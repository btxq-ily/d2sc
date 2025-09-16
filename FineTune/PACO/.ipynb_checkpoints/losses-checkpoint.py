import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class PaCoLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.0, supt=1.0, temperature=1.0, base_temperature=None, K=128, num_classes=1000):
        super(PaCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha 
        self.beta = beta 
        self.gamma = gamma 
        self.supt = supt
        self.num_classes = num_classes

    def forward(self, features, labels=None, sup_logits=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        ss = features.shape[0]
        batch_size = ( features.shape[0] - self.K ) // 2

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # add supervised logits
        anchor_dot_contrast = torch.cat(( (sup_logits) / self.supt, anchor_dot_contrast), dim=1)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        # add ground truth 
        one_hot_label = torch.nn.functional.one_hot(labels[:batch_size,].view(-1,), num_classes=self.num_classes).to(torch.float32)
        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

        # compute log_prob
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
       
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        
        return loss

def mmd_loss(x, y, sigma=1.0):
    """
    改进的MMD损失，使用高斯核
    x, y: [batch, feat_dim]
    """
    if x.size(0) == 0 or y.size(0) == 0:
        return torch.tensor(0.0, device=x.device)
    
    # 计算核矩阵
    x_expand = x.unsqueeze(1)  # [batch_x, 1, dim]
    y_expand = y.unsqueeze(0)  # [1, batch_y, dim]
    
    # 计算L2距离
    diff = x_expand - y_expand  # [batch_x, batch_y, dim]
    dist_sq = torch.sum(diff**2, dim=-1)  # [batch_x, batch_y]
    
    # 高斯核
    kernel = torch.exp(-dist_sq / (2 * sigma**2))
    
    # MMD统计量
    xx_kernel = torch.exp(-torch.cdist(x, x, p=2)**2 / (2 * sigma**2))
    yy_kernel = torch.exp(-torch.cdist(y, y, p=2)**2 / (2 * sigma**2))
    
    mmd = (xx_kernel.mean() + yy_kernel.mean() - 2 * kernel.mean())
    return mmd


def center_loss(features, centers, labels):
    """
    改进的Center Loss，增加边界检查
    features: [batch, feat_dim]
    centers: [num_classes, feat_dim]
    labels: [batch]
    """
    # 确保labels在有效范围内
    max_label = centers.size(0) - 1
    valid_labels = torch.clamp(labels, 0, max_label)
    
    # 获取对应的中心
    batch_centers = centers[valid_labels]
    
    # 计算到中心的距离
    dist = torch.sum((features - batch_centers) ** 2, dim=1)
    
    return dist.mean()


def balanced_classifier_loss(logits, labels, seen_classes, unseen_classes, seen_weight=1.0, unseen_weight=1.5):
    """
    平衡的分类器损失，为unseen类别增加权重
    """
    class_weights = torch.ones(logits.size(1), device=logits.device)
    
    # 为unseen类别增加权重
    for unseen_idx in unseen_classes:
        if unseen_idx < class_weights.size(0):
            class_weights[unseen_idx] = unseen_weight
    
    # 为seen类别设置权重
    for seen_idx in seen_classes:
        if seen_idx < class_weights.size(0):
            class_weights[seen_idx] = seen_weight
    
    ce_loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='none')
    return ce_loss.mean()


def adaptive_bias_loss(features, labels, seen_classes, unseen_classes):
    """
    自适应偏置损失，动态调整seen/unseen类别的特征分布
    """
    seen_mask = torch.isin(labels, seen_classes)
    unseen_mask = torch.isin(labels, unseen_classes)
    
    if seen_mask.sum() == 0 or unseen_mask.sum() == 0:
        return torch.tensor(0.0, device=features.device)
    
    seen_features = features[seen_mask]
    unseen_features = features[unseen_mask]
    
    # 计算特征分布的中心和方差
    seen_center = seen_features.mean(dim=0)
    unseen_center = unseen_features.mean(dim=0)
    seen_var = seen_features.var(dim=0).mean()
    unseen_var = unseen_features.var(dim=0).mean()
    
    # 自适应偏置：根据方差差异调整中心距离
    variance_ratio = unseen_var / (seen_var + 1e-8)
    adaptive_bias = torch.norm(seen_center - unseen_center) * variance_ratio
    
    # 目标：让unseen特征分布更紧凑，与seen特征分布保持适当距离
    target_distance = 0.5
    bias_loss = F.mse_loss(adaptive_bias, torch.tensor(target_distance, device=features.device))
    
    return bias_loss


def harmonic_balance_loss(seen_acc, unseen_acc, target_h=0.85):
    """
    谐波平衡损失，直接优化H-score
    """
    current_h = 2 * seen_acc * unseen_acc / (seen_acc + unseen_acc + 1e-8)
    h_loss = F.mse_loss(current_h, torch.tensor(target_h, device=seen_acc.device))
    balance_loss = F.mse_loss(seen_acc, unseen_acc)
    
    return h_loss + 0.1 * balance_loss
