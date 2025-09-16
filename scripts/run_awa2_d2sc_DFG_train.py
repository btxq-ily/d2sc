#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Qitong Fang
"""
import os
os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python d2sc_DFG_train.py \
--gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101 \
--class_embedding att --class_embedding_norm --nepoch 300 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--nclass_all 50 --dataroot datasets/xlsa17/data --dataset AWA2 --eval_interval 1 \
--batch_size 64 --noiseSize 312 --attSize 85 --resSize 2048 \
--lr 0.0005 --classifier_lr 0.001 --gamma_recons 0.5 --freeze_dec --dec_lr 0.0001 \
--gamma_ADV 10 --gamma_VAE 1.0 --embed_type VA --unseen_bias 1.3 \
--n_T 4 --dim_t 85 --gamma_x0 1.0 --gamma_xt 1.0 \
--split_percent 100 --syn_num 8000  --gamma_dist 5.0 --factor_dist 1.5 \
--gamma_mmd 0.25 --gamma_center 0.25  --gamma_recons 1.0 \
--netG_con_model_path ./out/AWA2/train_d2sc_DRG_100percent_att:att_b:64_lr:0.001_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:1800_zsl.tar \
''')
    
