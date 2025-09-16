#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Qitong Fang
"""
import os
os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python d2sc_DRG_train.py \
--gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101 \
--class_embedding att --class_embedding_norm --nepoch 300 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--nclass_all 50 --dataroot datasets/xlsa17/data --dataset AWA2 --eval_interval 1 \
--noiseSize 312 --attSize 85 --resSize 2048 \
--lr 0.001 --classifier_lr 0.001 --gamma_recons 1.0 --freeze_dec \
--gamma_ADV 10 --gamma_VAE 1.0 --embed_type VA \
--n_T 4 --dim_t 85 --gamma_x0 1.0 --gamma_xt 1.0 --gamma_dist 0.0 \
--batch_size 64 --syn_num 1800 --split_percent 100 \
--drg_neg_sampling none --gamma_CON_step 0.02 \
''')




