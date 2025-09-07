_base_ = ['../_base_.py']

feature_folder = 'F:/TAD/data/thumos14_6b'
name_format = '{}_spatial_feature.pt'
gt_path = 'data/thumos14/thumos14_soft_pseudo.json'
meta_info_path = 'data/thumos14/th14_i3d2s_act_ft_info.json'
dataset_name = 'thumos14'
num_classes = 20
binary = False
noise_scale = 0
stride = 4
base_frame = 16
default_fps = 30
mem_cache = True
resize = False

repeat_trainset = 1

# Evaluation
prime_metric = 'mAP_nms'
nms_mode = ['soft_nms', 'nms']
# nms_mode = ['nms']
nms_thr = 0.75
nms_sigma = 0.5
nms_multi_class = True
voting_thresh = -1
min_score = 0.001
duration_thresh = 0.05
extra_cls_path = None
iou_range = [0.3, 0.4, 0.5, 0.6, 0.7]
display_metric_indices = [0, 1, 2, 3, 4]


noise_scale = 0.0
noise_scaler = 0.0
seg_noise_scale = 0.0
label_smoothing = 0.0
eval_interval = 1
temperature = 10000
normalize = False

max_seq_len = 4096
downsample_rate = 1
base_scale = 32

eval_topk = 900
length_ratio = -1
eval_workers = None

enc_layers = 2
dec_layers = 6
num_cls_head_layers = 1
num_reg_head_layers = 3
num_feature_levels = 3
num_sampling_levels = 3
two_stage = True
emb_norm_type = 'ln'
emb_relu = True
kernel_size = 3
gc_kernel_size = 11
dc_level = 2

feature_dim = 3200
hidden_dim = 512

#
set_cost_class = 2
set_cost_seg = 0
set_cost_iou = 2
cls_loss_coef = 2
seg_loss_coef = 0
iou_loss_coef = 2
enc_loss_coef = 1

lr = 5e-5
weight_decay = 0
param_dict_type = 'default'
lr_backbone = 1e-05
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['sampling_offsets']
lr_linear_proj_mult = 0.1
clip_max_norm = 0.1

epochs = 80
lr_drop = 80
onecyclelr = False
multi_step_lr = False
lr_drop_list = [45, 50]
batch_size = 4
repeat_trainset = 1
save_checkpoint_interval = 100
optimizer = 'adamw'

use_checkpoint = False

pre_norm = False
dim_feedforward = 2048
enc_dropout = 0.0
dec_dropout = 0.0
emb_dropout = 0.0
attn_dropout = 0.0
n_heads = 8
n_deform_heads = 8
max_queries = 900
transformer_activation = 'relu'

enc_n_points = 4
dec_n_points = 4
aux_loss = True
focal_alpha = 0.25
query_selection_ratio = 0.5

# for ema
use_ema = False
ema_decay = 0.999
ema_epoch = 0


# for weakly

loss_lamb_1 = 2e-3
loss_lamb_2 = 5e-5
# loss_lamb_2 = 0.0
loss_lamb_3 = 2e-4
cls_threshold = 0.25
action_cls_num = 20