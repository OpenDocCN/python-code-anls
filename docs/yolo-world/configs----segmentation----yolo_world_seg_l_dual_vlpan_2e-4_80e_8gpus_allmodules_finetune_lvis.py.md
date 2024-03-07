# `.\YOLO-World\configs\segmentation\yolo_world_seg_l_dual_vlpan_2e-4_80e_8gpus_allmodules_finetune_lvis.py`

```
_base_ = (
    '../../third_party/mmyolo/configs/yolov8/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py'
)
# 定义基础配置文件路径

custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)
# 自定义导入模块，禁止导入失败

# 超参数
num_classes = 1203
num_training_classes = 80
max_epochs = 80  # 最大训练轮数
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 8
load_from = 'pretrained_models/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-0e566235.pth'
persistent_workers = False

# Polygon2Mask
downsample_ratio = 4
mask_overlap = False
use_mask2refine = True
max_aspect_ratio = 100
min_area_ratio = 0.01

# 模型设置
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='openai/clip-vit-base-patch32',
            frozen_modules=[])),
    neck=dict(type='YOLOWorldDualPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
              text_enhancder=dict(type='ImagePoolingAttentionModule',
                                  embed_channels=256,
                                  num_heads=8)),
    # 定义 YOLO 网络的头部结构，包括类型、模块类型、嵌入维度、类别数量、掩模通道数和原型通道数
    bbox_head=dict(type='YOLOWorldSegHead',
                   head_module=dict(type='YOLOWorldSegHeadModule',
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes,
                                    mask_channels=32,
                                    proto_channels=256),
                   mask_overlap=mask_overlap,
                   # 定义掩模损失函数，使用交叉熵损失，采用 sigmoid 函数，不进行降维
                   loss_mask=dict(type='mmdet.CrossEntropyLoss',
                                  use_sigmoid=True,
                                  reduction='none'),
                   # 定义掩模损失的权重
                   loss_mask_weight=1.0),
    # 定义训练配置，包括分配器和类别数量
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)),
    # 定义测试配置，包括二值化掩模阈值和快速测试标志
    test_cfg=dict(mask_thr_binary=0.5, fast_test=True))
# 定义数据预处理流程的前置转换操作
pre_transform = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),  # 从文件加载图像
    dict(type='LoadAnnotations',
         with_bbox=True,  # 加载边界框信息
         with_mask=True,  # 加载掩码信息
         mask2bbox=True)  # 将掩码转换为边界框
]

# 定义数据预处理流程的最终转换操作
last_transform = [
    dict(type='mmdet.Albu',
         transforms=_base_.albu_train_transforms,  # 使用指定的数据增强操作
         bbox_params=dict(type='BboxParams',
                          format='pascal_voc',
                          label_fields=['gt_bboxes_labels',
                                        'gt_ignore_flags']),  # 设置边界框参数
         keymap={
             'img': 'image',
             'gt_bboxes': 'bboxes'
         }),  # 映射关键字
    dict(type='YOLOv5HSVRandomAug'),  # 使用YOLOv5的HSV随机增强
    dict(type='mmdet.RandomFlip', prob=0.5),  # 随机翻转操作
    dict(type='Polygon2Mask',
         downsample_ratio=downsample_ratio,  # 设置下采样比例
         mask_overlap=mask_overlap)  # 设置掩码重叠参数
]

# 数据集设置
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),  # 设置负样本数量
         max_num_samples=num_training_classes,  # 设置最大样本数量
         padding_to_max=True,  # 填充到最大长度
         padding_value=''),  # 设置填充值
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))  # 打包检测输入信息
]

mosaic_affine_transform = [
    dict(type='MultiModalMosaic',
         img_scale=_base_.img_scale,  # 设置图像缩放比例
         pad_val=114.0,  # 设置填充值
         pre_transform=pre_transform),  # 设置前置转换操作
    dict(type='YOLOv5CopyPaste', prob=_base_.copypaste_prob),  # 使用YOLOv5的复制粘贴操作
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,  # 设置最大旋转角度
        max_shear_degree=0.0,  # 设置最大剪切角度
        max_aspect_ratio=100.,  # 设置最大长宽比
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),  # 设置缩放比例范围
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),  # 设置边界
        border_val=(114, 114, 114),  # 设置边界填充值
        min_area_ratio=_base_.min_area_ratio,  # 设置最小区域比例
        use_mask_refine=True)  # 使用掩码细化
]

train_pipeline = [
    *pre_transform, *mosaic_affine_transform,  # 将前置转换操作和镜像仿射变换操作合并到训练流程中
    # 创建一个字典，指定模型类型为YOLOv5MultiModalMixUp，概率为mixup_prob
    dict(type='YOLOv5MultiModalMixUp',
         prob=_base_.mixup_prob,
         # 将pre_transform和mosaic_affine_transform的元素合并到一个列表中
         pre_transform=[*pre_transform, *mosaic_affine_transform]),
    # 将last_transform和text_transform的元素合并到一个列表中
    *last_transform, *text_transform
# 定义训练管道的第二阶段，包括预处理、YOLOv5KeepRatioResize、LetterResize、YOLOv5RandomAffine等操作
_train_pipeline_stage2 = [
    *pre_transform,  # 将pre_transform中的操作添加到管道中
    dict(type='YOLOv5KeepRatioResize', scale=_base_.img_scale),  # 使用YOLOv5KeepRatioResize进行图像尺寸调整
    dict(type='LetterResize',  # 使用LetterResize进行图像尺寸调整
         scale=_base_.img_scale,  # 设置尺度
         allow_scale_up=True,  # 允许尺度放大
         pad_val=dict(img=114.0)),  # 设置填充值
    dict(type='YOLOv5RandomAffine',  # 使用YOLOv5RandomAffine进行随机仿射变换
         max_rotate_degree=0.0,  # 设置最大旋转角度
         max_shear_degree=0.0,  # 设置最大剪切角度
         scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),  # 设置缩放比例范围
         max_aspect_ratio=_base_.max_aspect_ratio,  # 设置最大长宽比
         border_val=(114, 114, 114),  # 设置边界值
         min_area_ratio=min_area_ratio,  # 设置最小区域比例
         use_mask_refine=use_mask2refine),  # 是否使用mask进行细化
    *last_transform  # 将last_transform中的操作添加到管道中
]
# 将_train_pipeline_stage2和text_transform合并为train_pipeline_stage2
train_pipeline_stage2 = [*_train_pipeline_stage2, *text_transform]

# 定义coco_train_dataset，包括数据集类型、数据根目录、注释文件、数据前缀等信息
coco_train_dataset = dict(
    _delete_=True,  # 删除该字段
    type='MultiModalDataset',  # 数据集类型为MultiModalDataset
    dataset=dict(type='YOLOv5LVISV1Dataset',  # 数据集类型为YOLOv5LVISV1Dataset
                 data_root='data/coco',  # 数据根目录
                 ann_file='lvis/lvis_v1_train_base.json',  # 注释文件
                 data_prefix=dict(img=''),  # 数据前缀
                 filter_cfg=dict(filter_empty_gt=True, min_size=32)),  # 过滤配置信息
    class_text_path='data/captions/lvis_v1_base_class_captions.json',  # 类别文本路径
    pipeline=train_pipeline)  # 数据处理管道为train_pipeline

# 定义train_dataloader，包括持久化工作进程、每个GPU的训练批次大小、数据集、数据集合并函数等信息
train_dataloader = dict(persistent_workers=persistent_workers,  # 持久化工作进程
                        batch_size=train_batch_size_per_gpu,  # 每个GPU的训练批次大小
                        collate_fn=dict(type='yolow_collate'),  # 数据集合并函数
                        dataset=coco_train_dataset)  # 数据集为coco_train_dataset

# 定义测试管道，包括基础测试管道、LoadText、mmdet.PackDetInputs等操作
test_pipeline = [
    *_base_.test_pipeline[:-1],  # 将基础测试管道中的操作添加到管道中，去掉最后一个操作
    dict(type='LoadText'),  # 加载文本
    dict(type='mmdet.PackDetInputs',  # 使用mmdet.PackDetInputs打包检测输入
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))  # 设置元数据键
]

# 训练设置
# 默认的钩子配置，包括参数调度器和检查点
default_hooks = dict(param_scheduler=dict(scheduler_type='linear',
                                          lr_factor=0.01,
                                          max_epochs=max_epochs),
                     checkpoint=dict(max_keep_ckpts=-1,
                                     save_best=None,
                                     interval=save_epoch_intervals))
# 自定义的钩子配置
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=train_pipeline_stage2)
]
# 训练配置
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=5,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])
# 优化器包装器配置
optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
                     paramwise_cfg=dict(bias_decay_mult=0.0,
                                        norm_decay_mult=0.0,
                                        custom_keys={
                                            'backbone.text_model':
                                            dict(lr_mult=0.01),
                                            'logit_scale':
                                            dict(weight_decay=0.0),
                                        }),
                     constructor='YOLOWv5OptimizerConstructor')

# 评估设置
coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5LVISV1Dataset',
                 data_root='data/coco/',
                 test_mode=True,
                 ann_file='lvis/lvis_v1_val.json',
                 data_prefix=dict(img=''),
                 batch_shapes_cfg=None),
    # 定义类别文本路径为'data/captions/lvis_v1_class_captions.json'，用于存储类别标签的文本信息
    class_text_path='data/captions/lvis_v1_class_captions.json',
    # 定义数据处理流程为test_pipeline，用于对数据进行预处理和增强操作
    pipeline=test_pipeline)
# 创建验证数据加载器，使用 COCO 验证数据集
val_dataloader = dict(dataset=coco_val_dataset)
# 将验证数据加载器赋值给测试数据加载器
test_dataloader = val_dataloader

# 创建验证评估器，类型为 'mmdet.LVISMetric'，使用 LVIS 验证注释文件，评估指标包括边界框和分割
val_evaluator = dict(type='mmdet.LVISMetric',
                     ann_file='data/coco/lvis/lvis_v1_val.json',
                     metric=['bbox', 'segm'])
# 将验证评估器赋值给测试评估器
test_evaluator = val_evaluator

# 设置参数为查找未使用的参数
find_unused_parameters = True
```