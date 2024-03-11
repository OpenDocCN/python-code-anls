# `.\YOLO-World\configs\finetune_coco\yolo_world_l_dual_vlpan_2e-4_80e_8gpus_finetune_coco.py`

```py
_base_ = (
    '../../third_party/mmyolo/configs/yolov8/'
    'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
# 定义自定义的导入模块和设置是否允许导入失败
custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False)

# 超参数设置
num_classes = 80
num_training_classes = 80
max_epochs = 80  # 最大训练轮数
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 16
load_from='pretrained_models/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-0e566235.pth'
persistent_workers = False

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
            frozen_modules=['all'])),
    neck=dict(type='YOLOWorldDualPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
              text_enhancder=dict(type='ImagePoolingAttentionModule',
                                  embed_channels=256,
                                  num_heads=8)),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# 数据集设置
text_transform = [
    # 定义一个字典，包含参数 type、num_neg_samples、max_num_samples、padding_to_max 和 padding_value
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    # 定义一个字典，包含参数 type 和 meta_keys
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
# 定义一个包含多个元素的列表，每个元素是一个字典，用于进行仿射变换
mosaic_affine_transform = [
    dict(
        type='MultiModalMosaic',
        img_scale=_base_.img_scale,
        pad_val=114.0,
        pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - _base_.affine_scale,
                             1 + _base_.affine_scale),
        # img_scale is (width, height)
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114))
]

# 定义训练数据处理流程的列表
train_pipeline = [
    *_base_.pre_transform,
    *mosaic_affine_transform,
    dict(
        type='YOLOv5MultiModalMixUp',
        prob=_base_.mixup_prob,
        pre_transform=[*_base_.pre_transform,
                       *mosaic_affine_transform]),
    *_base_.last_transform[:-1],
    *text_transform
]

# 定义第二阶段训练数据处理流程的列表
train_pipeline_stage2 = [
    *_base_.train_pipeline_stage2[:-1],
    *text_transform
]

# 定义 COCO 训练数据集的配置字典
coco_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data/coco',
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/coco_class_texts.json',
    pipeline=train_pipeline)

# 定义训练数据加载器的配置字典
train_dataloader = dict(
    persistent_workers=persistent_workers,
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=coco_train_dataset)

# 定义测试数据处理流程的列表
test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'texts'))
]

# 定义 COCO 验证数据集的配置字典
coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    # 定义数据集参数，指定数据集类型为YOLOv5CocoDataset
    dataset=dict(
        type='YOLOv5CocoDataset',
        # 数据集根目录
        data_root='data/coco',
        # 标注文件路径
        ann_file='annotations/instances_val2017.json',
        # 数据前缀，包含图片路径
        data_prefix=dict(img='val2017/'),
        # 过滤配置，设置不过滤空的ground truth，最小尺寸为32
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    # 类别文本路径
    class_text_path='data/texts/coco_class_texts.json',
    # 测试数据处理管道
    pipeline=test_pipeline)
# 创建验证数据加载器，使用 COCO 验证数据集
val_dataloader = dict(dataset=coco_val_dataset)
# 将验证数据加载器赋值给测试数据加载器
test_dataloader = val_dataloader

# 训练设置
# 默认钩子函数设置
default_hooks = dict(
    param_scheduler=dict(
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        max_keep_ckpts=-1,
        save_best=None,
        interval=save_epoch_intervals))
# 自定义钩子函数设置
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]
# 训练配置设置
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        _base_.val_interval_stage2)])
# 优化器包装器设置
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={'backbone.text_model': dict(lr_mult=0.01),
                     'logit_scale': dict(weight_decay=0.0)}),
    constructor='YOLOWv5OptimizerConstructor')

# 评估设置
# 验证评估器设置
val_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox')
```