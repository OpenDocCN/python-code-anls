# `.\YOLO-World\configs\finetune_coco\yolo_world_l_efficient_neck_2e-4_80e_8gpus_mask-refine_finetune_coco.py`

```py
_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# 定义基础配置文件路径和自定义导入配置
# _base_ 为基础配置文件路径
# custom_imports 为自定义导入配置，包含导入模块和是否允许导入失败

# hyper-parameters
num_classes = 80
num_training_classes = 80
max_epochs = 80  # Maximum training epochs
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 16
load_from = 'pretrained_models/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-0e566235.pth'
persistent_workers = False

# 定义超参数
# num_classes 为类别数量
# max_epochs 为最大训练轮数
# text_channels 为文本通道数
# neck_embed_channels 为颈部嵌入通道数
# neck_num_heads 为颈部注意力头数
# base_lr 为基础学习率
# weight_decay 为权重衰减
# train_batch_size_per_gpu 为每个 GPU 的训练批量大小
# load_from 为预训练模型路径
# persistent_workers 为是否持久化工作进程

# model settings
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
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='EfficientCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# 定义模型设置
# type 为模型类型
# mm_neck 为是否使用多模态颈部
# data_preprocessor 为数据预处理器类型
# backbone 为骨干网络配置
# neck 为颈部网络配置
# bbox_head 为边界框头部配置
# train_cfg 为训练配置

# dataset settings
text_transform = [

# 定义数据集设���
# text_transform 为文本转换器
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
# 定义一个包含多个数据增强操作的列表，用于对图像进行仿射变换
mosaic_affine_transform = [
    # 多模态镶嵌操作，设置图像缩放、填充值、预处理操作
    dict(type='MultiModalMosaic',
         img_scale=_base_.img_scale,
         pad_val=114.0,
         pre_transform=_base_.pre_transform),
    # YOLOv5CopyPaste 操作，设置复制粘贴的概率
    dict(type='YOLOv5CopyPaste', prob=_base_.copypaste_prob),
    # YOLOv5RandomAffine 操作，设置随机仿射变换的参数
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        # 图像缩放比例为 (宽度, 高度)
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=_base_.min_area_ratio,
        use_mask_refine=_base_.use_mask2refine)
]
# 训练数据处理流程，包括预处理、仿射变换、MixUp 操作等
train_pipeline = [
    *_base_.pre_transform, *mosaic_affine_transform,
    # YOLOv5MultiModalMixUp 操作，设置 MixUp 操作的概率和预处理操作
    dict(type='YOLOv5MultiModalMixUp',
         prob=_base_.mixup_prob,
         pre_transform=[*_base_.pre_transform, *mosaic_affine_transform]),
    # 最后的数据处理操作，除最后一个元素外，添加文本转换操作
    *_base_.last_transform[:-1], *text_transform
]
# 第二阶段训练数据处理流程，除最后一个元素外，添加文本转换操作
train_pipeline_stage2 = [*_base_.train_pipeline_stage2[:-1], *text_transform]
# COCO 训练数据集配置，设置数据集路径、注释文件、类别文本路径和数据处理流程
coco_train_dataset = dict(_delete_=True,
                          type='MultiModalDataset',
                          dataset=dict(
                              type='YOLOv5CocoDataset',
                              data_root='data/coco',
                              ann_file='annotations/instances_train2017.json',
                              data_prefix=dict(img='train2017/'),
                              filter_cfg=dict(filter_empty_gt=False,
                                              min_size=32)),
                          class_text_path='data/texts/coco_class_texts.json',
                          pipeline=train_pipeline)
# 训练数据加载器配置，设置持久化工作进程、每个 GPU 的批处理大小、数据集和数据整理函数
train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)
# 测试数据处理流程，除最后一个元素外，保持不变
test_pipeline = [
    *_base_.test_pipeline[:-1],
    # 创建一个字典，指定类型为'LoadText'
    dict(type='LoadText'),
    # 创建一个字典，指定类型为'mmdet.PackDetInputs'，并指定元数据的键值
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
# 以下代码存在语法错误，缺少左括号，需要修复
# 定义 coco_val_dataset 字典，包含数据集信息和数据预处理流程
coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5CocoDataset',
                 data_root='data/coco',
                 ann_file='annotations/instances_val2017.json',
                 data_prefix=dict(img='val2017/'),
                 filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/coco_class_texts.json',
    pipeline=test_pipeline)
# 定义 val_dataloader 字典，包含验证数据集信息
val_dataloader = dict(dataset=coco_val_dataset)
# 将验证数据集赋值给测试数据集
test_dataloader = val_dataloader
# 定义默认的训练设置
default_hooks = dict(param_scheduler=dict(scheduler_type='linear',
                                          lr_factor=0.01,
                                          max_epochs=max_epochs),
                     checkpoint=dict(max_keep_ckpts=-1,
                                     save_best=None,
                                     interval=save_epoch_intervals))
# 定义自定义的训练设置
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
# 定义训练配置
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=5,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])
# 定义优化器包装器
optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    weight_decay=weight_decay,
    # 设置每个 GPU 的训练批量大小
    batch_size_per_gpu=train_batch_size_per_gpu),
    # 针对参数进行配置，包括偏置项和归一化项的衰减倍数，以及自定义键值对
    paramwise_cfg=dict(bias_decay_mult=0.0,
                       norm_decay_mult=0.0,
                       custom_keys={
                           'backbone.text_model':
                           dict(lr_mult=0.01),
                           'logit_scale':
                           dict(weight_decay=0.0)
                       }),
    # 使用 YOLOWv5 优化器构造函数
    constructor='YOLOWv5OptimizerConstructor')
# 定义评估器的设置
val_evaluator = dict(_delete_=True,  # 删除原有的评估器设置
                     type='mmdet.CocoMetric',  # 使用 mmdet 库中的 CocoMetric 类
                     proposal_nums=(100, 1, 10),  # 提议框的数量设置
                     ann_file='data/coco/annotations/instances_val2017.json',  # COCO 数据集的标注文件路径
                     metric='bbox')  # 评估指标为边界框（bbox）
```