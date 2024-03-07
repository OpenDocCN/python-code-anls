# `.\YOLO-World\configs\pretrain\yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py`

```
# 设置基础配置文件路径
_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
# 自定义导入模块配置
custom_imports = dict(imports=['yolo_world'],
                      allow_failed_imports=False)

# 超参数设置
num_classes = 1203
num_training_classes = 80
max_epochs = 20  # 最大训练轮数
close_mosaic_epochs = 2
save_epoch_intervals = 2
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.025
train_batch_size_per_gpu = 4
load_from = "pretrained_models/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth"

img_scale = (1280, 1280)

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
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    use_bn_head=True,
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
# train_pipeline 列表定义，包含一系列数据处理步骤
train_pipeline = [
    *_base_.pre_transform,  # 将_base_.pre_transform中的元素添加到train_pipeline中
    dict(type='MultiModalMosaic',  # 使用MultiModalMosaic进行数据增强
         img_scale=img_scale,  # 图像缩放比例
         pad_val=114.0,  # 填充值
         pre_transform=_base_.pre_transform),  # 预处理步骤
    dict(
        type='YOLOv5RandomAffine',  # 使用YOLOv5RandomAffine进行数据增强
        max_rotate_degree=0.0,  # 最大旋转角度
        max_shear_degree=0.0,  # 最大剪切角度
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),  # 缩放比例范围
        max_aspect_ratio=_base_.max_aspect_ratio,  # 最大长宽比
        border=(-img_scale[0] // 2, -img_scale[1] // 2),  # 边界
        border_val=(114, 114, 114)),  # 边界填充值
    *_base_.last_transform[:-1],  # 将_base_.last_transform中的元素添加到train_pipeline中，除了最后一个元素
    *text_transform,  # 将text_transform中的元素添加到train_pipeline中
]

# train_pipeline_stage2 列表定义，包含一系列数据处理步骤
train_pipeline_stage2 = [
    *_base_.pre_transform,  # 将_base_.pre_transform中的元素添加到train_pipeline_stage2中
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),  # 使用YOLOv5KeepRatioResize进行数据增强
    dict(
        type='LetterResize',  # 使用LetterResize进行数据增强
        scale=img_scale,  # 图像缩放比例
        allow_scale_up=True,  # 允许缩放
        pad_val=dict(img=114.0)),  # 图像填充值
    dict(
        type='YOLOv5RandomAffine',  # 使用YOLOv5RandomAffine进行数据增强
        max_rotate_degree=0.0,  # 最大旋转角度
        max_shear_degree=0.0,  # 最大剪切角度
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),  # 缩放比例范围
        max_aspect_ratio=_base_.max_aspect_ratio,  # 最大长宽比
        border_val=(114, 114, 114)),  # 边界填充值
    *_base_.last_transform[:-1],  # 将_base_.last_transform中的元素添加到train_pipeline_stage2中，除了最后一个元素
    *text_transform  # 将text_transform中的元素添加到train_pipeline_stage2中
]

# obj365v1_train_dataset 字典定义，包含数据集相关信息和数据处理步骤
obj365v1_train_dataset = dict(
    type='MultiModalDataset',  # 多模态数据集
    dataset=dict(
        type='YOLOv5Objects365V1Dataset',  # 使用YOLOv5Objects365V1Dataset数据集
        data_root='data/objects365v1/',  # 数据根目录
        ann_file='annotations/objects365_train.json',  # 标注文件
        data_prefix=dict(img='train/'),  # 数据前缀
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),  # 过滤配置
    class_text_path='data/texts/obj365v1_class_texts.json',  # 类别文本路径
    pipeline=train_pipeline  # 数据处理步骤
)

# mg_train_dataset 字典定义，包含数据集相关信息和数据处理步骤
mg_train_dataset = dict(type='YOLOv5MixedGroundingDataset',  # 使用YOLOv5MixedGroundingDataset数据集
                        data_root='data/mixed_grounding/',  # 数据根目录
                        ann_file='annotations/final_mixed_train_no_coco.json',  # 标注文件
                        data_prefix=dict(img='gqa/images/'),  # 数据前缀
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),  # 过滤配置
                        pipeline=train_pipeline  # 数据处理步骤
)

# flickr_train_dataset 字典定义，包含数据集相关信息和数据处理步骤
flickr_train_dataset = dict(
    type='YOLOv5MixedGroundingDataset',
    # 数据根目录
    data_root='data/flickr/',
    # 注释文件路径
    ann_file='annotations/final_flickr_separateGT_train.json',
    # 数据前缀，包含图片路径
    data_prefix=dict(img='full_images/'),
    # 过滤配置，包含是否过滤空的 ground truth 和最小尺寸限制
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    # 训练管道
    pipeline=train_pipeline)
# 定义训练数据加载器，设置批量大小、数据集拼接方式、数据集列表和忽略的键
train_dataloader = dict(batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=dict(_delete_=True,
                                     type='ConcatDataset',
                                     datasets=[
                                         obj365v1_train_dataset,
                                         flickr_train_dataset, mg_train_dataset
                                     ],
                                     ignore_keys=['classes', 'palette']))

# 定义测试数据处理流程，包括加载图像、YOLOv5保持比例缩放、LetterResize、加载标注、加载文本和打包检测输入
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]

# 定义COCO验证数据集，设置数据集类型、数据根目录、测试模式、标注文件、数据前缀和批量形状配置
coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5LVISV1Dataset',
                 data_root='data/coco/',
                 test_mode=True,
                 ann_file='lvis/lvis_v1_minival_inserted_image_name.json',
                 data_prefix=dict(img=''),
                 batch_shapes_cfg=None),
    class_text_path='data/texts/lvis_v1_class_texts.json',
    pipeline=test_pipeline)

# 定义验证数据加载器，设置数据集为COCO验证数据集
val_dataloader = dict(dataset=coco_val_dataset)
# 将测试数据加载器设置为验证数据加载器
test_dataloader = val_dataloader

# 定义验证评估器，设置评估类型为bbox，标注文件为LVIS标注文件
val_evaluator = dict(type='mmdet.LVISMetric',
                     ann_file='data/coco/lvis/lvis_v1_minival_inserted_image_name.json',
                     metric='bbox')
# 将测试评估器设置为验证评估器
test_evaluator = val_evaluator

# 训练设置，包括默认钩子和自定义钩子
default_hooks = dict(param_scheduler=dict(max_epochs=max_epochs),
                     checkpoint=dict(interval=save_epoch_intervals,
                                     rule='greater'))
custom_hooks = [
    # 创建一个字典，包含EMAHook的相关参数
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    # 创建一个字典，包含PipelineSwitchHook的相关参数
    dict(type='mmdet.PipelineSwitchHook',
         # 计算切换pipeline的时机，根据最大训练轮数和关闭mosaic的轮数计算得出
         switch_epoch=max_epochs - close_mosaic_epochs,
         # 设置切换后的pipeline为train_pipeline_stage2
         switch_pipeline=train_pipeline_stage2)
# 创建一个字典，包含训练配置参数，如最大训练轮数、验证间隔等
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=10,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])

# 创建一个字典，包含优化器的配置参数，如优化器类型、学习率、权重衰减等
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
                                            dict(weight_decay=0.0)
                                        }),
                     constructor='YOLOWv5OptimizerConstructor')
```