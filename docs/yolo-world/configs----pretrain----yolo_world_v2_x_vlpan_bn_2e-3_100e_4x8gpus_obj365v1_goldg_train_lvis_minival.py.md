# `.\YOLO-World\configs\pretrain\yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py`

```
# 设置基础配置文件路径
_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_x_syncbn_fast_8xb16-500e_coco.py')
# 自定义导入模块
custom_imports = dict(imports=['yolo_world'],
                      allow_failed_imports=False)

# 超参数设置
num_classes = 1203
num_training_classes = 80
max_epochs = 100  # 最大训练轮数
close_mosaic_epochs = 2
save_epoch_intervals = 2
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-3
weight_decay = 0.05 / 2
train_batch_size_per_gpu = 16

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
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    # 创建一个字典对象，包含指定的键值对
    dict(type='mmdet.PackDetInputs',
         # 定义元数据的键名元组
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
# 定义训练管道，包含一系列数据处理和增强操作
train_pipeline = [
    *_base_.pre_transform,  # 使用基础预处理操作
    dict(type='MultiModalMosaic',  # 使用多模态马赛克操作
         img_scale=_base_.img_scale,  # 图像缩放比例
         pad_val=114.0,  # 填充值
         pre_transform=_base_.pre_transform),  # 预处理操作
    dict(
        type='YOLOv5RandomAffine',  # 使用YOLOv5随机仿射变换操作
        max_rotate_degree=0.0,  # 最大旋转角度
        max_shear_degree=0.0,  # 最大剪切角度
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),  # 缩放比例范围
        max_aspect_ratio=_base_.max_aspect_ratio,  # 最大长宽比
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),  # 边界
        border_val=(114, 114, 114)),  # 边界填充值
    *_base_.last_transform[:-1],  # 使用基础最后的转换操作
    *text_transform,  # 文本转换操作
]

# 定义第二阶段的训练管道
train_pipeline_stage2 = [*_base_.train_pipeline_stage2[:-1], *text_transform]

# 定义obj365v1训练数据集
obj365v1_train_dataset = dict(
    type='MultiModalDataset',  # 多模态数据集类型
    dataset=dict(
        type='YOLOv5Objects365V1Dataset',  # 使用YOLOv5 Objects365V1数据集
        data_root='data/objects365v1/',  # 数据根目录
        ann_file='annotations/objects365_train.json',  # 标注文件
        data_prefix=dict(img='train/'),  # 数据前缀
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),  # 过滤配置
    class_text_path='data/texts/obj365v1_class_texts.json',  # 类别文本路径
    pipeline=train_pipeline)  # 数据处理管道

# 定义mg训练数据集
mg_train_dataset = dict(type='YOLOv5MixedGroundingDataset',  # 使用YOLOv5混合定位数据集
                        data_root='data/mixed_grounding/',  # 数据根目录
                        ann_file='annotations/final_mixed_train_no_coco.json',  # 标注文件
                        data_prefix=dict(img='gqa/images/'),  # 数据前缀
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),  # 过滤配置
                        pipeline=train_pipeline)  # 数据处理管道

# 定义flickr训练数据集
flickr_train_dataset = dict(
    type='YOLOv5MixedGroundingDataset',  # 使用YOLOv5混合定位数据集
    data_root='data/flickr/',  # 数据根目录
    ann_file='annotations/final_flickr_separateGT_train.json',  # 标注文件
    data_prefix=dict(img='full_images/'),  # 数据前缀
    filter_cfg=dict(filter_empty_gt=True, min_size=32),  # 过滤配置
    pipeline=train_pipeline)  # 数据处理管道
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

# 定义测试数据处理流程，包括加载文本和打包检测输入
test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]

# 定义 COCO 验证数据集，设置数据集类型、根目录、测试模式、注释文件、数据前缀和批量形状配置
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

# 定义验证数据加载器，设置数据集为 COCO 验证数据集
val_dataloader = dict(dataset=coco_val_dataset)
# 将测试数据加载器设置为验证数据加载器
test_dataloader = val_dataloader

# 定义验证评估器，设置评估类型为 LVIS 检测指标，注释文件为 LVIS 最小验证集的插入图像名称文件
val_evaluator = dict(type='mmdet.LVISMetric',
                     ann_file='data/coco/lvis/lvis_v1_minival_inserted_image_name.json',
                     metric='bbox')
# 将测试评估器设置为验证评估器
test_evaluator = val_evaluator

# 训练设置
# 默认钩子，设置参数调度器和检查点保存间隔
default_hooks = dict(param_scheduler=dict(max_epochs=max_epochs),
                     checkpoint=dict(interval=save_epoch_intervals,
                                     rule='greater'))
# 自定义钩子，设置指数动量 EMA 钩子的参数
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    # 创建一个字典，包含PipelineSwitchHook的相关参数
    dict(type='mmdet.PipelineSwitchHook', 
         # 设置切换pipeline的时机为最大训练轮数减去关闭mosaic的轮数
         switch_epoch=max_epochs - close_mosaic_epochs, 
         # 设置切换的pipeline为train_pipeline_stage2
         switch_pipeline=train_pipeline_stage2)
# 创建一个字典，包含训练配置参数
train_cfg = dict(max_epochs=max_epochs,  # 最大训练轮数
                 val_interval=10,  # 验证间隔
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),  # 动态间隔
                                     _base_.val_interval_stage2)])  # 验证阶段2的间隔设置

# 定义优化器包装器的配置参数
optim_wrapper = dict(optimizer=dict(
    _delete_=True,  # 删除原有的优化器配置
    type='AdamW',  # 优化器类型为AdamW
    lr=base_lr,  # 学习率
    weight_decay=weight_decay,  # 权重衰减
    batch_size_per_gpu=train_batch_size_per_gpu),  # 每个GPU的批处理大小
                     paramwise_cfg=dict(bias_decay_mult=0.0,  # 偏置项衰减倍数
                                        norm_decay_mult=0.0,  # 归一化层衰减倍数
                                        custom_keys={  # 自定义键值对
                                            'backbone.text_model':  # 文本模型的键
                                            dict(lr_mult=0.01),  # 学习率倍数
                                            'logit_scale':  # 输出层缩放的键
                                            dict(weight_decay=0.0)  # 输出层权重衰减
                                        }),
                     constructor='YOLOWv5OptimizerConstructor')  # 构造函数为YOLOWv5OptimizerConstructor
```