# `.\YOLO-World\configs\finetune_coco\yolo_world_l_dual_vlpan_2e-4_80e_8gpus_mask-refine_finetune_coco.py`

```py
_base_ = (
    '../../third_party/mmyolo/configs/yolov8/'
    'yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False)

# 定义超参数
num_classes = 80  # 类别数
num_training_classes = 80  # 训练类别数
max_epochs = 80  # 最大训练轮数
close_mosaic_epochs = 10  # 关闭镶嵌的轮数
save_epoch_intervals = 5  # 保存模型的间隔
text_channels = 512  # 文本通道数
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]  # 颈部嵌入通道数
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]  # 颈部头数
base_lr = 2e-4  # 基础学习率
weight_decay = 0.05  # 权重衰减
train_batch_size_per_gpu = 16  # 每个 GPU 的训练批次大小
load_from='pretrained_models/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-0e566235.pth'  # 加载预训练模型路径
persistent_workers = False  # 持久化工作进程

# 模型设置
model = dict(
    type='YOLOWorldDetector',  # 模型类型
    mm_neck=True,  # 多模态颈部
    num_train_classes=num_training_classes,  # 训练类别数
    num_test_classes=num_classes,  # 测试类别数
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),  # 数据预处理器
    backbone=dict(
        _delete_=True,  # 删除原有的设置
        type='MultiModalYOLOBackbone',  # 多模态 YOLO 骨干网络
        image_model={{_base_.model.backbone}},  # 图像模型
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',  # 文本模型
            model_name='openai/clip-vit-base-patch32',  # 模型名称
            frozen_modules=['all'])),  # 冻结模块
    neck=dict(type='YOLOWorldDualPAFPN',  # 颈部设置
              guide_channels=text_channels,  # 引导通道数
              embed_channels=neck_embed_channels,  # 嵌入通道数
              num_heads=neck_num_heads,  # 头数
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),  # 块配置
              text_enhancder=dict(type='ImagePoolingAttentionModule',  # 文本增强器
                                  embed_channels=256,  # 嵌入通道数
                                  num_heads=8)),  # 头数
    bbox_head=dict(type='YOLOWorldHead',  # 边界框头部设置
                   head_module=dict(type='YOLOWorldHeadModule',  # 头部模块设置
                                    embed_dims=text_channels,  # 嵌入维度
                                    num_classes=num_training_classes)),  # 训练类别数
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))  # 训练配置

# 数据集设置
# 定义文本转换器，包含随机加载文本和打包检测输入两个步骤
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]

# 定义马赛克仿射变换器，包含多模态马赛克、YOLOv5复制粘贴和YOLOv5随机仿射三个步骤
mosaic_affine_transform = [
    dict(
        type='MultiModalMosaic',
        img_scale=_base_.img_scale,
        pad_val=114.0,
        pre_transform=_base_.pre_transform),
    dict(type='YOLOv5CopyPaste', prob=_base_.copypaste_prob),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - _base_.affine_scale,
                             1 + _base_.affine_scale),
        # img_scale is (width, height)
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=_base_.min_area_ratio,
        use_mask_refine=_base_.use_mask2refine)
]

# 定义训练管道，包含基础预处理、马赛克仿射变换、YOLOv5多模态混合等步骤
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

# 定义第二阶段训练管道，包含文本转换器
train_pipeline_stage2 = [
    *_base_.train_pipeline_stage2[:-1],
    *text_transform
]

# 定义COCO训练数据集，包含多模态数据集和管道
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

# 定义训练数据加载器，包含持久化工作进程设置
train_dataloader = dict(
    persistent_workers=persistent_workers,
    # 设置每个 GPU 的训练批量大小
    batch_size=train_batch_size_per_gpu,
    # 设置数据集的拼接函数为 yolow_collate
    collate_fn=dict(type='yolow_collate'),
    # 设置数据集为 coco_train_dataset
    dataset=coco_train_dataset)
# 定义测试数据处理流程，包括加载文本和打包检测输入
test_pipeline = [
    *_base_.test_pipeline[:-1],  # 复制基础测试数据处理流程，去掉最后一个元素
    dict(type='LoadText'),  # 加载文本数据
    dict(
        type='mmdet.PackDetInputs',  # 打包检测输入数据
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'texts'))  # 指定元数据的键值
]
# 定义 COCO 验证数据集，包括数据集信息、类别文本路径和数据处理流程
coco_val_dataset = dict(
    _delete_=True,  # 删除原有的数据集配置
    type='MultiModalDataset',  # 多模态数据集类型
    dataset=dict(
        type='YOLOv5CocoDataset',  # 使用 YOLOv5 格式的 COCO 数据集
        data_root='data/coco',  # 数据根目录
        ann_file='annotations/instances_val2017.json',  # 标注文件路径
        data_prefix=dict(img='val2017/'),  # 图像数据前缀
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),  # 数据过滤配置
    class_text_path='data/texts/coco_class_texts.json',  # 类别文本路径
    pipeline=test_pipeline)  # 数据处理流程
# 定义验证数据加载器，使用 COCO 验证数据集
val_dataloader = dict(dataset=coco_val_dataset)
# 测试数据加载器与验证数据加载器相同
test_dataloader = val_dataloader

# 训练设置
default_hooks = dict(
    param_scheduler=dict(
        scheduler_type='linear',  # 使用线性学习率调度器
        lr_factor=0.01,  # 学习率因子
        max_epochs=max_epochs),  # 最大训练轮数
    checkpoint=dict(
        max_keep_ckpts=-1,  # 保留的最大检查点数
        save_best=None,  # 保存最佳模型的配置
        interval=save_epoch_intervals))  # 保存检查点的间隔
custom_hooks = [
    dict(
        type='EMAHook',  # 指数移动平均钩子
        ema_type='ExpMomentumEMA',  # 指数动量 EMA 类型
        momentum=0.0001,  # 动量参数
        update_buffers=True,  # 更新缓冲区
        strict_load=False,  # 严格加载
        priority=49),  # 优先级
    dict(
        type='mmdet.PipelineSwitchHook',  # 数据处理流程切换钩子
        switch_epoch=max_epochs - close_mosaic_epochs,  # 切换数据处理流程的轮数
        switch_pipeline=train_pipeline_stage2)  # 切换后的数据处理流程
]
train_cfg = dict(
    max_epochs=max_epochs,  # 最大训练轮数
    val_interval=5,  # 验证间隔
    dynamic_intervals=[((max_epochs - close_mosaic_epochs), _base_.val_interval_stage2)])  # 动态间隔设置
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,  # 删除原有的优化器配置
        type='AdamW',  # 使用 AdamW 优化器
        lr=base_lr,  # 基础学习率
        weight_decay=weight_decay,  # 权重衰减
        batch_size_per_gpu=train_batch_size_per_gpu),  # 每个 GPU 的批处理大小
    paramwise_cfg=dict(
        bias_decay_mult=0.0,  # 偏置衰减倍数
        norm_decay_mult=0.0,  # 归一化层衰减倍数
        custom_keys={'backbone.text_model': dict(lr_mult=0.01),  # 自定义键值对，指定学习率倍数
                     'logit_scale': dict(weight_decay=0.0)}),  # 自定义键值对，指定权重衰减
    constructor='YOLOWv5OptimizerConstructor')  # 优化器构造器

# 评估设置
# 创建一个字典，用于配置评估器的参数
val_evaluator = dict(
    # 标记是否删除
    _delete_=True,
    # 评估器类型为 mmdet.CocoMetric
    type='mmdet.CocoMetric',
    # 提议框数量的元组
    proposal_nums=(100, 1, 10),
    # COCO 数据集的标注文件路径
    ann_file='data/coco/annotations/instances_val2017.json',
    # 评估指标为 bbox
    metric='bbox')
```