# `.\YOLO-World\configs\segmentation\yolo_world_seg_m_dual_vlpan_2e-4_80e_8gpus_allmodules_finetune_lvis.py`

```
_base_ = (
    '../../third_party/mmyolo/configs/yolov8/yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco.py'
)
# 定义基础配置文件路径
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)
# 自定义导入模块，禁止导入失败

# 超参数设置
num_classes = 1203
num_training_classes = 80
max_epochs = 80  # 最大训练周期数
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4

weight_decay = 0.05
train_batch_size_per_gpu = 8
load_from = 'pretrained_models/yolo_world_m_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-2b7bd1be.pth'
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
# 定义数据预处理的步骤列表，用于数据增强
pre_transform = [
    # 从文件加载图像
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    # 从文件加载标注信息，包括边界框和掩码
    dict(type='LoadAnnotations',
         with_bbox=True,
         with_mask=True,
         mask2bbox=True)
]

# 定义最后的数据转换步骤列表
last_transform = [
    # 使用 mmdet 库中的 Albu 进行数据增强
    dict(type='mmdet.Albu',
         transforms=_base_.albu_train_transforms,
         bbox_params=dict(type='BboxParams',
                          format='pascal_voc',
                          label_fields=['gt_bboxes_labels',
                                        'gt_ignore_flags']),
         keymap={
             'img': 'image',
             'gt_bboxes': 'bboxes'
         }),
    # 使用 YOLOv5HSVRandomAug 进行数据增强
    dict(type='YOLOv5HSVRandomAug'),
    # 随机翻转图像
    dict(type='mmdet.RandomFlip', prob=0.5),
    # 将多边形转换为掩码
    dict(type='Polygon2Mask',
         downsample_ratio=downsample_ratio,
         mask_overlap=mask_overlap),
]

# 数据集设置
text_transform = [
    # 随机加载文本
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    # 打包检测输入数据
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]

mosaic_affine_transform = [
    # 多模态镶嵌
    dict(type='MultiModalMosaic',
         img_scale=_base_.img_scale,
         pad_val=114.0,
         pre_transform=pre_transform),
    # YOLOv5CopyPaste 数据增强
    dict(type='YOLOv5CopyPaste', prob=_base_.copypaste_prob),
    # YOLOv5RandomAffine 数据增强
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        # 图像尺寸为 (宽度, 高度)
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=_base_.min_area_ratio,
        use_mask_refine=True)
]

# 训练管道
train_pipeline = [
    # 将预处理步骤和镶嵌仿射变换步骤合并
    *pre_transform, *mosaic_affine_transform,
    # 创建一个字典，指定模型类型为YOLOv5MultiModalMixUp，概率为mixup_prob
    dict(type='YOLOv5MultiModalMixUp',
         prob=_base_.mixup_prob,
         # 将pre_transform和mosaic_affine_transform的元素合并到一个列表中
         pre_transform=[*pre_transform, *mosaic_affine_transform]),
    # 将last_transform和text_transform的元素合并到一个列表中
    *last_transform, *text_transform
# 定义训练管道的第二阶段，包括预处理、YOLOv5KeepRatioResize、LetterResize、YOLOv5RandomAffine等操作
_train_pipeline_stage2 = [
    *pre_transform,  # 包含预处理操作
    dict(type='YOLOv5KeepRatioResize', scale=_base_.img_scale),  # YOLOv5KeepRatioResize操作
    dict(type='LetterResize',  # LetterResize操作
         scale=_base_.img_scale,
         allow_scale_up=True,
         pad_val=dict(img=114.0)),
    dict(type='YOLOv5RandomAffine',  # YOLOv5RandomAffine操作
         max_rotate_degree=0.0,
         max_shear_degree=0.0,
         scaling_ratio_range=(1 - _base_.affine_scale,
                              1 + _base_.affine_scale),
         max_aspect_ratio=_base_.max_aspect_ratio,
         border_val=(114, 114, 114),
         min_area_ratio=min_area_ratio,
         use_mask_refine=use_mask2refine), *last_transform  # 包含最后的变换操作
]
# 将_train_pipeline_stage2与text_transform合并为train_pipeline_stage2
train_pipeline_stage2 = [*_train_pipeline_stage2, *text_transform]
# 定义coco_train_dataset，包括数据集类型、数据根目录、注释文件、数据前缀等信息
coco_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5LVISV1Dataset',
                 data_root='data/coco',
                 ann_file='lvis/lvis_v1_train_base.json',
                 data_prefix=dict(img=''),
                 filter_cfg=dict(filter_empty_gt=True, min_size=32)),
    class_text_path='data/captions/lvis_v1_base_class_captions.json',
    pipeline=train_pipeline)  # 使用train_pipeline作为管道
# 定义train_dataloader，包括持久化工作进程、每个GPU的批量大小、数据集、数据收集函数等信息
train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)

# 定义测试管道，包括加载文本、PackDetInputs等操作
test_pipeline = [
    *_base_.test_pipeline[:-1],  # 包含基本测试管道的前几个操作
    dict(type='LoadText'),  # 加载文本操作
    dict(type='mmdet.PackDetInputs',  # PackDetInputs操作
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]

# 训练设置
# 默认的钩子配置，包括参数调度器和检查点
default_hooks = dict(param_scheduler=dict(scheduler_type='linear',
                                          lr_factor=0.01,
                                          max_epochs=max_epochs),
                     checkpoint=dict(max_keep_ckpts=-1,
                                     save_best=None,
                                     interval=save_epoch_intervals))
# 自定义的钩子配置列表
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
# 训练配置，包括最大训练轮数、验证间隔、动态间隔等
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=5,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])
# 优化器包装器配置，包括优化器类型、学习率、权重衰减等
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

# 评估设置，包括 COCO 验证数据集配置
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

# 创建验证评估器，类型为 'mmdet.LVISMetric'，使用 LVIS 验证数据集的注释文件，评估指标包括边界框和分割
val_evaluator = dict(type='mmdet.LVISMetric',
                     ann_file='data/coco/lvis/lvis_v1_val.json',
                     metric=['bbox', 'segm'])
# 将验证评估器赋值给测试评估器
test_evaluator = val_evaluator

# 设置参数为查找未使用的参数
find_unused_parameters = True
```