# `.\MinerU\magic_pdf\model\pek_sub_modules\layoutlmv3\model_init.py`

```
# 从当前包导入可视化器
from .visualizer import Visualizer
# 从当前包导入 RCNN 和 VL 模块
from .rcnn_vl import *
from .backbone import *

# 从 detectron2 导入配置相关函数
from detectron2.config import get_cfg
# 从 detectron2 导入配置节点类
from detectron2.config import CfgNode as CN
# 从 detectron2 导入元数据和数据集注册功能
from detectron2.data import MetadataCatalog, DatasetCatalog
# 从 detectron2 注册 COCO 数据集实例
from detectron2.data.datasets import register_coco_instances
# 从 detectron2 导入训练器和其他基础功能
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, DefaultPredictor


def add_vit_config(cfg):
    """
    为 VIT 添加配置。
    """
    _C = cfg  # 将 cfg 赋值给 _C，方便后续操作

    _C.MODEL.VIT = CN()  # 创建 VIT 模型的配置节点

    # CoaT 模型名称。
    _C.MODEL.VIT.NAME = ""  # 初始化 CoaT 模型名称为空

    # 从 CoaT 主干网络输出特征。
    _C.MODEL.VIT.OUT_FEATURES = ["layer3", "layer5", "layer7", "layer11"]  # 指定输出特征层

    _C.MODEL.VIT.IMG_SIZE = [224, 224]  # 设置输入图像大小

    _C.MODEL.VIT.POS_TYPE = "shared_rel"  # 设置位置类型为共享关系

    _C.MODEL.VIT.DROP_PATH = 0.  # 设置 Drop Path 的比率为 0

    _C.MODEL.VIT.MODEL_KWARGS = "{}"  # 初始化模型的关键词参数为空字典

    _C.SOLVER.OPTIMIZER = "ADAMW"  # 设置优化器为 ADAMW

    _C.SOLVER.BACKBONE_MULTIPLIER = 1.0  # 设置主干网络的倍增因子为 1.0

    _C.AUG = CN()  # 创建数据增强的配置节点

    _C.AUG.DETR = False  # 设置 DETR 增强为 False

    _C.MODEL.IMAGE_ONLY = True  # 设置模型为仅处理图像

    # 各种数据目录的初始化为空字符串
    _C.PUBLAYNET_DATA_DIR_TRAIN = ""  
    _C.PUBLAYNET_DATA_DIR_TEST = ""  
    _C.FOOTNOTE_DATA_DIR_TRAIN = ""  
    _C.FOOTNOTE_DATA_DIR_VAL = ""  
    _C.SCIHUB_DATA_DIR_TRAIN = ""  
    _C.SCIHUB_DATA_DIR_TEST = ""  
    _C.JIAOCAI_DATA_DIR_TRAIN = ""  
    _C.JIAOCAI_DATA_DIR_TEST = ""  
    _C.ICDAR_DATA_DIR_TRAIN = ""  
    _C.ICDAR_DATA_DIR_TEST = ""  
    _C.M6DOC_DATA_DIR_TEST = ""  
    _C.DOCSTRUCTBENCH_DATA_DIR_TEST = ""  
    _C.DOCSTRUCTBENCHv2_DATA_DIR_TEST = ""  
    _C.CACHE_DIR = ""  # 设置缓存目录为空字符串
    _C.MODEL.CONFIG_PATH = ""  # 设置模型配置路径为空字符串

    # 有效更新步骤将是 MAX_ITER/GRADIENT_ACCUMULATION_STEPS
    # 可能需要将 MAX_ITER *= GRADIENT_ACCUMULATION_STEPS
    _C.SOLVER.GRADIENT_ACCUMULATION_STEPS = 1  # 设置梯度累积步骤为 1


def setup(args, device):
    """
    创建配置并执行基本设置。
    """
    cfg = get_cfg()  # 获取默认配置

    # add_coat_config(cfg)  # 添加 CoA_T 配置（注释掉的功能）
    add_vit_config(cfg)  # 添加 VIT 配置
    cfg.merge_from_file(args.config_file)  # 从配置文件合并设置
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # 设置模型的测试阈值
    cfg.merge_from_list(args.opts)  # 从命令行参数合并设置

    # 使用统一的设备配置
    cfg.MODEL.DEVICE = device  # 设置模型设备

    cfg.freeze()  # 冻结配置，防止进一步修改
    default_setup(cfg, args)  # 执行默认设置

    # @todo 可以删掉这块？
    # 注册 COCO 数据集实例（被注释掉的代码）
    # register_coco_instances(
    #     "scihub_train",
    #     {},
    #     cfg.SCIHUB_DATA_DIR_TRAIN + ".json",
    #     cfg.SCIHUB_DATA_DIR_TRAIN
    # )

    return cfg  # 返回配置对象


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)  # 初始化父类字典

    def __getattr__(self, key):
        if key not in self.keys():  # 检查键是否在字典中
            return None  # 如果不在，则返回 None
        value = self[key]  # 获取对应的值
        if isinstance(value, dict):  # 如果值是字典
            value = DotDict(value)  # 将其转化为 DotDict 对象
        return value  # 返回值

    def __setattr__(self, key, value):
        self[key] = value  # 设置字典的属性


class Layoutlmv3_Predictor(object):  # 定义 Layoutlmv3 预测器类
    # 初始化方法，接收权重、配置文件和设备作为参数
    def __init__(self, weights, config_file, device):
        # 设置布局参数的字典，包括配置文件及其他选项
        layout_args = {
            "config_file": config_file,
            "resume": False,  # 不从之前的检查点恢复
            "eval_only": False,  # 不仅仅用于评估
            "num_gpus": 1,  # 使用的 GPU 数量
            "num_machines": 1,  # 使用的机器数量
            "machine_rank": 0,  # 当前机器的排名
            "dist_url": "tcp://127.0.0.1:57823",  # 分布式训练的 URL
            "opts": ["MODEL.WEIGHTS", weights],  # 指定模型权重
        }
        # 将布局参数转换为 DotDict 以支持点式访问
        layout_args = DotDict(layout_args)
    
        # 设置配置并返回配置对象
        cfg = setup(layout_args, device)
        # 定义类别映射列表
        self.mapping = ["title", "plain text", "abandon", "figure", "figure_caption", "table", "table_caption",
                        "table_footnote", "isolate_formula", "formula_caption"]
        # 设置训练数据集的类标签
        MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = self.mapping
        # 创建默认预测器实例
        self.predictor = DefaultPredictor(cfg)
    
    # 使对象可调用的方法，接收图像和可选的类别 ID 列表
    def __call__(self, image, ignore_catids=[]):
        # page_layout_result = {
        #     "layout_dets": []  # 原本的布局检测结果结构（已注释）
        # }
        # 初始化布局检测列表
        layout_dets = []
        # 使用预测器处理图像，获得输出结果
        outputs = self.predictor(image)
        # 获取预测框的坐标列表，并将其转换为 CPU 列表
        boxes = outputs["instances"].to("cpu")._fields["pred_boxes"].tensor.tolist()
        # 获取预测的类别标签，并转换为 CPU 列表
        labels = outputs["instances"].to("cpu")._fields["pred_classes"].tolist()
        # 获取预测分数，并转换为 CPU 列表
        scores = outputs["instances"].to("cpu")._fields["scores"].tolist()
        # 遍历每个预测框的索引
        for bbox_idx in range(len(boxes)):
            # 如果标签在忽略的类别 ID 中，则跳过
            if labels[bbox_idx] in ignore_catids:
                continue
            # 将符合条件的框和相关信息添加到布局检测列表
            layout_dets.append({
                "category_id": labels[bbox_idx],  # 类别 ID
                "poly": [  # 多边形顶点坐标
                    boxes[bbox_idx][0], boxes[bbox_idx][1],
                    boxes[bbox_idx][2], boxes[bbox_idx][1],
                    boxes[bbox_idx][2], boxes[bbox_idx][3],
                    boxes[bbox_idx][0], boxes[bbox_idx][3],
                ],
                "score": scores[bbox_idx]  # 预测分数
            })
        # 返回布局检测结果列表
        return layout_dets
```