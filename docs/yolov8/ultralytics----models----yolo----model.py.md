# `.\yolov8\ultralytics\models\yolo\model.py`

```py
# Ultralytics YOLO , AGPL-3.0 license

from pathlib import Path  # 导入路径操作模块Path

from ultralytics.engine.model import Model  # 导入Ultralytics的模型基类Model
from ultralytics.models import yolo  # 导入Ultralytics的YOLO模块
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel  # 导入Ultralytics的不同任务模型
from ultralytics.utils import ROOT, yaml_load  # 导入Ultralytics的工具函数ROOT和yaml_load


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        """Initialize YOLO model, switching to YOLOWorld if model filename contains '-world'."""
        path = Path(model)  # 使用路径模块Path创建路径对象path，指定模型文件名
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # 如果模型文件名包含'-world'并且文件类型是'.pt', '.yaml', '.yml'
            new_instance = YOLOWorld(path, verbose=verbose)  # 创建YOLOWorld的实例new_instance，传入模型路径和是否详细输出参数
            self.__class__ = type(new_instance)  # 设置当前对象的类为new_instance的类
            self.__dict__ = new_instance.__dict__  # 将当前对象的字典设置为new_instance的字典
        else:
            # Continue with default YOLO initialization
            super().__init__(model=model, task=task, verbose=verbose)  # 使用默认的YOLO模型初始化过程

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,  # 分类任务的模型类
                "trainer": yolo.classify.ClassificationTrainer,  # 分类任务的训练器类
                "validator": yolo.classify.ClassificationValidator,  # 分类任务的验证器类
                "predictor": yolo.classify.ClassificationPredictor,  # 分类任务的预测器类
            },
            "detect": {
                "model": DetectionModel,  # 检测任务的模型类
                "trainer": yolo.detect.DetectionTrainer,  # 检测任务的训练器类
                "validator": yolo.detect.DetectionValidator,  # 检测任务的验证器类
                "predictor": yolo.detect.DetectionPredictor,  # 检测任务的预测器类
            },
            "segment": {
                "model": SegmentationModel,  # 分割任务的模型类
                "trainer": yolo.segment.SegmentationTrainer,  # 分割任务的训练器类
                "validator": yolo.segment.SegmentationValidator,  # 分割任务的验证器类
                "predictor": yolo.segment.SegmentationPredictor,  # 分割任务的预测器类
            },
            "pose": {
                "model": PoseModel,  # 姿态估计任务的模型类
                "trainer": yolo.pose.PoseTrainer,  # 姿态估计任务的训练器类
                "validator": yolo.pose.PoseValidator,  # 姿态估计任务的验证器类
                "predictor": yolo.pose.PosePredictor,  # 姿态估计任务的预测器类
            },
            "obb": {
                "model": OBBModel,  # 目标边界框任务的模型类
                "trainer": yolo.obb.OBBTrainer,  # 目标边界框任务的训练器类
                "validator": yolo.obb.OBBValidator,  # 目标边界框任务的验证器类
                "predictor": yolo.obb.OBBPredictor,  # 目标边界框任务的预测器类
            },
        }


class YOLOWorld(Model):
    """YOLO-World object detection model."""
    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initializes the YOLOv8-World model with the given pre-trained model file. Supports *.pt and *.yaml formats.

        Args:
            model (str | Path): Path to the pre-trained model. Defaults to 'yolov8s-world.pt'.
        """
        # 调用父类的初始化方法，传入模型路径和任务类型为'detect'，同时设置是否详细输出信息
        super().__init__(model=model, task="detect", verbose=verbose)

        # 如果模型对象没有属性 'names'，则加载默认的 COCO 类别名称
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        # 返回一个字典，映射任务类型为'detect'时对应的模型类、验证器类、预测器类和训练器类
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes):
        """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e. ["person"].
        """
        # 调用模型对象的设置类别方法，设置新的类别列表
        self.model.set_classes(classes)
        
        # 如果类别列表中包含背景类别，将其移除
        background = " "
        if background in classes:
            classes.remove(background)
        
        # 更新模型的类别名称为新的类别列表
        self.model.names = classes

        # 重置预测器对象的类别名称
        if self.predictor:
            self.predictor.model.names = classes
```