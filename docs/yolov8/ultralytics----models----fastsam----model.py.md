# `.\yolov8\ultralytics\models\fastsam\model.py`

```py
# 导入必要的模块和类
from pathlib import Path
from ultralytics.engine.model import Model
from .predict import FastSAMPredictor
from .val import FastSAMValidator

# 定义 FastSAM 类，继承自 Model 类
class FastSAM(Model):
    """
    FastSAM 模型接口。

    Example:
        ```python
        from ultralytics import FastSAM

        model = FastSAM('last.pt')
        results = model.predict('ultralytics/assets/bus.jpg')
        ```py
    """

    def __init__(self, model="FastSAM-x.pt"):
        """初始化方法，调用父类（YOLO）的 __init__ 方法，使用更新后的默认模型名称。"""
        # 如果模型名称为 "FastSAM.pt"，则修改为 "FastSAM-x.pt"
        if str(model) == "FastSAM.pt":
            model = "FastSAM-x.pt"
        # 断言模型文件的后缀不是 .yaml 或 .yml，因为 FastSAM 模型只支持预训练模型
        assert Path(model).suffix not in {".yaml", ".yml"}, "FastSAM models only support pre-trained models."
        # 调用父类的初始化方法，传入模型名称和任务类型 "segment"
        super().__init__(model=model, task="segment")

    def predict(self, source, stream=False, bboxes=None, points=None, labels=None, texts=None, **kwargs):
        """
        对给定的图像或视频源进行分割预测。

        Args:
            source (str): 图像或视频文件的路径，或者是 PIL.Image 对象，或者是 numpy.ndarray 对象。
            stream (bool, optional): 如果为 True，则启用实时流处理。默认为 False。
            bboxes (list, optional): 提供分割提示的边界框坐标列表。默认为 None。
            points (list, optional): 提供分割提示的点列表。默认为 None。
            labels (list, optional): 提供分割提示的标签列表。默认为 None。
            texts (list, optional): 提供分割提示的文本列表。默认为 None。

        Returns:
            (list): 模型的预测结果列表。
        """
        # 将提示信息组织成字典
        prompts = dict(bboxes=bboxes, points=points, labels=labels, texts=texts)
        # 调用父类的 predict 方法进行预测，并传入参数和提示信息
        return super().predict(source, stream, prompts=prompts, **kwargs)

    @property
    def task_map(self):
        """返回一个字典，将分割任务映射到相应的预测器和验证器类。"""
        return {"segment": {"predictor": FastSAMPredictor, "validator": FastSAMValidator}}
```