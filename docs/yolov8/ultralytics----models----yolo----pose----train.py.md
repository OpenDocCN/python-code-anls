# `.\yolov8\ultralytics\models\yolo\pose\train.py`

```py
# 导入所需的模块和函数
from copy import copy

# 导入 Ultralytics 中的 YOLO 模型相关内容
from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.plotting import plot_images, plot_results


# 定义一个名为 PoseTrainer 的类，继承自 DetectionTrainer 类
class PoseTrainer(yolo.detect.DetectionTrainer):
    """
    PoseTrainer 类，扩展自 DetectionTrainer 类，用于基于姿态模型进行训练。

    示例:
        ```python
        from ultralytics.models.yolo.pose import PoseTrainer

        args = dict(model='yolov8n-pose.pt', data='coco8-pose.yaml', epochs=3)
        trainer = PoseTrainer(overrides=args)
        trainer.train()
        ```py
    """

    # 初始化 PoseTrainer 对象，设置配置和覆盖参数
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a PoseTrainer object with specified configurations and overrides."""
        if overrides is None:
            overrides = {}
        # 将任务类型设置为 "pose"
        overrides["task"] = "pose"
        super().__init__(cfg, overrides, _callbacks)

        # 如果设备类型为 "mps"，则输出警告信息
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    # 获取带有指定配置和权重的姿态估计模型
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get pose estimation model with specified configuration and weights."""
        model = PoseModel(cfg, ch=3, nc=self.data["nc"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose)
        if weights:
            model.load(weights)

        return model

    # 设置 PoseModel 的关键点形状属性
    def set_model_attributes(self):
        """Sets keypoints shape attribute of PoseModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    # 获取 PoseValidator 类的实例，用于验证
    def get_validator(self):
        """Returns an instance of the PoseValidator class for validation."""
        # 设置损失名称
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return yolo.pose.PoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    # 绘制一批训练样本的图像，包括类标签、边界框和关键点的注释
    def plot_training_samples(self, batch, ni):
        """Plot a batch of training samples with annotated class labels, bounding boxes, and keypoints."""
        images = batch["img"]
        kpts = batch["keypoints"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]
        batch_idx = batch["batch_idx"]
        # 调用 plot_images 函数绘制图像
        plot_images(
            images,
            batch_idx,
            cls,
            bboxes,
            kpts=kpts,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    # 绘制训练/验证指标的图表
    def plot_metrics(self):
        """Plots training/val metrics."""
        # 调用 plot_results 函数绘制训练/验证指标的图表并保存为 results.png 文件
        plot_results(file=self.csv, pose=True, on_plot=self.on_plot)  # save results.png
```