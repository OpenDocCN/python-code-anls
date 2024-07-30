# `.\yolov8\ultralytics\models\yolo\classify\train.py`

```py
# 导入PyTorch库
import torch

# 从Ultralytics库中导入所需的模块和类
from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.utils.torch_utils import is_parallel, strip_optimizer, torch_distributed_zero_first

# 定义一个名为ClassificationTrainer的类，继承自BaseTrainer类，用于分类模型的训练
class ClassificationTrainer(BaseTrainer):
    """
    一个扩展了BaseTrainer类的类，用于基于分类模型的训练。

    Notes:
        - Torchvision的分类模型也可以通过'model'参数传递，例如model='resnet18'。

    Example:
        ```python
        from ultralytics.models.yolo.classify import ClassificationTrainer

        args = dict(model='yolov8n-cls.pt', data='imagenet10', epochs=3)
        trainer = ClassificationTrainer(overrides=args)
        trainer.train()
        ```py
    """

    # 初始化ClassificationTrainer对象，可选配置覆盖和回调函数
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a ClassificationTrainer object with optional configuration overrides and callbacks."""
        # 如果overrides为None，则设为一个空字典
        if overrides is None:
            overrides = {}
        # 设置任务为分类
        overrides["task"] = "classify"
        # 如果未设置imgsz，则设为224
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 224
        # 调用父类的初始化方法
        super().__init__(cfg, overrides, _callbacks)

    # 设置YOLO模型的类名，从加载的数据集中获取
    def set_model_attributes(self):
        """Set the YOLO model's class names from the loaded dataset."""
        self.model.names = self.data["names"]

    # 获取模型的方法，返回一个配置好的用于YOLO训练的修改后的PyTorch模型
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Returns a modified PyTorch model configured for training YOLO."""
        # 创建分类模型对象
        model = ClassificationModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        # 如果有指定的权重，则加载模型权重
        if weights:
            model.load(weights)

        # 遍历模型的所有模块
        for m in model.modules():
            # 如果未使用预训练模型并且模块具有reset_parameters方法，则重置其参数
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            # 如果模块是Dropout类型并且启用了dropout，则设置其p属性
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # set dropout
        # 设置模型所有参数为需要梯度计算（用于训练）
        for p in model.parameters():
            p.requires_grad = True  # for training
        return model

    # 设置模型的方法，加载、创建或下载模型，适用于任何任务
    def setup_model(self):
        """Load, create or download model for any task."""
        # 导入torchvision库，以便更快速地在作用域内导入ultralytics
        import torchvision  # scope for faster 'import ultralytics'

        # 如果self.model名称存在于torchvision.models.__dict__中
        if str(self.model) in torchvision.models.__dict__:
            # 使用指定的self.model创建torchvision中的模型实例
            self.model = torchvision.models.__dict__[self.model](
                weights="IMAGENET1K_V1" if self.args.pretrained else None
            )
            # 检查点设为None
            ckpt = None
        else:
            # 否则调用父类的setup_model方法，获取检查点
            ckpt = super().setup_model()
        # 调整分类模型的输出维度为self.data["nc"]
        ClassificationModel.reshape_outputs(self.model, self.data["nc"])
        return ckpt
    def build_dataset(self, img_path, mode="train", batch=None):
        """Creates a ClassificationDataset instance given an image path, and mode (train/test etc.)."""
        # 使用给定的图片路径和模式创建一个 ClassificationDataset 实例
        return ClassificationDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Returns PyTorch DataLoader with transforms to preprocess images for inference."""
        # 在分布式数据并行训练环境中，仅在 rank 为 0 时初始化 dataset 的缓存
        with torch_distributed_zero_first(rank):
            # 使用 build_dataset 方法创建数据集对象
            dataset = self.build_dataset(dataset_path, mode)

        # 使用 build_dataloader 函数创建 PyTorch DataLoader 对象，用于加载数据批次
        loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank)
        
        # 如果不是训练模式，将推理转换添加到模型中
        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        # 返回 DataLoader 对象
        return loader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images and classes."""
        # 将批次中的图像和类别转移到设备上（GPU）
        batch["img"] = batch["img"].to(self.device)
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def progress_string(self):
        """Returns a formatted string showing training progress."""
        # 返回格式化后的训练进度字符串，包括当前训练轮次、GPU内存占用和各种损失的名称
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def get_validator(self):
        """Returns an instance of ClassificationValidator for validation."""
        # 设置损失名称为 "loss" 并返回一个用于验证的 ClassificationValidator 实例
        self.loss_names = ["loss"]
        return yolo.classify.ClassificationValidator(self.test_loader, self.save_dir, _callbacks=self.callbacks)

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        # 根据指定的前缀和损失名称，返回带标签的训练损失字典
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(loss_items), 5)]
        return dict(zip(keys, loss_items))

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        # 从 CSV 文件中绘制指标图表，用于分类任务，并保存为 results.png
        plot_results(file=self.csv, classify=True, on_plot=self.on_plot)
    # 定义一个方法，用于评估训练好的模型并保存验证结果
    def final_eval(self):
        """Evaluate trained model and save validation results."""
        # 遍历最后和最佳模型文件
        for f in self.last, self.best:
            # 检查文件是否存在
            if f.exists():
                # 去除模型文件中的优化器信息
                strip_optimizer(f)  # strip optimizers
                # 如果当前文件是最佳模型文件
                if f is self.best:
                    # 记录信息：正在验证最佳模型文件
                    LOGGER.info(f"\nValidating {f}...")
                    # 设置验证器参数
                    self.validator.args.data = self.args.data
                    self.validator.args.plots = self.args.plots
                    # 使用验证器评估模型
                    self.metrics = self.validator(model=f)
                    # 移除评估结果中的 fitness 指标（如果存在）
                    self.metrics.pop("fitness", None)
                    # 执行回调函数：在每个训练周期结束时
                    self.run_callbacks("on_fit_epoch_end")
        # 记录信息：结果保存到指定目录
        LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")

    # 定义一个方法，用于绘制训练样本和它们的标注
    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        # 调用 plot_images 函数，绘制训练样本图片
        plot_images(
            images=batch["img"],
            # 创建一个张量，包含批次内所有图片的索引
            batch_idx=torch.arange(len(batch["img"])),
            # 警告：对于分类模型，使用 .view() 而不是 .squeeze() 来展平类别数据
            cls=batch["cls"].view(-1),
            # 图片文件名，保存在指定目录下，并包含批次号
            fname=self.save_dir / f"train_batch{ni}.jpg",
            # 在绘图时执行的回调函数
            on_plot=self.on_plot,
        )
```