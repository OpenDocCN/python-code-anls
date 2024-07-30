# `.\yolov8\ultralytics\engine\trainer.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolov8n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

# 导入需要的库和模块
import gc  # 垃圾回收模块
import math  # 数学计算模块
import os  # 系统操作模块
import subprocess  # 子进程管理模块
import time  # 时间操作模块
import warnings  # 警告处理模块
from copy import deepcopy  # 深拷贝函数
from datetime import datetime, timedelta  # 日期时间操作模块
from pathlib import Path  # 路径操作模块

import numpy as np  # 数组操作模块
import torch  # PyTorch深度学习库
from torch import distributed as dist  # 分布式训练模块
from torch import nn, optim  # 神经网络模块，优化器模块

# 导入Ultralytics库中的各种功能和工具
from ultralytics.cfg import get_cfg, get_save_dir  # 获取配置和保存目录
from ultralytics.data.utils import check_cls_dataset, check_det_dataset  # 检查分类和检测数据集的实用工具
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights  # 加载模型权重的函数
from ultralytics.utils import (
    DEFAULT_CFG,  # 默认配置
    LOGGER,  # 日志记录器
    RANK,  # 排名
    TQDM,  # 进度条
    __version__,  # Ultralytics版本号
    callbacks,  # 回调函数
    clean_url,  # 清理URL
    colorstr,  # 字符串颜色处理
    emojis,  # 表情符号
    yaml_save,  # 保存YAML文件
)
from ultralytics.utils.autobatch import check_train_batch_size  # 检查训练批次大小
from ultralytics.utils.checks import (
    check_amp,  # 检查AMP（自动混合精度）支持
    check_file,  # 检查文件
    check_imgsz,  # 检查图像尺寸
    check_model_file_from_stem,  # 从stem检查模型文件
    print_args,  # 打印参数
)
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command  # 分布式数据并行清理和命令生成
from ultralytics.utils.files import get_latest_run  # 获取最新运行文件
from ultralytics.utils.torch_utils import (
    EarlyStopping,  # 早停
    ModelEMA,  # 模型指数移动平均
    autocast,  # 自动类型转换
    convert_optimizer_state_dict_to_fp16,  # 将优化器状态转换为FP16
    init_seeds,  # 初始化随机种子
    one_cycle,  # 单循环学习率策略
    select_device,  # 选择设备
    strip_optimizer,  # 剥离优化器
    torch_distributed_zero_first,  # 分布式训练的首次清零
)

# 定义基础训练器类
class BaseTrainer:
    """
    BaseTrainer.

    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        trainset (torch.utils.data.Dataset): Training dataset.
        testset (torch.utils.data.Dataset): Testing dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
    """
    # 初始化 BaseTrainer 类的构造函数
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
            _callbacks (list, optional): List of callbacks. Defaults to None.
        """
        # 获取配置参数
        self.args = get_cfg(cfg, overrides)
        # 检查是否需要恢复训练状态
        self.check_resume(overrides)
        # 选择设备
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None  # 初始化验证器为 None
        self.metrics = None  # 初始化指标为 None
        self.plots = {}  # 初始化图形字典为空

        # 初始化随机种子
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # 目录设置
        self.save_dir = get_save_dir(self.args)  # 获取保存目录
        self.args.name = self.save_dir.name  # 更新日志记录器的名称
        self.wdir = self.save_dir / "weights"  # 权重保存目录
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # 创建目录
            self.args.save_dir = str(self.save_dir)  # 设置保存目录路径
            yaml_save(self.save_dir / "args.yaml", vars(self.args))  # 保存运行参数到 YAML 文件
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # 检查点文件路径
        self.save_period = self.args.save_period  # 保存周期设置

        self.batch_size = self.args.batch  # 批量大小
        self.epochs = self.args.epochs  # 训练周期数
        self.start_epoch = 0  # 起始周期
        if RANK == -1:
            print_args(vars(self.args))  # 打印参数信息

        # 设备设置
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # 若使用 CPU 或者 mps 设备，设置 workers 为 0，提升训练速度

        # 模型和数据集
        self.model = check_model_file_from_stem(self.args.model)  # 检查模型文件，添加后缀（例如 yolov8n -> yolov8n.pt）
        with torch_distributed_zero_first(RANK):  # 避免多次自动下载数据集
            self.trainset, self.testset = self.get_dataset()  # 获取训练集和测试集
        self.ema = None  # 指数移动平均参数初始化为 None

        # 优化器工具初始化
        self.lf = None  # 损失函数初始化为 None
        self.scheduler = None  # 调度器初始化为 None

        # 每个周期的指标
        self.best_fitness = None  # 最佳适应度初始化为 None
        self.fitness = None  # 适应度初始化为 None
        self.loss = None  # 损失初始化为 None
        self.tloss = None  # 总损失初始化为 None
        self.loss_names = ["Loss"]  # 损失名称列表
        self.csv = self.save_dir / "results.csv"  # CSV 文件路径
        self.plot_idx = [0, 1, 2]  # 绘图索引

        # HUB
        self.hub_session = None  # HUB 会话初始化为 None

        # 回调函数
        self.callbacks = _callbacks or callbacks.get_default_callbacks()  # 获取默认或自定义回调函数列表
        if RANK in {-1, 0}:
            callbacks.add_integration_callbacks(self)  # 添加集成回调函数

    def add_callback(self, event: str, callback):
        """Appends the given callback to the event."""
        self.callbacks[event].append(callback)  # 将给定的回调函数添加到事件对应的回调函数列表

    def set_callback(self, event: str, callback):
        """Sets the given callback for the event, overriding any existing callbacks."""
        self.callbacks[event] = [callback]  # 设置给定事件的回调函数，覆盖现有的所有回调函数

    def run_callbacks(self, event: str):
        """Runs all existing callbacks associated with the given event."""
        for callback in self.callbacks.get(event, []):  # 遍历给定事件关联的所有回调函数
            callback(self)  # 执行回调函数
    def train(self):
        """
        Allow device='', device=None on Multi-GPU systems to default to device=0.
        """
        # Determine the number of GPUs to be used based on the provided device configuration
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0

        # Run subprocess if DDP (Distributed Data Parallel) training is enabled and not in a subprocess
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # Argument checks and adjustments for Multi-GPU training
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                LOGGER.warning(
                    "WARNING ⚠️ 'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting "
                    "default 'batch=16'"
                )
                self.args.batch = 16

            # Generate and execute the DDP command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f'{colorstr("DDP:")} debug command {" ".join(cmd)}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                # Clean up DDP resources
                ddp_cleanup(self, str(file))

        else:
            # Perform single-GPU or CPU training
            self._do_train(world_size)

    def _setup_scheduler(self):
        """
        Initialize training learning rate scheduler based on arguments.
        """
        if self.args.cos_lr:
            # Use cosine learning rate schedule
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            # Use linear learning rate schedule
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        # Initialize LambdaLR scheduler with the defined learning rate function
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        """
        Initializes and configures DistributedDataParallel (DDP) for training.
        """
        # Set CUDA device for the current process rank
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        # Configure environment for DDP training
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        # Initialize the process group for distributed training
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size,
        )
    def save_model(self):
        """
        Save model training checkpoints with additional metadata.
        """
        import io  # 导入io模块，用于处理字节流操作

        import pandas as pd  # 导入pandas模块，用于处理CSV文件和数据结构

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()  # 创建一个字节流缓冲区对象
        torch.save(
            {
                "epoch": self.epoch,  # 保存当前训练轮次
                "best_fitness": self.best_fitness,  # 保存最佳训练效果指标
                "model": None,  # 模型信息（由EMA派生的恢复和最终检查点）
                "ema": deepcopy(self.ema.ema).half(),  # 深度复制EMA对象的EMA部分，并转换为半精度
                "updates": self.ema.updates,  # 保存EMA对象的更新次数
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),  # 转换优化器状态字典为半精度
                "train_args": vars(self.args),  # 保存训练参数的字典形式
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},  # 保存训练指标和适应度指标
                "train_results": {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient="list").items()},  # 读取CSV文件并将结果保存为字典
                "date": datetime.now().isoformat(),  # 保存当前时间的ISO格式字符串
                "version": __version__,  # 保存当前版本号
                "license": "AGPL-3.0 (https://ultralytics.com/license)",  # 许可证信息
                "docs": "https://docs.ultralytics.com",  # 文档链接
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # 获取序列化后的检查点内容以保存

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # 保存最后的检查点文件（last.pt）
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # 若当前适应度等于最佳适应度，则保存最佳检查点文件（best.pt）
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # 根据当前轮次保存检查点文件，如'epoch3.pt'

    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        try:
            if self.args.task == "classify":  # 如果任务为分类
                data = check_cls_dataset(self.args.data)  # 检查并获取分类数据集
            elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:  # 如果数据文件格式为yaml或yml，或者任务是检测、分割、姿态或obb
                data = check_det_dataset(self.args.data)  # 检查并获取检测数据集
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # 更新self.args.data以验证'yolo train data=url.zip'的使用
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e  # 抛出数据集错误异常
        self.data = data  # 将获取的数据集保存到self.data
        return data["train"], data.get("val") or data.get("test")  # 返回训练集路径，如果不存在则返回验证集或测试集路径
    def setup_model(self):
        """
        Load/create/download model for any task.

        Checks if a model is already loaded. If not, attempts to load weights from a '.pt' file or a specified pretrained path,
        then initializes the model using the loaded configuration and weights.
        """
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            # Attempt to load weights and configuration from a '.pt' file
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            # Attempt to load weights from a specified pretrained path
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        
        # Initialize the model using the retrieved configuration and weights
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """
        Perform a single step of the training optimizer with gradient clipping and EMA update.

        Unscales gradients, clips gradients to a maximum norm of 10.0, performs optimizer step, updates scaler,
        and zeroes the gradients of the optimizer. If an Exponential Moving Average (EMA) is enabled, updates the EMA.
        """
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """
        Allows custom preprocessing model inputs and ground truths depending on task type.

        This function can be overridden to implement custom preprocessing logic for batches of data before feeding them
        into the model.
        """
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator.

        Returns metrics and fitness. Updates the best fitness if a new best fitness is found during validation.
        """
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Get model and raise NotImplementedError for loading cfg files.

        Raises NotImplementedError indicating that loading configuration files (cfg) is not supported by this method.
        """
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """
        Returns a NotImplementedError when the get_validator function is called.

        Raises NotImplementedError indicating that the get_validator function is not implemented in this trainer.
        """
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """
        Returns dataloader derived from torch.data.Dataloader.

        Raises NotImplementedError indicating that the get_dataloader function is not implemented in this trainer.
        """
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build dataset.

        Raises NotImplementedError indicating that the build_dataset function is not implemented in this trainer.
        """
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        This function provides a labeled dictionary of loss items. It's particularly useful for tasks like segmentation and detection.
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]
    # 设置或更新模型训练前的参数
    def set_model_attributes(self):
        """To set or update model parameters before training."""
        self.model.names = self.data["names"]

    # 构建用于训练 YOLO 模型的目标张量
    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        pass

    # 返回描述训练进度的字符串
    def progress_string(self):
        """Returns a string describing training progress."""
        return ""

    # TODO: 可能需要将以下函数放入回调函数中
    # 绘制 YOLO 训练过程中的训练样本图像
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLO training."""
        pass

    # 绘制 YOLO 模型的训练标签
    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    # 将训练指标保存到 CSV 文件中
    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = "" if self.csv.exists() else (("%23s," * n % tuple(["epoch"] + keys)).rstrip(",") + "\n")  # header
        with open(self.csv, "a") as f:
            f.write(s + ("%23.5g," * n % tuple([self.epoch + 1] + vals)).rstrip(",") + "\n")

    # 绘制并可视化训练指标
    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    # 注册绘图信息（例如供回调函数使用）
    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    # 执行物体检测 YOLO 模型的最终评估和验证
    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # 去除优化器信息
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")
    # 检查是否存在恢复检查点，并根据需要更新参数
    def check_resume(self, overrides):
        # 获取恢复路径
        resume = self.args.resume
        # 如果有恢复路径
        if resume:
            try:
                # 检查恢复路径是否是字符串或路径对象并且存在
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                # 获取最新的运行检查点路径
                last = Path(check_file(resume) if exists else get_latest_run())

                # 检查恢复的数据 YAML 文件是否存在，否则强制重新下载数据集
                ckpt_args = attempt_load_weights(last).args
                if not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                # 标记为需要恢复
                resume = True
                # 获取配置对象，并更新参数
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # 恢复模型路径
                # 遍历可能需要更新的参数，例如减少内存占用或更新设备
                for k in "imgsz", "batch", "device":
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                # 如果出现异常，抛出文件未找到的错误
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        # 将恢复状态更新到对象中
        self.resume = resume

    # 从给定的检查点恢复 YOLO 训练
    def resume_training(self, ckpt):
        # 如果检查点为空或者没有设置恢复标志，则直接返回
        if ckpt is None or not self.resume:
            return
        # 初始化最佳适应度和开始的 epoch
        best_fitness = 0.0
        start_epoch = ckpt.get("epoch", -1) + 1
        # 如果检查点中包含优化器状态，则加载优化器状态
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # 加载优化器状态
            best_fitness = ckpt["best_fitness"]
        # 如果存在 EMA 并且检查点中包含 EMA 数据，则加载 EMA 状态
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # 加载 EMA 状态
            self.ema.updates = ckpt["updates"]
        # 确保开始的 epoch 大于 0，否则输出错误信息
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        # 记录恢复训练的信息
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        # 如果总 epoch 小于开始的 epoch，则输出调试信息进行微调
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # 进行额外的微调 epoch
        # 更新最佳适应度和开始的 epoch
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        # 如果开始的 epoch 大于总 epoch 减去关闭混合图块的数量，则关闭数据加载器中的混合图块
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()
    # 定义一个方法，用于关闭数据加载器的马赛克增强功能
    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        # 检查训练数据加载器的数据集是否具有 'mosaic' 属性
        if hasattr(self.train_loader.dataset, "mosaic"):
            # 将数据集的 'mosaic' 属性设置为 False，停止使用马赛克增强
            self.train_loader.dataset.mosaic = False
        # 再次检查训练数据加载器的数据集是否具有 'close_mosaic' 方法
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            # 记录信息到日志，指示正在关闭数据加载器的马赛克增强
            LOGGER.info("Closing dataloader mosaic")
            # 调用数据集的 'close_mosaic' 方法，关闭马赛克增强，并传递额外的参数 'hyp=self.args'
            self.train_loader.dataset.close_mosaic(hyp=self.args)
```