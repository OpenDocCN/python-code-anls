# `.\yolov8\ultralytics\engine\tuner.py`

```py
"""
This module provides functionalities for hyperparameter tuning of the Ultralytics YOLO models for object detection,
instance segmentation, image classification, pose estimation, and multi-object tracking.

Hyperparameter tuning is the process of systematically searching for the optimal set of hyperparameters
that yield the best model performance. This is particularly crucial in deep learning models like YOLO,
where small changes in hyperparameters can lead to significant differences in model accuracy and efficiency.

Example:
    Tune hyperparameters for YOLOv8n on COCO8 at imgsz=640 and epochs=30 for 300 tuning iterations.
    ```python
    from ultralytics import YOLO

    model = YOLO('yolov8n.pt')
    model.tune(data='coco8.yaml', epochs=10, iterations=300, optimizer='AdamW', plots=False, save=False, val=False)
    ```py
"""

import random
import shutil
import subprocess
import time

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG, LOGGER, callbacks, colorstr, remove_colorstr, yaml_print, yaml_save
from ultralytics.utils.plotting import plot_tune_results


class Tuner:
    """
    Class responsible for hyperparameter tuning of YOLO models.

    The class evolves YOLO model hyperparameters over a given number of iterations
    by mutating them according to the search space and retraining the model to evaluate their performance.

    Attributes:
        space (dict): Hyperparameter search space containing bounds and scaling factors for mutation.
        tune_dir (Path): Directory where evolution logs and results will be saved.
        tune_csv (Path): Path to the CSV file where evolution logs are saved.

    Methods:
        _mutate(hyp: dict) -> dict:
            Mutates the given hyperparameters within the bounds specified in `self.space`.

        __call__():
            Executes the hyperparameter evolution across multiple iterations.

    Example:
        Tune hyperparameters for YOLOv8n on COCO8 at imgsz=640 and epochs=30 for 300 tuning iterations.
        ```python
        from ultralytics import YOLO

        model = YOLO('yolov8n.pt')
        model.tune(data='coco8.yaml', epochs=10, iterations=300, optimizer='AdamW', plots=False, save=False, val=False)
        ```py

        Tune with custom search space.
        ```python
        from ultralytics import YOLO

        model = YOLO('yolov8n.pt')
        model.tune(space={key1: val1, key2: val2})  # custom search space dictionary
        ```
    """
    def __init__(self, args=DEFAULT_CFG, _callbacks=None):
        """
        Initialize the Tuner with configurations.

        Args:
            args (dict, optional): Configuration for hyperparameter evolution.
        """
        # 将参数中的'space'键弹出，如果不存在则使用默认空间字典
        self.space = args.pop("space", None) or {  # key: (min, max, gain(optional))
            # 初始学习率范围 (例如 SGD=1E-2, Adam=1E-3)
            "lr0": (1e-5, 1e-1),
            # 最终的 OneCycleLR 学习率范围 (lr0 * lrf)
            "lrf": (0.0001, 0.1),
            # SGD 动量/Adam beta1 范围
            "momentum": (0.7, 0.98, 0.3),
            # 优化器权重衰减范围
            "weight_decay": (0.0, 0.001),
            # 温升 epochs 范围 (可以是小数)
            "warmup_epochs": (0.0, 5.0),
            # 温升初始动量范围
            "warmup_momentum": (0.0, 0.95),
            # box 损失增益范围
            "box": (1.0, 20.0),
            # cls 损失增益范围 (与像素缩放相关)
            "cls": (0.2, 4.0),
            # dfl 损失增益范围
            "dfl": (0.4, 6.0),
            # 图像 HSV-Hue 增强范围 (分数)
            "hsv_h": (0.0, 0.1),
            # 图像 HSV-Saturation 增强范围 (分数)
            "hsv_s": (0.0, 0.9),
            # 图像 HSV-Value 增强范围 (分数)
            "hsv_v": (0.0, 0.9),
            # 图像旋转范围 (+/- 度数)
            "degrees": (0.0, 45.0),
            # 图像平移范围 (+/- 分数)
            "translate": (0.0, 0.9),
            # 图像缩放范围 (+/- 增益)
            "scale": (0.0, 0.95),
            # 图像剪切范围 (+/- 度数)
            "shear": (0.0, 10.0),
            # 图像透视范围 (+/- 分数)，范围 0-0.001
            "perspective": (0.0, 0.001),
            # 图像上下翻转概率
            "flipud": (0.0, 1.0),
            # 图像左右翻转概率
            "fliplr": (0.0, 1.0),
            # 图像通道 bgr 变换概率
            "bgr": (0.0, 1.0),
            # 图像混合概率
            "mosaic": (0.0, 1.0),
            # 图像 mixup 概率
            "mixup": (0.0, 1.0),
            # 分割复制粘贴概率
            "copy_paste": (0.0, 1.0),
        }
        # 使用参数获取配置并初始化
        self.args = get_cfg(overrides=args)
        # 获取保存目录路径
        self.tune_dir = get_save_dir(self.args, name="tune")
        # 定义保存结果的 CSV 文件路径
        self.tune_csv = self.tune_dir / "tune_results.csv"
        # 获取回调函数或者使用默认回调函数列表
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        # 设置前缀字符串
        self.prefix = colorstr("Tuner: ")
        # 添加整合回调函数
        callbacks.add_integration_callbacks(self)
        # 记录初始化信息
        LOGGER.info(
            f"{self.prefix}Initialized Tuner instance with 'tune_dir={self.tune_dir}'\n"
            f"{self.prefix}💡 Learn about tuning at https://docs.ultralytics.com/guides/hyperparameter-tuning"
        )
    # 根据指定的参数变异超参数，基于self.space中指定的边界和缩放因子。
    def _mutate(self, parent="single", n=5, mutation=0.8, sigma=0.2):
        """
        Mutates the hyperparameters based on bounds and scaling factors specified in `self.space`.

        Args:
            parent (str): Parent selection method: 'single' or 'weighted'.
            n (int): Number of parents to consider.
            mutation (float): Probability of a parameter mutation in any given iteration.
            sigma (float): Standard deviation for Gaussian random number generator.

        Returns:
            (dict): A dictionary containing mutated hyperparameters.
        """
        if self.tune_csv.exists():  # if CSV file exists: select best hyps and mutate
            # Select parent(s)
            x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
            fitness = x[:, 0]  # first column
            n = min(n, len(x))  # number of previous results to consider
            x = x[np.argsort(-fitness)][:n]  # top n mutations
            w = x[:, 0] - x[:, 0].min() + 1e-6  # weights (sum > 0)
            if parent == "single" or len(x) == 1:
                # x = x[random.randint(0, n - 1)]  # random selection
                x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
            elif parent == "weighted":
                x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

            # Mutate
            r = np.random  # method
            r.seed(int(time.time()))
            g = np.array([v[2] if len(v) == 3 else 1.0 for k, v in self.space.items()])  # gains 0-1
            ng = len(self.space)
            v = np.ones(ng)
            while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                v = (g * (r.random(ng) < mutation) * r.randn(ng) * r.random() * sigma + 1).clip(0.3, 3.0)
            hyp = {k: float(x[i + 1] * v[i]) for i, k in enumerate(self.space.keys())}
        else:
            # 如果没有调优CSV文件，则使用self.args中的值初始化超参数
            hyp = {k: getattr(self.args, k) for k in self.space.keys()}

        # Constrain to limits
        # 将超参数限制在定义的边界内
        for k, v in self.space.items():
            hyp[k] = max(hyp[k], v[0])  # lower limit
            hyp[k] = min(hyp[k], v[1])  # upper limit
            hyp[k] = round(hyp[k], 5)  # significant digits

        return hyp
```