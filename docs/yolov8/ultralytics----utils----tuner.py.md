# `.\yolov8\ultralytics\utils\tuner.py`

```py
# 导入 subprocess 模块，用于执行外部命令
import subprocess

# 从 ultralytics 的 cfg 模块中导入 TASK2DATA、TASK2METRIC、get_save_dir 函数
from ultralytics.cfg import TASK2DATA, TASK2METRIC, get_save_dir
# 从 ultralytics 的 utils 模块中导入 DEFAULT_CFG、DEFAULT_CFG_DICT、LOGGER、NUM_THREADS、checks 函数
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER, NUM_THREADS, checks


# 定义函数 run_ray_tune，用于运行 Ray Tune 进行超参数调优
def run_ray_tune(
    model, space: dict = None, grace_period: int = 10, gpu_per_trial: int = None, max_samples: int = 10, **train_args
):
    """
    Runs hyperparameter tuning using Ray Tune.

    Args:
        model (YOLO): Model to run the tuner on.
        space (dict, optional): The hyperparameter search space. Defaults to None.
        grace_period (int, optional): The grace period in epochs of the ASHA scheduler. Defaults to 10.
        gpu_per_trial (int, optional): The number of GPUs to allocate per trial. Defaults to None.
        max_samples (int, optional): The maximum number of trials to run. Defaults to 10.
        train_args (dict, optional): Additional arguments to pass to the `train()` method. Defaults to {}.

    Returns:
        (dict): A dictionary containing the results of the hyperparameter search.

    Example:
        ```python
        from ultralytics import YOLO

        # Load a YOLOv8n model
        model = YOLO('yolov8n.pt')

        # Start tuning hyperparameters for YOLOv8n training on the COCO8 dataset
        result_grid = model.tune(data='coco8.yaml', use_ray=True)
        ```
    """

    # 在日志中输出一条信息，引导用户了解 RayTune 的文档链接
    LOGGER.info("💡 Learn about RayTune at https://docs.ultralytics.com/integrations/ray-tune")
    
    # 如果 train_args 为 None，则将其设为一个空字典
    if train_args is None:
        train_args = {}

    try:
        # 尝试安装 Ray Tune 相关依赖
        subprocess.run("pip install ray[tune]".split(), check=True)  # do not add single quotes here

        # 导入必要的 Ray Tune 模块
        import ray
        from ray import tune
        from ray.air import RunConfig
        from ray.air.integrations.wandb import WandbLoggerCallback
        from ray.tune.schedulers import ASHAScheduler
    except ImportError:
        # 若导入失败，则抛出 ModuleNotFoundError 异常
        raise ModuleNotFoundError('Ray Tune required but not found. To install run: pip install "ray[tune]"')

    try:
        # 尝试导入 wandb 模块，并确保其有 __version__ 属性
        import wandb
        assert hasattr(wandb, "__version__")
    except (ImportError, AssertionError):
        # 若导入失败或缺少 __version__ 属性，则将 wandb 设为 False
        wandb = False

    # 使用 checks 模块检查 ray 的版本是否符合要求
    checks.check_version(ray.__version__, ">=2.0.0", "ray")
    # 默认的超参数搜索空间，包含各种超参数的取值范围或概率分布
    default_space = {
        # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
        "lr0": tune.uniform(1e-5, 1e-1),  # 初始学习率范围
        "lrf": tune.uniform(0.01, 1.0),  # 最终OneCycleLR学习率 (lr0 * lrf)
        "momentum": tune.uniform(0.6, 0.98),  # SGD动量/Adam beta1
        "weight_decay": tune.uniform(0.0, 0.001),  # 优化器权重衰减
        "warmup_epochs": tune.uniform(0.0, 5.0),  # 预热 epochs 数量（允许小数）
        "warmup_momentum": tune.uniform(0.0, 0.95),  # 预热初始动量
        "box": tune.uniform(0.02, 0.2),  # 盒损失增益
        "cls": tune.uniform(0.2, 4.0),  # 分类损失增益（与像素缩放比例相关）
        "hsv_h": tune.uniform(0.0, 0.1),  # 图像HSV-Hue增强（比例）
        "hsv_s": tune.uniform(0.0, 0.9),  # 图像HSV-Saturation增强（比例）
        "hsv_v": tune.uniform(0.0, 0.9),  # 图像HSV-Value增强（比例）
        "degrees": tune.uniform(0.0, 45.0),  # 图像旋转角度范围（+/- 度）
        "translate": tune.uniform(0.0, 0.9),  # 图像平移范围（+/- 比例）
        "scale": tune.uniform(0.0, 0.9),  # 图像缩放范围（+/- 增益）
        "shear": tune.uniform(0.0, 10.0),  # 图像剪切范围（+/- 度）
        "perspective": tune.uniform(0.0, 0.001),  # 图像透视变换范围（+/- 比例），范围 0-0.001
        "flipud": tune.uniform(0.0, 1.0),  # 图像上下翻转概率
        "fliplr": tune.uniform(0.0, 1.0),  # 图像左右翻转概率
        "bgr": tune.uniform(0.0, 1.0),  # 图像通道BGR转换概率
        "mosaic": tune.uniform(0.0, 1.0),  # 图像混合（mosaic）概率
        "mixup": tune.uniform(0.0, 1.0),  # 图像混合（mixup）概率
        "copy_paste": tune.uniform(0.0, 1.0),  # 分割复制粘贴概率
    }

    # 将模型放入Ray存储
    task = model.task
    model_in_store = ray.put(model)

    def _tune(config):
        """
        使用指定的超参数和额外的参数训练YOLO模型。

        Args:
            config (dict): 用于训练的超参数字典。

        Returns:
            dict: 训练结果字典。
        """
        model_to_train = ray.get(model_in_store)  # 从Ray存储中获取要调整的模型
        model_to_train.reset_callbacks()  # 重置回调函数
        config.update(train_args)  # 更新训练参数
        results = model_to_train.train(**config)  # 使用给定的超参数进行训练
        return results.results_dict  # 返回训练结果字典

    # 获取搜索空间
    if not space:
        space = default_space  # 如果未提供搜索空间，则使用默认搜索空间
        LOGGER.warning("WARNING ⚠️ 没有提供搜索空间，使用默认搜索空间。")

    # 获取数据集
    data = train_args.get("data", TASK2DATA[task])  # 获取数据集
    space["data"] = data  # 将数据集添加到搜索空间
    if "data" not in train_args:
        LOGGER.warning(f'WARNING ⚠️ 没有提供数据集，使用默认数据集 "data={data}".')

    # 定义带有分配资源的可训练函数
    trainable_with_resources = tune.with_resources(_tune, {"cpu": NUM_THREADS, "gpu": gpu_per_trial or 0})
    # 定义 ASHA 调度器用于超参数搜索
    asha_scheduler = ASHAScheduler(
        time_attr="epoch",  # 指定时间属性为 epoch
        metric=TASK2METRIC[task],  # 设置优化的指标为任务对应的指标
        mode="max",  # 设定优化模式为最大化
        max_t=train_args.get("epochs") or DEFAULT_CFG_DICT["epochs"] or 100,  # 设置最大的训练轮数
        grace_period=grace_period,  # 设置优雅期间
        reduction_factor=3,  # 设置收敛因子
    )

    # 定义用于超参数搜索的回调函数
    tuner_callbacks = [WandbLoggerCallback(project="YOLOv8-tune")] if wandb else []

    # 创建 Ray Tune 的超参数搜索调谐器
    tune_dir = get_save_dir(DEFAULT_CFG, name="tune").resolve()  # 获取保存调优结果的绝对路径，确保目录存在
    tune_dir.mkdir(parents=True, exist_ok=True)  # 创建保存调优结果的目录，如果不存在则创建
    tuner = tune.Tuner(
        trainable_with_resources,  # 可训练函数
        param_space=space,  # 参数空间
        tune_config=tune.TuneConfig(scheduler=asha_scheduler, num_samples=max_samples),  # 调优配置
        run_config=RunConfig(callbacks=tuner_callbacks, storage_path=tune_dir),  # 运行配置
    )

    # 运行超参数搜索
    tuner.fit()

    # 返回超参数搜索的结果
    return tuner.get_results()
```