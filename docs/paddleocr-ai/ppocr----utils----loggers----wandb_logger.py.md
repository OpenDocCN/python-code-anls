# `.\PaddleOCR\ppocr\utils\loggers\wandb_logger.py`

```
# 导入 os 模块
# 从当前目录下的 base_logger 模块中导入 BaseLogger 类
import os
from .base_logger import BaseLogger

# 定义 WandbLogger 类，继承自 BaseLogger 类
class WandbLogger(BaseLogger):
    # 初始化方法，接受多个参数
    def __init__(self, 
        project=None, 
        name=None, 
        id=None, 
        entity=None, 
        save_dir=None, 
        config=None,
        **kwargs):
        # 尝试导入 wandb 模块
        try:
            import wandb
            self.wandb = wandb
        # 如果导入失败，抛出 ModuleNotFoundError 异常
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install wandb using `pip install wandb`"
                )

        # 初始化实例变量
        self.project = project
        self.name = name
        self.id = id
        self.save_dir = save_dir
        self.config = config
        self.kwargs = kwargs
        self.entity = entity
        self._run = None
        # 初始化 wandb 的配置参数
        self._wandb_init = dict(
            project=self.project,
            name=self.name,
            id=self.id,
            entity=self.entity,
            dir=self.save_dir,
            resume="allow"
        )
        self._wandb_init.update(**kwargs)

        # 调用 run 属性

        # 如果有配置参数，则更新 run 的配置
        if self.config:
            self.run.config.update(self.config)

    # run 属性，用于获取或初始化 wandb 的 run 对象
    @property
    def run(self):
        if self._run is None:
            # 如果已经有 wandb 的 run 对象存在，则提示信息
            if self.wandb.run is not None:
                logger.info(
                    "There is a wandb run already in progress "
                    "and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()`"
                    "before instantiating `WandbLogger`."
                )
                self._run = self.wandb.run
            # 否则初始化一个新的 run 对象
            else:
                self._run = self.wandb.init(**self._wandb_init)
        return self._run

    # 记录指标数据的方法
    def log_metrics(self, metrics, prefix=None, step=None):
        # 如果没有指定前缀，则设为空字符串
        if not prefix:
            prefix = ""
        # 更新指标数据的键名，加上前缀
        updated_metrics = {prefix.lower() + "/" + k: v for k, v in metrics.items()}
        
        # 调用 run 对象的 log 方法，记录更新后的指标数据和步数
        self.run.log(updated_metrics, step=step)
    # 记录模型信息到日志，包括是否为最佳模型、前缀和元数据
    def log_model(self, is_best, prefix, metadata=None):
        # 拼接模型保存路径
        model_path = os.path.join(self.save_dir, prefix + '.pdparams')
        # 创建一个 Wandb Artifact 对象，用于存储模型
        artifact = self.wandb.Artifact('model-{}'.format(self.run.id), type='model', metadata=metadata)
        # 将模型文件添加到 Artifact 中
        artifact.add_file(model_path, name="model_ckpt.pdparams")

        # 设置模型的别名，如果是最佳模型则添加 "best" 别名
        aliases = [prefix]
        if is_best:
            aliases.append("best")

        # 将 Artifact 记录到当前运行日志中，设置别名
        self.run.log_artifact(artifact, aliases=aliases)

    # 结束当前运行
    def close(self):
        self.run.finish()
```