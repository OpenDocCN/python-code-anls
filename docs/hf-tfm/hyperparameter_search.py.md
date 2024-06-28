# `.\hyperparameter_search.py`

```
# 从 integrations 模块中导入必要的函数和变量
from .integrations import (
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)
# 从 trainer_utils 模块中导入必要的类和函数
from .trainer_utils import (
    HPSearchBackend,
    default_hp_space_optuna,
    default_hp_space_ray,
    default_hp_space_sigopt,
    default_hp_space_wandb,
)
# 从 utils 模块中导入 logging 函数
from .utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义超参数搜索后端基类
class HyperParamSearchBackendBase:
    name: str
    pip_package: str = None

    @staticmethod
    def is_available():
        # 抽象方法，子类需要实现该方法来检查后端是否可用
        raise NotImplementedError

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        # 抽象方法，子类需要实现该方法来执行超参数搜索
        raise NotImplementedError

    def default_hp_space(self, trial):
        # 抽象方法，子类需要实现该方法来定义默认的超参数空间
        raise NotImplementedError

    def ensure_available(self):
        # 确保后端可用，否则抛出运行时异常
        if not self.is_available():
            raise RuntimeError(
                f"You picked the {self.name} backend, but it is not installed. Run {self.pip_install()}."
            )

    @classmethod
    def pip_install(cls):
        # 返回安装当前后端所需的 pip 命令字符串
        return f"`pip install {cls.pip_package or cls.name}`"


# 定义 Optuna 后端类，继承自 HyperParamSearchBackendBase
class OptunaBackend(HyperParamSearchBackendBase):
    name = "optuna"

    @staticmethod
    def is_available():
        # 检查 Optuna 是否可用
        return is_optuna_available()

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        # 使用 Optuna 执行超参数搜索
        return run_hp_search_optuna(trainer, n_trials, direction, **kwargs)

    def default_hp_space(self, trial):
        # 返回 Optuna 的默认超参数空间
        return default_hp_space_optuna(trial)


# 定义 Ray Tune 后端类，继承自 HyperParamSearchBackendBase
class RayTuneBackend(HyperParamSearchBackendBase):
    name = "ray"
    pip_package = "'ray[tune]'"

    @staticmethod
    def is_available():
        # 检查 Ray Tune 是否可用
        return is_ray_tune_available()

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        # 使用 Ray Tune 执行超参数搜索
        return run_hp_search_ray(trainer, n_trials, direction, **kwargs)

    def default_hp_space(self, trial):
        # 返回 Ray Tune 的默认超参数空间
        return default_hp_space_ray(trial)


# 定义 SigOpt 后端类，继承自 HyperParamSearchBackendBase
class SigOptBackend(HyperParamSearchBackendBase):
    name = "sigopt"

    @staticmethod
    def is_available():
        # 检查 SigOpt 是否可用
        return is_sigopt_available()

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        # 使用 SigOpt 执行超参数搜索
        return run_hp_search_sigopt(trainer, n_trials, direction, **kwargs)

    def default_hp_space(self, trial):
        # 返回 SigOpt 的默认超参数空间
        return default_hp_space_sigopt(trial)


# 定义 Wandb 后端类，继承自 HyperParamSearchBackendBase
class WandbBackend(HyperParamSearchBackendBase):
    name = "wandb"

    @staticmethod
    def is_available():
        # 检查 Wandb 是否可用
        return is_wandb_available()

    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        # 使用 Wandb 执行超参数搜索
        return run_hp_search_wandb(trainer, n_trials, direction, **kwargs)

    def default_hp_space(self, trial):
        # 返回 Wandb 的默认超参数空间
        return default_hp_space_wandb(trial)
    # 定义静态方法，用于检查是否安装了 Weights & Biases 库
    @staticmethod
    def is_available():
        # 调用 is_wandb_available 函数，检查 Weights & Biases 库是否可用
        return is_wandb_available()

    # 定义方法，用于运行超参数搜索
    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        # 调用 run_hp_search_wandb 函数，运行基于 Weights & Biases 的超参数搜索
        return run_hp_search_wandb(trainer, n_trials, direction, **kwargs)

    # 定义方法，返回默认的超参数空间
    def default_hp_space(self, trial):
        # 调用 default_hp_space_wandb 函数，返回基于 Weights & Biases 的默认超参数空间
        return default_hp_space_wandb(trial)
# 创建一个字典，将各个超参数搜索后端与其名称关联起来
ALL_HYPERPARAMETER_SEARCH_BACKENDS = {
    HPSearchBackend(backend.name): backend for backend in [OptunaBackend, RayTuneBackend, SigOptBackend, WandbBackend]
}

# 定义一个函数，用于获取默认的超参数搜索后端名称
def default_hp_search_backend() -> str:
    # 获取所有可用的超参数搜索后端
    available_backends = [backend for backend in ALL_HYPERPARAMETER_SEARCH_BACKENDS.values() if backend.is_available()]
    
    # 如果至少有一个可用的后端，则选择第一个作为默认值
    if len(available_backends) > 0:
        name = available_backends[0].name
        
        # 如果有多个可用的后端，记录日志并使用第一个作为默认
        if len(available_backends) > 1:
            logger.info(
                f"{len(available_backends)} hyperparameter search backends available. Using {name} as the default."
            )
        
        # 返回选定的后端名称
        return name
    
    # 如果没有可用的后端，则抛出运行时错误，并给出安装信息
    raise RuntimeError(
        "No hyperparameter search backend available.\n"
        + "\n".join(
            f" - To install {backend.name} run {backend.pip_install()}"
            for backend in ALL_HYPERPARAMETER_SEARCH_BACKENDS.values()
        )
    )
```