# `.\transformers\hyperparameter_search.py`

```py
# 导入相关的集成模块和函数
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
# 导入与超参数搜索相关的实用函数和类
from .trainer_utils import (
    HPSearchBackend,
    default_hp_space_optuna,
    default_hp_space_ray,
    default_hp_space_sigopt,
    default_hp_space_wandb,
)
# 导入日志记录工具
from .utils import logging

# 获取与当前模块相关的日志记录器
logger = logging.get_logger(__name__)

# 定义超参数搜索后端基类
class HyperParamSearchBackendBase:
    name: str
    pip_package: str = None

    # 检查当前后端是否可用的静态方法
    @staticmethod
    def is_available():
        raise NotImplementedError

    # 运行超参数搜索的方法
    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        raise NotImplementedError

    # 获取默认超参数空间的方法
    def default_hp_space(self, trial):
        raise NotImplementedError

    # 确保当前后端可用的方法
    def ensure_available(self):
        if not self.is_available():
            # 如果当前后端不可用，则抛出运行时错误
            raise RuntimeError(
                f"You picked the {self.name} backend, but it is not installed. Run {self.pip_install()}."
            )

    # 获取安装当前后端所需的 pip 命令的类方法
    @classmethod
    def pip_install(cls):
        return f"`pip install {cls.pip_package or cls.name}`"

# 定义 Optuna 超参数搜索后端类
class OptunaBackend(HyperParamSearchBackendBase):
    name = "optuna"

    # 检查 Optuna 是否可用的静态方法
    @staticmethod
    def is_available():
        return is_optuna_available()

    # 运行基于 Optuna 的超参数搜索的方法
    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        return run_hp_search_optuna(trainer, n_trials, direction, **kwargs)

    # 获取默认的超参数空间定义的方法
    def default_hp_space(self, trial):
        return default_hp_space_optuna(trial)

# 定义 Ray Tune 超参数搜索后端类
class RayTuneBackend(HyperParamSearchBackendBase):
    name = "ray"
    pip_package = "'ray[tune]'"

    # 检查 Ray Tune 是否可用的静态方法
    @staticmethod
    def is_available():
        return is_ray_tune_available()

    # 运行基于 Ray Tune 的超参数搜索的方法
    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        return run_hp_search_ray(trainer, n_trials, direction, **kwargs)

    # 获取默认的超参数空间定义的方法
    def default_hp_space(self, trial):
        return default_hp_space_ray(trial)

# 定义 SigOpt 超参数搜索后端类
class SigOptBackend(HyperParamSearchBackendBase):
    name = "sigopt"

    # 检查 SigOpt 是否可用的静态方法
    @staticmethod
    def is_available():
        return is_sigopt_available()

    # 运行基于 SigOpt 的超参数搜索的方法
    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        return run_hp_search_sigopt(trainer, n_trials, direction, **kwargs)

    # 获取默认的超参数空间定义的方法
    def default_hp_space(self, trial):
        return default_hp_space_sigopt(trial)

# 定义 Wandb 超参数搜索后端类
class WandbBackend(HyperParamSearchBackendBase):
    name = "wandb"
    # 静态方法：检查是否可用
    @staticmethod
    def is_available():
        # 调用外部函数检查是否可用
        return is_wandb_available()

    # 运行超参数搜索
    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        # 调用外部函数执行超参数搜索，并返回结果
        return run_hp_search_wandb(trainer, n_trials, direction, **kwargs)

    # 获取默认的超参数空间
    def default_hp_space(self, trial):
        # 调用外部函数获取默认的超参数空间
        return default_hp_space_wandb(trial)
# 定义一个字典，用于存储所有超参数搜索后端，键为HPSearchBackend对象，值为相应的后端类
ALL_HYPERPARAMETER_SEARCH_BACKENDS = {
    HPSearchBackend(backend.name): backend for backend in [OptunaBackend, RayTuneBackend, SigOptBackend, WandbBackend]
}

# 定义一个函数，用于获取默认的超参数搜索后端名称
def default_hp_search_backend() -> str:
    # 获取所有可用的超参数搜索后端列表
    available_backends = [backend for backend in ALL_HYPERPARAMETER_SEARCH_BACKENDS.values() if backend.is_available()]
    # 如果存在可用的后端
    if len(available_backends) > 0:
        # 获取第一个可用后端的名称
        name = available_backends[0].name
        # 如果存在多个可用后端
        if len(available_backends) > 1:
            # 记录日志，指出正在使用的默认后端名称
            logger.info(
                f"{len(available_backends)} hyperparameter search backends available. Using {name} as the default."
            )
        # 返回默认后端名称
        return name
    # 如果没有可用的后端，则引发运行时错误
    raise RuntimeError(
        "No hyperparameter search backend available.\n"
        + "\n".join(
            f" - To install {backend.name} run {backend.pip_install()}"
            for backend in ALL_HYPERPARAMETER_SEARCH_BACKENDS.values()
        )
    )
```