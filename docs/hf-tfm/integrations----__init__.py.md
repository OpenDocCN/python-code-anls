# `.\transformers\integrations\__init__.py`

```
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入懒加载模块
from ..utils import _LazyModule

# 定义模块导入结构
_import_structure = {
    "awq": ["fuse_awq_modules", "replace_with_awq_linear"],
    "bitsandbytes": [
        "get_keys_to_not_convert",
        "replace_8bit_linear",
        "replace_with_bnb_linear",
        "set_module_8bit_tensor_to_device",
        "set_module_quantized_tensor_to_device",
    ],
    "deepspeed": [
        "HfDeepSpeedConfig",
        "HfTrainerDeepSpeedConfig",
        "deepspeed_config",
        "deepspeed_init",
        "deepspeed_load_checkpoint",
        "deepspeed_optim_sched",
        "is_deepspeed_available",
        "is_deepspeed_zero3_enabled",
        "set_hf_deepspeed_config",
        "unset_hf_deepspeed_config",
    ],
    "integration_utils": [
        "INTEGRATION_TO_CALLBACK",
        "AzureMLCallback",
        "ClearMLCallback",
        "CodeCarbonCallback",
        "CometCallback",
        "DagsHubCallback",
        "DVCLiveCallback",
        "FlyteCallback",
        "MLflowCallback",
        "NeptuneCallback",
        "NeptuneMissingConfiguration",
        "TensorBoardCallback",
        "WandbCallback",
        "get_available_reporting_integrations",
        "get_reporting_integration_callbacks",
        "hp_params",
        "is_azureml_available",
        "is_clearml_available",
        "is_codecarbon_available",
        "is_comet_available",
        "is_dagshub_available",
        "is_dvclive_available",
        "is_flyte_deck_standard_available",
        "is_flytekit_available",
        "is_mlflow_available",
        "is_neptune_available",
        "is_optuna_available",
        "is_ray_available",
        "is_ray_tune_available",
        "is_sigopt_available",
        "is_tensorboard_available",
        "is_wandb_available",
        "rewrite_logs",
        "run_hp_search_optuna",
        "run_hp_search_ray",
        "run_hp_search_sigopt",
        "run_hp_search_wandb",
    ],
    "peft": ["PeftAdapterMixin"],
}

# 如果是类型检查模式，则导入特定模块
if TYPE_CHECKING:
    from .awq import fuse_awq_modules, replace_with_awq_linear
    from .bitsandbytes import (
        get_keys_to_not_convert,
        replace_8bit_linear,
        replace_with_bnb_linear,
        set_module_8bit_tensor_to_device,
        set_module_quantized_tensor_to_device,
    )
    # 从当前包中导入相关模块和函数
    
    from .deepspeed import (
        # 导入深度加速相关配置类和函数
        HfDeepSpeedConfig,
        HfTrainerDeepSpeedConfig,
        deepspeed_config,
        deepspeed_init,
        deepspeed_load_checkpoint,
        deepspeed_optim_sched,
        is_deepspeed_available,
        is_deepspeed_zero3_enabled,
        set_hf_deepspeed_config,
        unset_hf_deepspeed_config,
    )
    
    # 从当前包中导入集成工具相关模块和函数
    from .integration_utils import (
        # 导入集成工具到回调函数的映射关系
        INTEGRATION_TO_CALLBACK,
        # 导入各种集成工具的回调函数
        AzureMLCallback,
        ClearMLCallback,
        CodeCarbonCallback,
        CometCallback,
        DagsHubCallback,
        DVCLiveCallback,
        FlyteCallback,
        MLflowCallback,
        NeptuneCallback,
        NeptuneMissingConfiguration,
        TensorBoardCallback,
        WandbCallback,
        # 导入获取可用报告集成工具和回调函数的函数
        get_available_reporting_integrations,
        get_reporting_integration_callbacks,
        # 导入超参数相关函数
        hp_params,
        # 导入判断各种集成工具是否可用的函数
        is_azureml_available,
        is_clearml_available,
        is_codecarbon_available,
        is_comet_available,
        is_dagshub_available,
        is_dvclive_available,
        is_flyte_deck_standard_available,
        is_flytekit_available,
        is_mlflow_available,
        is_neptune_available,
        is_optuna_available,
        is_ray_available,
        is_ray_tune_available,
        is_sigopt_available,
        is_tensorboard_available,
        is_wandb_available,
        # 导入日志重写函数
        rewrite_logs,
        # 导入基于 Optuna、Ray、SigOpt 和 Wandb 的超参数搜索函数
        run_hp_search_optuna,
        run_hp_search_ray,
        run_hp_search_sigopt,
        run_hp_search_wandb,
    )
    
    # 从当前包中导入 PEFT 适配器混合类
    from .peft import PeftAdapterMixin
# 如果不在主模块中，则导入sys模块
import sys
# 将当前模块添加到sys.modules字典中，使用_LazyModule延迟加载模块
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```