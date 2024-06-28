# `.\integrations\__init__.py`

```
# 版权声明及许可信息
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入类型检查模块
from typing import TYPE_CHECKING

# 从当前包的utils模块中导入_LazyModule类
from ..utils import _LazyModule

# 定义模块的导入结构
_import_structure = {
    "aqlm": ["replace_with_aqlm_linear"],  # 导入aqlm模块的replace_with_aqlm_linear函数
    "awq": [
        "fuse_awq_modules",  # 导入awq模块的fuse_awq_modules函数
        "post_init_awq_exllama_modules",  # 导入awq模块的post_init_awq_exllama_modules函数
        "replace_with_awq_linear",  # 导入awq模块的replace_with_awq_linear函数
    ],
    "bitsandbytes": [
        "get_keys_to_not_convert",  # 导入bitsandbytes模块的get_keys_to_not_convert函数
        "replace_8bit_linear",  # 导入bitsandbytes模块的replace_8bit_linear函数
        "replace_with_bnb_linear",  # 导入bitsandbytes模块的replace_with_bnb_linear函数
        "set_module_8bit_tensor_to_device",  # 导入bitsandbytes模块的set_module_8bit_tensor_to_device函数
        "set_module_quantized_tensor_to_device",  # 导入bitsandbytes模块的set_module_quantized_tensor_to_device函数
    ],
    "deepspeed": [
        "HfDeepSpeedConfig",  # 导入deepspeed模块的HfDeepSpeedConfig类
        "HfTrainerDeepSpeedConfig",  # 导入deepspeed模块的HfTrainerDeepSpeedConfig类
        "deepspeed_config",  # 导入deepspeed模块的deepspeed_config函数
        "deepspeed_init",  # 导入deepspeed模块的deepspeed_init函数
        "deepspeed_load_checkpoint",  # 导入deepspeed模块的deepspeed_load_checkpoint函数
        "deepspeed_optim_sched",  # 导入deepspeed模块的deepspeed_optim_sched函数
        "is_deepspeed_available",  # 导入deepspeed模块的is_deepspeed_available函数
        "is_deepspeed_zero3_enabled",  # 导入deepspeed模块的is_deepspeed_zero3_enabled函数
        "set_hf_deepspeed_config",  # 导入deepspeed模块的set_hf_deepspeed_config函数
        "unset_hf_deepspeed_config",  # 导入deepspeed模块的unset_hf_deepspeed_config函数
    ],
    "integration_utils": [
        "INTEGRATION_TO_CALLBACK",  # 导入integration_utils模块的INTEGRATION_TO_CALLBACK常量
        "AzureMLCallback",  # 导入integration_utils模块的AzureMLCallback类
        "ClearMLCallback",  # 导入integration_utils模块的ClearMLCallback类
        "CodeCarbonCallback",  # 导入integration_utils模块的CodeCarbonCallback类
        "CometCallback",  # 导入integration_utils模块的CometCallback类
        "DagsHubCallback",  # 导入integration_utils模块的DagsHubCallback类
        "DVCLiveCallback",  # 导入integration_utils模块的DVCLiveCallback类
        "FlyteCallback",  # 导入integration_utils模块的FlyteCallback类
        "MLflowCallback",  # 导入integration_utils模块的MLflowCallback类
        "NeptuneCallback",  # 导入integration_utils模块的NeptuneCallback类
        "NeptuneMissingConfiguration",  # 导入integration_utils模块的NeptuneMissingConfiguration异常类
        "TensorBoardCallback",  # 导入integration_utils模块的TensorBoardCallback类
        "WandbCallback",  # 导入integration_utils模块的WandbCallback类
        "get_available_reporting_integrations",  # 导入integration_utils模块的get_available_reporting_integrations函数
        "get_reporting_integration_callbacks",  # 导入integration_utils模块的get_reporting_integration_callbacks函数
        "hp_params",  # 导入integration_utils模块的hp_params函数
        "is_azureml_available",  # 导入integration_utils模块的is_azureml_available函数
        "is_clearml_available",  # 导入integration_utils模块的is_clearml_available函数
        "is_codecarbon_available",  # 导入integration_utils模块的is_codecarbon_available函数
        "is_comet_available",  # 导入integration_utils模块的is_comet_available函数
        "is_dagshub_available",  # 导入integration_utils模块的is_dagshub_available函数
        "is_dvclive_available",  # 导入integration_utils模块的is_dvclive_available函数
        "is_flyte_deck_standard_available",  # 导入integration_utils模块的is_flyte_deck_standard_available函数
        "is_flytekit_available",  # 导入integration_utils模块的is_flytekit_available函数
        "is_mlflow_available",  # 导入integration_utils模块的is_mlflow_available函数
        "is_neptune_available",  # 导入integration_utils模块的is_neptune_available函数
        "is_optuna_available",  # 导入integration_utils模块的is_optuna_available函数
        "is_ray_available",  # 导入integration_utils模块的is_ray_available函数
        "is_ray_tune_available",  # 导入integration_utils模块的is_ray_tune_available函数
        "is_sigopt_available",  # 导入integration_utils模块的is_sigopt_available函数
        "is_tensorboard_available",  # 导入integration_utils模块的is_tensorboard_available函数
        "is_wandb_available",  # 导入integration_utils模块的is_wandb_available函数
        "rewrite_logs",  # 导入integration_utils模块的rewrite_logs函数
        "run_hp_search_optuna",  # 导入integration_utils模块的run_hp_search_optuna函数
        "run_hp_search_ray",  # 导入integration_utils模块的run_hp_search_ray函数
        "run_hp_search_sigopt",  # 导入integration_utils模块的run_hp_search_sigopt函数
        "run_hp_search_wandb",  # 导入integration_utils模块的run_hp_search_wandb函数
    ],
    "peft": ["PeftAdapterMixin"],  # 导入peft模块的PeftAdapterMixin类
    "quanto": ["replace_with_quanto_layers"],  # 导入quanto模块的replace_with_quanto_layers函数
}

# 如果支持类型检查，则导入以下类型
if TYPE_CHECKING:
    from .aqlm import replace_with_aqlm_linear  # 导入aqlm模块的replace_with_aqlm_linear函数
    from .awq import (
        fuse_awq_modules,  # 导入awq模块的fuse_awq_modules函数
        post_init_awq_exllama_modules,  # 导入awq模块的post_init_awq_exllama_modules函数
        replace_with_awq_linear,  # 导入awq模块的replace_with_awq_linear函数
    )
    # 导入从bitsandbytes模块中的函数和类
    from .bitsandbytes import (
        get_keys_to_not_convert,               # 获取不转换的键
        replace_8bit_linear,                  # 替换为8位线性
        replace_with_bnb_linear,              # 替换为BNB线性
        set_module_8bit_tensor_to_device,     # 将模块的8位张量设置到设备
        set_module_quantized_tensor_to_device # 将模块的量化张量设置到设备
    )
    
    # 导入从deepspeed模块中的函数和类
    from .deepspeed import (
        HfDeepSpeedConfig,                    # Hugging Face DeepSpeed 配置
        HfTrainerDeepSpeedConfig,             # Hugging Face Trainer DeepSpeed 配置
        deepspeed_config,                     # DeepSpeed 配置
        deepspeed_init,                       # DeepSpeed 初始化
        deepspeed_load_checkpoint,            # 加载 DeepSpeed 检查点
        deepspeed_optim_sched,                # DeepSpeed 优化和调度
        is_deepspeed_available,               # 判断 DeepSpeed 是否可用
        is_deepspeed_zero3_enabled,           # 判断 DeepSpeed Zero3 是否启用
        set_hf_deepspeed_config,              # 设置 Hugging Face DeepSpeed 配置
        unset_hf_deepspeed_config             # 取消设置 Hugging Face DeepSpeed 配置
    )
    
    # 导入从integration_utils模块中的函数和类
    from .integration_utils import (
        INTEGRATION_TO_CALLBACK,              # 集成到回调函数的映射
        AzureMLCallback,                      # AzureML 回调函数
        ClearMLCallback,                      # ClearML 回调函数
        CodeCarbonCallback,                   # CodeCarbon 回调函数
        CometCallback,                        # Comet 回调函数
        DagsHubCallback,                      # DagsHub 回调函数
        DVCLiveCallback,                      # DVCLive 回调函数
        FlyteCallback,                        # Flyte 回调函数
        MLflowCallback,                       # MLflow 回调函数
        NeptuneCallback,                      # Neptune 回调函数
        NeptuneMissingConfiguration,          # Neptune 缺少配置
        TensorBoardCallback,                  # TensorBoard 回调函数
        WandbCallback,                        # Wandb 回调函数
        get_available_reporting_integrations, # 获取可用的报告集成
        get_reporting_integration_callbacks,  # 获取报告集成回调函数
        hp_params,                            # 超参数配置
        is_azureml_available,                 # 判断 AzureML 是否可用
        is_clearml_available,                 # 判断 ClearML 是否可用
        is_codecarbon_available,              # 判断 CodeCarbon 是否可用
        is_comet_available,                   # 判断 Comet 是否可用
        is_dagshub_available,                 # 判断 DagsHub 是否可用
        is_dvclive_available,                 # 判断 DVCLive 是否可用
        is_flyte_deck_standard_available,     # 判断 Flyte Deck Standard 是否可用
        is_flytekit_available,                # 判断 Flytekit 是否可用
        is_mlflow_available,                  # 判断 MLflow 是否可用
        is_neptune_available,                 # 判断 Neptune 是否可用
        is_optuna_available,                  # 判断 Optuna 是否可用
        is_ray_available,                     # 判断 Ray 是否可用
        is_ray_tune_available,                # 判断 Ray Tune 是否可用
        is_sigopt_available,                  # 判断 SigOpt 是否可用
        is_tensorboard_available,             # 判断 TensorBoard 是否可用
        is_wandb_available,                   # 判断 Wandb 是否可用
        rewrite_logs,                         # 重写日志
        run_hp_search_optuna,                 # 运行 Optuna 的超参数搜索
        run_hp_search_ray,                    # 运行 Ray 的超参数搜索
        run_hp_search_sigopt,                 # 运行 SigOpt 的超参数搜索
        run_hp_search_wandb                   # 运行 Wandb 的超参数搜索
    )
    
    # 导入从peft模块中的类
    from .peft import PeftAdapterMixin         # Peft 适配器混合类
    
    # 导入从quanto模块中的函数
    from .quanto import replace_with_quanto_layers  # 替换为 Quanto 层
else:
    # 导入 sys 模块，用于动态管理模块
    import sys

    # 将当前模块注册为一个懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```