# `.\pytorch\torch\quantization\observer.py`

```py
# flake8: noqa: F401
"""
这个文件正在迁移到 `torch/ao/quantization`，在迁移过程中保留在这里以保持兼容性。
如果你正在添加新的条目/功能，请将其添加到 `torch/ao/quantization/observer.py`，同时在这里添加一个导入语句。
"""
# 从 `torch.ao.quantization.observer` 模块导入多个特定的符号（函数、类等）
from torch.ao.quantization.observer import (
    _is_activation_post_process,
    _is_per_channel_script_obs_instance,
    _ObserverBase,
    _PartialWrapper,
    _with_args,
    _with_callable_args,
    ABC,
    default_debug_observer,
    default_dynamic_quant_observer,
    default_float_qparams_observer,
    default_histogram_observer,
    default_observer,
    default_per_channel_weight_observer,
    default_placeholder_observer,
    default_weight_observer,
    get_observer_state_dict,
    HistogramObserver,
    load_observer_state_dict,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    NoopObserver,
    ObserverBase,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
    RecordingObserver,
)
```