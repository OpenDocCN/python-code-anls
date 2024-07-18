# `.\graphrag\graphrag\config\input_models\parallelization_parameters_input.py`

```py
# 版权声明和许可证信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入所需的模块和库
"""LLM Parameters model."""
# 定义一个类型字典类，用于表示并行化参数输入
from typing_extensions import NotRequired, TypedDict

# 并行化参数输入的类型字典类
class ParallelizationParametersInput(TypedDict):
    """LLM Parameters model."""

    # stagger 参数，可以是 float、str 或者 None，非必需
    stagger: NotRequired[float | str | None]
    # num_threads 参数，可以是 int、str 或者 None，非必需
    num_threads: NotRequired[int | str | None]
```