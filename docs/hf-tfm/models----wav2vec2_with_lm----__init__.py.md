# `.\models\wav2vec2_with_lm\__init__.py`

```py
# 版权声明和版权许可信息
# Copyright 2021 The HuggingFace Team. All rights reserved.
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

# 引入类型检查模块的导入声明
from typing import TYPE_CHECKING

# 引入延迟加载模块工具
from ...utils import _LazyModule

# 定义模块的导入结构
_import_structure = {"processing_wav2vec2_with_lm": ["Wav2Vec2ProcessorWithLM"]}

# 如果是类型检查模式
if TYPE_CHECKING:
    # 从具体的子模块中导入 Wav2Vec2ProcessorWithLM 类型
    from .processing_wav2vec2_with_lm import Wav2Vec2ProcessorWithLM
# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys

    # 将当前模块替换为一个延迟加载模块，使用 LazyModule 类型
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```