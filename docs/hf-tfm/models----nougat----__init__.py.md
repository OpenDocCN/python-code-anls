# `.\models\nougat\__init__.py`

```
# 版权声明及许可信息，指出此代码的版权归HuggingFace团队所有，并按Apache License, Version 2.0许可使用
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

# 导入类型检查模块的类型检查工具
from typing import TYPE_CHECKING

# 导入必要的异常和模块，用于处理可选依赖项未安装的情况，以及懒加载模块和可用性检查工具
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_vision_available

# 定义要导入的结构，初始化处理模块的列表
_import_structure = {
    "processing_nougat": ["NougatProcessor"],
}

# 检查是否安装了tokenizers，若未安装则抛出OptionalDependencyNotAvailable异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若安装了tokenizers，则添加tokenization_nougat_fast模块到导入结构中
    _import_structure["tokenization_nougat_fast"] = ["NougatTokenizerFast"]

# 检查是否安装了vision模块，若未安装则抛出OptionalDependencyNotAvailable异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若安装了vision模块，则添加image_processing_nougat模块到导入结构中
    _import_structure["image_processing_nougat"] = ["NougatImageProcessor"]

# 如果当前是类型检查模式
if TYPE_CHECKING:
    # 导入processing_nougat模块中的NougatProcessor类
    from .processing_nougat import NougatProcessor

    # 检查是否安装了tokenizers，若安装则导入tokenization_nougat_fast模块中的NougatTokenizerFast类
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_nougat_fast import NougatTokenizerFast

    # 检查是否安装了vision模块，若安装则导入image_processing_nougat模块中的NougatImageProcessor类
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_nougat import NougatImageProcessor

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块注册为一个懒加载模块，使用_LazyModule进行延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```