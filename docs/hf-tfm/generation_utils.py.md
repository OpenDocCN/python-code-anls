# `.\transformers\generation_utils.py`

```
# 设置文件编码为 UTF-8
# 版权声明，包括作者信息和许可证信息
# 依据 Apache 许可证 2.0 版本，对该文件的使用受限
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以了解特定语言的权限和限制

# 导入警告模块
import warnings

# 从 generation 模块导入 GenerationMixin 类

# 定义 GenerationMixin 类，继承自 GenerationMixin 类
class GenerationMixin(GenerationMixin):
    # 在导入时发出警告
    warnings.warn(
        "Importing `GenerationMixin` from `src/transformers/generation_utils.py` is deprecated and will "
        "be removed in Transformers v5. Import as `from transformers import GenerationMixin` instead.",
        FutureWarning,
    )
```