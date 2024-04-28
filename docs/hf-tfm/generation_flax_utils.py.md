# `.\transformers\generation_flax_utils.py`

```
# 设置文件编码为 UTF-8
# 版权声明，包括作者信息和版权声明
# 根据 Apache 许可证 2.0 版本，对该文件的使用受限制
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证副本
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以了解特定语言的权限和限制

# 导入警告模块
import warnings

# 从 generation 模块导入 FlaxGenerationMixin 类

# 定义 FlaxGenerationMixin 类，继承自 FlaxGenerationMixin 类
class FlaxGenerationMixin(FlaxGenerationMixin):
    # 在导入时发出警告
    warnings.warn(
        "Importing `FlaxGenerationMixin` from `src/transformers/generation_flax_utils.py` is deprecated and will "
        "be removed in Transformers v5. Import as `from transformers import FlaxGenerationMixin` instead.",
        FutureWarning,
    )
```