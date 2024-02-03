# `.\PaddleOCR\ppocr\data\imaug\text_image_aug\__init__.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 从 augment 模块中导入 tia_perspective、tia_distort、tia_stretch 函数
from .augment import tia_perspective, tia_distort, tia_stretch

# 导出 tia_distort、tia_stretch、tia_perspective 函数
__all__ = ['tia_distort', 'tia_stretch', 'tia_perspective']
```