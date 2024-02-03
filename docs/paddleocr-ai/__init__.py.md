# `.\PaddleOCR\__init__.py`

```
# 版权声明和许可证信息
# 版权所有 (c) 2020 PaddlePaddle 作者。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“按原样”分发的，没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。
# 导入 paddleocr 模块
from .paddleocr import *

# 版本号
__version__ = paddleocr.VERSION
# 导出的模块列表
__all__ = [
    'PaddleOCR', 'PPStructure', 'draw_ocr', 'draw_structure_result',
    'save_structure_res', 'download_with_progressbar', 'sorted_layout_boxes',
    'convert_info_docx', 'to_excel'
]
```