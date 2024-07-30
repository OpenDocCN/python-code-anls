# `.\comic-translate\modules\ocr\pororo\pororo\__init__.py`

```py
# 从相对当前位置的路径导入模块版本号作为 __version__，并忽略 PEP8 的规范检查
from ..pororo.__version__ import version as __version__  # noqa

# 从相对当前位置的路径导入 Pororo 类，用于后续的自然语言处理任务
from ..pororo.pororo import Pororo  # noqa
```