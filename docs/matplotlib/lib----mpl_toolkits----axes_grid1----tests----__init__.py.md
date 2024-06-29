# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axes_grid1\tests\__init__.py`

```py
# 导入Path类，用于处理文件和目录路径
from pathlib import Path

# 检查测试目录是否存在
if not (Path(__file__).parent / "baseline_images").exists():
    # 如果基准图像目录不存在，抛出OSError异常
    raise OSError(
        'The baseline image directory does not exist. '
        'This is most likely because the test data is not installed. '
        'You may need to install matplotlib from source to get the '
        'test data.')
```