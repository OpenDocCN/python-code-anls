# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axisartist\tests\__init__.py`

```
# 导入 Path 类
from pathlib import Path

# 检查测试目录是否存在
if not (Path(__file__).parent / "baseline_images").exists():
    # 如果基准图像目录不存在，则抛出 OSError 异常
    raise OSError(
        'The baseline image directory does not exist. '
        'This is most likely because the test data is not installed. '
        'You may need to install matplotlib from source to get the '
        'test data.')
```