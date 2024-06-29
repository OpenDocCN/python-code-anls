# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\__init__.py`

```
# 导入Path类，用于处理文件和目录路径
from pathlib import Path

# 检查测试目录是否存在
# 使用__file__获取当前文件的路径，通过Path(__file__).parent获取其父目录路径
# 使用 / 运算符将父目录路径与 'baseline_images' 目录名连接起来，然后调用exists()方法检查目录是否存在
if not (Path(__file__).parent / 'baseline_images').exists():
    # 如果 'baseline_images' 目录不存在，抛出OSError异常
    raise OSError(
        'The baseline image directory does not exist. '
        'This is most likely because the test data is not installed. '
        'You may need to install matplotlib from source to get the '
        'test data.')
```