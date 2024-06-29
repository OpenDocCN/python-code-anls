# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\mplot3d\tests\__init__.py`

```
# 导入 Path 类，用于处理文件路径
from pathlib import Path

# 检查测试目录是否存在
# 使用 __file__ 获取当前脚本文件的路径，再使用 parent 获取其父目录，
# 然后拼接字符串 "baseline_images" 构成目标路径，
# 最后使用 exists() 方法检查该路径是否存在。
if not (Path(__file__).parent / "baseline_images").exists():
    # 如果目标路径不存在，抛出 OSError 异常
    raise OSError(
        'The baseline image directory does not exist. '
        'This is most likely because the test data is not installed. '
        'You may need to install matplotlib from source to get the '
        'test data.')
```