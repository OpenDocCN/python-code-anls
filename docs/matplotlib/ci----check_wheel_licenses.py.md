# `D:\src\scipysrc\matplotlib\ci\check_wheel_licenses.py`

```py
#!/usr/bin/env python3
"""
Check that all specified .whl files have the correct LICENSE files included.

To run:
    $ python3 -m build --wheel
    $ ./ci/check_wheel_licenses.py dist/*.whl
"""

# 导入所需模块
from pathlib import Path  # 导入处理路径的模块
import sys  # 导入系统相关模块，用于处理命令行参数和退出
import zipfile  # 导入处理 ZIP 文件的模块


# 检查命令行参数是否至少包含一个 .whl 文件名，否则退出
if len(sys.argv) <= 1:
    sys.exit('At least one wheel must be specified in command-line arguments.')

# 获取项目目录和 LICENSE 目录
project_dir = Path(__file__).parent.resolve().parent  # 获取脚本所在目录的父级目录作为项目目录
license_dir = project_dir / 'LICENSE'  # 设置 LICENSE 文件目录路径

# 获取 LICENSE 目录下所有文件的文件名，并放入集合中
license_file_names = {path.name for path in sorted(license_dir.glob('*'))}

# 遍历每个指定的 .whl 文件进行 LICENSE 文件检查
for wheel in sys.argv[1:]:
    print(f'Checking LICENSE files in: {wheel}')
    # 使用 zipfile 模块打开当前 .whl 文件
    with zipfile.ZipFile(wheel) as f:
        # 获取 .whl 文件中所有以 .dist-info/LICENSE 结尾的文件名，并放入集合中
        wheel_license_file_names = {Path(path).name
                                    for path in sorted(f.namelist())
                                    if '.dist-info/LICENSE' in path}
        # 检查是否 .whl 文件中的 LICENSE 文件名集合包含所有项目 LICENSE 目录下的文件名
        if not (len(wheel_license_file_names) and
                wheel_license_file_names.issuperset(license_file_names)):
            # 如果不包含所有文件名，则输出错误信息并退出程序
            sys.exit(f'LICENSE file(s) missing:\n'
                     f'{wheel_license_file_names} !=\n'
                     f'{license_file_names}')
```