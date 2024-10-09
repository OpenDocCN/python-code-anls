# `.\MinerU\update_version.py`

```
# 导入操作系统模块
import os
# 导入子进程模块，用于执行外部命令
import subprocess


# 定义获取版本号的函数
def get_version():
    # 定义要执行的 Git 命令，以获取标签信息
    command = ["git", "describe", "--tags"]
    # 尝试执行命令并获取输出
    try:
        # 获取命令输出并解码为字符串，同时去除首尾空白
        version = subprocess.check_output(command).decode().strip()
        # 按照 "-" 分隔版本字符串，得到各部分
        version_parts = version.split("-")
        # 检查版本部分是否符合预期格式
        if len(version_parts) > 1 and version_parts[0].startswith("magic_pdf"):
            # 如果符合格式，返回版本号
            return version_parts[1]
        else:
            # 否则，抛出值错误异常，提示格式错误
            raise ValueError(f"Invalid version tag {version}. Expected format is magic_pdf-<version>-released.")
    # 捕获所有异常
    except Exception as e:
        # 打印异常信息
        print(e)
        # 返回默认版本号
        return "0.0.0"


# 定义将版本号写入文件的函数
def write_version_to_commons(version):
    # 构建版本文件的路径
    commons_path = os.path.join(os.path.dirname(__file__), 'magic_pdf', 'libs', 'version.py')
    # 打开版本文件以写入
    with open(commons_path, 'w') as f:
        # 写入版本号信息
        f.write(f'__version__ = "{version}"\n')


# 如果当前脚本是主程序
if __name__ == '__main__':
    # 获取版本名称
    version_name = get_version()
    # 将版本名称写入文件
    write_version_to_commons(version_name)
```