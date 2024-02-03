# `stable-diffusion-webui\modules\restart.py`

```py
# 导入必要的模块
import os
from pathlib import Path

# 从内部路径模块中导入脚本路径
from modules.paths_internal import script_path

# 检查环境变量中是否存在 SD_WEBUI_RESTART，如果存在则返回 True，表示 webui 可以重新启动
def is_restartable() -> bool:
    """
    Return True if the webui is restartable (i.e. there is something watching to restart it with)
    """
    return bool(os.environ.get('SD_WEBUI_RESTART'))

# 重新启动程序的函数
def restart_program() -> None:
    """creates file tmp/restart and immediately stops the process, which webui.bat/webui.sh interpret as a command to start webui again"""

    # 获取临时目录路径
    tmpdir = Path(script_path) / "tmp"
    # 创建临时目录，如果不存在则创建
    tmpdir.mkdir(parents=True, exist_ok=True)
    # 在临时目录下创建名为 "restart" 的文件
    (tmpdir / "restart").touch()

    # 停止当前程序的函数
    stop_program()

# 停止程序的函数
def stop_program() -> None:
    # 退出程序，参数为 0 表示正常退出
    os._exit(0)
```