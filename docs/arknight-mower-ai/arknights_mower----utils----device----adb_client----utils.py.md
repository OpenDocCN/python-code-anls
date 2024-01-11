# `arknights-mower\arknights_mower\utils\device\adb_client\utils.py`

```
# 导入必要的模块和类型提示
from __future__ import annotations
import shutil
import subprocess
from typing import Union

# 导入特定的模块和变量
from .... import __system__
from ... import config
from ...log import logger
from ..utils import download_file

# 定义内置 ADB 工具的下载链接和文件列表
ADB_BUILDIN_URL = 'https://oss.nano.ac/arknights_mower/adb-binaries'
ADB_BUILDIN_FILELIST = {
    'linux': ['adb'],
    'windows': ['adb.exe', 'AdbWinApi.dll', 'AdbWinUsbApi.dll'],
    'darwin': ['adb'],
}

# 定义下载内置 ADB 工具的函数
def adb_buildin() -> None:
    """ download adb_bin """
    # 初始化 ADB 内置工具的文件夹
    folder = config.init_adb_buildin()
    folder.mkdir(exist_ok=True, parents=True)
    # 检查当前系统是否在支持的系统列表中
    if __system__ not in ADB_BUILDIN_FILELIST.keys():
        raise NotImplementedError(f'Unknown system: {__system__}')
    # 遍历当前系统所需的文件列表，下载并拷贝到指定文件夹
    for file in ADB_BUILDIN_FILELIST[__system__]:
        target_path = folder / file
        if not target_path.exists():
            url = f'{ADB_BUILDIN_URL}/{__system__}/{file}'
            logger.debug(f'adb_buildin: {url}')
            tmp_path = download_file(url)
            shutil.copy(tmp_path, str(target_path))
    # 设置内置 ADB 工具的路径，并修改权限为可执行
    config.ADB_BUILDIN = folder / ADB_BUILDIN_FILELIST[__system__][0]
    config.ADB_BUILDIN.chmod(0o744)

# 定义运行命令的函数
def run_cmd(cmd: list[str], decode: bool = False) -> Union[bytes, str]:
    # 打印要执行的命令
    logger.debug(f"run command: {cmd}")
    try:
        # 执行命令并获取输出
        r = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        # 如果命令执行出错，打印错误输出并抛出异常
        logger.debug(e.output)
        raise e
    # 如果需要解码输出，则进行解码并返回字符串，否则返回字节流
    if decode:
        return r.decode('utf8')
    return r
```