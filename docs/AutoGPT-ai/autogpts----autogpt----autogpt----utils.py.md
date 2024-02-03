# `.\AutoGPT\autogpts\autogpt\autogpt\utils.py`

```py
# 导入必要的模块
from pathlib import Path
import yaml
from colorama import Fore

# 定义函数，用于验证 YAML 文件的有效性
def validate_yaml_file(file: str | Path):
    try:
        # 打开文件，指定编码为 utf-8
        with open(file, encoding="utf-8") as fp:
            # 加载 YAML 文件内容，使用 yaml.FullLoader 加载器
            yaml.load(fp.read(), Loader=yaml.FullLoader)
    except FileNotFoundError:
        # 处理文件未找到的异常
        return (False, f"The file {Fore.CYAN}`{file}`{Fore.RESET} wasn't found")
    except yaml.YAMLError as e:
        # 处理 YAML 文件读取异常
        return (
            False,
            f"There was an issue while trying to read with your AI Settings file: {e}",
        )

    # 返回验证结果
    return (True, f"Successfully validated {Fore.CYAN}`{file}`{Fore.RESET}!")
```