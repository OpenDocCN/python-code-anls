# `stable-diffusion-webui\modules\sysinfo.py`

```
# 导入所需的模块
import json
import os
import sys

import platform
import hashlib
import pkg_resources
import psutil
import re

# 导入自定义模块
import launch
from modules import paths_internal, timer, shared, extensions, errors

# 定义校验令牌和环境白名单
checksum_token = "DontStealMyGamePlz__WINNERS_DONT_USE_DRUGS__DONT_COPY_THAT_FLOPPY"
environment_whitelist = {
    "GIT",
    "INDEX_URL",
    "WEBUI_LAUNCH_LIVE_OUTPUT",
    "GRADIO_ANALYTICS_ENABLED",
    "PYTHONPATH",
    "TORCH_INDEX_URL",
    "TORCH_COMMAND",
    "REQS_FILE",
    "XFORMERS_PACKAGE",
    "CLIP_PACKAGE",
    "OPENCLIP_PACKAGE",
    "STABLE_DIFFUSION_REPO",
    "K_DIFFUSION_REPO",
    "CODEFORMER_REPO",
    "BLIP_REPO",
    "STABLE_DIFFUSION_COMMIT_HASH",
    "K_DIFFUSION_COMMIT_HASH",
    "CODEFORMER_COMMIT_HASH",
    "BLIP_COMMIT_HASH",
    "COMMANDLINE_ARGS",
    "IGNORE_CMD_ARGS_ERRORS",
}

# 定义函数，将字节数转换为可读的格式
def pretty_bytes(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]:
        if abs(num) < 1024 or unit == 'Y':
            return f"{num:.0f}{unit}{suffix}"
        num /= 1024

# 获取结果并返回为 JSON 格式的字符串
def get():
    res = get_dict()

    text = json.dumps(res, ensure_ascii=False, indent=4)

    h = hashlib.sha256(text.encode("utf8"))
    text = text.replace(checksum_token, h.hexdigest())

    return text

# 正则表达式用于匹配校验和
re_checksum = re.compile(r'"Checksum": "([0-9a-fA-F]{64})')

# 检查给定字符串中是否包含校验和
def check(x):
    m = re.search(re_checksum, x)
    if not m:
        return False

    replaced = re.sub(re_checksum, f'"Checksum": "{checksum_token}"', x)

    h = hashlib.sha256(replaced.encode("utf8"))
    return h.hexdigest() == m.group(1)

# 获取系统内存信息
def get_dict():
    ram = psutil.virtual_memory()
    # 创建一个包含系统信息的字典
    res = {
        # 获取操作系统平台信息
        "Platform": platform.platform(),
        # 获取 Python 版本信息
        "Python": platform.python_version(),
        # 获取启动时的 Git 标签
        "Version": launch.git_tag(),
        # 获取启动时的提交哈希值
        "Commit": launch.commit_hash(),
        # 获取脚本路径
        "Script path": paths_internal.script_path,
        # 获取数据路径
        "Data path": paths_internal.data_path,
        # 获取扩展目录路径
        "Extensions dir": paths_internal.extensions_dir,
        # 获取校验和
        "Checksum": checksum_token,
        # 获取命令行参数
        "Commandline": get_argv(),
        # 获取 Torch 环境信息
        "Torch env info": get_torch_sysinfo(),
        # 获取异常信息
        "Exceptions": errors.get_exceptions(),
        # 获取 CPU 信息
        "CPU": {
            # 获取 CPU 型号
            "model": platform.processor(),
            # 获取逻辑 CPU 数量
            "count logical": psutil.cpu_count(logical=True),
            # 获取物理 CPU 数量
            "count physical": psutil.cpu_count(logical=False),
        },
        # 获取 RAM 信息
        "RAM": {
            # 遍历 RAM 属性，获取指定属性的字节大小并格式化
            x: pretty_bytes(getattr(ram, x, 0)) for x in ["total", "used", "free", "active", "inactive", "buffers", "cached", "shared"] if getattr(ram, x, 0) != 0
        },
        # 获取已启用的扩展信息
        "Extensions": get_extensions(enabled=True),
        # 获取未启用的扩展信息
        "Inactive extensions": get_extensions(enabled=False),
        # 获取环境信息
        "Environment": get_environment(),
        # 获取配置信息
        "Config": get_config(),
        # 获取启动记录
        "Startup": timer.startup_record,
        # 获取已安装包的信息
        "Packages": sorted([f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set]),
    }
    
    # 返回系统信息字典
    return res
# 返回环境变量中在白名单中的键值对
def get_environment():
    return {k: os.environ[k] for k in sorted(os.environ) if k in environment_whitelist}


# 获取命令行参数，对于包含敏感信息的参数进行隐藏处理
def get_argv():
    res = []

    for v in sys.argv:
        if shared.cmd_opts.gradio_auth and shared.cmd_opts.gradio_auth == v:
            res.append("<hidden>")
            continue

        if shared.cmd_opts.api_auth and shared.cmd_opts.api_auth == v:
            res.append("<hidden>")
            continue

        res.append(v)

    return res

# 编译正则表达式，用于匹配换行符
re_newline = re.compile(r"\r*\n")


# 获取 Torch 系统信息
def get_torch_sysinfo():
    try:
        import torch.utils.collect_env
        info = torch.utils.collect_env.get_env_info()._asdict()

        return {k: re.split(re_newline, str(v)) if "\n" in str(v) else v for k, v in info.items()}
    except Exception as e:
        return str(e)


# 获取已启用或未启用的扩展信息
def get_extensions(*, enabled):

    try:
        def to_json(x: extensions.Extension):
            return {
                "name": x.name,
                "path": x.path,
                "version": x.version,
                "branch": x.branch,
                "remote": x.remote,
            }

        return [to_json(x) for x in extensions.extensions if not x.is_builtin and x.enabled == enabled]
    except Exception as e:
        return str(e)


# 获取配置信息
def get_config():
    try:
        return shared.opts.data
    except Exception as e:
        return str(e)
```