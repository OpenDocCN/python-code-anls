# `.\yolov8\ultralytics\utils\dist.py`

```py
# 导入必要的模块和函数
import os  # 系统操作模块
import shutil  # 文件操作模块
import socket  # 网络通信模块
import sys  # 系统模块
import tempfile  # 临时文件模块

from . import USER_CONFIG_DIR  # 导入当前目录下的 USER_CONFIG_DIR 变量
from .torch_utils import TORCH_1_9  # 导入 TORCH_1_9 变量和函数

# 查找本地空闲网络端口的函数
def find_free_network_port() -> int:
    """
    Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))  # 绑定到本地地址，端口号自动分配
        return s.getsockname()[1]  # 返回分配的端口号


# 生成并返回一个 DDP 文件的函数
def generate_ddp_file(trainer):
    """Generates a DDP file and returns its file name."""
    module, name = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)

    content = f"""
# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)
overrides = {vars(trainer)}

if __name__ == "__main__":
    from {module} import {name}
    from ultralytics.utils import DEFAULT_CFG_DICT

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # 处理额外的键 'save_dir'
    trainer = {name}(cfg=cfg, overrides=overrides)
    trainer.args.model = "{getattr(trainer.hub_session, 'model_url', trainer.args.model)}"
    results = trainer.train()
"""
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)  # 创建存放 DDP 文件的目录
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(trainer)}.py",
        mode="w+",
        encoding="utf-8",
        dir=USER_CONFIG_DIR / "DDP",
        delete=False,
    ) as file:
        file.write(content)  # 写入临时文件内容
    return file.name  # 返回临时文件的文件名


# 生成并返回分布式训练命令的函数
def generate_ddp_command(world_size, trainer):
    """Generates and returns command for distributed training."""
    import __main__  # 本地导入，避免特定问题

    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # 删除保存目录
    file = generate_ddp_file(trainer)  # 生成 DDP 文件
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    port = find_free_network_port()  # 获取空闲端口号
    cmd = [sys.executable, "-m", dist_cmd, "--nproc_per_node", f"{world_size}", "--master_port", f"{port}", file]
    return cmd, file  # 返回生成的命令和文件名


# 清理函数，删除生成的临时文件
def ddp_cleanup(trainer, file):
    """Delete temp file if created."""
    if f"{id(trainer)}.py" in file:  # 如果文件名包含临时文件的标识
        os.remove(file)  # 删除临时文件
```