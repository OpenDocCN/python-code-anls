# `.\pytorch\torch\distributed\elastic\multiprocessing\subprocess_handler\subprocess_handler.py`

```py
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os  # 导入操作系统相关功能模块
import signal  # 导入信号处理模块
import subprocess  # 导入子进程管理模块
import sys  # 导入系统相关模块
from typing import Any, Dict, Optional, Tuple  # 导入类型提示相关模块


__all__ = ["SubprocessHandler"]  # 指定导出的模块成员列表

IS_WINDOWS = sys.platform == "win32"  # 判断当前操作系统是否为Windows


def _get_default_signal() -> signal.Signals:
    """Get the default termination signal. SIGTERM for unix, CTRL_C_EVENT for windows."""
    if IS_WINDOWS:
        return signal.CTRL_C_EVENT  # 返回Windows系统的默认终止信号 CTRL_C_EVENT
    else:
        return signal.SIGTERM  # 返回Unix系统的默认终止信号 SIGTERM


class SubprocessHandler:
    """
    Convenience wrapper around python's ``subprocess.Popen``. Keeps track of
    meta-objects associated to the process (e.g. stdout and stderr redirect fds).
    """

    def __init__(
        self,
        entrypoint: str,
        args: Tuple,
        env: Dict[str, str],
        stdout: str,
        stderr: str,
        local_rank_id: int,
    ):
        self._stdout = open(stdout, "w") if stdout else None  # 打开标准输出文件，若无则设为None
        self._stderr = open(stderr, "w") if stderr else None  # 打开标准错误输出文件，若无则设为None
        # inherit parent environment vars
        env_vars = os.environ.copy()  # 复制当前环境变量
        env_vars.update(env)  # 更新环境变量，添加额外的环境变量参数

        args_str = (entrypoint, *[str(e) for e in args])  # 构建命令行参数列表
        self.local_rank_id = local_rank_id  # 设置本地进程ID
        self.proc: subprocess.Popen = self._popen(args_str, env_vars)  # 使用给定的命令和环境变量启动子进程

    def _popen(self, args: Tuple, env: Dict[str, str]) -> subprocess.Popen:
        kwargs: Dict[str, Any] = {}
        if not IS_WINDOWS:
            kwargs["start_new_session"] = True  # 对于非Windows系统，设置启动新会话
        return subprocess.Popen(
            args=args,  # 子进程的命令行参数
            env=env,  # 子进程的环境变量
            stdout=self._stdout,  # 子进程的标准输出重定向
            stderr=self._stderr,  # 子进程的标准错误输出重定向
            **kwargs,  # 其他参数（如start_new_session）
        )

    def close(self, death_sig: Optional[signal.Signals] = None) -> None:
        if not death_sig:
            death_sig = _get_default_signal()  # 获取默认的终止信号
        if IS_WINDOWS:
            self.proc.send_signal(death_sig)  # 发送终止信号到子进程（仅限Windows）
        else:
            os.killpg(self.proc.pid, death_sig)  # 终止进程组中的所有进程（非Windows系统）
        if self._stdout:
            self._stdout.close()  # 关闭标准输出文件
        if self._stderr:
            self._stderr.close()  # 关闭标准错误输出文件
```