# `.\pytorch\torch\distributed\elastic\timer\debug_info_logging.py`

```py
#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 导入必要的模块和函数
from typing import Dict, List  # 导入类型提示需要的 Dict 和 List

# 从 torch.distributed.elastic.utils.logging 模块中导入 get_logger 函数
from torch.distributed.elastic.utils.logging import get_logger

# 获取 logger 对象
logger = get_logger(__name__)

# 定义公开的函数列表
__all__ = ["log_debug_info_for_expired_timers"]


# 定义函数 log_debug_info_for_expired_timers，接受两个参数
def log_debug_info_for_expired_timers(
    run_id: str,
    expired_timers: Dict[int, List[str]],
):
    # 检查 expired_timers 是否非空
    if expired_timers:
        # 使用 logger 对象记录信息，指出运行ID和过期的计时器信息
        logger.info("Timers expired for run:[%s] [%s].", run_id, expired_timers)
```