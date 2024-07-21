# `.\pytorch\torch\distributed\rpc\_utils.py`

```
# mypy: allow-untyped-defs
# 导入日志模块
import logging
# 导入上下文管理器模块
from contextlib import contextmanager
# 导入类型转换工具
from typing import cast

# 从当前目录导入 api 和 TensorPipeAgent 模块
from . import api, TensorPipeAgent

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义一个上下文管理器函数，用于管理群组成员管理操作
@contextmanager
def _group_membership_management(store, name, is_join):
    # 定义群组管理令牌的键名
    token_key = "RpcGroupManagementToken"
    # 根据是否加入群组确定操作类型
    join_or_leave = "join" if is_join else "leave"
    # 创建当前操作的令牌名
    my_token = f"Token_for_{name}_{join_or_leave}"
    
    while True:
        # 从存储中获取令牌，用于表示开始加入/离开群组的关键段
        returned = store.compare_set(token_key, "", my_token).decode()
        # 如果返回的令牌与当前令牌相同
        if returned == my_token:
            # 让出控制权，执行上下文管理器包装的函数
            yield
            # 完成后，清除令牌，表示关键段结束
            store.set(token_key, "")
            # 设置当前令牌为 "Done"，以通知其他等待的进程可以执行
            store.set(my_token, "Done")
            break
        else:
            # 存储将等待令牌被释放
            try:
                # 等待其他令牌被释放
                store.wait([returned])
            except RuntimeError:
                # 如果等待超时，记录错误日志
                logger.error(
                    "Group membership token %s timed out waiting for %s to be released.",
                    my_token,
                    returned,
                )
                raise

# 更新群组成员关系的私有函数
def _update_group_membership(worker_info, my_devices, reverse_device_map, is_join):
    # 强制类型转换获取当前的 TensorPipeAgent 实例
    agent = cast(TensorPipeAgent, api._get_current_rpc_agent())
    # 调用 TensorPipeAgent 的成员函数来更新群组成员关系
    ret = agent._update_group_membership(
        worker_info, my_devices, reverse_device_map, is_join
    )
    return ret
```