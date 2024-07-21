# `.\pytorch\torch\distributed\__init__.py`

```
# mypy: allow-untyped-defs
# 引入pdb模块，用于调试和分析程序执行过程中的问题
import pdb
# 引入sys模块，提供对Python解释器进行访问的标准库
import sys

# 引入torch模块，PyTorch深度学习框架的主要入口
import torch


def is_available() -> bool:
    """
    返回一个布尔值，指示是否可用分布式包。

    否则，
    ``torch.distributed`` 不会暴露任何其他API。当前，
    ``torch.distributed`` 在Linux、MacOS和Windows上可用。在构建PyTorch时设置``USE_DISTRIBUTED=1``以启用它。
    目前，默认值是Linux和Windows上的``USE_DISTRIBUTED=1``，
    MacOS上为``USE_DISTRIBUTED=0``。
    """
    return hasattr(torch._C, "_c10d_init")


if is_available() and not torch._C._c10d_init():
    raise RuntimeError("Failed to initialize torch.distributed")

# 自定义从分布式包抛出的运行时错误
DistError = torch._C._DistError
DistBackendError = torch._C._DistBackendError
DistNetworkError = torch._C._DistNetworkError
DistStoreError = torch._C._DistStoreError

if is_available():
    from torch._C._distributed_c10d import (
        _broadcast_coalesced,
        _compute_bucket_assignment_by_size,
        _ControlCollectives,
        _DEFAULT_FIRST_BUCKET_BYTES,
        _make_nccl_premul_sum,
        _register_builtin_comm_hook,
        _register_comm_hook,
        _StoreCollectives,
        _test_python_store,
        _verify_params_across_processes,
        Backend as _Backend,
        BuiltinCommHookType,
        DebugLevel,
        FileStore,
        get_debug_level,
        GradBucket,
        Logger,
        PrefixStore,
        ProcessGroup as ProcessGroup,
        Reducer,
        set_debug_level,
        set_debug_level_from_env,
        Store,
        TCPStore,
        Work as _Work,
    )

    class _DistributedPdb(pdb.Pdb):
        """
        支持从多进程子进程内使用PDB调试。

        用法:
        _DistributedPdb().set_trace()
        """

        def interaction(self, *args, **kwargs):
            _stdin = sys.stdin
            try:
                sys.stdin = open("/dev/stdin")
                pdb.Pdb.interaction(self, *args, **kwargs)
            finally:
                sys.stdin = _stdin
    # 定义一个函数 `breakpoint`，用于设置断点，但只在指定的进程编号上生效。其他进程将在断点处等待。
    def breakpoint(rank: int = 0):
        """
        Set a breakpoint, but only on a single rank.  All other ranks will wait for you to be
        done with the breakpoint before continuing.

        Args:
            rank (int): Which rank to break on.  Default: ``0``
        """
        # 如果当前进程编号等于指定的断点进程编号 `rank`，则创建 `_DistributedPdb` 对象
        if get_rank() == rank:
            pdb = _DistributedPdb()
            # 输出调试信息，提示用户输入 'up' 来回到调用 `dist.breakpoint(rank={rank})` 的帧
            pdb.message(
                "\n!!! ATTENTION !!!\n\n"
                f"Type 'up' to get to the frame that called dist.breakpoint(rank={rank})\n"
            )
            # 设置断点，等待用户交互
            pdb.set_trace()
        
        # 如果 TLS（线程本地存储）中包含 Meta/Python 键，则确保忽略它们并使用 CPU/CUDA 的默认实现来执行 barrier
        meta_in_tls = torch._C._meta_in_tls_dispatch_include()
        # 创建一个 `_DisableTorchDispatch` 对象作为保护锁
        guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
        # 设置 Meta/Python 键不包含在 TLS 中
        torch._C._set_meta_in_tls_dispatch_include(False)
        try:
            # 执行 barrier 操作
            barrier()
        finally:
            # 恢复 TLS 中的 Meta/Python 键的状态，并删除保护锁
            torch._C._set_meta_in_tls_dispatch_include(meta_in_tls)
            del guard

    # 如果操作系统平台不是 Windows
    if sys.platform != "win32":
        # 从 `torch._C._distributed_c10d` 中导入 `_round_robin_process_groups` 和 `HashStore`
        from torch._C._distributed_c10d import _round_robin_process_groups, HashStore

    # 从 `device_mesh` 模块中导入 `DeviceMesh` 和 `init_device_mesh` 函数
    from .device_mesh import DeviceMesh, init_device_mesh

    # 变量名以下划线开头的不会自动导入
    # 参见 `distributed_c10d.py` 中关于 `_backend` 的注释，解释了为什么需要将其暴露出来
    from .distributed_c10d import *  # noqa: F403

    # 从 `distributed_c10d` 模块中导入以下指定的符号
    from .distributed_c10d import (
        _all_gather_base,
        _coalescing_manager,
        _CoalescingManager,
        _create_process_group_wrapper,
        _get_process_group_name,
        _rank_not_in_group,
        _reduce_scatter_base,
        get_node_local_rank,
    )

    # 从 `remote_device` 模块中导入 `_remote_device` 符号
    from .remote_device import _remote_device

    # 从 `rendezvous` 模块中导入以下指定的符号
    from .rendezvous import (
        _create_store_from_options,
        register_rendezvous_handler,
        rendezvous,
    )

    # 根据环境变量设置调试级别
    set_debug_level_from_env()
else:
    # 当 USE_DISTRIBUTED=0 时，为了确保
    # python test/test_public_bindings.py -k test_correct_module_names 能正常工作，这个存根就足够了。
    # 如果需要，可以随时添加更多的存根。
    # 由于会混淆 pyre，我们无法直接定义存根。

    # 定义一个名为 _ProcessGroupStub 的空类作为存根
    class _ProcessGroupStub:
        pass

    # 将 _ProcessGroupStub 赋给 sys.modules["torch.distributed"].ProcessGroup
    # type: ignore[attr-defined] 表示忽略对 ProcessGroup 属性定义的类型检查
    sys.modules["torch.distributed"].ProcessGroup = _ProcessGroupStub
```