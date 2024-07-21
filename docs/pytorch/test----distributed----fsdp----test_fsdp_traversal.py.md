# `.\pytorch\test\distributed\fsdp\test_fsdp_traversal.py`

```
# Owner(s): ["oncall: distributed"]

# 导入系统模块 sys
import sys

# 导入 torch 分布式相关模块
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    NestedWrappedModule,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN

# 如果分布式不可用，输出消息并退出
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果使用 dev-asan 测试模式，由于 torch 和 multiprocessing spawn 存在已知问题，跳过测试
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 测试类 TestTraversal 继承自 FSDPTest 类
class TestTraversal(FSDPTest):
    
    # 定义 world_size 属性为 2
    @property
    def world_size(self):
        return 2

    # 修饰器，如果 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_fsdp_modules(self):
        # 初始化 NestedWrappedModule 对象
        nested_wrapped_module = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE,
        )
        # 获取 FSDP 模块列表
        modules = FSDP.fsdp_modules(nested_wrapped_module)
        # 断言模块列表内容
        self.assertEqual(
            modules,
            [
                nested_wrapped_module.module.get_submodule("1"),
                nested_wrapped_module.module.get_submodule("1").get_submodule("0"),
                nested_wrapped_module.module.get_submodule("2"),
            ],
        )
        # 获取仅根模块的 FSDP 模块列表
        modules = FSDP.fsdp_modules(nested_wrapped_module, root_only=True)
        # 断言模块列表内容
        self.assertEqual(
            modules,
            [
                nested_wrapped_module.module.get_submodule("1"),
                nested_wrapped_module.module.get_submodule("2"),
            ],
        )


# 如果当前文件为主程序，运行测试
if __name__ == "__main__":
    run_tests()
```