# `.\pytorch\test\distributed\fsdp\test_distributed_checkpoint.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入系统库和临时文件库
import sys
import tempfile

# 导入PyTorch相关库
import torch
from torch import distributed as dist
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load_state_dict,
    save_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, SkipModel
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)

# 如果分布式不可用，输出错误信息并退出
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果使用开发调试的ASAN（地址检测），输出相应信息并退出，因为torch和多进程生成存在已知问题
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 定义支持的分布式状态字典类型集合
_DISTRIBUTED_STATE_DICT_IMPLS = {
    StateDictType.LOCAL_STATE_DICT,
    StateDictType.SHARDED_STATE_DICT,
}


# 定义测试类 TestDistributedCheckpoint，继承自 FSDPTest
class TestDistributedCheckpoint(FSDPTest):

    # 定义 world_size 属性为 2，指定测试时的分布式环境中节点数
    @property
    def world_size(self):
        return 2

    # 装饰器，要求至少有两个GPU，否则跳过测试
    @skip_if_lt_x_gpu(2)

    # 参数化装饰器，用于定义测试函数的参数 state_dict_type，参数值来自 _DISTRIBUTED_STATE_DICT_IMPLS
    @parametrize("state_dict_type", _DISTRIBUTED_STATE_DICT_IMPLS)
    # 定义一个测试函数，用于测试分布式检查点功能，接受状态字典类型参数
    def test_distributed_checkpoint(self, state_dict_type) -> None:
        # 启用分布式包装器 FSDP，并设定包装器类为 FSDP
        with enable_wrap(wrapper_cls=FSDP):
            # 设定随机种子为 100
            torch.manual_seed(100)
            # 创建一个使用 FSDP 包装的 SkipModel 模型实例，双重嵌套
            model = wrap(SkipModel(double_nest=True))
            # 设定随机种子为 200
            torch.manual_seed(200)
            # 创建另一个使用 FSDP 包装的 SkipModel 模型实例，双重嵌套
            new_model = wrap(SkipModel(double_nest=True))

        # 将 model 和 new_model 的参数分片化，并且检查它们的参数不相等
        with FullyShardedDataParallel.summon_full_params(
            model
        ), FullyShardedDataParallel.summon_full_params(new_model):
            params = list(model.parameters())
            new_params = list(new_model.parameters())
            self.assertNotEqual(params, new_params)

        # 在临时目录中创建一个临时文件夹，并进行路径广播
        with tempfile.TemporaryDirectory() as path:
            paths = [path]
            dist.broadcast_object_list(paths)
            path = paths[0]
            # 创建一个 FileSystemWriter 实例用于写入文件系统
            writer = FileSystemWriter(path)
            # 创建一个 FileSystemReader 实例用于读取文件系统
            reader = FileSystemReader(path)
            # 使用 FSDP 包装器的状态字典类型，保存 model 的状态字典
            with FSDP.state_dict_type(model, state_dict_type), FSDP.state_dict_type(
                new_model, state_dict_type
            ):
                state_dict = model.state_dict()

            # 将状态字典保存到文件系统中
            save_state_dict(state_dict, writer)

            # 使用 FSDP 包装器的状态字典类型，加载 new_model 的状态字典并应用
            with FSDP.state_dict_type(model, state_dict_type), FSDP.state_dict_type(
                new_model, state_dict_type
            ):
                state_dict = new_model.state_dict()
                # 加载状态字典到 new_model 中
                load_state_dict(state_dict, reader)
                new_model.load_state_dict(state_dict)

        # 再次将 model 和 new_model 的参数分片化，并且检查它们的参数相等
        with FullyShardedDataParallel.summon_full_params(
            model
        ), FullyShardedDataParallel.summon_full_params(new_model):
            params = list(model.parameters())
            new_params = list(new_model.parameters())
            self.assertEqual(params, new_params)

        # 待添加重新分片测试用例。
# 调用一个函数来实例化参数化测试，传入的参数为 TestDistributedCheckpoint 类
instantiate_parametrized_tests(TestDistributedCheckpoint)

# 检查当前模块是否作为主程序运行，如果是，则执行测试
if __name__ == "__main__":
    run_tests()
```