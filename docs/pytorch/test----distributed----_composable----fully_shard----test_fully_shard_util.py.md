# `.\pytorch\test\distributed\_composable\fully_shard\test_fully_shard_util.py`

```py
# Owner(s): ["oncall: distributed"]  # 标识这段代码的所有者，分布式系统中的责任人

import sys  # 导入系统相关的库，用于处理系统相关操作

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式通信模块
from torch.distributed._composable import fully_shard  # 导入PyTorch的全分片函数
from torch.distributed.fsdp._debug_utils import (
    _get_sharded_module_tree_with_module_name_to_fqns,  # 导入调试工具中的函数
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy  # 导入模块包装策略
from torch.testing._internal.common_dist_composable import CompositeModel, UnitModule  # 导入测试相关的模块和类
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入跳过GPU数目不足的装饰器函数
from torch.testing._internal.common_fsdp import FSDPTest  # 导入FSDP测试相关的类
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN  # 导入运行测试和测试调试标志

if not dist.is_available():  # 如果分布式功能不可用
    print("Distributed not available, skipping tests", file=sys.stderr)  # 打印消息并输出到标准错误流
    sys.exit(0)  # 终止程序

if TEST_WITH_DEV_DBG_ASAN:  # 如果设置了开发调试ASAN（地址检查器）标志
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,  # 打印消息并输出到标准错误流
    )
    sys.exit(0)  # 终止程序


class TestUtils(FSDPTest):  # 定义一个名为TestUtils的测试类，继承自FSDPTest类
    @property
    def world_size(self):  # 定义属性world_size，返回值为2
        return 2

    @property
    def process_group(self):  # 定义属性process_group，返回默认的分布式进程组
        return dist.distributed_c10d._get_default_group()

    @skip_if_lt_x_gpu(2)  # 使用装饰器，跳过GPU数目小于2的测试
    # 测试函数：测试通过模块名到全限定名的字典获取分片模块树信息
    def test_get_sharded_module_tree_with_module_name_to_fqns(self):
        # 创建一个 CompositeModel 对象，使用 CUDA 设备
        model = CompositeModel(torch.device("cuda"))
        # 对模型进行完全分片，使用 UnitModule 的模块包装策略
        fully_shard(
            model,
            policy=ModuleWrapPolicy({UnitModule}),
        )
        # 调用函数获取分片模块树信息和模块名到全限定名字典
        (
            sharded_tree_info,
            sharded_module_name_to_fqns,
        ) = _get_sharded_module_tree_with_module_name_to_fqns(model)
        # 断言模块名到全限定名字典的键值对列表
        self.assertEqual(
            list(sharded_module_name_to_fqns.keys()),
            ["[CompositeModel]", "u1[UnitModule]", "u2[UnitModule]"],
        )
        # 断言模块名到全限定名字典的值列表
        self.assertEqual(
            list(sharded_module_name_to_fqns.values()),
            [
                ["l1.weight", "l1.bias", "l2.weight", "l2.bias"],
                [
                    "u1.l1.weight",
                    "u1.l1.bias",
                    "u1.seq.1.weight",
                    "u1.seq.1.bias",
                    "u1.l2.weight",
                    "u1.l2.bias",
                ],
                [
                    "u2.l1.weight",
                    "u2.l1.bias",
                    "u2.seq.1.weight",
                    "u2.seq.1.bias",
                    "u2.l2.weight",
                    "u2.l2.bias",
                ],
            ],
        )
        # 测试嵌套的 fully_shard 调用
        # 创建一个新的 CompositeModel 对象，使用 CUDA 设备
        new_model = CompositeModel(torch.device("cuda"))
        # 对 new_model.u1 模块进行完全分片
        fully_shard(new_model.u1)
        # 对整个 new_model 模型进行完全分片
        fully_shard(new_model)
        # 再次调用函数获取分片模块树信息和模块名到全限定名字典
        (
            sharded_tree_info,
            sharded_module_name_to_fqns,
        ) = _get_sharded_module_tree_with_module_name_to_fqns(new_model)
        # 断言模块名到全限定名字典的键值对列表
        self.assertEqual(
            list(sharded_module_name_to_fqns.keys()),
            ["[CompositeModel]", "u1[UnitModule]"],
        )
        # 断言模块名到全限定名字典的值列表
        self.assertEqual(
            list(sharded_module_name_to_fqns.values()),
            [
                [
                    "l1.weight",
                    "l1.bias",
                    "u2.l1.weight",
                    "u2.l1.bias",
                    "u2.seq.1.weight",
                    "u2.seq.1.bias",
                    "u2.l2.weight",
                    "u2.l2.bias",
                    "l2.weight",
                    "l2.bias",
                ],
                [
                    "u1.l1.weight",
                    "u1.l1.bias",
                    "u1.seq.1.weight",
                    "u1.seq.1.bias",
                    "u1.l2.weight",
                    "u1.l2.bias",
                ],
            ],
        )
# 如果当前脚本作为主程序运行（而不是被导入到其他模块中执行），则执行以下代码块
if __name__ == "__main__":
    # 调用名为 run_tests 的函数来执行测试
    run_tests()
```