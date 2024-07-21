# `.\pytorch\test\distributed\pipelining\test_pipe.py`

```
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# 从模型注册表中导入所需的模块
from model_registry import MLPModule, ModelWithParamAlias

# 导入PyTorch相关模块
import torch
from torch.distributed.pipelining import pipe_split, pipeline
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

# 定义隐藏层维度和微批量大小
d_hid = 512
microbatch_size = 16

# 设置随机种子
torch.manual_seed(0)


# 定义一个示例模型类 ExampleCode
class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化模型参数
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        # 定义线性层
        self.lin1 = torch.nn.Linear(d_hid, d_hid)
        self.lin2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y):
        # 第一次使用 mm_param1 参数进行矩阵乘法
        x = torch.mm(x, self.mm_param1)  # mutli-use param
        skip_connection = x  # 保留当前状态用于后续跳跃连接
        x = x + y  # 加上输入 y
        x = torch.relu(x)  # ReLU 激活函数
        pipe_split()  # 管道分割点
        # 再次使用 mm_param1 参数进行矩阵乘法
        x = torch.mm(x, self.mm_param1)  # mutli-use param
        x = self.lin1(x)  # 第一个线性层
        pipe_split()  # 管道分割点
        x = torch.relu(x)  # ReLU 激活函数
        x = x + skip_connection  # 添加之前保留的 skip_connection
        x = torch.mm(x, self.mm_param2)  # 使用 mm_param2 参数进行矩阵乘法
        pipe_split()  # 管道分割点
        x = self.lin2(x)  # 第二个线性层
        x = torch.relu(x)  # ReLU 激活函数
        return x


# 定义多个 MLP 的模型类 MultiMLP
class MultiMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化多个 MLP 模块
        self.mlp0 = MLPModule(d_hid)
        self.mlp1 = MLPModule(d_hid)
        self.mlp2 = MLPModule(d_hid)
        self.mlp3 = MLPModule(d_hid)

    def forward(self, x, y):
        x = self.mlp0(x)  # 第一个 MLP 模块
        pipe_split()  # 管道分割点
        x = self.mlp1(x)  # 第二个 MLP 模块
        pipe_split()  # 管道分割点
        x = self.mlp2(x)  # 第三个 MLP 模块
        pipe_split()  # 管道分割点
        x = self.mlp3(x)  # 第四个 MLP 模块
        return x - y  # 返回结果与输入 y 的差值


# 定义期望的管道阶段数目字典
EXPECTED_N_STAGES = {
    ExampleCode: 4,
    MultiMLP: 4,
    ModelWithParamAlias: 2,
}

# 当前不强制要求原始模型和流水线化模型在 FQN 集合上完全相等
# 因为在多次使用参数的情况下，流水线将从 state_dict 中去重 FQN
# TODO: 进一步处理 FQN 集合相等性的检查
CHECK_FQN_SET_EQUALITY = False


# 定义管道测试类 PipeTests，继承自 TestCase
class PipeTests(TestCase):
    @parametrize("ModelClass", [ExampleCode, MultiMLP, ModelWithParamAlias])
    # 定义一个测试模型分割的方法，接受一个 ModelClass 参数
    def test_model_split(self, ModelClass):
        # 实例化给定的 ModelClass 类来创建模型对象
        mod = ModelClass()
        # 生成一个形状为 (microbatch_size, d_hid) 的随机张量 x
        x = torch.randn(microbatch_size, d_hid)
        # 生成一个形状为 (microbatch_size, d_hid) 的随机张量 y
        y = torch.randn(microbatch_size, d_hid)

        # 创建一个管道对象，使用给定的模型 mod 和输入参数 x, y
        pipe = pipeline(
            mod,
            mb_args=(x, y),
        )

        # 断言管道的阶段数等于预期的阶段数 EXPECTED_N_STAGES[ModelClass]
        assert (
            pipe.num_stages == EXPECTED_N_STAGES[ModelClass]
        ), f"nstages = {pipe.num_stages}, expect {EXPECTED_N_STAGES[ModelClass]}"

        # 计算模型直接调用的输出 ref_out
        ref_out = mod(x, y)
        # 使用管道调用模型的输出 out
        out = pipe(x, y)[0]
        # 使用 torch.testing.assert_close 断言 out 和 ref_out 的近似程度
        torch.testing.assert_close(out, ref_out)
        # 打印验证通过的信息，显示出 out 的总和和 ref_out 的总和
        print(f"equivalence test passed {torch.sum(out)} ref {torch.sum(ref_out)}")

        # 检查限定名称（qualname）
        # state_dict.keys 包括参数和持久缓冲区
        old_names = set(mod.state_dict().keys())
        new_names = set()
        # 遍历管道的每个阶段
        for idx in range(pipe.num_stages):
            # 获取当前阶段的模块 stage_mod
            stage_mod = pipe.get_stage_module(idx)
            # 获取当前阶段模块的 state_dict 键集合 stage_fqns
            stage_fqns = set(stage_mod.state_dict().keys())
            # 断言当前阶段的 state_dict 键集合是 old_names 的子集
            assert stage_fqns.issubset(old_names)
            # 更新 new_names 集合
            new_names.update(stage_fqns)

        # 如果 CHECK_FQN_SET_EQUALITY 为真，则断言 old_names 和 new_names 相等
        if CHECK_FQN_SET_EQUALITY:
            assert (
                old_names == new_names
            ), f"""
            old names {old_names}
            new names {new_names}
            """
        # 打印限定名称检查通过的信息
        print("Qualname check passed")
# 实例化一个带参数的测试类，参数为 PipeTests，用于测试管道功能
instantiate_parametrized_tests(PipeTests)

# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 运行测试函数
    run_tests()
```