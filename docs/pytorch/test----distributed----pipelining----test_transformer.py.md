# `.\pytorch\test\distributed\pipelining\test_transformer.py`

```py
# 导入PyTorch库
import torch
# 从torch.distributed.pipelining模块中导入pipeline和SplitPoint
from torch.distributed.pipelining import pipeline, SplitPoint
# 从torch.testing._internal.common_utils导入run_tests和TestCase类
from torch.testing._internal.common_utils import run_tests, TestCase

# 定义MLPModule的隐藏层维度
d_hid = 16
# 定义TransformerLike模型的层数
n_layers = 8
# 定义微批次大小
microbatch_size = 4

# 定义MLPModule类，继承自torch.nn.Module
class MLPModule(torch.nn.Module):
    # 构造函数，初始化隐藏层维度
    def __init__(self, d_hid):
        super().__init__()
        # 定义第一个线性层，输入和输出维度都为d_hid
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        # 定义ReLU激活函数
        self.relu = torch.nn.ReLU()
        # 定义第二个线性层，输入和输出维度都为d_hid
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    # 前向传播函数
    def forward(self, x):
        # 输入x经过第一个线性层和ReLU激活函数
        x = self.net1(x)
        x = self.relu(x)
        # 经过第二个线性层
        x = self.net2(x)
        return x

# 定义TransformerLike类，继承自torch.nn.Module
class TransformerLike(torch.nn.Module):
    # 构造函数
    def __init__(self) -> None:
        super().__init__()
        # 创建包含n_layers个MLPModule实例的序列
        self.layers = torch.nn.Sequential(*[MLPModule(d_hid) for _ in range(n_layers)])

    # 前向传播函数，输入x为torch.Tensor类型，返回值也是torch.Tensor类型
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x经过self.layers中的每个MLPModule实例
        return self.layers(x)

# 定义TransformerTests类，继承自TestCase类
class TransformerTests(TestCase):
    # 定义测试函数test_ir
    def test_ir(self):
        # 创建TransformerLike模型实例
        transformer = TransformerLike()
        # 创建输入张量x，大小为(microbatch_size, d_hid)，内容为随机数
        x = torch.randn(microbatch_size, d_hid)

        # 将pipeline分成2个阶段
        num_stages = 2
        # 定义分阶段的规格，将self.layers中的层分成num_stages个阶段
        split_spec = {f"layers.{n_layers // num_stages}": SplitPoint.BEGINNING}

        # 创建pipeline对象pipe，使用transformer模型和输入x，应用分阶段规格split_spec
        pipe = pipeline(
            transformer,
            (x,),
            split_spec=split_spec,
        )
        # 断言pipe的阶段数等于num_stages
        assert pipe.num_stages == num_stages, f"{pipe.num_stages=}, expect {num_stages}"

        # 定义获取模块层名称的辅助函数get_layers
        def get_layers(module):
            # 返回module.layers中每个子模块的名称列表
            layers = [name for name, _ in module.layers.named_children()]
            return layers

        # 收集pipe中所有阶段的层名称
        layers = []
        for stage_idx in range(pipe.num_stages):
            # 获取第stage_idx阶段的模块
            stage_mod = pipe.get_stage_module(stage_idx)
            # 将该阶段模块的层名称添加到layers列表中
            layers += get_layers(stage_mod)

        # 检查层的完整性
        orig_layers = get_layers(transformer)
        # 断言排序后的layers列表与orig_layers列表相等
        assert sorted(layers) == sorted(orig_layers), f"{layers} != {orig_layers}"
        # 输出层匹配的提示信息
        print("Layers matched!")

        # 检查等效性
        ref = transformer(x)
        # 使用pipe进行前向传播，得到输出out
        out = pipe(x)[0]
        # 使用torch.testing.assert_close检查out和ref的近似程度
        torch.testing.assert_close(out, ref)
        # 输出等效性测试通过的提示信息，显示out和ref的总和
        print(f"Equivalence test passed {torch.sum(out)} ref {torch.sum(ref)}")

# 如果当前脚本为主程序，则运行测试
if __name__ == "__main__":
    run_tests()
```