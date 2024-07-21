# `.\pytorch\test\jit\test_graph_rewrite_passes.py`

```py
# 导入PyTorch相关库
import torch
import torch._C
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase

# 测试类继承自JitTestCase，用于测试图重写的通道
class TestGraphRewritePasses(JitTestCase):
    
    # 测试线性融合函数
    def test_fuse_linear(self):
        
        # 定义一个模块用于仿真线性函数
        class FunctionalLinear(torch.nn.Module):
            def __init__(self, weight, bias):
                super().__init__()
                self.weight = weight
                self.bias = bias
            
            # 定义前向传播函数
            def forward(self, x):
                # 计算输入和权重的矩阵乘积
                res = torch.matmul(x, self.weight.t())
                # 如果有偏置，则加上偏置
                if self.bias is not None:
                    res.add_(self.bias)
                return res
        
        # 创建随机输入、权重和偏置
        x1 = torch.rand(3)
        w1 = torch.rand(5, 3)
        b1 = torch.rand(5)
        
        # 对于有偏置和无偏置的两种情况进行测试
        for has_bias in [True, False]:
            bias = b1 if has_bias else None
            # 使用torch.jit.trace跟踪模型
            model = torch.jit.trace(FunctionalLinear(w1, bias), [x1])
            
            # 遍历模型的图中的每个节点
            for node in model.graph.nodes():
                if node.kind() == "aten::matmul":
                    # 获取第一个节点的源范围
                    source_range_1 = node.sourceRange()
            
            # 执行线性融合的JIT通道
            torch._C._jit_pass_fuse_linear(model.graph)
            
            # 再次遍历模型的图中的每个节点
            for node in model.graph.nodes():
                if node.kind() == "aten::linear":
                    # 获取第二个节点的源范围
                    source_range_2 = node.sourceRange()
            
            # 使用FileCheck检查图中是否存在"aten::linear"节点
            FileCheck().check("aten::linear").run(model.graph)
            
            # 使用FileCheck检查图中是否不存在特定节点类型
            check_not = ["aten::matmul", "aten::addmm", "aten::add_", "aten::t("]
            for cn in check_not:
                FileCheck().check_not(cn).run(model.graph)
            
            # 断言两个源范围是否相等
            self.assertTrue(source_range_1 == source_range_2)
            
            # 确保模型可以成功运行
            model(x1)
        
        # 定义用于检查矩阵乘积的模块
        class Matmul(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight
            
            # 定义前向传播函数
            def forward(self, x):
                return torch.matmul(x, self.weight)
        
        # 创建随机输入和权重
        x = torch.rand(5, 6, 5)
        w = torch.rand(5, 5, 100)
        
        # 使用torch.jit.trace跟踪模型
        model = torch.jit.trace(Matmul(w), [x])
        
        # 执行线性融合的JIT通道
        torch._C._jit_pass_fuse_linear(model.graph)
        
        # 使用FileCheck检查图中是否存在"aten::matmul"节点
        FileCheck().check("aten::matmul").run(model.graph)
        
        # 使用FileCheck检查图中是否不存在"aten::linear"节点
        FileCheck().check_not("aten::linear").run(model.graph)
        
        # 确保模型可以成功运行
        model(x)
```