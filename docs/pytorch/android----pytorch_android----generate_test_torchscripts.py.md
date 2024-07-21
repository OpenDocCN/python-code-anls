# `.\pytorch\android\pytorch_android\generate_test_torchscripts.py`

```
# 引入必要的类型注解
from typing import Dict, List, Optional, Tuple

# 引入 PyTorch 库
import torch
from torch import Tensor

# 定义输出目录常量
OUTPUT_DIR = "src/androidTest/assets/"

# 定义函数，用于将模型脚本化并保存
def scriptAndSave(module, fileName):
    # 打印分隔线
    print("-" * 80)
    # 对模型进行脚本化
    script_module = torch.jit.script(module)
    # 打印脚本化模型的计算图
    print(script_module.graph)
    # 设置输出文件名
    outputFileName = OUTPUT_DIR + fileName
    # 使用轻量级解释器模型保存完整 JIT 模型
    script_module._save_for_lite_interpreter(outputFileName)
    # 打印保存成功信息
    print("Saved to " + outputFileName)
    # 打印分隔线
    print("=" * 80)

# 定义测试类，继承自 torch.jit.ScriptModule
class Test(torch.jit.ScriptModule):
    
    # 定义前向传播方法
    @torch.jit.script_method
    def forward(self, input):
        return None
    
    # 定义返回布尔值的方法
    @torch.jit.script_method
    def eqBool(self, input: bool) -> bool:
        return input
    
    # 定义返回整数的方法
    @torch.jit.script_method
    def eqInt(self, input: int) -> int:
        return input
    
    # 定义返回浮点数的方法
    @torch.jit.script_method
    def eqFloat(self, input: float) -> float:
        return input
    
    # 定义返回字符串的方法
    @torch.jit.script_method
    def eqStr(self, input: str) -> str:
        return input
    
    # 定义返回张量的方法
    @torch.jit.script_method
    def eqTensor(self, input: Tensor) -> Tensor:
        return input
    
    # 定义返回键为字符串、值为整数的字典的方法
    @torch.jit.script_method
    def eqDictStrKeyIntValue(self, input: Dict[str, int]) -> Dict[str, int]:
        return input
    
    # 定义返回键为整数、值为整数的字典的方法
    @torch.jit.script_method
    def eqDictIntKeyIntValue(self, input: Dict[int, int]) -> Dict[int, int]:
        return input
    
    # 定义返回键为浮点数、值为整数的字典的方法
    @torch.jit.script_method
    def eqDictFloatKeyIntValue(self, input: Dict[float, int]) -> Dict[float, int]:
        return input
    
    # 定义返回整数列表及其总和的元组的方法
    @torch.jit.script_method
    def listIntSumReturnTuple(self, input: List[int]) -> Tuple[List[int], int]:
        sum = 0
        for x in input:
            sum += x
        return (input, sum)
    
    # 定义返回布尔值列表的逻辑与结果的方法
    @torch.jit.script_method
    def listBoolConjunction(self, input: List[bool]) -> bool:
        res = True
        for x in input:
            res = res and x
        return res
    
    # 定义返回布尔值列表的逻辑或结果的方法
    @torch.jit.script_method
    def listBoolDisjunction(self, input: List[bool]) -> bool:
        res = False
        for x in input:
            res = res or x
        return res
    
    # 定义返回三个整数元组及其总和的元组的方法
    @torch.jit.script_method
    def tupleIntSumReturnTuple(
        self, input: Tuple[int, int, int]
    ) -> Tuple[Tuple[int, int, int], int]:
        sum = 0
        for x in input:
            sum += x
        return (input, sum)
    
    # 定义返回可选整数是否为 None 的方法
    @torch.jit.script_method
    def optionalIntIsNone(self, input: Optional[int]) -> bool:
        return input is None
    
    # 定义返回整数是否等于 0 或者为 None 的方法
    @torch.jit.script_method
    def intEq0None(self, input: int) -> Optional[int]:
        if input == 0:
            return None
        return input
    
    # 定义返回输入字符串重复三次的方法
    @torch.jit.script_method
    def str3Concat(self, input: str) -> str:
        return input + input + input
    
    # 定义返回包含输入项的新张量的方法
    @torch.jit.script_method
    def newEmptyShapeWithItem(self, input):
        return torch.tensor([int(input.item())])[0]
    
    # 定义返回张量列表的方法
    @torch.jit.script_method
    def testAliasWithOffset(self) -> List[Tensor]:
        x = torch.tensor([100, 200])
        a = [x[0], x[1]]
        return a
    @torch.jit.script_method
    def testNonContiguous(self):
        # 创建一个包含元素 [100, 200, 300] 的张量 x，取步长为2的切片
        x = torch.tensor([100, 200, 300])[::2]
        # 断言张量 x 不是连续的
        assert not x.is_contiguous()
        # 断言张量 x 的第一个元素为 100
        assert x[0] == 100
        # 断言张量 x 的第二个元素为 300
        assert x[1] == 300
        # 返回张量 x
        return x

    @torch.jit.script_method
    def conv2d(self, x: Tensor, w: Tensor, toChannelsLast: bool) -> Tensor:
        # 对输入张量 x 和权重张量 w 执行二维卷积操作
        r = torch.nn.functional.conv2d(x, w)
        # 如果指定要转换为 channels_last 格式
        if toChannelsLast:
            # 将结果张量 r 转换为 channels_last 内存格式
            r = r.contiguous(memory_format=torch.channels_last)
        else:
            # 否则，保持结果张量 r 的连续性
            r = r.contiguous()
        # 返回转换后的结果张量 r
        return r

    @torch.jit.script_method
    def conv3d(self, x: Tensor, w: Tensor, toChannelsLast: bool) -> Tensor:
        # 对输入张量 x 和权重张量 w 执行三维卷积操作
        r = torch.nn.functional.conv3d(x, w)
        # 如果指定要转换为 channels_last_3d 格式
        if toChannelsLast:
            # 将结果张量 r 转换为 channels_last_3d 内存格式
            r = r.contiguous(memory_format=torch.channels_last_3d)
        else:
            # 否则，保持结果张量 r 的连续性
            r = r.contiguous()
        # 返回转换后的结果张量 r
        return r

    @torch.jit.script_method
    def contiguous(self, x: Tensor) -> Tensor:
        # 返回输入张量 x 的连续版本
        return x.contiguous()

    @torch.jit.script_method
    def contiguousChannelsLast(self, x: Tensor) -> Tensor:
        # 返回将输入张量 x 转换为 channels_last 内存格式的连续版本
        return x.contiguous(memory_format=torch.channels_last)

    @torch.jit.script_method
    def contiguousChannelsLast3d(self, x: Tensor) -> Tensor:
        # 返回将输入张量 x 转换为 channels_last_3d 内存格式的连续版本
        return x.contiguous(memory_format=torch.channels_last_3d)
# 调用名为 scriptAndSave 的函数，传入 Test 类的实例和字符串 "test.pt" 作为参数
scriptAndSave(Test(), "test.pt")
```