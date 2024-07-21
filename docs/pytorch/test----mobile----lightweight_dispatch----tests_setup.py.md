# `.\pytorch\test\mobile\lightweight_dispatch\tests_setup.py`

```py
# 导入 functools 模块，用于创建包装函数
# 导入 os 模块，提供对操作系统功能的访问
# 导入 shutil 模块，用于高级文件操作
import functools
import os
import shutil

# 导入 sys 模块，提供对 Python 解释器的访问
import sys
# 导入 BytesIO 类，用于操作字节流数据
from io import BytesIO

# 导入 torch 模块，主要用于机器学习任务和深度学习模型
import torch
# 导入 torch.jit.mobile 模块中的函数和类
from torch.jit.mobile import _export_operator_list, _load_for_lite_interpreter

# 全局变量，存储导出的操作符集合
_OPERATORS = set()
# 全局变量，存储保存的文件名列表
_FILENAMES = []
# 全局变量，存储保存的模型列表
_MODELS = []


def save_model(cls):
    """装饰器函数，用于保存模型并导出所有操作符"""

    @functools.wraps(cls)
    def wrapper_save():
        # 将传入的类添加到模型列表中
        _MODELS.append(cls)
        # 创建模型实例
        model = cls()
        # 对模型进行脚本化
        scripted = torch.jit.script(model)
        # 将脚本化的模型保存到字节流缓冲区中
        buffer = BytesIO(scripted._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        # 使用 Lite 解释器加载模型字节流缓冲区
        mobile_module = _load_for_lite_interpreter(buffer)
        # 导出 Lite 解释器中的操作符列表
        ops = _export_operator_list(mobile_module)
        # 将导出的操作符列表添加到全局操作符集合中
        _OPERATORS.update(ops)
        # 构造保存模型的路径
        path = f"./{cls.__name__}.ptl"
        # 将路径添加到保存文件名列表中
        _FILENAMES.append(path)
        # 将脚本化的模型保存为 Lite 解释器可以读取的格式
        scripted._save_for_lite_interpreter(path)

    return wrapper_save


# 使用装饰器保存模型的示例类
@save_model
class ModelWithDTypeDeviceLayoutPinMemory(torch.nn.Module):
    def forward(self, x: int):
        # 创建一个指定形状和参数的张量
        a = torch.ones(
            size=[3, x],
            dtype=torch.int64,
            layout=torch.strided,
            device="cpu",
            pin_memory=False,
        )
        return a


# 使用装饰器保存模型的示例类
@save_model
class ModelWithTensorOptional(torch.nn.Module):
    def forward(self, index):
        # 创建一个指定形状的零张量，并对其进行赋值
        a = torch.zeros(2, 2)
        a[0][1] = 1
        a[1][0] = 2
        a[1][1] = 3
        return a[index]


# 使用装饰器保存模型的示例类
@save_model
class ModelWithScalarList(torch.nn.Module):
    def forward(self, a: int):
        # 创建一个包含指定数值的张量
        values = torch.tensor(
            [4.0, 1.0, 1.0, 16.0],
        )
        if a == 0:
            # 计算张量的梯度，使用指定的间距值
            return torch.gradient(
                values, spacing=torch.scalar_tensor(2.0, dtype=torch.float64)
            )
        elif a == 1:
            # 计算张量的梯度，使用列表形式的间距值
            return torch.gradient(values, spacing=[torch.tensor(1.0).item()])


# 使用装饰器保存模型的示例类
@save_model
class ModelWithFloatList(torch.nn.Upsample):
    def __init__(self):
        super().__init__(
            scale_factor=(2.0,),
            mode="linear",
            align_corners=False,
            recompute_scale_factor=True,
        )


# 使用装饰器保存模型的示例类
@save_model
class ModelWithListOfOptionalTensors(torch.nn.Module):
    def forward(self, index):
        # 创建一个包含指定数值的二维张量
        values = torch.tensor([[4.0, 1.0, 1.0, 16.0]])
        # 使用索引从张量中取值
        return values[torch.tensor(0), index]


# 使用装饰器保存模型的示例类
@save_model
class ModelWithArrayOfInt(torch.nn.Conv2d):
    def __init__(self):
        super().__init__(1, 2, (2, 2), stride=(1, 1), padding=(1, 1))


# 使用装饰器保存模型的示例类
# 定义一个继承自 torch.nn.Module 的模型类，用于处理包含张量的操作
class ModelWithTensors(torch.nn.Module):
    # 定义前向传播方法，接受参数 a
    def forward(self, a):
        # 创建一个与 a 同样形状的张量 b，所有元素为 1
        b = torch.ones_like(a)
        # 返回 a 和 b 相加的结果作为输出
        return a + b


# 使用装饰器 @save_model 标记的模型类，继承自 torch.nn.Module
@save_model
class ModelWithStringOptional(torch.nn.Module):
    # 定义前向传播方法，接受参数 b
    def forward(self, b):
        # 创建一个整数张量 a，数值为 3，数据类型为 torch.int64
        a = torch.tensor(3, dtype=torch.int64)
        # 创建一个形状为 [1] 的空张量 out，数据类型为 torch.float
        out = torch.empty(size=[1], dtype=torch.float)
        # 将 b 除以 a，并将结果存入 out 中
        torch.div(b, a, out=out)
        # 返回两个张量的列表：b 除以 a 的结果（截断模式下的整数除法）和 out
        return [torch.div(b, a, rounding_mode="trunc"), out]


# 使用装饰器 @save_model 标记的模型类，继承自 torch.nn.Module
@save_model
class ModelWithMultipleOps(torch.nn.Module):
    # 初始化方法，创建一个包含多个操作的序列容器 self.ops
    def __init__(self):
        super().__init__()
        self.ops = torch.nn.Sequential(
            torch.nn.ReLU(),  # ReLU 激活函数层
            torch.nn.Flatten(),  # 将输入张量展平的层
        )

    # 定义前向传播方法，接受参数 x
    def forward(self, x):
        # 修改输入张量 x 中索引为 1 的元素为 -2
        x[1] = -2
        # 将输入 x 经过 self.ops 中的层序列处理，并返回处理结果
        return self.ops(x)


# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 从命令行参数中获取指令和操作的 YAML 文件路径
    command = sys.argv[1]
    ops_yaml = sys.argv[2]
    # 创建备份文件路径，添加 .bak 后缀
    backup = ops_yaml + ".bak"
    # 如果指令为 "setup"
    if command == "setup":
        # 创建多个模型对象实例的列表 tests
        tests = [
            ModelWithDTypeDeviceLayoutPinMemory(),
            ModelWithTensorOptional(),
            ModelWithScalarList(),
            ModelWithFloatList(),
            ModelWithListOfOptionalTensors(),
            ModelWithArrayOfInt(),
            ModelWithTensors(),
            ModelWithStringOptional(),
            ModelWithMultipleOps(),
        ]
        # 复制操作的 YAML 文件到备份文件
        shutil.copyfile(ops_yaml, backup)
        # 打开操作的 YAML 文件，以追加模式写入操作列表 _OPERATORS 的内容
        with open(ops_yaml, "a") as f:
            for op in _OPERATORS:
                f.write(f"- {op}\n")
    # 如果指令为 "shutdown"
    elif command == "shutdown":
        # 遍历 _MODELS 列表中的文件名
        for file in _MODELS:
            # 如果文件存在，则删除该文件
            if os.path.isfile(file):
                os.remove(file)
        # 将备份文件移动回原操作的 YAML 文件路径
        shutil.move(backup, ops_yaml)
```