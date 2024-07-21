# `.\pytorch\test\cpp\aoti_inference\test.py`

```py
# 导入PyTorch库
import torch
# 导入AOT编译器
from torch._export import aot_compile
# 导入维度处理模块
from torch.export import Dim

# 设置随机种子，以保证可重复性
torch.manual_seed(1337)

# 定义一个神经网络模型类
class Net(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        # 初始化权重矩阵 w_pre 和 w_add，使用设备指定的设备（CPU或GPU）
        self.w_pre = torch.randn(4, 4, device=device)
        self.w_add = torch.randn(4, 4, device=device)

    def forward(self, x):
        # 计算权重矩阵 w_pre 的转置
        w_transpose = torch.transpose(self.w_pre, 0, 1)
        # 对转置后的矩阵应用ReLU激活函数
        w_relu = torch.nn.functional.relu(w_transpose)
        # 将ReLU后的矩阵与 w_add 相加，得到权重矩阵 w
        w = w_relu + self.w_add
        # 返回输入 x 与权重矩阵 w 的矩阵乘积
        return torch.matmul(x, w)

# 定义一个带有张量常数的神经网络模型类
class NetWithTensorConstants(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化权重张量 w，使用 CUDA 设备
        self.w = torch.randn(30, 1, device="cuda")

    def forward(self, x, y):
        # 计算张量 z，为 w 乘以输入张量 x 和 y 的元素积
        z = self.w * x * y
        # 返回张量 z 中指定索引的子集
        return z[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17]]

# 初始化数据字典
data = {}
# 初始化带有张量常数的数据字典
data_with_tensor_constants = {}

# 生成基本AOTI模型测试
def generate_basic_tests():
    # 遍历设备类型和运行时常量折叠选项
    for device in ["cpu", "cuda"]:
        for use_runtime_constant_folding in [True, False]:
            # 如果设备为 CPU 并且启用运行时常量折叠，跳过该情况
            if device == "cpu" and use_runtime_constant_folding:
                continue
            # 创建模型实例并移动到指定设备
            model = Net(device).to(device=device)
            # 生成随机输入张量 x
            x = torch.randn((4, 4), device=device)
            # 在无梯度下计算参考输出
            with torch.no_grad():
                ref_output = model(x)

            # 重置动态编译器状态
            torch._dynamo.reset()
            # 在无梯度下，定义动态形状和模型编译选项
            with torch.no_grad():
                dim0_x = Dim("dim0_x", min=1, max=1024)
                dynamic_shapes = {"x": {0: dim0_x}}
                model_so_path = aot_compile(
                    model,
                    (x,),
                    dynamic_shapes=dynamic_shapes,
                    options={
                        "aot_inductor.use_runtime_constant_folding": use_runtime_constant_folding
                    },
                )

            # 构建后缀以区分设备和运行时常量折叠选项
            suffix = f"{device}"
            if use_runtime_constant_folding:
                suffix += "_use_runtime_constant_folding"
            # 更新数据字典，包括模型路径、输入、输出、权重矩阵等信息
            data.update(
                {
                    f"model_so_path_{suffix}": model_so_path,
                    f"inputs_{suffix}": [x],
                    f"outputs_{suffix}": [ref_output],
                    f"w_pre_{suffix}": model.w_pre,
                    f"w_add_{suffix}": model.w_add,
                }
            )

# 生成带有额外张量的AOTI模型测试
def generate_test_with_additional_tensors():
    # 创建带有张量常数的模型实例
    model = NetWithTensorConstants()
    # 生成随机输入张量 x 和 y，使用 CUDA 设备
    x = torch.randn((30, 1), device="cuda")
    y = torch.randn((30, 1), device="cuda")
    # 在无梯度下计算参考输出
    with torch.no_grad():
        ref_output = model(x, y)

    # 重置动态编译器状态
    torch._dynamo.reset()
    # 在无梯度下，进行模型编译
    with torch.no_grad():
        model_so_path = aot_compile(model, (x, y))

    # 更新带有张量常数的数据字典，包括模型路径、输入、输出、权重张量等信息
    data_with_tensor_constants.update(
        {
            "model_so_path": model_so_path,
            "inputs": [x, y],
            "outputs": [ref_output],
            "w": model.w,
        }
    )

# 调用函数生成基本AOTI模型测试
generate_basic_tests()
# 调用函数生成带有额外张量的AOTI模型测试
generate_test_with_additional_tensors()
# 用于将张量传递给 C++ 代码的序列化器类
class Serializer(torch.nn.Module):
    # 初始化方法，接受一个数据字典作为参数
    def __init__(self, data):
        super().__init__()
        # 遍历数据字典中的每个键值对
        for key in data:
            # 使用 setattr 方法将每个键值对设置为当前对象的属性
            setattr(self, key, data[key])

# 对包含数据的 Serializer 对象进行 TorchScript 脚本化，并保存为 "data.pt" 文件
torch.jit.script(Serializer(data)).save("data.pt")
# 对包含张量常量数据的 Serializer 对象进行 TorchScript 脚本化，并保存为 "data_with_tensor_constants.pt" 文件
torch.jit.script(Serializer(data_with_tensor_constants)).save(
    "data_with_tensor_constants.pt"
)
```