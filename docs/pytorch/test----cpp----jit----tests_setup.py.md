# `.\pytorch\test\cpp\jit\tests_setup.py`

```
# 导入必要的模块
import os  # 导入操作系统相关的功能模块
import sys  # 导入系统相关的功能模块

import torch  # 导入 PyTorch 深度学习库


# 设置类，定义了设置和关闭的接口，但没有具体实现
class Setup:
    def setup(self):
        raise NotImplementedError  # 抛出未实现异常

    def shutdown(self):
        raise NotImplementedError  # 抛出未实现异常


# 文件设置类，包含路径属性和关闭方法
class FileSetup:
    path = None  # 文件路径默认为空

    def shutdown(self):
        if os.path.exists(self.path):  # 如果文件路径存在
            os.remove(self.path)  # 移除文件
            pass  # 占位符，无实际功能


# EvalModeForLoadedModule 类继承自 FileSetup，设定了特定路径和模型保存逻辑
class EvalModeForLoadedModule(FileSetup):
    path = "dropout_model.pt"  # 模型文件路径为 dropout_model.pt

    def setup(self):
        # 内部定义了一个名为 Model 的 Torch 脚本模块
        class Model(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout(0.1)

            @torch.jit.script_method
            def forward(self, x):
                x = self.dropout(x)
                return x

        model = Model()  # 创建 Model 实例
        model = model.train()  # 设置模型为训练模式
        model.save(self.path)  # 保存模型到指定路径


# SerializationInterop 类继承自 FileSetup，设定了特定路径和数据序列化逻辑
class SerializationInterop(FileSetup):
    path = "ivalue.pt"  # 数据文件路径为 ivalue.pt

    def setup(self):
        ones = torch.ones(2, 2)  # 创建一个 2x2 的全 1 张量
        twos = torch.ones(3, 5) * 2  # 创建一个 3x5 的全 2 张量

        value = (ones, twos)  # 组成一个元组

        # 使用新的 ZIP 文件序列化功能保存数据到指定路径
        torch.save(value, self.path, _use_new_zipfile_serialization=True)


# TorchSaveError 类继承自 FileSetup，设定了特定路径和数据序列化逻辑（不使用新 ZIP 文件）
class TorchSaveError(FileSetup):
    path = "eager_value.pt"  # 数据文件路径为 eager_value.pt

    def setup(self):
        ones = torch.ones(2, 2)  # 创建一个 2x2 的全 1 张量
        twos = torch.ones(3, 5) * 2  # 创建一个 3x5 的全 2 张量

        value = (ones, twos)  # 组成一个元组

        # 使用传统的数据保存方法保存数据到指定路径
        torch.save(value, self.path, _use_new_zipfile_serialization=False)


# TorchSaveJitStream_CUDA 类继承自 FileSetup，设定了特定路径和 CUDA 模型保存逻辑
class TorchSaveJitStream_CUDA(FileSetup):
    path = "saved_stream_model.pt"  # 模型文件路径为 saved_stream_model.pt

    def setup(self):
        if not torch.cuda.is_available():  # 如果 CUDA 不可用，则返回
            return

        # 内部定义了一个名为 Model 的 Torch 模块
        class Model(torch.nn.Module):
            def forward(self):
                s = torch.cuda.Stream()  # 创建一个 CUDA 流对象
                a = torch.rand(3, 4, device="cuda")  # 在 CUDA 设备上生成随机张量 a
                b = torch.rand(3, 4, device="cuda")  # 在 CUDA 设备上生成随机张量 b

                with torch.cuda.stream(s):  # 在指定 CUDA 流上运行以下代码块
                    is_stream_s = (
                        torch.cuda.current_stream(s.device_index()).id() == s.id()
                    )  # 检查当前流是否为 s
                    c = torch.cat((a, b), 0).to("cuda")  # 在 CUDA 设备上拼接张量 a 和 b

                s.synchronize()  # 同步 CUDA 流
                return is_stream_s, a, b, c  # 返回结果

        model = Model()  # 创建 Model 实例

        # 脚本化模型并保存到指定路径
        script_model = torch.jit.script(model)
        torch.jit.save(script_model, self.path)


# 创建测试对象列表
tests = [
    EvalModeForLoadedModule(),  # 加载模型测试
    SerializationInterop(),  # 数据序列化交互测试
    TorchSaveError(),  # 数据序列化错误测试
    TorchSaveJitStream_CUDA(),  # CUDA 模型保存测试
]


# 设置函数，依次调用每个测试对象的 setup 方法
def setup():
    for test in tests:
        test.setup()


# 关闭函数，依次调用每个测试对象的 shutdown 方法
def shutdown():
    for test in tests:
        test.shutdown()


# 主程序入口
if __name__ == "__main__":
    command = sys.argv[1]  # 获取命令行参数
    if command == "setup":  # 如果命令为 setup
        setup()  # 调用 setup 函数
    elif command == "shutdown":  # 如果命令为 shutdown
        shutdown()  # 调用 shutdown 函数
```