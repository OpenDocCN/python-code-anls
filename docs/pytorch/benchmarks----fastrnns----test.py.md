# `.\pytorch\benchmarks\fastrnns\test.py`

```
import argparse  # 导入 argparse 模块，用于处理命令行参数

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 神经网络模块

from .factory import pytorch_lstm_creator, varlen_pytorch_lstm_creator  # 从本地工厂模块中导入两个函数
from .runner import get_nn_runners  # 从本地运行器模块中导入函数

def barf():  # 定义函数 barf，用于调试时设置断点
    import pdb  # 导入 Python 调试器模块

    pdb.set_trace()  # 设置调试断点

def assertEqual(tensor, expected, threshold=0.001):  # 定义函数 assertEqual，用于比较张量和期望值
    if isinstance(tensor, (list, tuple)):  # 如果 tensor 是列表或元组
        for t, e in zip(tensor, expected):  # 遍历 tensor 和 expected
            assertEqual(t, e)  # 递归调用 assertEqual 函数
    else:
        if (tensor - expected).abs().max() > threshold:  # 如果张量和期望值之间的最大差异大于阈值
            barf()  # 调用 barf 函数进行调试

def filter_requires_grad(tensors):  # 定义函数 filter_requires_grad，用于筛选出需要梯度的张量
    return [t for t in tensors if t.requires_grad]  # 返回所有需要梯度的张量列表

def test_rnns(  # 定义函数 test_rnns，用于测试循环神经网络模型
    experim_creator,  # 实验模型创建器
    control_creator,  # 控制模型创建器
    check_grad=True,  # 是否检查梯度，默认为 True
    verbose=False,  # 是否详细输出，默认为 False
    seqLength=100,  # 序列长度，默认为 100
    numLayers=1,  # 神经网络层数，默认为 1
    inputSize=512,  # 输入尺寸，默认为 512
    hiddenSize=512,  # 隐藏层尺寸，默认为 512
    miniBatch=64,  # 小批量大小，默认为 64
    device="cuda",  # 设备类型，默认为 "cuda"
    seed=17,  # 随机种子，默认为 17
):
    creator_args = dict(  # 创建参数字典 creator_args
        seqLength=seqLength,  # 序列长度
        numLayers=numLayers,  # 神经网络层数
        inputSize=inputSize,  # 输入尺寸
        hiddenSize=hiddenSize,  # 隐藏层尺寸
        miniBatch=miniBatch,  # 小批量大小
        device=device,  # 设备类型
        seed=seed,  # 随机种子
    )

    print("Setting up...")  # 打印设置中...

    control = control_creator(**creator_args)  # 使用控制模型创建器创建控制模型
    experim = experim_creator(**creator_args)  # 使用实验模型创建器创建实验模型

    # Precondition
    assertEqual(experim.inputs, control.inputs)  # 断言实验模型的输入与控制模型的输入相等
    assertEqual(experim.params, control.params)  # 断言实验模型的参数与控制模型的参数相等

    print("Checking outputs...")  # 打印检查输出...

    control_outputs = control.forward(*control.inputs)  # 获取控制模型的前向传播输出
    experim_outputs = experim.forward(*experim.inputs)  # 获取实验模型的前向传播输出
    assertEqual(experim_outputs, control_outputs)  # 断言实验模型的输出与控制模型的输出相等

    print("Checking grads...")  # 打印检查梯度...

    assert control.backward_setup is not None  # 断言控制模型的反向传播设置不为 None
    assert experim.backward_setup is not None  # 断言实验模型的反向传播设置不为 None
    assert control.backward is not None  # 断言控制模型的反向传播函数不为 None
    assert experim.backward is not None  # 断言实验模型的反向传播函数不为 None

    control_backward_inputs = control.backward_setup(control_outputs, seed)  # 获取控制模型的反向传播输入
    experim_backward_inputs = experim.backward_setup(experim_outputs, seed)  # 获取实验模型的反向传播输入

    control.backward(*control_backward_inputs)  # 对控制模型进行反向传播
    experim.backward(*experim_backward_inputs)  # 对实验模型进行反向传播

    control_grads = [p.grad for p in control.params]  # 获取控制模型的梯度
    experim_grads = [p.grad for p in experim.params]  # 获取实验模型的梯度
    assertEqual(experim_grads, control_grads)  # 断言实验模型的梯度与控制模型的梯度相等

    if verbose:  # 如果 verbose 为 True
        print(experim.forward.graph_for(*experim.inputs))  # 打印实验模型前向传播的计算图
    print("")  # 打印空行

def test_vl_py(**test_args):  # 定义函数 test_vl_py，用于测试 varlen_pytorch_lstm_creator
    # XXX: This compares vl_py with vl_lstm.
    # It's done this way because those two don't give the same outputs so
    # the result isn't an apples-to-apples comparison right now.
    control_creator = varlen_pytorch_lstm_creator  # 设置控制模型创建器为 varlen_pytorch_lstm_creator
    name, experim_creator, context = get_nn_runners("vl_py")[0]  # 获取名为 "vl_py" 的神经网络运行器的名称、实验模型创建器和上下文
    # 进入上下文管理器，执行以下代码块
    with context():
        # 打印正在测试的名称
        print(f"testing {name}...")
        # 定义需要传递给创建器函数的键列表
        creator_keys = [
            "seqLength",    # 序列长度
            "numLayers",    # 层数
            "inputSize",    # 输入大小
            "hiddenSize",   # 隐藏层大小
            "miniBatch",    # 小批量大小
            "device",       # 设备
            "seed",         # 种子值
        ]
        # 根据 creator_keys 从 test_args 中创建参数字典
        creator_args = {key: test_args[key] for key in creator_keys}

        # 打印设置过程
        print("Setting up...")
        # 使用 creator_args 调用 control_creator 函数创建控制对象
        control = control_creator(**creator_args)
        # 使用 creator_args 调用 experim_creator 函数创建实验对象
        experim = experim_creator(**creator_args)

        # 预置条件检查
        assertEqual(experim.inputs, control.inputs[:2])  # 检查实验对象的输入
        assertEqual(experim.params, control.params)      # 检查实验对象的参数与控制对象相同

        # 打印检查输出过程
        print("Checking outputs...")
        # 调用控制对象的 forward 方法获取控制输出及隐藏状态
        control_out, control_hiddens = control.forward(*control.inputs)
        control_hx, control_cx = control_hiddens
        # 调用实验对象的 forward 方法获取实验输出及隐藏状态
        experim_out, experim_hiddens = experim.forward(*experim.inputs)
        experim_hx, experim_cx = experim_hiddens

        # 对实验对象输出进行填充以适应神经网络要求
        experim_padded = nn.utils.rnn.pad_sequence(experim_out).squeeze(-2)
        # 断言实验对象填充后的输出与控制对象的输出相等
        assertEqual(experim_padded, control_out)
        # 断言实验对象的隐藏状态与控制对象的隐藏状态连接后相等
        assertEqual(torch.cat(experim_hx, dim=1), control_hx)
        assertEqual(torch.cat(experim_cx, dim=1), control_cx)

        # 打印检查梯度过程
        print("Checking grads...")
        # 断言控制对象及实验对象的反向传播设置不为 None
        assert control.backward_setup is not None
        assert experim.backward_setup is not None
        # 断言控制对象及实验对象的反向传播函数不为 None
        assert control.backward is not None
        assert experim.backward is not None
        # 调用控制对象的反向传播设置函数，获取反向传播输入
        control_backward_inputs = control.backward_setup(
            (control_out, control_hiddens), test_args["seed"]
        )
        # 调用实验对象的反向传播设置函数，获取反向传播输入
        experim_backward_inputs = experim.backward_setup(
            (experim_out, experim_hiddens), test_args["seed"]
        )

        # 调用控制对象的反向传播函数
        control.backward(*control_backward_inputs)
        # 调用实验对象的反向传播函数
        experim.backward(*experim_backward_inputs)

        # 获取控制对象及实验对象的梯度列表
        control_grads = [p.grad for p in control.params]
        experim_grads = [p.grad for p in experim.params]
        # 断言实验对象的梯度与控制对象的梯度相等
        assertEqual(experim_grads, control_grads)

        # 如果 verbose 为 True，则打印实验对象的前向图
        if test_args["verbose"]:
            print(experim.forward.graph_for(*experim.inputs))
        # 打印空行
        print("")
if __name__ == "__main__":
    # 创建参数解析器对象，用于解析命令行参数并生成帮助信息
    parser = argparse.ArgumentParser(description="Test lstm correctness")

    # 添加命令行参数
    parser.add_argument("--seqLength", default="100", type=int)  # 序列长度，默认为100，整数类型
    parser.add_argument("--numLayers", default="1", type=int)  # LSTM 层的数量，默认为1，整数类型
    parser.add_argument("--inputSize", default="512", type=int)  # 输入大小，默认为512，整数类型
    parser.add_argument("--hiddenSize", default="512", type=int)  # 隐状态大小，默认为512，整数类型
    parser.add_argument("--miniBatch", default="64", type=int)  # 小批量大小，默认为64，整数类型
    parser.add_argument("--device", default="cuda", type=str)  # 设备选择，默认为"cuda"，字符串类型
    parser.add_argument("--check-grad", "--check_grad", default="True", type=bool)  # 是否检查梯度，默认为True，布尔类型
    parser.add_argument("--variable-lstms", "--variable_lstms", action="store_true")  # 是否使用可变长度的 LSTM，布尔类型
    parser.add_argument("--seed", default="17", type=int)  # 随机种子，默认为17，整数类型
    parser.add_argument("--verbose", action="store_true")  # 是否打印详细信息，布尔类型
    parser.add_argument("--rnns", nargs="*", help="What to run. jit_premul, jit, etc")  # 要运行的 RNN 类型列表，可变参数

    # 解析命令行参数
    args = parser.parse_args()

    # 如果未指定 --rnns 参数，则默认为 ["jit_premul", "jit"]
    if args.rnns is None:
        args.rnns = ["jit_premul", "jit"]

    # 打印解析后的参数信息
    print(args)

    # 如果设备选项中包含 "cuda"，则断言 CUDA 可用
    if "cuda" in args.device:
        assert torch.cuda.is_available()

    # 获取 RNN 运行器列表
    rnn_runners = get_nn_runners(*args.rnns)

    # 是否测试可变长度的 LSTM 模型
    should_test_varlen_lstms = args.variable_lstms

    # 复制参数字典并移除 "rnns" 和 "variable_lstms" 键
    test_args = vars(args)
    del test_args["rnns"]
    del test_args["variable_lstms"]

    # 如果需要测试可变长度的 LSTM 模型，则调用 test_vl_py 函数
    if should_test_varlen_lstms:
        test_vl_py(**test_args)

    # 遍历 RNN 运行器列表，并执行测试
    for name, creator, context in rnn_runners:
        # 在上下文环境中执行测试
        with context():
            print(f"testing {name}...")
            # 调用 test_rnns 函数，传入相应参数
            test_rnns(creator, pytorch_lstm_creator, **test_args)
```