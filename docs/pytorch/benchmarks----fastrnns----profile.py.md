# `.\pytorch\benchmarks\fastrnns\profile.py`

```
import argparse
import datetime
import subprocess
import sys
import time

import torch

from .runner import get_nn_runners


def run_rnn(
    name,
    rnn_creator,
    nloops=5,
    seqLength=100,
    numLayers=1,
    inputSize=512,
    hiddenSize=512,
    miniBatch=64,
    device="cuda",
    seed=None,
):
    def run_iter(modeldef):
        # 执行神经网络模型的前向传播
        forward_output = modeldef.forward(*modeldef.inputs)

        # 计算损失并进行反向传播
        if modeldef.backward_setup is not None:
            backward_input = modeldef.backward_setup(forward_output)
        else:
            backward_input = forward_output
        if modeldef.backward is not None:
            modeldef.backward(*backward_input)

        # 更新模型参数
        if modeldef.backward is not None:
            with torch.no_grad():
                for param in modeldef.params:
                    param.grad.zero_()
        torch.cuda.synchronize()

    assert device == "cuda"
    creator_args = dict(
        seqLength=seqLength,
        numLayers=numLayers,
        inputSize=inputSize,
        hiddenSize=hiddenSize,
        miniBatch=miniBatch,
        device=device,
        seed=seed,
    )
    modeldef = rnn_creator(**creator_args)

    [run_iter(modeldef) for _ in range(nloops)]


def profile(
    rnns,
    sleep_between_seconds=1,
    nloops=5,
    internal_run=True,  # 未使用，可以移除，TODO
    seqLength=100,
    numLayers=1,
    inputSize=512,
    hiddenSize=512,
    miniBatch=64,
    device="cuda",
    seed=None,
):
    params = dict(
        seqLength=seqLength,
        numLayers=numLayers,
        inputSize=inputSize,
        hiddenSize=hiddenSize,
        miniBatch=miniBatch,
        device=device,
        seed=seed,
    )
    for name, creator, context in get_nn_runners(*rnns):
        with context():
            run_rnn(name, creator, nloops, **params)
            time.sleep(sleep_between_seconds)


def system(command):
    """执行系统命令，并返回执行结果的元组(return-code, stdout, stderr)"""
    print(f"[system] {command}")
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, err = p.communicate()
    rc = p.returncode
    output = output.decode("ascii")
    err = err.decode("ascii")
    return rc, output, err


def describe_sizes(**sizes):
    # 根据输入的尺寸参数生成描述字符串，格式为s{seqLength}-l{numLayers}-i{inputSize}-h{hiddenSize}-b{miniBatch}
    return "s{}-l{}-i{}-h{}-b{}".format(
        sizes["seqLength"],
        sizes["numLayers"],
        sizes["inputSize"],
        sizes["hiddenSize"],
        sizes["miniBatch"],
    )


OUTPUT_DIR = "~/profout/"


def nvprof_output_filename(rnns, **params):
    # 根据给定的神经网络和参数生成 NVProf 输出文件名
    rnn_tag = "-".join(rnns)
    size_tag = describe_sizes(**params)
    date_tag = datetime.datetime.now().strftime("%m%d%y-%H%M")
    return f"{OUTPUT_DIR}prof_{rnn_tag}_{size_tag}_{date_tag}.nvvp"


def nvprof(cmd, outpath):
    # 执行 NVProf 分析命令，并将结果输出到指定路径
    return system(f"nvprof -o {outpath} {cmd}")


def full_profile(rnns, **args):
    profile_args = []
    # 遍历参数字典args，将每个键值对转换为格式为"--key=value"的字符串，并添加到profile_args列表中
    for k, v in args.items():
        profile_args.append(f"--{k}={v}")

    # 将列表rnns中的元素用空格连接成一个字符串，并添加到profile_args列表中作为一个参数选项
    profile_args.append(f"--rnns {' '.join(rnns)}")

    # 添加一个固定的内部运行参数选项到profile_args列表中
    profile_args.append("--internal-run")

    # 根据给定的rnns和其他参数args生成nvprof输出文件的路径，并赋值给outpath变量
    outpath = nvprof_output_filename(rnns, **args)

    # 构建命令字符串cmd，调用fastrnns.profile模块进行性能分析，使用profile_args作为命令的参数
    cmd = f"{sys.executable} -m fastrnns.profile {' '.join(profile_args)}"

    # 使用nvprof命令执行cmd命令，并将返回的状态码(rc)、标准输出(stdout)和标准错误(stderr)分别赋值给变量
    rc, stdout, stderr = nvprof(cmd, outpath)

    # 如果执行的返回码rc不等于0，抛出运行时异常，包含stderr和stdout的内容
    if rc != 0:
        raise RuntimeError(f"stderr: {stderr}\nstdout: {stdout}")
# 如果脚本作为主程序运行，则执行以下代码块
if __name__ == "__main__":
    # 创建一个参数解析器对象，描述为“Profile RNNs”
    parser = argparse.ArgumentParser(description="Profile RNNs")

    # 添加整数类型的命令行参数：序列长度，默认为100
    parser.add_argument("--seqLength", default="100", type=int)
    # 添加整数类型的命令行参数：RNN 层数，默认为1
    parser.add_argument("--numLayers", default="1", type=int)
    # 添加整数类型的命令行参数：输入大小，默认为512
    parser.add_argument("--inputSize", default="512", type=int)
    # 添加整数类型的命令行参数：隐藏层大小，默认为512
    parser.add_argument("--hiddenSize", default="512", type=int)
    # 添加整数类型的命令行参数：小批量大小，默认为64
    parser.add_argument("--miniBatch", default="64", type=int)
    # 添加整数类型的命令行参数：运行间隔秒数，默认为1
    parser.add_argument(
        "--sleep-between-seconds", "--sleep_between_seconds", default="1", type=int
    )
    # 添加整数类型的命令行参数：循环次数，默认为5
    parser.add_argument("--nloops", default="5", type=int)

    # 添加一个可变长度的命令行参数：指定要运行的 RNN 类型，如 cudnn、aten、jit 等
    parser.add_argument("--rnns", nargs="*", help="What to run. cudnn, aten, jit, etc")

    # 添加一个布尔类型的命令行参数：内部运行标志，若设置则表示要直接运行 RNN
    # 若未设置，则会使用 nvprof 与 internal_run=True 来运行
    parser.add_argument(
        "--internal-run",
        "--internal_run",
        default=False,
        action="store_true",
        help="Don't use this",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 如果未指定要运行的 RNN 类型，则默认为 cudnn、aten、jit
    if args.rnns is None:
        args.rnns = ["cudnn", "aten", "jit"]

    # 打印解析后的命令行参数信息
    print(args)

    # 如果设置了 --internal-run 标志
    if args.internal_run:
        # 调用 profile 函数，传递命令行参数的字典形式作为参数
        profile(**vars(args))
    else:
        # 否则调用 full_profile 函数，同样传递命令行参数的字典形式作为参数
        full_profile(**vars(args))
```