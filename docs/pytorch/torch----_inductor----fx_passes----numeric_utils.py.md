# `.\pytorch\torch\_inductor\fx_passes\numeric_utils.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和库
import gc  # 垃圾回收模块
import logging  # 日志记录模块
import os  # 系统操作模块
import random  # 随机数生成模块
import traceback  # 异常追踪模块

import numpy  # 数值计算库

import torch  # PyTorch深度学习库
import torch.optim as optim  # PyTorch优化器模块

from .. import config  # 导入上级目录的配置文件

# 设置日志记录器
logger: logging.Logger = logging.getLogger(__name__)

# 设置主随机种子
MAIN_RANDOM_SEED = 1337

# 设置CUBLAS_WORKSPACE_CONFIG环境变量，用于CUDA加速库配置
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# 如果前向函数涉及非确定性操作（如某些类型的并行性或异步执行），这可能导致不同的输出。
def set_deterministic() -> None:
    """使得 torch 的随机种子确定性化."""

    torch.manual_seed(MAIN_RANDOM_SEED)  # 设置PyTorch随机种子
    random.seed(MAIN_RANDOM_SEED)  # 设置Python随机种子
    numpy.random.seed(MAIN_RANDOM_SEED)  # 设置NumPy随机种子
    torch.use_deterministic_algorithms(True)  # 使用确定性算法


def clean_memory() -> None:
    """清理内存以避免OOM（内存耗尽错误）."""
    gc.collect()  # 执行垃圾回收
    torch.cuda.empty_cache()  # 清空CUDA缓存


# 比较字典中张量的数值结果，用于检查前后梯度处理后数值是否一致
def compare_dict_tensors(dict_base, dict_control, precision):
    if len(set(dict_base.keys())) != len(set(dict_control.keys())):
        logger.warning("前后梯度处理前后找到键不匹配。")
        logger.debug("前梯度处理前的键 %s", dict_base.keys())
        logger.debug("前梯度处理后的键 %s", dict_control.keys())
        return False
    is_allclose = True
    for key in dict_base.keys():
        if key not in dict_control:
            logger.warning(
                "梯度参数名称 %s 在前后梯度处理后不存在。",
                key,
            )
        # 一些参数为 `None`，并非每个参数都有有效的 .grad 字段，因此我们跳过它们
        if dict_base[key] is None or dict_control[key] is None:
            continue
        if not torch.allclose(
            dict_base[key],
            dict_control[key],
            rtol=precision,
            atol=precision,
            equal_nan=True,
        ):
            logger.warning(
                "前后梯度处理前后找到参数值不匹配。"
            )
            logger.debug("前梯度处理前的值 %s", dict_base[key])
            logger.debug("前梯度处理后的值 %s", dict_control[key])
            is_allclose = False
    return is_allclose


# 比较元组中张量的数值结果，用于检查前后梯度处理后数值是否一致
def compare_tuple_tensors(tuple_base, tuple_control, precision):
    if len(tuple_base) != len(tuple_control):
        logger.warning(
            "前向输出长度不匹配。转换前：%s，转换后：%s",
            len(tuple_base),
            len(tuple_control),
        )
        return False
    is_allclose = True
    for i in range(len(tuple_base)):
        # 检查 tuple_base 和 tuple_control 中是否有 None，如果有则跳过当前循环
        if tuple_base[i] is None or tuple_control[i] is None:
            continue
        # 使用 torch.allclose 检查 tuple_base[i] 和 tuple_control[i] 是否在指定的精度范围内相等
        if not torch.allclose(
            tuple_base[i],
            tuple_control[i],
            rtol=precision,
            atol=precision,
            equal_nan=True,
        ):
            # 如果不相等，记录调试信息，显示 tuple_base[i] 的值
            logger.debug(
                "forward output before pre/post grad fx passes %s", tuple_base[i]
            )
            # 如果不相等，记录调试信息，显示 tuple_control[i] 的值
            logger.debug(
                "forward output after pre/post grad fx passes %s", tuple_control[i]
            )
            # 设置标志为 False，表示不是所有的 tuple_base[i] 和 tuple_control[i] 都在指定精度范围内相等
            is_allclose = False
    # 返回最终的比较结果，是否所有的 tuple_base 和 tuple_control 都在指定精度范围内相等
    return is_allclose
# 比较两个模型的参数，并返回比较结果
def compare_parameters(model_base, model_control, precision):
    return compare_dict_tensors(
        dict(model_base.named_parameters()),  # 提取基准模型的参数并转换为字典形式
        dict(model_control.named_parameters()),  # 提取对照模型的参数并转换为字典形式
        precision,  # 比较精度阈值
    )


# 比较两个模型的前向输出，并返回比较结果
def compare_forward_output(pred_base, pred_control, precision):
    return compare_tuple_tensors(
        pred_base,  # 基准模型的前向输出
        pred_control,  # 对照模型的前向输出
        precision,  # 比较精度阈值
    )


# 比较两个模型的梯度，并返回比较结果
def compare_gradients(model_base, model_control, precision):
    # 提取基准模型的梯度信息，并转换为字典形式
    grad_base = {key: param.grad for key, param in model_base.named_parameters()}
    # 提取对照模型的梯度信息，并转换为字典形式
    grad_pt2 = {key: param.grad for key, param in model_control.named_parameters()}
    return compare_dict_tensors(
        grad_base,  # 基准模型的梯度字典
        grad_pt2,  # 对照模型的梯度字典
        precision,  # 比较精度阈值
    )


# 运行模型，并执行指定次数的迭代
def run_model(
    model_base, model_control, model_input, num_iterations=10, precision=1e-4
):
    clean_memory()  # 清理内存
    # 对于指定次数的迭代循环
    for i in range(num_iterations):
        # 记录当前迭代开始的日志信息
        logger.info("start %s iteration", i)
        # 设置模型的确定性行为
        set_deterministic()
        # 使用基础模型对给定输入进行预测
        pred_base = model_base(*model_input)
        # 再次设置模型的确定性行为
        set_deterministic()
        # 使用控制模型对给定输入进行预测
        pred_control = model_control(*model_input)

        # 比较基础模型和控制模型的参数
        res = compare_parameters(model_base, model_control, precision)
        # 记录参数比较的结果信息
        logger.info("compare parameters. Numerical result : %s", res)

        # 比较基础模型和控制模型的前向输出结果
        res = compare_forward_output(pred_base, pred_control, precision)
        # 记录损失/预测比较的结果信息
        logger.info("compare loss/predict. Numerical result : %s", res)
        
        # 尝试对基础模型和控制模型的第一个预测结果进行反向传播梯度计算
        # 注意：张量可能没有 grad_fn
        try:
            _ = pred_base[0].sum().backward(retain_graph=True)
            _ = pred_control[0].sum().backward(retain_graph=True)
            # 比较基础模型和控制模型的参数梯度
            res = compare_gradients(model_base, model_control, precision)
            # 记录参数梯度比较的结果信息
            logger.info("compare param grad. Numerical result : %s", res)
        except Exception:
            # 如果出现异常，记录异常信息并打印堆栈追踪
            logger.exception("Exception when comparing gradients")
            traceback.print_exc()

        # 如果配置要求进行数值检查
        if config.fx_passes_numeric_check["requires_optimizer"]:
            try:
                # 创建基础模型的 SGD 优化器
                optimizer_base = optim.SGD(
                    [param for name, param in model_base.named_parameters()], lr=0.01
                )
                # 执行基础模型的优化步骤
                optimizer_base.step()

                # 创建控制模型的 SGD 优化器
                optimizer_control = optim.SGD(
                    [param for name, param in model_control.named_parameters()], lr=0.01
                )
                # 执行控制模型的优化步骤
                optimizer_control.step()

                # 再次比较基础模型和控制模型的参数
                res = compare_parameters(model_base, model_control, precision)
                # 记录添加优化器后的参数比较结果信息
                logger.info(
                    "compare parameters with optimizer added. Numerical result : %s",
                    res,
                )
            except Exception as e:
                # 如果出现异常，记录异常信息并打印堆栈追踪
                logger.exception(
                    "Exception when optimizer is added to check parameter names"
                )
                traceback.print_exc()
        else:
            # 如果配置不要求使用优化器进行检查，记录警告信息
            logger.warning(
                "no parameter with optimizer to compare with length %s before transformation"
                " and the length %s after transformation",
                len(dict(model_base.named_parameters())),
                len(dict(model_control.named_parameters())),
            )
# 定义一个函数，用于执行数值检查，确认是否启用
def numeric_check_if_enabled(
    gm_before_fx_passes,  # 输入参数：在执行前的图模块
    gm_after_fx_passes,   # 输入参数：在执行后的图模块
    example_inputs,       # 输入参数：示例输入数据
    num_iterations,       # 输入参数：迭代次数
    precision,            # 输入参数：精度
):
    # 尝试设置自动检测异常的上下文，用于运行模型
    try:
        with torch.autograd.set_detect_anomaly(True):
            # 调用运行模型的函数，传入相关参数
            run_model(
                gm_before_fx_passes,  # 在执行前的图模块
                gm_after_fx_passes,   # 在执行后的图模块
                example_inputs,       # 示例输入数据
                num_iterations=num_iterations,  # 指定的迭代次数
                precision=precision,   # 指定的精度
            )
    except Exception as e:
        # 如果发生异常，记录警告信息，并打印异常的堆栈信息
        logger.warning(
            "Runtime numeric check failed in pre grad fx passes with error: %s", e
        )
        traceback.print_exc()
```