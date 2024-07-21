# `.\pytorch\torchgen\static_runtime\config.py`

```
# 导入必要的模块或类
from __future__ import annotations

from torchgen.model import NativeFunctionsGroup, NativeFunctionsViewGroup

# 定义一个函数，根据给定的 NativeFunctionsGroup 或 NativeFunctionsViewGroup 对象返回一个字符串
def func_name_base_str(g: NativeFunctionsGroup | NativeFunctionsViewGroup) -> str:
    # 如果对象是 NativeFunctionsGroup 类型，则返回其函数名的基本字符串表示
    if isinstance(g, NativeFunctionsGroup):
        return str(g.functional.func.name.name.base)
    else:
        # 否则返回视图对象的根名称的字符串表示
        return str(g.view.root_name)

# 预定义一组手写的操作名称，使用 frozenset 以提高效率和不可变性
is_hand_written_ops_ = frozenset(
    (
        "abs", "add", "addmm", "all", "any", "argmin", "bmm", "clamp", "clamp_min",
        "cumsum", "div", "fmod", "index_select", "leaky_relu", "linear", "log", 
        "matmul", "mul", "narrow_copy", "nonzero", "pow", "remainder", "sigmoid", 
        "sign", "sub", "tanh", "detach", "expand_as", "flatten", "narrow", 
        "reshape_as", "select", "slice", "softmax", "split", "squeeze", 
        "transpose", "view", "where",
    )
)

# 定义一个函数，检查给定的 NativeFunctionsGroup 或 NativeFunctionsViewGroup 是否为手写操作
def is_hand_written(g: NativeFunctionsGroup | NativeFunctionsViewGroup) -> bool:
    # 获取函数名的基本字符串表示
    name_base = func_name_base_str(g)
    # 检查基本字符串是否在预定义的手写操作集合中
    return name_base in is_hand_written_ops_

# 定义一个函数，用于根据操作名和索引覆盖测试值
def override_test_values(arg_map: dict[str, str], op_name: str, index: int) -> None:
    # 确保索引值为 0 或 1
    assert index == 0 or index == 1
    # 根据操作名设置不同的测试值
    if op_name == "addr":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6})"
            arg_map["vec1"] = "at::rand({6})"
            arg_map["vec2"] = "at::rand({6})"
        else:
            arg_map["self"] = "at::rand({22, 22})"
            arg_map["vec1"] = "at::rand({22})"
            arg_map["vec2"] = "at::rand({22})"
        return
    elif op_name == "mv":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6})"
            arg_map["vec"] = "at::rand({6})"
        else:
            arg_map["self"] = "at::rand({22, 22})"
            arg_map["vec"] = "at::rand({22})"
        return
    elif op_name == "addbmm":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6})"
        else:
            arg_map["self"] = "at::rand({22, 22})"
        return
    elif op_name == "cross":
        if index == 0:
            arg_map["self"] = "at::rand({3, 3, 3})"
            arg_map["other"] = "at::rand({3, 3, 3})"
        else:
            arg_map["self"] = "at::rand({22, 3, 22})"
            arg_map["other"] = "at::rand({22, 3, 22})"
        return
    elif op_name == "take":
        if index == 0:
            arg_map["index"] = "at::randint(0, 216, {20}, torch::kInt64)"
        else:
            arg_map["index"] = "at::randint(0, 1000, {100}, torch::kInt64)"
        return
    elif op_name == "take_along_dim":
        if index == 0:
            arg_map["indices"] = "at::argsort(self0, 1, true)"
        else:
            arg_map["indices"] = "at::argsort(self1, 1, true)"
        return
    # 如果操作名为 "masked_select"
    if op_name == "masked_select":
        # 如果索引为 0，则设置 "mask" 参数为一个 6x6x6 的随机张量，大于 0.5 的元素为 true
        if index == 0:
            arg_map["mask"] = "at::randn({6, 6, 6}) > 0.5"
        # 否则，设置 "mask" 参数为一个 22x22x22 的随机张量，大于 0.5 的元素为 true
        else:
            arg_map["mask"] = "at::rand({22, 22, 22}) > 0.5"
        return

    # 如果操作名为 "orgqr"
    if op_name == "orgqr":
        # 如果索引为 0，则设置 "input2" 参数为一个 6x6 的随机张量
        if index == 0:
            arg_map["input2"] = "at::rand({6, 6})"
        # 否则，设置 "input2" 参数为一个 22x22 的随机张量
        else:
            arg_map["input2"] = "at::rand({22, 22})"
        return

    # 如果操作名为 "ormqr"
    if op_name == "ormqr":
        # 如果索引为 0，则设置 "input2" 参数为一个 6x6 的随机张量
        if index == 0:
            arg_map["input2"] = "at::rand({6, 6})"
        # 否则，设置 "input2" 参数为一个 22x22 的随机张量
        else:
            arg_map["input2"] = "at::rand({22, 22})"
        return

    # 如果操作名为 "quantile"
    if op_name == "quantile":
        # 如果索引为 0
        if index == 0:
            # 设置 "q" 参数为一个长度为 6 的随机张量
            arg_map["q"] = "at::rand({6})"
            # 设置 "interpolation" 参数为 "linear"
            arg_map["interpolation"] = '"linear"'
        else:
            # 否则，设置 "q" 参数为一个长度为 22 的随机张量
            arg_map["q"] = "at::rand({22})"
            # 设置 "interpolation" 参数为 "linear"
            arg_map["interpolation"] = '"linear"'
        return

    # 如果操作名为 "nanquantile"
    if op_name == "nanquantile":
        # 如果索引为 0
        if index == 0:
            # 设置 "q" 参数为一个长度为 6 的随机张量
            arg_map["q"] = "at::rand({6})"
            # 设置 "interpolation" 参数为 "linear"
            arg_map["interpolation"] = '"linear"'
        else:
            # 否则，设置 "q" 参数为一个长度为 22 的随机张量
            arg_map["q"] = "at::rand({22})"
            # 设置 "interpolation" 参数为 "linear"
            arg_map["interpolation"] = '"linear"'
        return

    # 如果操作名为 "multi_margin_loss"
    if op_name == "multi_margin_loss":
        # 如果索引为 0
        if index == 0:
            # 设置 "self" 参数为一个 6x6 的随机张量
            arg_map["self"] = "at::rand({6, 6})"
            # 设置 "target" 参数为一个长度为 6 的随机整数张量
            arg_map["target"] = "at::randint(6, {6}, torch::kInt64)"
            # 设置 "weight" 参数为一个长度为 6 的随机张量
            arg_map["weight"] = "at::rand({6})"
        else:
            # 否则，设置 "self" 参数为一个 22x22 的随机张量
            arg_map["self"] = "at::rand({22, 22})"
            # 设置 "target" 参数为一个长度为 22 的随机整数张量
            arg_map["target"] = "at::randint(22, {22}, torch::kInt64)"
            # 设置 "weight" 参数为一个长度为 22 的随机张量
            arg_map["weight"] = "at::rand({22})"
        return

    # 如果操作名为 "multilabel_margin_loss"
    if op_name == "multilabel_margin_loss":
        # 如果索引为 0
        if index == 0:
            # 设置 "self" 参数为一个 6x6 的随机张量
            arg_map["self"] = "at::rand({6, 6})"
            # 设置 "target" 参数为一个长度为 6x6 的随机整数张量
            arg_map["target"] = "at::randint(6, {6, 6}, torch::kInt64)"
        else:
            # 否则，设置 "self" 参数为一个 22x22 的随机张量
            arg_map["self"] = "at::rand({22, 22})"
            # 设置 "target" 参数为一个长度为 22x22 的随机整数张量
            arg_map["target"] = "at::randint(22, {22, 22}, torch::kInt64)"
        return

    # 如果操作名为 "nll_loss"
    if op_name == "nll_loss":
        # 如果索引为 0
        if index == 0:
            # 设置 "self" 参数为一个 6x6 的随机张量
            arg_map["self"] = "at::rand({6, 6})"
            # 设置 "target" 参数为一个长度为 6 的随机整数张量
            arg_map["target"] = "at::randint(6, {6}, torch::kInt64)"
            # 设置 "weight" 参数为一个长度为 6 的随机张量
            arg_map["weight"] = "at::rand({6})"
        else:
            # 否则，设置 "self" 参数为一个 22x22 的随机张量
            arg_map["self"] = "at::rand({22, 22})"
            # 设置 "target" 参数为一个长度为 22 的随机整数张量
            arg_map["target"] = "at::randint(22, {22}, torch::kInt64)"
            # 设置 "weight" 参数为一个长度为 22 的随机张量
            arg_map["weight"] = "at::rand({22})"
        return

    # 如果操作名为 "nll_loss2d"
    if op_name == "nll_loss2d":
        # 如果索引为 0
        if index == 0:
            # 设置 "self" 参数为一个 6x6x6x6 的随机张量
            arg_map["self"] = "at::rand({6, 6, 6, 6})"
            # 设置 "target" 参数为一个长度为 6x6x6 的随机整数张量
            arg_map["target"] = "at::randint(6, {6, 6, 6}, torch::kInt64)"
            # 设置 "weight" 参数为一个长度为 6 的随机张量
            arg_map["weight"] = "at::rand({6})"
        else:
            # 否则，设置 "self" 参数为一个 22x22x22x22 的随机张量
            arg_map["self"] = "at::rand({22, 22, 22, 22})"
            # 设置 "target" 参数为一个长度为 22x22x22 的随机整数张量
            arg_map["target"] = "at::randint(22, {22, 22, 22}, torch::kInt64)"
            # 设置 "weight" 参数为一个长度为 22 的随机张量
            arg_map["weight"] = "at::rand({22})"
        return

    # 如果操作名为 "fft_fft", "fft_ifft", "fft_rfft", "fft_irfft", "fft_hfft", "fft_ihfft" 中的任意一个
    if op_name in (
        "fft_fft",
        "fft_ifft",
        "fft_rfft",
        "fft_irfft",
        "fft_hfft",
        "fft_ihfft",
    ):
        # 设置 "norm" 参数为 "forward"
        arg_map["norm"] = '"forward"'
        return
    # 如果操作名称是 "linalg_tensorinv"
    if op_name == "linalg_tensorinv":
        # 如果索引是 0，设置参数字典中的 "self" 和 "ind"
        if index == 0:
            arg_map["self"] = "at::rand({6, 6, 6, 6})"
            arg_map["ind"] = "2"
        else:
            # 否则，设置参数字典中的 "self" 和 "ind"
            arg_map["self"] = "at::rand({22, 22, 22, 22})"
            arg_map["ind"] = "2"
        return
    
    # 如果操作名称是 "addmv"
    if op_name == "addmv":
        # 如果索引是 0，设置参数字典中的 "self"、"mat" 和 "vec"
        if index == 0:
            arg_map["self"] = "at::rand({2})"
            arg_map["mat"] = "at::rand({2, 2})"
            arg_map["vec"] = "at::rand({2})"
        else:
            # 否则，设置参数字典中的 "self"、"mat" 和 "vec"
            arg_map["self"] = "at::rand({35})"
            arg_map["mat"] = "at::rand({35, 35})"
            arg_map["vec"] = "at::rand({35})"
        return
    
    # 如果操作名称是 "acosh"
    if op_name == "acosh":
        # 如果索引是 0，设置参数字典中的 "self"
        if index == 0:
            arg_map["self"] = "at::rand({2, 2, 2}) + at::ones({2, 2, 2})"
        else:
            # 否则，设置参数字典中的 "self"
            arg_map["self"] = "at::rand({5, 5, 5}) + at::ones({5, 5, 5})"
        return
    
    # 如果操作名称是 "adaptive_max_pool2d_backward"
    if op_name == "adaptive_max_pool2d_backward":
        # 如果索引是 0，设置参数字典中的 "grad_output"、"self" 和 "indices"
        if index == 0:
            arg_map["grad_output"] = "at::rand({2, 2, 2}, at::kFloat)"
            arg_map["self"] = "at::rand({2, 2, 2}, at::kFloat)"
            arg_map["indices"] = "at::randint(0, 1, {2, 2, 2}, at::kLong)"
        else:
            # 否则，设置参数字典中的 "grad_output"、"self" 和 "indices"
            arg_map["grad_output"] = "at::rand({3, 3, 3}, at::kFloat)"
            arg_map["self"] = "at::rand({3, 3, 3}, at::kFloat)"
            arg_map["indices"] = "at::randint(0, 1, {3, 3, 3}, at::kLong)"
        return
    
    # 如果操作名称是 "adaptive_max_pool3d_backward"
    if op_name == "adaptive_max_pool3d_backward":
        # 如果索引是 0，设置参数字典中的 "grad_output"、"self" 和 "indices"
        if index == 0:
            arg_map["grad_output"] = "at::rand({2, 2, 2, 2}, at::kFloat)"
            arg_map["self"] = "at::rand({2, 2, 2, 2}, at::kFloat)"
            arg_map["indices"] = "at::randint(0, 1, {2, 2, 2, 2}, at::kLong)"
        else:
            # 否则，设置参数字典中的 "grad_output"、"self" 和 "indices"
            arg_map["grad_output"] = "at::rand({3, 3, 3, 3}, at::kFloat)"
            arg_map["self"] = "at::rand({3, 3, 3, 3}, at::kFloat)"
            arg_map["indices"] = "at::randint(0, 1, {3, 3, 3, 3}, at::kLong)"
        return
    
    # 如果操作名称是 "bitwise_left_shift"
    if op_name == "bitwise_left_shift":
        # 如果索引是 0，设置参数字典中的 "self" 和 "other"
        if index == 0:
            arg_map["self"] = "at::randint(1, 1 << 4, {6, 6, 6}, at::kInt)"
            arg_map["other"] = "at::randint(1, 26, {6, 6, 6}, at::kInt)"
        else:
            # 否则，设置参数字典中的 "self" 和 "other"
            arg_map["self"] = "at::randint(1, 1 << 4, {22, 22, 22}, at::kInt)"
            arg_map["other"] = "at::randint(1, 26, {22, 22, 22}, at::kInt)"
        return
    
    # 如果操作名称是 "bitwise_right_shift"
    if op_name == "bitwise_right_shift":
        # 如果索引是 0，设置参数字典中的 "self" 和 "other"
        if index == 0:
            arg_map["self"] = "at::randint(1 << 21, 1 << 30, {6, 6, 6}, at::kInt)"
            arg_map["other"] = "at::randint(1, 22, {6, 6, 6}, at::kInt)"
        else:
            # 否则，设置参数字典中的 "self" 和 "other"
            arg_map["self"] = "at::randint(1 << 21, 1 << 30, {22, 22, 22}, at::kInt)"
            arg_map["other"] = "at::randint(1, 22, {22, 22, 22}, at::kInt)"
        return
    # 如果操作名称为 "gather"
    if op_name == "gather":
        # 如果索引为 0
        if index == 0:
            # 设定自变量为随机整数张量，形状为 {2,2,2}，数据类型为整数
            arg_map["self"] = "at::randint(1, 100, {2,2,2}, at::kInt)"
            # 设定维度为 1
            arg_map["dim"] = "1"
            # 设定索引为随机整数张量，形状为 {2,2,2}，数据类型为 int64
            arg_map["index"] = "at::randint(0, 1, {2,2,2}, torch::kInt64)"
            # 设定稀疏梯度为假
            arg_map["sparse_grad"] = "false"
        else:
            # 设定自变量为随机整数张量，形状为 {5,5,5}，数据类型为整数
            arg_map["self"] = "at::randint(1, 100, {5,5,5}, at::kInt)"
            # 设定维度为 1
            arg_map["dim"] = "1"
            # 设定索引为随机整数张量，形状为 {5,5,5}，数据类型为 int64
            arg_map["index"] = "at::randint(0, 4, {5,5,5}, torch::kInt64)"
            # 设定稀疏梯度为假
            arg_map["sparse_grad"] = "false"
        # 返回结果
        return
    
    # 如果操作名称为 "gelu"
    if op_name == "gelu":
        # 如果索引为 0
        if index == 0:
            # 设定自变量为形状为 {6, 6, 6} 的随机张量
            arg_map["self"] = "at::rand({6, 6, 6})"
            # 设定近似值为 "tanh"
            arg_map["approximate"] = '"tanh"'
        else:
            # 设定自变量为形状为 {22, 22, 22} 的随机张量
            arg_map["self"] = "at::rand({22, 22, 22})"
            # 设定近似值为 "tanh"
            arg_map["approximate"] = '"tanh"'
        # 返回结果
        return
    
    # 如果操作名称为 "gelu_backward"
    if op_name == "gelu_backward":
        # 如果索引为 0
        if index == 0:
            # 设定梯度输出为形状为 {6, 6, 6} 的随机张量
            arg_map["grad_output"] = "at::rand({6, 6, 6})"
            # 设定自变量为形状为 {6, 6, 6} 的随机张量
            arg_map["self"] = "at::rand({6, 6, 6})"
            # 设定近似值为 "tanh"
            arg_map["approximate"] = '"tanh"'
        else:
            # 设定梯度输出为形状为 {22, 22, 22} 的随机张量
            arg_map["grad_output"] = "at::rand({22, 22, 22})"
            # 设定自变量为形状为 {22, 22, 22} 的随机张量
            arg_map["self"] = "at::rand({22, 22, 22})"
            # 设定近似值为 "tanh"
            arg_map["approximate"] = '"tanh"'
        # 返回结果
        return
    
    # 如果操作名称为 "index_add"
    if op_name == "index_add":
        # 如果索引为 0
        if index == 0:
            # 设定自变量为形状为 {2} 的随机张量
            arg_map["self"] = "at::rand({2})"
            # 设定维度为 0
            arg_map["dim"] = "0"
            # 设定索引为随机整数张量，形状为 {2}，数据类型为整数
            arg_map["index"] = "at::randint(0, 1, {2}, at::kInt)"
            # 设定源张量为形状为 {2} 的随机张量
            arg_map["source"] = "at::rand({2})"
            # 设定 alpha 参数为 2
            arg_map["alpha"] = "2"
        else:
            # 设定自变量为形状为 {16} 的随机张量
            arg_map["self"] = "at::rand({16})"
            # 设定维度为 0
            arg_map["dim"] = "0"
            # 设定索引为随机整数张量，形状为 {16}，数据类型为整数
            arg_map["index"] = "at::randint(0, 10, {16}, at::kInt)"
            # 设定源张量为形状为 {16} 的随机张量
            arg_map["source"] = "at::rand({16})"
            # 设定 alpha 参数为 2
            arg_map["alpha"] = "2"
        # 返回结果
        return
    
    # 如果操作名称为 "index_copy"
    if op_name == "index_copy":
        # 如果索引为 0
        if index == 0:
            # 设定自变量为形状为 {2} 的随机张量
            arg_map["self"] = "at::rand({2})"
            # 设定维度为 0
            arg_map["dim"] = "0"
            # 设定索引为随机整数张量，形状为 {2}，数据类型为长整型
            arg_map["index"] = "at::randint(0, 1, {2}, at::kLong)"
            # 设定源张量为形状为 {2} 的随机张量
            arg_map["source"] = "at::rand({2})"
        else:
            # 设定自变量为形状为 {32} 的随机张量
            arg_map["self"] = "at::rand({32})"
            # 设定维度为 0
            arg_map["dim"] = "0"
            # 设定索引为随机整数张量，形状为 {32}，数据类型为长整型
            arg_map["index"] = "at::randint(0, 10, {32}, at::kLong)"
            # 设定源张量为形状为 {32} 的随机张量
            arg_map["source"] = "at::rand({32})"
        # 返回结果
        return
    
    # 如果操作名称为 "linalg_cross"
    if op_name == "linalg_cross":
        # 如果索引为 0
        if index == 0:
            # 设定自变量为形状为 {6, 3, 6} 的随机张量
            arg_map["self"] = "at::rand({6, 3, 6})"
            # 设定其他参数为形状为 {6, 3, 6} 的随机张量
            arg_map["other"] = "at::rand({6, 3, 6})"
            # 设定维度为 1
            arg_map["dim"] = "1"
        else:
            # 设定自变量为形状为 {22, 3, 22} 的随机张量
            arg_map["self"] = "at::rand({22, 3, 22})"
            # 设定其他参数为形状为 {22, 3, 22} 的随机张量
            arg_map["other"] = "at::rand({22, 3, 22})"
            # 设定维度为 1
            arg_map["dim"] = "1"
        # 返回结果
        return
    # 如果操作名为 "nll_loss_backward"，根据索引不同设置不同的参数映射
    if op_name == "nll_loss_backward":
        if index == 0:
            arg_map["grad_output"] = "at::rand({})"
            arg_map["self"] = "at::rand({6})"
            arg_map["target"] = "at::randint(0, 5, {6}, torch::kInt64)"
            arg_map["weight"] = "at::rand({6})"
            arg_map["reduction"] = "1"
            arg_map["ignore_index"] = "1"
            arg_map["total_weight"] = "at::rand({})"
        else:
            arg_map["grad_output"] = "at::rand({})"
            arg_map["self"] = "at::rand({36})"
            arg_map["target"] = "at::randint(0, 11, {36}, torch::kInt64)"
            arg_map["weight"] = "at::rand({36})"
            arg_map["reduction"] = "1"
            arg_map["ignore_index"] = "1"
            arg_map["total_weight"] = "at::rand({})"
        return

    # 如果操作名在列表 ["scatter", "scatter_add", "_scatter_reduce"] 中，根据索引不同设置不同的参数映射
    if op_name in ["scatter", "scatter_add", "_scatter_reduce"]:
        if index == 0:
            arg_map["self"] = "at::randint(1, 100, {2,2,2}, torch::kInt64)"
            arg_map["index"] = "at::randint(0, 1, {2,2,2}, torch::kInt64)"
            arg_map["src"] = "at::randint(1, 100, {2,2,2}, torch::kInt64)"
        else:
            arg_map["self"] = "at::randint(1, 100, {5,5,5}, torch::kInt64)"
            arg_map["index"] = "at::randint(0, 1, {5,5,5}, torch::kInt64)"
            arg_map["src"] = "at::randint(1, 100, {5,5,5}, torch::kInt64)"
        
        # 如果参数映射中有 "reduce" 字段，并且操作名为 "_scatter_reduce"，则设置 reduce 为 "sum"，否则为 "add"
        if "reduce" in arg_map:
            arg_map["reduce"] = '"sum"' if op_name == "_scatter_reduce" else '"add"'
        return

    # 如果操作名为 "scatter_reduce"，根据索引不同设置不同的参数映射
    if op_name == "scatter_reduce":
        arg_map["reduce"] = '"mean"'
        if index == 0:
            arg_map["index"] = "at::randint(6, {6, 6, 6}, torch::kInt64)"
        else:
            arg_map["index"] = "at::randint(22, {22, 22, 22}, torch::kInt64)"
        return

    # 如果操作名为 "special_zeta"，根据索引不同设置不同的参数映射
    if op_name == "special_zeta":
        if index == 0:
            arg_map["self"] = "at::rand({2,2,2}, at::kDouble) + at::ones({2,2,2})"
            arg_map["other"] = "at::rand({2,2,2}, at::kDouble) + at::ones({2,2,2})"
        else:
            arg_map["self"] = "at::rand({5,5,5}, at::kDouble) + at::ones({5,5,5})"
            arg_map["other"] = "at::rand({5,5,5}, at::kDouble) + at::ones({5,5,5})"
        return

    # 如果操作名为 "_convert_indices_from_csr_to_coo"，根据索引不同设置不同的参数映射
    if op_name == "_convert_indices_from_csr_to_coo":
        if index == 0:
            arg_map["crow_indices"] = "torch::tensor({1}, torch::kInt32)"
            arg_map["col_indices"] = "torch::tensor({0, 1, 0}, torch::kInt32)"
            arg_map["out_int32"] = "false"
        else:
            arg_map["crow_indices"] = "torch::tensor({0}, torch::kInt32)"
            arg_map[
                "col_indices"
            ] = "torch::tensor({0, 1, 0, 2, 1, 2, 0, 1, 0, 2, 1, 2}, torch::kInt32)"
            arg_map["out_int32"] = "false"
        return
    # 如果操作名称是 "_convert_indices_from_coo_to_csr"，根据索引设置参数映射
    if op_name == "_convert_indices_from_coo_to_csr":
        # 如果索引为 0，设置特定参数值
        if index == 0:
            arg_map["self"] = "at::randint(0, 3, {2}, at::kInt)"
            arg_map["size"] = "10"
            arg_map["out_int32"] = "false"
        else:
            # 否则，设置另一组参数值
            arg_map["self"] = "at::randint(0, 3, {12}, at::kInt)"
            arg_map["size"] = "24"
            arg_map["out_int32"] = "false"
        # 函数返回，不再执行后续代码
        return
    
    # 如果操作名称是 "diagonal" 或 "linalg_diagonal"，设置特定参数映射
    if op_name in ("diagonal", "linalg_diagonal"):
        arg_map["offset"] = "0"
        arg_map["dim1"] = "2"
        arg_map["dim2"] = "1"
        # 函数返回，不再执行后续代码
        return
```