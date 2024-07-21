# `.\pytorch\torch\fx\experimental\graph_gradual_typechecker.py`

```py
# mypy: allow-untyped-defs
# 导入 reduce 函数和 torch 库
from functools import reduce
import torch
# 导入 operator 模块
import operator
# 导入与类型推断相关的类和函数
from torch.fx.tensor_type import Dyn, is_consistent, TensorType, is_more_precise
# 导入类型和函数字典
from typing import Callable, Dict
# 导入节点类和目标类
from torch.fx.node import Target, Node
# 导入批归一化和卷积层类
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
# 导入实验性质的相等性
from torch.fx.experimental.refinement_types import Equality
# 导入 itertools 库
import itertools

# 导入变量类
from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]

# 导入 sympy 库
import sympy

# 创建三个空的规则字典
_INFERENCE_RULES: Dict[Target, Callable] = {}
_REFINEMENT_RULES: Dict[Target, Callable] = {}
_RULES: Dict[Target, Callable] = {}


def expand_to_tensor_dim(t, n):
    """
    尝试将给定类型扩展到指定的张量维度
    如果不可能则抛出错误
    - t 是给定的类型
    - n 是要扩展到的维度数
    """
    if t == Dyn:
        # 如果类型为 Dyn，则创建全为 Dyn 的维度
        dims = [Dyn] * n
        return TensorType(tuple(dims))
    elif isinstance(t, TensorType):
        if len(t.__args__) != n:
            # 如果类型为 TensorType 但维度与 n 不匹配，则抛出错误
            raise TypeError(f'Cannot extend tensor. Tensor {t} has rank {len(t.__args__)}. It should have rank {n}')
        return t
    else:
        # 如果类型不是 Dyn 或 TensorType，则抛出类型错误
        raise TypeError(f'Cannot match the type {t}')


def broadcast_types(t1, t2):
    """
    对给定的两个类型应用广播，使它们彼此一致，并返回两个新的结果类型
    """
    # 如果其中一个类型是 Dyn 或者包含 Var，则不进行广播，因为类型已经一致
    if t1 == Dyn or t2 == Dyn or isinstance(t1, Var) or isinstance(t2, Var):
        return t1, t2

    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        s1 = len(t1.__args__)
        s2 = len(t2.__args__)

        new_t1 = list(t1.__args__)
        new_t2 = list(t2.__args__)

        # 使两个类型具有相同的长度，这是一致性的第一个要求
        if s1 > s2:
            for i in range(s1 - s2):
                new_t2.insert(0, 1)
        elif s2 > s1:
            for i in range(s2 - s1):
                new_t1.insert(0, 1)

        # 将每个类型中的 "1" 替换为另一个类型中对应的维度
        for i, (x, y) in enumerate(zip(new_t1, new_t2)):
            if x == 1:
                new_t1[i] = y
            elif y == 1:
                new_t2[i] = x

        # 现在我们的张量应该是一致的，可以应用逐元素操作，并找到操作的输出维度
        return TensorType(tuple(new_t1)), TensorType(tuple(new_t2))
    else:
        # 如果两个类型不都是 TensorType，则抛出类型错误
        raise TypeError(f'Cannot broadcast types {t1} and {t2}')


def register_inference_rule(call_target):
    """
    注册推断规则的装饰器函数
    """
    def register(fn):
        if call_target in _INFERENCE_RULES:
            raise RuntimeError(f'Inference rule already registered for {call_target}!')
        _INFERENCE_RULES[call_target] = fn
        return fn
    return register
# 定义一个装饰器函数，用于注册细化规则
def register_refinement_rule(call_target):
    # 内部函数，用于注册具体的细化规则函数
    def register(fn):
        # 如果给定的调用目标已经在_REFINEMENT_RULES字典中存在，则抛出运行时错误
        if call_target in _REFINEMENT_RULES:
            raise RuntimeError(f'Refinement rule already registered for {call_target}!')
        # 否则将该细化规则函数注册到_REFINEMENT_RULES字典中
        _REFINEMENT_RULES[call_target] = fn
        return fn
    return register

# 定义一个装饰器函数，用于注册代数推断规则
def register_algebraic_expressions_inference_rule(call_target):
    # 内部函数，用于注册具体的代数推断规则函数
    def register(fn):
        # 如果给定的调用目标已经在_RULES字典中存在，则抛出运行时错误
        if call_target in _RULES:
            raise RuntimeError(f'Rule already registered for {call_target}!')
        # 否则将该代数推断规则函数注册到_RULES字典中
        _RULES[call_target] = fn
        return fn
    return register

# 使用@register_inference_rule装饰器将torch.add和operator.add注册为推断规则
@register_inference_rule(torch.add)
@register_inference_rule(operator.add)
def add_inference_rule(n: Node):
    """
    应用加法推断规则。包括：
    - 标量加法
    - 广播语义

    注意我们总是返回操作数（在应用广播后）之间最不精确的类型作为操作的最终类型

    注意我们在应用广播后不会修改操作数本身的类型。我们只使用它们来计算最终类型
    """
    assert isinstance(n.args[0], Node)
    assert isinstance(n.args[1], Node)
    t1 = n.args[0].type
    t2 = n.args[1].type

    # 处理标量加法
    if t1 == int and isinstance(t2, TensorType):
        n.type = t2
        return n.type

    elif t2 == int and isinstance(t1, TensorType):
        n.type = t1
        return n.type

    # 将新类型带到检查一致性的点
    # 任何不一致性都不会是广播引起的
    (new_t1, new_t2) = broadcast_types(t1, t2)

    if new_t1 != t1 or new_t2 != t2:
        n.meta['broadcast'] = True
        n.meta[str(n.args[0])] = new_t1
        n.meta[str(n.args[1])] = new_t2

    else:
        n.meta['broadcast'] = False

    new_t1 = t1 if not n.meta['broadcast'] else new_t1
    new_t2 = t2 if not n.meta['broadcast'] else new_t2

    # 检查新类型之间的一致性
    if is_consistent(new_t1, new_t2):
        # 返回较不精确的类型，因为可能已经发生了广播
        if is_more_precise(new_t1, new_t2):
            n.type = new_t2
        else:
            n.type = new_t1
        return n.type
    else:
        # 如果类型不一致，则抛出类型错误
        raise TypeError(f'Cannot add arguments {n.args[0]} ({ n.args[0].type}) and {n.args[1]} ({ n.args[1].type}) in node {n}.'
                        f' Types should match ')

# 使用@register_inference_rule装饰器将getattr函数注册为推断规则
@register_inference_rule(getattr)
def get_attr_inference_rule(n: Node, traced):
    """
    当前getattr规则只处理shape属性
    可以扩展到其他属性
    我们目前最具代表性的类型是"Dyn"，但系统可以扩展更多类型，比如用于表示形状的类型
    """
    attr_node = n.args[0]
    attr_name = n.args[1]
    # 如果属性名为 "shape"
    if attr_name == "shape":
        # 将节点 n 的类型设置为 Dyn
        n.type = Dyn
    else:
        # 抛出类型错误异常，说明尚未实现处理该属性名的情况
        raise TypeError("Not yet implemented")

    # TODO. 目前保留此处代码，直到我们添加一种类型来表示张量的大小
    # 返回节点 n 的类型
    return n.type
@register_inference_rule(torch.transpose)
def transpose_inference_rule(n: Node):
    """
    注册推断规则，处理 torch.transpose 操作的节点 n

    确保转置操作的维度在节点的张量类型范围内
    """
    if n.target == torch.transpose:
        assert isinstance(n.args[0], Node)
        # 获取节点的输入类型
        t = n.args[0].type

        assert isinstance(n.args[1], int)
        assert isinstance(n.args[2], int)
        # 获取转置操作的两个维度
        dim1, dim2 = n.args[1], n.args[2]

        if t == Dyn:
            # 如果原始张量维度未知，返回动态类型
            n.type = Dyn
            return n.type

        elif isinstance(t, TensorType):
            if 0 <= dim1 < len(t.__args__) and 0 <= dim2 < len(t.__args__):
                # 交换维度并生成新的张量类型
                new_type = list(t.__args__)
                new_type[dim1], new_type[dim2] = new_type[dim2], new_type[dim1]
                final = TensorType(new_type)
                # 获取节点类型的最大上界
                n.type = get_greatest_upper_bound(n.type, final)
                return n.type
            else:
                raise TypeError(f'节点 {n} 的类型 {t} 中无法进行维度 {dim1} 和 {dim2} 的转置')
        else:
            raise TypeError(f'节点 {n} 的类型 {t} 不支持进行维度 {dim1} 和 {dim2} 的转置')


@register_inference_rule(torch.reshape)
def reshape_inference_rule(n: Node):
    """
    注册推断规则，处理 torch.reshape 操作的节点 n

    在不涉及动态维度的情况下，检查输入张量的元素乘积
    是否等于所需形状的元素乘积。逐步处理动态输入的情况，
    以及其中一些张量维度未知的情况。
    """
    assert isinstance(n.args[0], Node)
    # 获取节点的输入类型
    t1 = n.args[0].type

    assert isinstance(n.args[1], list)
    # 获取目标形状
    t2 = n.args[1]
    t2_type = TensorType([Dyn if elem == -1 else elem for elem in t2])

    # 如果原始张量维度未知，返回所需的目标形状
    if t1 == Dyn:
        n.type = t2_type
        return t2_type

    # 如果任何维度未知，检查是否可分割
    elif isinstance(t1, TensorType):
        assert isinstance(t1, TensorType)
        a = [e if e != Dyn else 1 for e in t1.__args__]
        p1 = reduce(operator.mul, a)
        p2 = reduce(operator.mul, t2)
        if p1 % p2 == 0 or p2 % p1 == 0:
            # 若可分割，则返回目标形状类型
            n.type = t2_type
            return t2_type
        else:
            raise TypeError(f'节点 {n} 中无法从类型 {t1} 转换到 {t2_type}')
    else:
        raise TypeError(f'节点 {n} 中无法从类型 {t1} 转换到 {t2_type}')


@register_inference_rule(BatchNorm2d)
def bn2d_inference_rule(n: Node, module_instance):
    """
    注册推断规则，处理 BatchNorm2d 实例及其节点 n

    检查以下条件：
    - 输入类型可以扩展为大小为 4 的张量：t = (x_1, x_2, x_3, x_4)
    - 当前节点类型可以扩展为大小为 4 的张量：t' = (x_1', x_2', x_3', x_4')
    - t 与 t' 一致
    - x_2 与模块的 num_features 一致
    - x_2' 与模块的 num_features 一致
    """
    """
    output type: the more precise type of t and t'
    """
    # 确保 n.args[0] 是一个 Node 类的实例
    assert isinstance(n.args[0], Node)
    # 将 n.args[0] 的类型扩展为张量维度为 4
    n.args[0].type = expand_to_tensor_dim(n.args[0].type, 4)
    # 获取扩展后的参数类型
    arg_type = n.args[0].type
    # 将 n 的类型也扩展为张量维度为 4
    n.type = expand_to_tensor_dim(n.type, 4)

    # 检查传入参数的条件和任何现有的注解
    # 同时检查两者之间的一致性
    if is_consistent(arg_type.__args__[1], module_instance.num_features) and \
            is_consistent(n.type.__args__[1], module_instance.num_features) and \
            is_consistent(arg_type, n.type):
        
        # 选择更精确的类型作为节点的类型
        # 如果传入参数有更多的类型信息，将节点的类型设为传入参数的类型
        n.type = get_greatest_upper_bound(arg_type, n.type)
        # 返回节点的类型
        return n.type
    else:
        # 抛出类型错误，说明无法将 module_instance 应用到给定输入类型上
        raise TypeError(f'Cannot apply {module_instance} with input type {arg_type} and existing type {n.type} on {n}')
# 计算输出维度中的高度或宽度，根据给定的 conv2D 文档
def calculate_out_dimension(d_in, module_instance, index):
    # 如果填充是整数，转换为元组形式；否则保持原样
    padding = (module_instance.padding, module_instance.padding) \
        if isinstance(module_instance.padding, int) else module_instance.padding
    # 如果卷积核大小是整数，转换为元组形式；否则保持原样
    kernel_size = (module_instance.kernel_size, module_instance.kernel_size) \
        if isinstance(module_instance.kernel_size, int) else module_instance.kernel_size
    # 如果步长是整数，转换为元组形式；否则保持原样
    stride = (module_instance.stride, module_instance.stride) \
        if isinstance(module_instance.stride, int) else module_instance.stride
    # 如果膨胀率是整数，转换为元组形式；否则保持原样
    dilation = (module_instance.dilation, module_instance.dilation) \
        if isinstance(module_instance.dilation, int) else module_instance.dilation

    # 支持的维度类型，包括整数和符号类型
    DIMENSION_TYPES = (int, sympy.Symbol)

    # 如果输入维度是动态的，直接返回动态维度
    if d_in == Dyn:
        return Dyn

    # 如果输入维度是整数或符号类型
    elif isinstance(d_in, DIMENSION_TYPES):
        # 根据卷积层参数计算输出维度的一维度
        n = d_in + 2 * padding[index] - \
            dilation[index] * \
            (kernel_size[index] - 1) - 1

        # 返回计算得到的输出维度
        return (n // stride[0]) + 1

    else:
        # 如果输入维度既不是动态的，也不是整数或符号类型，则引发类型错误
        raise TypeError(f'{d_in} in {module_instance} must be a number or Dyn. Received {type(d_in)}')


# 获取两个类型中的最精确的上界类型
def get_greatest_upper_bound(type1, type2):
    # 如果第一个类型是动态类型，则返回第二个类型
    if type1 == Dyn:
        return type2
    # 如果第二个类型是动态类型，则返回第一个类型
    elif type2 == Dyn:
        return type1
    # 如果两个类型都是 TensorType，并且类型一致，则找到最精确的类型
    elif isinstance(type1, TensorType) and isinstance(type2, TensorType):
        # 如果类型不一致，引发类型错误
        if not is_consistent(type1, type2):
            raise TypeError(f'Inconsistent types {type1}, {type2}')
        # 遍历每个维度元素，选取更精确的类型
        gub = [t1 if is_more_precise(t1, t2) else t2 for (t1, t2) in zip(type1.__args__, type2.__args__)]
        # 返回 TensorType 类型的最精确上界
        return TensorType(tuple(gub))


# 注册推理规则为 Conv2d 的推理规则
@register_inference_rule(Conv2d)
def conv2d_inference_rule(n: Node, module_instance):
    # 确保第一个参数是 Node 类型
    assert isinstance(n.args[0], Node)
    # 将第一个参数的类型扩展为 4 维张量
    n.args[0].type = expand_to_tensor_dim(n.args[0].type, 4)
    # 获取扩展后的参数类型
    arg_type = n.args[0].type
    # 将当前节点的类型扩展为 4 维张量
    curr_node_type = expand_to_tensor_dim(n.type, 4)

    # 检查第一个参数的第二维度是否与模块的输入通道数一致
    if is_consistent(arg_type.__args__[1], module_instance.in_channels):
        # 获取输入维度的宽度和高度
        w_in = arg_type.__args__[3]
        h_in = arg_type.__args__[2]
        # 根据卷积层计算输出的高度和宽度
        h_out = calculate_out_dimension(h_in, module_instance, 0)
        w_out = calculate_out_dimension(w_in, module_instance, 1)
        # 构建新的类型，表示输出为 (x_1, out_channels, H_out, W_out)
        new_type = TensorType((arg_type.__args__[0], module_instance.out_channels, h_out, w_out))
        # 获取当前节点类型与新类型的最精确上界
        gub = get_greatest_upper_bound(new_type, curr_node_type)
        # 更新当前节点的类型为计算得到的最精确上界类型
        n.type = gub
        # 返回更新后的节点类型
        return n.type
    else:
        # 如果不满足前述所有条件，则抛出类型错误异常，指明无法应用给定模块实例（module_instance）到节点（n）上，
        # 因为输入类型（arg_type）与节点现有类型（n.type）不兼容。
        raise TypeError(f'Cannot apply {module_instance} with input type {arg_type} and existing type {n.type} on {n}')
@register_inference_rule(torch.nn.ReLU)
def relu_inference_rule(n: Node, module_instance):
    """
    Input and output shapes should be equal.
    """
    assert isinstance(n.args[0], Node)

    # 如果输入节点类型为动态(Dyn)且当前节点类型为张量类型(TensorType)，则将输入节点类型扩展到与当前节点类型的维度相同
    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.__args__))

    # 如果输入节点类型为张量类型，则将当前节点类型设置为输入节点类型与当前类型的最大上界
    if isinstance(n.args[0].type, TensorType):
        n.type = get_greatest_upper_bound(n.args[0].type, n.type)
    return n.type


def maxpool2d_check(typ, module_instance):
    """
    Applies the maxpool2d shape information to the input
    this affects the last two dimensions
    """
    new_type_list = list(typ.__args__)
    # 如果输入类型参数为4个或3个，则进行以下操作
    if len(new_type_list) == 4 or len(new_type_list) == 3:
        w_in = new_type_list[-1]
        h_in = new_type_list[-2]

        # 计算输出尺寸的高度和宽度
        h_out = calculate_out_dimension(h_in, module_instance, 0)
        w_out = calculate_out_dimension(w_in, module_instance, 1)

        # 更新输入类型参数的最后两个维度为计算得到的输出尺寸
        new_type_list[-1] = w_out
        new_type_list[-2] = h_out
        return TensorType(tuple(new_type_list))

    else:
        # 如果输入类型参数不是4个或3个，则引发类型错误
        raise TypeError(f'Wrong size {typ} for {module_instance}')


@register_inference_rule(torch.nn.MaxPool2d)
def maxpool2d_inference_rule(n: Node, module_instance):
    """
    Given a MaxPool2D instance and a node check the following conditions:
    - Input size matches size 3 or 4
    - Current node type is consistent with the output type we will calculate
    - Input size matches output size and the last two dimensions of the output
      are w_out and h_out. The remaining dimensions are the same as the input
    - Our final result is the greatest upper bound of the output we calculate
      and the current node type.
    """
    assert isinstance(n.args[0], Node)

    # 如果输入节点类型为动态(Dyn)且当前节点类型为张量类型(TensorType)，则将输入节点类型扩展到与当前节点类型的维度相同
    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.__args__))
    # 如果输入节点类型为张量类型，则根据 maxpool2d_check 函数计算输出类型，并将当前节点类型设置为计算结果与当前类型的最大上界
    if isinstance(n.args[0].type, TensorType):
        output = maxpool2d_check(n.args[0].type, module_instance)
        n.type = get_greatest_upper_bound(output, n.type)
    return n.type



def linear_check(tensor_type, module_instance):
    """
    Checks that an input tensor type satisfies the conditions for linear operation
    and returns the output type based on in and out features given by module_instance
    """
    # 如果张量类型参数的长度大于等于2，则执行以下操作
    if len(tensor_type.__args__) >= 2:
        # 如果输入特征和张量类型参数的最后一个元素一致，则更新类型参数的最后一个元素为输出特征数
        if is_consistent(module_instance.in_features, tensor_type.__args__[-1]):
            new_type_args = list(tensor_type.__args__)
            new_type_args[-1] = module_instance.out_features
            return TensorType(tuple(new_type_args))
        else:
            # 如果输入特征和张量类型参数的最后一个元素不一致，则引发类型错误
            raise TypeError(f'Inconsistent {module_instance.in_features} and {tensor_type.__args__[-1]} in {module_instance}')
    else:
        # 如果张量类型参数的长度小于2，则引发类型错误
        raise TypeError(f'Type {tensor_type} must have rank 2 or more.')


@register_inference_rule(torch.nn.Linear)
def linear_inference_rule(n: Node, module_instance):
    """
    Given a Linear instance and a node, check and update the node's type based on input and output features.
    """
    # 断言输入的第一个参数是节点类型
    assert isinstance(n.args[0], Node)

    # 如果输入节点类型为动态(Dyn)且当前节点类型为张量类型(TensorType)，则将输入节点类型扩展到与当前节点类型的维度相同
    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.__args__))
    # 如果输入节点类型为张量类型，则根据 linear_check 函数计算输出类型，并将当前节点类型设置为计算结果与当前类型的最大上界
    if isinstance(n.args[0].type, TensorType):
        output = linear_check(n.args[0].type, module_instance)
        n.type = get_greatest_upper_bound(output, n.type)
    return n.type
    """
    Applies the shape information to the input then gets the greatest upper bound
    of the resulting type and the existing type
    """
    # 断言输入参数 n.args[0] 是一个 Node 对象
    assert isinstance(n.args[0], Node)
    
    # 如果输入参数 n.args[0] 的类型是 Dyn，而 n.type 是 TensorType 类型的实例，则扩展 n.args[0] 的类型以匹配 n.type 的张量维度
    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.__args__))
    
    # 如果 n.args[0] 的类型是 TensorType 类型
    if isinstance(n.args[0].type, TensorType):
        # 对 n.args[0].type 进行线性检查，返回输出类型
        output_type = linear_check(n.args[0].type, module_instance)
        # 获取输出类型 output_type 和 n.type 的最大上界
        n.type = get_greatest_upper_bound(output_type, n.type)
    
    # 返回更新后的类型 n.type
    return n.type
def adaptiveavgpool2d_check(tensor_type, module_instance):
    # 获取模块实例的输出大小
    output_size = module_instance.output_size

    # 如果输出大小是整数，则转换为列表形式 [output_size, output_size]
    if isinstance(output_size, int):
        output_size = [output_size, output_size]
    # 如果输出大小是元组，则转换为列表形式，并确保其中的 None 值得到处理
    elif isinstance(output_size, tuple):
        output_size = list(output_size)
        if output_size[0] is None:
            output_size[0] = output_size[1]
        if output_size[1] is None:
            output_size[1] = output_size[0]

    # 获取张量类型的参数列表
    new_type_list = list(tensor_type.__args__)

    # 如果张量类型的参数列表长度为 4 或 3
    if len(tensor_type.__args__) == 4 or len(tensor_type.__args__) == 3:
        # 更新张量类型的参数列表中倒数第二和倒数第一位，分别为输出大小的宽度和高度
        new_type_list[-1] = output_size[1]
        new_type_list[-2] = output_size[0]

        # 返回更新后的张量类型
        return TensorType(tuple(new_type_list))

    else:
        # 抛出类型错误，要求张量秩必须为 3 或 4
        raise TypeError(f'Tensor ranks must be 3 or 4. Got {tensor_type}')

@register_inference_rule(torch.nn.AdaptiveAvgPool2d)
def adaptiveavgpool2d_inference_rule(n: Node, module_instance):
    """
    应用自适应平均池化的推断规则，确保输入和输出的尺寸在除了最后两个维度（宽度和高度）之外都相同
    """
    assert isinstance(n.args[0], Node)

    # 如果输入节点的类型是动态的，并且输出类型是张量类型
    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        # 扩展输入节点的类型，使其维数等于输出类型的维数
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.__args__))

    # 如果输入节点的类型是张量类型
    if isinstance(n.args[0].type, TensorType):
        # 运行自适应平均池化的检查函数，获取输出类型
        output_type = adaptiveavgpool2d_check(n.args[0].type, module_instance)
        # 获取输出类型和现有类型之间的最大上界
        n.type = get_greatest_upper_bound(n.type, output_type)

    # 返回推断后的节点类型
    return n.type

def flatten_check(tensor_type, start_dim, end_dim):
    # 获取张量类型的参数列表长度
    l = len(tensor_type.__args__)

    # 处理负索引的起始维度和结束维度
    start_dim = l if start_dim == -1 else abs(start_dim)
    end_dim = l + end_dim + 1 if end_dim < 0 else end_dim + 1

    # 如果起始维度和结束维度在有效范围内
    if 0 <= start_dim <= (l - 1) and 0 <= end_dim <= l and start_dim < end_dim:
        # 复制张量类型的参数列表
        my_args = list(tensor_type.__args__)
        # 分割参数列表并根据需要重新排列维度
        lhs = my_args[0:start_dim]
        rhs = my_args[end_dim:]
        mid = my_args[start_dim:end_dim]
        if Dyn in mid:
            mid = [Dyn]
        else:
            mid = [reduce(operator.mul, my_args[start_dim:end_dim])]
        new_type_list = lhs + mid + rhs
        # 返回更新后的张量类型
        return TensorType(tuple(new_type_list))
    else:
        # 抛出类型错误，指出维度不兼容
        raise TypeError(f'Incompatible dimensions {start_dim}, {end_dim - 1} in type {tensor_type}')

@register_inference_rule(torch.flatten)
def flatten_inference_rule(n: Node):
    """
    应用展平操作的推断规则，应用到输入形状信息并获取其与现有类型的最大上界
    """
    assert isinstance(n.args[0], Node)

    # 设置默认的起始和结束维度
    start_dim = 1
    end_dim = -1

    # 如果节点参数多于一个，则检查并设置起始和结束维度
    if len(n.args) > 1:
        assert isinstance(n.args[1], int)
        start_dim = n.args[1]

    if len(n.args) > 2:
        assert isinstance(n.args[2], int)
        end_dim = n.args[2]

    # 如果输入节点的类型是动态的，并且输出类型是张量类型
    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        # 扩展输入节点的类型，使其维数等于输出类型的维数
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.__args__))

    # 返回推断后的节点类型
    return n.type
    # 检查 n.args[0] 的类型是否为 TensorType 类型
    if isinstance(n.args[0].type, TensorType):
        # 调用 flatten_check 函数对 n.args[0].type 进行扁平化处理，返回结果赋给 output_type
        output_type = flatten_check(n.args[0].type, start_dim, end_dim)
        # 调用 get_greatest_upper_bound 函数获取 output_type 和 n.type 的最大上界，并将结果赋给 n.type
        n.type = get_greatest_upper_bound(output_type , n.type)
    
    # 返回变量 n 的类型作为函数的返回值
    return n.type
class GraphTypeChecker:
    # 初始化方法，接收环境变量和跟踪对象作为参数
    def __init__(self, env, traced):
        self.env = env  # 设置环境变量
        self.traced = traced  # 设置跟踪对象

    # 类型检查方法
    def type_check(self):
        """
        逐个节点进行渐进类型检查
        如果有任何节点未能通过类型检查，则返回 False
        """
        graph = self.traced.graph

        # 遍历图中的每个节点，进行类型检查
        for n in graph.nodes:
            self.type_check_node(n)
        return True  # 返回 True 表示所有节点均通过类型检查

    # 类型检查单个节点的方法
    def type_check_node(self, n: Node):
        """
        对给定的 FX 节点进行类型检查
        目前支持的操作包括：Reshape、Transpose、Add、Relu、conv2d、batchnorm2d、flatten、maxpool2d、adaptiveavgpool2d、linear
        """
        if n.type is None:
            n.type = Dyn  # 如果节点类型为空，设置为动态类型 Dyn

        if n.op == 'placeholder':
            return n.type  # 如果节点操作为 placeholder，则返回节点类型

        elif n.op == 'get_attr':
            t = get_parameter(self.traced, n.target)  # 获取目标节点的参数
            if isinstance(t.data, torch.Tensor):
                n.type = TensorType(t.data.shape)  # 如果参数是 torch.Tensor，则设置节点类型为对应的 TensorType
            return n.type

        elif n.op == 'call_function':
            if n.target == getattr:
                assert getattr in _INFERENCE_RULES
                return _INFERENCE_RULES[n.target](n, self.traced)  # 调用注册的推断规则进行推断

            elif n.target in _INFERENCE_RULES:
                return _INFERENCE_RULES[n.target](n)  # 调用注册的推断规则进行推断
            else:
                raise RuntimeError(f'未为目标 {n.target} 注册推断规则！')

        elif n.op == 'call_module':
            module_instance = self.traced.get_submodule(n.target)
            if type(module_instance) in _INFERENCE_RULES:
                return _INFERENCE_RULES[type(module_instance)](n, module_instance)  # 调用注册的推断规则进行推断
            else:
                raise RuntimeError(f'未为类 {type(module_instance)} 注册推断规则！')

        elif n.op == 'output':
            def get_node_type(a):
                return a.type
            n.type = torch.fx.node.map_arg(n.args[0], get_node_type)  # 设置节点类型为其参数节点的类型
            return n.type

        else:
            raise NotImplementedError(f"方法 {n.op} 尚未实现")

# 注册卷积层 Conv2d 的细化规则
@register_refinement_rule(Conv2d)
def conv_refinement_rule(n: Node):
    """
    输入和输出的第一个维度应满足相等约束
    """
    res = []
    assert isinstance(n.args[0], Node)
    arg_type = n.args[0].type
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        res = [Equality(arg_type.__args__[0], n.type.__args__[0])]
        return res

# 注册线性层 torch.nn.Linear 的细化规则
@register_refinement_rule(torch.nn.Linear)
def linear_refinement_rule(n: Node):
    """
    输入和输出的第一个维度应满足相等约束
    """
    res = []
    assert isinstance(n.args[0], Node)
    arg_type = n.args[0].type
    # 检查 arg_type 是否为 TensorType 类型，并且 n.type 也是 TensorType 类型
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        # 如果条件满足，创建一个包含单个元素的列表，元素为 arg_type 的第一个类型参数与 n.type 的第一个类型参数的相等性比较
        res = [Equality(arg_type.__args__[0], n.type.__args__[0])]
    # 返回结果列表 res
    return res
@register_refinement_rule(BatchNorm2d)
@register_refinement_rule(torch.nn.ReLU)
def all_eq(n: Node):
    """
    For operations where the input shape is equal to the output shape
    """
    # 初始化结果列表
    res = []
    # 断言第一个参数是一个节点对象
    assert isinstance(n.args[0], Node)
    # 获取第一个参数的类型
    arg_type = n.args[0].type
    # 如果第一个参数和节点类型都是 TensorType
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        # 获取参数类型的参数列表
        args1 = arg_type.__args__
        # 获取节点类型的参数列表
        args2 = n.type.__args__
        # 生成每个维度相等的约束条件
        res = [Equality(args1[i], args2[i]) for i in range(len(args1))]
    return res


@register_refinement_rule(torch.nn.AdaptiveAvgPool2d)
@register_refinement_rule(torch.nn.MaxPool2d)
def first_two_eq(n: Node):
    """
    For operations where the first two dimensions of the input and output shape
    are equal
    """
    # 初始化结果列表
    res = []
    # 断言第一个参数是一个节点对象
    assert isinstance(n.args[0], Node)
    # 获取第一个参数的类型
    arg_type = n.args[0].type
    # 如果第一个参数和节点类型都是 TensorType
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        # 获取参数类型的参数列表
        args1 = arg_type.__args__
        # 获取节点类型的参数列表
        args2 = n.type.__args__
        # 生成前两个维度相等的约束条件
        res = [Equality(args1[0], args2[0]), Equality(args1[1], args2[1])]
    return res


@register_refinement_rule(torch.add)
@register_refinement_rule(operator.add)
def element_wise_eq(n: Node):
    """
    For element-wise operations and handles broadcasting.
    Note that after applying broadcasting to the arguments
    we are able to determine if certain dimensions have not been broadcast
    if they are symbolicallu equal.

    in this case, we can establish equality between those dimensions and the
    corresponding output dimensions.

    Note that it takes two iterations for this result. One iteration to establish
    equality between certain dimensions of the operands (requiring the whole solver
    including unification) and another iteration to establish equality between the operands
    and the resulting type, requiring another round of constraint generation and unificaiton.
    """
    # 初始化结果列表
    res = []
    # 如果前两个参数都是节点对象
    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        # 获取第一个参数的类型
        arg_type1 = n.args[0].type
        # 获取第二个参数的类型
        arg_type2 = n.args[1].type
        # 如果两个参数类型都是 TensorType，并且节点类型也是 TensorType
        if isinstance(arg_type1, TensorType) and isinstance(arg_type2, TensorType) and isinstance(n.type, TensorType):
            # 对参数类型进行广播，获取广播后的参数列表
            args1, args2 = broadcast_types(arg_type1, arg_type2)
            # 确定各个参数列表中的维度相等性约束
            a1 = args1.__args__
            a2 = args2.__args__
            a3 = n.type.__args__

            # 第二次迭代，建立操作数类型维度与结果类型维度的相等性
            r = []
            for x, y, z in zip(a1, a2, a3):
                if x == y:
                    r.append(Equality(x, z))
            res = r
    return res


@register_refinement_rule(torch.flatten)
def flatten_refinement_rule(n: Node):
    """
    Generates equality constraints between the dimensions of the input and output
    that will not be involved in the flatten operation
    """
    # 断言第一个参数是一个节点对象
    assert isinstance(n.args[0], Node)
    # 初始化空列表，用于存储等式约束
    eq_const = []

    # 默认起始维度为1，即从第一个维度开始
    start_dim = 1
    # 默认结束维度为-1，表示倒数第一个维度
    end_dim = -1

    # 如果参数 n 的 args 属性中包含多于一个元素
    if len(n.args) > 1:
        # 断言第二个参数是整数
        assert isinstance(n.args[1], int)
        # 将起始维度设定为第二个参数的值
        start_dim = n.args[1]

    # 如果参数 n 的 args 属性中包含多于两个元素
    if len(n.args) > 2:
        # 断言第三个参数是整数
        assert isinstance(n.args[2], int)
        # 将结束维度设定为第三个参数的值
        end_dim = n.args[2]

    # 如果 n 的类型是 TensorType，且第一个参数的类型也是 TensorType
    if isinstance(n.type, TensorType) and isinstance(n.args[0].type, TensorType):
        # 获取 n 的类型参数的长度
        l = len(n.type.__args__)
        # 获取第一个参数的类型
        arg_type = n.args[0].type
        # 如果起始维度为 -1，则设定为 n 的类型参数的长度
        start_dim = l if start_dim == -1 else start_dim
        # 如果结束维度为负数，则进行计算，否则直接使用结束维度
        end_dim = l + end_dim + 1 if end_dim < 0 else end_dim + 1

        # 对起始维度之前的维度进行约束等式的生成
        for t1, t2 in zip(n.type.__args__[0:start_dim], arg_type.__args__[0:start_dim]):
            eq_const.append(Equality(t1, t2))

        # 对结束维度之后的维度进行约束等式的生成
        for t1, t2 in zip(n.type.__args__[end_dim:], arg_type.__args__[end_dim:]):
            eq_const.append(Equality(t1, t2))

    # 返回生成的等式约束列表
    return eq_const
# 注册一个函数作为 Conv2d 类型的代数表达式推断规则的装饰器
@register_algebraic_expressions_inference_rule(Conv2d)
def conv_rule(n: Node, module_instance):
    """
    Represents the output in terms of an algebraic expression with respect to
    the input when possible
    """
    # 断言节点的第一个参数是 Node 类型的对象
    assert isinstance(n.args[0], Node)
    # 获取第一个参数的类型
    arg_type = n.args[0].type
    # 如果第一个参数的类型是 TensorType 并且 n 的类型也是 TensorType
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        # 获取输入张量的宽度和高度
        w_in = arg_type.__args__[3]
        h_in = arg_type.__args__[2]
        # 计算输出张量的高度和宽度
        h_out = calculate_out_dimension(h_in, module_instance, 0)
        w_out = calculate_out_dimension(w_in, module_instance, 1)
        # 创建新的张量类型，表示输出的形状
        new_type = TensorType((n.type.__args__[0], n.type.__args__[1], h_out, w_out))
        # 将节点 n 的类型更新为新的类型
        n.type = new_type
        # 返回新的类型对象
        return new_type

# Refine 类，用于符号形状推断
class Refine:
    """
    Symbolic shape inference.
    Generates constraints over type variables.
    Currently all constraints are equality constraints.
    """
    def __init__(self, traced):
        # 初始化约束列表和追踪对象
        self.constraints = []
        self.traced = traced
        # 符号迭代器，用于生成唯一符号标识符
        self.symbol_iter = itertools.count(start=0, step=1)

    def refine(self):
        """
        Generates constraints for
        every node in the graph based on
        the operation.
        """
        # 获取追踪对象的计算图
        graph = self.traced.graph
        # 遍历图中的每个节点，并为每个节点生成约束条件
        for n in graph.nodes:
            self.refine_node(n)
        # 返回 True，表示约束生成完成
        return True

    def symbolic_relations(self):
        """
        Infers algebraic relations
        """
        # 获取追踪对象的计算图
        graph = self.traced.graph
        # 遍历图中的每个节点，推断代数关系
        for n in graph.nodes:
            self.infer_symbolic_relations(n)
        # 返回 True，表示代数关系推断完成
        return True

    def replace_dyn_with_fresh_var(self, typ):
        """
        Replace all unknown types with fresh type variables.
        """
        # 如果类型是 Dyn，则用新的类型变量替换
        if typ == Dyn:
            new_symbol = Var(next(self.symbol_iter))
            return new_symbol
        # 如果类型是 TensorType，则递归地用新的类型变量替换其元素
        elif isinstance(typ, TensorType):
            new_args = [self.replace_dyn_with_fresh_var(a) for a in typ.__args__]
            return TensorType(tuple(new_args))
        # 如果类型是列表，则递归地替换列表中的每个元素
        elif isinstance(typ, list):
            return [self.replace_dyn_with_fresh_var(t) for t in typ]
        # 如果类型是元组，则递归地替换元组中的每个元素
        elif isinstance(typ, tuple):
            return tuple(self.replace_dyn_with_fresh_var(t) for t in typ)
        else:
            # 对于其他类型，直接返回原始类型
            return typ

    def convert_to_sympy_symbols(self, typ):
        """
        Replace all unknown types with fresh type variables.
        """
        # 如果类型是 Var，则转换为 SymPy 符号
        if isinstance(typ, Var):
            return sympy.symbols(str(typ))
        # 如果类型是 TensorType，则递归地将其元素转换为 SymPy 符号
        elif isinstance(typ, TensorType):
            new_args = [self.convert_to_sympy_symbols(a) for a in typ.__args__]
            return TensorType(tuple(new_args))
        # 如果类型是列表，则递归地转换列表中的每个元素为 SymPy 符号
        elif isinstance(typ, list):
            return [self.convert_to_sympy_symbols(t) for t in typ]
        # 如果类型是元组，则递归地转换元组中的每个元素为 SymPy 符号
        elif isinstance(typ, tuple):
            return tuple(self.convert_to_sympy_symbols(t) for t in typ)
        else:
            # 对于其他类型，直接返回原始类型
            return typ
    # 定义一个方法用于优化节点类型信息，接受一个 Node 类型的参数 n
    def refine_node(self, n: Node):
        """
        Returns a list of equality constraints for
        call_module and call_function nodes.
        Models the relation between input and output dimensions
        using constraints in case they are both tensors.
        All operations used in resnet50 are defined.
        """
        # 如果节点类型为空，将其设为动态类型 Dyn
        if n.type is None:
            n.type = Dyn

        # 用一个新的变量替换 Dyn 类型
        n.type = self.replace_dyn_with_fresh_var(n.type)

        # 如果节点操作是 'call_function'
        if n.op == 'call_function':
            # 如果节点的目标函数在 _REFINEMENT_RULES 中
            if n.target in _REFINEMENT_RULES:
                # 将调用目标函数后得到的约束添加到 constraints 中
                self.constraints += _REFINEMENT_RULES[n.target](n)
            else:
                pass

        # 如果节点操作是 'call_module'
        if n.op == 'call_module':
            # 获取模块实例
            module_instance = self.traced.get_submodule(n.target)
            # 如果模块实例的类型在 _REFINEMENT_RULES 中
            if type(module_instance) in _REFINEMENT_RULES:
                # 将调用模块实例后得到的约束添加到 constraints 中
                self.constraints += _REFINEMENT_RULES[type(module_instance)](n)
            else:
                pass

        # 如果节点操作是 'output'
        if n.op == 'output':
            # 定义一个函数，用于获取节点的类型
            def get_node_type(a):
                return a.type
            # 将节点的类型映射为其参数的类型
            n.type = torch.fx.node.map_arg(n.args[0], get_node_type)
            # 返回节点的类型
            return n.type

        else:
            pass

    # 定义一个方法用于推断符号关系，接受一个 Node 类型的参数 n
    def infer_symbolic_relations(self, n: Node):
        # 将节点类型转换为 sympy 符号
        n.type = self.convert_to_sympy_symbols(n.type)

        # 如果节点操作是 'call_function'
        if n.op == 'call_function':
            # 如果节点的目标函数在 _RULES 中
            if n.target in _RULES:
                # 调用目标函数后返回结果
                return _RULES[n.target](n)
            else:
                pass

        # 如果节点操作是 'call_module'
        if n.op == 'call_module':
            # 获取模块实例
            module_instance = self.traced.get_submodule(n.target)
            # 如果模块实例的类型在 _RULES 中
            if type(module_instance) in _RULES:
                # 调用模块实例后返回结果
                return _RULES[type(module_instance)](n, module_instance)
            else:
                pass

        # 如果节点操作是 'output'
        if n.op == 'output':
            # 定义一个函数，用于获取节点的类型
            def get_node_type(a):
                return a.type
            # 将节点的类型映射为其参数的类型
            n.type = torch.fx.node.map_arg(n.args[0], get_node_type)
            # 返回节点的类型
            return n.type

        else:
            pass
# 根据给定的参数路径从traced对象中获取对应的torch.nn.Parameter参数。
def get_parameter(traced, target: str):
    """
    如果存在，返回由 ``target`` 指定的参数，否则抛出错误。

    有关此方法功能的更详细说明以及如何正确指定 ``target``，请参见
    ``get_submodule`` 的文档字符串。

    Args:
        target: 要查找的参数的完全限定字符串名称。
            （查看 ``get_submodule`` 以了解如何指定完全限定字符串。）

    Returns:
        torch.nn.Parameter: 由 ``target`` 引用的参数

    Raises:
        AttributeError: 如果目标字符串引用无效路径，或解析为不是
            ``nn.Parameter`` 的内容时。
    """
    # 将目标路径分解为模块路径和参数名
    module_path, _, param_name = target.rpartition(".")

    # 从traced对象中获取模块路径对应的子模块
    mod: torch.nn.Module = traced.get_submodule(module_path)

    # 如果模块没有对应的参数名，抛出属性错误异常
    if not hasattr(mod, param_name):
        raise AttributeError(mod._get_name() + " has no attribute `" + param_name + "`")

    # 获取模块中的参数对象
    param: torch.nn.Parameter = getattr(mod, param_name)

    # 返回获取到的参数对象
    return param
```