# `.\pytorch\torch\fx\experimental\migrate_gradual_types\constraint_generator.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和函数
import torch
import operator
import warnings
from typing import Callable, Dict, Iterable

# 导入具体的函数和类
from torch.fx._symbolic_trace import _assert_is_none
from torch.fx.experimental.migrate_gradual_types.constraint import ApplyBroadcasting, CalcProduct, \
    Disj, TGreatestUpperBound, CalcMaxPool, CalcConv, Conj, BinConstraintT, CanReshape, BinConstraintD, GetItem, T, F, \
    TVar, DVar, GetItemTensor, IndexSelect, Transpose, DGreatestUpperBound
from torch.fx.experimental.migrate_gradual_types.operation import \
    op_eq, op_matching, op_consistency, op_leq, op_precision, op_gt, op_div, op_sub, op_neq, op_lt, op_add, op_mul
from torch.fx.node import Target, Node
from torch.fx.experimental.migrate_gradual_types.util import gen_tensor_dims, gen_nat_constraints, gen_dvar, gen_tvar, \
    gen_bvar
from torch.fx.tensor_type import Dyn, TensorType
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d

# 初始化推理规则字典
_INFERENCE_RULES: Dict[Target, Callable] = {}

# 定义最大张量秩
MAX_TENSOR_RANK = 4

# 注册推理规则的装饰器函数
def register_inference_rule(call_target):
    def register(fn):
        # 检查是否已经注册过该调用目标的推理规则
        if call_target in _INFERENCE_RULES:
            raise RuntimeError(f'Inference rule already registered for {call_target}!')
        # 将推理函数注册到规则字典中
        _INFERENCE_RULES[call_target] = fn
        return fn
    return register

# 生成展平约束条件的函数
def generate_flatten_constraints(start_dim, end_dim, input, flattened, n, counter):
    # 生成张量维度变量
    d, counter = gen_tensor_dims(n, counter)
    # 创建输入张量类型约束
    c1 = BinConstraintT(input, TensorType(d), op_eq)
    # 计算展平操作约束
    start_dim = n if start_dim == -1 else abs(start_dim)
    end_dim = n + end_dim + 1 if end_dim < 0 else end_dim + 1
    c2 = CalcProduct(start_dim, end_dim, flattened, d)
    # 生成自然约束条件
    nat_constraints = gen_nat_constraints(d)
    # 返回联合约束条件和更新后的计数器
    return Conj([c1, c2, *nat_constraints]), counter

# 注册getattr函数的推理规则
@register_inference_rule(getattr)
def get_attr_inference_rule(n: Node, symbols, constraints, counter):
    """
    If the attribute is "device" then the tensor shape is preserved
    """
    # 断言确保第一个参数是Node类型，第二个参数是字符串类型
    assert isinstance(n.args[0], Node)
    assert isinstance(n.args[1], str)
    
    # 生成新的张量类型变量
    output, counter = gen_tvar(counter)
    symbols[n] = output

    # 获取getattr的输入和属性名
    input = symbols[n.args[0]]
    attr = n.args[1]

    # 如果属性是'device'，则返回输入与输出类型相等的约束条件
    if attr == 'device':
        return [BinConstraintT(input, output, op_eq)], counter
    else:
        # 如果属性未实现，则抛出NotImplementedError
        raise NotImplementedError('Not yet implemented')

# 注册torch.bmm函数的推理规则
@register_inference_rule(torch.bmm)
def bmm_inference_rule(n: Node, symbols, constraints, counter):
    """
    Constraints that match the input to a size 3 tensor
    and switch the dimensions according to the rules
    of batch multiplication
    """
    # 断言确保前两个参数都是Node类型
    assert isinstance(n.args[0], Node)
    assert isinstance(n.args[1], Node)

    # 生成新的张量类型变量
    bmm_output, counter = gen_tvar(counter)
    symbols[n] = bmm_output

    # 获取torch.bmm的两个输入节点的类型变量
    bmm_input1 = symbols[n.args[0]]
    bmm_input2 = symbols[n.args[1]]

    # 生成大小为3的张量维度变量
    dims_input1, counter = gen_tensor_dims(3, counter)
    dims_input2, counter = gen_tensor_dims(3, counter)
    # 定义包含动态约束的输入变量列表
    inputs_dyn = Conj([
        # 将bmm_input1约束为动态类型Dyn
        BinConstraintT(bmm_input1, Dyn, op_eq),
        # 将bmm_input2约束为动态类型Dyn
        BinConstraintT(bmm_input2, Dyn, op_eq),
        # 将bmm_output约束为动态类型Dyn
        BinConstraintT(bmm_output, Dyn, op_eq)
    ])
    
    # 定义包含部分动态约束的输入1变量列表
    input1_dyn = Conj([
        # 将bmm_input1约束为动态类型Dyn
        BinConstraintT(bmm_input1, Dyn, op_eq),
        # 将bmm_input2约束为指定维度的TensorType
        BinConstraintT(bmm_input2, TensorType(dims_input2), op_eq),
        # 将bmm_output约束为具有部分动态维度的TensorType
        BinConstraintT(bmm_output, TensorType([dims_input2[0], Dyn, dims_input2[2]]), op_eq)
    ])
    
    # 定义包含部分动态约束的输入2变量列表
    input2_dyn = Conj([
        # 将bmm_input2约束为动态类型Dyn
        BinConstraintT(bmm_input2, Dyn, op_eq),
        # 将bmm_input1约束为指定维度的TensorType
        BinConstraintT(bmm_input1, TensorType(dims_input1), op_eq),
        # 将bmm_output约束为具有部分动态维度的TensorType
        BinConstraintT(bmm_output, TensorType([dims_input1[0], dims_input1[1], Dyn]), op_eq)
    ])
    
    # 定义一致性约束列表，确保dims_input1[0]与dims_input2[0]一致
    consistency_constraints = [BinConstraintD(dims_input1[0], dims_input2[0], op_consistency)]
    
    # 生成动态变量batch_size和计数器counter
    batch_size, counter = gen_dvar(counter)
    
    # 定义输入变量都为张量的约束条件列表
    inputs_are_tensors = Conj([
        # 将bmm_input1约束为指定维度的TensorType
        BinConstraintT(bmm_input1, TensorType(dims_input1), op_eq),
        # 将bmm_input2约束为指定维度的TensorType
        BinConstraintT(bmm_input2, TensorType(dims_input2), op_eq),
        # 将bmm_output约束为指定维度的TensorType，其中batch_size为动态维度
        BinConstraintT(bmm_output, TensorType([batch_size, dims_input1[1], dims_input2[2]]), op_eq),
        # 添加所有一致性约束
        *consistency_constraints,
        # 保证batch_size是dims_input1[0]与dims_input2[0]的最大上界
        DGreatestUpperBound(batch_size, dims_input1[0], dims_input2[0])
    ])
    
    # 返回包含不同约束的多个逻辑语句的列表及计数器
    return [Disj([inputs_dyn, input1_dyn, input2_dyn, inputs_are_tensors])], counter
# 注册推断规则的装饰器，指定函数名称为 "index_select"
@register_inference_rule("index_select")
def index_select_inference_rule(n: Node, symbols, constraints, counter):
    """
    We constrain the second argument to a vector or Dyn.
    The output replaces the input with the shape of the vector
    at the position given by the index (first argument)
    """
    # 确认第一个参数是节点对象
    assert isinstance(n.args[0], Node)
    # 确认第二个参数是整数
    assert isinstance(n.args[1], int)
    # 确认第三个参数是节点对象

    # 生成一个类型变量和一个计数器
    index_select, counter = gen_tvar(counter)
    # 将当前节点 n 对应的符号映射为 index_select
    symbols[n] = index_select

    # 生成一个维度列表，长度为 1
    dims, counter = gen_tensor_dims(1, counter)

    # 创建一个等式约束，约束 symbols[n.args[2]] 的类型为 TensorType(dims)
    is_size_1 = BinConstraintT(symbols[n.args[2]], TensorType(dims), op_eq)
    # 创建一个等式约束，约束 symbols[n.args[2]] 的类型为 Dyn
    is_dyn = BinConstraintT(symbols[n.args[2]], Dyn, op_eq)

    # 创建一个复合约束 c2，包含 is_size_1 和一系列 IndexSelect 约束
    c2 = Conj([is_size_1, Disj([IndexSelect(i + 1, symbols[n.args[0]], dims[0], n.args[1], index_select)
                                for i in range(MAX_TENSOR_RANK)])])
    # 创建一个复合约束 c3，包含 is_dyn 和一系列 IndexSelect 约束
    c3 = Conj([is_dyn, Disj([IndexSelect(i + 1, symbols[n.args[0]], Dyn, n.args[1], index_select)
                             for i in range(MAX_TENSOR_RANK)])])

    # 返回由 c2 和 c3 构成的析取约束列表，以及更新后的计数器
    return [Disj([c2, c3])], counter


# 注册推断规则的装饰器，指定函数名称为 "expand"
@register_inference_rule("expand")
def expand_inference_rule(n: Node, symbols, constraints, counter):
    """
    We generate the exact constraints as we do for tensor additions but we constraint
    the rank of this expression to be equal to len(n.args[1:]) so that only
    those cases get considered for the output
    """
    # 确认第一个参数是节点对象
    assert isinstance(n.args[0], Node)

    # 生成一个类型变量和一个计数器
    expand, counter = gen_tvar(counter)
    # 将当前节点 n 对应的符号映射为 expand
    symbols[n] = expand

    # 将 symbols[n.args[0]] 的值赋给 e1
    e1 = symbols[n.args[0]]
    # 生成一个类型变量和一个计数器，用于 e2
    e2, counter = gen_tvar(counter)

    # 生成 e2 的自然数约束列表
    e2_nat_constraints = []
    for arg in n.args[1:]:
        # 确认每个参数是节点对象或整数
        assert isinstance(arg, (Node, int))
        if isinstance(arg, Node):
            # 确认 symbols[arg] 是 DVar 类型
            assert isinstance(symbols[arg], DVar)
            # 添加 e2 的二元约束，限制其小于等于 symbols[arg]
            e2_nat_constraints.append(BinConstraintD(0, symbols[arg], op_leq))

    # 创建 e2 的类型约束，限制其为 TensorType([arg if isinstance(arg, int) else symbols[arg] for arg in n.args[1:]])
    e2_constraint = BinConstraintT(e2, TensorType([arg if isinstance(arg, int) else symbols[arg] for arg in n.args[1:]]), op_eq)

    # 生成广播约束，并更新约束列表和计数器
    constraints, counter = gen_broadcasting_constraints(e1, e2, symbols, counter, expand)

    # 生成长度为 len(n.args[1:]) 的维度列表，并生成对应的自然数约束列表
    dims, counter = gen_tensor_dims(len(n.args[1:]), counter)
    nat_constraints = gen_nat_constraints(dims)
    # 创建约束列表 c，包含 expand 的类型约束、自然数约束、e2 的类型约束和 e2 的二元约束
    c = [BinConstraintT(expand, TensorType(dims), op_eq), *nat_constraints, e2_constraint, *e2_nat_constraints]
    # 将约束列表 c 添加到现有约束列表中
    constraints += c

    # 返回更新后的约束列表和计数器
    return constraints, counter


# 注册推断规则的装饰器，指定函数名称为 torch.nn.functional.gelu
@register_inference_rule(torch.nn.functional.gelu)
# 同样方式注册其他函数的推断规则
@register_inference_rule(torch.nn.functional.dropout)
@register_inference_rule(torch.nn.functional.softmax)
@register_inference_rule("detach")
@register_inference_rule("to")
@register_inference_rule("int")
@register_inference_rule("long")
@register_inference_rule("contiguous")
@register_inference_rule(torch.ones)
@register_inference_rule(torch.zeros)
def equality_inference_rule(n: Node, symbols, constraints, counter):
    """
    """
    We generate the constraint: input = output
    """
    # 生成约束条件：input = output

    output, counter = gen_tvar(counter)
    # 调用 gen_tvar 函数生成一个新的类型变量 output，并更新计数器 counter
    symbols[n] = output
    # 将节点 n 对应的符号映射为 output

    if isinstance(n.args[0], Node):
        # 如果 n 的第一个参数是 Node 类型
        input = symbols[n.args[0]]
        # 将输入指定为节点 n 的第一个参数对应的符号

        if isinstance(input, TVar):
            # 如果输入是类型变量 TVar
            return [BinConstraintT(input, output, op_eq)], counter
            # 返回一个包含 input = output 的二元约束条件列表，并返回计数器 counter

        # then we have dimension variables
        else:
            # 否则，我们有维度变量
            for arg in n.args:
                assert isinstance(symbols[arg], DVar)
            # 遍历 n 的所有参数，确保它们都是维度变量 DVar
            my_size = [symbols[arg] for arg in n.args]
            # 生成参数列表的符号表示 my_size
            return [BinConstraintT(output, TensorType(my_size), op_eq)], counter
            # 返回一个包含 output = TensorType(my_size) 的二元约束条件列表，并返回计数器 counter

    elif isinstance(n.args[0], tuple):
        # 如果 n 的第一个参数是元组
        # then the tuple is the size
        assert len(n.args[0]) <= 4
        # 断言元组长度不超过 4
        my_size = [symbols[arg] for arg in n.args[0]]
        # 生成元组中每个元素的符号表示 my_size
        return [BinConstraintT(output, TensorType(my_size), op_eq)], counter
        # 返回一个包含 output = TensorType(my_size) 的二元约束条件列表，并返回计数器 counter

    else:
        # 否则，抛出未实现的错误
        raise NotImplementedError('Method not yet implemented')
@register_inference_rule("transpose")
def transpose_inference_rule(n: Node, symbols, constraints, counter):
    """
    注册推理规则为 "transpose"
    """
    assert isinstance(n.args[0], Node)
    assert isinstance(n.args[1], int)
    assert isinstance(n.args[2], int)

    output, counter = gen_tvar(counter)
    symbols[n] = output

    from_arg = symbols[n.args[0]]
    assert isinstance(from_arg, TVar)

    # 输入和输出都是动态类型
    is_dyn = Conj([BinConstraintT(from_arg, Dyn, op_eq), BinConstraintT(output, Dyn, op_eq)])

    # 或者输入是张量，实际执行置换
    c3 = Disj([Transpose(i + 1, from_arg, n.args[1], n.args[2], output) for i in range(MAX_TENSOR_RANK)])

    return [Disj([is_dyn, c3])], counter


@register_inference_rule("type_as")
def type_inference_rule(n: Node, symbols, constraints, counter):
    """
    生成约束条件：输入等于输出
    """
    assert isinstance(n.args[0], Node)
    assert isinstance(n.args[1], Node)

    output, counter = gen_tvar(counter)
    symbols[n] = output

    from_arg = symbols[n.args[0]]
    to_arg = symbols[n.args[1]]

    assert isinstance(from_arg, TVar)
    assert isinstance(to_arg, TVar)

    return [BinConstraintT(from_arg, to_arg, op_consistency),
            BinConstraintT(output, to_arg, op_eq)], counter


@register_inference_rule("masked_fill_")
def masked_fill_inference_rule(n: Node, symbols, constraints, counter):
    """
    类似于加法。目前我们实现了当参数是布尔张量时的约束条件。还有一种情况是条件的情况，我们暂时不处理。
    """
    assert isinstance(n.args[0], Node)
    assert isinstance(n.args[1], Node)

    # 从符号表中获取类型变量，并确认它们是张量变量
    e1 = symbols[n.args[0]]
    e2 = symbols[n.args[1]]

    if isinstance(e1, TVar) and isinstance(e2, TVar):
        masked_fill_tensor, counter = gen_tvar(counter)
        symbols[n] = masked_fill_tensor
        return gen_broadcasting_constraints(e1, e2, symbols, counter, masked_fill_tensor)
    else:
        raise NotImplementedError('Not yet implemented')


@register_inference_rule(torch.nn.functional.embedding)
def embedding_inference_rule_functional(n: Node, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)

    embedding_dim_weights = symbols[n.args[1]]

    # 将其视为静态形状。因此，我们不使用匹配。
    weight_dims, counter = gen_tensor_dims(2, counter)
    equality_constraint = BinConstraintT(embedding_dim_weights, TensorType(weight_dims), op_eq)
    embedding_dim = weight_dims[1]
    constraints, counter = gen_embedding_rules(n, symbols, embedding_dim, counter)
    return [equality_constraint] + constraints, counter


@register_inference_rule(torch.nn.modules.sparse.Embedding)



注释：
- `@register_inference_rule("transpose")`：注册推理规则为 "transpose"。
- `def transpose_inference_rule(n: Node, symbols, constraints, counter):`：定义推理规则函数，接受节点 `n`，符号表 `symbols`，约束列表 `constraints` 和计数器 `counter` 作为参数。
- `""" Can be considered as a sequence of two index selects, so we generate constraints accordingly """`：可以视为两个索引选择的序列，因此我们相应地生成约束条件。
- `assert isinstance(n.args[0], Node)`：断言 `n.args[0]` 是一个节点对象。
- `assert isinstance(n.args[1], int)`：断言 `n.args[1]` 是一个整数。
- `assert isinstance(n.args[2], int)`：断言 `n.args[2]` 是一个整数。
- `output, counter = gen_tvar(counter)`：生成一个新的类型变量并更新计数器。
- `symbols[n] = output`：将节点 `n` 对应的符号设置为 `output`。
- `from_arg = symbols[n.args[0]]`：获取节点 `n` 的第一个参数对应的符号。
- `assert isinstance(from_arg, TVar)`：断言 `from_arg` 是一个类型变量。
- `is_dyn = Conj([BinConstraintT(from_arg, Dyn, op_eq), BinConstraintT(output, Dyn, op_eq)])`：生成输入和输出都是动态类型的约束条件。
- `c3 = Disj([Transpose(i + 1, from_arg, n.args[1], n.args[2], output) for i in range(MAX_TENSOR_RANK)])`：根据最大张量秩生成置换约束条件的析取列表。
- `return [Disj([is_dyn, c3])], counter`：返回由动态类型和置换约束条件构成的析取列表，以及更新后的计数器。

（后续代码均类似，注释采用相同格式解释每一行的作用）
def embedding_inference_rule(n: Node, module_instance, symbols, constraints, counter):
    """
    The output shape differs from the input shape in the last dimension
    """
    assert isinstance(n.args[0], Node)
    # 生成嵌入规则并返回
    return gen_embedding_rules(n, symbols, module_instance.embedding_dim, counter)


def gen_embedding_rules(n: Node, symbols, embedding_dim, counter):
    """
    Generates embedding rules for given node `n`.

    Args:
        n (Node): The node for which embedding rules are generated.
        symbols (dict): Symbol table mapping nodes to their respective symbols.
        embedding_dim (int): Dimensionality of the embedding.
        counter (int): Counter for generating unique variables.

    Returns:
        list: List containing constraints for embedding rules.
        int: Updated counter value after generating constraints.
    """

    embedding_output, counter = gen_tvar(counter)
    symbols[n] = embedding_output
    embedding_input = symbols[n.args[0]]

    input_dyn = BinConstraintT(embedding_input, Dyn, op_eq)
    output_dyn = BinConstraintT(embedding_output, Dyn, op_eq)

    c1 = Conj([input_dyn, output_dyn])
    c2 = []

    for i in range(1, MAX_TENSOR_RANK):
        new_dims, counter = gen_tensor_dims(i, counter)
        nat_constraints = gen_nat_constraints(new_dims)

        # we consider all tensor sizes and append embedding_dim to the end of the output dimension in all cases
        c_tensor_i = Conj([BinConstraintT(embedding_input, TensorType(new_dims), op_eq),
                           BinConstraintT(embedding_output, TensorType(new_dims + [embedding_dim]), op_eq)] +
                          nat_constraints)
        c2.append(c_tensor_i)

    return [Disj([c1, Disj(c2)])], counter


@register_inference_rule(torch.tensor)
def tensor_inference_rule(n: Node, symbols, constraints, counter):
    """
    If the tensor is a scalar, we will skip it since we
    do not support scalars yet. We will add support in the future
    if it's needed. For our examples so far, scalars are not needed.
    """
    return [], counter


@register_inference_rule("reshape")
@register_inference_rule("view")
def view_inference_rule(n: Node, symbols, constraints, counter):
    """
    Similar to reshape but with an extra condition on the strides
    """
    assert isinstance(n.args[0], Node)
    # 生成新变量作为视图
    my_view, counter = gen_tvar(counter)
    symbols[n] = my_view

    src_var = symbols[n.args[0]]
    t2 = [symbols[elem] if isinstance(elem, Node) else elem for elem in n.args[1:]]  # target shape
    t2_type = []
    num_constraints = []

    for t in t2:
        if t == -1:
            var, counter = gen_dvar(counter)
            t2_type.append(var)
            num_constraints.append(BinConstraintD(var, Dyn, op_neq))
        else:
            num_constraints.append(BinConstraintD(t, Dyn, op_neq))
            t2_type.append(t)

    t2_type = TensorType(t2_type)  # type: ignore[assignment]

    c1 = BinConstraintT(my_view, t2_type, op_eq)
    c2 = CanReshape(src_var, t2_type)

    # TODO: add the extra check mentioned here:
    # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view

    return [c1, c2] + num_constraints, counter  # type: ignore[operator]


@register_inference_rule("size")
def size_inference_rule(n: Node, symbols, constraints, counter):
    """
    The constraint is just lhs = rhs.
    Ex: size = input_ids.size()
    """
    # 待实现，此处应添加代码实现size规则的推断
    if len(n.args) == 1:
        # 如果节点 n 的参数数量为 1，执行以下操作：
        # 生成新变量 size 和 counter
        size, counter = gen_tvar(counter)
        # 将节点 n 添加到 symbols 字典中，并关联到 size
        symbols[n] = size
        # 获取 n 的第一个参数对应的符号变量
        input = symbols[n.args[0]]
        # 创建一个二进制约束条件，要求 input 的大小等于 size
        c = BinConstraintT(input, size, op_eq)
        # 返回包含约束 c 和 counter 的列表
        return [c], counter

    elif len(n.args) == 2:
        # 如果节点 n 的参数数量为 2，执行以下操作：
        # TODO: review this rule; should input = dyn; output = dyn be included here?
        # 检查第二个参数是否为整数
        if isinstance(n.args[1], int):
            # 生成新的动态变量 size_index 和 counter
            size_index, counter = gen_dvar(counter)
            # 将节点 n 添加到 symbols 字典中，并关联到 size_index
            symbols[n] = size_index
            # 获取 n 的第一个参数对应的符号变量
            input = symbols[n.args[0]]
            # 创建一组 GetItem 约束条件，将其保存在列表 c2 中
            c2 = [GetItem(i + 1, n.args[1], size_index, input) for i in range(MAX_TENSOR_RANK)]
            # 创建一个二进制约束条件，要求 size_index <= 0
            c3 = BinConstraintD(0, size_index, op_leq)
            
            # 创建 input_dyn 和 output_dyn 的二进制约束条件
            input_dyn = BinConstraintT(input, Dyn, op_eq)
            output_dyn = BinConstraintD(size_index, Dyn, op_eq)
            c1 = Conj([input_dyn, output_dyn])  # 创建 c1，包含 input_dyn 和 output_dyn

            # 返回包含 Disj([c1, Conj([Disj(c2), c3])]) 和 counter 的列表
            return [Disj([c1, Conj([Disj(c2), c3])])], counter

        else:
            # 如果第二个参数不是整数，抛出未实现错误
            raise NotImplementedError

    else:
        # 如果节点 n 的参数数量既不是 1 也不是 2，抛出未实现错误
        raise NotImplementedError
@register_inference_rule(torch.cumsum)
def cumsum_inference_rule(n: Node, symbols, constraints, counter):
    """
    Input and output shapes should be equal
    We should verify that the index is valid
    """
    # 断言第一个参数是Node类型
    assert isinstance(n.args[0], Node)
    # 如果参数个数大于1，则使用第二个参数；否则使用关键字参数"dim"
    arg_1 = n.args[1] if len(n.args) > 1 else n.kwargs["dim"]
    # 断言第二个参数是整数类型
    assert isinstance(arg_1, int)

    # 生成新的类型变量和计数器
    output, counter = gen_tvar(counter)
    symbols[n] = output
    # 获取输入符号
    input = symbols[n.args[0]]

    # 创建输入动态约束
    input_dyn = BinConstraintT(input, Dyn, op_eq)
    # 创建输出动态约束
    output_dyn = BinConstraintT(output, Dyn, op_eq)
    # 构建第一个约束：输入和输出的动态约束
    c1 = Conj([input_dyn, output_dyn])

    c2 = []
    # 遍历1到最大张量秩的范围
    for i in range(1, MAX_TENSOR_RANK + 1):
        # 生成张量维度和新的计数器
        new_dims, counter = gen_tensor_dims(i, counter)
        # 生成自然数约束
        nat_constraints = gen_nat_constraints(new_dims)

        # 创建张量约束：输入和输出的张量类型相等，以及索引范围检查和自然数约束
        c_tensor_i = Conj([BinConstraintT(input, TensorType(new_dims), op_eq),
                           BinConstraintT(output, TensorType(new_dims), op_eq)] +
                          [range_check(arg_1, i)] + nat_constraints)

        c2.append(c_tensor_i)

    # 创建动态或张量的析取约束
    dyn_or_tensor = Disj([c1, Disj(c2)])
    return [dyn_or_tensor], counter


@register_inference_rule(_assert_is_none)
def assert_inference_rule(n: Node, symbols, constraints, counter):
    # 断言节点没有用户
    assert len(n.users) == 0
    # 返回空约束列表和计数器
    return [], counter


@register_inference_rule(operator.getitem)
def getitem_inference_rule(n: Node, symbols, constraints, counter):
    # 断言第一个参数是Node类型
    assert isinstance(n.args[0], Node)

    # 如果第二个参数是整数，处理维度输出情况
    if isinstance(n.args[1], int):
        # 生成新的维度变量和计数器
        get_item_output, counter = gen_dvar(counter)
        symbols[n] = get_item_output

        # 获取getitem的参数变量
        get_item_arg = symbols[n.args[0]]
        assert isinstance(get_item_arg, TVar)

        # 如果输入是动态的，接受任何索引并返回动态维度作为输出
        input_dyn = BinConstraintT(get_item_arg, Dyn, op_eq)
        output_dyn = BinConstraintD(get_item_output, Dyn, op_eq)
        c1 = Conj([input_dyn, output_dyn])

        # 如果输入是张量，生成基于张量维度扩展的getitem约束
        c2 = [GetItem(i + 1, n.args[1], get_item_output, get_item_arg) for i in range(MAX_TENSOR_RANK)]

        # 输出是维度，确保它是自然数并添加为c2析取的一个合取
        c3 = BinConstraintD(0, get_item_output, op_leq)
        return [Disj([c1, Conj([Disj(c2), c3])])], counter

    # 张量输出情况
    elif isinstance(n.args[1], tuple):
        # 如果 n.args[1] 是一个元组，则执行以下代码块
        # 创建并存储新的张量变量
        get_item_output, counter = gen_tvar(counter)
        symbols[n] = get_item_output

        # 获取参数变量
        if n.args[0] in symbols:
            # 如果 n.args[0] 在符号表中已存在
            get_item_arg = symbols[n.args[0]]
            assert isinstance(get_item_arg, TVar)

            # 创建输入和输出动态约束
            input_dyn = BinConstraintT(get_item_arg, Dyn, op_eq)
            output_dyn = BinConstraintT(get_item_output, Dyn, op_eq)  # type: ignore[assignment]
            c1 = Conj([input_dyn, output_dyn])

            # 生成多个获取张量项的约束
            c2 = [GetItemTensor(i + 1, n.args[1], get_item_output, get_item_arg)  # type: ignore[misc]
                  for i in range(MAX_TENSOR_RANK)]
        else:
            # 如果 n.args[0] 不存在于符号表中，则记录错误并返回空列表和计数器
            # TODO: 我们应该弄清楚为什么会出现键错误（KeyError）
            return [], counter

        # 返回包含约束的列表和更新后的计数器
        return [Disj([c1, *c2])], counter

    else:
        # 如果 n.args[1] 不是元组，则抛出运行时错误
        raise RuntimeError('Method not yet implemented')
@register_inference_rule(operator.gt)
# 定义针对大于操作符的推断规则注册函数
def gt_inference_rule(n: Node, symbols, constraints, counter):
    # 断言第一个操作数是Node类型或整数类型
    assert isinstance(n.args[0], (Node, int))
    # 断言第二个操作数是Node类型或整数类型

    assert isinstance(n.args[1], (Node, int))

    # 确保此节点不会再次使用。我们不生成关于该节点的约束，只生成操作数的约束。

    # 如果第一个操作数是Node类型，将其解析为相应的符号
    e1 = symbols[n.args[0]] if isinstance(n.args[0], Node) else n.args[0]
    # 如果第二个操作数是Node类型，将其解析为相应的符号
    e2 = symbols[n.args[1]] if isinstance(n.args[1], Node) else n.args[1]

    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        if isinstance(e1, TVar) and isinstance(e2, TVar):
            # 生成新的Tensor变量和广播约束
            gt_tensor, counter = gen_tvar(counter)
            symbols[n] = gt_tensor
            return gen_broadcasting_constraints(e1, e2, symbols, counter, gt_tensor)

        elif isinstance(e1, DVar) and isinstance(e2, DVar):
            # 仅用于流分析
            gt_constraint = BinConstraintD(e1, e2, op_gt)

            my_gt, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_gt, gt_constraint, op_eq)
            return [equality_constraint], counter

        else:
            # 类型不匹配错误
            raise RuntimeError('Sort Mismatch')

    elif isinstance(n.args[0], Node) and not isinstance(n.args[1], Node):
        if isinstance(e1, DVar):
            # 仅用于流分析
            gt_constraint = BinConstraintD(e1, e2, op_gt)

            my_gt, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_gt, gt_constraint, op_eq)
            return [equality_constraint], counter

        elif isinstance(e1, TVar) and isinstance(e2, int):
            # 关于错误的假设的警告
            warnings.warn(f'Made the wrong assumption for node {n}. Correctness not guaranteed.')

            # 生成新的数据变量并更新符号表
            new_e1, counter = gen_dvar(counter)
            symbols[n.args[0]] = new_e1

            # 生成大于约束
            gt_constraint = BinConstraintD(new_e1, e2, op_gt)

            my_gt, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_gt, gt_constraint, op_eq)
            return [equality_constraint], counter

        else:
            # 方法尚未实现的错误
            raise NotImplementedError('Method not yet implemented')

    else:
        # 方法尚未实现的错误
        raise NotImplementedError('Method not yet implemented')


@register_inference_rule(operator.eq)
# 定义针对等于操作符的推断规则注册函数
def eq_inference_rule(n: Node, symbols, constraints, counter):
    # 断言第一个操作数是Node类型或整数类型
    assert isinstance(n.args[0], (Node, int))
    # 断言第二个操作数是Node类型或整数类型

    assert isinstance(n.args[1], (Node, int))

    # 如果第一个操作数是Node类型，将其解析为相应的符号
    e1 = symbols[n.args[0]] if isinstance(n.args[0], Node) else n.args[0]
    # 如果第二个操作数是Node类型，将其解析为相应的符号
    e2 = symbols[n.args[1]] if isinstance(n.args[1], Node) else n.args[1]
    # 检查第一个参数和第二个参数是否都是 Node 对象
    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        # 如果 e1 和 e2 都是 TVar 类型的变量
        if isinstance(e1, TVar) and isinstance(e2, TVar):
            # 生成一个新的类型变量和计数器，并将其映射到 symbols 中
            eq_tensor, counter = gen_tvar(counter)
            symbols[n] = eq_tensor
            # 生成广播约束并返回
            return gen_broadcasting_constraints(e1, e2, symbols, counter, eq_tensor)

        # 如果 e1 和 e2 都是 DVar 类型的变量
        elif isinstance(e1, DVar) and isinstance(e2, DVar):
            # 创建一个二元约束对象 BinConstraintD，表示 e1 等于 e2
            eq_constraint = BinConstraintD(e1, e2, op_eq)

            # 生成一个新的布尔变量和计数器
            my_eq, counter = gen_bvar(counter)
            # 创建一个新的二元约束，表示 my_eq 等于之前的约束 eq_constraint
            equality_constraint = BinConstraintD(my_eq, eq_constraint, op_eq)
            # 返回包含这个约束的列表和更新后的计数器
            return [equality_constraint], counter

        else:
            # 如果 e1 和 e2 的类型不匹配，抛出运行时错误
            raise RuntimeError('Sort Mismatch')

    # 如果第一个参数是 Node 对象而第二个参数不是
    elif isinstance(n.args[0], Node) and not isinstance(n.args[1], Node):
        # 如果 e1 是 DVar 类型的变量
        if isinstance(e1, DVar):
            # 创建一个二元约束对象 BinConstraintD，表示 e1 等于 e2
            eq_constraint = BinConstraintD(e1, e2, op_eq)

            # 生成一个新的布尔变量和计数器
            my_eq, counter = gen_bvar(counter)
            # 创建一个新的二元约束，表示 my_eq 等于之前的约束 eq_constraint
            equality_constraint = BinConstraintD(my_eq, eq_constraint, op_eq)
            # 返回包含这个约束的列表和更新后的计数器
            return [equality_constraint], counter
        else:
            # 如果不支持这种情况，抛出未实现错误
            raise NotImplementedError('Method not yet implemented')
    else:
        # 如果以上情况都不符合，抛出未实现错误
        raise NotImplementedError('Method not yet implemented')
@register_inference_rule(operator.ne)
def neq_inference_rule(n: Node, symbols, constraints, counter):
    """
    Translates to inconsistent in gradual types.
    To prove inequality, we should prove that
    tensors are either different sizes or
    disagree on at least one dimension

    This is a WIP (works when the condition
    is false. We are working on making this operation work
    when the condition is true as well)
    """
    # 确保 n.args[0] 是 Node 类型的对象
    assert isinstance(n.args[0], Node)
    # 确保 n.args[1] 是一个元组
    assert isinstance(n.args[1], tuple)

    # 对于长度为 3 的情况进行实现
    if len(n.args[1]) == 3:
        # 确保 n.args[1][0] 是 Node 或 int 类型的对象
        assert isinstance(n.args[1][0], (Node, int))
        # 确保 n.args[1][1] 是 Node 或 int 类型的对象
        assert isinstance(n.args[1][1], (Node, int))
        # 确保 n.args[1][2] 是 Node 或 int 类型的对象
        assert isinstance(n.args[1][2], (Node, int))

        # 获取 symbols 中 n.args[0] 对应的符号
        lhs = symbols[n.args[0]]

        # 生成一个新的张量维度 b，并更新计数器
        b, counter = gen_tensor_dims(4, counter)
        # 创建一个等式约束，将输入标记为大小为 [b[0], b[1], b[2]]
        input_is_size3 = BinConstraintT(lhs, TensorType([b[0], b[1], b[2]]), op_eq)

        # 确定维度 d1, d2, d3 的具体值或符号
        d1 = n.args[1][0] if isinstance(n.args[1][0], int) else symbols[n.args[1][0]]
        d2 = n.args[1][1] if isinstance(n.args[1][1], int) else symbols[n.args[1][1]]
        d3 = n.args[1][2] if isinstance(n.args[1][2], int) else symbols[n.args[1][2]]

        # 创建维度不相等的约束
        my_ne, counter = gen_bvar(counter)
        neq_1 = BinConstraintD(d1, b[0], op_neq)
        neq_2 = BinConstraintD(d2, b[1], op_neq)
        neq_3 = BinConstraintD(d3, b[2], op_neq)

        # 创建维度不一致的约束组合
        dims_inconsistent1 = Conj([BinConstraintD(d1, Dyn, op_neq), BinConstraintD(b[0], Dyn, op_neq), neq_1])
        dims_inconsistent2 = Conj([BinConstraintD(d2, Dyn, op_neq), BinConstraintD(b[1], Dyn, op_neq), neq_2])
        dims_inconsistent3 = Conj([BinConstraintD(d3, Dyn, op_neq), BinConstraintD(b[2], Dyn, op_neq), neq_3])
        dims_inconsistent = Disj([dims_inconsistent1, dims_inconsistent2, dims_inconsistent3])

        # 创建维度不一致的合并约束
        ne_constraint = Conj([input_is_size3, dims_inconsistent])

        # 创建一个新的布尔变量，用于表示不相等关系
        my_ne, counter = gen_bvar(counter)
        equality_constraint = BinConstraintD(my_ne, ne_constraint, op_eq)
    # 如果参数 n 的第二个元素长度为 4
    elif len(n.args[1]) == 4:

        # 确保 n.args[1][0] 是 Node 或者整数类型
        assert isinstance(n.args[1][0], (Node, int))
        # 确保 n.args[1][1] 是 Node 或者整数类型
        assert isinstance(n.args[1][1], (Node, int))
        # 确保 n.args[1][2] 是 Node 或者整数类型
        assert isinstance(n.args[1][2], (Node, int))
        # 确保 n.args[1][3] 是 Node 或者整数类型
        assert isinstance(n.args[1][3], (Node, int))

        # 获取 symbols 字典中 n.args[0] 对应的符号对象
        lhs = symbols[n.args[0]]

        # 使用 gen_dvar 函数生成新的动态变量 b1, b2, b3, b4，并更新 counter
        b1, counter = gen_dvar(counter)
        b2, counter = gen_dvar(counter)
        b3, counter = gen_dvar(counter)
        b4, counter = gen_dvar(counter)

        # 创建一个 BinConstraintT 对象，表示 lhs 与 TensorType([b1, b2, b3, b4]) 相等的约束条件
        input_is_size4 = BinConstraintT(lhs, TensorType([b1, b2, b3, b4]), op_eq)

        # 如果 n.args[1][0] 是整数，则直接使用它；否则从 symbols 字典中获取对应的符号对象
        d1 = n.args[1][0] if isinstance(n.args[1][0], int) else symbols[n.args[1][0]]
        # 类似地处理 n.args[1][1]
        d2 = n.args[1][1] if isinstance(n.args[1][1], int) else symbols[n.args[1][1]]
        # 类似地处理 n.args[1][2]
        d3 = n.args[1][2] if isinstance(n.args[1][2], int) else symbols[n.args[1][2]]
        # 类似地处理 n.args[1][3]
        d4 = n.args[1][3] if isinstance(n.args[1][3], int) else symbols[n.args[1][3]]

        # 创建 BinConstraintD 对象，表示 d1, d2, d3, d4 与 b1, b2, b3, b4 不相等的约束条件
        # 这些约束分别为 dimensions not equal
        neq_1 = BinConstraintD(d1, b1, op_neq)
        neq_2 = BinConstraintD(d2, b2, op_neq)
        neq_3 = BinConstraintD(d3, b3, op_neq)
        neq_4 = BinConstraintD(d4, b4, op_neq)

        # 创建多个 Conj 对象，表示各维度不一致的约束条件
        dims_inconsistent1 = Conj([BinConstraintD(d1, Dyn, op_neq), BinConstraintD(b1, Dyn, op_neq), neq_1])
        dims_inconsistent2 = Conj([BinConstraintD(d2, Dyn, op_neq), BinConstraintD(b2, Dyn, op_neq), neq_2])
        dims_inconsistent3 = Conj([BinConstraintD(d3, Dyn, op_neq), BinConstraintD(b3, Dyn, op_neq), neq_3])
        dims_inconsistent4 = Conj([BinConstraintD(d4, Dyn, op_neq), BinConstraintD(b4, Dyn, op_neq), neq_4])

        # 创建一个 Disj 对象，表示维度不一致的总体约束条件
        dims_inconsistent = Disj([dims_inconsistent1, dims_inconsistent2, dims_inconsistent3, dims_inconsistent4])

        # 创建一个 Conj 对象，将输入大小为 4 的约束条件和维度不一致的总体约束条件组合起来
        ne_constraint = Conj([input_is_size4, dims_inconsistent])

        # 生成一个新的布尔变量 my_ne，并更新 counter
        my_ne, counter = gen_bvar(counter)

        # 创建 BinConstraintD 对象，表示 my_ne 与 ne_constraint 相等的约束条件
        equality_constraint = BinConstraintD(my_ne, ne_constraint, op_eq)

    else:
        # 如果参数 n 的第二个元素长度不为 4，则抛出未实现的错误
        raise NotImplementedError('Method not yet implemented')

    # 返回一个包含 equality_constraint 和 counter 的列表
    return [equality_constraint], counter
@register_inference_rule(operator.lt)
def lt_inference_rule(n: Node, symbols, constraints, counter):
    # 确保 n 的操作数是 Node 或整数类型
    assert isinstance(n.args[0], (Node, int))
    assert isinstance(n.args[1], (Node, int))

    # 获取操作数的符号表示，如果是 Node 类型则从 symbols 中获取
    e1 = symbols[n.args[0]] if isinstance(n.args[0], Node) else n.args[0]
    e2 = symbols[n.args[1]] if isinstance(n.args[1], Node) else n.args[1]

    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        if isinstance(e1, TVar) and isinstance(e2, TVar):
            # 生成一个新的类型变量和广播约束
            lt_tensor, counter = gen_tvar(counter)
            symbols[n] = lt_tensor
            return gen_broadcasting_constraints(e1, e2, symbols, counter, lt_tensor)

        elif isinstance(e1, DVar) and isinstance(e2, DVar):
            # 生成一个二进制约束用于流分析
            lt_constraint = BinConstraintD(e1, e2, op_lt)

            my_lt, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_lt, lt_constraint, op_eq)
            return [equality_constraint], counter

        else:
            raise RuntimeError('Sort Mismatch')

    elif isinstance(n.args[0], Node) and not isinstance(n.args[1], Node):
        if isinstance(e1, DVar):
            # 生成一个二进制约束用于流分析
            lt_constraint = BinConstraintD(e1, e2, op_lt)

            my_lt, counter = gen_bvar(counter)
            equality_constraint = BinConstraintD(my_lt, lt_constraint, op_eq)
            return [equality_constraint], counter
        else:
            raise NotImplementedError('Method not yet implemented')

    else:
        raise NotImplementedError('Method not yet implemented')


@register_inference_rule(torch.full)
def full_inference_rule(n: Node, symbols, constraints, counter):
    # 生成一个新的类型变量和计数器
    full, counter = gen_tvar(counter)
    symbols[n] = full
    res = []

    # 确保 n 的第一个参数是可迭代的
    assert isinstance(n.args[0], Iterable)
    for arg in n.args[0]:
        dim = arg if isinstance(arg, int) else symbols[arg]
        res.append(dim)
    # 创建一个张量类型的二进制约束
    c = BinConstraintT(full, TensorType(list(res)), op_eq)  # type: ignore[arg-type]
    return [c], counter


# TODO normalize index
@register_inference_rule(torch.arange)
def arange_inference_rule(n: Node, symbols, constraints, counter):
    start = 0
    step = 1

    if len(n.args) == 1:
        end = symbols[n.args[0]]
    else:
        raise NotImplementedError('Not yet implemented')

    # 计算 int((end - start) / step) 并生成一个动态变量和大小约束
    d1, counter = gen_dvar(counter)
    size_constraint = BinConstraintD(d1, BinConstraintD(BinConstraintD(end, start, op_sub), step, op_div), op_eq)
    arange, counter = gen_tvar(counter)
    symbols[n] = arange

    # 生成一些约束条件用于参数 a 是数值或者动态变量的情况
    c1 = Disj([BinConstraintD(end, Dyn, op_eq),
               BinConstraintD(start, Dyn, op_eq),
               BinConstraintD(step, Dyn, op_eq)])
    c2 = BinConstraintD(d1, Dyn, op_eq)
    both_dyn = Conj([c1, c2])
    # 创建一个 Conj 对象 c11，包含三个 BinConstraintD 约束条件：end != Dyn，start != Dyn，step != Dyn
    c11 = Conj([BinConstraintD(end, Dyn, op_neq),
                BinConstraintD(start, Dyn, op_neq),
                BinConstraintD(step, Dyn, op_neq)])
    
    # 创建一个 BinConstraintD 约束条件 c22：d1 != Dyn
    c22 = BinConstraintD(d1, Dyn, op_neq)
    
    # 创建一个 Conj 对象 both_numbers，包含 c11、c22 和 size_constraint 三个约束条件
    both_numbers = Conj([c11, c22, size_constraint])

    # 返回一个列表，包含两个元素：
    # 1. BinConstraintT 对象，约束 arange 为 TensorType([d1])，即 arange 的类型为包含一个维度 d1 的张量
    # 2. Disj 对象，包含 both_dyn 和 both_numbers 两个子句的析取
    return [BinConstraintT(arange, TensorType([d1]), op_eq), Disj([both_dyn, both_numbers])], counter
# 生成广播约束条件的函数
def gen_broadcasting_constraints(e1, e2, symbols, counter, output_var):
    # 生成两个不与表达式对应的附加变量，并更新计数器
    e11, counter = gen_tvar(counter)
    e22, counter = gen_tvar(counter)

    # 创建最大上界约束
    c1 = TGreatestUpperBound(output_var, e11, e22)
    # 应用广播操作的约束
    c2 = ApplyBroadcasting(e11, e22, e1, e2)
    # 二元约束类型的一致性约束
    c3 = BinConstraintT(e11, e22, op_consistency)
    return [c1, c2, c3], counter


# 注册推理规则函数，处理加法和乘法操作
@register_inference_rule(operator.mul)
@register_inference_rule(torch.ne)
@register_inference_rule("ne")
@register_inference_rule(torch.add)
@register_inference_rule(operator.add)
def broadcasting_inference_rule(n: Node, symbols, constraints, counter):
    # 根据操作符确定操作码
    op_code = None
    if n.target == operator.add or n.target == torch.add:
        op_code = op_add
    elif n.target == operator.mul:
        op_code = op_mul

    # 如果两个参数都是节点对象
    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        # 如果符号表中对应的值是类型变量
        if isinstance(symbols[n.args[0]], TVar) and isinstance(symbols[n.args[1]], TVar):
            # 生成一个新的类型变量作为输出，并更新符号表和计数器
            my_output, counter = gen_tvar(counter)
            symbols[n] = my_output
            e1 = symbols[n.args[0]]
            e2 = symbols[n.args[1]]

            # 调用生成广播约束的函数
            return gen_broadcasting_constraints(e1, e2, symbols, counter, my_output)
        else:
            raise NotImplementedError('Method not yet implemented')

    # 如果第一个参数是节点对象，第二个参数是数值（int或float）
    elif isinstance(n.args[0], Node) and isinstance(n.args[1], (int, float)):
        # 如果符号表中对应的值是类型变量
        if isinstance(symbols[n.args[0]], TVar):
            # 生成一个新的类型变量作为输出，并更新符号表和计数器，生成二元约束等式
            my_output, counter = gen_tvar(counter)
            symbols[n] = my_output
            e1 = symbols[n.args[0]]
            return [BinConstraintT(my_output, e1, op_eq)], counter
        # 如果符号表中对应的值是数据变量
        elif isinstance(symbols[n.args[0]], DVar):
            # 生成一个新的数据变量作为输出，并更新符号表和计数器，生成数据约束
            my_output, counter = gen_dvar(counter)
            symbols[n] = my_output
            e1 = symbols[n.args[0]]

            # 根据运行时值生成数据约束
            c = Conj([BinConstraintD(my_output, BinConstraintD(e1, n.args[1], op_code), op_eq),
                      BinConstraintD(0, my_output, op_leq)])
            return [c], counter

    # 如果第二个参数是节点对象，第一个参数是数值（int或float）
    elif isinstance(n.args[1], Node) and isinstance(n.args[0], (int, float)):
        # 如果符号表中对应的值是类型变量
        if isinstance(symbols[n.args[1]], TVar):
            # 生成一个新的类型变量作为输出，并更新符号表和计数器，生成二元约束等式
            my_output, counter = gen_tvar(counter)
            symbols[n] = my_output
            e2 = symbols[n.args[1]]
            return [BinConstraintT(my_output, e2, op_eq)], counter
        # 如果符号表中对应的值是数据变量
        elif isinstance(symbols[n.args[1]], DVar):
            # 生成一个新的数据变量作为输出，并更新符号表和计数器，生成数据约束
            my_output, counter = gen_dvar(counter)
            symbols[n] = my_output
            e2 = symbols[n.args[1]]

            # 根据运行时值生成数据约束
            c = Conj([BinConstraintD(my_output, BinConstraintD(e2, n.args[0], op_code), op_eq),
                      BinConstraintD(0, my_output, op_leq)])
            return [c], counter

        else:
            raise NotImplementedError('Method not yet implemented')
    else:
        # 如果程序执行到这里，表示当前情况尚未实现
        # 抛出未实现错误，提示加法操作尚未被实现
        raise NotImplementedError('Addition not yet implemented')
@register_inference_rule(torch.flatten)
def flatten_inference_rule(n: Node, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)

    # 生成新的临时变量
    flattened, counter = gen_tvar(counter)
    symbols[n] = flattened

    input = symbols[n.args[0]]

    # 设置默认的起始和结束维度
    start_dim = 1
    end_dim = -1

    if len(n.args) > 1:
        assert isinstance(n.args[1], int)
        start_dim = n.args[1]

    if len(n.args) > 2:
        assert isinstance(n.args[2], int)
        end_dim = n.args[2]

    # 创建约束条件 c1 和 c2
    c1 = BinConstraintT(input, Dyn, op_eq)
    c2 = BinConstraintT(flattened, Dyn, op_eq)
    both_dyn = Conj([c1, c2])

    const = []
    for i in range(1, MAX_TENSOR_RANK + 1):
        # 生成展平约束条件，并添加到 const 列表中
        c, counter = generate_flatten_constraints(start_dim, end_dim, input, flattened, i, counter)
        const.append(c)

    # 返回展平推断规则的约束条件列表和更新后的计数器
    return [Disj([both_dyn, *const])], counter


@register_inference_rule(torch.nn.functional.layer_norm)
def layer_norm_functional(n: Node, symbols, constraints, counter):
    """
    生成约束条件：input = output
    """
    assert isinstance(n.args[0], Node)
    return gen_layer_norm_constraints(n, n.args[1], symbols, counter)


@register_inference_rule(torch.nn.LayerNorm)
def layer_norm_inference_rule(n: Node, module_instance, symbols, constraints, counter):
    """
    输入和输出的形状应该相等。
    输入应该与normalized_shape一致。
    """
    assert isinstance(n.args[0], Node)
    return gen_layer_norm_constraints(n, module_instance.normalized_shape, symbols, counter)


def gen_layer_norm_constraints(n: Node, normalized_shape, symbols, counter):
    output, counter = gen_tvar(counter)
    symbols[n] = output
    input = symbols[n.args[0]]

    input_dyn = BinConstraintT(input, Dyn, op_eq)
    output_dyn = BinConstraintT(output, Dyn, op_eq)

    c1 = Conj([input_dyn, output_dyn])

    c2 = []
    for i in range(1, MAX_TENSOR_RANK + 1):
        new_dims_rhs, counter = gen_tensor_dims(i, counter)
        nat_constraints = gen_nat_constraints(new_dims_rhs)

        # 生成张量约束条件 c_tensor_i，并添加到 c2 列表中
        c_tensor_i = Conj([BinConstraintT(input, TensorType(new_dims_rhs), op_eq),
                           BinConstraintT(output, TensorType(new_dims_rhs), op_eq)] +
                          add_layer_norm_constraints(new_dims_rhs, list(normalized_shape)) +
                          nat_constraints)
        c2.append(c_tensor_i)
    # 返回层归一化推断规则的约束条件列表和更新后的计数器
    return [Disj([c1, Disj(c2)])], counter


@register_inference_rule(torch.nn.Dropout)
@register_inference_rule(torch.nn.ReLU)
def relu_inference_rule(n: Node, module_instance, symbols, constraints, counter):
    """
    输入和输出的形状应该相等。
    """
    assert isinstance(n.args[0], Node)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    input = symbols[n.args[0]]
    assert isinstance(input, TVar)
    return [BinConstraintT(input, output, op_eq)], counter


@register_inference_rule(torch.nn.Linear)
# 定义线性推理规则函数，处理节点n、模块实例、符号、约束和计数器
def linear_inference_rule(n: Node, module_instance, symbols, constraints, counter):
    """
    Input and output sizes should be the same except for the last dimension
    If the input is Dyn, then so should the output
    """
    # 断言节点n的第一个参数是Node类型
    assert isinstance(n.args[0], Node)
    # 调用linear_constraints函数，返回线性约束列表
    return linear_constraints(n, module_instance.in_features, module_instance.out_features, symbols, counter)


# 注册推理规则函数"dim"，处理节点n、符号、约束和计数器
@register_inference_rule("dim")  # type: ignore[attr-defined]
def torch_dim_inference_rule(n: Node, symbols, constraints, counter):
    # 断言节点n的第一个参数是Node类型
    assert isinstance(n.args[0], Node)
    # 生成一个新的维度变量my_dim和更新后的计数器
    my_dim, counter = gen_dvar(counter)
    # 将符号表中节点n对应的符号设置为my_dim
    symbols[n] = my_dim
    # 获取节点n的第一个参数的符号
    input = symbols[n.args[0]]

    # 创建输入和输出的动态约束
    input_dyn = BinConstraintT(input, Dyn, op_eq)
    output_dyn = BinConstraintD(my_dim, Dyn, op_eq)

    # 初始化空列表c1
    c1 = []

    # 循环生成1到MAX_TENSOR_RANK的张量维度约束
    for i in range(1, MAX_TENSOR_RANK + 1):
        # 生成一个新的张量维度new_dims_rhs_1和更新后的计数器
        new_dims_rhs_1, counter = gen_tensor_dims(i, counter)

        # 创建张量约束c_tensor_i，包括输入和输出的张量类型约束和维度约束
        c_tensor_i = Conj([BinConstraintT(input, TensorType(new_dims_rhs_1), op_eq),
                           BinConstraintD(my_dim, i, op_eq)])
        # 将c_tensor_i添加到列表c1中
        c1.append(c_tensor_i)

    # 返回包含输入和输出动态约束及张量维度约束的列表，以及更新后的计数器
    return [Disj([Conj([input_dyn, output_dyn]), Disj(c1)])], counter


# 注册推理规则函数torch._C._nn.linear，处理节点n、符号、约束和计数器
@register_inference_rule(torch._C._nn.linear)  # type: ignore[attr-defined]
def torch_linear_inference_rule(n: Node, symbols, constraints, counter):
    # 断言节点n的第一个参数是Node类型
    assert isinstance(n.args[0], Node)
    # 生成一个2维的张量维度weight_dims和更新后的计数器
    weight_dims, counter = gen_tensor_dims(2, counter)
    # 创建等式约束，要求节点n的第二个参数的张量类型与weight_dims相等
    equality_constraint = BinConstraintT(symbols[n.args[1]], TensorType(weight_dims), op_eq)
    # 调用linear_constraints函数，返回线性约束列表，并更新计数器
    constraints, counter = linear_constraints(n, weight_dims[1], weight_dims[0], symbols, counter)
    # 返回包含等式约束和线性约束的列表，以及更新后的计数器
    return [equality_constraint] + constraints, counter


# 定义线性约束函数，处理节点n、输入特征数、输出特征数、符号和计数器
def linear_constraints(n: Node, in_features, out_features, symbols, counter):
    # 生成一个新的线性输出变量linear_output和更新后的计数器
    linear_output, counter = gen_tvar(counter)
    # 将符号表中节点n对应的符号设置为linear_output
    symbols[n] = linear_output
    # 获取节点n的第一个参数的符号
    linear_input = symbols[n.args[0]]

    # 创建输入和输出的动态约束
    input_dyn = BinConstraintT(linear_input, Dyn, op_eq)
    output_dyn = BinConstraintT(linear_output, Dyn, op_eq)

    # 创建合取约束c1，包括输入和输出的动态约束
    c1 = Conj([input_dyn, output_dyn])

    # 初始化空列表c2
    c2 = []
    # 循环生成1到MAX_TENSOR_RANK的张量维度约束
    for i in range(1, MAX_TENSOR_RANK + 1):
        # 生成两个新的张量维度new_dims_rhs_1和new_dims_rhs_2，以及更新后的计数器
        new_dims_rhs_1, counter = gen_tensor_dims(i, counter)
        new_dims_rhs_2, counter = gen_tensor_dims(i, counter)

        # 生成自然数约束nat_constraints，要求new_dims_rhs_1 + new_dims_rhs_2是自然数
        nat_constraints = gen_nat_constraints(new_dims_rhs_1 + new_dims_rhs_2)

        # 创建合取约束c_tensor_i，包括输入和输出的张量类型约束、线性约束和自然数约束
        c_tensor_i = Conj([BinConstraintT(linear_input, TensorType(new_dims_rhs_1), op_eq),
                           BinConstraintT(linear_output, TensorType(new_dims_rhs_2), op_eq)] +
                          add_linear_constraints(new_dims_rhs_1, new_dims_rhs_2, in_features, out_features) +
                          nat_constraints)
        # 将c_tensor_i添加到列表c2中
        c2.append(c_tensor_i)

    # 返回包含合取约束c1和析取约束c2的列表，以及更新后的计数器
    return [Disj([c1, Disj(c2)])], counter


# 定义增加层归一化约束函数，处理输入维度和归一化维度
def add_layer_norm_constraints(input_dim, normalized_dim):
    """
    The constraints say that the type has te form: [*, 1024, 1024]
     while the normalized_dim have the form [1024, 1024]
    Args:
        input_dim: Input shape of layer norm
        normalized_dim: normalized_dim parameter of the module instance

    """
    # 如果标准化后的维度列表长度大于输入维度列表长度，则返回一个包含单个 F() 函数调用的列表，表示模式不匹配的情况
    if len(normalized_dim) > len(input_dim):
        return [F()]
    
    else:
        # 初始化一个空列表来存储约束条件
        constraints = []
        # 使用反向迭代器遍历输入维度和标准化后的维度列表，并创建约束对象 BinConstraintD，加入约束列表中
        for i, n in zip(reversed(input_dim), reversed(normalized_dim)):
            constraints.append(BinConstraintD(i, n, op_consistency))
        # 返回由创建的约束对象组成的约束列表
        return constraints
# 确保 dims1 和 dims2 的长度相等
assert len(dims1) == len(dims2)
# 初始化约束列表
constraints = []
# 遍历 dims1 和 dims2 中的每个维度
for i in range(len(dims1)):
    # 如果是最后一个维度
    if i == len(dims1) - 1:
        # 添加约束：dims1[i] 和 in_features 的二元约束
        constraints.append(BinConstraintD(dims1[i], in_features, op_consistency))
        # 添加约束：dims2[i] 和 out_features 的二元约束
        constraints.append(BinConstraintD(dims2[i], out_features, op_eq))
    else:
        # 添加约束：dims1[i] 和 dims2[i] 的二元约束
        constraints.append(BinConstraintD(dims1[i], dims2[i], op_eq))

# 返回生成的约束列表
return constraints


@register_inference_rule(torch.reshape)
def reshape_inference_rule(n: Node, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)

    # 生成新变量并更新计数器
    my_reshape, counter = gen_tvar(counter)
    # 将新变量与节点 n 关联
    symbols[n] = my_reshape

    # 获取节点 n 的输入符号变量
    src_var = symbols[n.args[0]]
    # 获取 reshape 操作的目标形状
    t2 = n.args[1]
    # 创建目标形状的张量类型，其中 Dyn 表示 -1 的位置
    t2_type = TensorType([Dyn if elem == -1 else elem for elem in t2])
    # 创建二元约束：my_reshape 和 t2_type 的相等约束
    c1 = BinConstraintT(my_reshape, t2_type, op_eq)
    # 创建可重塑约束：src_var 可重塑为 t2_type
    c2 = CanReshape(src_var, t2_type)

    # 返回约束列表和更新后的计数器
    return [c1, c2], counter


@register_inference_rule(BatchNorm2d)
def batchnorm_inference_rule(n: Node, module_instance, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)

    # 生成新变量并更新计数器
    batchnorm_output, counter = gen_tvar(counter)
    # 将新变量与节点 n 关联
    symbols[n] = batchnorm_output
    # 获取节点 n 的输入符号变量
    batchnorm_input = symbols[n.args[0]]

    # 生成维度变量
    d1, counter = gen_dvar(counter)
    d2, counter = gen_dvar(counter)
    d3, counter = gen_dvar(counter)
    d4, counter = gen_dvar(counter)

    # 生成自然约束
    nat_constraints = gen_nat_constraints([d1, d2, d3, d4])

    # 创建二元约束：batchnorm_input 和 [d1, d2, d3, d4] 的匹配约束
    c1 = BinConstraintT(batchnorm_input, TensorType([d1, d2, d3, d4]), op_matching)
    # 创建二元约束：batchnorm_input 和 batchnorm_output 的相等约束
    c2 = BinConstraintT(batchnorm_input, batchnorm_output, op_eq)

    # 返回约束列表、自然约束列表和更新后的计数器
    return [c1, c2, *nat_constraints], counter


@register_inference_rule(torch.nn.AdaptiveAvgPool2d)
def adaptive_inference_rule(n: Node, module_instance, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)

    # 生成新变量并更新计数器
    avg_pool, counter = gen_tvar(counter)

    # 将新变量与节点 n 关联
    symbols[n] = avg_pool
    # 获取节点 n 的输入符号变量
    input_var = symbols[n.args[0]]

    # 生成维度变量
    d1, counter = gen_dvar(counter)
    d2, counter = gen_dvar(counter)
    d3, counter = gen_dvar(counter)
    d4, counter = gen_dvar(counter)

    # 生成自然约束
    nat_constraints = gen_nat_constraints([d1, d2, d3, d4])

    # 创建二元约束：input_var 和 [d1, d2, d3, d4] 的匹配约束
    c1 = BinConstraintT(input_var, TensorType([d1, d2, d3, d4]), op_matching)
    # 创建二元约束：avg_pool 和 [d1, d2, module_instance.output_size[0], module_instance.output_size[1]] 的相等约束
    c2 = BinConstraintT(avg_pool, TensorType([d1, d2, module_instance.output_size[0], module_instance.output_size[1]]), op_eq)

    # 返回约束列表、自然约束列表和更新后的计数器
    return [c1, c2, *nat_constraints], counter


@register_inference_rule(Conv2d)
def conv2d_inference_rule(n: Node, module_instance, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)

    # 生成新变量并更新计数器
    my_conv, counter = gen_tvar(counter)
    # 将新变量与节点 n 关联
    symbols[n] = my_conv
    # 获取节点 n 的输入符号变量
    input_var = symbols[n.args[0]]

    # 生成张量维度变量
    [d1, d2, d3, d4], counter = gen_tensor_dims(MAX_TENSOR_RANK, counter)

    # 返回未完成的代码注释，因为缺少后续
    # 创建二进制约束条件 c1，传入输入变量、张量类型和操作匹配
    c1 = BinConstraintT(input_var, TensorType([d1, d2, d3, d4]), op_matching)

    # 创建维度一致性约束条件 c2，传入模块实例的输入通道数和维度 d2，以及操作一致性
    c2 = BinConstraintD(module_instance.in_channels, d2, op_consistency)

    # 计算卷积操作的约束条件 c3，传入我的卷积对象 my_conv、输入变量、输出通道数、卷积核大小、填充、步长、膨胀率以及维度列表 [d1, d2, d3, d4]
    c3 = CalcConv(my_conv, input_var,
                  module_instance.out_channels,
                  module_instance.kernel_size,
                  module_instance.padding,
                  module_instance.stride,
                  module_instance.dilation, [d1, d2, d3, d4])

    # 生成自然约束条件，传入维度列表 [d1, d2, d3, d4]
    nat_constraints = gen_nat_constraints([d1, d2, d3, d4])

    # 返回由 c1、c2、c3 和自然约束条件组成的列表，以及计数器 counter
    return [c1, c2, c3, *nat_constraints], counter
@register_inference_rule(torch.nn.MaxPool2d)
def maxpool_inference_rule(n: Node, module_instance, symbols, constraints, counter):
    assert isinstance(n.args[0], Node)
    # 生成一个新的符号变量 maxpool，并更新计数器
    maxpool, counter = gen_tvar(counter)
    # 将符号变量 maxpool 与当前节点 n 关联起来
    symbols[n] = maxpool
    # 获取当前节点 n 的输入变量
    input_var = symbols[n.args[0]]

    # 生成张量维度的符号变量 d1, d2, d3, d4，并更新计数器
    [d1, d2, d3, d4], counter = gen_tensor_dims(MAX_TENSOR_RANK, counter)

    # 创建输入变量 input_var 的约束条件 c1，包括张量类型和操作匹配
    c1 = BinConstraintT(input_var, TensorType([d1, d2, d3, d4]), op_matching)

    # 生成最大池化操作的约束条件 c2，涉及 maxpool、input_var、kernel_size 等参数
    c2 = CalcMaxPool(maxpool, input_var, module_instance.kernel_size, module_instance.padding,
                     module_instance.stride, module_instance.dilation, [d1, d2, d3, d4])

    # 生成自然约束条件 nat_constraints，基于维度变量 [d1, d2, d3, d4]
    nat_constraints = gen_nat_constraints([d1, d2, d3, d4])

    # 返回所有约束条件的列表，包括 c1、c2 和自然约束条件，以及更新后的计数器
    return [c1, c2, *nat_constraints], counter


class ConstraintGenerator:
    def __init__(self, traced, graph=None):
        # 初始化 ConstraintGenerator 类，设置 traced 属性为输入的 traced 对象或其根节点
        self.traced = traced  # traced or tracer.root
        # 从 traced 对象中提取所有命名的参数，并存储在 traced_params 字典中
        self.traced_params = dict(self.traced.named_parameters())
        # 初始化约束条件列表、符号字典和图对象属性
        self.constraints = []
        self.symbol_dict = {}
        self.graph = traced.graph if hasattr(traced, 'graph') else graph


    def generate_constraints(self, counter=0):
        """
        Iterate through every node and generate constraints
        Effect: self.constraints will be populated with the final constraints
        """
        # 获取当前实例的图对象
        graph = self.graph

        # 存储所有节点生成的约束条件的列表
        all_constraints = []

        # 遍历图中的每个节点，生成约束条件并累加到 all_constraints 中
        for n in graph.nodes:
            (constraints, counter) = self.generate_constraints_node(n, counter)
            all_constraints += constraints

        # 将所有生成的约束条件列表连接成一个 Conjunction 类型的对象，并返回以及更新后的计数器
        return Conj(all_constraints), counter
    def generate_constraints_node(self, n: Node, counter):
        """
        生成给定节点的约束条件:
        当前支持的操作有:
        - Reshape （重塑操作）
        - Add （加法操作）
        - conv2d （二维卷积操作）
        """

        if n.op == 'placeholder':
            # 生成新的类型变量并将其与节点 n 关联
            x, counter = gen_tvar(counter)
            self.symbol_dict[n] = x

            # 获取节点的类型信息
            my_type = n.type

            # 如果节点类型不是动态类型并且不是 TensorType 的实例
            if n.type != Dyn and (not isinstance(n.type, TensorType)):
                if n.type == torch.nn.parameter.Parameter:
                    # 对于参数类型，假设其形状是静态的
                    assert 'example_value' in n.meta
                    my_type = TensorType(n.meta['example_value'].size())
                else:
                    my_type = Dyn

            # 生成两个二元约束条件
            c1 = BinConstraintT(my_type, x, op_precision)
            c2 = BinConstraintT(x, MAX_TENSOR_RANK, op_leq)
            return [c1, c2], counter

        elif n.op == 'call_function':
            if n.target in _INFERENCE_RULES:
                # 如果函数调用有注册的推断规则，则调用对应的规则函数
                return _INFERENCE_RULES[n.target](n, self.symbol_dict, self.constraints, counter)
            else:
                # 如果没有注册的推断规则，则抛出运行时错误
                raise RuntimeError(f'No inference rule registered for target {n.target}!')

        elif n.op == 'call_module':

            # 获取模块实例并检查其类型是否有注册的推断规则
            module_instance = self.traced.get_submodule(n.target)
            if type(module_instance) in _INFERENCE_RULES:
                return _INFERENCE_RULES[type(module_instance)](n,
                                                               module_instance,
                                                               self.symbol_dict,
                                                               self.constraints, counter)
            else:
                # 如果没有注册的推断规则，则抛出运行时错误
                raise RuntimeError(f'No inference rule registered for class {type(module_instance)}!')

        elif n.op == 'call_method':
            if n.target in _INFERENCE_RULES:
                # 如果方法调用有注册的推断规则，则调用对应的规则函数
                return _INFERENCE_RULES[n.target](n, self.symbol_dict, self.constraints, counter)
            else:
                # 如果没有注册的推断规则，则抛出运行时错误
                raise RuntimeError(f'No inference rule registered for target {n.target}!')

        elif n.op == 'get_attr':
            # 获取节点 n 的目标属性，并根据其类型生成约束条件
            t = self.traced_params.get(n.target, None)

            if isinstance(t, torch.Tensor):
                if len(t.shape) > 0:
                    # 如果张量的形状不为空，则创建形状约束条件
                    res = list(t.shape)
                    attr_type = TensorType(res)
                    output, counter = gen_tvar(counter)
                    self.symbol_dict[n] = output
                    return [BinConstraintT(output, attr_type, op_eq)], counter
                else:
                    # 如果张量是标量，则返回空约束条件列表
                    return [], counter
            else:
                # 如果属性不是张量，则返回空约束条件列表
                return [], counter

        elif n.op == 'output':
            # 对于输出节点，返回空约束条件列表
            return [], counter

        else:
            # 如果节点的操作不在支持的操作列表中，则抛出未实现错误
            raise NotImplementedError(f"Method {n.op} not yet implemented")
```