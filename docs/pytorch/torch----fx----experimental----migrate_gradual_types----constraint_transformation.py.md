# `.\pytorch\torch\fx\experimental\migrate_gradual_types\constraint_transformation.py`

```py
# 忽略类型检查错误
# 导入必要的模块
import copy  # 导入copy模块，用于对象的深拷贝操作
import itertools  # 导入itertools模块，用于高效的迭代操作
from torch.fx.experimental.migrate_gradual_types.constraint_generator import BinConstraintT, MAX_TENSOR_RANK  # 从指定模块导入BinConstraintT和MAX_TENSOR_RANK
from torch.fx.experimental.migrate_gradual_types.constraint import T, BinConstraintD, Conj, Constraint, DVar, TVar, \  # 从指定模块导入多个类和函数
    Transpose  # 导入Transpose类
from torch.fx.experimental.migrate_gradual_types.constraint import Disj, TGreatestUpperBound  # 从指定模块导入Disj和TGreatestUpperBound
from torch.fx.experimental.migrate_gradual_types.constraint import DGreatestUpperBound  # 导入DGreatestUpperBound类
from torch.fx.experimental.migrate_gradual_types.constraint import CalcConv, CalcMaxPool  # 导入CalcConv和CalcMaxPool类
from torch.fx.experimental.migrate_gradual_types.constraint import CalcProduct, CanReshape  # 导入CalcProduct和CanReshape类
from torch.fx.experimental.migrate_gradual_types.constraint import ApplyBroadcasting, Prod, F, GetItem, GetItemTensor, IndexSelect  # 导入多个类和函数
from torch.fx.experimental.migrate_gradual_types.operation import op_eq, op_precision, op_leq, op_matching  # 从指定模块导入多个操作函数
from torch.fx.experimental.migrate_gradual_types.operation import op_consistency, op_neq  # 导入op_consistency和op_neq函数
from torch.fx.experimental.migrate_gradual_types.operation import op_mul, op_add, op_sub, op_div, op_mod  # 导入多个数学操作函数
from torch.fx.experimental.migrate_gradual_types.util import gen_tensor_dims, gen_nat_constraints, gen_dvar  # 从指定模块导入多个实用函数
from torch.fx.tensor_type import TensorType, Dyn  # 从指定模块导入TensorType和Dyn类
from typing import Callable, Dict, List  # 导入类型提示中的Callable、Dict和List类

_TRANSFORMATION_RULES: Dict[Constraint, Callable] = {}  # 定义一个空的全局字典_TRANSFORMATION_RULES，用于存储变换规则和对应的函数


def register_transformation_rule(call_target):
    # 定义注册变换规则的函数
    def register(fn):
        # 内部函数，注册变换规则的具体实现函数
        if call_target in _TRANSFORMATION_RULES:
            # 如果给定的调用目标(call_target)已经在_TRANSFORMATION_RULES中存在，抛出运行时错误
            raise RuntimeError(f'Transformation rule already registered for {call_target}!')
        _TRANSFORMATION_RULES[call_target] = fn  # 将给定的调用目标和函数注册到_TRANSFORMATION_RULES中
        return fn  # 返回注册的函数
    return register  # 返回内部函数


def valid_index(index, dims):
    """
    Given a list of dimensions, checks if an index is valid in the list
    """
    try:
        dims[index]  # 尝试索引列表dims中的索引index
        return T()  # 如果成功，返回T()
    except IndexError:
        return F()  # 如果索引超出范围，返回F()


@register_transformation_rule(Transpose)
def transform_transpose(constraint, counter):
    """
    Similar to a sequence of two index-selects
    """
    dims, counter = gen_tensor_dims(constraint.tensor_size, counter)  # 生成张量的维度信息和更新计数器
    is_valid_index1 = valid_index(constraint.index1, dims)  # 检查第一个索引是否在维度列表中有效
    is_valid_index2 = valid_index(constraint.index2, dims)  # 检查第二个索引是否在维度列表中有效
    new_dims = copy.deepcopy(dims)  # 深度复制维度列表

    nat_constraints = gen_nat_constraints(dims)  # 生成基于维度的自然数约束

    if is_valid_index1 == T() and is_valid_index2 == T():
        # 如果两个索引都是有效的
        new_dims[constraint.index1] = dims[constraint.index2]  # 交换第一个索引的维度
        new_dims[constraint.index2] = dims[constraint.index1]  # 交换第二个索引的维度

    # 构建变换后的约束，包括张量类型匹配和自然数约束
    transformed_constraint = Conj([BinConstraintT(constraint.input_var, TensorType(dims), op_eq),
                                   *nat_constraints,
                                   is_valid_index1, is_valid_index2,
                                   BinConstraintT(constraint.output, TensorType(new_dims), op_eq)])
    return transformed_constraint, counter  # 返回变换后的约束和更新后的计数器


@register_transformation_rule(IndexSelect)
def transform_index_select(constraint, counter):
    """
    The constraints consider the given tensor size, checks if the index is valid
    """
    dims, counter = gen_tensor_dims(constraint.tensor_size, counter)
    # 生成张量维度和计数器，并返回

    is_valid_index = valid_index(constraint.index, dims)
    # 检查给定索引是否在有效范围内

    nat_constraints = gen_nat_constraints(dims)
    # 生成与张量维度相关的自然数约束条件

    # 如果索引有效，则替换输入维度为新的维度
    # 否则维度不会被替换，子句将包含 False
    if is_valid_index == T():
        new_dims = copy.deepcopy(dims)
        new_dims[constraint.index] = constraint.dim_replace
        # 使用深拷贝创建新维度列表，并将指定索引的维度替换为新的维度值

    transformed_constraint = Conj([
        BinConstraintT(constraint.input_var, TensorType(dims), op_eq),
        *nat_constraints,
        is_valid_index,
        BinConstraintT(constraint.output, TensorType(new_dims), op_eq)
    ])
    # 构建转换后的约束条件，包括输入变量的张量类型约束、自然数约束、索引有效性检查以及输出变量的张量类型约束

    # 返回转换后的约束条件和更新后的计数器
    return transformed_constraint, counter
@register_transformation_rule(GetItem)
def transform_get_item(constraint, counter):
    """
    生成形如以下的等式:
    t = [a1, ..., an]
    然后生成检查给定索引是否有效的约束条件
    给定特定张量大小的情况下。
    如果索引有效，则生成用于获取项的约束条件。
    注意，我们已经在前一个步骤中处理了Dyn输入的情况。
    参数:
        constraint: GetItem，假设我们从张量中获取项（而不是Dyn）
        counter: 跟踪变量
    返回: GetItem的简化约束条件

    """
    # 生成张量维度信息和计数器
    dims, counter = gen_tensor_dims(constraint.tensor_size, counter)
    # 生成与张量维度相关的自然数约束
    nat_constraints = gen_nat_constraints(dims)

    # 检查索引是否有效
    is_valid_index = valid_index(constraint.index, dims)

    # 组合所有约束条件
    all_constraints = [BinConstraintT(constraint.input_var, TensorType(dims), op_eq),
                       *nat_constraints,
                       is_valid_index]

    # 如果索引有效，则生成用于获取项的约束条件
    # 否则由于索引错误，此子句将会不可满足（UNSAT）
    if is_valid_index == T():
        all_constraints.append(BinConstraintD(constraint.res, dims[constraint.index], op_eq))

    return Conj(all_constraints), counter

def valid_index_tensor(index, dims):
    """
    如果切片实例超过了维度的长度，则这是类型错误，因此返回False
    """
    slice_count = 0
    for s in index:
        if isinstance(s, slice):
            slice_count += 1
    if slice_count > len(dims):
        return F()
    else:
        return T()

@register_transformation_rule(GetItemTensor)
def transform_get_item_tensor(constraint, counter):
    """
    当索引是元组时，输出将是一个张量
    TODO: 我们必须检查所有HF模型是否都是这种情况

    这里我们涵盖的情况是元组中包含:
     - 具有默认参数的切片
     - None

     None会将输入张量的维度增加1
     因此每个'None'的出现都会增加1的秩

     具有默认参数的切片不会改变秩
    """
    assert isinstance(constraint.index_tuple, tuple)

    # 生成预期大小的结果张量
    dims, counter = gen_tensor_dims(constraint.tensor_size, counter)
    # 生成与张量维度相关的自然数约束
    nat_constraints = gen_nat_constraints(dims)

    # 生成正确秩的占位符列表
    # 其中"slice"不会增加秩，而"None"会
    none_c = constraint.index_tuple.count(None)
    resulting_tensor_dims = (none_c + len(dims)) * [None]

    dim_index = 0
    for i in range(len(constraint.index_tuple)):
        # 在结果张量的正确位置添加1
        if constraint.index_tuple[i] is None:
            resulting_tensor_dims[i] = 1

        elif constraint.index_tuple[i] == slice(None, None, None):
            pass

        else:
            raise NotImplementedError('Method not yet implemented')
    # 将剩余的维度添加到正确的位置
    dim_index = 0  # 初始化维度索引
    for i in range(len(resulting_tensor_dims)):  # 遍历结果张量的维度列表
        if resulting_tensor_dims[i] is None:  # 如果当前位置的维度为空
            resulting_tensor_dims[i] = dims[dim_index]  # 使用给定的维度填充到结果张量的相应位置
            dim_index += 1  # 移动到下一个待填充的维度

    # 检查索引是否有效
    is_valid_index = valid_index_tensor(constraint.index_tuple, dims)

    # 检查结果张量是否在边界内
    if len(resulting_tensor_dims) > 4:  # 如果结果张量的维度超过4个
        return F(), counter  # 返回空函数和计数器

    else:
        # 构建约束条件列表，包括二元约束和其他自然约束
        constraints = [
            BinConstraintT(constraint.input_var, TensorType(dims), op_eq),  # 输入变量的维度约束
            BinConstraintT(constraint.res, TensorType(resulting_tensor_dims), op_eq),  # 结果张量的维度约束
            *nat_constraints,  # 其他自然约束（如果有）
            is_valid_index  # 索引有效性约束
        ]
        return Conj(constraints), counter  # 返回约束的合取和计数器
# 注册二进制约束转换规则，处理 BinConstraintT 类型的约束对象
@register_transformation_rule(BinConstraintT)
def generate_binconstraint_t(constraint, counter):
    """
    Transform binary constraints for tensors
    """

    # 处理精度约束
    if constraint.op == op_precision:
        # 如果左操作数是动态维度(Dyn)，返回空的变换结果和计数器
        if constraint.lhs == Dyn:
            return T(), counter
        # 如果左操作数是张量类型
        elif isinstance(constraint.lhs, TensorType):
            # 检查是否所有维度都是完全静态的
            is_fully_static = all(d != Dyn for d in constraint.lhs.__args__)
            if is_fully_static:
                # 返回左操作数与右操作数之间精度相等的约束转换结果和计数器
                return BinConstraintT(constraint.lhs, constraint.rhs, op_eq), counter
            else:
                new_dims = []

                # 生成新的动态维度变量
                for _ in range(len(constraint.lhs.__args__)):
                    dim, counter = gen_dvar(counter)
                    new_dims.append(dim)

                # 构建新的约束条件列表，包括维度转换和精度匹配约束
                new_dim_constraints = [BinConstraintD(old_dim, new_dim, op_precision) for
                                       new_dim, old_dim in zip(new_dims, constraint.lhs.__args__)] + \
                                      [BinConstraintT(constraint.rhs, TensorType(new_dims), op_eq)] + \
                                      [BinConstraintD(1, new_dim, op_leq) for
                                       new_dim in new_dims]
                return Conj(new_dim_constraints), counter

    # 处理匹配约束
    elif constraint.op == op_matching:
        assert isinstance(constraint.rhs, TensorType)
        d1 = constraint.rhs.__args__[0]
        d2 = constraint.rhs.__args__[1]
        d3 = constraint.rhs.__args__[2]
        d4 = constraint.rhs.__args__[3]

        # 构建约束条件列表，确保左操作数与动态维度相等，右操作数与指定维度相等，以及整体张量类型相等
        conj = [BinConstraintT(constraint.lhs, Dyn, op_eq),
                BinConstraintD(d1, Dyn, op_eq),
                BinConstraintD(d2, Dyn, op_eq),
                BinConstraintD(d3, Dyn, op_eq),
                BinConstraintD(d4, Dyn, op_eq)]
        return Disj([Conj(conj),
                     BinConstraintT(constraint.lhs, TensorType([d1, d2, d3, d4]), op_eq)]), counter

    # 处理一致性约束
    elif constraint.op == op_consistency:
        # 构建一致性约束，包括对动态维度的处理和生成的张量约束
        c_dyn = Disj([BinConstraintT(constraint.lhs, Dyn, op_eq), BinConstraintT(constraint.rhs, Dyn, op_eq)])
        [c_tensor_1, c_tensor_2, c_tensor_3, c_tensor_4], counter = gen_consistency_constraints(constraint, counter)

        return Disj([c_dyn, c_tensor_1, c_tensor_2, c_tensor_3, c_tensor_4]), counter

    # 处理小于等于约束
    elif constraint.op == op_leq:
        assert isinstance(constraint.rhs, int)
        disj = [BinConstraintT(constraint.lhs, Dyn, op_eq)]
        # 生成多个维度约束，确保左操作数与生成的张量类型相等
        for i in range(1, constraint.rhs + 1):
            dims = []
            for j in range(1, i + 1):
                dim_var, counter = gen_dvar(counter)
                dims.append(dim_var)
            disj.append(BinConstraintT(constraint.lhs, TensorType(dims), op_eq))
        return Disj(disj), counter
    else:
        # 对于未知操作，直接返回原始约束和计数器
        return constraint, counter
    # 如果约束的操作符等于 op_precision
    if constraint.op == op_precision:
        # 如果约束的左操作数是整数
        if isinstance(constraint.lhs, int):
            # 返回一个 BinConstraintD 对象和计数器
            return BinConstraintD(constraint.lhs, constraint.rhs, op_eq), counter
        # 如果约束的左操作数是 Dyn
        elif constraint.lhs == Dyn:
            # 返回一个 T() 对象和计数器
            return T(), counter
    
    # 如果约束的操作符等于 op_consistency
    elif constraint.op == op_consistency:
        # 返回一个 Disj 对象，包含三个 BinConstraintD 对象，并返回计数器
        return Disj([
            BinConstraintD(constraint.lhs, constraint.rhs, op_eq),
            BinConstraintD(constraint.rhs, Dyn, op_eq),
            BinConstraintD(constraint.lhs, Dyn, op_eq)
        ]), counter
    
    # 如果约束的操作符不是上述两种情况
    else:
        # 返回原始约束对象和计数器
        return constraint, counter
# 注册转换规则，处理 Conjunction（逻辑与）约束
@register_transformation_rule(Conj)
def generate_conj(constraint, counter):
    """
    Transform conjunctions

    Args:
        constraint: Conjunction constraint object
        counter: Counter for transformation steps

    Returns:
        Conj: Transformed conjunction constraint
        int: Updated counter value
    """
    new = []
    # 遍历每个逻辑与操作中的子约束
    for c in constraint.conjucts:
        # 对每个子约束进行转换，并更新计数器
        new_c, counter = transform_constraint(c, counter)
        new.append(new_c)
    return Conj(new), counter


# 注册转换规则，处理 Disjunction（逻辑或）约束
@register_transformation_rule(Disj)
def generate_disj(constraint, counter):
    """
    Transform disjunctions

    Args:
        constraint: Disjunction constraint object
        counter: Counter for transformation steps

    Returns:
        Disj: Transformed disjunction constraint
        int: Updated counter value
    """
    new = []
    # 遍历每个逻辑或操作中的子约束
    for c in constraint.disjuncts:
        # 对每个子约束进行转换，并更新计数器
        new_c, counter = transform_constraint(c, counter)
        new.append(new_c)
    return Disj(new), counter


# 注册转换规则，处理 TGreatestUpperBound（最大上界）约束
@register_transformation_rule(TGreatestUpperBound)
def generate_gub(constraint, counter):
    """
    Transform greatest upper bound for tensors. Results in equality and Greatest Upper Bound
    on dimensions

    Args:
        constraint: TGreatestUpperBound constraint object
        counter: Counter for transformation steps

    Returns:
        Disj: Transformed disjunction constraint containing transformed constraints
        int: Updated counter value
    """
    # 创建第一个约束 c1，包含三个子约束
    c1 = Conj([
        Disj([
            BinConstraintT(constraint.rhs1, Dyn, op_eq),
            BinConstraintT(constraint.rhs2, Dyn, op_eq)
        ]),
        BinConstraintT(constraint.res, Dyn, op_eq)
    ])

    # 调用辅助函数生成剩余的四个约束
    [c2, c3, c4, c5], counter = gen_greatest_upper_bound(constraint, counter)

    # 返回包含所有约束的逻辑或操作
    return Disj([c1, c2, c3, c4, c5]), counter


# 注册转换规则，处理 DGreatestUpperBound（维度最大上界）约束
@register_transformation_rule(DGreatestUpperBound)
def generate_d_gub(constraint, counter):
    """
    Transform greatest upper bound for dimensions into equality constraints

    Args:
        constraint: DGreatestUpperBound constraint object
        counter: Counter for transformation steps

    Returns:
        Disj: Transformed disjunction constraint containing transformed constraints
        int: Updated counter value
    """
    # 创建三个包含两个子约束的逻辑与操作，分别为 c1, c2, c3
    c1 = Conj([
        BinConstraintD(constraint.rhs1, Dyn, op_eq),
        BinConstraintD(constraint.res, constraint.rhs2, op_eq)
    ])
    c2 = Conj([
        BinConstraintD(constraint.rhs2, Dyn, op_eq),
        BinConstraintD(constraint.res, constraint.rhs1, op_eq)
    ])
    c3 = Conj([
        BinConstraintD(constraint.rhs2, constraint.rhs1, op_eq),
        BinConstraintD(constraint.res, constraint.rhs1, op_eq)
    ])
    # 返回包含所有约束的逻辑或操作
    return Disj([c1, c2, c3]), counter


# 注册转换规则，处理 CalcConv（卷积计算）约束
@register_transformation_rule(CalcConv)
def generate_calc_conv(constraint, counter):
    """
    Transform convolution constraints

    Args:
        constraint: CalcConv constraint object
        counter: Counter for transformation steps

    Returns:
        Conj: Transformed conjunction constraint containing transformed constraints
        int: Updated counter value
    """
    # 生成一个四维张量的维度信息，并更新计数器
    d, counter = gen_tensor_dims(4, counter)
    # 创建卷积结果的张量类型
    conv_result = TensorType([d[0], d[1], d[2], d[3]])

    # 创建五个约束 c1 到 c5
    c1 = BinConstraintT(constraint.conv_result, conv_result, op_eq)
    c2 = Conj([
        BinConstraintD(d[1], constraint.c_out, op_eq),
        BinConstraintD(d[1], Dyn, op_neq)
    ])
    c3 = BinConstraintD(constraint.matching_constraint[0], d[0], op_eq)
    c4, c5 = calc_last_two_dims(constraint, d)

    # 创建四个小于等于零的约束
    leq_constraints = Conj([
        BinConstraintD(0, d[0], op_leq),
        BinConstraintD(0, d[1], op_leq),
        BinConstraintD(0, d[2], op_leq),
        BinConstraintD(0, d[3], op_leq)
    ])

    # 返回包含所有约束的逻辑与操作
    return Conj([c1, c2, c3, c4, c5, leq_constraints]), counter


# 注册转换规则，处理 CalcMaxPool（最大池化计算）约束
@register_transformation_rule(CalcMaxPool)
def generate_calc_maxpool(constraint, counter):
    """
    Transform maxpool constraints

    Args:
        constraint: CalcMaxPool constraint object
        counter: Counter for transformation steps

    Returns:
        None
    """
    d, counter = gen_tensor_dims(4, counter)
    maxpool_result = TensorType([d[0], d[1], d[2], d[3]])

    # the maxpool result is a tensor of size 4
    # 略，后续代码未提供，无需添加注释
    # 创建 BinConstraintT 对象，约束条件是 maxpool_result 等于 constraint.maxpool_result
    c1 = BinConstraintT(constraint.maxpool_result, maxpool_result, op_eq)

    # 创建 BinConstraintD 对象，约束条件是 d 的第二维等于 constraint.matching_constraint 的第二维
    c2 = BinConstraintD(constraint.matching_constraint[1], d[1], op_eq)
    # 创建 BinConstraintD 对象，约束条件是 d 的第一维等于 constraint.matching_constraint 的第一维
    c3 = BinConstraintD(constraint.matching_constraint[0], d[0], op_eq)
    
    # 调用 calc_last_two_dims 函数计算最后两维的约束条件，并分别赋给 c4 和 c5
    c4, c5 = calc_last_two_dims(constraint, d)

    # 创建 Conj 对象，包含四个约束条件，每个约束条件是 d 的一维与0之间的小于等于关系
    leq_constraints = Conj([BinConstraintD(0, d[0], op_leq),
                            BinConstraintD(0, d[1], op_leq),
                            BinConstraintD(0, d[2], op_leq),
                            BinConstraintD(0, d[3], op_leq)])

    # 返回一个 Conj 对象，包含所有创建的约束条件 c1 到 c5 以及 leq_constraints，同时返回 counter
    return Conj([c1, c2, c3, c4, c5, leq_constraints]), counter
# 注册转换规则，将 CalcProduct 类型的约束转换为新的约束表达式
@register_transformation_rule(CalcProduct)
def generate_calc_product(constraint, counter):
    """
    Transform flatten constraints
    """
    # 从约束对象中获取起始位置、结束位置、需要展平的维度和展平后的维度
    start = constraint.start
    end = constraint.end
    dims = constraint.dims_to_flatten
    flattened = constraint.flattened
    n = len(constraint.dims_to_flatten)

    # 边界检查，确保起始位置在合理范围内
    boundary_check = (0 <= start and start < end and end <= n)

    # 根据边界检查结果，决定是否创建一个真值或假值的约束
    c_boundary = T() if boundary_check else F()

    # 拆分展平维度，分为左侧、中间和右侧部分
    lhs = dims[0:start]
    rhs = dims[end:]
    mid = dims[start:end]

    # 生成所有中间维度的动态整数可能性
    all_possibilities = generate_all_int_dyn_dim_possibilities(mid)

    # 存储所有的约束条件
    all_constraints = []

    # 遍历所有可能性
    for p in all_possibilities:
        p = list(p)
        
        # 检查当前可能性是否包含动态变量
        contains_dyn = not all(constraint.op == op_neq for constraint in p)
        
        if contains_dyn:
            mid_var = [Dyn]
            total_constraints = lhs + mid_var + rhs
            # 如果总约束数大于4，则添加假值约束
            if len(total_constraints) > 4:
                all_constraints.append(F())
            else:
                # 否则创建一个包含展平约束和所有可能性约束的合取式
                all_constraints.append(Conj([BinConstraintT(flattened, TensorType(lhs + mid_var + rhs), op_eq)] + p))
        else:
            # 否则创建一个新的动态变量，并生成对应的中间等式乘积约束
            new_var, counter = gen_dvar(counter)
            mid_eq_prod = Conj([BinConstraintD(new_var, Prod(mid), op_eq), BinConstraintD(new_var, Dyn, op_neq)])
            mid_var = [new_var]
            total_constraints = lhs + mid_var + rhs
            if len(total_constraints) > 4:
                all_constraints.append(F())
            else:
                # 创建一个包含展平约束、中间等式乘积约束和所有可能性约束的合取式
                all_constraints.append(Conj([BinConstraintT(flattened, TensorType(lhs + mid_var + rhs), op_eq), mid_eq_prod] + p))

    # 返回所有约束的析取式和边界约束的合取式
    return Conj([Disj(all_constraints), c_boundary]), counter


# 注册转换规则，将 CanReshape 类型的约束转换为新的约束表达式
@register_transformation_rule(CanReshape)
def generate_reshape(constraint, counter):
    """
    Transform reshape constraints
    """
    # 生成一个四维张量的维度，其中每个维度变量和一个计数器相关联
    d, counter = gen_tensor_dims(4, counter)

    # 分别获取四维张量的各个维度
    d1 = d[0]
    d2 = d[1]
    d3 = d[2]
    d4 = d[3]

    # 获取目标维度
    target = constraint.target.__args__

    # 判断目标维度是否完全静态
    is_fully_static = all(d != Dyn for d in target)

    # 创建动态张量约束条件
    c1_dyn = BinConstraintT(constraint.src, Dyn, op_eq)
    c2_tensor1 = BinConstraintT(constraint.src, TensorType([d1]), op_eq)
    c2_tensor2 = BinConstraintT(constraint.src, TensorType([d1, d2]), op_eq)
    c2_tensor3 = BinConstraintT(constraint.src, TensorType([d1, d2, d3]), op_eq)
    c2_tensor4 = BinConstraintT(constraint.src, TensorType([d1, d2, d3, d4]), op_eq)

    # 创建动态张量的维度相等和不等的约束条件
    d1_eq_dyn = BinConstraintD(d1, Dyn, op_eq)
    d1_neq_dyn = BinConstraintD(d1, Dyn, op_neq)

    d2_eq_dyn = BinConstraintD(d2, Dyn, op_eq)
    d2_neq_dyn = BinConstraintD(d2, Dyn, op_neq)

    d3_eq_dyn = BinConstraintD(d3, Dyn, op_eq)
    d3_neq_dyn = BinConstraintD(d3, Dyn, op_neq)

    d4_eq_dyn = BinConstraintD(d3, Dyn, op_eq)  # 注意这里是 d3，可能是个错误
    d4_neq_dyn = BinConstraintD(d3, Dyn, op_neq)  # 同样注意这里是 d3，可能是个错误

    # 创建维度为自然数的约束条件
    nat_d1 = BinConstraintD(0, d1, op_leq)
    nat_d2 = BinConstraintD(0, d2, op_leq)
    nat_d3 = BinConstraintD(0, d3, op_leq)
    nat_d4 = BinConstraintD(0, d4, op_leq)
    # 如果 is_fully_static 为 True，则生成静态张量表达式
    if is_fully_static:
        # tensor 大小为 1
        c3_tensor1 = Disj([d1_eq_dyn,
                           (Conj([d1_neq_dyn,
                                  BinConstraintD(d1, Prod(target), op_eq)]))])
        all_tensor_1 = Conj([c2_tensor1, c3_tensor1])

        # tensor 大小为 2
        all_tensor_2 = Conj([c2_tensor2, gen_all_reshape_possibilities([d1, d2], target)])

        # tensor 大小为 3
        all_tensor_3 = Conj([c2_tensor3, gen_all_reshape_possibilities([d1, d2, d3], target)])

        # tensor 大小为 4
        all_tensor_4 = Conj([c2_tensor4, gen_all_reshape_possibilities([d1, d2, d3, d4], target)])

        # 返回静态情况下的约束条件和计数器
        return Conj([Disj([c1_dyn, all_tensor_1, all_tensor_2, all_tensor_3, all_tensor_4]),
                     nat_d1, nat_d2, nat_d3, nat_d4]), counter

    # 如果不是完全静态，则需确保 target 中仅有一个动态尺寸 Dyn 的出现
    else:
        new_target = []

        # 遍历 target，移除所有非 Dyn 的尺寸
        for n in target:
            if n != Dyn:
                new_target.append(n)

        # tensor 1
        c3_tensor1 = Disj([d1_eq_dyn,
                           (Conj([d1_neq_dyn,
                                  is_dim_div_by_target(new_target, d1)]))])
        all_tensor_1 = Conj([c2_tensor1, c3_tensor1])

        # tensor 2
        c21 = Disj([d1_eq_dyn, d2_eq_dyn])
        c22 = Conj([d1_neq_dyn, d2_neq_dyn, is_dim_div_by_target(new_target, Prod([d1, d2]))])
        all_tensor_2 = Conj([c2_tensor2, Disj([c21, c22])])

        # tensor 3
        c31 = Disj([d1_eq_dyn, d2_eq_dyn, d3_eq_dyn])
        c32 = Conj([d1_neq_dyn, d2_neq_dyn, d3_neq_dyn, is_dim_div_by_target(new_target, Prod([d1, d2, d3]))])
        all_tensor_3 = Conj([c2_tensor3, Disj([c31, c32])])

        # tensor 4
        c41 = Disj([d1_eq_dyn, d2_eq_dyn, d3_eq_dyn, d4_eq_dyn])
        c42 = Conj([d1_neq_dyn, d2_neq_dyn, d3_neq_dyn, d4_neq_dyn, is_dim_div_by_target(new_target, Prod([d1, d2, d3, d4]))])
        all_tensor_4 = Conj([c2_tensor4, Disj([c41, c42])])

        # 返回非静态情况下的约束条件和计数器
        return Conj([Disj([c1_dyn, all_tensor_1, all_tensor_2, all_tensor_3, all_tensor_4]),
                     nat_d1, nat_d2, nat_d3, nat_d4]), counter
@register_transformation_rule(ApplyBroadcasting)
def generate_broadcasting(constraint, counter):
    """
    Transform broadcasting constraints
    """
    # 从约束条件中获取结果的表达式
    e11, e12 = constraint.res1, constraint.res2
    # 从约束条件中获取输入的表达式
    e1, e2 = constraint.input1, constraint.input2

    # 创建一个约束条件，将 e1 限制为动态维度
    e1_dyn = BinConstraintT(e1, Dyn, op_eq)
    # 创建一个约束条件，将 e2 限制为动态维度
    e2_dyn = BinConstraintT(e2, Dyn, op_eq)

    # 引入维度相等的约束条件
    e1_equal_e11 = BinConstraintT(e1, e11, op_eq)
    e2_equal_e12 = BinConstraintT(e2, e12, op_eq)

    # 动态维度可能性的约束条件
    e1_dyn_constraint = Conj([e1_dyn, e1_equal_e11, e2_equal_e12])
    e2_dyn_constraint = Conj([e2_dyn, e1_equal_e11, e2_equal_e12])

    # 张量可能性
    # 生成维度以创建大小为1的张量
    final_tensor_1_constraint, _, _, nat_dims_1, counter = \
        gen_broadcasting_constraints(e1, e2, e11, e12, 1, counter)

    # 生成维度以创建大小为2的张量
    final_tensor_2_constraint_no_padding, final_tensor_2_constraint_padding_arg1, \
        final_tensor_2_constraint_padding_arg2, nat_dims_2, counter = \
        gen_broadcasting_constraints(e1, e2, e11, e12, 2, counter)

    # 生成维度以创建大小为3的张量
    final_tensor_3_constraint_no_padding, final_tensor_3_constraint_padding_arg1, \
        final_tensor_3_constraint_padding_arg2, nat_dims_3, counter = \
        gen_broadcasting_constraints(e1, e2, e11, e12, 3, counter)

    # 生成维度以创建大小为4的张量
    final_tensor_4_constraint_no_padding, final_tensor_4_constraint_padding_arg1, \
        final_tensor_4_constraint_padding_arg2, nat_dims_4, counter = \
        gen_broadcasting_constraints(e1, e2, e11, e12, 4, counter)

    # 最终的结果为所有可能性的析取
    final_result = Disj([
        e1_dyn_constraint,
        e2_dyn_constraint,
        final_tensor_1_constraint,
        final_tensor_2_constraint_no_padding,
        final_tensor_2_constraint_padding_arg1,
        final_tensor_2_constraint_padding_arg2,
        final_tensor_3_constraint_no_padding,
        final_tensor_3_constraint_padding_arg1,
        final_tensor_3_constraint_padding_arg2,
        final_tensor_4_constraint_no_padding,
        final_tensor_4_constraint_padding_arg1,
        final_tensor_4_constraint_padding_arg2
    ])

    # 返回所有维度约束的合取以及自然维度的列表和计数器
    return Conj([final_result, *nat_dims_1, *nat_dims_2, *nat_dims_3, *nat_dims_4]), counter


def transform_constraint(constraint: Constraint, counter: int):
    """
    Transforms a constraint into a simpler constraint.
    Ex: precision and consistency are transformed to equality
    Args:
        constraint: constraint to be transformed
        counter: for variable tracking

    Returns: Constraint

    """
    # 如果约束类型在_TRANSFORMATION_RULES字典中，则应用相应的转换规则
    if type(constraint) in _TRANSFORMATION_RULES:
        return _TRANSFORMATION_RULES[type(constraint)](constraint, counter)

    else:
        # 否则直接返回原始约束和计数器
        return constraint, counter




def calc_last_two_dims(constraint, d: List[DVar]):
    """
    Generates constraints for the last two dimensions of a convolution or a maxpool output
    """
    Args:
        constraint: CalcConv or CalcMaxPool  # 接受一个 CalcConv 或 CalcMaxPool 对象作为约束条件
        d: The list of output dimensions  # 输出维度的列表

    Returns: Constraints for calculating the last two dimensions of the output
    # 返回用于计算输出的最后两个维度的约束条件

    """

    assert isinstance(constraint, (CalcConv, CalcMaxPool))  # 断言 constraint 是 CalcConv 或 CalcMaxPool 类的实例

    b3 = constraint.matching_constraint[2]  # 从 constraint 中获取第三个匹配约束
    b4 = constraint.matching_constraint[3]  # 从 constraint 中获取第四个匹配约束

    b3_dyn = Conj([BinConstraintD(d[2], Dyn, op_eq), BinConstraintD(b3, Dyn, op_eq)])
    # 创建一个包含两个二进制约束的合取条件，用于表示 d[2] 和 b3 是动态的相等约束
    b4_dyn = Conj([BinConstraintD(d[3], Dyn, op_eq), BinConstraintD(b4, Dyn, op_eq)])
    # 创建一个包含两个二进制约束的合取条件，用于表示 d[3] 和 b4 是动态的相等约束

    d3_not_dyn = Conj([BinConstraintD(d[2], Dyn, op_neq), BinConstraintD(b3, Dyn, op_neq)])
    # 创建一个包含两个二进制约束的合取条件，用于表示 d[2] 和 b3 是非动态的不相等约束
    d4_not_dyn = Conj([BinConstraintD(d[3], Dyn, op_neq), BinConstraintD(b4, Dyn, op_neq)])
    # 创建一个包含两个二进制约束的合取条件，用于表示 d[3] 和 b4 是非动态的不相等约束

    # 将参数转换为元组，以防它们不是已经是元组
    padding = (constraint.padding, constraint.padding) \
        if isinstance(constraint.padding, int) else constraint.padding
    kernel = (constraint.kernel, constraint.kernel) \
        if isinstance(constraint.kernel, int) else constraint.kernel
    stride = (constraint.stride, constraint.stride) \
        if isinstance(constraint.stride, int) else constraint.stride
    dilation = (constraint.dilation, constraint.dilation) \
        if isinstance(constraint.dilation, int) else constraint.dilation

    f1 = BinConstraintD(b3, BinConstraintD(2, padding[0], op_mul), op_add)
    # 计算 f1 的二进制约束，包含 b3、padding[0] 和乘法相加操作
    f2 = BinConstraintD(dilation[0], BinConstraintD(kernel[0], 1, op_sub), op_mul)
    # 计算 f2 的二进制约束，包含 dilation[0]、kernel[0] 和减法乘法操作
    f3 = BinConstraintD(BinConstraintD(BinConstraintD(f1, f2, op_sub), 1, op_sub), stride[0], op_div)
    # 计算 f3 的二进制约束，包含 f1、f2 和减法、除法操作
    f4 = BinConstraintD(f3, 1, op_add)
    # 计算 f4 的二进制约束，包含 f3 和加法操作

    c4 = Disj([b3_dyn, Conj([d3_not_dyn, BinConstraintD(d[2], f4, op_eq)])])
    # 创建 c4 的析取条件，包含 b3_dyn 和 d3_not_dyn 的合取，以及 d[2] 和 f4 的相等约束

    f11 = BinConstraintD(b4, BinConstraintD(2, padding[1], op_mul), op_add)
    # 计算 f11 的二进制约束，包含 b4、padding[1] 和乘法相加操作
    f22 = BinConstraintD(dilation[1], BinConstraintD(kernel[1], 1, op_sub), op_mul)
    # 计算 f22 的二进制约束，包含 dilation[1]、kernel[1] 和减法乘法操作
    f33 = BinConstraintD(BinConstraintD(BinConstraintD(f11, f22, op_sub), 1, op_sub), stride[1], op_div)
    # 计算 f33 的二进制约束，包含 f11、f22 和减法、除法操作
    f44 = BinConstraintD(f33, 1, op_add)
    # 计算 f44 的二进制约束，包含 f33 和加法操作

    c5 = Disj([b4_dyn, Conj([d4_not_dyn, BinConstraintD(d[3], f44, op_eq)])])
    # 创建 c5 的析取条件，包含 b4_dyn 和 d4_not_dyn 的合取，以及 d[3] 和 f44 的相等约束

    return c4, c5
# 生成所有可能性，即对于输入的每个维度变量，生成其等于或不等于dyn的所有可能性的约束
def generate_all_int_dyn_dim_possibilities(my_list: List[DVar]):
    """
    Generate all possibilities of being equal or not equal to dyn for my_list
    Args:
        my_list: List of tensor dimensions

    Returns: A list of a list of constraints. Each list of constraints corresponds to
    one possibility about the values of the dimension variables
    """
    # 生成等于dyn或不等于dyn的约束列表
    eq_possibilities = [BinConstraintD(my_list[i], Dyn, op_eq) for i in range(len(my_list))]
    neq_possibilities = [BinConstraintD(my_list[i], Dyn, op_neq) for i in range(len(my_list))]
    d_possibilities = []

    # 将等于和不等于dyn的约束组合成一对一对的列表
    for i in zip(eq_possibilities, neq_possibilities):
        d_possibilities.append(list(i))
    # 计算所有可能的组合情况
    all_possibilities = list(itertools.product(*d_possibilities))
    return all_possibilities


# 生成约束，检查目标维度是否可以被输入维度整除
def is_target_div_by_dim(target: List[int], dim: List[DVar]):
    """
    Generate constraints to check if the target dimensions are divisible by the input dimensions
    Args:
        target: Target dimensions
        dim: Input dimensions

    Returns: Constraints to check divisibility

    """
    # 构建约束，检查目标维度是否可以被输入维度整除
    return BinConstraintD(BinConstraintD(Prod(target), dim, op_mod), 0, op_eq)


# 生成约束，检查输入维度是否可以被目标维度整除
def is_dim_div_by_target(target: List[int], dim: List[DVar]):
    """
    Generate constraints to check if the input dimensions is divisible by the target dimensions
    Args:
        target: Target dimensions
        dim:  Input dimensions

    Returns: Constraints to check divisibility

    """
    # 构建约束，检查输入维度是否可以被目标维度整除
    return BinConstraintD(BinConstraintD(dim, Prod(target), op_mod), 0, op_eq)


# 生成所有reshape可能性的约束
def gen_all_reshape_possibilities(list_of_dims, target):
    """
    Consider all possibilities what the input dimensions could be (number or dynamic)
    Then generate the appropriate constraints using multiplication or mod depending on the possibility
    The possibilities we consider here are the cross product of being equal to dyn or not equal to dyn
    for the input. Target is fixed because at most one dimension could be dyn.
    We have different cases for this.

    Args:
        list_of_dims: The input list of dimensions
        target: The tensor we want to reshape to

    Returns: A disjunction of transformed reshape constraints

    """
    # 生成所有整数和动态维度的可能性
    all_possibilities = generate_all_int_dyn_dim_possibilities(list_of_dims)

    # 初始化约束列表
    all_constraints = []
    # 对所有可能性进行遍历
    for p in all_possibilities:
        # 初始化一个空列表，用于存储需要乘法的约束条件
        to_multiply = []

        # 将 p 转换为列表，便于后续操作
        p = list(p)

        # 遍历 p 中的每一个约束条件
        for constraint in p:
            # 断言 constraint 是 BinConstraintD 类的实例
            assert isinstance(constraint, BinConstraintD)
            # 如果约束条件是不等式 (op_neq)，将其左侧的值添加到 to_multiply 列表中
            if constraint.op == op_neq:
                to_multiply.append(constraint.lhs)

        # 如果 to_multiply 列表为空，将 p 转换为 Conj 对象并添加到 all_constraints 列表中
        if not to_multiply:
            all_constraints.append(Conj(p))

        # 如果 to_multiply 列表长度小于 list_of_dims 的长度
        elif len(to_multiply) < len(list_of_dims):
            # 将 p 扩展为包含一个新约束条件，该约束条件为 is_target_div_by_dim(target, Prod(to_multiply))
            all_constraints.append(Conj(p + [is_target_div_by_dim(target, Prod(to_multiply))]))

        # 否则，将 p 扩展为包含一个新约束条件，该约束条件为 BinConstraintD(Prod(list_of_dims), Prod(target), op_eq)
        else:
            all_constraints.append(Conj(p + [BinConstraintD(Prod(list_of_dims),
                                                            Prod(target), op_eq)]))

    # 返回 Disj 对象，其中包含所有的 Conj 对象列表 all_constraints
    return Disj(all_constraints)
# 定义一个函数，用于在给定的张量（tensor）输入中对指定索引的维度进行广播操作
def broadcast_dim(tensor_input1, tensor_input2, res1, res2, index, padding=False):
    """
    Apply broadcasting to the 'index' dimension of tensor_input1.
    Args:
        tensor_input1: 应表示为 [d1, ..., d_index, ...]，其中 d_index = 1
                       （表示第 index 维度的长度为 1）
        tensor_input2: 第二个输入张量
        res1: 广播后的结果1
        res2: 广播后的结果2
        index: 需要进行广播的维度的索引
        padding: 如果使用了填充，则 tensor_input1[index] 不存在

    Returns:
        返回一个约束列表，以应用于广播结果的约束条件
    """

    # 如果 tensor_input1 的指定索引处为 None，则应确保 padding 参数为 True
    if tensor_input1[index] is None:
        assert padding

    # 如果没有使用填充
    if not padding:
        # 那么输入应该具有相同的长度，因此它们在 "index" 维度上都应相等
        return Conj([BinConstraintD(tensor_input1[index], 1, op_eq),
                     BinConstraintD(res1[index], res2[index], op_eq),
                     BinConstraintD(res2[index], tensor_input2[index], op_eq)])
    else:
        # 如果存在填充，则不设置输入维度为 1，因为它不存在
        return Conj([BinConstraintD(res1[index], res2[index], op_eq),
                     BinConstraintD(res2[index], tensor_input2[index], op_eq)])


def apply_padding(e1_var: TVar,
                  e11: BinConstraintT,
                  e2: BinConstraintT,
                  e12: BinConstraintT,
                  d2: List[DVar],
                  d11: List[DVar],
                  d12: List[DVar],
                  counter: int):
    """
    We are considering the possibility where one input has less dimensions than
    another input, so we apply padding to the broadcasted results

    Args:
        e1_var: Variable representing the first input where padding will be
        e11: constraint of the form e11 = Tensortype[d1, ..., dn]
        e2:  constraint of the form e2 = Tensortype[d1, ..., dn]
        e12: constraint of the form e11 = Tensortype[d1, ..., dn]
        d2: Tensor variables for the second input
        d11: Tensor variables for the broadcasted first input
        d12: Tensor variables for the broadcasted second input
        counter: variable tracking

    Returns: A new constraint whose goal is to apply padding to the broadcasted result

    """

    res = []

    # 用 None 填充较短的输入，以便将其传递给广播辅助函数
    # 遍历从1到d2的长度减一的范围，生成索引i
    for i in range(1, len(d2)):

        # 调用函数gen_tensor_dims生成张量维度d1，并更新计数器counter
        d1, counter = gen_tensor_dims(i, counter)

        # 根据维度d1, d2, d11, d12生成自然约束条件
        nat_constraints = gen_nat_constraints(d1 + d2 + d11 + d12)

        # 创建二元约束条件对象e1，要求e1_var的张量类型与d1相等
        e1 = BinConstraintT(e1_var, TensorType(d1), op_eq)

        # 创建一个长度为(len(d2) - i)的空列表simulate_padding
        simulate_padding = [None] * (len(d2) - i)

        # 断言simulate_padding与d1的长度之和等于d2的长度
        assert len(simulate_padding + d1) == len(d2)

        # 创建空的广播填充列表broadcast_padding
        broadcast_padding = []

        # 对于每一个填充大小，考虑广播操作
        for j in range(len(d2) - i):
            # 调用broadcast_dim函数生成广播维度，并将结果添加到broadcast_padding列表中
            broadcast_padding.append(broadcast_dim(simulate_padding, d2, d11, d12, j, True))

        # 生成不带填充的所有广播可能性的约束条件
        all_broadcasting_possibilities = generate_all_broadcasting_possibilities_no_padding(d1,
                                                                                            d2[(len(d2) - i):],
                                                                                            d11[(len(d2) - i):],
                                                                                            d12[(len(d2) - i):])
        # 将所有约束条件合并成一个合取(conjunction)
        c = Conj([e1, e11, e2, e12,
                  *broadcast_padding,
                  all_broadcasting_possibilities,
                  *nat_constraints
                  ])
        # 将合取条件添加到结果列表res中
        res.append(c)

    # 返回一个不带填充的所有合取条件的析取(disjunction)以及更新后的计数器counter
    return Disj(res), counter
def no_broadcast_dim_with_index(d1: List[DVar],
                                d2: List[DVar],
                                d3: List[DVar],
                                d4: List[DVar],
                                i: int):
    """
    Args:
        d1: input 1
            - List of variables representing dimensions for input 1
        d2: input 2
            - List of variables representing dimensions for input 2
        d3: simulated broadcasting for input 1
            - List of variables representing dimensions for input 1 after simulated broadcasting
        d4: simulated broadcasting for input 2
            - List of variables representing dimensions for input 2 after simulated broadcasting
        i: the rank of the resulting tensor addition
            - Integer representing the index for which the constraints are applied

    Returns: Constraints for when no broadcasting occurs
    """
    return Conj([
        Disj([
            Conj([BinConstraintD(d1[i], 1, op_eq),
                  BinConstraintD(d2[i], 1, op_eq)]),
                # Constraint: d1[i] == 1 and d2[i] == 1

            Conj([BinConstraintD(d1[i], 1, op_neq),
                  BinConstraintD(d2[i], 1, op_neq)])]),
                # Constraint: d1[i] != 1 or d2[i] != 1

        BinConstraintD(d1[i], d3[i], op_eq),
            # Constraint: d1[i] == d3[i] (simulated broadcasting for input 1)

        BinConstraintD(d2[i], d4[i], op_eq)])
            # Constraint: d2[i] == d4[i] (simulated broadcasting for input 2)


def gen_lists_of_dims(num_tensors: int, dim_size: int, counter: int):
    """
    Generate lists of DVar to represent tensor dimensions
    Args:
        num_tensors: the required number of tensors
            - Integer specifying how many tensors to generate
        dim_size: the number of dimensions for each tensor
            - Integer specifying the size of each tensor's dimension list
        counter: variable tracking
            - Integer for tracking state or counting

    Returns: A list of a list of tensor dimensions
        - A list where each element is a list of variables representing dimensions
    """
    res = []

    for _ in range(num_tensors):
        dims, counter = gen_tensor_dims(dim_size, counter)
            # Generate tensor dimensions using helper function gen_tensor_dims
        res.append(dims)

    return res, counter


def create_equality_constraints_for_broadcasting(e1: TVar,
                                                 e2: TVar,
                                                 e11: TVar,
                                                 e12: TVar,
                                                 d1: List[DVar],
                                                 d2: List[DVar],
                                                 d11: List[DVar],
                                                 d12: List[DVar]):
    """
    Create equality constraints for when no broadcasting occurs
    Args:
        e1: Input 1
            - Tensor variable representing input 1
        e2: Input 2
            - Tensor variable representing input 2
        e11: Broadcasted input 1
            - Tensor variable representing input 1 after broadcasting
        e12: Broadcasted input 2
            - Tensor variable representing input 2 after broadcasting
        d1: Variables that store dimensions for e1
            - List of variables representing dimensions for input 1
        d2: Variables that store dimensions for e2
            - List of variables representing dimensions for input 2
        d11: Variables that store dimensions for e11
            - List of variables representing dimensions for broadcasted input 1
        d12: Variables that store dimensions for e22
            - List of variables representing dimensions for broadcasted input 2

    Returns: Four equality constraints
        - List of constraints ensuring equality between tensors and their respective dimensions
    """

    e1_tensor = BinConstraintT(e1, TensorType(d1), op_eq)
        # Constraint: e1 is of type TensorType with dimensions d1
    e11_tensor = BinConstraintT(e11, TensorType(d11), op_eq)
        # Constraint: e11 is of type TensorType with dimensions d11
    e2_tensor = BinConstraintT(e2, TensorType(d2), op_eq)
        # Constraint: e2 is of type TensorType with dimensions d2
    e12_tensor = BinConstraintT(e12, TensorType(d12), op_eq)
        # Constraint: e12 is of type TensorType with dimensions d12
    return [e1_tensor, e11_tensor, e2_tensor, e12_tensor]


def gen_consistency_constraints(constraint: Constraint, counter: int):
    """
    Args:
        constraint: Consistency constraint on tensors
            - Constraint object representing consistency constraints on tensors
        counter: for variable tracking
            - Integer used for tracking or counting

    Returns: Equality and consistency constraints on dimensions
        - Returns constraints related to the equality and consistency of dimensions
    """

    all_constraints = []
        # Initialize an empty list to store all constraints
    # 循环生成张量的约束条件，从秩 1 到 MAX_TENSOR_RANK
    for i in range(1, MAX_TENSOR_RANK + 1):
        # 生成第一个张量的维度及更新计数器
        new_dims_rhs_1, counter = gen_tensor_dims(i, counter)
        # 生成第二个张量的维度及更新计数器
        new_dims_rhs_2, counter = gen_tensor_dims(i, counter)

        # 生成自然约束条件，结合两个张量的维度
        nat_constraints = gen_nat_constraints(new_dims_rhs_1 + new_dims_rhs_2)

        # 创建张量约束对象 c_tensor_i，包括等式约束和维度一致性约束
        c_tensor_i = Conj([BinConstraintT(constraint.lhs, TensorType(new_dims_rhs_1), op_eq),
                           BinConstraintT(constraint.rhs, TensorType(new_dims_rhs_2), op_eq)] +
                          [BinConstraintD(d1, d2, op_consistency) for
                           d1, d2 in zip(new_dims_rhs_1, new_dims_rhs_2)] + nat_constraints)

        # 将当前张量约束对象加入总约束列表
        all_constraints.append(c_tensor_i)

    # 返回所有生成的约束列表以及最终更新后的计数器
    return all_constraints, counter
def gen_greatest_upper_bound(constraint: TGreatestUpperBound, counter: int):
    """
    Args:
        constraint: 最大张量上界约束
        counter: 变量追踪计数器

    Returns: 相等约束集合和最大张量上界约束集合
    """

    all_constraints = []

    # 循环遍历张量秩的可能取值范围
    for i in range(1, MAX_TENSOR_RANK + 1):
        c = []
        
        # 生成张量维度
        dims1, counter = gen_tensor_dims(i, counter)
        c1tensor = TensorType(dims1)

        dims2, counter = gen_tensor_dims(i, counter)
        c2tensor = TensorType(dims2)

        dims3, counter = gen_tensor_dims(i, counter)
        c3tensor = TensorType(dims3)

        # 添加二进制约束条件，要求三个张量类型与给定的约束条件相等
        c += [BinConstraintT(constraint.rhs1, c1tensor, op_eq),
              BinConstraintT(constraint.rhs2, c2tensor, op_eq),
              BinConstraintT(constraint.res, c3tensor, op_eq)] + \
             gen_nat_constraints(dims1 + dims2 + dims3)

        # 断言三个张量的维度数相等
        assert len(c3tensor.__args__) == len(c1tensor.__args__) == len(c2tensor.__args__)
        
        # 添加最大张量上界约束
        for j in range(len(c3tensor.__args__)):
            c.append(DGreatestUpperBound(c3tensor.__args__[j],
                                         c1tensor.__args__[j],
                                         c2tensor.__args__[j]))

        # 将所有约束添加到一个合取条件中
        all_constraints.append(Conj(c))
    
    # 返回所有生成的约束条件集合和最新的计数器值
    return all_constraints, counter


def generate_all_broadcasting_possibilities_no_padding(d1: List[DVar], d2: List[DVar], d11: List[DVar], d12: List[DVar]):
    """
    不考虑填充情况下生成广播约束。广播可以发生在任何维度上。
    Args:
        d1: 输入1的维度
        d2: 输入2的维度
        d11: 广播后输入1的维度
        d12: 广播后输入2的维度

    Returns: 关联输入维度和广播后维度的广播约束
    """

    size = len(d1)
    res2 = []

    # 对所有维度进行循环
    for i in range(size):
        # 计算在第i个维度上的广播约束
        t1 = broadcast_dim(d1, d2, d11, d12, i)
        t2 = broadcast_dim(d2, d1, d12, d11, i)
        t3 = no_broadcast_dim_with_index(d1, d2, d11, d12, i)

        # 将三种可能性作为析取条件添加到结果列表中
        res2.append(Disj([t1, t2, t3]))

    # 返回所有析取条件作为合取条件的结果
    return Conj(res2)


def gen_broadcasting_constraints(e1: TVar, e2: TVar, e11: TVar, e12: TVar, i: int, counter: int):
    """
    模拟对e1和e2进行广播，并返回结果分别在e11和e12中。由于渐进类型，
    e1和e2可能不相等。同样，e11和e12可能不相等。应保证e11和e12一致，
    因为它们表示广播后要相加的张量的形状。
    Args:
        e1: 表示输入1类型的TVar
        e2: 表示输入2类型的TVar
        e11: 表示广播后输入1类型的TVar
        e12: 表示广播后输入2类型的TVar
        i: 结果类型的秩
        counter: 变量追踪计数器

    Returns: 简化的广播约束
    """
    # 生成大小为 i 的维度列表和计数器
    dims, counter = gen_lists_of_dims(4, i, counter)
    # 将生成的维度列表解包为 d1, d2, d3, d4
    [d1, d2, d3, d4] = dims
    # 生成自然约束条件
    nat_dims_i = gen_nat_constraints(list(itertools.chain.from_iterable(dims)))

    # 创建不带填充的广播相等约束条件
    initialize_tensors_constraints = create_equality_constraints_for_broadcasting(e1, e2, e11, e12,
                                                                                  d1, d2, d3, d4)

    # 解包初始化的张量约束条件
    [e1_tensor, e11_tensor, e2_tensor, e12_tensor] = initialize_tensors_constraints

    # 创建不带填充的最终张量约束条件，包括所有广播可能性
    final_tensor_constraint_no_padding = Conj([*initialize_tensors_constraints,
                                               generate_all_broadcasting_possibilities_no_padding(d1, d2, d3, d4)])

    # 应用填充后，创建带填充的最终张量约束条件（第一种情况）
    final_tensor_constraint_padding_arg1, counter = \
        apply_padding(e1, e11_tensor, e2_tensor, e12_tensor, d2, d3, d4, counter)

    # 应用填充后，创建带填充的最终张量约束条件（第二种情况）
    final_tensor_constraint_padding_arg2, counter = \
        apply_padding(e2, e12_tensor, e1_tensor, e11_tensor, d1, d4, d3, counter)

    # 返回结果：不带填充的最终张量约束条件、两种填充情况的最终张量约束条件、自然约束条件、和更新后的计数器
    return final_tensor_constraint_no_padding, \
        final_tensor_constraint_padding_arg1, \
        final_tensor_constraint_padding_arg2, nat_dims_i, counter
```