# `.\pytorch\torch\fx\experimental\migrate_gradual_types\constraint.py`

```py
# mypy: allow-untyped-defs
# 导入需要的操作函数
from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_sub, op_mul, op_div, \
    op_mod, op_gt, op_lt, op_neq, op_eq
# 导入张量类型相关的类
from torch.fx.tensor_type import TensorType, Dyn


# 定义约束的基类
class Constraint:
    pass


# 定义并实现逻辑“与”约束类
class Conj(Constraint):
    def __init__(self, conjuncts):
        """
        :param conjuncts: 约束的逻辑“与”操作列表
        """
        self.conjucts = conjuncts

    def __eq__(self, other):
        if isinstance(other, Conj):
            return self.conjucts == other.conjucts and self.conjucts == other.conjucts
        else:
            return False

    def __repr__(self):
        return f'And({self.conjucts})'


# 定义并实现逻辑“或”约束类
class Disj(Constraint):
    def __init__(self, disjuncts):
        """
        :param disjuncts: 约束的逻辑“或”操作列表
        """
        self.disjuncts = disjuncts

    def __eq__(self, other):
        if isinstance(other, Disj):
            return self.disjuncts == other.disjuncts and self.disjuncts == other.disjuncts
        else:
            return False

    def __repr__(self):
        return f'Or({self.disjuncts})'


# 定义并实现乘积约束类
class Prod(Constraint):
    def __init__(self, products):
        """
        :param products: 需要相乘的维度列表
        """
        self.products = products

    def __eq__(self, other):
        if isinstance(other, Prod):
            return self.products == other.products and self.products == other.products
        else:
            return False

    def __repr__(self):
        return f'Product({self.products})'


# 定义并实现逻辑“真”约束类
class T(Constraint):
    """
    True
    """
    def __init__(self):
        pass

    def __eq__(self, other):
        return isinstance(other, T)

    def __repr__(self):
        return 'True'


# 定义并实现逻辑“假”约束类
class F(Constraint):
    """
    False
    """
    def __init__(self):
        pass

    def __eq__(self, other):
        return isinstance(other, F)

    def __repr__(self):
        return 'False'


# 定义二元约束基类
class BinaryConstraint(Constraint):
    """
    Represents all binary operations
    """
    def __init__(self, lhs, rhs, op):
        """
        :param lhs: 约束的左操作数
        :param rhs: 约束的右操作数
        :param op: 表示操作的字符串
        """
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def __eq__(self, other):
        if isinstance(other, BinaryConstraint):
            return self.lhs == other.lhs and self.rhs == other.rhs and self.op == other.op
        else:
            return False

    def __repr__(self):
        return f'({self.lhs} {self.op} {self.rhs})'


# 定义并实现关于张量的二元约束类
class BinConstraintT(BinaryConstraint):
    """
    Binary constraints about tensors
    """
    def __init__(self, lhs, rhs, op):
        assert (isinstance(lhs, (TVar, TensorType, int)) or lhs == Dyn) and \
               (isinstance(rhs, (TVar, TensorType, int)) or rhs == Dyn)
        super().__init__(lhs, rhs, op)

    def __eq__(self, other):
        return super().__eq__(other)
class BinConstraintD(BinaryConstraint):
    """
    Binary constraints about dimensions
    """
    def __init__(self, lhs, rhs, op):
        # 断言左操作数为代数表达式、维度或布尔表达式
        assert is_algebraic_expression(lhs) or is_dim(lhs) or is_bool_expr(lhs)
        # 断言右操作数为代数表达式、维度或布尔表达式
        assert is_algebraic_expression(rhs) or is_dim(rhs) or is_bool_expr(rhs)

        # 调用父类的初始化方法
        super().__init__(lhs, rhs, op)

    def __eq__(self, other):
        # 调用父类的相等性判断方法
        return super().__eq__(other)


class TGreatestUpperBound(Constraint):
    """
    Greatest Upper bound for tensors with dynamic type
    """
    def __init__(self, res, rhs1, rhs2):
        """
        :param res: tensor variable that stores the result of the output
        :param rhs1: tensor or tensor variable
        :param rhs2: tensor or tensor variable
        """
        # 初始化函数，将参数分配给实例变量
        self.res = res
        self.rhs1 = rhs1
        self.rhs2 = rhs2

    def __repr__(self):
        # 返回对象的字符串表示，显示最大上界的形式
        return f'{self.res} = {self.rhs1}\u2294*{self.rhs2}'

    def __eq__(self, other):
        # 判断对象是否相等的方法
        if isinstance(other, TGreatestUpperBound):
            return self.res == other.res and self.rhs1 == other.rhs1 and self.rhs2 == other.rhs2
        else:
            return False


class DGreatestUpperBound(Constraint):
    """
    Greatest Upper bound for dimensions
    """
    def __init__(self, res, rhs1, rhs2):
        """
        :param res: Dimension variable to store the result
        :param rhs1: dimension variable 1
        :param rhs2: dimension variable 2
        """
        # 断言确保所有参数都是维度变量
        assert is_dim(res)
        assert is_dim(rhs1)
        assert is_dim(rhs2)

        # 初始化函数，将参数分配给实例变量
        self.res = res
        self.rhs1 = rhs1
        self.rhs2 = rhs2

    def __repr__(self):
        # 返回对象的字符串表示，显示最大上界的形式
        return f'{self.res} = {self.rhs1}\u2294{self.rhs2}'

    def __eq__(self, other):
        # 判断对象是否相等的方法
        if isinstance(other, DGreatestUpperBound):
            return self.res == other.res and self.rhs1 == other.rhs1 and self.rhs2 == other.rhs2
        else:
            return False


class CanReshape(Constraint):
    """
    can_reshape constraint
    """
    def __init__(self, src, target):
        """
        :param src: tensor variable
        :param target: tensor
        """
        # 初始化函数，将参数分配给实例变量
        self.src = src
        self.target = target

    def __repr__(self):
        # 返回对象的字符串表示，显示可以重塑的约束形式
        return f'can-reshape({self.src}, {self.target})'

    def __eq__(self, other):
        # 判断对象是否相等的方法
        if isinstance(other, CanReshape):
            return self.src == other.src and self.target == other.target
        else:
            return False


class IndexSelect(Constraint):
    # 在这里继续补充注释


注释：
    # 初始化方法，用于创建 IndexSelect 对象
    def __init__(self, tensor_size, input_var, dim_replace, index, output):
        """
        Args:
            input_var: index_select 的输入变量
            tensor_size: 我们考虑的张量大小
            dim_replace: "index" 处输出的维度
            index: 要替换输入中维度的位置
            output: 存储结果的变量
        """
        # 断言确保输入变量和输出变量符合预期类型
        assert isinstance(input_var, TVar)
        assert isinstance(output, TVar)
        # 断言确保 dim_replace 是动态维度变量或者特定的维度变量
        assert isinstance(dim_replace, DVar) or dim_replace == Dyn
        assert isinstance(index, int)

        # 初始化对象的属性
        self.input_var = input_var
        self.tensor_size = tensor_size
        self.dim_replace = dim_replace
        self.index = index
        self.output = output

    # 返回对象的字符串表示形式，用于调试和显示
    def __repr__(self):
        return f' {self.output} = ' \
               f'IndexSelect({self.input_var}, ' \
               f'tensor_size: {self.tensor_size}, ' \
               f'{self.dim_replace}, ' \
               f'{self.index})'

    # 判断两个 IndexSelect 对象是否相等的方法
    def __eq__(self, other):
        if isinstance(other, IndexSelect):
            # 比较各个属性是否相等
            return self.tensor_size == other.tensor_size and \
                self.dim_replace == other.dim_replace and \
                self.index == other.index and \
                self.output == other.output and \
                self.input_var == other.input_var
        else:
            return False
class Transpose(Constraint):
    # Transpose 类，继承自 Constraint 约束类

    def __init__(self, tensor_size, input_var, index1, index2, output):
        """
        Args:
            tensor_size: 当前张量大小
            input_var: 用于保存输入的变量
            index1: 维度 1
            index2: 维度 2
            output: 用于存储结果的输出
        """
        assert isinstance(input_var, TVar)  # 断言 input_var 是 TVar 类型
        assert isinstance(output, TVar)  # 断言 output 是 TVar 类型
        assert isinstance(index1, int)  # 断言 index1 是整数类型
        assert isinstance(index2, int)  # 断言 index2 是整数类型

        self.input_var = input_var  # 初始化 input_var 实例变量
        self.tensor_size = tensor_size  # 初始化 tensor_size 实例变量
        self.index1 = index1  # 初始化 index1 实例变量
        self.index2 = index2  # 初始化 index2 实例变量
        self.output = output  # 初始化 output 实例变量

    def __repr__(self):
        # 返回对象的字符串表示形式
        return f' {self.output} = ' \
               f'Transpose({self.input_var}, ' \
               f'tensor_size: {self.tensor_size}, ' \
               f'{self.index1}, ' \
               f'{self.index2})'

    def __eq__(self, other):
        # 判断两个 Transpose 对象是否相等
        if isinstance(other, Transpose):
            return self.tensor_size == other.tensor_size and \
                self.index1 == other.index1 and \
                self.index2 == other.index2 and \
                self.output == other.output and \
                self.input_var == other.input_var
        else:
            return False


class GetItem(Constraint):
    # GetItem 类，继承自 Constraint 约束类

    def __init__(self, tensor_size, index, res, input_var):
        """
        Constraint for getting item given a tensor size
        :param tensor_size: 实际数字表示大小
        :param index: 实际数字表示索引
        :param res: 维度变量，用于携带我们获取的项目
        :param input_var: 我们将获取项目的张量变量
        """
        assert isinstance(res, DVar)  # 断言 res 是 DVar 类型

        self.res = res  # 初始化 res 实例变量
        self.tensor_size = tensor_size  # 初始化 tensor_size 实例变量
        self.index = index  # 初始化 index 实例变量
        self.input_var = input_var  # 初始化 input_var 实例变量

    def __repr__(self):
        # 返回对象的字符串表示形式
        return f' {self.res} = GetItem({self.input_var}, tensor_size: {self.tensor_size}, {self.index})'

    def __eq__(self, other):
        # 判断两个 GetItem 对象是否相等
        if isinstance(other, GetItem):
            return self.res == other.res and \
                self.tensor_size == other.tensor_size and \
                self.index == other.index and \
                self.input_var == other.input_var
        else:
            return False

class GetItemTensor(Constraint):
    # GetItemTensor 类，继承自 Constraint 约束类

    def __init__(self, tensor_size, index_tuple, res, input_var):
        """
        Constraint for getting item given a tensor size
        However, when the argument is a tuple, we will
        expect a tensor
        :param tensor_size: 实际数字表示秩
        :param index_tuple: 用于索引的元组
        :param res: 张量变量，用于携带我们获取的项目
        :param input_var: 我们将获取项目的张量变量
        """
        assert isinstance(res, TVar)  # 断言 res 是 TVar 类型

        self.res = res  # 初始化 res 实例变量
        self.tensor_size = tensor_size  # 初始化 tensor_size 实例变量
        self.index_tuple = index_tuple  # 初始化 index_tuple 实例变量
        self.input_var = input_var  # 初始化 input_var 实例变量
    # 定义对象的字符串表示形式，用于返回对象的描述信息
    def __repr__(self):
        # 返回包含对象属性的字符串表示形式，显示结果值、输入变量、张量尺寸和索引元组
        return f' {self.res} = GetItemT({self.input_var}, tensor_size: {self.tensor_size}, {self.index_tuple})'

    # 定义对象的相等性比较方法
    def __eq__(self, other):
        # 如果另一个对象是 GetItemTensor 类型，则比较对象的各属性是否相等
        if isinstance(other, GetItemTensor):
            return self.res == other.res and \
                self.tensor_size == other.tensor_size and \
                self.index_tuple == other.index_tuple and \
                self.input_var == other.input_var
        else:
            # 如果另一个对象不是 GetItemTensor 类型，则返回 False
            return False
class CalcConv(Constraint):
    def __init__(self, conv_result, input_var, c_out, kernel, padding, stride, dilation, matching_constraint_vars):
        """
        :param conv_result: 卷积的结果
        :param input_var: 卷积的输入
        :param c_out: 输出通道类型
        :param kernel: 卷积核的元组
        """
        self.conv_result = conv_result  # 初始化卷积结果属性
        self.input_var = input_var  # 初始化输入属性
        self.c_out = c_out  # 初始化输出通道类型属性
        self.kernel = kernel  # 初始化卷积核属性
        self.padding = padding  # 初始化填充属性
        self.stride = stride  # 初始化步长属性
        self.dilation = dilation  # 初始化扩展属性
        self.matching_constraint = matching_constraint_vars  # 初始化匹配约束属性

    def __repr__(self):
        """
        返回对象的字符串表示形式
        """
        return f'{self.conv_result} =' \
               f' calc-conv({self.input_var},' \
               f' {self.c_out}, {self.kernel}, ' \
               f'{self.padding}, {self.stride},' \
               f' {self.dilation})'

    def __eq__(self, other):
        """
        判断两个对象是否相等
        """
        if isinstance(other, CalcConv):
            return self.conv_result == other.conv_result and self.input_var == other.input_var and \
                self.c_out == other.c_out and self.kernel == other.kernel and self.padding == other.padding \
                and self.stride == other.stride and self.dilation == other.dilation \
                and self.matching_constraint == other.matching_constraint
        else:
            return False


class CalcMaxPool(Constraint):
    def __init__(self, maxpool_result, input_var, kernel, padding, stride, dilation, matching_constraint_vars):
        """
        :param maxpool_result: 最大池化的结果
        :param input_var: 池化的输入
        :param kernel: 池化核的元组
        """
        self.maxpool_result = maxpool_result  # 初始化最大池化结果属性
        self.input_var = input_var  # 初始化输入属性
        self.kernel = kernel  # 初始化池化核属性
        self.padding = padding  # 初始化填充属性
        self.stride = stride  # 初始化步长属性
        self.dilation = dilation  # 初始化扩展属性
        self.matching_constraint = matching_constraint_vars  # 初始化匹配约束属性

    def __repr__(self):
        """
        返回对象的字符串表示形式
        """
        return f'{self.maxpool_result} =' \
               f' calc-maxpool({self.input_var},' \
               f'  {self.kernel}, ' \
               f'{self.padding}, {self.stride},' \
               f' {self.dilation})'

    def __eq__(self, other):
        """
        判断两个对象是否相等
        """
        if isinstance(other, CalcMaxPool):
            return self.maxpool_result == other.maxpool_result and self.input_var == other.input_var \
                and self.kernel == other.kernel and self.padding == other.padding \
                and self.stride == other.stride and self.dilation == other.dilation \
                and self.matching_constraint == other.matching_constraint
        else:
            return False


class ApplyBroadcasting(Constraint):
    # 初始化方法，用于创建一个ApplyBroadcasting对象
    def __init__(self, res1, res2, input1, input2):
        """
        :param res1: resulting tensor 1 结果张量 1
        :param res2: resulting tensor 2 结果张量 2
        :param input1: tensor variable 1 张量变量 1
        :param input2: tensor variable 2 张量变量 2
        """
        # 将参数赋值给对象的属性
        self.res1 = res1  # 结果张量 1
        self.res2 = res2  # 结果张量 2
        self.input1 = input1  # 张量变量 1
        self.input2 = input2  # 张量变量 2

    # 等式判断方法，用于比较两个ApplyBroadcasting对象是否相等
    def __eq__(self, other):
        if isinstance(other, ApplyBroadcasting):  # 检查other是否为ApplyBroadcasting类型的对象
            # 比较各个属性是否相等，返回比较结果
            return self.res1 == other.res1 \
                and self.res2 == other.res2 \
                and self.input1 == other.input1 \
                and self.input2 == other.input2
        else:
            return False  # 如果other不是ApplyBroadcasting类型的对象，返回False

    # 字符串表示方法，返回对象的字符串表示
    def __repr__(self):
        # 返回格式化的字符串，描述ApplyBroadcasting对象的属性和内容
        return f'{self.res1}, {self.res2} = apply-broadcasting({self.input1}, {self.input2})'
class CalcProduct(Constraint):
    """
    Given correct dimensions, calculate the product for flatten accounting for Dyn
    """
    # 初始化函数，设置起始索引、结束索引、用于存储结果的变量以及需要进行flatten的维度类型列表
    def __init__(self, start, end, flattened, dims_to_flatten):
        """
        :param start: start index
        :param end: end index
        :param flattened: variable to store the product
        :param dims_to_flatten: the type which we will flatten
        """
        # 断言确保参数的类型正确
        assert isinstance(dims_to_flatten, list)
        assert isinstance(flattened, TVar)
        assert isinstance(start, int)
        assert isinstance(end, int)

        # 将参数赋值给实例变量
        self.start = start
        self.end = end
        self.dims_to_flatten = dims_to_flatten
        self.flattened = flattened

    # 判断两个 CalcProduct 对象是否相等的方法
    def __eq__(self, other):
        if isinstance(other, CalcProduct):
            return self.start == other.start and self.end == other.end and \
                self.dims_to_flatten == other.dims_to_flatten and self.flattened == other.flattened
        else:
            return False

    # 返回对象的字符串表示形式
    def __repr__(self):
        return f'{self.flattened} = CalcProduct({self.start}, {self.end}, {self.dims_to_flatten})'


class TVar:
    """
    Tensor variable with no tensor constructor
    """
    # 初始化函数，用于创建 Tensor 变量对象
    def __init__(self, tvar):
        """
        :param tvar: tensor variable
        """
        # 将参数赋值给实例变量
        self.tvar = tvar

    # 返回对象的字符串表示形式
    def __repr__(self):
        return f'TV({self.tvar})'

    # 判断两个 TVar 对象是否相等的方法
    def __eq__(self, other):
        if isinstance(other, TVar):
            return self.tvar == other.tvar
        else:
            return False


class DVar:
    """
    Dimension variable
    """
    # 初始化函数，用于创建 Dimension 变量对象
    def __init__(self, c):
        """
        :param c: character or number
        """
        # 将参数赋值给实例变量
        self.c = c

    # 返回对象的字符串表示形式
    def __repr__(self):
        return f'DV({self.c})'

    # 判断两个 DVar 对象是否相等的方法
    def __eq__(self, other):
        if isinstance(other, DVar):
            return self.c == other.c
        else:
            return False


class BVar:
    """
    Boolean variable
    """
    # 初始化函数，用于创建 Boolean 变量对象
    def __init__(self, c):
        """
        :param c: character or number
        """
        # 将参数赋值给实例变量
        self.c = c

    # 返回对象的字符串表示形式
    def __repr__(self):
        return f'BV({self.c})'

    # 判断两个 BVar 对象是否相等的方法
    def __eq__(self, other):
        if isinstance(other, BVar):
            return self.c == other.c
        else:
            return False


# 判断约束是否是代数表达式的方法
def is_algebraic_expression(constraint):
    if isinstance(constraint, BinConstraintD):
        return constraint.op in [op_add, op_sub, op_div, op_mul, op_mod]
    else:
        return isinstance(constraint, Prod)


# 判断约束是否是布尔表达式的方法
def is_bool_expr(constraint):
    if isinstance(constraint, BinConstraintD):
        return constraint.op in [op_gt, op_lt, op_neq, op_eq]
    else:
        return isinstance(constraint, (BVar, Conj, Disj))


# 判断变量是否是维度对象的方法
def is_dim(d):
    return isinstance(d, (DVar, int)) or d == Dyn
```