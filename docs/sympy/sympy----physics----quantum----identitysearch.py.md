# `D:\src\scipysrc\sympy\sympy\physics\quantum\identitysearch.py`

```
# 导入必要的库和模块
from collections import deque  # 导入 deque 数据结构
from sympy.core.random import randint  # 导入 randint 函数

from sympy.external import import_module  # 导入 import_module 函数，用于动态导入模块
from sympy.core.basic import Basic  # 导入 SymPy 核心基本类 Basic
from sympy.core.mul import Mul  # 导入 SymPy 核心乘法类 Mul
from sympy.core.numbers import Number, equal_valued  # 导入 SymPy 核心数字类 Number 和 equal_valued 函数
from sympy.core.power import Pow  # 导入 SymPy 核心幂函数类 Pow
from sympy.core.singleton import S  # 导入 SymPy 核心单例类 S
from sympy.physics.quantum.represent import represent  # 导入量子物理模块中的 represent 函数
from sympy.physics.quantum.dagger import Dagger  # 导入量子物理模块中的 Dagger 类

__all__ = [
    # 公共接口
    'generate_gate_rules',
    'generate_equivalent_ids',
    'GateIdentity',
    'bfs_identity_search',
    'random_identity_search',

    # "私有"函数
    'is_scalar_sparse_matrix',
    'is_scalar_nonsparse_matrix',
    'is_degenerate',
    'is_reducible',
]

np = import_module('numpy')  # 尝试导入 numpy 库，赋值给 np
scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})  # 尝试导入 scipy 库的 sparse 子模块，赋值给 scipy


def is_scalar_sparse_matrix(circuit, nqubits, identity_only, eps=1e-11):
    """Checks if a given scipy.sparse matrix is a scalar matrix.

    A scalar matrix is such that B = bI, where B is the scalar
    matrix, b is some scalar multiple, and I is the identity
    matrix.  A scalar matrix would have only the element b along
    it's main diagonal and zeroes elsewhere.

    Parameters
    ==========

    circuit : Gate tuple
        Sequence of quantum gates representing a quantum circuit
    nqubits : int
        Number of qubits in the circuit
    identity_only : bool
        Check for only identity matrices
    eps : number
        The tolerance value for zeroing out elements in the matrix.
        Values in the range [-eps, +eps] will be changed to a zero.
    """

    if not np or not scipy:
        pass  # 如果 numpy 或 scipy 未成功导入，则不执行任何操作

    matrix = represent(Mul(*circuit), nqubits=nqubits,
                       format='scipy.sparse')

    # 在某些情况下，represent 返回一个一维的标量值，而不是多维标量矩阵
    if isinstance(matrix, int):
        return matrix == 1 if identity_only else True

    # 如果 represent 返回一个矩阵，则检查该矩阵是否为对角矩阵，并且主对角线上的每个元素都相同
    else:
        # 由于浮点运算的影响，需要将稠密矩阵中非常小的元素置零
        # 参见参数的默认值。

        # 获取稠密矩阵的 ndarray 版本
        dense_matrix = matrix.todense().getA()
        # 由于复数值无法直接比较，需要将矩阵分解为实部和虚部
        # 寻找实部在 -eps 和 eps 之间的值
        bool_real = np.logical_and(dense_matrix.real > -eps,
                                   dense_matrix.real < eps)
        # 寻找虚部在 -eps 和 eps 之间的值
        bool_imag = np.logical_and(dense_matrix.imag > -eps,
                                   dense_matrix.imag < eps)
        # 将在 -eps 和 eps 之间的值替换为 0
        corrected_real = np.where(bool_real, 0.0, dense_matrix.real)
        corrected_imag = np.where(bool_imag, 0.0, dense_matrix.imag)
        # 将实部的矩阵转换为具有虚部的复数值
        corrected_imag = corrected_imag * complex(1j)
        # 重新组合实部和虚部成为修正后的稠密矩阵
        corrected_dense = corrected_real + corrected_imag

        # 检查是否为对角矩阵
        row_indices = corrected_dense.nonzero()[0]
        col_indices = corrected_dense.nonzero()[1]
        # 检查行索引和列索引是否相同
        # 如果相同，则表示矩阵只包含沿对角线的元素
        bool_indices = row_indices == col_indices
        is_diagonal = bool_indices.all()

        first_element = corrected_dense[0][0]
        # 如果第一个元素是零，则无法重新缩放矩阵
        # 而且肯定不是对角矩阵
        if (first_element == 0.0 + 0.0j):
            return False

        # 修正后的稠密矩阵的迹应该是 first_element 的倍数
        trace_of_corrected = (corrected_dense / first_element).trace()
        expected_trace = pow(2, nqubits)
        has_correct_trace = trace_of_corrected == expected_trace

        # 如果仅寻找单位矩阵
        # 第一个元素必须是 1
        real_is_one = abs(first_element.real - 1.0) < eps
        imag_is_zero = abs(first_element.imag) < eps
        is_one = real_is_one and imag_is_zero
        is_identity = is_one if identity_only else True
        return bool(is_diagonal and has_correct_trace and is_identity)
def _get_min_qubits(a_gate):
    # 如果给定门是一个幂运算，返回其底数门的最小量子比特数
    if isinstance(a_gate, Pow):
        return a_gate.base.min_qubits
    else:
        # 否则返回该门自身的最小量子比特数
        return a_gate.min_qubits


def ll_op(left, right):
    """Perform a LL operation.

    A LL operation multiplies both left and right circuits
    with the dagger of the left circuit's leftmost gate, and
    the dagger is multiplied on the left side of both circuits.

    If a LL is possible, it returns the new gate rule as a
    2-tuple (LHS, RHS), where LHS is the left circuit and
    and RHS is the right circuit of the new rule.
    If a LL is not possible, None is returned.

    Parameters
    ==========

    left : Gate tuple
        The left circuit of a gate rule expression.
    right : Gate tuple
        The right circuit of a gate rule expression.

    Examples
    ========

    Generate a new gate rule using a LL operation:

    >>> from sympy.physics.quantum.identitysearch import ll_op
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> ll_op((x, y, z), ())
    ((Y(0), Z(0)), (X(0),))

    >>> ll_op((y, z), (x,))
    ((Z(0),), (Y(0), X(0)))
    """
    # 省略了具体的 LL 操作实现细节
    pass
    # 如果左侧门列表不为空，则执行以下操作
    if (len(left) > 0):
        # 获取左侧列表的第一个门
        ll_gate = left[0]
        # 检查左侧门是否是单位矩阵
        ll_gate_is_unitary = is_scalar_matrix(
            (Dagger(ll_gate), ll_gate), _get_min_qubits(ll_gate), True)
    
    # 如果左侧门列表不为空且左侧门是单位矩阵
    if (len(left) > 0 and ll_gate_is_unitary):
        # 获取移除了最左侧门的新左侧列表
        new_left = left[1:len(left)]
        # 将最左侧门加入到右侧列表的左侧位置
        new_right = (Dagger(ll_gate),) + right
        # 返回新的门规则组合
        return (new_left, new_right)
    
    # 如果条件不满足，则返回空值
    return None
# 定义一个函数，执行 LR 操作

def lr_op(left, right):
    """Perform a LR operation.

    A LR operation multiplies both left and right circuits
    with the dagger of the left circuit's rightmost gate, and
    the dagger is multiplied on the right side of both circuits.

    If a LR is possible, it returns the new gate rule as a
    2-tuple (LHS, RHS), where LHS is the left circuit and
    and RHS is the right circuit of the new rule.
    If a LR is not possible, None is returned.

    Parameters
    ==========

    left : Gate tuple
        The left circuit of a gate rule expression.
    right : Gate tuple
        The right circuit of a gate rule expression.

    Examples
    ========

    Generate a new gate rule using a LR operation:

    >>> from sympy.physics.quantum.identitysearch import lr_op
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> lr_op((x, y, z), ())
    ((X(0), Y(0)), (Z(0),))

    >>> lr_op((x, y), (z,))
    ((X(0),), (Z(0), Y(0)))
    """

    # 检查左边电路是否非空
    if (len(left) > 0):
        # 获取左边电路最右侧的门
        lr_gate = left[len(left) - 1]
        # 检查最右侧门是否为单位矩阵
        lr_gate_is_unitary = is_scalar_matrix(
            (Dagger(lr_gate), lr_gate), _get_min_qubits(lr_gate), True)

    # 如果左边电路非空且最右侧门是单位矩阵
    if (len(left) > 0 and lr_gate_is_unitary):
        # 创建一个不包含最右侧门的新左边电路
        new_left = left[0:len(left) - 1]
        # 将最右侧门添加到右边电路的正确位置
        new_right = right + (Dagger(lr_gate),)
        # 返回新的门规则
        return (new_left, new_right)

    # 如果无法进行 LR 操作，则返回 None
    return None


# 定义一个函数，执行 RL 操作

def rl_op(left, right):
    """Perform a RL operation.

    A RL operation multiplies both left and right circuits
    with the dagger of the right circuit's leftmost gate, and
    the dagger is multiplied on the left side of both circuits.

    If a RL is possible, it returns the new gate rule as a
    2-tuple (LHS, RHS), where LHS is the left circuit and
    and RHS is the right circuit of the new rule.
    If a RL is not possible, None is returned.

    Parameters
    ==========

    left : Gate tuple
        The left circuit of a gate rule expression.
    right : Gate tuple
        The right circuit of a gate rule expression.

    Examples
    ========

    Generate a new gate rule using a RL operation:

    >>> from sympy.physics.quantum.identitysearch import rl_op
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> rl_op((x,), (y, z))
    ((Y(0), X(0)), (Z(0),))

    >>> rl_op((x, y), (z,))
    ((Z(0), X(0), Y(0)), ())
    """

    # 检查右边电路是否非空
    if (len(right) > 0):
        # 获取右边电路最左侧的门
        rl_gate = right[0]
        # 检查最左侧门是否为单位矩阵
        rl_gate_is_unitary = is_scalar_matrix(
            (Dagger(rl_gate), rl_gate), _get_min_qubits(rl_gate), True)
    # 如果右侧操作序列非空且 rl_gate_is_unitary 为真，则执行以下操作
    if (len(right) > 0 and rl_gate_is_unitary):
        # 去除右侧操作序列的最左边的门（即第一个元素）
        new_right = right[1:len(right)]
        # 将 rl_gate 的逆操作添加到左侧操作序列的最左边
        new_left = (Dagger(rl_gate),) + left
        # 返回更新后的操作规则，其中左右操作序列都已更新
        return (new_left, new_right)
    
    # 如果条件不满足，则返回空值
    return None
def rr_op(left, right):
    """Perform a RR operation.

    A RR operation multiplies both left and right circuits
    with the dagger of the right circuit's rightmost gate, and
    the dagger is multiplied on the right side of both circuits.

    If a RR is possible, it returns the new gate rule as a
    2-tuple (LHS, RHS), where LHS is the left circuit and
    and RHS is the right circuit of the new rule.
    If a RR is not possible, None is returned.

    Parameters
    ==========

    left : Gate tuple
        The left circuit of a gate rule expression.
    right : Gate tuple
        The right circuit of a gate rule expression.

    Examples
    ========

    Generate a new gate rule using a RR operation:

    >>> from sympy.physics.quantum.identitysearch import rr_op
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> rr_op((x, y), (z,))
    ((X(0), Y(0), Z(0)), ())

    >>> rr_op((x,), (y, z))
    ((X(0), Z(0)), (Y(0),))
    """

    # Check if the right circuit is not empty
    if (len(right) > 0):
        # Get the rightmost gate from the right circuit
        rr_gate = right[len(right) - 1]
        # Check if the rightmost gate is unitary
        rr_gate_is_unitary = is_scalar_matrix(
            (Dagger(rr_gate), rr_gate), _get_min_qubits(rr_gate), True)

    # If the right circuit is not empty and the rightmost gate is unitary
    if (len(right) > 0 and rr_gate_is_unitary):
        # Create a new right circuit without the rightmost gate
        new_right = right[0:len(right) - 1]
        # Append the dagger of the rightmost gate to the left circuit
        new_left = left + (Dagger(rr_gate),)
        # Return the new gate rule as a tuple of left and right circuits
        return (new_left, new_right)

    # Return None if a RR operation is not possible
    return None


def generate_gate_rules(gate_seq, return_as_muls=False):
    """Returns a set of gate rules.  Each gate rules is represented
    as a 2-tuple of tuples or Muls.  An empty tuple represents an arbitrary
    scalar value.

    This function uses the four operations (LL, LR, RL, RR)
    to generate the gate rules.

    A gate rule is an expression such as ABC = D or AB = CD, where
    A, B, C, and D are gates.  Each value on either side of the
    equal sign represents a circuit.  The four operations allow
    one to find a set of equivalent circuits from a gate identity.
    The letters denoting the operation tell the user what
    activities to perform on each expression.  The first letter
    indicates which side of the equal sign to focus on.  The
    second letter indicates which gate to focus on given the
    side.  Once this information is determined, the inverse
    of the gate is multiplied on both circuits to create a new
    gate rule.

    For example, given the identity, ABCD = 1, a LL operation
    means look at the left value and multiply both left sides by the
    inverse of the leftmost gate A.  If A is Hermitian, the inverse
    of A is still A.  The resulting new rule is BCD = A.

    The following is a summary of the four operations.  Assume
    """
    # 如果 gate_seq 是一个数字（标量矩阵）
    if isinstance(gate_seq, Number):
        # 如果 return_as_muls 为 True，则返回一个包含单一元组 (1, 1) 的集合
        if return_as_muls:
            return {(S.One, S.One)}
        else:
            # 否则返回一个包含空元组的集合
            return {((), ())}

    # 如果 gate_seq 是一个 Mul 对象，则提取其参数作为 gate_seq
    elif isinstance(gate_seq, Mul):
        gate_seq = gate_seq.args

    # queue 是一个双端队列，每个元素都是一个三元组：
    #   i)   第一个元素是等式的左侧
    #   ii)  第二个元素是等式的右侧
    #   iii) 第三个元素是已执行的操作数量
    queue = deque()
    
    # rules 是一个集合，用于存储门规则
    rules = set()
    
    # max_ops 是要执行的最大操作数量，等于 gate_seq 的长度
    max_ops = len(gate_seq)
    # 定义处理新规则的函数，将新规则添加到规则集合中，同时将操作次数加一后添加到队列中
    def process_new_rule(new_rule, ops):
        # 如果新规则不为空
        if new_rule is not None:
            # 解包新规则
            new_left, new_right = new_rule

            # 如果新规则不在规则集合中，并且其反向规则也不在规则集合中，则添加该新规则到规则集合中
            if new_rule not in rules and (new_right, new_left) not in rules:
                rules.add(new_rule)

            # 如果操作次数加一后未达到最大操作数限制
            if ops + 1 < max_ops:
                # 将新规则和增加后的操作次数添加到队列中
                queue.append(new_rule + (ops + 1,))

    # 将初始门序列和空元组作为初始规则添加到队列中
    queue.append((gate_seq, (), 0))

    # 将初始门序列和空元组作为初始规则添加到规则集合中
    rules.add((gate_seq, ()))

    # 当队列不为空时循环处理
    while len(queue) > 0:
        # 从队列左侧取出左部、右部和操作次数
        left, right, ops = queue.popleft()

        # 执行 LL 操作
        new_rule = ll_op(left, right)
        # 处理新规则
        process_new_rule(new_rule, ops)

        # 执行 LR 操作
        new_rule = lr_op(left, right)
        # 处理新规则
        process_new_rule(new_rule, ops)

        # 执行 RL 操作
        new_rule = rl_op(left, right)
        # 处理新规则
        process_new_rule(new_rule, ops)

        # 执行 RR 操作
        new_rule = rr_op(left, right)
        # 处理新规则
        process_new_rule(new_rule, ops)

    # 如果需要将规则作为乘法表达式返回
    if return_as_muls:
        # 将每个规则元组转换为乘法表达式，并添加到乘法规则集合中
        mul_rules = set()
        for rule in rules:
            left, right = rule
            mul_rules.add((Mul(*left), Mul(*right)))

        # 将规则集合更新为乘法规则集合
        rules = mul_rules

    # 返回最终的规则集合
    return rules
def generate_equivalent_ids(gate_seq, return_as_muls=False):
    """Returns a set of equivalent gate identities.

    A gate identity is a quantum circuit such that the product
    of the gates in the circuit is equal to a scalar value.
    For example, XYZ = i, where X, Y, Z are the Pauli gates and
    i is the imaginary value, is considered a gate identity.

    This function uses the four operations (LL, LR, RL, RR)
    to generate the gate rules and, subsequently, to locate equivalent
    gate identities.

    Note that all equivalent identities are reachable in n operations
    from the starting gate identity, where n is the number of gates
    in the sequence.

    The max number of gate identities is 2n, where n is the number
    of gates in the sequence (unproven).

    Parameters
    ==========

    gate_seq : Gate tuple, Mul, or Number
        A variable length tuple or Mul of Gates whose product is equal to
        a scalar matrix.
    return_as_muls: bool
        True to return as Muls; False to return as tuples

    Examples
    ========

    Find equivalent gate identities from the current circuit with tuples:

    >>> from sympy.physics.quantum.identitysearch import generate_equivalent_ids
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> generate_equivalent_ids((x, x))
    {(X(0), X(0))}

    >>> generate_equivalent_ids((x, y, z))
    {(X(0), Y(0), Z(0)), (X(0), Z(0), Y(0)), (Y(0), X(0), Z(0)),
     (Y(0), Z(0), X(0)), (Z(0), X(0), Y(0)), (Z(0), Y(0), X(0))}

    Find equivalent gate identities from the current circuit with Muls:

    >>> generate_equivalent_ids(x*x, return_as_muls=True)
    {1}

    >>> generate_equivalent_ids(x*y*z, return_as_muls=True)
    {X(0)*Y(0)*Z(0), X(0)*Z(0)*Y(0), Y(0)*X(0)*Z(0),
     Y(0)*Z(0)*X(0), Z(0)*X(0)*Y(0), Z(0)*Y(0)*X(0)}
    """

    # If gate_seq is a scalar (Number), return the identity element
    if isinstance(gate_seq, Number):
        return {S.One}
    # If gate_seq is a product (Mul) of gates, extract individual gates
    elif isinstance(gate_seq, Mul):
        gate_seq = gate_seq.args

    # Initialize an empty set to store equivalent gate identities
    eq_ids = set()

    # Generate gate rules based on the given gate sequence
    gate_rules = generate_gate_rules(gate_seq)

    # Iterate through each gate rule
    for rule in gate_rules:
        l, r = rule
        # Check if the left side of the rule is empty, add right to eq_ids
        if l == ():
            eq_ids.add(r)
        # Check if the right side of the rule is empty, add left to eq_ids
        elif r == ():
            eq_ids.add(l)

    # Convert eq_ids to Muls if return_as_muls is True
    if return_as_muls:
        convert_to_mul = lambda id_seq: Mul(*id_seq)
        eq_ids = set(map(convert_to_mul, eq_ids))

    return eq_ids
    def __new__(cls, *args):
        # 使用基类 Basic 的构造方法创建新的 GateIdentity 对象
        obj = Basic.__new__(cls, *args)
        # 根据传入的参数 args 构建乘积表达式，表示电路的乘积形式
        obj._circuit = Mul(*args)
        # 根据传入的参数 args 生成门操作的规则
        obj._rules = generate_gate_rules(args)
        # 根据传入的参数 args 生成等效门操作的组合形式
        obj._eq_ids = generate_equivalent_ids(args)

        return obj

    @property
    def circuit(self):
        # 返回 GateIdentity 对象的电路表示形式
        return self._circuit

    @property
    def gate_rules(self):
        # 返回 GateIdentity 对象的门操作规则
        return self._rules

    @property
    def equivalent_ids(self):
        # 返回 GateIdentity 对象的等效门操作的组合形式集合
        return self._eq_ids

    @property
    def sequence(self):
        # 返回 GateIdentity 对象的参数 args，即门操作的序列
        return self.args

    def __str__(self):
        """Returns the string of gates in a tuple."""
        # 返回 GateIdentity 对象的电路表示形式的字符串形式
        return str(self.circuit)
# 从给定的门列表构建门身份的集合
def bfs_identity_search(gate_list, nqubits, max_depth=None,
       identity_only=False):
    """Constructs a set of gate identities from the list of possible gates.

    Performs a breadth first search over the space of gate identities.
    This allows the finding of the shortest gate identities first.

    Parameters
    ==========

    gate_list : list, Gate
        A list of Gates from which to search for gate identities.
    # 如果 max_depth 未指定或者小于等于 0，则设为 gate_list 的长度
    if max_depth is None or max_depth <= 0:
        max_depth = len(gate_list)

    # 根据 identity_only 参数确定是否只搜索可以约化为单位矩阵的门的身份
    id_only = identity_only

    # 使用双端队列创建一个空的初始序列（隐含包含一个 IdentityGate）
    queue = deque([()])

    # 创建一个空的集合，用于存储找到的门的身份
    ids = set()

    # 开始在给定空间中搜索门的身份
    while (len(queue) > 0):
        # 从队列左侧取出当前电路序列
        current_circuit = queue.popleft()

        # 遍历门的列表，依次将每个门添加到当前电路序列中
        for next_gate in gate_list:
            new_circuit = current_circuit + (next_gate,)

            # 判断是否严格子电路是标量矩阵
            circuit_reducible = is_reducible(new_circuit, nqubits,
                                             1, len(new_circuit))

            # 如果新序列可以约化为单位矩阵，并且不是退化的，并且不是标量矩阵
            if (is_scalar_matrix(new_circuit, nqubits, id_only) and
                not is_degenerate(ids, new_circuit) and
                not circuit_reducible):
                # 将门的身份添加到集合中
                ids.add(GateIdentity(*new_circuit))

            # 如果新序列的长度小于 max_depth，并且不是标量矩阵
            elif (len(new_circuit) < max_depth and
                  not circuit_reducible):
                # 将新序列添加到队列中继续搜索
                queue.append(new_circuit)

    # 返回找到的门的身份集合
    return ids
`
# 随机选择 gate_list 中的 numgates 个门，并检查是否形成门身份识别电路

def random_identity_search(gate_list, numgates, nqubits):
    """Randomly selects numgates from gate_list and checks if it is
    a gate identity.

    If the circuit is a gate identity, the circuit is returned;
    Otherwise, None is returned.
    """

    gate_size = len(gate_list)  # 获取 gate_list 的长度
    circuit = ()  # 初始化电路为空元组

    for i in range(numgates):  # 遍历 numgates 次
        next_gate = gate_list[randint(0, gate_size - 1)]  # 从 gate_list 中随机选择一个门
        circuit = circuit + (next_gate,)  # 将选中的门添加到电路中

    is_scalar = is_scalar_matrix(circuit, nqubits, False)  # 判断电路是否为标量矩阵

    return circuit if is_scalar else None  # 如果电路是标量矩阵则返回电路，否则返回 None
```