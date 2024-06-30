# `D:\src\scipysrc\sympy\sympy\physics\quantum\circuitutils.py`

```
# 导入所需模块和函数
from functools import reduce

# 导入排序相关的函数
from sympy.core.sorting import default_sort_key
# 导入元组容器类
from sympy.core.containers import Tuple
# 导入乘法相关类
from sympy.core.mul import Mul
# 导入符号类
from sympy.core.symbol import Symbol
# 导入 sympify 函数，用于将字符串转换为符号表达式
from sympy.core.sympify import sympify
# 导入用于生成编号符号的函数
from sympy.utilities import numbered_symbols
# 导入量子门类
from sympy.physics.quantum.gate import Gate

# 将以下变量添加到 __all__ 中，便于外部访问
__all__ = [
    'kmp_table',
    'find_subcircuit',
    'replace_subcircuit',
    'convert_to_symbolic_indices',
    'convert_to_real_indices',
    'random_reduce',
    'random_insert'
]

def kmp_table(word):
    """构建 Knuth-Morris-Pratt 算法的 '部分匹配' 表格。

    注意：适用于字符串或表示量子电路的元组。
    """

    # 当前在子电路中的位置
    pos = 2
    # 可能稍后在 word 中重新出现的候选子字符串的起始位置
    cnd = 0
    # '部分匹配' 表格，用于帮助确定下一个开始子字符串搜索的位置
    table = []
    table.append(-1)
    table.append(0)

    while pos < len(word):
        if word[pos - 1] == word[cnd]:
            cnd = cnd + 1
            table.append(cnd)
            pos = pos + 1
        elif cnd > 0:
            cnd = table[cnd]
        else:
            table.append(0)
            pos = pos + 1

    return table


def find_subcircuit(circuit, subcircuit, start=0, end=0):
    """在 circuit 中查找 subcircuit 是否存在。

    解释
    ==============

    如果 subcircuit 存在，则返回其在 circuit 中起始位置的索引；
    否则返回 -1。使用的算法是 Knuth-Morris-Pratt 算法。

    参数
    ==============

    circuit : tuple, Gate or Mul
        表示量子电路的 Gate 或 Mul 元组
    subcircuit : tuple, Gate or Mul
        要在 circuit 中查找的 Gate 或 Mul 元组
    start : int
        开始查找 subcircuit 的位置。如果 start 与 end 相同或超过 end，则返回 -1。
    end : int
        查找 subcircuit 的最后位置。如果 end 小于 1，则取 circuit 的长度为 end。

    示例
    ==============

    查找第一个 subcircuit 的实例：

    >>> from sympy.physics.quantum.circuitutils import find_subcircuit
    >>> from sympy.physics.quantum.gate import X, Y, Z, H
    >>> circuit = X(0)*Z(0)*Y(0)*H(0)
    >>> subcircuit = Z(0)*Y(0)
    >>> find_subcircuit(circuit, subcircuit)
    1

    从特定位置开始查找第一个实例：

    >>> find_subcircuit(circuit, subcircuit, start=1)
    1

    >>> find_subcircuit(circuit, subcircuit, start=2)
    -1

    >>> circuit = circuit*subcircuit
    >>> find_subcircuit(circuit, subcircuit, start=2)
    4

    在某个区间内查找 subcircuit：

    >>> find_subcircuit(circuit, subcircuit, start=2, end=2)
    -1
    """

    if isinstance(circuit, Mul):
        circuit = circuit.args
    # 如果子电路是乘法表达式的实例，将其转换为其参数列表
    if isinstance(subcircuit, Mul):
        subcircuit = subcircuit.args

    # 如果子电路长度为0或超过电路长度，返回-1表示没有找到匹配
    if len(subcircuit) == 0 or len(subcircuit) > len(circuit):
        return -1

    # 如果结束位置小于1，则将其设置为电路的长度
    if end < 1:
        end = len(circuit)

    # 在电路中的起始位置
    pos = start
    # 在子电路中的位置
    index = 0
    # Knuth-Morris-Pratt（KMP）算法的部分匹配表
    table = kmp_table(subcircuit)

    # 使用KMP算法在电路中查找子电路的位置
    while (pos + index) < end:
        if subcircuit[index] == circuit[pos + index]:
            index = index + 1
        else:
            # 如果不匹配，根据部分匹配表调整位置
            pos = pos + index - table[index]
            index = table[index] if table[index] > -1 else 0

        # 如果index等于子电路长度，表示找到了匹配位置，返回起始位置pos
        if index == len(subcircuit):
            return pos

    # 如果未找到匹配，返回-1
    return -1
# 将子电路替换为另一个子电路，如果存在的话。
def replace_subcircuit(circuit, subcircuit, replace=None, pos=0):
    """Replaces a subcircuit with another subcircuit in circuit,
    if it exists.

    Explanation
    ===========

    If multiple instances of subcircuit exists, the first instance is
    replaced.  The position to begin searching from (if different from
    0) may be optionally given.  If subcircuit cannot be found, circuit
    is returned.

    Parameters
    ==========

    circuit : tuple, Gate or Mul
        A quantum circuit.
    subcircuit : tuple, Gate or Mul
        The circuit to be replaced.
    replace : tuple, Gate or Mul
        The replacement circuit.
    pos : int
        The location to start search and replace
        subcircuit, if it exists.  This may be used
        if it is known beforehand that multiple
        instances exist, and it is desirable to
        replace a specific instance.  If a negative number
        is given, pos will be defaulted to 0.

    Examples
    ========

    Find and remove the subcircuit:

    >>> from sympy.physics.quantum.circuitutils import replace_subcircuit
    >>> from sympy.physics.quantum.gate import X, Y, Z, H
    >>> circuit = X(0)*Z(0)*Y(0)*H(0)*X(0)*H(0)*Y(0)
    >>> subcircuit = Z(0)*Y(0)
    >>> replace_subcircuit(circuit, subcircuit)
    (X(0), H(0), X(0), H(0), Y(0))

    Remove the subcircuit given a starting search point:

    >>> replace_subcircuit(circuit, subcircuit, pos=1)
    (X(0), H(0), X(0), H(0), Y(0))

    >>> replace_subcircuit(circuit, subcircuit, pos=2)
    (X(0), Z(0), Y(0), H(0), X(0), H(0), Y(0))

    Replace the subcircuit:

    >>> replacement = H(0)*Z(0)
    >>> replace_subcircuit(circuit, subcircuit, replace=replacement)
    (X(0), H(0), Z(0), H(0), X(0), H(0), Y(0))
    """

    if pos < 0:
        pos = 0  # 如果位置为负数，将其设为默认值 0

    if isinstance(circuit, Mul):
        circuit = circuit.args  # 如果电路是 Mul 类型，取其参数

    if isinstance(subcircuit, Mul):
        subcircuit = subcircuit.args  # 如果子电路是 Mul 类型，取其参数

    if isinstance(replace, Mul):
        replace = replace.args  # 如果替换电路是 Mul 类型，取其参数
    elif replace is None:
        replace = ()  # 如果没有指定替换电路，设为空元组

    # 在位置 pos 开始查找子电路
    loc = find_subcircuit(circuit, subcircuit, start=pos)

    # 如果找到子电路
    if loc > -1:
        # 获取子电路左边的门
        left = circuit[0:loc]
        # 获取子电路右边的门
        right = circuit[loc + len(subcircuit):len(circuit)]
        # 重新组合左右两侧的门形成新电路
        circuit = left + replace + right

    return circuit


def _sympify_qubit_map(mapping):
    new_map = {}
    for key in mapping:
        new_map[key] = sympify(mapping[key])
    return new_map


def convert_to_symbolic_indices(seq, start=None, gen=None, qubit_map=None):
    """Returns the circuit with symbolic indices and the
    dictionary mapping symbolic indices to real indices.

    The mapping is 1 to 1 and onto (bijective).

    Parameters
    ==========

    seq : list
        A sequence representing the quantum circuit.
    start : int or None
        Optional starting index for symbolic indices.
    gen : generator or None
        Optional generator for generating symbolic indices.
    qubit_map : dict or None
        Optional mapping from symbolic indices to real indices.

    Returns
    =======

    tuple
        A tuple containing:
        - The modified circuit with symbolic indices.
        - A dictionary mapping symbolic indices to real indices.

    Examples
    ========

    (Examples will be provided in the subsequent continuation.)
    """
    """
    seq : tuple, Gate/Integer/tuple or Mul
        A tuple of Gate, Integer, or tuple objects, or a Mul
    start : Symbol
        An optional starting symbolic index
    gen : object
        An optional numbered symbol generator
    qubit_map : dict
        An existing mapping of symbolic indices to real indices

    All symbolic indices have the format 'i#', where # is
    some number >= 0.
    """

    if isinstance(seq, Mul):
        # If seq is an instance of Mul, unpack its arguments
        seq = seq.args

    # A numbered symbol generator
    index_gen = numbered_symbols(prefix='i', start=-1)
    # Obtain the first symbol from the generator
    cur_ndx = next(index_gen)

    # keys are symbolic indices; values are real indices
    ndx_map = {}

    def create_inverse_map(symb_to_real_map):
        # Function to create an inverse mapping of symb_to_real_map
        rev_items = lambda item: (item[1], item[0])
        return dict(map(rev_items, symb_to_real_map.items()))

    if start is not None:
        if not isinstance(start, Symbol):
            # Raise an error if start is not an instance of Symbol
            msg = 'Expected Symbol for starting index, got %r.' % start
            raise TypeError(msg)
        cur_ndx = start

    if gen is not None:
        if not isinstance(gen, numbered_symbols().__class__):
            # Raise an error if gen is not an instance of the same class as numbered_symbols()
            msg = 'Expected a generator, got %r.' % gen
            raise TypeError(msg)
        index_gen = gen

    if qubit_map is not None:
        if not isinstance(qubit_map, dict):
            # Raise an error if qubit_map is not a dictionary
            msg = ('Expected dict for existing map, got ' +
                   '%r.' % qubit_map)
            raise TypeError(msg)
        ndx_map = qubit_map

    # Ensure that ndx_map contains symbolic keys
    ndx_map = _sympify_qubit_map(ndx_map)
    # Create an inverse map where keys are real indices and values are symbolic indices
    inv_map = create_inverse_map(ndx_map)

    sym_seq = ()
    for item in seq:
        # If item is an instance of Gate, convert its arguments to symbolic indices
        if isinstance(item, Gate):
            result = convert_to_symbolic_indices(item.args,
                                                 qubit_map=ndx_map,
                                                 start=cur_ndx,
                                                 gen=index_gen)
            sym_item, new_map, cur_ndx, index_gen = result
            ndx_map.update(new_map)
            inv_map = create_inverse_map(ndx_map)

        elif isinstance(item, (tuple, Tuple)):
            # If item is a tuple, recursively convert its elements to symbolic indices
            result = convert_to_symbolic_indices(item,
                                                 qubit_map=ndx_map,
                                                 start=cur_ndx,
                                                 gen=index_gen)
            sym_item, new_map, cur_ndx, index_gen = result
            ndx_map.update(new_map)
            inv_map = create_inverse_map(ndx_map)

        elif item in inv_map:
            # If item is already in inv_map, use its corresponding symbolic index
            sym_item = inv_map[item]

        else:
            # Generate a new symbolic index and update mappings
            cur_ndx = next(gen)
            ndx_map[cur_ndx] = item
            inv_map[item] = cur_ndx
            sym_item = cur_ndx

        if isinstance(item, Gate):
            # If item is a Gate instance, recreate it with symbolic indices
            sym_item = item.__class__(*sym_item)

        # Append the symbolic item to sym_seq
        sym_seq = sym_seq + (sym_item,)

    # Return the tuple of symbolic sequence, ndx_map, current index, and generator
    return sym_seq, ndx_map, cur_ndx, index_gen
# 定义函数，将符号索引转换为实际索引
def convert_to_real_indices(seq, qubit_map):
    """Returns the circuit with real indices.

    Parameters
    ==========

    seq : tuple, Gate/Integer/tuple or Mul
        A tuple of Gate, Integer, or tuple objects or a Mul
    qubit_map : dict
        A dictionary mapping symbolic indices to real indices.

    Examples
    ========

    Change the symbolic indices to real integers:

    >>> from sympy import symbols
    >>> from sympy.physics.quantum.circuitutils import convert_to_real_indices
    >>> from sympy.physics.quantum.gate import X, Y, H
    >>> i0, i1 = symbols('i:2')
    >>> index_map = {i0 : 0, i1 : 1}
    >>> convert_to_real_indices(X(i0)*Y(i1)*H(i0)*X(i1), index_map)
    (X(0), Y(1), H(0), X(1))
    """

    # 如果输入的 seq 是 Mul 对象，则提取其 args
    if isinstance(seq, Mul):
        seq = seq.args

    # 检查 qubit_map 是否为字典类型，否则抛出类型错误
    if not isinstance(qubit_map, dict):
        msg = 'Expected dict for qubit_map, got %r.' % qubit_map
        raise TypeError(msg)

    # 将 qubit_map 的值转换为符号表达式
    qubit_map = _sympify_qubit_map(qubit_map)
    # 初始化一个空的实数索引序列
    real_seq = ()
    # 遍历 seq 中的每个元素
    for item in seq:
        # 如果 item 是 Gate 对象，则递归地转换其参数为实数索引
        if isinstance(item, Gate):
            real_item = convert_to_real_indices(item.args, qubit_map)

        # 如果 item 是 tuple 或 Tuple 对象，则递归地转换其中的元素为实数索引
        elif isinstance(item, (tuple, Tuple)):
            real_item = convert_to_real_indices(item, qubit_map)

        # 否则，直接将 item 替换为其对应的实数索引
        else:
            real_item = qubit_map[item]

        # 如果 item 是 Gate 对象，则用转换后的参数重新构造 Gate 对象
        if isinstance(item, Gate):
            real_item = item.__class__(*real_item)

        # 将转换后的元素添加到 real_seq 中
        real_seq = real_seq + (real_item,)

    # 返回转换后的实数索引序列
    return real_seq


def random_reduce(circuit, gate_ids, seed=None):
    """Shorten the length of a quantum circuit.

    Explanation
    ===========

    random_reduce looks for circuit identities in circuit, randomly chooses
    one to remove, and returns a shorter yet equivalent circuit.  If no
    identities are found, the same circuit is returned.

    Parameters
    ==========

    circuit : Gate tuple of Mul
        A tuple of Gates representing a quantum circuit
    gate_ids : list, GateIdentity
        List of gate identities to find in circuit
    seed : int or list
        seed used for _randrange; to override the random selection, provide a
        list of integers: the elements of gate_ids will be tested in the order
        given by the list

    """
    # 导入 _randrange 函数用于随机数生成
    from sympy.core.random import _randrange

    # 如果 gate_ids 为空列表，则直接返回原始电路
    if not gate_ids:
        return circuit

    # 如果 circuit 是 Mul 对象，则提取其 args
    if isinstance(circuit, Mul):
        circuit = circuit.args

    # 将 gate_ids 扁平化处理
    ids = flatten_ids(gate_ids)

    # 使用给定的种子创建随机整数生成器
    randrange = _randrange(seed)

    # 在电路中查找电路标识
    while ids:
        i = randrange(len(ids))
        id = ids.pop(i)
        # 如果在电路中找到了标识，就从中删除并返回新的电路
        if find_subcircuit(circuit, id) != -1:
            break
    else:
        # 如果没有找到任何标识，则返回原始电路
        return circuit

    # 返回删除了标识的新电路
    return replace_subcircuit(circuit, id)


def random_insert(circuit, choices, seed=None):
    """Insert a circuit into another quantum circuit.

    Explanation
    ===========

    random_insert selects a point in circuit and inserts choices, a circuit or
    Gate, at that point.  If choices is a Mul, its args are used in the
    insertion; if choices is a Gate, it is used as is.

    Parameters
    ==========

    circuit : Gate tuple of Mul
        A tuple of Gates representing a quantum circuit
    choices : Gate/tuple of Gate/Mul
        A Gate, a tuple of Gates, or a Mul to be inserted into circuit
    seed : int or list
        seed used for _randrange; to override the random selection, provide a
        list of integers: the elements of gate_ids will be tested in the order
        given by the list

    """
    random_insert randomly chooses a location in the circuit to insert
    a randomly selected circuit from amongst the given choices.

    Parameters
    ==========

    circuit : Gate tuple or Mul
        A tuple or Mul of Gates representing a quantum circuit
    choices : list
        Set of circuit choices
    seed : int or list
        seed used for _randrange; to override the random selections, give
        a list two integers, [i, j] where i is the circuit location where
        choice[j] will be inserted.

    Notes
    =====

    Indices for insertion should be [0, n] if n is the length of the
    circuit.
    """
    # 导入 _randrange 函数用于生成随机数
    from sympy.core.random import _randrange

    # 如果 choices 列表为空，则直接返回原始的 circuit
    if not choices:
        return circuit

    # 如果 circuit 是 Mul 类型，则将其转换为其元组表示
    if isinstance(circuit, Mul):
        circuit = circuit.args

    # 使用 seed 生成一个随机数生成器
    randrange = _randrange(seed)

    # 随机选择插入位置 loc，范围是 [0, n+1]，其中 n 是 circuit 的长度
    loc = randrange(len(circuit) + 1)

    # 从 choices 中随机选择一个元素作为要插入的电路部分
    choice = choices[randrange(len(choices))]

    # 将 circuit 转换为列表，将 choice 插入到 loc 位置
    circuit = list(circuit)
    circuit[loc: loc] = choice

    # 返回修改后的 circuit，类型为元组
    return tuple(circuit)
# 将 GateIdentity 对象（带有门规则）展平成一个单一列表

def flatten_ids(ids):
    # 定义一个 lambda 函数 collapse，用于将等效 ID 的列表合并到累加器 acc 中，并按默认排序键排序
    collapse = lambda acc, an_id: acc + sorted(an_id.equivalent_ids,
                                               key=default_sort_key)
    # 使用 reduce 函数将 ids 列表中的所有 GateIdentity 对象的 equivalent_ids 属性展开到一个单一的列表中
    ids = reduce(collapse, ids, [])
    # 对最终的 ids 列表按默认排序键进行排序
    ids.sort(key=default_sort_key)
    # 返回展平后的 ids 列表
    return ids
```