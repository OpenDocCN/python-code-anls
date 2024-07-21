# `.\pytorch\torch\_inductor\runtime\triton_helpers.py`

```py
# mypy: allow-untyped-defs
# 引入 triton 库及其语言模块
import triton
import triton.language as tl

# 尝试导入最新版本的 triton，其中数学函数被重组到不同模块中
try:
    from triton.language.extra import libdevice

    # 设置 libdevice 为 tl.extra.libdevice
    libdevice = tl.extra.libdevice  # noqa: F811
    # 设置 math 为 tl.math
    math = tl.math
except ImportError:
    # 处理导入错误，根据不同的硬件平台选择 libdevice 和 math 模块
    if hasattr(tl.extra, "cuda") and hasattr(tl.extra.cuda, "libdevice"):
        libdevice = tl.extra.cuda.libdevice
        math = tl.math
    elif hasattr(tl.extra, "intel") and hasattr(tl.extra.intel, "libdevice"):
        libdevice = tl.extra.intel.libdevice
        math = tl.math
    else:
        # 如果以上条件都不满足，使用 tl.math
        libdevice = tl.math
        math = tl.math

# 尝试导入 _log2 函数，若失败则定义为抛出 NotImplementedError 的函数
try:
    from triton.language.standard import _log2
except ImportError:

    def _log2(x):
        raise NotImplementedError

# Triton jit 装饰器，用于编译函数为 Triton 可执行的代码
@triton.jit
def promote_to_tensor(x):
    # 加法操作将 x 提升为张量（tensor）
    return x + tl.zeros((1,), tl.int1)

# Triton jit 装饰器，判断 x 是否为浮点数类型
@triton.jit
def is_floating(x):
    return promote_to_tensor(x).dtype.is_floating()

# Triton jit 装饰器，计算两个数的乘积
@triton.jit
def _prod_accumulate(a, b):
    return a * b

# Triton jit 装饰器，沿着指定轴计算输入数组的乘积
@triton.jit
def prod(input, axis):
    return tl.reduce(input, axis, _prod_accumulate)

# Triton jit 装饰器，返回两个数中的较小值
@triton.jit
def minimum(a, b):
    mask = a < b
    if is_floating(a):
        mask |= a != a
    return tl.where(mask, a, b)

# Triton jit 装饰器，返回两个数中的较大值
@triton.jit
def maximum(a, b):
    mask = a > b
    if is_floating(a):
        mask |= a != a
    return tl.where(mask, a, b)

# Triton jit 装饰器，沿着指定轴计算输入数组的最小值
@triton.jit
def min2(a, dim):
    return tl.reduce(a, dim, minimum)

# Triton jit 装饰器，沿着指定轴计算输入数组的最大值
@triton.jit
def max2(a, dim):
    return tl.reduce(a, dim, maximum)

# Triton jit 装饰器，返回两个值中的较小值及其索引
@triton.jit
def minimum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value < b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan and not b_isnan
        # 将 NaN 视为相等
        equal |= a_isnan and b_isnan

    # 若值相等，则选择较小的索引
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)

# Triton jit 装饰器，返回两个值中的较大值及其索引
@triton.jit
def maximum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value > b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan and not b_isnan
        # 将 NaN 视为相等
        equal |= a_isnan and b_isnan

    # 若值相等，则选择较小的索引
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)

# Triton jit 装饰器，沿着指定轴返回数组中的最小值及其索引
@triton.jit
def min_with_index(value, index, dim):
    return tl.reduce((value, index), dim, minimum_with_index)

# Triton jit 装饰器，沿着指定轴返回数组中的最大值及其索引
@triton.jit
def max_with_index(value, index, dim):
    return tl.reduce((value, index), dim, maximum_with_index)

# Triton jit 装饰器，实现 Welford 方法的归约操作
@triton.jit
def welford_reduce(value, mean, m2, weight, first_iteration):
    # 如果是第一次迭代
    if first_iteration:
        # 创建一个形状与weight相同，元素全部为1的张量，并使用weight的数据类型
        new_weight = tl.full(weight.shape, 1, weight.dtype)
        # 设置新的均值为当前值
        new_mean = value
        # 创建一个与m2形状相同，元素全部为0的张量
        new_m2 = tl.zeros_like(m2)
    else:
        # 计算当前值与均值的差值
        delta = value - mean
        # 更新权重：原始权重加1
        new_weight = weight + 1
        # 更新均值：根据加权平均公式更新均值
        new_mean = mean + delta / new_weight
        # 更新m2：累积平方差，用于计算方差
        new_m2 = m2 + delta * (value - new_mean)
    # 返回更新后的均值、m2和权重
    return new_mean, new_m2, new_weight
@triton.jit
# 使用 Triton 编译装饰器将函数编译为 Triton JIT 函数
def welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    # 计算两组统计量的差值
    delta = mean_2 - mean_1
    # 计算合并后的权重
    new_weight = weight_1 + weight_2
    # 计算权重比例
    w2_over_w = tl.where(new_weight == 0.0, 0.0, weight_2 / new_weight)
    # 返回合并后的均值、平方差、以及新的权重
    return (
        mean_1 + delta * w2_over_w,
        m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w,
        new_weight,
    )


@triton.jit
# 使用 Triton 编译装饰器将函数编译为 Triton JIT 函数
def welford(mean, m2, weight, dim):
    # 使用 reduce 函数对指定维度上的数据应用 welford_combine 函数
    return tl.reduce((mean, m2, weight), dim, welford_combine)


@triton.jit
# 使用 Triton 编译装饰器将函数编译为 Triton JIT 函数
def device_assert_then(cond, msg, r):
    # 在设备上进行条件断言，如果条件不满足，输出消息 msg
    tl.device_assert(cond, msg)
    # 返回结果 r
    return r


@triton.jit
# 使用 Triton 编译装饰器将函数编译为 Triton JIT 函数
def randint64(seed, offset, low, high):
    # 使用四个随机数生成 64 位整数
    r0, r1, r2, r3 = tl.randint4x(seed, offset)
    # 将 r0 和 r1 转换为 64 位无符号整数
    r0 = r0.to(tl.uint64)
    r1 = r1.to(tl.uint64)
    # 合并两个 64 位整数得到一个结果
    result = r0 | (r1 << 32)
    # 计算结果在指定范围内的余数
    size = high - low
    result = result % size.to(tl.uint64)
    # 将结果转换为 64 位有符号整数并加上低限
    result = result.to(tl.int64) + low
    # 返回结果
    return result


@triton.jit
# 使用 Triton 编译装饰器将函数编译为 Triton JIT 函数
def _any_combine(a, b):
    # 对两个值进行逻辑或操作
    return a | b


@triton.jit
# 使用 Triton 编译装饰器将函数编译为 Triton JIT 函数
def any(a, dim):
    # 在指定维度上应用 _any_combine 函数进行 reduce 操作
    return tl.reduce(a, dim, _any_combine)


@triton.jit
# 使用 Triton 编译装饰器将函数编译为 Triton JIT 函数
def bucketize_binary_search(
    values,  # 1D tensor，输入的数值
    offsets_ptr,  # 指向偏移值的指针
    indexing_dtype,  # 索引数据类型
    right,  # bool: 是否右侧闭区间
    OFFSETS_SIZE: int,  # 偏移值大小
    BLOCK_SHAPE,  # tuple/list of block shape，块形状
):
    """
    See [Note: Inductor bucketize op]
    """
    # 创建全零的张量作为下限
    low = tl.zeros(BLOCK_SHAPE, dtype=indexing_dtype)
    # 创建全局变量大小的张量作为上限
    high = tl.full(BLOCK_SHAPE, OFFSETS_SIZE, dtype=indexing_dtype)

    full_range = OFFSETS_SIZE + 1
    # 使用二分查找算法查找桶的位置
    while full_range > 1:
        mid = (high + low) // 2
        mask = mid < OFFSETS_SIZE
        # 加载偏移指针处的中间值作为桶的上界
        bucket_upper_bound = tl.load(offsets_ptr + mid, mask=mask, other=0.0)
        # 根据 right 参数决定比较方式
        if right:
            is_above = values >= bucket_upper_bound
        else:
            is_above = values > bucket_upper_bound

        # 根据比较结果更新 low 和 high
        low = tl.where(is_above & mask, mid + 1, low)
        high = tl.where(is_above, high, mid)

        full_range = (full_range + 1) // 2

    # 返回最终的桶的索引
    return low


@triton.jit
# 使用 Triton 编译装饰器将函数编译为 Triton JIT 函数
def pack_value_flag(
    value,
    flag,
    DTYPE_VALUE_AS_UINT: tl.constexpr,
    DTYPE_PACK: tl.constexpr,
):
    # 解决 Triton 中 tensor.to 函数不支持 constexpr 值的问题
    DTYPE_VALUE_AS_UINT = tl.core._constexpr_to_value(DTYPE_VALUE_AS_UINT)
    bitwidth = DTYPE_VALUE_AS_UINT.primitive_bitwidth
    # 将数值转换为指定的无符号整数类型并进行位操作
    uv = value.to(DTYPE_VALUE_AS_UINT, bitcast=True).to(DTYPE_PACK)
    # 将标志位转换为指定的打包类型并进行位操作
    return flag.to(DTYPE_PACK) | (uv << bitwidth)


@triton.jit
# 使用 Triton 编译装饰器将函数编译为 Triton JIT 函数
def unpack_value(
    pack,
    DTYPE_VALUE,
    DTYPE_VALUE_AS_UINT,
):
    # 解决 Triton 中 tensor.to 函数不支持 constexpr 值的问题
    DTYPE_VALUE = tl.core._constexpr_to_value(DTYPE_VALUE)
    DTYPE_VALUE_AS_UINT = tl.core._constexpr_to_value(DTYPE_VALUE_AS_UINT)
    bitwidth = DTYPE_VALUE_AS_UINT.primitive_bitwidth
    # 从打包数据中提取数值并转换为指定类型
    value_uint = (pack >> bitwidth).to(DTYPE_VALUE_AS_UINT)
    return value_uint.to(DTYPE_VALUE, bitcast=True)


@triton.jit
# 使用 Triton 编译装饰器将函数编译为 Triton JIT 函数
def unpack_flag(pack, DTYPE_FLAG):
    # 将打包数据转换为指定的标志位类型
    return pack.to(DTYPE_FLAG)


@triton.jit
# 使用 Triton 编译装饰器将函数编译为 Triton JIT 函数
def exclusive_scan_decoupled_lookback(
    # 定义一系列变量，这些变量可能在程序的其他部分有定义或者被引用
    scratch_base,
    block_value,
    index,
    combine_fn,
    # 使用 tl.constexpr 常量值作为 DTYPE_VALUE_AS_UINT 的值
    DTYPE_VALUE_AS_UINT: tl.constexpr,
    # 使用 tl.constexpr 常量值作为 DTYPE_PACK 的值
    DTYPE_PACK: tl.constexpr,
@triton.jit
def exclusive_scan_decoupled_lookback_64(scratch_base, block_value, index, combine_fn):
    """Compute exclusive scan of a scalar value between blocks

    Ref: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

    scratch_base: Pointer to scratch space in global memory
    block_value: Scalar value for this block, must be 64-bits wide
    index: Scalar index of this block relative to the current scan
    combine_fn: Function ``(value, value) -> value`` which is scanned over
    init: Scalar value equal to the identity of combine_fn
    """
    # Publish block sum so subsequent blocks don't get stuck waiting for us

    # Determine the data type of block_value
    DTYPE_VALUE = block_value.dtype

    # Pack the block_value and a flag using the pack_value_flag function
    pack = pack_value_flag(
        block_value,
        tl.full(block_value.shape, 1, DTYPE_VALUE_AS_UINT),
        DTYPE_VALUE_AS_UINT,
        DTYPE_PACK,
    )

    # If the index is greater than 0, atomically exchange the pack value into scratch_base + index
    if index > 0:
        tl.atomic_xchg(scratch_base + index, pack, sem="relaxed")

    # Initialize exclusive_prefix for the exclusive prefix scan
    exclusive_prefix = tl.zeros([], DTYPE_VALUE)
    prefix_valid = False
    test_target = index - 1

    # Perform a loop to calculate the exclusive prefix scan
    while test_target >= 0:
        # Load the pack value atomically from scratch_base + test_target
        flag = tl.full([], 0, DTYPE_VALUE_AS_UINT)
        while flag == 0:
            pack = tl.atomic_add(scratch_base + test_target, 0, sem="relaxed")
            flag = unpack_flag(pack, DTYPE_VALUE_AS_UINT)

        # Unpack the value from the pack
        value = unpack_value(pack, DTYPE_VALUE, DTYPE_VALUE_AS_UINT)

        # Update exclusive_prefix according to combine_fn
        if prefix_valid:
            exclusive_prefix = combine_fn(value, exclusive_prefix)
        else:
            exclusive_prefix = value
            prefix_valid = True

        # Check if the flag indicates the end of the scan
        if flag == 2:
            test_target = -1
        else:
            test_target = test_target - 1

    # Calculate inclusive_prefix for the inclusive block sum
    if prefix_valid:
        inclusive_prefix = combine_fn(exclusive_prefix, block_value)
    else:
        inclusive_prefix = block_value

    # Pack the inclusive_prefix and a flag using the pack_value_flag function
    pack = pack_value_flag(
        inclusive_prefix,
        tl.full([], 2, DTYPE_VALUE_AS_UINT),
        DTYPE_VALUE_AS_UINT,
        DTYPE_PACK,
    )

    # Atomically exchange the pack value into scratch_base + index
    tl.atomic_xchg(scratch_base + index, pack, sem="relaxed")

    # Return the exclusive_prefix which represents the result of the exclusive scan
    return exclusive_prefix
    # 如果索引大于0，则执行以下操作
    if index > 0:
        # 将 block_value 转换为 uint64 类型，并进行位转换
        block_value_u64 = block_value.to(tl.uint64, bitcast=True)
        # 将 block_value_u64 存储到指定位置（scratch_base + 3 * index + 1）
        tl.store(scratch_base + 3 * index + 1, block_value_u64)
        # 执行调试屏障操作
        tl.debug_barrier()
        # 创建一个值为1的 uint64 类型的 full tensor
        flag_one = tl.full([], 1, tl.uint64)
        # 使用原子交换操作将 flag_one 存储到指定位置（scratch_base + 3 * index + 0），并释放信号量
        tl.atomic_xchg(scratch_base + 3 * index + 0, flag_one, sem="release")

    # 计算独占前缀扫描
    exclusive_prefix = tl.zeros([], block_value.dtype)
    # 前缀有效性标志
    prefix_valid = False
    # 测试目标为当前索引的前一个索引
    test_target = index - 1
    # 当 test_target 大于等于0时执行循环
    while test_target >= 0:
        # 创建一个值为0的 uint64 类型的 full tensor，并进行原子加操作
        flag = tl.full([], 0, tl.uint64)
        while flag == 0:
            # 使用原子加操作获取标志（scratch_base + 3 * test_target + 0），并获取信号量
            flag = tl.atomic_add(scratch_base + 3 * test_target + 0, 0, sem="acquire")

        # 从指定位置（scratch_base + 3 * test_target + flag.to(tl.int32)）加载数据，并转换为指定类型
        value_u64 = tl.load(scratch_base + 3 * test_target + flag.to(tl.int32))
        value = value_u64.to(block_value.dtype, bitcast=True)
        
        # 如果前缀有效，则将当前值与 exclusive_prefix 结合
        if prefix_valid:
            exclusive_prefix = combine_fn(value, exclusive_prefix)
        else:
            exclusive_prefix = value
            prefix_valid = True

        # 如果 flag 等于2，则设置 test_target 为-1，结束循环
        if flag == 2:
            test_target = -1
        else:
            test_target = test_target - 1

    # 将 inclusive block sum 对其他块可见
    if prefix_valid:
        # 将 exclusive_prefix 与 block_value 结合，得到 inclusive_prefix
        inclusive_prefix = combine_fn(exclusive_prefix, block_value)
    else:
        inclusive_prefix = block_value
    # 将 inclusive_prefix 转换为 uint64 类型，并进行位转换
    inclusive_prefix_u64 = inclusive_prefix.to(tl.uint64, bitcast=True)
    # 将 inclusive_prefix_u64 存储到指定位置（scratch_base + 3 * index + 2）
    tl.store(scratch_base + 3 * index + 2, inclusive_prefix_u64)
    # 执行调试屏障操作
    tl.debug_barrier()
    # 创建一个值为2的 uint64 类型的 full tensor
    flag_two = tl.full([], 2, tl.uint64)
    # 使用原子交换操作将 flag_two 存储到指定位置（scratch_base + 3 * index + 0），并释放信号量
    tl.atomic_xchg(scratch_base + 3 * index + 0, flag_two, sem="release")

    # 返回计算得到的 exclusive_prefix
    return exclusive_prefix
@triton.jit
def frexp(x):
    # TODO(isuruf): use inline_asm_elementwise here
    # 计算浮点数 x 的指数和尾数
    y = libdevice.ilogb(x) + 1  # 计算 x 的对数绝对值并加 1
    exponent = tl.where(x == 0, 0, y)  # 如果 x 为零则指数为零，否则为 y
    mantissa = tl.where(x == 0, 0, libdevice.ldexp(x, -y))  # 如果 x 为零则尾数为零，否则计算 x 乘以 2 的 -y 次幂
    return mantissa, exponent


@triton.jit
def _compare_and_swap_with_index(
    x,
    idxs,
    valid_mask,
    flip,
    i: tl.constexpr,
    n_dims: tl.constexpr,
    stable: tl.constexpr,
    descending: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims  # 计算 x 的元素数量右移 n_dims 位
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]  # 计算形状参数

    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)  # 获取整数数据类型

    y = tl.reshape(x, shape)  # 将 x 重新形状为 shape
    iy = y.to(idtype, bitcast=True)  # 将 y 转换为 idtype 类型并进行位转换
    # 使用左右 'stride' 2**(n_dims - i - 1) 切片
    right_mask = tl.arange(0, 2)[None, :, None].to(idtype)  # 创建右侧掩码
    left_mask = (1 - right_mask).to(idtype)  # 创建左侧掩码
    ileft = tl.broadcast_to(tl.sum(iy * left_mask, 1)[:, None, :], shape)  # 计算左侧和
    iright = tl.broadcast_to(tl.sum(iy * right_mask, 1)[:, None, :], shape)  # 计算右侧和
    ileft = tl.reshape(ileft, x.shape)  # 将 ileft 重新形状为 x 的形状
    iright = tl.reshape(iright, x.shape)  # 将 iright 重新形状为 x 的形状
    left = ileft.to(x.dtype, bitcast=True)  # 将 ileft 转换为 x 的数据类型，并进行位转换
    right = iright.to(x.dtype, bitcast=True)  # 将 iright 转换为 x 的数据类型，并进行位转换

    # idx
    y_idx = tl.reshape(idxs, shape)  # 将 idxs 重新形状为 shape
    left_idx = tl.broadcast_to(
        tl.sum(y_idx * left_mask.to(y_idx.dtype), 1)[:, None, :], shape
    )  # 计算左侧索引和
    right_idx = tl.broadcast_to(
        tl.sum(y_idx * right_mask.to(y_idx.dtype), 1)[:, None, :], shape
    )  # 计算右侧索引和
    left_idx = tl.reshape(left_idx, x.shape)  # 将 left_idx 重新形状为 x 的形状
    right_idx = tl.reshape(right_idx, x.shape)  # 将 right_idx 重新形状为 x 的形状

    # valid
    if valid_mask is None:
        left_valid_mask = tl.full(x.shape, True, tl.int1)  # 创建全为 True 的左侧有效掩码
        right_valid_mask = tl.full(x.shape, True, tl.int1)  # 创建全为 True 的右侧有效掩码
    else:
        y_valid_mask = tl.reshape(valid_mask, shape)  # 将 valid_mask 重新形状为 shape
        left_valid_mask = tl.broadcast_to(
            tl.sum(y_valid_mask * left_mask.to(tl.int8), 1)[:, None, :], shape
        ).to(tl.int1)  # 计算左侧有效掩码
        right_valid_mask = tl.broadcast_to(
            tl.sum(y_valid_mask * right_mask.to(tl.int8), 1)[:, None, :], shape
        ).to(tl.int1)  # 计算右侧有效掩码
        left_valid_mask = tl.reshape(left_valid_mask, x.shape)  # 将 left_valid_mask 重新形状为 x 的形状
        right_valid_mask = tl.reshape(right_valid_mask, x.shape)  # 将 right_valid_mask 重新形状为 x 的形状

    # actual compare-and-swap
    ix = x.to(idtype, bitcast=True)  # 将 x 转换为 idtype 类型，并进行位转换

    if descending:
        cond = left < right  # 如果降序，比较左侧和右侧
    else:
        cond = left > right  # 如果升序，比较左侧和右侧

    if stable:
        # 当稳定排序时，通过索引进行打破平局
        cond = cond | ((left == right) & (left_idx > right_idx))

    cond = (right_valid_mask > left_valid_mask) | (
        (right_valid_mask == left_valid_mask) & cond
    )  # 比较有效掩码和条件
    cond = cond ^ flip  # 条件进行异或操作
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))  # 计算返回值
    new_idxs = idxs ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(idxs))  # 计算新索引
    if valid_mask is None:
        new_valid_mask = tl.full(x.shape, True, tl.int1)  # 创建全为 True 的新有效掩码
    else:
        new_valid_mask = valid_mask ^ tl.where(
            cond, left_valid_mask ^ right_valid_mask, tl.zeros_like(valid_mask)
        )  # 计算新的有效掩码
    # 将 ret 张量转换为指定的数据类型 x.dtype，并通过位转换（bitcast）设置为 True
    return ret.to(x.dtype, bitcast=True), new_idxs, new_valid_mask
@triton.jit
# 使用 Triton 编译器进行即时编译的装饰器
def _bitonic_merge_with_index(
    x,
    idxs,
    mask,
    stage: tl.constexpr,
    alternating: tl.constexpr,
    n_dims: tl.constexpr,
    stable: tl.constexpr,
    descending: tl.constexpr,
):
    # 计算每个外部循环的元素数量
    n_outer: tl.constexpr = x.numel >> n_dims
    # 静态断言：确保当前阶段不超过维度数量
    tl.static_assert(stage <= n_dims)
    
    # flip 表示是否按升序或降序重新排列子序列的元素
    # 如果 flip = 00000000...，则所有元素将在此阶段按升序重新排列
    # 如果 flip = 00110011...，则所有元素将在此阶段以交替方式重新排列（步长为 2）
    if alternating:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = False
    
    # 执行 `stage` 轮 `compare-and-swap` 操作
    next_mask = mask
    for i in tl.static_range(stage):
        # 调用 `_compare_and_swap_with_index` 函数执行比较和交换操作
        x, idxs, next_mask = _compare_and_swap_with_index(
            x, idxs, mask, flip, i + (n_dims - stage), n_dims, stable, descending
        )
        if mask is not None:
            mask = next_mask
    
    return x, idxs, next_mask


@triton.jit
# 使用 Triton 编译器进行即时编译的装饰器
def sort_with_index(
    x,  # 值
    idxs,  # 索引
    mask,  # 如果当前值有效，则使用的掩码（无效值排序到末尾）
    dim: tl.constexpr = None,
    stable: tl.constexpr = tl.constexpr(False),
    descending: tl.constexpr = tl.constexpr(False),
):
    # 广播值 `x` 和索引 `idxs`，使它们具有相同的形状
    x, idxs = tl.broadcast(x, idxs)
    
    # 如果存在掩码，则广播掩码 `mask` 和值 `x`，使它们具有相同的形状
    if mask is not None:
        x, mask = tl.broadcast(x, mask)
    
    # 处理默认维度或检查其是否是最小的维度
    _dim: tl.constexpr = len(x.shape) - 1 if dim is None else dim
    tl.static_assert(
        _dim == len(x.shape) - 1, "only minor dimension is currently supported"
    )
    
    # 迭代执行比特洛尼克合并排序步骤
    n_dims: tl.constexpr = _log2(x.shape[_dim])

    for i in tl.static_range(1, n_dims + 1):
        # 调用 `_bitonic_merge_with_index` 函数执行比特洛尼克合并排序
        x, idxs, next_mask = _bitonic_merge_with_index(
            x,
            idxs,
            mask,
            i,
            alternating=i < n_dims,
            n_dims=n_dims,
            stable=stable,
            descending=descending,
        )
        if mask is not None:
            mask = next_mask
    
    return x, idxs
```