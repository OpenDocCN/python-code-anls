# `.\numpy\numpy\_core\tests\test_mem_overlap.py`

```py
import itertools  # 导入 itertools 模块，用于生成迭代器的工具函数
import pytest  # 导入 pytest 模块，用于编写和运行测试用例

import numpy as np  # 导入 NumPy 库，并将其命名为 np
from numpy._core._multiarray_tests import solve_diophantine, internal_overlap  # 导入 NumPy 内部的函数和类
from numpy._core import _umath_tests  # 导入 NumPy 内部的数学测试函数
from numpy.lib.stride_tricks import as_strided  # 导入 NumPy 的 as_strided 函数
from numpy.testing import (  # 导入 NumPy 测试模块中的一些断言函数
    assert_, assert_raises, assert_equal, assert_array_equal
    )

ndims = 2  # 定义维度数量为 2
size = 10  # 定义数组的大小为 10
shape = tuple([size] * ndims)  # 创建一个形状为 (10, 10) 的元组

MAY_SHARE_BOUNDS = 0  # 定义常量 MAY_SHARE_BOUNDS 为 0
MAY_SHARE_EXACT = -1  # 定义常量 MAY_SHARE_EXACT 为 -1


def _indices_for_nelems(nelems):
    """Returns slices of length nelems, from start onwards, in direction sign."""
    # 如果 nelems 为 0，则返回一个包含 size // 2 的整数索引的列表
    if nelems == 0:
        return [size // 2]  # int index

    res = []
    # 遍历步长为 1 和 2，符号为 -1 和 1 的组合
    for step in (1, 2):
        for sign in (-1, 1):
            start = size // 2 - nelems * step * sign // 2
            stop = start + nelems * step * sign
            res.append(slice(start, stop, step * sign))

    return res  # 返回生成的切片列表


def _indices_for_axis():
    """Returns (src, dst) pairs of indices."""
    res = []
    # 遍历不同的 nelems 值，生成对应的索引切片组合
    for nelems in (0, 2, 3):
        ind = _indices_for_nelems(nelems)
        res.extend(itertools.product(ind, ind))  # 使用 itertools.product 生成所有 nelems 大小的切片组合

    return res  # 返回生成的 (src, dst) 索引对列表


def _indices(ndims):
    """Returns ((axis0_src, axis0_dst), (axis1_src, axis1_dst), ... ) index pairs."""
    ind = _indices_for_axis()
    return itertools.product(ind, repeat=ndims)  # 使用 itertools.product 生成指定维度的索引对组合


def _check_assignment(srcidx, dstidx):
    """Check assignment arr[dstidx] = arr[srcidx] works."""
    arr = np.arange(np.prod(shape)).reshape(shape)  # 创建一个形状为 shape 的 NumPy 数组，填充为按序排列的元素

    cpy = arr.copy()  # 复制原始数组 arr

    cpy[dstidx] = arr[srcidx]  # 将 arr[srcidx] 的值复制到 cpy[dstidx]
    arr[dstidx] = arr[srcidx]  # 将 arr[srcidx] 的值赋给 arr[dstidx]

    assert_(np.all(arr == cpy),  # 使用 NumPy 的 assert_ 函数检查 arr 是否等于 cpy
            'assigning arr[%s] = arr[%s]' % (dstidx, srcidx))  # 如果不等则输出错误信息


def test_overlapping_assignments():
    # Test automatically generated assignments which overlap in memory.
    inds = _indices(ndims)  # 生成 ndims 维度的所有索引对组合

    for ind in inds:
        srcidx = tuple([a[0] for a in ind])  # 获取源索引 srcidx
        dstidx = tuple([a[1] for a in ind])  # 获取目标索引 dstidx

        _check_assignment(srcidx, dstidx)  # 调用 _check_assignment 函数进行赋值检查


@pytest.mark.slow  # 用 pytest.mark.slow 标记这是一个较慢的测试用例
def test_diophantine_fuzz():
    # Fuzz test the diophantine solver
    rng = np.random.RandomState(1234)  # 使用种子为 1234 的随机状态生成器

    max_int = np.iinfo(np.intp).max  # 获取 NumPy 中 np.intp 类型的最大整数值
    # 对于每个维度从0到9循环
    for ndim in range(10):
        # 可行解计数器和不可行解计数器初始化为0
        feasible_count = 0
        infeasible_count = 0

        # 计算最小计数，确保在小整数问题和大整数问题之间平衡
        min_count = 500 // (ndim + 1)

        # 当可行解计数器和不可行解计数器中的较小值小于最小计数时继续循环
        while min(feasible_count, infeasible_count) < min_count:
            # 确保大整数和小整数问题
            A_max = 1 + rng.randint(0, 11, dtype=np.intp)**6
            U_max = rng.randint(0, 11, dtype=np.intp)**6

            # 将A_max和U_max限制在max_int的范围内
            A_max = min(max_int, A_max)
            U_max = min(max_int - 1, U_max)

            # 生成长度为ndim的A和U元组，元素为随机整数
            A = tuple(int(rng.randint(1, A_max + 1, dtype=np.intp))
                      for j in range(ndim))
            U = tuple(int(rng.randint(0, U_max + 2, dtype=np.intp))
                      for j in range(ndim))

            # 计算b_ub作为A和U的线性组合的和，限制在max_int-2内
            b_ub = min(max_int - 2, sum(a * ub for a, ub in zip(A, U)))
            # 生成b作为介于-1和b_ub+2之间的随机整数
            b = int(rng.randint(-1, b_ub + 2, dtype=np.intp))

            # 如果ndim为0且可行解计数器小于最小计数，则强制b为0
            if ndim == 0 and feasible_count < min_count:
                b = 0

            # 解决给定A、U、b的丢番图方程，返回解X
            X = solve_diophantine(A, U, b)

            # 如果无解X
            if X is None:
                # 检查简化的决策问题是否一致
                X_simplified = solve_diophantine(A, U, b, simplify=1)
                assert_(X_simplified is None, (A, U, b, X_simplified))

                # 检查不存在解（前提是问题足够小，暴力检查不会花费太长时间）
                ranges = tuple(range(0, a * ub + 1, a) for a, ub in zip(A, U))

                size = 1
                for r in ranges:
                    size *= len(r)
                # 如果问题规模小于100000，检查是否存在任何解满足和为b
                if size < 100000:
                    assert_(not any(sum(w) == b for w in itertools.product(*ranges)))
                    infeasible_count += 1
            else:
                # 检查简化的决策问题是否一致
                X_simplified = solve_diophantine(A, U, b, simplify=1)
                assert_(X_simplified is not None, (A, U, b, X_simplified))

                # 检查解的有效性，确保线性组合的结果等于b
                assert_(sum(a * x for a, x in zip(A, X)) == b)
                # 确保解的每个元素在其对应的上界内
                assert_(all(0 <= x <= ub for x, ub in zip(X, U)))
                feasible_count += 1
# 定义一个测试函数，用于检验整数溢出检测功能
def test_diophantine_overflow():
    # 获取平台下整数指针的最大值
    max_intp = np.iinfo(np.intp).max
    # 获取平台下 int64 类型整数的最大值
    max_int64 = np.iinfo(np.int64).max

    # 检查是否 int64 的最大值小于等于整数指针的最大值
    if max_int64 <= max_intp:
        # 确保算法在内部使用 128 位解决问题；
        # 解决这个问题需要处理大量的中间数值
        A = (max_int64//2, max_int64//2 - 10)
        U = (max_int64//2, max_int64//2 - 10)
        b = 2*(max_int64//2) - 10

        # 断言求解二次丢番图方程的结果为 (1, 1)
        assert_equal(solve_diophantine(A, U, b), (1, 1))


# 检查两个数组是否可能共享内存的确切情况
def check_may_share_memory_exact(a, b):
    # 使用 MAY_SHARE_EXACT 参数检查 a 和 b 是否可能共享内存
    got = np.may_share_memory(a, b, max_work=MAY_SHARE_EXACT)

    # 断言默认参数下的共享内存结果与带界限参数的结果相同
    assert_equal(np.may_share_memory(a, b),
                 np.may_share_memory(a, b, max_work=MAY_SHARE_BOUNDS))

    # 修改数组 a 和 b 的内容
    a.fill(0)
    b.fill(0)
    a.fill(1)
    # 确定数组 b 是否有任何非零元素
    exact = b.any()

    # 如果共享内存的结果与实际情况不符，则记录错误消息
    err_msg = ""
    if got != exact:
        err_msg = "    " + "\n    ".join([
            "base_a - base_b = %r" % (a.__array_interface__['data'][0] - b.__array_interface__['data'][0],),
            "shape_a = %r" % (a.shape,),
            "shape_b = %r" % (b.shape,),
            "strides_a = %r" % (a.strides,),
            "strides_b = %r" % (b.strides,),
            "size_a = %r" % (a.size,),
            "size_b = %r" % (b.size,)
        ])

    # 断言得到的共享内存结果与预期结果一致
    assert_equal(got, exact, err_msg=err_msg)


# 手动测试 may_share_memory 函数的测试用例
def test_may_share_memory_manual():
    # 基础数组
    xs0 = [
        np.zeros([13, 21, 23, 22], dtype=np.int8),
        np.zeros([13, 21, 23*2, 22], dtype=np.int8)[:,:,::2,:]
    ]

    # 生成所有负步幅组合
    xs = []
    for x in xs0:
        for ss in itertools.product(*(([slice(None), slice(None, None, -1)],)*4)):
            xp = x[ss]
            xs.append(xp)

    for x in xs:
        # 默认情况下简单检查范围是否重叠
        assert_(np.may_share_memory(x[:,0,:], x[:,1,:]))
        assert_(np.may_share_memory(x[:,0,:], x[:,1,:], max_work=None))

        # 确切检查
        check_may_share_memory_exact(x[:,0,:], x[:,1,:])
        check_may_share_memory_exact(x[:,::7], x[:,3::3])

        try:
            xp = x.ravel()
            if xp.flags.owndata:
                continue
            xp = xp.view(np.int16)
        except ValueError:
            continue

        # 0 大小的数组不可能重叠
        check_may_share_memory_exact(x.ravel()[6:6],
                                     xp.reshape(13, 21, 23, 11)[:,::7])

        # 测试项目大小处理情况
        check_may_share_memory_exact(x[:,::7],
                                     xp.reshape(13, 21, 23, 11))
        check_may_share_memory_exact(x[:,::7],
                                     xp.reshape(13, 21, 23, 11)[:,3::3])
        check_may_share_memory_exact(x.ravel()[6:7],
                                     xp.reshape(13, 21, 23, 11)[:,::7])

    # 检查单位大小
    x = np.zeros([1], dtype=np.int8)
    check_may_share_memory_exact(x, x)
    check_may_share_memory_exact(x, x.copy())
# 定义函数 iter_random_view_pairs，用于生成随机视图对
def iter_random_view_pairs(x, same_steps=True, equal_size=False):
    # 使用种子 1234 初始化随机数生成器
    rng = np.random.RandomState(1234)

    # 如果 equal_size 和 same_steps 同时为 True，则抛出 ValueError 异常
    if equal_size and same_steps:
        raise ValueError()

    # 定义生成随机切片的函数 random_slice
    def random_slice(n, step):
        # 随机生成起始位置 start 和终止位置 stop
        start = rng.randint(0, n+1, dtype=np.intp)
        stop = rng.randint(start, n+1, dtype=np.intp)
        # 随机决定是否反向切片
        if rng.randint(0, 2, dtype=np.intp) == 0:
            stop, start = start, stop
            step *= -1
        return slice(start, stop, step)

    # 定义生成固定大小随机切片的函数 random_slice_fixed_size
    def random_slice_fixed_size(n, step, size):
        # 随机生成起始位置 start 和终止位置 stop
        start = rng.randint(0, n+1 - size*step)
        stop = start + (size-1)*step + 1
        # 随机决定是否反向切片
        if rng.randint(0, 2) == 0:
            stop, start = start-1, stop-1
            if stop < 0:
                stop = None
            step *= -1
        return slice(start, stop, step)

    # 首先生成一些常规视图对
    # 返回原始数组 x 与自身的视图
    yield x, x
    # 循环生成一些特定偏移量 j 的视图对
    for j in range(1, 7, 3):
        yield x[j:], x[:-j]
        yield x[...,j:], x[...,:-j]

    # 生成一个具有零步长内部重叠的数组视图对
    strides = list(x.strides)
    strides[0] = 0
    xp = as_strided(x, shape=x.shape, strides=strides)
    yield x, xp
    yield xp, xp

    # 生成一个具有非零步长内部重叠的数组视图对
    strides = list(x.strides)
    if strides[0] > 1:
        strides[0] = 1
    xp = as_strided(x, shape=x.shape, strides=strides)
    yield x, xp
    yield xp, xp

    # 然后生成不连续的视图对
    while True:
        # 随机生成各维度的步长 steps
        steps = tuple(rng.randint(1, 11, dtype=np.intp)
                      if rng.randint(0, 5, dtype=np.intp) == 0 else 1
                      for j in range(x.ndim))
        # 随机生成一个切片 s1
        s1 = tuple(random_slice(p, s) for p, s in zip(x.shape, steps))

        # 随机打乱维度顺序 t1
        t1 = np.arange(x.ndim)
        rng.shuffle(t1)

        # 如果 equal_size 为 True，则 t2 与 t1 相同；否则重新随机排序维度
        if equal_size:
            t2 = t1
        else:
            t2 = np.arange(x.ndim)
            rng.shuffle(t2)

        # 取出切片 s1 对应的子数组 a
        a = x[s1]

        # 根据条件生成切片 s2
        if equal_size:
            if a.size == 0:
                continue
            steps2 = tuple(rng.randint(1, max(2, p//(1+pa)))
                           if rng.randint(0, 5) == 0 else 1
                           for p, s, pa in zip(x.shape, s1, a.shape))
            s2 = tuple(random_slice_fixed_size(p, s, pa)
                       for p, s, pa in zip(x.shape, steps2, a.shape))
        elif same_steps:
            steps2 = steps
        else:
            steps2 = tuple(rng.randint(1, 11, dtype=np.intp)
                           if rng.randint(0, 5, dtype=np.intp) == 0 else 1
                           for j in range(x.ndim))

        # 如果 equal_size 为 False，则重新生成切片 s2
        if not equal_size:
            s2 = tuple(random_slice(p, s) for p, s in zip(x.shape, steps2))

        # 对子数组 a 和 b 进行维度重排
        a = a.transpose(t1)
        b = x[s2].transpose(t2)

        # 生成视图对 (a, b)
        yield a, b


# 定义函数 check_may_share_memory_easy_fuzz，用于检查具有通用步长的重叠问题
def check_may_share_memory_easy_fuzz(get_max_work, same_steps, min_count):
    # 创建一个形状为 [17, 34, 71, 97] 的全零数组 x
    x = np.zeros([17,34,71,97], dtype=np.int16)

    # 初始化可行和不可行的计数器
    feasible = 0
    infeasible = 0

    # 生成 x 的随机视图对的迭代器 pair_iter
    pair_iter = iter_random_view_pairs(x, same_steps)
    # 当 feasible 和 infeasible 中的最小值小于 min_count 时循环执行以下操作
    while min(feasible, infeasible) < min_count:
        # 从 pair_iter 中获取下一对数据 a 和 b
        a, b = next(pair_iter)

        # 检查 a 和 b 是否共享内存，返回布尔值
        bounds_overlap = np.may_share_memory(a, b)
        
        # 检查 a 和 b 是否共享内存，返回布尔值
        may_share_answer = np.may_share_memory(a, b)
        
        # 检查 a 和 b 是否共享内存，返回布尔值，考虑到最大工作量
        easy_answer = np.may_share_memory(a, b, max_work=get_max_work(a, b))
        
        # 检查 a 和 b 是否共享内存，返回布尔值，使用精确的共享内存策略
        exact_answer = np.may_share_memory(a, b, max_work=MAY_SHARE_EXACT)

        # 如果简易答案与精确答案不同，触发断言错误
        if easy_answer != exact_answer:
            # assert_equal 操作较慢，用于验证 easy_answer 和 exact_answer 相等
            assert_equal(easy_answer, exact_answer)

        # 如果 may_share_answer 与 bounds_overlap 不同，触发断言错误
        if may_share_answer != bounds_overlap:
            assert_equal(may_share_answer, bounds_overlap)

        # 如果 bounds_overlap 为 True
        if bounds_overlap:
            # 如果 exact_answer 为 True，增加 feasible 计数
            if exact_answer:
                feasible += 1
            # 否则，增加 infeasible 计数
            else:
                infeasible += 1
@pytest.mark.slow
def test_may_share_memory_easy_fuzz():
    # 检查使用常见步长时，重叠问题是否总是可以用较少的工作解决

    # 调用检查函数，设定最大工作量为1，同步步长，最小计数为2000
    check_may_share_memory_easy_fuzz(get_max_work=lambda a, b: 1,
                                     same_steps=True,
                                     min_count=2000)


@pytest.mark.slow
def test_may_share_memory_harder_fuzz():
    # 对于不一定有共同步长的重叠问题，需要更多的工作量
    #
    # 下面的工作上限无法减少太多。更难的问题也可能存在，但不会在这里检测到，因为问题集来自RNG（随机数生成器）。

    # 调用检查函数，设定最大工作量为数组大小的一半，非同步步长，最小计数为2000
    check_may_share_memory_easy_fuzz(get_max_work=lambda a, b: max(a.size, b.size)//2,
                                     same_steps=False,
                                     min_count=2000)


def test_shares_memory_api():
    x = np.zeros([4, 5, 6], dtype=np.int8)

    # 检查数组是否共享内存
    assert_equal(np.shares_memory(x, x), True)
    assert_equal(np.shares_memory(x, x.copy()), False)

    # 创建切片a和b，检查它们是否共享内存
    a = x[:,::2,::3]
    b = x[:,::3,::2]
    assert_equal(np.shares_memory(a, b), True)
    assert_equal(np.shares_memory(a, b, max_work=None), True)
    
    # 断言引发异常，检查最大工作量设置是否正常
    assert_raises(
        np.exceptions.TooHardError, np.shares_memory, a, b, max_work=1
    )


def test_may_share_memory_bad_max_work():
    x = np.zeros([1])
    
    # 断言引发溢出错误，检查最大工作量设置是否正常
    assert_raises(OverflowError, np.may_share_memory, x, x, max_work=10**100)
    assert_raises(OverflowError, np.shares_memory, x, x, max_work=10**100)


def test_internal_overlap_diophantine():
    def check(A, U, exists=None):
        X = solve_diophantine(A, U, 0, require_ub_nontrivial=1)

        if exists is None:
            exists = (X is not None)

        if X is not None:
            # 断言检查解是否符合特定条件
            assert_(sum(a*x for a, x in zip(A, X)) == sum(a*u//2 for a, u in zip(A, U)))
            assert_(all(0 <= x <= u for x, u in zip(X, U)))
            assert_(any(x != u//2 for x, u in zip(X, U)))

        if exists:
            assert_(X is not None, repr(X))
        else:
            assert_(X is None, repr(X))

    # Smoke tests
    # 调用check函数进行烟雾测试，确保特定输入条件下函数正常工作
    check((3, 2), (2*2, 3*2), exists=True)
    check((3*2, 2), (15*2, (3-1)*2), exists=False)


def test_internal_overlap_slices():
    # 切片数组永远不会生成内部重叠

    x = np.zeros([17,34,71,97], dtype=np.int16)

    rng = np.random.RandomState(1234)

    def random_slice(n, step):
        start = rng.randint(0, n+1, dtype=np.intp)
        stop = rng.randint(start, n+1, dtype=np.intp)
        if rng.randint(0, 2, dtype=np.intp) == 0:
            stop, start = start, stop
            step *= -1
        return slice(start, stop, step)

    cases = 0
    min_count = 5000
    # 在达到最小计数之前循环执行以下操作
    while cases < min_count:
        # 生成一个元组 `steps`，其中每个元素是在指定范围内随机选择的步长值或者固定值 1
        steps = tuple(rng.randint(1, 11, dtype=np.intp)
                      if rng.randint(0, 5, dtype=np.intp) == 0 else 1
                      for j in range(x.ndim))
        
        # 创建一个长度等于 `x` 的维度数的数组 `t1`，其中包含从 0 到 `x.ndim-1` 的整数
        t1 = np.arange(x.ndim)
        
        # 对数组 `t1` 进行随机重排列
        rng.shuffle(t1)
        
        # 根据 `x` 的形状和步长 `steps` 生成一个元组 `s1`，其中每个元素是随机切片对象
        s1 = tuple(random_slice(p, s) for p, s in zip(x.shape, steps))
        
        # 根据切片 `s1` 从数组 `x` 中获取数据，并根据 `t1` 的顺序进行转置，得到数组 `a`
        a = x[s1].transpose(t1)
        
        # 断言确保数组 `a` 中不存在内部重叠
        assert_(not internal_overlap(a))
        
        # 增加计数器 `cases` 的值
        cases += 1
# 检查输入数组是否存在内部重叠
def check_internal_overlap(a, manual_expected=None):
    # 调用内部函数计算得到数组的内部重叠情况
    got = internal_overlap(a)

    # 使用暴力方法进行检查
    m = set()
    # 生成各维度的范围
    ranges = tuple(range(n) for n in a.shape)
    # 遍历数组所有可能的索引组合
    for v in itertools.product(*ranges):
        # 计算偏移量
        offset = sum(s*w for s, w in zip(a.strides, v))
        # 如果偏移量已经存在于集合中，则表示存在重叠
        if offset in m:
            expected = True
            break
        else:
            m.add(offset)
    else:
        expected = False

    # 比较实际结果和预期结果
    if got != expected:
        # 使用断言检查结果是否相等，如果不相等则会抛出异常
        assert_equal(got, expected, err_msg=repr((a.strides, a.shape)))
    # 如果手动预期结果不为None且与计算得到的结果不一致，则使用断言检查
    if manual_expected is not None and expected != manual_expected:
        assert_equal(expected, manual_expected)
    # 返回计算得到的内部重叠结果
    return got


def test_internal_overlap_manual():
    # Stride tricks可以构建具有内部重叠的数组

    # 我们不关心内存边界，数组不是读/写访问
    x = np.arange(1).astype(np.int8)

    # 检查低维特殊情况

    # 检查一维情况
    check_internal_overlap(x, False)
    # 检查零维情况
    check_internal_overlap(x.reshape([]), False)

    # 使用指定步长和形状创建数组
    a = as_strided(x, strides=(3, 4), shape=(4, 4))
    check_internal_overlap(a, False)

    # 使用指定步长和形状创建数组
    a = as_strided(x, strides=(3, 4), shape=(5, 4))
    check_internal_overlap(a, True)

    # 使用指定步长和形状创建数组
    a = as_strided(x, strides=(0,), shape=(0,))
    check_internal_overlap(a, False)

    # 使用指定步长和形状创建数组
    a = as_strided(x, strides=(0,), shape=(1,))
    check_internal_overlap(a, False)

    # 使用指定步长和形状创建数组
    a = as_strided(x, strides=(0,), shape=(2,))
    check_internal_overlap(a, True)

    # 使用指定步长和形状创建数组
    a = as_strided(x, strides=(0, -9993), shape=(87, 22))
    check_internal_overlap(a, True)

    # 使用指定步长和形状创建数组
    a = as_strided(x, strides=(0, -9993), shape=(1, 22))
    check_internal_overlap(a, False)

    # 使用指定步长和形状创建数组
    a = as_strided(x, strides=(0, -9993), shape=(0, 22))
    check_internal_overlap(a, False)


def test_internal_overlap_fuzz():
    # 模糊测试；暴力检查速度较慢

    x = np.arange(1).astype(np.int8)

    overlap = 0
    no_overlap = 0
    min_count = 100

    rng = np.random.RandomState(1234)

    while min(overlap, no_overlap) < min_count:
        # 随机生成数组的维度和步长
        ndim = rng.randint(1, 4, dtype=np.intp)
        strides = tuple(rng.randint(-1000, 1000, dtype=np.intp)
                        for j in range(ndim))
        shape = tuple(rng.randint(1, 30, dtype=np.intp)
                      for j in range(ndim))

        # 使用指定步长和形状创建数组
        a = as_strided(x, strides=strides, shape=shape)
        # 调用检查函数，获取结果
        result = check_internal_overlap(a)

        # 根据结果增加相应的计数
        if result:
            overlap += 1
        else:
            no_overlap += 1


def test_non_ndarray_inputs():
    # gh-5604的回归检查

    class MyArray:
        def __init__(self, data):
            self.data = data

        @property
        def __array_interface__(self):
            return self.data.__array_interface__

    class MyArray2:
        def __init__(self, data):
            self.data = data

        def __array__(self, dtype=None, copy=None):
            return self.data
    # 遍历包含 MyArray 和 MyArray2 两个类的列表
    for cls in [MyArray, MyArray2]:
        # 创建一个包含五个元素的 NumPy 数组
        x = np.arange(5)
    
        # 断言：检查 cls(x[::2]) 和 x[1::2] 是否可能共享内存
        assert_(np.may_share_memory(cls(x[::2]), x[1::2]))
        # 断言：检查 cls(x[::2]) 和 x[1::2] 是否确实不共享内存
        assert_(not np.shares_memory(cls(x[::2]), x[1::2]))
    
        # 断言：检查 cls(x[1::3]) 和 x[::2] 是否共享内存
        assert_(np.shares_memory(cls(x[1::3]), x[::2]))
        # 断言：检查 cls(x[1::3]) 和 x[::2] 是否可能共享内存
        assert_(np.may_share_memory(cls(x[1::3]), x[::2]))
# 定义一个函数，用于创建一个视图数组，查看输入数组 `x` 中每个元素的第一个字节
def view_element_first_byte(x):
    # 导入需要的模块
    from numpy.lib._stride_tricks_impl import DummyArray
    # 创建一个包含 x 数组接口信息的字典副本
    interface = dict(x.__array_interface__)
    # 设置视图数组的类型字符串为 '|b1'，表示每个元素只有一个字节
    interface['typestr'] = '|b1'
    # 设置描述符，每个元素都是一个字节
    interface['descr'] = [('', '|b1')]
    # 使用接口信息创建并返回一个视图数组，该数组查看了 x 的每个元素的第一个字节
    return np.asarray(DummyArray(interface, x))


# 定义一个函数，验证操作 operation(*args, out=out) 和 out[...] = operation(*args, out=out.copy()) 的等价性
def assert_copy_equivalent(operation, args, out, **kwargs):
    # 设置关键字参数 'out' 为给定的 out
    kwargs['out'] = out
    # 复制 kwargs，并设置其 'out' 参数为 out 的副本
    kwargs2 = dict(kwargs)
    kwargs2['out'] = out.copy()

    # 备份原始的 out 数组
    out_orig = out.copy()
    # 将 out[...] 设置为 operation(*args, **kwargs2) 的结果
    out[...] = operation(*args, **kwargs2)
    # 复制得到预期的结果
    expected = out.copy()
    # 恢复 out 到原始状态
    out[...] = out_orig

    # 调用 operation(*args, **kwargs) 得到实际的结果，并复制该结果
    got = operation(*args, **kwargs).copy()

    # 检查实际结果与预期结果是否有不同的地方，如果有则触发断言错误
    if (got != expected).any():
        assert_equal(got, expected)


# 定义一个测试类 TestUFunc，用于测试 ufunc 调用时的内存重叠处理
class TestUFunc:
    """
    Test ufunc call memory overlap handling
    """
    # 定义一个方法，用于检查一元操作的模糊测试情况
    def check_unary_fuzz(self, operation, get_out_axis_size, dtype=np.int16,
                             count=5000):
        # 定义不同形状的数组维度
        shapes = [7, 13, 8, 21, 29, 32]

        # 创建一个随机数生成器，种子为1234
        rng = np.random.RandomState(1234)

        # 遍历不同维度的数组形状
        for ndim in range(1, 6):
            # 生成随机数组，元素为指定数据类型的随机整数
            x = rng.randint(0, 2**16, size=shapes[:ndim]).astype(dtype)

            # 获取随机视图对的迭代器
            it = iter_random_view_pairs(x, same_steps=False, equal_size=True)

            # 根据维度大小确定最小循环次数
            min_count = count // (ndim + 1)**2

            # 初始化重叠计数
            overlapping = 0
            # 当重叠次数小于最小计数时循环
            while overlapping < min_count:
                # 获取下一对随机视图
                a, b = next(it)

                # 复制原始数组a和b
                a_orig = a.copy()
                b_orig = b.copy()

                # 如果输出轴大小为None
                if get_out_axis_size is None:
                    # 断言通过复制等效执行操作后，输出结果b与a等价
                    assert_copy_equivalent(operation, [a], out=b)

                    # 如果a和b共享内存，则增加重叠计数
                    if np.shares_memory(a, b):
                        overlapping += 1
                else:
                    # 对于每个轴，包括None
                    for axis in itertools.chain(range(ndim), [None]):
                        # 恢复a和b的原始值
                        a[...] = a_orig
                        b[...] = b_orig

                        # 确定减少轴的大小（如果是标量则为None）
                        outsize, scalarize = get_out_axis_size(a, b, axis)
                        # 如果输出大小为'skip'，则跳过本次循环
                        if outsize == 'skip':
                            continue

                        # 切片b以获取正确大小的输出数组
                        sl = [slice(None)] * ndim
                        if axis is None:
                            if outsize is None:
                                sl = [slice(0, 1)] + [0]*(ndim - 1)
                            else:
                                sl = [slice(0, outsize)] + [0]*(ndim - 1)
                        else:
                            if outsize is None:
                                k = b.shape[axis]//2
                                if ndim == 1:
                                    sl[axis] = slice(k, k + 1)
                                else:
                                    sl[axis] = k
                            else:
                                assert b.shape[axis] >= outsize
                                sl[axis] = slice(0, outsize)
                        b_out = b[tuple(sl)]

                        # 如果需要标量化，则重塑b_out
                        if scalarize:
                            b_out = b_out.reshape([])

                        # 如果a和b_out共享内存，则增加重叠计数
                        if np.shares_memory(a, b_out):
                            overlapping += 1

                        # 断言通过复制等效执行操作后，输出结果b_out与a等价，对于指定轴axis
                        assert_copy_equivalent(operation, [a], out=b_out, axis=axis)

    # 标记为慢速测试，测试一元ufunc调用的模糊测试
    @pytest.mark.slow
    def test_unary_ufunc_call_fuzz(self):
        # 测试np.invert的一元ufunc调用
        self.check_unary_fuzz(np.invert, None, np.int16)

    # 标记为慢速测试，测试复数型一元ufunc调用的模糊测试
    @pytest.mark.slow
    def test_unary_ufunc_call_complex_fuzz(self):
        # 复数型通常具有比itemsize更小的对齐方式
        # 测试np.negative的复数型一元ufunc调用
        self.check_unary_fuzz(np.negative, None, np.complex128, count=500)
    # 定义一个测试函数 test_binary_ufunc_accumulate_fuzz，用于测试二元通用函数 np.add.accumulate 的模糊测试
    def test_binary_ufunc_accumulate_fuzz(self):
        
        # 定义内部函数 get_out_axis_size，用于确定输出轴的大小
        def get_out_axis_size(a, b, axis):
            # 如果 axis 为 None
            if axis is None:
                # 如果 a 的维度为 1
                if a.ndim == 1:
                    # 返回 a 的大小和 False（表示不跳过）
                    return a.size, False
                else:
                    # 返回 'skip' 和 False，表示跳过该情况（accumulate 不支持多维数组）
                    return 'skip', False  # accumulate doesn't support this
            else:
                # 返回 a 在指定 axis 上的形状和 False（表示不跳过）
                return a.shape[axis], False

        # 调用 self.check_unary_fuzz 方法，测试 np.add.accumulate 的模糊测试，指定 dtype=np.int16 和 count=500
        self.check_unary_fuzz(np.add.accumulate, get_out_axis_size,
                              dtype=np.int16, count=500)

    # 定义一个测试函数 test_binary_ufunc_reduce_fuzz，用于测试二元通用函数 np.add.reduce 的模糊测试
    def test_binary_ufunc_reduce_fuzz(self):
        
        # 定义内部函数 get_out_axis_size，用于确定输出轴的大小
        def get_out_axis_size(a, b, axis):
            # 总是返回 None 和 (axis is None or a.ndim == 1)
            return None, (axis is None or a.ndim == 1)

        # 调用 self.check_unary_fuzz 方法，测试 np.add.reduce 的模糊测试，指定 dtype=np.int16 和 count=500
        self.check_unary_fuzz(np.add.reduce, get_out_axis_size,
                              dtype=np.int16, count=500)

    # 定义一个测试函数 test_binary_ufunc_reduceat_fuzz，用于测试二元通用函数 np.add.reduceat 的模糊测试
    def test_binary_ufunc_reduceat_fuzz(self):
        
        # 定义内部函数 get_out_axis_size，用于确定输出轴的大小
        def get_out_axis_size(a, b, axis):
            # 如果 axis 为 None
            if axis is None:
                # 如果 a 的维度为 1
                if a.ndim == 1:
                    # 返回 a 的大小和 False（表示不跳过）
                    return a.size, False
                else:
                    # 返回 'skip' 和 False，表示跳过该情况（reduceat 不支持多维数组）
                    return 'skip', False  # reduceat doesn't support this
            else:
                # 返回 a 在指定 axis 上的形状和 False（表示不跳过）
                return a.shape[axis], False

        # 定义内部函数 do_reduceat，用于执行 np.add.reduceat 的操作
        def do_reduceat(a, out, axis):
            # 如果 axis 为 None
            if axis is None:
                # 计算数组 a 的大小和步长
                size = len(a)
                step = size // len(out)
            else:
                # 计算数组 a 在指定 axis 上的大小和步长
                size = a.shape[axis]
                step = a.shape[axis] // out.shape[axis]
            # 构造索引数组 idx，用于 reduceat 操作
            idx = np.arange(0, size, step)
            # 执行 np.add.reduceat 操作，并将结果存入 out，指定 axis
            return np.add.reduceat(a, idx, out=out, axis=axis)

        # 调用 self.check_unary_fuzz 方法，测试 do_reduceat 函数的模糊测试，指定 dtype=np.int16 和 count=500
        self.check_unary_fuzz(do_reduceat, get_out_axis_size,
                              dtype=np.int16, count=500)

    # 定义一个测试函数 test_binary_ufunc_reduceat_manual，用于手动测试二元通用函数 np.add.reduceat 的功能
    def test_binary_ufunc_reduceat_manual(self):
        
        # 定义内部函数 check，用于检查 reduceat 操作的输出
        def check(ufunc, a, ind, out):
            # 分别使用复制的输入数组执行 reduceat 操作，并比较结果
            c1 = ufunc.reduceat(a.copy(), ind.copy(), out=out.copy())
            c2 = ufunc.reduceat(a, ind, out=out)
            # 使用 assert_array_equal 断言，确保两次操作的结果相等
            assert_array_equal(c1, c2)

        # 测试情况1：完全相同的输入和输出数组
        a = np.arange(10000, dtype=np.int16)
        check(np.add, a, a[::-1].copy(), a)

        # 测试情况2：索引数组与输入数组重叠
        a = np.arange(10000, dtype=np.int16)
        check(np.add, a, a[::-1], a)

    # 使用 pytest.mark.slow 标记该测试函数为慢速测试
    # 定义一个测试函数，用于测试一元通用函数的模糊测试
    def test_unary_gufunc_fuzz(self):
        # 定义不同形状的数组
        shapes = [7, 13, 8, 21, 29, 32]
        # 获取欧几里得距离计算函数的引用
        gufunc = _umath_tests.euclidean_pdist

        # 使用种子为1234的随机数生成器
        rng = np.random.RandomState(1234)

        # 循环测试数组维度从2到5
        for ndim in range(2, 6):
            # 根据当前维度随机生成数组 x
            x = rng.rand(*shapes[:ndim])

            # 生成随机视图对的迭代器
            it = iter_random_view_pairs(x, same_steps=False, equal_size=True)

            # 计算最小重叠次数
            min_count = 500 // (ndim + 1)**2

            # 初始化重叠计数
            overlapping = 0
            # 当重叠次数未达到最小要求时循环
            while overlapping < min_count:
                # 获取下一对视图 a, b
                a, b = next(it)

                # 如果 a 或 b 的最小形状小于2，或者 a 的最后一个维度小于2，则继续下一轮循环
                if min(a.shape[-2:]) < 2 or min(b.shape[-2:]) < 2 or a.shape[-1] < 2:
                    continue

                # 确保形状满足欧几里得距离计算函数的要求
                if b.shape[-1] > b.shape[-2]:
                    b = b[...,0,:]
                else:
                    b = b[...,:,0]

                # 计算 a 的形状
                n = a.shape[-2]
                # 计算可能的对数
                p = n * (n - 1) // 2
                # 如果 p 小于等于 b 的最后一个维度并且大于0，则截取 b 的形状
                if p <= b.shape[-1] and p > 0:
                    b = b[...,:p]
                else:
                    # 否则重新计算 n，使得 n 大于等于2，并计算新的 p
                    n = max(2, int(np.sqrt(b.shape[-1]))//2)
                    p = n * (n - 1) // 2
                    # 截取 a 和 b 的新形状
                    a = a[...,:n,:]
                    b = b[...,:p]

                # 检查 a 和 b 是否共享内存，若是则增加重叠计数
                if np.shares_memory(a, b):
                    overlapping += 1

                # 忽略溢出和无效操作的错误，调用 gufunc 并断言输出与 b 等效
                with np.errstate(over='ignore', invalid='ignore'):
                    assert_copy_equivalent(gufunc, [a], out=b)

    # 定义测试 ufunc 在指定位置进行操作的函数
    def test_ufunc_at_manual(self):
        # 定义一个内部函数 check，用于验证 ufunc 在指定位置进行操作的正确性
        def check(ufunc, a, ind, b=None):
            # 复制数组 a 到 a0
            a0 = a.copy()
            # 如果 b 为空，则调用 ufunc.at(a0, ind) 和 ufunc.at(a, ind)，并断言结果相等
            if b is None:
                ufunc.at(a0, ind.copy())
                c1 = a0.copy()
                ufunc.at(a, ind)
                c2 = a.copy()
            else:
                # 否则调用 ufunc.at(a0, ind, b) 和 ufunc.at(a, ind, b)，并断言结果相等
                ufunc.at(a0, ind.copy(), b.copy())
                c1 = a0.copy()
                ufunc.at(a, ind, b)
                c2 = a.copy()
            assert_array_equal(c1, c2)

        # 测试反转数组 a 并与其索引重叠的情况
        a = np.arange(10000, dtype=np.int16)
        check(np.invert, a[::-1], a)

        # 测试在第二个数据数组上进行 ufunc 操作
        a = np.arange(100, dtype=np.int16)
        ind = np.arange(0, 100, 2, dtype=np.int16)
        check(np.add, a, ind, a[25:75])
    def test_unary_ufunc_1d_manual(self):
        # 测试一维数组的单目通用函数（ufunc）的快速路径（避免创建 `np.nditer`）

        def check(a, b):
            # 复制输入数组，以便在每次测试前恢复初始状态
            a_orig = a.copy()
            b_orig = b.copy()

            # 使用输出数组 b0 进行 ufunc 操作，并保存结果到 c1
            b0 = b.copy()
            c1 = ufunc(a, out=b0)
            # 使用相同的输出数组 b 进行 ufunc 操作，并保存结果到 c2
            c2 = ufunc(a, out=b)
            # 检查 c1 和 c2 是否相等
            assert_array_equal(c1, c2)

            # 触发“复杂 ufunc 循环”代码路径
            # 使用视图操作获取 b 的第一个字节，并转换为布尔类型的掩码
            mask = view_element_first_byte(b).view(np.bool)

            # 恢复输入数组和输出数组的初始状态
            a[...] = a_orig
            b[...] = b_orig
            # 使用复制的输出数组和复制的掩码进行 ufunc 操作，并保存结果到 c1
            c1 = ufunc(a, out=b.copy(), where=mask.copy()).copy()

            # 恢复输入数组和输出数组的初始状态
            a[...] = a_orig
            b[...] = b_orig
            # 使用原始输出数组和复制的掩码进行 ufunc 操作，并保存结果到 c2
            c2 = ufunc(a, out=b, where=mask.copy()).copy()

            # 恢复输入数组和输出数组的初始状态
            a[...] = a_orig
            b[...] = b_orig
            # 使用原始输出数组和原始掩码进行 ufunc 操作，并保存结果到 c3
            c3 = ufunc(a, out=b, where=mask).copy()

            # 检查 c1、c2 和 c3 是否相等
            assert_array_equal(c1, c2)
            assert_array_equal(c1, c3)

        # 定义被测试的数据类型列表
        dtypes = [np.int8, np.int16, np.int32, np.int64, np.float32,
                  np.float64, np.complex64, np.complex128]
        dtypes = [np.dtype(x) for x in dtypes]

        # 遍历每种数据类型进行测试
        for dtype in dtypes:
            # 根据数据类型选择对应的 ufunc 函数
            if np.issubdtype(dtype, np.integer):
                ufunc = np.invert
            else:
                ufunc = np.reciprocal

            # 设定测试中的数组大小和索引表达式
            n = 1000
            k = 10
            indices = [
                np.index_exp[:n],
                np.index_exp[k:k+n],
                np.index_exp[n-1::-1],
                np.index_exp[k+n-1:k-1:-1],
                np.index_exp[:2*n:2],
                np.index_exp[k:k+2*n:2],
                np.index_exp[2*n-1::-2],
                np.index_exp[k+2*n-1:k-1:-2],
            ]

            # 使用 itertools 生成索引对，并进行测试
            for xi, yi in itertools.product(indices, indices):
                v = np.arange(1, 1 + n*2 + k, dtype=dtype)
                x = v[xi]
                y = v[yi]

                # 忽略所有错误状态，执行 check 函数进行测试
                with np.errstate(all='ignore'):
                    check(x, y)

                    # 测试标量情况
                    check(x[:1], y)
                    check(x[-1:], y)
                    check(x[:1].reshape([]), y)
                    check(x[-1:].reshape([]), y)

    def test_unary_ufunc_where_same(self):
        # 检查 wheremask 重叠时的行为
        ufunc = np.invert

        def check(a, out, mask):
            # 使用复制的掩码和输出数组进行 ufunc 操作，并保存结果到 c1
            c1 = ufunc(a, out=out.copy(), where=mask.copy())
            # 使用原始掩码和输出数组进行 ufunc 操作，并保存结果到 c2
            c2 = ufunc(a, out=out, where=mask)
            # 检查 c1 和 c2 是否相等
            assert_array_equal(c1, c2)

        # 使用布尔类型的数组 x 进行测试
        x = np.arange(100).astype(np.bool)
        # 测试不同的输入和输出组合及其重叠的掩码情况
        check(x, x, x)
        check(x, x.copy(), x)
        check(x, x, x.copy())

    @pytest.mark.slow
    # 定义一个测试函数，用于测试一维数组的逐元素操作（ufunc）
    def test_binary_ufunc_1d_manual(self):
        # 定义要测试的二元通用函数为 np.add
        ufunc = np.add

        # 定义检查函数，用于比较三个数组的逐元素运算结果
        def check(a, b, c):
            # 复制 c 到 c0
            c0 = c.copy()
            # 使用 ufunc 对 a 和 b 执行逐元素运算，结果存入 c0
            c1 = ufunc(a, b, out=c0)
            # 直接使用 ufunc 对 a 和 b 执行逐元素运算，结果存入 c
            c2 = ufunc(a, b, out=c)
            # 断言 c1 和 c2 相等
            assert_array_equal(c1, c2)

        # 遍历不同的数据类型进行测试
        for dtype in [np.int8, np.int16, np.int32, np.int64,
                      np.float32, np.float64, np.complex64, np.complex128]:
            # 检查不同数据依赖顺序的情况

            n = 1000  # 设置数组长度
            k = 10    # 设置步长

            indices = []
            # 使用 itertools.product 生成三重循环的索引组合
            for p in [1, 2]:
                indices.extend([
                    np.index_exp[:p*n:p],           # 切片索引
                    np.index_exp[k:k+p*n:p],        # 起始偏移索引
                    np.index_exp[p*n-1::-p],        # 反向步长索引
                    np.index_exp[k+p*n-1:k-1:-p],   # 起始偏移反向步长索引
                ])

            # 对 indices 中的组合进行排列组合，形成测试样例
            for x, y, z in itertools.product(indices, indices, indices):
                # 创建一个长度为 6*n 的数组，并转换为指定数据类型 dtype
                v = np.arange(6*n).astype(dtype)
                # 根据索引 x，y，z 分别取出对应的子数组
                x = v[x]
                y = v[y]
                z = v[z]

                # 调用 check 函数检查逐元素操作结果
                check(x, y, z)

                # 对标量情况进行检查
                check(x[:1], y, z)
                check(x[-1:], y, z)
                check(x[:1].reshape([]), y, z)
                check(x[-1:].reshape([]), y, z)
                check(x, y[:1], z)
                check(x, y[-1:], z)
                check(x, y[:1].reshape([]), z)
                check(x, y[-1:].reshape([]), z)

    # 定义一个测试函数，用于测试简单的原地操作
    def test_inplace_op_simple_manual(self):
        # 创建一个随机数生成器，并生成一个 200x200 的随机数组 x
        rng = np.random.RandomState(1234)
        x = rng.rand(200, 200)  # 大于缓冲区大小

        # 执行 x 和其转置矩阵的原地相加操作
        x += x.T
        # 断言 x 减去其转置矩阵后的结果全为 0
        assert_array_equal(x - x.T, 0)
```