# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_decomp_update.py`

```
# 导入 itertools 库，用于生成迭代器
import itertools

# 导入 numpy 库，并从中导入需要使用的测试函数 assert_, assert_allclose, assert_equal
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal

# 导入 pytest 库，并将 raises 别名为 assert_raises
from pytest import raises as assert_raises

# 导入 scipy 库，并从中导入 linalg 模块
from scipy import linalg

# 导入 scipy.linalg._decomp_update 模块，并从中导入 qr_delete, qr_update, qr_insert 函数
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert

# 定义一个 assert_unitary 函数，用于验证矩阵是否为单位矩阵
def assert_unitary(a, rtol=None, atol=None, assert_sqr=True):
    # 如果 rtol 未指定，则设为默认值
    if rtol is None:
        rtol = 10.0 ** -(np.finfo(a.dtype).precision-2)
    # 如果 atol 未指定，则设为默认值
    if atol is None:
        atol = 10*np.finfo(a.dtype).eps

    # 如果 assert_sqr 为真，则验证矩阵为方阵
    if assert_sqr:
        assert_(a.shape[0] == a.shape[1], 'unitary matrices must be square')
    
    # 计算矩阵的共轭转置与自身的乘积，并验证其接近单位矩阵
    aTa = np.dot(a.T.conj(), a)
    assert_allclose(aTa, np.eye(a.shape[1]), rtol=rtol, atol=atol)

# 定义一个 assert_upper_tri 函数，用于验证矩阵是否为上三角矩阵
def assert_upper_tri(a, rtol=None, atol=None):
    # 如果 rtol 未指定，则设为默认值
    if rtol is None:
        rtol = 10.0 ** -(np.finfo(a.dtype).precision-2)
    # 如果 atol 未指定，则设为默认值
    if atol is None:
        atol = 2*np.finfo(a.dtype).eps
    
    # 创建一个上三角矩阵的掩码
    mask = np.tri(a.shape[0], a.shape[1], -1, np.bool_)
    # 验证矩阵在上三角位置的元素接近于 0
    assert_allclose(a[mask], 0.0, rtol=rtol, atol=atol)

# 定义一个 check_qr 函数，用于验证 QR 分解的正确性
def check_qr(q, r, a, rtol, atol, assert_sqr=True):
    # 验证 Q 矩阵为单位矩阵
    assert_unitary(q, rtol, atol, assert_sqr)
    # 验证 R 矩阵为上三角矩阵
    assert_upper_tri(r, rtol, atol)
    # 验证 Q*R 等于原始矩阵 A
    assert_allclose(q.dot(r), a, rtol=rtol, atol=atol)

# 定义一个 make_strided 函数，用于创建具有指定步幅的数组视图
def make_strided(arrs):
    # 定义一组步幅
    strides = [(3, 7), (2, 2), (3, 4), (4, 2), (5, 4), (2, 3), (2, 1), (4, 5)]
    # 计算步幅列表的长度
    kmax = len(strides)
    k = 0
    ret = []
    # 遍历输入的数组列表
    for a in arrs:
        # 如果数组是一维的
        if a.ndim == 1:
            # 取出当前步幅
            s = strides[k % kmax]
            k += 1
            # 创建一个全零数组作为基础数组
            base = np.zeros(s[0]*a.shape[0]+s[1], a.dtype)
            # 从基础数组中取出视图，按步幅将原始数组复制进去
            view = base[s[1]::s[0]]
            view[...] = a
        # 如果数组是二维的
        elif a.ndim == 2:
            # 取出当前步幅和下一个步幅
            s = strides[k % kmax]
            t = strides[(k+1) % kmax]
            k += 2
            # 创建一个全零数组作为基础数组，形状为根据两个步幅计算得出
            base = np.zeros((s[0]*a.shape[0]+s[1], t[0]*a.shape[1]+t[1]),
                            a.dtype)
            # 从基础数组中取出视图，按两个步幅将原始数组复制进去
            view = base[s[1]::s[0], t[1]::t[0]]
            view[...] = a
        else:
            # 抛出错误，仅支持一维或二维数组
            raise ValueError('make_strided only works for ndim = 1 or'
                             ' 2 arrays')
        ret.append(view)
    return ret

# 定义一个 negate_strides 函数，用于生成数组的逆步幅视图
def negate_strides(arrs):
    ret = []
    # 遍历输入的数组列表
    for a in arrs:
        # 创建一个与输入数组相同形状的全零数组
        b = np.zeros_like(a)
        # 如果数组是二维的，将其逆序排列
        if b.ndim == 2:
            b = b[::-1, ::-1]
        # 如果数组是一维的，将其逆序排列
        elif b.ndim == 1:
            b = b[::-1]
        else:
            # 抛出错误，仅支持一维或二维数组
            raise ValueError('negate_strides only works for ndim = 1 or'
                             ' 2 arrays')
        # 将原始数组的值复制到逆步幅数组中
        b[...] = a
        ret.append(b)
    return ret

# 定义一个 nonitemsize_strides 函数，用于生成具有非本机字节顺序的数组
def nonitemsize_strides(arrs):
    out = []
    # 遍历输入的数组列表
    for a in arrs:
        # 获取数组的数据类型
        a_dtype = a.dtype
        # 创建一个形状与输入数组相同的数组，但数据类型为结构化数组
        b = np.zeros(a.shape, [('a', a_dtype), ('junk', 'S1')])
        # 从结构化数组中取出原始数据类型的视图
        c = b.getfield(a_dtype)
        # 将原始数组的值复制到结构化数组的视图中
        c[...] = a
        out.append(c)
    return out

# 定义一个 make_nonnative 函数，用于生成具有非本机字节顺序的数组
def make_nonnative(arrs):
    # 将输入数组转换为具有与当前系统不同字节顺序的数组
    return [a.astype(a.dtype.newbyteorder()) for a in arrs]

# 定义一个 BaseQRdeltas 类，用于测试 QR 分解时的设置
class BaseQRdeltas:
    # 初始化方法，设置默认的相对误差和绝对误差
    def setup_method(self):
        self.rtol = 10.0 ** -(np.finfo(self.dtype).precision-2)
        self.atol = 10 * np.finfo(self.dtype).eps
    # 定义一个生成矩阵的方法，接受类型 `type` 和模式 `mode` 参数，其中模式默认为 'full'
    def generate(self, type, mode='full'):
        # 设置随机种子，确保生成的随机数可复现性
        np.random.seed(29382)
        # 根据不同的类型选择相应的矩阵形状
        shape = {'sqr': (8, 8), 'tall': (12, 7), 'fat': (7, 12),
                 'Mx1': (8, 1), '1xN': (1, 8), '1x1': (1, 1)}[type]
        # 生成一个指定形状的随机实数数组
        a = np.random.random(shape)
        # 如果数据类型包含复数，则生成一个相同形状的随机实数数组作为虚部，并合成复数数组
        if np.iscomplexobj(self.dtype.type(1)):
            b = np.random.random(shape)
            a = a + 1j * b
        # 将数组 `a` 转换为指定的数据类型 `self.dtype`
        a = a.astype(self.dtype)
        # 对数组 `a` 进行 QR 分解，得到正交矩阵 `q` 和上三角矩阵 `r`
        q, r = linalg.qr(a, mode=mode)
        # 返回生成的数组 `a`、正交矩阵 `q` 和上三角矩阵 `r`
        return a, q, r
# 继承自 BaseQRdeltas 类，用于测试 QR 分解中删除行或列的功能
class BaseQRdelete(BaseQRdeltas):

    # 测试删除单行的情况
    def test_sqr_1_row(self):
        # 生成一个方阵 a、Q、R
        a, q, r = self.generate('sqr')
        # 遍历 R 的每一行
        for row in range(r.shape[0]):
            # 删除第 row 行后得到新的 Q、R
            q1, r1 = qr_delete(q, r, row, overwrite_qr=False)
            # 删除数组 a 的第 row 行
            a1 = np.delete(a, row, 0)
            # 检查新的 Q、R 是否满足 QR 分解的要求
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试删除多行的情况
    def test_sqr_p_row(self):
        a, q, r = self.generate('sqr')
        # 从删除两行到五行的情况进行遍历
        for ndel in range(2, 6):
            # 从第一行开始遍历到可以保留 ndel 行之前的每一行
            for row in range(a.shape[0]-ndel):
                # 删除从第 row 行开始的 ndel 行，得到新的 Q、R
                q1, r1 = qr_delete(q, r, row, ndel, overwrite_qr=False)
                # 删除数组 a 的第 row 至 row+ndel 行
                a1 = np.delete(a, slice(row, row+ndel), 0)
                # 检查新的 Q、R 是否满足 QR 分解的要求
                check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试删除单列的情况
    def test_sqr_1_col(self):
        a, q, r = self.generate('sqr')
        # 遍历 R 的每一列
        for col in range(r.shape[1]):
            # 删除第 col 列后得到新的 Q、R
            q1, r1 = qr_delete(q, r, col, which='col', overwrite_qr=False)
            # 删除数组 a 的第 col 列
            a1 = np.delete(a, col, 1)
            # 检查新的 Q、R 是否满足 QR 分解的要求
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试删除多列的情况
    def test_sqr_p_col(self):
        a, q, r = self.generate('sqr')
        for ndel in range(2, 6):
            # 从第一列开始遍历到可以保留 ndel 列之前的每一列
            for col in range(r.shape[1]-ndel):
                # 删除从第 col 列开始的 ndel 列，得到新的 Q、R
                q1, r1 = qr_delete(q, r, col, ndel, which='col', overwrite_qr=False)
                # 删除数组 a 的第 col 至 col+ndel 列
                a1 = np.delete(a, slice(col, col+ndel), 1)
                # 检查新的 Q、R 是否满足 QR 分解的要求
                check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试删除单行的情况（矩阵为纵向长条形）
    def test_tall_1_row(self):
        a, q, r = self.generate('tall')
        for row in range(r.shape[0]):
            q1, r1 = qr_delete(q, r, row, overwrite_qr=False)
            a1 = np.delete(a, row, 0)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试删除多行的情况（矩阵为纵向长条形）
    def test_tall_p_row(self):
        a, q, r = self.generate('tall')
        for ndel in range(2, 6):
            for row in range(a.shape[0]-ndel):
                q1, r1 = qr_delete(q, r, row, ndel, overwrite_qr=False)
                a1 = np.delete(a, slice(row, row+ndel), 0)
                check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试删除单列的情况（矩阵为横向长条形）
    def test_tall_1_col(self):
        a, q, r = self.generate('tall')
        for col in range(r.shape[1]):
            q1, r1 = qr_delete(q, r, col, which='col', overwrite_qr=False)
            a1 = np.delete(a, col, 1)
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试删除多列的情况（矩阵为横向长条形）
    def test_tall_p_col(self):
        a, q, r = self.generate('tall')
        for ndel in range(2, 6):
            for col in range(r.shape[1]-ndel):
                q1, r1 = qr_delete(q, r, col, ndel, which='col', overwrite_qr=False)
                a1 = np.delete(a, slice(col, col+ndel), 1)
                check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试删除单行的情况（矩阵为横向长条形）
    def test_fat_1_row(self):
        a, q, r = self.generate('fat')
        for row in range(r.shape[0]):
            q1, r1 = qr_delete(q, r, row, overwrite_qr=False)
            a1 = np.delete(a, row, 0)
            check_qr(q1, r1, a1, self.rtol, self.atol)
    # 测试行数大于列数的情况下的 QR 分解删除操作
    def test_fat_p_row(self):
        # 生成一个“fat”型矩阵的数据
        a, q, r = self.generate('fat')
        # 对于删除操作的行数从2到5
        for ndel in range(2, 6):
            # 对矩阵每一行（不包括被删除的行）进行操作
            for row in range(a.shape[0]-ndel):
                # 执行 QR 分解的删除操作，不覆写 Q 和 R 矩阵
                q1, r1 = qr_delete(q, r, row, ndel, overwrite_qr=False)
                # 删除原始矩阵中的指定行
                a1 = np.delete(a, slice(row, row+ndel), 0)
                # 检查删除后的 QR 分解结果与矩阵是否匹配
                check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试列数大于行数的情况下的 QR 分解删除操作
    def test_fat_1_col(self):
        # 生成一个“fat”型矩阵的数据
        a, q, r = self.generate('fat')
        # 对于每一列进行操作
        for col in range(r.shape[1]):
            # 执行 QR 分解的列删除操作，不覆写 Q 和 R 矩阵
            q1, r1 = qr_delete(q, r, col, which='col', overwrite_qr=False)
            # 删除原始矩阵中的指定列
            a1 = np.delete(a, col, 1)
            # 检查删除后的 QR 分解结果与矩阵是否匹配
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试行数大于列数的情况下的 QR 分解删除操作（经济型）
    def test_fat_p_col(self):
        # 生成一个“fat”型矩阵的数据
        a, q, r = self.generate('fat')
        # 对于删除操作的列数从2到5
        for ndel in range(2, 6):
            # 对矩阵每一列（不包括被删除的列）进行操作
            for col in range(r.shape[1]-ndel):
                # 执行 QR 分解的删除操作，不覆写 Q 和 R 矩阵
                q1, r1 = qr_delete(q, r, col, ndel, which='col',
                                   overwrite_qr=False)
                # 删除原始矩阵中的指定列
                a1 = np.delete(a, slice(col, col+ndel), 1)
                # 检查删除后的 QR 分解结果与矩阵是否匹配
                check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试行数小于列数的情况下的 QR 分解删除操作（经济型）
    def test_economic_1_row(self):
        # 生成一个“tall”型经济型矩阵的数据
        a, q, r = self.generate('tall', 'economic')
        # 对矩阵的每一行进行操作
        for row in range(r.shape[0]):
            # 执行 QR 分解的行删除操作，不覆写 Q 和 R 矩阵
            q1, r1 = qr_delete(q, r, row, overwrite_qr=False)
            # 删除原始矩阵中的指定行
            a1 = np.delete(a, row, 0)
            # 检查删除后的 QR 分解结果与矩阵是否匹配，不允许 Q 和 R 重写
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    # 经济型矩阵行删除操作的基础测试函数
    # a - prow = eco, a - prow = sqr, a - prow = fat
    def base_economic_p_row_xxx(self, ndel):
        # 生成一个“tall”型经济型矩阵的数据
        a, q, r = self.generate('tall', 'economic')
        # 对矩阵从第一行开始到要删除的行数之前的每一行进行操作
        for row in range(a.shape[0]-ndel):
            # 执行 QR 分解的行删除操作，不覆写 Q 和 R 矩阵
            q1, r1 = qr_delete(q, r, row, ndel, overwrite_qr=False)
            # 删除原始矩阵中的指定行
            a1 = np.delete(a, slice(row, row+ndel), 0)
            # 检查删除后的 QR 分解结果与矩阵是否匹配，不允许 Q 和 R 重写
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    # 测试经济型矩阵行删除操作后保持经济型矩阵的情况
    def test_economic_p_row_economic(self):
        # (12, 7) - (3, 7) = (9,7) --> 保持经济型
        self.base_economic_p_row_xxx(3)

    # 测试经济型矩阵行删除操作后变为方阵的情况
    def test_economic_p_row_sqr(self):
        # (12, 7) - (5, 7) = (7, 7) --> 变为方阵
        self.base_economic_p_row_xxx(5)

    # 测试经济型矩阵行删除操作后变为“fat”型矩阵的情况
    def test_economic_p_row_fat(self):
        # (12, 7) - (7,7) = (5, 7) --> 变为“fat”型
        self.base_economic_p_row_xxx(7)

    # 测试经济型矩阵列删除操作
    def test_economic_1_col(self):
        # 生成一个“tall”型经济型矩阵的数据
        a, q, r = self.generate('tall', 'economic')
        # 对矩阵的每一列进行操作
        for col in range(r.shape[1]):
            # 执行 QR 分解的列删除操作，不覆写 Q 和 R 矩阵
            q1, r1 = qr_delete(q, r, col, which='col', overwrite_qr=False)
            # 删除原始矩阵中的指定列
            a1 = np.delete(a, col, 1)
            # 检查删除后的 QR 分解结果与矩阵是否匹配，不允许 Q 和 R 重写
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    # 测试经济型矩阵列删除操作
    def test_economic_p_col(self):
        # 生成一个“tall”型经济型矩阵的数据
        a, q, r = self.generate('tall', 'economic')
        # 对于删除操作的列数从2到5
        for ndel in range(2, 6):
            # 对矩阵的每一列（不包括被删除的列）进行操作
            for col in range(r.shape[1]-ndel):
                # 执行 QR 分解的列删除操作，不覆写 Q 和 R 矩阵
                q1, r1 = qr_delete(q, r, col, ndel, which='col',
                                   overwrite_qr=False)
                # 删除原始矩阵中的指定列
                a1 = np.delete(a, slice(col, col+ndel), 1)
                # 检查删除后的 QR 分解结果与矩阵是否匹配，不允许 Q 和 R 重写
                check_qr(q1, r1, a1, self.rtol, self.atol, False)
    # 测试删除单行的情况
    def test_Mx1_1_row(self):
        # 生成测试数据
        a, q, r = self.generate('Mx1')
        # 遍历矩阵的每一行
        for row in range(r.shape[0]):
            # 调用 qr_delete 函数删除指定行，不覆盖原始 QR 分解结果
            q1, r1 = qr_delete(q, r, row, overwrite_qr=False)
            # 使用 NumPy 删除指定行的数据
            a1 = np.delete(a, row, 0)
            # 检查 QR 分解结果是否正确
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试删除多行的情况
    def test_Mx1_p_row(self):
        # 生成测试数据
        a, q, r = self.generate('Mx1')
        # 循环不同删除数量的情况
        for ndel in range(2, 6):
            # 遍历矩阵中每一行，注意不包括被删除的行
            for row in range(a.shape[0]-ndel):
                # 调用 qr_delete 函数删除指定行数的行，不覆盖原始 QR 分解结果
                q1, r1 = qr_delete(q, r, row, ndel, overwrite_qr=False)
                # 使用 NumPy 删除指定行范围的数据
                a1 = np.delete(a, slice(row, row+ndel), 0)
                # 检查 QR 分解结果是否正确
                check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试删除单列的情况
    def test_1xN_1_col(self):
        # 生成测试数据
        a, q, r = self.generate('1xN')
        # 遍历矩阵的每一列
        for col in range(r.shape[1]):
            # 调用 qr_delete 函数删除指定列，不覆盖原始 QR 分解结果
            q1, r1 = qr_delete(q, r, col, which='col', overwrite_qr=False)
            # 使用 NumPy 删除指定列的数据
            a1 = np.delete(a, col, 1)
            # 检查 QR 分解结果是否正确
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试删除多列的情况
    def test_1xN_p_col(self):
        # 生成测试数据
        a, q, r = self.generate('1xN')
        # 循环不同删除数量的情况
        for ndel in range(2, 6):
            # 遍历矩阵中每一列，注意不包括被删除的列
            for col in range(r.shape[1]-ndel):
                # 调用 qr_delete 函数删除指定列数的列，不覆盖原始 QR 分解结果
                q1, r1 = qr_delete(q, r, col, ndel, which='col',
                                   overwrite_qr=False)
                # 使用 NumPy 删除指定列范围的数据
                a1 = np.delete(a, slice(col, col+ndel), 1)
                # 检查 QR 分解结果是否正确
                check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试删除单行（经济模式）的情况
    def test_Mx1_economic_1_row(self):
        # 生成测试数据（经济模式）
        a, q, r = self.generate('Mx1', 'economic')
        # 遍历矩阵的每一行
        for row in range(r.shape[0]):
            # 调用 qr_delete 函数删除指定行，不覆盖原始 QR 分解结果
            q1, r1 = qr_delete(q, r, row, overwrite_qr=False)
            # 使用 NumPy 删除指定行的数据
            a1 = np.delete(a, row, 0)
            # 检查 QR 分解结果是否正确，不检查 Q 矩阵的正交性
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    # 测试删除多行（经济模式）的情况
    def test_Mx1_economic_p_row(self):
        # 生成测试数据（经济模式）
        a, q, r = self.generate('Mx1', 'economic')
        # 循环不同删除数量的情况
        for ndel in range(2, 6):
            # 遍历矩阵中每一行，注意不包括被删除的行
            for row in range(a.shape[0]-ndel):
                # 调用 qr_delete 函数删除指定行数的行，不覆盖原始 QR 分解结果
                q1, r1 = qr_delete(q, r, row, ndel, overwrite_qr=False)
                # 使用 NumPy 删除指定行范围的数据
                a1 = np.delete(a, slice(row, row+ndel), 0)
                # 检查 QR 分解结果是否正确，不检查 Q 矩阵的正交性
                check_qr(q1, r1, a1, self.rtol, self.atol, False)

    # 测试删除最后一行的情况
    def test_delete_last_1_row(self):
        # 生成测试数据（1xN）
        a, q, r = self.generate('1xN')
        # 调用 qr_delete 函数删除第一行，删除一个行
        q1, r1 = qr_delete(q, r, 0, 1, 'row')
        # 断言删除后的 Q 矩阵为空矩阵
        assert_equal(q1, np.ndarray(shape=(0, 0), dtype=q.dtype))
        # 断言删除后的 R 矩阵为全零矩阵，形状与原始 R 矩阵的列数相同
        assert_equal(r1, np.ndarray(shape=(0, r.shape[1]), dtype=r.dtype))

    # 测试删除最后多行的情况
    def test_delete_last_p_row(self):
        # 生成测试数据（高瘦矩阵，全列填充模式）
        a, q, r = self.generate('tall', 'full')
        # 调用 qr_delete 函数删除所有行，即删除整个矩阵
        q1, r1 = qr_delete(q, r, 0, a.shape[0], 'row')
        # 断言删除后的 Q 矩阵为空矩阵
        assert_equal(q1, np.ndarray(shape=(0, 0), dtype=q.dtype))
        # 断言删除后的 R 矩阵为全零矩阵，形状与原始 R 矩阵的列数相同
        assert_equal(r1, np.ndarray(shape=(0, r.shape[1]), dtype=r.dtype))

        # 生成测试数据（高瘦矩阵，经济模式）
        a, q, r = self.generate('tall', 'economic')
        # 调用 qr_delete 函数删除所有行，即删除整个矩阵
        q1, r1 = qr_delete(q, r, 0, a.shape[0], 'row')
        # 断言删除后的 Q 矩阵为空矩阵
        assert_equal(q1, np.ndarray(shape=(0, 0), dtype=q.dtype))
        # 断言删除后的 R 矩阵为全零矩阵，形状与原始 R 矩阵的列数相同
        assert_equal(r1, np.ndarray(shape=(0, r.shape[1]), dtype=r.dtype))
    # 定义测试函数，用于测试删除最后一列的情况
    def test_delete_last_1_col(self):
        # 生成测试数据
        a, q, r = self.generate('Mx1', 'economic')
        # 调用 qr_delete 函数删除第 1 列
        q1, r1 = qr_delete(q, r, 0, 1, 'col')
        # 断言删除后的结果符合预期：q1 应为零矩阵
        assert_equal(q1, np.ndarray(shape=(q.shape[0], 0), dtype=q.dtype))
        # 断言删除后的结果符合预期：r1 应为零矩阵
        assert_equal(r1, np.ndarray(shape=(0, 0), dtype=r.dtype))

        # 生成测试数据
        a, q, r = self.generate('Mx1', 'full')
        # 再次调用 qr_delete 函数删除第 1 列
        q1, r1 = qr_delete(q, r, 0, 1, 'col')
        # 断言删除后的结果：q1 应为单位矩阵
        assert_unitary(q1)
        # 断言 q1 的数据类型与 q 相同
        assert_(q1.dtype == q.dtype)
        # 断言 q1 的形状与 q 相同
        assert_(q1.shape == q.shape)
        # 断言删除后的结果符合预期：r1 应为零矩阵
        assert_equal(r1, np.ndarray(shape=(r.shape[0], 0), dtype=r.dtype))

    # 定义测试函数，用于测试删除最后 p 列的情况
    def test_delete_last_p_col(self):
        # 生成测试数据
        a, q, r = self.generate('tall', 'full')
        # 调用 qr_delete 函数删除最后 p 列
        q1, r1 = qr_delete(q, r, 0, a.shape[1], 'col')
        # 断言删除后的结果：q1 应为单位矩阵
        assert_unitary(q1)
        # 断言 q1 的数据类型与 q 相同
        assert_(q1.dtype == q.dtype)
        # 断言 q1 的形状与 q 相同
        assert_(q1.shape == q.shape)
        # 断言删除后的结果符合预期：r1 应为零矩阵
        assert_equal(r1, np.ndarray(shape=(r.shape[0], 0), dtype=r.dtype))

        # 生成测试数据
        a, q, r = self.generate('tall', 'economic')
        # 再次调用 qr_delete 函数删除最后 p 列
        q1, r1 = qr_delete(q, r, 0, a.shape[1], 'col')
        # 断言删除后的结果符合预期：q1 应为零矩阵
        assert_equal(q1, np.ndarray(shape=(q.shape[0], 0), dtype=q.dtype))
        # 断言删除后的结果符合预期：r1 应为零矩阵
        assert_equal(r1, np.ndarray(shape=(0, 0), dtype=r.dtype))

    # 定义测试函数，用于测试删除 1x1 矩阵行和列的情况
    def test_delete_1x1_row_col(self):
        # 生成测试数据
        a, q, r = self.generate('1x1')
        # 调用 qr_delete 函数删除第 1 行
        q1, r1 = qr_delete(q, r, 0, 1, 'row')
        # 断言删除后的结果符合预期：q1 应为零矩阵
        assert_equal(q1, np.ndarray(shape=(0, 0), dtype=q.dtype))
        # 断言删除后的结果符合预期：r1 的行数为零，列数与 r 相同
        assert_equal(r1, np.ndarray(shape=(0, r.shape[1]), dtype=r.dtype))

        # 生成测试数据
        a, q, r = self.generate('1x1')
        # 再次调用 qr_delete 函数删除第 1 列
        q1, r1 = qr_delete(q, r, 0, 1, 'col')
        # 断言删除后的结果：q1 应为单位矩阵
        assert_unitary(q1)
        # 断言 q1 的数据类型与 q 相同
        assert_(q1.dtype == q.dtype)
        # 断言 q1 的形状与 q 相同
        assert_(q1.shape == q.shape)
        # 断言删除后的结果符合预期：r1 应为零矩阵
        assert_equal(r1, np.ndarray(shape=(r.shape[0], 0), dtype=r.dtype))

    # 对所有完整的 QR 分解、行删除和单列删除应能处理任何非负步长。（只使用行向量和列向量操作。）
    # p 列删除需要 Fortran 排序的 Q 和 R，并根据需要进行复制。
    # 经济型 QR 分解行删除需要连续的 q。
    # 定义一个方法，用于处理非简单步幅的基础情况
    def base_non_simple_strides(self, adjust_strides, ks, p, which,
                                overwriteable):
        # 如果 which 参数为 'row'，设置 qind 和 rind 用于行索引
        if which == 'row':
            qind = (slice(p,None), slice(p,None))  # qind 表示行的切片索引
            rind = (slice(p,None), slice(None))   # rind 表示列的切片索引
        else:
            qind = (slice(None), slice(None))      # qind 表示所有行的切片索引
            rind = (slice(None), slice(None,-p))   # rind 表示除了最后 p 列之外的所有列的切片索引

        # 使用 itertools 生成器组合 type 和 ks 中的元素
        for type, k in itertools.product(['sqr', 'tall', 'fat'], ks):
            # 根据 type 生成矩阵 a, q0, r0
            a, q0, r0, = self.generate(type)
            # 调整步幅并获取 qs, rs
            qs, rs = adjust_strides((q0, r0))

            # 根据 p 的不同情况删除矩阵 a 的某些行或列
            if p == 1:
                a1 = np.delete(a, k, 0 if which == 'row' else 1)
            else:
                s = slice(k,k+p)
                if k < 0:
                    s = slice(k, k + p +
                              (a.shape[0] if which == 'row' else a.shape[1]))
                a1 = np.delete(a, s, 0 if which == 'row' else 1)

            # 复制 q0 和 r0，并按 Fortran 顺序重新排列
            q = q0.copy('F')
            r = r0.copy('F')

            # 使用 qr_delete 函数删除矩阵 q, r 的第 k 行或列，并确保未覆盖原始数据
            q1, r1 = qr_delete(qs, r, k, p, which, False)
            check_qr(q1, r1, a1, self.rtol, self.atol)

            # 使用 qr_delete 函数删除矩阵 q, r 的第 k 行或列，并确保覆盖原始数据
            q1o, r1o = qr_delete(qs, r, k, p, which, True)
            check_qr(q1o, r1o, a1, self.rtol, self.atol)

            # 如果 overwriteable 为 True，检查覆盖后的数据与预期值的接近程度
            if overwriteable:
                assert_allclose(q1o, qs[qind], rtol=self.rtol, atol=self.atol)
                assert_allclose(r1o, r[rind], rtol=self.rtol, atol=self.atol)

            # 复制 q0 和 r0，并按 Fortran 顺序重新排列
            q = q0.copy('F')
            r = r0.copy('F')

            # 使用 qr_delete 函数删除矩阵 q, r 的第 k 行或列，并确保未覆盖原始数据
            q2, r2 = qr_delete(q, rs, k, p, which, False)
            check_qr(q2, r2, a1, self.rtol, self.atol)

            # 使用 qr_delete 函数删除矩阵 q, r 的第 k 行或列，并确保覆盖原始数据
            q2o, r2o = qr_delete(q, rs, k, p, which, True)
            check_qr(q2o, r2o, a1, self.rtol, self.atol)

            # 如果 overwriteable 为 True，检查覆盖后的数据与预期值的接近程度
            if overwriteable:
                assert_allclose(q2o, q[qind], rtol=self.rtol, atol=self.atol)
                assert_allclose(r2o, rs[rind], rtol=self.rtol, atol=self.atol)

            # 复制 q0 和 r0，并按 Fortran 顺序重新排列
            q = q0.copy('F')
            r = r0.copy('F')

            # 重新调整步幅并获取 qs, rs
            qs, rs = adjust_strides((q, r))

            # 使用 qr_delete 函数删除矩阵 qs, rs 的第 k 行或列，并确保未覆盖原始数据
            q3, r3 = qr_delete(qs, rs, k, p, which, False)
            check_qr(q3, r3, a1, self.rtol, self.atol)

            # 使用 qr_delete 函数删除矩阵 qs, rs 的第 k 行或列，并确保覆盖原始数据
            q3o, r3o = qr_delete(qs, rs, k, p, which, True)
            check_qr(q3o, r3o, a1, self.rtol, self.atol)

            # 如果 overwriteable 为 True，检查覆盖后的数据与预期值的接近程度
            if overwriteable:
                assert_allclose(q2o, qs[qind], rtol=self.rtol, atol=self.atol)
                assert_allclose(r3o, rs[rind], rtol=self.rtol, atol=self.atol)

    # 测试非单位步幅下的行删除情况
    def test_non_unit_strides_1_row(self):
        self.base_non_simple_strides(make_strided, [0], 1, 'row', True)

    # 测试非单位步幅下的行删除情况（p=3）
    def test_non_unit_strides_p_row(self):
        self.base_non_simple_strides(make_strided, [0], 3, 'row', True)

    # 测试非单位步幅下的列删除情况
    def test_non_unit_strides_1_col(self):
        self.base_non_simple_strides(make_strided, [0], 1, 'col', True)
    # 调用基础非简单步幅测试方法，测试列的情况
    def test_non_unit_strides_p_col(self):
        self.base_non_simple_strides(make_strided, [0], 3, 'col', False)

    # 调用基础非简单步幅测试方法，测试负步幅的单行情况
    def test_neg_strides_1_row(self):
        self.base_non_simple_strides(negate_strides, [0], 1, 'row', False)

    # 调用基础非简单步幅测试方法，测试负步幅的多行情况
    def test_neg_strides_p_row(self):
        self.base_non_simple_strides(negate_strides, [0], 3, 'row', False)

    # 调用基础非简单步幅测试方法，测试负步幅的单列情况
    def test_neg_strides_1_col(self):
        self.base_non_simple_strides(negate_strides, [0], 1, 'col', False)

    # 调用基础非简单步幅测试方法，测试负步幅的多列情况
    def test_neg_strides_p_col(self):
        self.base_non_simple_strides(negate_strides, [0], 3, 'col', False)

    # 调用基础非简单步幅测试方法，测试非itemsize步幅的单行情况
    def test_non_itemize_strides_1_row(self):
        self.base_non_simple_strides(nonitemsize_strides, [0], 1, 'row', False)

    # 调用基础非简单步幅测试方法，测试非itemsize步幅的多行情况
    def test_non_itemize_strides_p_row(self):
        self.base_non_simple_strides(nonitemsize_strides, [0], 3, 'row', False)

    # 调用基础非简单步幅测试方法，测试非itemsize步幅的单列情况
    def test_non_itemize_strides_1_col(self):
        self.base_non_simple_strides(nonitemsize_strides, [0], 1, 'col', False)

    # 调用基础非简单步幅测试方法，测试非itemsize步幅的多列情况
    def test_non_itemize_strides_p_col(self):
        self.base_non_simple_strides(nonitemsize_strides, [0], 3, 'col', False)

    # 调用基础非简单步幅测试方法，测试非本地字节顺序的单行情况
    def test_non_native_byte_order_1_row(self):
        self.base_non_simple_strides(make_nonnative, [0], 1, 'row', False)

    # 调用基础非简单步幅测试方法，测试非本地字节顺序的多行情况
    def test_non_native_byte_order_p_row(self):
        self.base_non_simple_strides(make_nonnative, [0], 3, 'row', False)

    # 调用基础非简单步幅测试方法，测试非本地字节顺序的单列情况
    def test_non_native_byte_order_1_col(self):
        self.base_non_simple_strides(make_nonnative, [0], 1, 'col', False)

    # 调用基础非简单步幅测试方法，测试非本地字节顺序的多列情况
    def test_non_native_byte_order_p_col(self):
        self.base_non_simple_strides(make_nonnative, [0], 3, 'col', False)

    # 生成一个特定类型的测试矩阵和相关参数，然后进行负k值的删除操作
    def test_neg_k(self):
        a, q, r = self.generate('sqr')
        # 使用 itertools 的 product 方法生成所有可能的 k, p, w 组合
        for k, p, w in itertools.product([-3, -7], [1, 3], ['row', 'col']):
            # 调用 qr_delete 方法执行矩阵的删除操作
            q1, r1 = qr_delete(q, r, k, p, w, overwrite_qr=False)
            if w == 'row':
                # 如果删除行，则使用 np.delete 删除相应行
                a1 = np.delete(a, slice(k+a.shape[0], k+p+a.shape[0]), 0)
            else:
                # 如果删除列，则使用 np.delete 删除相应列
                a1 = np.delete(a, slice(k+a.shape[0], k+p+a.shape[1]), 1)
            # 检查 qr 分解后的结果是否符合预期
            check_qr(q1, r1, a1, self.rtol, self.atol)
    def base_overwrite_qr(self, which, p, test_C, test_F, mode='full'):
        # 如果 mode 是 'full'，则需要进行完整的 SQR 断言检查
        assert_sqr = True if mode == 'full' else False
        # 根据 which 参数确定 qind 和 rind 的切片范围
        if which == 'row':
            qind = (slice(p,None), slice(p,None))
            rind = (slice(p,None), slice(None))
        else:
            qind = (slice(None), slice(None))
            rind = (slice(None), slice(None,-p))
        # 生成 'sqr' 模式下的 a, q0, r0
        a, q0, r0 = self.generate('sqr', mode)
        # 根据 p 的不同值选择要删除的行或列，并生成修改后的数组 a1
        if p == 1:
            a1 = np.delete(a, 3, 0 if which == 'row' else 1)
        else:
            a1 = np.delete(a, slice(3, 3+p), 0 if which == 'row' else 1)

        # 不覆盖原始数据，复制 q0 和 r0
        q = q0.copy('F')
        r = r0.copy('F')
        # 在不修改原始数据的情况下删除第三行或列
        q1, r1 = qr_delete(q, r, 3, p, which, False)
        # 检查删除操作后的 QR 分解结果是否符合预期
        check_qr(q1, r1, a1, self.rtol, self.atol, assert_sqr)
        # 检查不修改原始数据的 QR 分解结果是否保持不变
        check_qr(q, r, a, self.rtol, self.atol, assert_sqr)

        if test_F:
            # 如果测试 F 标志为真，则复制一份 F 序列的 q0 和 r0
            q = q0.copy('F')
            r = r0.copy('F')
            # 在允许覆盖的情况下删除第三行或列
            q2, r2 = qr_delete(q, r, 3, p, which, True)
            # 检查覆盖操作后的 QR 分解结果是否符合预期
            check_qr(q2, r2, a1, self.rtol, self.atol, assert_sqr)
            # 验证覆盖后的结果与预期切片是否一致
            assert_allclose(q2, q[qind], rtol=self.rtol, atol=self.atol)
            assert_allclose(r2, r[rind], rtol=self.rtol, atol=self.atol)

        if test_C:
            # 如果测试 C 标志为真，则复制一份 C 序列的 q0 和 r0
            q = q0.copy('C')
            r = r0.copy('C')
            # 在允许覆盖的情况下删除第三行或列
            q3, r3 = qr_delete(q, r, 3, p, which, True)
            # 检查覆盖操作后的 QR 分解结果是否符合预期
            check_qr(q3, r3, a1, self.rtol, self.atol, assert_sqr)
            # 验证覆盖后的结果与预期切片是否一致
            assert_allclose(q3, q[qind], rtol=self.rtol, atol=self.atol)
            assert_allclose(r3, r[rind], rtol=self.rtol, atol=self.atol)

    def test_overwrite_qr_1_row(self):
        # 测试行覆盖模式下的基础 QR 分解操作
        self.base_overwrite_qr('row', 1, True, True)

    def test_overwrite_economic_qr_1_row(self):
        # 测试行覆盖模式下经济型 QR 分解的基础操作
        self.base_overwrite_qr('row', 1, True, True, 'economic')

    def test_overwrite_qr_1_col(self):
        # 测试列覆盖模式下的基础 QR 分解操作
        # 完整和经济型共享相同的代码路径
        self.base_overwrite_qr('col', 1, True, True)

    def test_overwrite_qr_p_row(self):
        # 测试行覆盖模式下的基础 QR 分解操作，p 大于 1
        self.base_overwrite_qr('row', 3, True, True)

    def test_overwrite_economic_qr_p_row(self):
        # 测试行覆盖模式下经济型 QR 分解的基础操作，p 大于 1
        self.base_overwrite_qr('row', 3, True, True, 'economic')

    def test_overwrite_qr_p_col(self):
        # 测试列覆盖模式下的基础 QR 分解操作，p 大于 1
        # 仅 F 序列的 q 和 r 可以被覆盖
        # 完整和经济型共享相同的代码路径
        self.base_overwrite_qr('col', 3, False, True)

    def test_bad_which(self):
        # 测试错误的 which 参数抛出 ValueError 异常
        a, q, r = self.generate('sqr')
        assert_raises(ValueError, qr_delete, q, r, 0, which='foo')
    # 定义测试方法，用于检查在不良条件下的 qr_delete 函数行为
    def test_bad_k(self):
        # 生成 'tall' 配置的数据集 a, q, r
        a, q, r = self.generate('tall')
        # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查 k 参数为 q.shape[0] 时的情况
        assert_raises(ValueError, qr_delete, q, r, q.shape[0], 1)
        # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查 k 参数为 -q.shape[0]-1 时的情况
        assert_raises(ValueError, qr_delete, q, r, -q.shape[0]-1, 1)
        # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查 r 删除列时 k 超出边界的情况
        assert_raises(ValueError, qr_delete, q, r, r.shape[0], 1, 'col')
        # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查 r 删除列时 k 超出边界的情况
        assert_raises(ValueError, qr_delete, q, r, -r.shape[0]-1, 1, 'col')

    # 定义测试方法，用于检查在不良条件下的 qr_delete 函数行为
    def test_bad_p(self):
        # 生成 'tall' 配置的数据集 a, q, r
        a, q, r = self.generate('tall')
        # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查 p 参数为负数的情况
        assert_raises(ValueError, qr_delete, q, r, 0, -1)
        # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查 p 参数为负数的情况，删除列
        assert_raises(ValueError, qr_delete, q, r, 0, -1, 'col')

        # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查 p 参数为零的情况
        assert_raises(ValueError, qr_delete, q, r, 0, 0)
        # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查 p 参数为零的情况，删除列
        assert_raises(ValueError, qr_delete, q, r, 0, 0, 'col')

        # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查删除行时不足 k+p 行的情况
        assert_raises(ValueError, qr_delete, q, r, 3, q.shape[0]-2)
        # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查删除列时不足 k+p 列的情况
        assert_raises(ValueError, qr_delete, q, r, 3, r.shape[1]-2, 'col')

    # 定义测试方法，用于检查在不良条件下的 qr_delete 函数行为
    def test_empty_q(self):
        # 生成 'tall' 配置的数据集 a, q, r
        a, q, r = self.generate('tall')
        # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查 q 参数为空数组的情况
        assert_raises(ValueError, qr_delete, np.array([]), r, 0, 1)

    # 定义测试方法，用于检查在不良条件下的 qr_delete 函数行为
    def test_empty_r(self):
        # 生成 'tall' 配置的数据集 a, q, r
        a, q, r = self.generate('tall')
        # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查 r 参数为空数组的情况
        assert_raises(ValueError, qr_delete, q, np.array([]), 0, 1)

    # 定义测试方法，用于检查在不匹配 q 和 r 的情况下的 qr_delete 函数行为
    def test_mismatched_q_and_r(self):
        # 生成 'tall' 配置的数据集 a, q, r
        a, q, r = self.generate('tall')
        # 改变 r，使其少一个行
        r = r[1:]
        # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查 q 和 r 维度不匹配的情况
        assert_raises(ValueError, qr_delete, q, r, 0, 1)

    # 定义测试方法，用于检查在不支持的数据类型下的 qr_delete 函数行为
    def test_unsupported_dtypes(self):
        # 定义不支持的数据类型列表
        dts = ['int8', 'int16', 'int32', 'int64',
               'uint8', 'uint16', 'uint32', 'uint64',
               'float16', 'longdouble', 'clongdouble',
               'bool']
        # 生成 'tall' 配置的数据集 a, q0, r0
        a, q0, r0 = self.generate('tall')
        # 遍历不支持的数据类型
        for dtype in dts:
            # 将 q0 转换为当前数据类型的实数部分
            q = q0.real.astype(dtype)
            # 通过设置错误状态处理无效值，将 r0 转换为当前数据类型的实数部分
            with np.errstate(invalid="ignore"):
                r = r0.real.astype(dtype)
            # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查删除行操作时不支持的数据类型
            assert_raises(ValueError, qr_delete, q, r0, 0, 1, 'row')
            # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查删除行操作时不支持的数据类型
            assert_raises(ValueError, qr_delete, q, r0, 0, 2, 'row')
            # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查删除列操作时不支持的数据类型
            assert_raises(ValueError, qr_delete, q, r0, 0, 1, 'col')
            # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查删除列操作时不支持的数据类型
            assert_raises(ValueError, qr_delete, q, r0, 0, 2, 'col')

            # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查删除行操作时不支持的数据类型
            assert_raises(ValueError, qr_delete, q0, r, 0, 1, 'row')
            # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查删除行操作时不支持的数据类型
            assert_raises(ValueError, qr_delete, q0, r, 0, 2, 'row')
            # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查删除列操作时不支持的数据类型
            assert_raises(ValueError, qr_delete, q0, r, 0, 1, 'col')
            # 断言：调用 qr_delete 函数，期望抛出 ValueError 异常，检查删除列操作时不支持的数据类型
            assert_raises(ValueError, qr_delete, q0, r, 0, 2, 'col')
    # 定义测试函数 test_check_finite(self)，用于检查 qr_delete 函数在特定情况下是否能正确引发 ValueError 异常

        # 使用 self.generate('tall') 生成的结果来初始化变量 a0, q0, r0
        a0, q0, r0 = self.generate('tall')

        # 复制 q0 并将其元素的数值类型改为 'F' (float)
        q = q0.copy('F')
        # 将 q 的第二行第二列的元素设置为 NaN
        q[1,1] = np.nan
        # 断言调用 qr_delete 函数，期望它引发 ValueError 异常，删除行号为 0 的行
        assert_raises(ValueError, qr_delete, q, r0, 0, 1, 'row')
        # 断言调用 qr_delete 函数，期望它引发 ValueError 异常，删除行号为 0 到 2 的行
        assert_raises(ValueError, qr_delete, q, r0, 0, 3, 'row')
        # 断言调用 qr_delete 函数，期望它引发 ValueError 异常，删除列号为 0 的列
        assert_raises(ValueError, qr_delete, q, r0, 0, 1, 'col')
        # 断言调用 qr_delete 函数，期望它引发 ValueError 异常，删除列号为 0 到 2 的列
        assert_raises(ValueError, qr_delete, q, r0, 0, 3, 'col')

        # 复制 r0 并将其元素的数值类型改为 'F' (float)
        r = r0.copy('F')
        # 将 r 的第二行第二列的元素设置为 NaN
        r[1,1] = np.nan
        # 断言调用 qr_delete 函数，期望它引发 ValueError 异常，删除行号为 0 的行
        assert_raises(ValueError, qr_delete, q0, r, 0, 1, 'row')
        # 断言调用 qr_delete 函数，期望它引发 ValueError 异常，删除行号为 0 到 2 的行
        assert_raises(ValueError, qr_delete, q0, r, 0, 3, 'row')
        # 断言调用 qr_delete 函数，期望它引发 ValueError 异常，删除列号为 0 的列
        assert_raises(ValueError, qr_delete, q0, r, 0, 1, 'col')
        # 断言调用 qr_delete 函数，期望它引发 ValueError 异常，删除列号为 0 到 2 的列
        assert_raises(ValueError, qr_delete, q0, r, 0, 3, 'col')

    # 定义测试函数 test_qr_scalar(self)，用于检查 qr_delete 函数在处理标量参数时是否能正确引发 ValueError 异常

        # 使用 self.generate('1x1') 生成的结果来初始化变量 a, q, r
        a, q, r = self.generate('1x1')
        # 断言调用 qr_delete 函数，期望它引发 ValueError 异常，其中 q 是一个标量，删除行号为 0 的行
        assert_raises(ValueError, qr_delete, q[0, 0], r, 0, 1, 'row')
        # 断言调用 qr_delete 函数，期望它引发 ValueError 异常，其中 r 是一个标量，删除行号为 0 的行
        assert_raises(ValueError, qr_delete, q, r[0, 0], 0, 1, 'row')
        # 断言调用 qr_delete 函数，期望它引发 ValueError 异常，其中 q 是一个标量，删除列号为 0 的列
        assert_raises(ValueError, qr_delete, q[0, 0], r, 0, 1, 'col')
        # 断言调用 qr_delete 函数，期望它引发 ValueError 异常，其中 r 是一个标量，删除列号为 0 的列
        assert_raises(ValueError, qr_delete, q, r[0, 0], 0, 1, 'col')
class TestQRdelete_f(BaseQRdelete):
    # 定义测试类 TestQRdelete_f，继承自 BaseQRdelete
    dtype = np.dtype('f')  # 设置 dtype 为单精度浮点数类型

class TestQRdelete_F(BaseQRdelete):
    # 定义测试类 TestQRdelete_F，继承自 BaseQRdelete
    dtype = np.dtype('F')  # 设置 dtype 为复数的单精度浮点数类型

class TestQRdelete_d(BaseQRdelete):
    # 定义测试类 TestQRdelete_d，继承自 BaseQRdelete
    dtype = np.dtype('d')  # 设置 dtype 为双精度浮点数类型

class TestQRdelete_D(BaseQRdelete):
    # 定义测试类 TestQRdelete_D，继承自 BaseQRdelete
    dtype = np.dtype('D')  # 设置 dtype 为复数的双精度浮点数类型

class BaseQRinsert(BaseQRdeltas):
    # 定义基类 BaseQRinsert，继承自 BaseQRdeltas
    def generate(self, type, mode='full', which='row', p=1):
        # 生成数据的方法
        a, q, r = super().generate(type, mode)  # 调用父类方法生成数据
        assert_(p > 0)  # 断言 p 大于 0

        # 调用 super 设置了随机种子...

        if which == 'row':
            # 如果 which 为 'row'
            if p == 1:
                u = np.random.random(a.shape[1])  # 生成与 a 列数相同的随机数数组
            else:
                u = np.random.random((p, a.shape[1]))  # 生成 p 行 a 列的随机数数组
        elif which == 'col':
            # 如果 which 为 'col'
            if p == 1:
                u = np.random.random(a.shape[0])  # 生成与 a 行数相同的随机数数组
            else:
                u = np.random.random((a.shape[0], p))  # 生成 a 行 p 列的随机数数组
        else:
            ValueError('which should be either "row" or "col"')  # 抛出数值错误，which 应为 'row' 或 'col'

        if np.iscomplexobj(self.dtype.type(1)):
            # 如果 dtype 是复数类型
            b = np.random.random(u.shape)  # 生成与 u 形状相同的随机数数组
            u = u + 1j * b  # 使用虚数单位乘以 b 添加虚数部分

        u = u.astype(self.dtype)  # 将 u 数组类型转换为 self.dtype 类型
        return a, q, r, u  # 返回生成的数据

    def test_sqr_1_row(self):
        # 测试在方阵情况下插入一行数据
        a, q, r, u = self.generate('sqr', which='row')
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row)  # 在 Q、R 中插入 u 到第 row 行
            a1 = np.insert(a, row, u, 0)  # 在 a 中第 row 行插入 u
            check_qr(q1, r1, a1, self.rtol, self.atol)  # 检查插入后的 Q、R、a 是否符合容差要求

    def test_sqr_p_row(self):
        # 测试在方阵情况下插入多行数据
        # sqr + rows --> fat always
        a, q, r, u = self.generate('sqr', which='row', p=3)  # 生成 3 行数据
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row)  # 在 Q、R 中插入 u 到第 row 行
            a1 = np.insert(a, np.full(3, row, np.intp), u, 0)  # 在 a 中指定 3 行位置插入 u
            check_qr(q1, r1, a1, self.rtol, self.atol)  # 检查插入后的 Q、R、a 是否符合容差要求

    def test_sqr_1_col(self):
        # 测试在方阵情况下插入一列数据
        a, q, r, u = self.generate('sqr', which='col')
        for col in range(r.shape[1] + 1):
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)  # 在 Q、R 中插入 u 到第 col 列
            a1 = np.insert(a, col, u, 1)  # 在 a 中第 col 列插入 u
            check_qr(q1, r1, a1, self.rtol, self.atol)  # 检查插入后的 Q、R、a 是否符合容差要求

    def test_sqr_p_col(self):
        # 测试在方阵情况下插入多列数据
        # sqr + cols --> fat always
        a, q, r, u = self.generate('sqr', which='col', p=3)  # 生成 3 列数据
        for col in range(r.shape[1] + 1):
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)  # 在 Q、R 中插入 u 到第 col 列
            a1 = np.insert(a, np.full(3, col, np.intp), u, 1)  # 在 a 中指定 3 列位置插入 u
            check_qr(q1, r1, a1, self.rtol, self.atol)  # 检查插入后的 Q、R、a 是否符合容差要求

    def test_tall_1_row(self):
        # 测试在高瘦矩阵情况下插入一行数据
        a, q, r, u = self.generate('tall', which='row')
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row)  # 在 Q、R 中插入 u 到第 row 行
            a1 = np.insert(a, row, u, 0)  # 在 a 中第 row 行插入 u
            check_qr(q1, r1, a1, self.rtol, self.atol)  # 检查插入后的 Q、R、a 是否符合容差要求

    def test_tall_p_row(self):
        # 测试在高瘦矩阵情况下插入多行数据
        # tall + rows --> tall always
        a, q, r, u = self.generate('tall', which='row', p=3)  # 生成 3 行数据
        for row in range(r.shape[0] + 1):
            q1, r1 = qr_insert(q, r, u, row)  # 在 Q、R 中插入 u 到第 row 行
            a1 = np.insert(a, np.full(3, row, np.intp), u, 0)  # 在 a 中指定 3 行位置插入 u
            check_qr(q1, r1, a1, self.rtol, self.atol)  # 检查插入后的 Q、R、a 是否符合容差要求
    # 测试在 "tall" 类型矩阵中插入列的情况
    def test_tall_1_col(self):
        # 生成 "tall" 类型矩阵及相关变量
        a, q, r, u = self.generate('tall', which='col')
        # 遍历矩阵的列数加一次
        for col in range(r.shape[1] + 1):
            # 插入列并进行 QR 分解
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            # 在原始矩阵中插入列并生成新矩阵
            a1 = np.insert(a, col, u, 1)
            # 检查插入列后的 QR 分解结果是否符合预期
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 插入列到 "tall" 类型矩阵有三种情况需要测试
    # tall + pcol --> tall
    # tall + pcol --> sqr
    # tall + pcol --> fat
    def base_tall_p_col_xxx(self, p):
        # 生成 "tall" 类型矩阵及相关变量，其中 p 表示插入的列数
        a, q, r, u = self.generate('tall', which='col', p=p)
        # 遍历矩阵的列数加一次
        for col in range(r.shape[1] + 1):
            # 插入列并进行 QR 分解
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            # 在原始矩阵中插入列并生成新矩阵，插入位置由 np.full(p, col, np.intp) 指定
            a1 = np.insert(a, np.full(p, col, np.intp), u, 1)
            # 检查插入列后的 QR 分解结果是否符合预期
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试在 "tall" 类型矩阵中插入 p 列后仍保持 "tall" 类型
    def test_tall_p_col_tall(self):
        # 12x7 + 12x3 = 12x10 --> 保持 "tall"
        self.base_tall_p_col_xxx(3)

    # 测试在 "tall" 类型矩阵中插入 p 列后变为 "sqr" 类型
    def test_tall_p_col_sqr(self):
        # 12x7 + 12x5 = 12x12 --> 变为 "sqr"
        self.base_tall_p_col_xxx(5)

    # 测试在 "tall" 类型矩阵中插入 p 列后变为 "fat" 类型
    def test_tall_p_col_fat(self):
        # 12x7 + 12x7 = 12x14 --> 变为 "fat"
        self.base_tall_p_col_xxx(7)

    # 测试在 "fat" 类型矩阵中插入行的情况
    def test_fat_1_row(self):
        # 生成 "fat" 类型矩阵及相关变量
        a, q, r, u = self.generate('fat', which='row')
        # 遍历矩阵的行数加一次
        for row in range(r.shape[0] + 1):
            # 插入行并进行 QR 分解
            q1, r1 = qr_insert(q, r, u, row)
            # 在原始矩阵中插入行并生成新矩阵
            a1 = np.insert(a, row, u, 0)
            # 检查插入行后的 QR 分解结果是否符合预期
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 插入行到 "fat" 类型矩阵有三种情况需要测试
    # fat + prow --> fat
    # fat + prow --> sqr
    # fat + prow --> tall
    def base_fat_p_row_xxx(self, p):
        # 生成 "fat" 类型矩阵及相关变量，其中 p 表示插入的行数
        a, q, r, u = self.generate('fat', which='row', p=p)
        # 遍历矩阵的行数加一次
        for row in range(r.shape[0] + 1):
            # 插入行并进行 QR 分解
            q1, r1 = qr_insert(q, r, u, row)
            # 在原始矩阵中插入行并生成新矩阵，插入位置由 np.full(p, row, np.intp) 指定
            a1 = np.insert(a, np.full(p, row, np.intp), u, 0)
            # 检查插入行后的 QR 分解结果是否符合预期
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试在 "fat" 类型矩阵中插入 p 行后仍保持 "fat" 类型
    def test_fat_p_row_fat(self):
        # 7x12 + 3x12 = 10x12 --> 保持 "fat"
        self.base_fat_p_row_xxx(3)

    # 测试在 "fat" 类型矩阵中插入 p 行后变为 "sqr" 类型
    def test_fat_p_row_sqr(self):
        # 7x12 + 5x12 = 12x12 --> 变为 "sqr"
        self.base_fat_p_row_xxx(5)

    # 测试在 "fat" 类型矩阵中插入 p 行后变为 "tall" 类型
    def test_fat_p_row_tall(self):
        # 7x12 + 7x12 = 14x12 --> 变为 "tall"
        self.base_fat_p_row_xxx(7)

    # 测试在 "fat" 类型矩阵中插入列的情况
    def test_fat_1_col(self):
        # 生成 "fat" 类型矩阵及相关变量
        a, q, r, u = self.generate('fat', which='col')
        # 遍历矩阵的列数加一次
        for col in range(r.shape[1] + 1):
            # 插入列并进行 QR 分解
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            # 在原始矩阵中插入列并生成新矩阵
            a1 = np.insert(a, col, u, 1)
            # 检查插入列后的 QR 分解结果是否符合预期
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试在 "fat" 类型矩阵中插入 p 列的情况
    def test_fat_p_col(self):
        # 生成 "fat" 类型矩阵及相关变量，其中 p 表示插入的列数
        a, q, r, u = self.generate('fat', which='col', p=3)
        # 遍历矩阵的列数加一次
        for col in range(r.shape[1] + 1):
            # 插入列并进行 QR 分解
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            # 在原始矩阵中插入列并生成新矩阵，插入位置由 np.full(3, col, np.intp) 指定
            a1 = np.insert(a, np.full(3, col, np.intp), u, 1)
            # 检查插入列后的 QR 分解结果是否符合预期
            check_qr(q1, r1, a1, self.rtol, self.atol)
    # 测试插入行操作对经济型矩阵的影响
    def test_economic_1_row(self):
        # 生成 'tall'、'economic'、'row' 类型的矩阵及其分解结果
        a, q, r, u = self.generate('tall', 'economic', 'row')
        # 对矩阵每一行执行插入操作
        for row in range(r.shape[0] + 1):
            # 在经济型 QR 分解中插入新行
            q1, r1 = qr_insert(q, r, u, row, overwrite_qru=False)
            # 在原始矩阵中插入新行
            a1 = np.insert(a, row, u, 0)
            # 检查插入后的 QR 分解结果是否符合预期
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    # 测试插入行操作对经济型矩阵（带批量插入）的影响
    def test_economic_p_row(self):
        # 生成 'tall'、'economic'、'row' 类型的矩阵及其分解结果，批量插入 3 行
        a, q, r, u = self.generate('tall', 'economic', 'row', 3)
        # 对矩阵每一行执行插入操作
        for row in range(r.shape[0] + 1):
            # 在经济型 QR 分解中插入新行
            q1, r1 = qr_insert(q, r, u, row, overwrite_qru=False)
            # 在原始矩阵中批量插入新行
            a1 = np.insert(a, np.full(3, row, np.intp), u, 0)
            # 检查插入后的 QR 分解结果是否符合预期
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    # 测试插入列操作对经济型矩阵的影响
    def test_economic_1_col(self):
        # 生成 'tall'、'economic'、列 类型的矩阵及其分解结果
        a, q, r, u = self.generate('tall', 'economic', which='col')
        # 对矩阵每一列执行插入操作
        for col in range(r.shape[1] + 1):
            # 在经济型 QR 分解中插入新列
            q1, r1 = qr_insert(q, r, u.copy(), col, 'col', overwrite_qru=False)
            # 在原始矩阵中插入新列
            a1 = np.insert(a, col, u, 1)
            # 检查插入后的 QR 分解结果是否符合预期
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    # 测试插入列操作对经济型矩阵的不良更新情况
    def test_economic_1_col_bad_update(self):
        # 当要添加的列位于 Q 的列向量生成的空间内时，插入操作无意义，会抛出 LinAlgError 异常
        q = np.eye(5, 3, dtype=self.dtype)
        r = np.eye(3, dtype=self.dtype)
        u = np.array([1, 0, 0, 0, 0], self.dtype)
        assert_raises(linalg.LinAlgError, qr_insert, q, r, u, 0, 'col')

    # 测试插入列操作对不同类型的经济型矩阵的影响（eco、sqr、fat）
    def base_economic_p_col_xxx(self, p):
        # 生成 'tall'、'economic'、列 类型的矩阵及其分解结果，批量插入 p 列
        a, q, r, u = self.generate('tall', 'economic', which='col', p=p)
        # 对矩阵每一列执行插入操作
        for col in range(r.shape[1] + 1):
            # 在经济型 QR 分解中插入新列
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            # 在原始矩阵中批量插入新列
            a1 = np.insert(a, np.full(p, col, np.intp), u, 1)
            # 检查插入后的 QR 分解结果是否符合预期
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    # 测试插入列操作对经济型矩阵的影响，保持矩阵类型为经济型
    def test_economic_p_col_eco(self):
        # 12x7 + 12x3 = 12x10 --> 保持经济型
        self.base_economic_p_col_xxx(3)

    # 测试插入列操作对经济型矩阵的影响，使矩阵类型变为方阵
    def test_economic_p_col_sqr(self):
        # 12x7 + 12x5 = 12x12 --> 变为方阵
        self.base_economic_p_col_xxx(5)

    # 测试插入列操作对经济型矩阵的影响，使矩阵类型变为宽矩阵
    def test_economic_p_col_fat(self):
        # 12x7 + 12x7 = 12x14 --> 变为宽矩阵
        self.base_economic_p_col_xxx(7)

    # 测试插入行操作对 Mx1 矩阵的影响
    def test_Mx1_1_row(self):
        # 生成 'Mx1'、行 类型的矩阵及其分解结果
        a, q, r, u = self.generate('Mx1', which='row')
        # 对矩阵每一行执行插入操作
        for row in range(r.shape[0] + 1):
            # 在 QR 分解中插入新行
            q1, r1 = qr_insert(q, r, u, row)
            # 在原始矩阵中插入新行
            a1 = np.insert(a, row, u, 0)
            # 检查插入后的 QR 分解结果是否符合预期
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试插入行操作对 Mx1 矩阵（带批量插入）的影响
    def test_Mx1_p_row(self):
        # 生成 'Mx1'、行 类型的矩阵及其分解结果，批量插入 3 行
        a, q, r, u = self.generate('Mx1', which='row', p=3)
        # 对矩阵每一行执行插入操作
        for row in range(r.shape[0] + 1):
            # 在 QR 分解中插入新行
            q1, r1 = qr_insert(q, r, u, row)
            # 在原始矩阵中批量插入新行
            a1 = np.insert(a, np.full(3, row, np.intp), u, 0)
            # 检查插入后的 QR 分解结果是否符合预期
            check_qr(q1, r1, a1, self.rtol, self.atol)
    # 测试生成 'Mx1' 类型的数据，按列插入新向量 u 到各列，并验证结果
    def test_Mx1_1_col(self):
        a, q, r, u = self.generate('Mx1', which='col')
        # 对每一列（包括添加列）进行插入操作
        for col in range(r.shape[1] + 1):
            # 在 (q, r) QR 分解中按列插入新向量 u
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            # 在矩阵 a 中按列插入新向量 u
            a1 = np.insert(a, col, u, 1)
            # 检查插入后的 QR 分解结果是否与预期一致
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试生成 'Mx1' 类型的数据，按列插入多个新向量 u 到各列，并验证结果
    def test_Mx1_p_col(self):
        a, q, r, u = self.generate('Mx1', which='col', p=3)
        # 对每一列（包括添加列）进行插入操作
        for col in range(r.shape[1] + 1):
            # 在 (q, r) QR 分解中按列插入多个新向量 u
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            # 在矩阵 a 中按列插入多个新向量 u
            a1 = np.insert(a, np.full(3, col, np.intp), u, 1)
            # 检查插入后的 QR 分解结果是否与预期一致
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试生成 'Mx1' 类型的数据，经济模式下按行插入新向量 u 到各行，并验证结果
    def test_Mx1_economic_1_row(self):
        a, q, r, u = self.generate('Mx1', 'economic', 'row')
        # 对每一行（包括添加行）进行插入操作
        for row in range(r.shape[0] + 1):
            # 在 (q, r) QR 分解中按行插入新向量 u
            q1, r1 = qr_insert(q, r, u, row)
            # 在矩阵 a 中按行插入新向量 u
            a1 = np.insert(a, row, u, 0)
            # 检查插入后的 QR 分解结果是否与预期一致，经济模式下不进行 Q 值覆写
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    # 测试生成 'Mx1' 类型的数据，经济模式下按行插入多个新向量 u 到各行，并验证结果
    def test_Mx1_economic_p_row(self):
        a, q, r, u = self.generate('Mx1', 'economic', 'row', 3)
        # 对每一行（包括添加行）进行插入操作
        for row in range(r.shape[0] + 1):
            # 在 (q, r) QR 分解中按行插入多个新向量 u
            q1, r1 = qr_insert(q, r, u, row)
            # 在矩阵 a 中按行插入多个新向量 u
            a1 = np.insert(a, np.full(3, row, np.intp), u, 0)
            # 检查插入后的 QR 分解结果是否与预期一致，经济模式下不进行 Q 值覆写
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    # 测试生成 'Mx1' 类型的数据，经济模式下按列插入新向量 u 到各列，并验证结果
    def test_Mx1_economic_1_col(self):
        a, q, r, u = self.generate('Mx1', 'economic', 'col')
        # 对每一列（包括添加列）进行插入操作
        for col in range(r.shape[1] + 1):
            # 在 (q, r) QR 分解中按列插入新向量 u
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            # 在矩阵 a 中按列插入新向量 u
            a1 = np.insert(a, col, u, 1)
            # 检查插入后的 QR 分解结果是否与预期一致，经济模式下不进行 Q 值覆写
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    # 测试生成 'Mx1' 类型的数据，经济模式下按列插入多个新向量 u 到各列，并验证结果
    def test_Mx1_economic_p_col(self):
        a, q, r, u = self.generate('Mx1', 'economic', 'col', 3)
        # 对每一列（包括添加列）进行插入操作
        for col in range(r.shape[1] + 1):
            # 在 (q, r) QR 分解中按列插入多个新向量 u
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            # 在矩阵 a 中按列插入多个新向量 u
            a1 = np.insert(a, np.full(3, col, np.intp), u, 1)
            # 检查插入后的 QR 分解结果是否与预期一致，经济模式下不进行 Q 值覆写
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    # 测试生成 '1xN' 类型的数据，按行插入新向量 u 到各行，并验证结果
    def test_1xN_1_row(self):
        a, q, r, u = self.generate('1xN', which='row')
        # 对每一行（包括添加行）进行插入操作
        for row in range(r.shape[0] + 1):
            # 在 (q, r) QR 分解中按行插入新向量 u
            q1, r1 = qr_insert(q, r, u, row)
            # 在矩阵 a 中按行插入新向量 u
            a1 = np.insert(a, row, u, 0)
            # 检查插入后的 QR 分解结果是否与预期一致
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试生成 '1xN' 类型的数据，按行插入多个新向量 u 到各行，并验证结果
    def test_1xN_p_row(self):
        a, q, r, u = self.generate('1xN', which='row', p=3)
        # 对每一行（包括添加行）进行插入操作
        for row in range(r.shape[0] + 1):
            # 在 (q, r) QR 分解中按行插入多个新向量 u
            q1, r1 = qr_insert(q, r, u, row)
            # 在矩阵 a 中按行插入多个新向量 u
            a1 = np.insert(a, np.full(3, row, np.intp), u, 0)
            # 检查插入后的 QR 分解结果是否与预期一致
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 测试生成 '1xN' 类型的数据，按列插入新向量 u 到各列，并验证结果
    def test_1xN_1_col(self):
        a, q, r, u = self.generate('1xN', which='col')
        # 对每一列（包括添加列）进行插入操作
        for col in range(r.shape[1] + 1):
            # 在 (q, r) QR 分解
    # 定义一个测试方法，用于测试在特定条件下的 QR 插入操作（1xN 列操作）
    def test_1xN_p_col(self):
        # 生成测试数据，包括矩阵 a、Q、R 和向量 u，其中 p=3
        a, q, r, u = self.generate('1xN', which='col', p=3)
        
        # 遍历 R 矩阵的每一列，包括额外的一列
        for col in range(r.shape[1] + 1):
            # 执行 QR 插入操作，得到新的 Q1 和 R1
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            
            # 在矩阵 a 中插入向量 u，位置由 np.full(3, col, np.intp) 确定，插入在列上
            a1 = np.insert(a, np.full(3, col, np.intp), u, 1)
            
            # 检查插入操作后的 QR 分解结果是否符合预期
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 定义一个测试方法，用于测试在特定条件下的 QR 插入操作（1x1 行操作）
    def test_1x1_1_row(self):
        # 生成测试数据，包括矩阵 a、Q、R 和向量 u，只有一行
        a, q, r, u = self.generate('1x1', which='row')
        
        # 遍历 R 矩阵的每一行，包括额外的一行
        for row in range(r.shape[0] + 1):
            # 执行 QR 插入操作，得到新的 Q1 和 R1
            q1, r1 = qr_insert(q, r, u, row)
            
            # 在矩阵 a 中插入向量 u，插入在行上，位置由 row 确定
            a1 = np.insert(a, row, u, 0)
            
            # 检查插入操作后的 QR 分解结果是否符合预期
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 定义一个测试方法，用于测试在特定条件下的 QR 插入操作（1x1 行操作，p=3）
    def test_1x1_p_row(self):
        # 生成测试数据，包括矩阵 a、Q、R 和向量 u，只有一行，p=3
        a, q, r, u = self.generate('1x1', which='row', p=3)
        
        # 遍历 R 矩阵的每一行，包括额外的一行
        for row in range(r.shape[0] + 1):
            # 执行 QR 插入操作，得到新的 Q1 和 R1
            q1, r1 = qr_insert(q, r, u, row)
            
            # 在矩阵 a 中插入向量 u，插入在行上，位置由 np.full(3, row, np.intp) 确定
            a1 = np.insert(a, np.full(3, row, np.intp), u, 0)
            
            # 检查插入操作后的 QR 分解结果是否符合预期
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 定义一个测试方法，用于测试在特定条件下的 QR 插入操作（1x1 列操作）
    def test_1x1_1_col(self):
        # 生成测试数据，包括矩阵 a、Q、R 和向量 u，只有一列
        a, q, r, u = self.generate('1x1', which='col')
        
        # 遍历 R 矩阵的每一列，包括额外的一列
        for col in range(r.shape[1] + 1):
            # 执行 QR 插入操作，得到新的 Q1 和 R1
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            
            # 在矩阵 a 中插入向量 u，插入在列上，位置由 col 确定
            a1 = np.insert(a, col, u, 1)
            
            # 检查插入操作后的 QR 分解结果是否符合预期
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 定义一个测试方法，用于测试在特定条件下的 QR 插入操作（1x1 列操作，p=3）
    def test_1x1_p_col(self):
        # 生成测试数据，包括矩阵 a、Q、R 和向量 u，只有一列，p=3
        a, q, r, u = self.generate('1x1', which='col', p=3)
        
        # 遍历 R 矩阵的每一列，包括额外的一列
        for col in range(r.shape[1] + 1):
            # 执行 QR 插入操作，得到新的 Q1 和 R1
            q1, r1 = qr_insert(q, r, u, col, 'col', overwrite_qru=False)
            
            # 在矩阵 a 中插入向量 u，插入在列上，位置由 np.full(3, col, np.intp) 确定
            a1 = np.insert(a, np.full(3, col, np.intp), u, 1)
            
            # 检查插入操作后的 QR 分解结果是否符合预期
            check_qr(q1, r1, a1, self.rtol, self.atol)

    # 定义一个测试方法，用于测试在特定条件下的 QR 插入操作（1x1 标量操作）
    def test_1x1_1_scalar(self):
        # 生成测试数据，包括矩阵 a、Q、R 和向量 u，只有一行
        a, q, r, u = self.generate('1x1', which='row')
        
        # 检验在插入标量情况下是否引发 ValueError 异常
        assert_raises(ValueError, qr_insert, q[0, 0], r, u, 0, 'row')
        assert_raises(ValueError, qr_insert, q, r[0, 0], u, 0, 'row')
        assert_raises(ValueError, qr_insert, q, r, u[0], 0, 'row')

        assert_raises(ValueError, qr_insert, q[0, 0], r, u, 0, 'col')
        assert_raises(ValueError, qr_insert, q, r[0, 0], u, 0, 'col')
        assert_raises(ValueError, qr_insert, q, r, u[0], 0, 'col')
    def base_non_simple_strides(self, adjust_strides, k, p, which):
        # 对于三种类型的矩阵（'sqr', 'tall', 'fat'），生成对应类型的矩阵和相关变量
        for type in ['sqr', 'tall', 'fat']:
            a, q0, r0, u0 = self.generate(type, which=which, p=p)
            # 根据给定的调整函数adjust_strides，调整步幅q0, r0, u0，并返回调整后的结果
            qs, rs, us = adjust_strides((q0, r0, u0))

            if p == 1:
                # 将u0插入矩阵a的第k行（如果which为'row'）或第k列（如果which为'col'）
                ai = np.insert(a, k, u0, 0 if which == 'row' else 1)
            else:
                # 将u0插入矩阵a的第k个位置（根据p次重复），在'row'模式下插入行，在'col'模式下插入列
                ai = np.insert(a, np.full(p, k, np.intp),
                        u0 if which == 'row' else u0,
                        0 if which == 'row' else 1)

            # 针对每个变量（q, r, u），尝试不同步幅和overwrite=False的情况，
            # 然后尝试overwrite=True的情况。由于只有按列顺序排列的Q可以在添加列时进行覆盖，因此不检查是否可以覆盖。
            
            # 复制q0, r0, u0为F顺序的副本
            q = q0.copy('F')
            r = r0.copy('F')
            u = u0.copy('F')
            # 在(k, which)位置插入qs
            q1, r1 = qr_insert(qs, r, u, k, which, overwrite_qru=False)
            # 检查插入后的QR分解结果
            check_qr(q1, r1, ai, self.rtol, self.atol)
            # 在(k, which)位置插入qs，并使用overwrite_qru=True
            q1o, r1o = qr_insert(qs, r, u, k, which, overwrite_qru=True)
            # 检查插入后的QR分解结果
            check_qr(q1o, r1o, ai, self.rtol, self.atol)

            q = q0.copy('F')
            r = r0.copy('F')
            u = u0.copy('F')
            # 在(k, which)位置插入rs
            q2, r2 = qr_insert(q, rs, u, k, which, overwrite_qru=False)
            # 检查插入后的QR分解结果
            check_qr(q2, r2, ai, self.rtol, self.atol)
            # 在(k, which)位置插入rs，并使用overwrite_qru=True
            q2o, r2o = qr_insert(q, rs, u, k, which, overwrite_qru=True)
            # 检查插入后的QR分解结果
            check_qr(q2o, r2o, ai, self.rtol, self.atol)

            q = q0.copy('F')
            r = r0.copy('F')
            u = u0.copy('F')
            # 在(k, which)位置插入us
            q3, r3 = qr_insert(q, r, us, k, which, overwrite_qru=False)
            # 检查插入后的QR分解结果
            check_qr(q3, r3, ai, self.rtol, self.atol)
            # 在(k, which)位置插入us，并使用overwrite_qru=True
            q3o, r3o = qr_insert(q, r, us, k, which, overwrite_qru=True)
            # 检查插入后的QR分解结果
            check_qr(q3o, r3o, ai, self.rtol, self.atol)

            q = q0.copy('F')
            r = r0.copy('F')
            u = u0.copy('F')
            # 重新调整步幅并返回qs, rs, us
            qs, rs, us = adjust_strides((q, r, u))
            # 在(k, which)位置插入qs
            q5, r5 = qr_insert(qs, rs, us, k, which, overwrite_qru=False)
            # 检查插入后的QR分解结果
            q5o, r5o = qr_insert(qs, rs, us, k, which, overwrite_qru=True)
            # 检查插入后的QR分解结果
            check_qr(q5o, r5o, ai, self.rtol, self.atol)

    def test_non_unit_strides_1_row(self):
        # 调用base_non_simple_strides方法，使用make_strided函数，p=1，which='row'进行测试
        self.base_non_simple_strides(make_strided, 0, 1, 'row')

    def test_non_unit_strides_p_row(self):
        # 调用base_non_simple_strides方法，使用make_strided函数，p=3，which='row'进行测试
        self.base_non_simple_strides(make_strided, 0, 3, 'row')

    def test_non_unit_strides_1_col(self):
        # 调用base_non_simple_strides方法，使用make_strided函数，p=1，which='col'进行测试
        self.base_non_simple_strides(make_strided, 0, 1, 'col')

    def test_non_unit_strides_p_col(self):
        # 调用base_non_simple_strides方法，使用make_strided函数，p=3，which='col'进行测试
        self.base_non_simple_strides(make_strided, 0, 3, 'col')

    def test_neg_strides_1_row(self):
        # 调用base_non_simple_strides方法，使用negate_strides函数，p=1，which='row'进行测试
        self.base_non_simple_strides(negate_strides, 0, 1, 'row')

    def test_neg_strides_p_row(self):
        # 调用base_non_simple_strides方法，使用negate_strides函数，p=3，which='row'进行测试
        self.base_non_simple_strides(negate_strides, 0, 3, 'row')
    def test_neg_strides_1_col(self):
        # 调用基础的非简单步幅函数，测试负步幅在列方向的情况
        self.base_non_simple_strides(negate_strides, 0, 1, 'col')

    def test_neg_strides_p_col(self):
        # 调用基础的非简单步幅函数，测试负步幅在列方向的情况（带有更大步幅）
        self.base_non_simple_strides(negate_strides, 0, 3, 'col')

    def test_non_itemsize_strides_1_row(self):
        # 调用基础的非简单步幅函数，测试非项目大小步幅在行方向的情况
        self.base_non_simple_strides(nonitemsize_strides, 0, 1, 'row')

    def test_non_itemsize_strides_p_row(self):
        # 调用基础的非简单步幅函数，测试非项目大小步幅在行方向的情况（带有更大步幅）
        self.base_non_simple_strides(nonitemsize_strides, 0, 3, 'row')

    def test_non_itemsize_strides_1_col(self):
        # 调用基础的非简单步幅函数，测试非项目大小步幅在列方向的情况
        self.base_non_simple_strides(nonitemsize_strides, 0, 1, 'col')

    def test_non_itemsize_strides_p_col(self):
        # 调用基础的非简单步幅函数，测试非项目大小步幅在列方向的情况（带有更大步幅）
        self.base_non_simple_strides(nonitemsize_strides, 0, 3, 'col')

    def test_non_native_byte_order_1_row(self):
        # 调用制造非本地字节顺序的函数，测试在行方向插入的情况
        self.base_non_simple_strides(make_nonnative, 0, 1, 'row')

    def test_non_native_byte_order_p_row(self):
        # 调用制造非本地字节顺序的函数，测试在行方向插入的情况（带有更大步幅）
        self.base_non_simple_strides(make_nonnative, 0, 3, 'row')

    def test_non_native_byte_order_1_col(self):
        # 调用制造非本地字节顺序的函数，测试在列方向插入的情况
        self.base_non_simple_strides(make_nonnative, 0, 1, 'col')

    def test_non_native_byte_order_p_col(self):
        # 调用制造非本地字节顺序的函数，测试在列方向插入的情况（带有更大步幅）
        self.base_non_simple_strides(make_nonnative, 0, 3, 'col')

    def test_overwrite_qu_rank_1(self):
        # 当插入行时，Q 和 R 的大小都会改变，因此只有列插入可以覆盖 Q。
        # 只有使用 C 顺序的复杂列插入才能覆盖 U。任何连续的 Q 在插入一个列时都会被覆盖。
        
        # 生成 'sqr' 类型的数据，其中 'col' 为主，并插入 1 列
        a, q0, r, u, = self.generate('sqr', which='col', p=1)
        # 对 Q 进行 C 顺序复制
        q = q0.copy('C')
        # 复制 U
        u0 = u.copy()
        
        # 不覆盖情况
        # 在 Q 的开头插入列，返回新的 Q 和 R
        q1, r1 = qr_insert(q, r, u, 0, 'col', overwrite_qru=False)
        # 在 A 的开头插入 U0，返回新的 A1
        a1 = np.insert(a, 0, u0, 1)
        # 检查 QR 分解的结果是否符合指定的相对误差和绝对误差
        check_qr(q1, r1, a1, self.rtol, self.atol)
        # 检查 Q, R, A 是否符合指定的相对误差和绝对误差
        check_qr(q, r, a, self.rtol, self.atol)

        # 尝试覆盖情况
        # 在 Q 的开头插入列，允许覆盖
        q2, r2 = qr_insert(q, r, u, 0, 'col', overwrite_qru=True)
        # 检查 QR 分解的结果是否符合指定的相对误差和绝对误差
        check_qr(q2, r2, a1, self.rtol, self.atol)
        # 验证覆盖后 Q 的正确性
        assert_allclose(q2, q, rtol=self.rtol, atol=self.atol)
        # 验证 U 是否正确共轭
        assert_allclose(u, u0.conj(), self.rtol, self.atol)

        # 现在尝试使用 Fortran 顺序的 Q
        qF = q0.copy('F')
        # 复制 U0
        u1 = u0.copy()
        # 在 QF 的开头插入列，不覆盖
        q3, r3 = qr_insert(qF, r, u1, 0, 'col', overwrite_qru=False)
        # 检查 QR 分解的结果是否符合指定的相对误差和绝对误差
        check_qr(q3, r3, a1, self.rtol, self.atol)
        # 检查 QF, R, A 是否符合指定的相对误差和绝对误差
        check_qr(qF, r, a, self.rtol, self.atol)

        # 尝试覆盖情况
        # 在 QF 的开头插入列，允许覆盖
        q4, r4 = qr_insert(qF, r, u1, 0, 'col', overwrite_qru=True)
        # 检查 QR 分解的结果是否符合指定的相对误差和绝对误差
        check_qr(q4, r4, a1, self.rtol, self.atol)
        # 验证覆盖后 QF 的正确性
        assert_allclose(q4, qF, rtol=self.rtol, atol=self.atol)
    def test_overwrite_qu_rank_p(self):
        # 当插入行时，Q 和 R 的大小都会改变，因此只有列插入可能会覆盖 Q。实际上，只有按列顺序的 Q 会被 rank p 更新覆盖。

        # 生成测试数据：a 是生成的矩阵，q0 是 Q 矩阵，r 是 R 矩阵，u 是要插入的列向量
        a, q0, r, u, = self.generate('sqr', which='col', p=3)
        
        # 将 q0 按列拷贝为 q
        q = q0.copy('F')
        
        # 在 a 的第一列插入 u，形成新的矩阵 a1
        a1 = np.insert(a, np.zeros(3, np.intp), u, 1)

        # 不覆盖 Q 的情况下插入
        q1, r1 = qr_insert(q, r, u, 0, 'col', overwrite_qru=False)
        
        # 检查插入后的 QR 分解结果是否正确
        check_qr(q1, r1, a1, self.rtol, self.atol)
        
        # 检查未修改的 Q 和 R 是否保持不变
        check_qr(q, r, a, self.rtol, self.atol)

        # 尝试覆盖 Q 的情况下插入
        q2, r2 = qr_insert(q, r, u, 0, 'col', overwrite_qru=True)
        
        # 检查覆盖 Q 后的 QR 分解结果是否正确
        check_qr(q2, r2, a1, self.rtol, self.atol)
        
        # 断言 q2 是否与原始 q 相等
        assert_allclose(q2, q, rtol=self.rtol, atol=self.atol)

    def test_empty_inputs(self):
        # 测试空输入的情况下是否能正确引发 ValueError 异常

        # 生成测试数据：a 是生成的矩阵，q 是 Q 矩阵，r 是 R 矩阵，u 是要插入的行或列向量
        a, q, r, u = self.generate('sqr', which='row')
        
        # 测试在空矩阵上进行插入是否会引发异常
        assert_raises(ValueError, qr_insert, np.array([]), r, u, 0, 'row')
        assert_raises(ValueError, qr_insert, q, np.array([]), u, 0, 'row')
        assert_raises(ValueError, qr_insert, q, r, np.array([]), 0, 'row')
        assert_raises(ValueError, qr_insert, np.array([]), r, u, 0, 'col')
        assert_raises(ValueError, qr_insert, q, np.array([]), u, 0, 'col')
        assert_raises(ValueError, qr_insert, q, r, np.array([]), 0, 'col')

    def test_mismatched_shapes(self):
        # 测试不匹配的形状是否能正确引发 ValueError 异常

        # 生成测试数据：a 是生成的矩阵，q 是 Q 矩阵，r 是 R 矩阵，u 是要插入的行或列向量
        a, q, r, u = self.generate('tall', which='row')
        
        # 测试在不匹配形状的情况下进行插入是否会引发异常
        assert_raises(ValueError, qr_insert, q, r[1:], u, 0, 'row')
        assert_raises(ValueError, qr_insert, q[:-2], r, u, 0, 'row')
        assert_raises(ValueError, qr_insert, q, r, u[1:], 0, 'row')
        assert_raises(ValueError, qr_insert, q, r[1:], u, 0, 'col')
        assert_raises(ValueError, qr_insert, q[:-2], r, u, 0, 'col')
        assert_raises(ValueError, qr_insert, q, r, u[1:], 0, 'col')

    def test_unsupported_dtypes(self):
        # 测试不支持的数据类型是否能正确引发 ValueError 异常

        # 支持测试的数据类型列表
        dts = ['int8', 'int16', 'int32', 'int64',
               'uint8', 'uint16', 'uint32', 'uint64',
               'float16', 'longdouble', 'clongdouble',
               'bool']
        
        # 生成测试数据：a 是生成的矩阵，q0 是 Q 矩阵，r0 是 R 矩阵，u0 是要插入的行或列向量
        a, q0, r0, u0 = self.generate('sqr', which='row')
        
        # 遍历所有测试的数据类型
        for dtype in dts:
            # 将 q0 和 r0 转换为当前的数据类型，并且在无效操作时忽略错误
            q = q0.real.astype(dtype)
            with np.errstate(invalid="ignore"):
                r = r0.real.astype(dtype)
            u = u0.real.astype(dtype)
            
            # 测试在不支持的数据类型下进行插入是否会引发异常
            assert_raises(ValueError, qr_insert, q, r0, u0, 0, 'row')
            assert_raises(ValueError, qr_insert, q, r0, u0, 0, 'col')
            assert_raises(ValueError, qr_insert, q0, r, u0, 0, 'row')
            assert_raises(ValueError, qr_insert, q0, r, u0, 0, 'col')
            assert_raises(ValueError, qr_insert, q0, r0, u, 0, 'row')
            assert_raises(ValueError, qr_insert, q0, r0, u, 0, 'col')
    # 定义测试方法，用于检查 QR 插入函数的边界条件
    def test_check_finite(self):
        # 生成测试所需的矩阵和向量
        a0, q0, r0, u0 = self.generate('sqr', which='row', p=3)

        # 复制矩阵 q0 并确保其数据类型为 float
        q = q0.copy('F')
        # 将第 (1,1) 位置的元素设置为 NaN
        q[1,1] = np.nan
        # 断言在插入具有 NaN 的 q 矩阵时会引发 ValueError 异常
        assert_raises(ValueError, qr_insert, q, r0, u0[:,0], 0, 'row')
        assert_raises(ValueError, qr_insert, q, r0, u0, 0, 'row')
        assert_raises(ValueError, qr_insert, q, r0, u0[:,0], 0, 'col')
        assert_raises(ValueError, qr_insert, q, r0, u0, 0, 'col')

        # 复制矩阵 r0 并确保其数据类型为 float
        r = r0.copy('F')
        # 将第 (1,1) 位置的元素设置为 NaN
        r[1,1] = np.nan
        # 断言在插入具有 NaN 的 r 矩阵时会引发 ValueError 异常
        assert_raises(ValueError, qr_insert, q0, r, u0[:,0], 0, 'row')
        assert_raises(ValueError, qr_insert, q0, r, u0, 0, 'row')
        assert_raises(ValueError, qr_insert, q0, r, u0[:,0], 0, 'col')
        assert_raises(ValueError, qr_insert, q0, r, u0, 0, 'col')

        # 复制矩阵 u0 并确保其数据类型为 float
        u = u0.copy('F')
        # 将第 (0,0) 位置的元素设置为 NaN
        u[0,0] = np.nan
        # 断言在插入具有 NaN 的 u 矩阵时会引发 ValueError 异常
        assert_raises(ValueError, qr_insert, q0, r0, u[:,0], 0, 'row')
        assert_raises(ValueError, qr_insert, q0, r0, u, 0, 'row')
        assert_raises(ValueError, qr_insert, q0, r0, u[:,0], 0, 'col')
        assert_raises(ValueError, qr_insert, q0, r0, u, 0, 'col')
class TestQRinsert_f(BaseQRinsert):
    dtype = np.dtype('f')  # 设置数据类型为单精度浮点数

class TestQRinsert_F(BaseQRinsert):
    dtype = np.dtype('F')  # 设置数据类型为单精度复数

class TestQRinsert_d(BaseQRinsert):
    dtype = np.dtype('d')  # 设置数据类型为双精度浮点数

class TestQRinsert_D(BaseQRinsert):
    dtype = np.dtype('D')  # 设置数据类型为双精度复数

class BaseQRupdate(BaseQRdeltas):
    def generate(self, type, mode='full', p=1):
        a, q, r = super().generate(type, mode)  # 调用父类方法生成矩阵a, q, r

        # 超类调用设置了种子...

        if p == 1:
            u = np.random.random(q.shape[0])  # 创建长度为q行数的随机向量u
            v = np.random.random(r.shape[1])  # 创建长度为r列数的随机向量v
        else:
            u = np.random.random((q.shape[0], p))  # 创建形状为(q行数, p)的随机矩阵u
            v = np.random.random((r.shape[1], p))  # 创建形状为(r列数, p)的随机矩阵v

        if np.iscomplexobj(self.dtype.type(1)):
            b = np.random.random(u.shape)  # 创建与u形状相同的随机矩阵b
            u = u + 1j * b  # 将u转为复数类型，虚部为b

            c = np.random.random(v.shape)  # 创建与v形状相同的随机矩阵c
            v = v + 1j * c  # 将v转为复数类型，虚部为c

        u = u.astype(self.dtype)  # 将u转换为指定的数据类型
        v = v.astype(self.dtype)  # 将v转换为指定的数据类型
        return a, q, r, u, v  # 返回生成的矩阵a, q, r以及处理后的u, v

    def test_sqr_rank_1(self):
        a, q, r, u, v = self.generate('sqr')  # 生成'sqr'类型的矩阵a, q, r, u, v
        q1, r1 = qr_update(q, r, u, v, False)  # 使用qr_update函数更新q, r得到q1, r1
        a1 = a + np.outer(u, v.conj())  # 计算更新后的a1
        check_qr(q1, r1, a1, self.rtol, self.atol)  # 检查更新后的q1, r1, a1是否满足误差要求

    def test_sqr_rank_p(self):
        # 在这里也进行ndim=2，rank 1更新的测试
        for p in [1, 2, 3, 5]:
            a, q, r, u, v = self.generate('sqr', p=p)  # 生成'sqr'类型的矩阵a, q, r, u, v，指定参数p
            if p == 1:
                u = u.reshape(u.size, 1)  # 将u调整为列向量
                v = v.reshape(v.size, 1)  # 将v调整为列向量
            q1, r1 = qr_update(q, r, u, v, False)  # 使用qr_update函数更新q, r得到q1, r1
            a1 = a + np.dot(u, v.T.conj())  # 计算更新后的a1
            check_qr(q1, r1, a1, self.rtol, self.atol)  # 检查更新后的q1, r1, a1是否满足误差要求

    def test_tall_rank_1(self):
        a, q, r, u, v = self.generate('tall')  # 生成'tall'类型的矩阵a, q, r, u, v
        q1, r1 = qr_update(q, r, u, v, False)  # 使用qr_update函数更新q, r得到q1, r1
        a1 = a + np.outer(u, v.conj())  # 计算更新后的a1
        check_qr(q1, r1, a1, self.rtol, self.atol)  # 检查更新后的q1, r1, a1是否满足误差要求

    def test_tall_rank_p(self):
        for p in [1, 2, 3, 5]:
            a, q, r, u, v = self.generate('tall', p=p)  # 生成'tall'类型的矩阵a, q, r, u, v，指定参数p
            if p == 1:
                u = u.reshape(u.size, 1)  # 将u调整为列向量
                v = v.reshape(v.size, 1)  # 将v调整为列向量
            q1, r1 = qr_update(q, r, u, v, False)  # 使用qr_update函数更新q, r得到q1, r1
            a1 = a + np.dot(u, v.T.conj())  # 计算更新后的a1
            check_qr(q1, r1, a1, self.rtol, self.atol)  # 检查更新后的q1, r1, a1是否满足误差要求

    def test_fat_rank_1(self):
        a, q, r, u, v = self.generate('fat')  # 生成'fat'类型的矩阵a, q, r, u, v
        q1, r1 = qr_update(q, r, u, v, False)  # 使用qr_update函数更新q, r得到q1, r1
        a1 = a + np.outer(u, v.conj())  # 计算更新后的a1
        check_qr(q1, r1, a1, self.rtol, self.atol)  # 检查更新后的q1, r1, a1是否满足误差要求

    def test_fat_rank_p(self):
        for p in [1, 2, 3, 5]:
            a, q, r, u, v = self.generate('fat', p=p)  # 生成'fat'类型的矩阵a, q, r, u, v，指定参数p
            if p == 1:
                u = u.reshape(u.size, 1)  # 将u调整为列向量
                v = v.reshape(v.size, 1)  # 将v调整为列向量
            q1, r1 = qr_update(q, r, u, v, False)  # 使用qr_update函数更新q, r得到q1, r1
            a1 = a + np.dot(u, v.T.conj())  # 计算更新后的a1
            check_qr(q1, r1, a1, self.rtol, self.atol)  # 检查更新后的q1, r1, a1是否满足误差要求

    def test_economic_rank_1(self):
        a, q, r, u, v = self.generate('tall', 'economic')  # 生成'tall'类型的矩阵a, q, r, u, v，经济模式
        q1, r1 = qr_update(q, r, u, v, False)  # 使用qr_update函数更新q, r得到q1, r1
        a1 = a + np.outer(u, v.conj())  # 计算更新后的a1
        check_qr(q1, r1, a1, self.rtol, self.atol, False)  # 检查更新后的q1, r1, a1是否满足误差要求，不检查经济模式
    # 定义测试函数 test_economic_rank_p，用于测试在不同参数 p 下的 QR 分解更新
    def test_economic_rank_p(self):
        # 遍历参数列表 [1, 2, 3, 5]
        for p in [1, 2, 3, 5]:
            # 调用 generate 方法生成矩阵 a 和向量 u, v，以及 q, r 矩阵
            a, q, r, u, v = self.generate('tall', 'economic', p)
            # 如果 p 等于 1，需要对 u 和 v 进行形状调整
            if p == 1:
                u = u.reshape(u.size, 1)
                v = v.reshape(v.size, 1)
            # 调用 qr_update 函数进行 QR 更新，返回更新后的 q1, r1
            q1, r1 = qr_update(q, r, u, v, False)
            # 计算 a1，更新后的矩阵 a
            a1 = a + np.dot(u, v.T.conj())
            # 调用 check_qr 函数检查更新后的 QR 分解结果的正确性
            check_qr(q1, r1, a1, self.rtol, self.atol, False)

    # 定义测试函数 test_Mx1_rank_1，测试 Mx1 类型且 rank 1 的 QR 更新
    def test_Mx1_rank_1(self):
        # 调用 generate 方法生成矩阵 a 和向量 u, v，以及 q, r 矩阵
        a, q, r, u, v = self.generate('Mx1')
        # 调用 qr_update 函数进行 QR 更新，返回更新后的 q1, r1
        q1, r1 = qr_update(q, r, u, v, False)
        # 计算 a1，更新后的矩阵 a
        a1 = a + np.outer(u, v.conj())
        # 调用 check_qr 函数检查更新后的 QR 分解结果的正确性
        check_qr(q1, r1, a1, self.rtol, self.atol)

    # 定义测试函数 test_Mx1_rank_p，测试 Mx1 类型且 rank p 的 QR 更新
    def test_Mx1_rank_p(self):
        # 当 M 或 N 等于 1 时，只支持 rank 1 更新。虽然不是基本限制，但代码不支持此功能。
        # 调用 generate 方法生成矩阵 a 和向量 u, v，以及 q, r 矩阵，其中 p 被设置为 1
        a, q, r, u, v = self.generate('Mx1', p=1)
        # 将 u 和 v 进行形状调整
        u = u.reshape(u.size, 1)
        v = v.reshape(v.size, 1)
        # 调用 qr_update 函数进行 QR 更新，返回更新后的 q1, r1
        q1, r1 = qr_update(q, r, u, v, False)
        # 计算 a1，更新后的矩阵 a
        a1 = a + np.dot(u, v.T.conj())
        # 调用 check_qr 函数检查更新后的 QR 分解结果的正确性
        check_qr(q1, r1, a1, self.rtol, self.atol)

    # 定义测试函数 test_Mx1_economic_rank_1，测试 Mx1 economic 类型且 rank 1 的 QR 更新
    def test_Mx1_economic_rank_1(self):
        # 调用 generate 方法生成矩阵 a 和向量 u, v，以及 q, r 矩阵
        a, q, r, u, v = self.generate('Mx1', 'economic')
        # 调用 qr_update 函数进行 QR 更新，返回更新后的 q1, r1
        q1, r1 = qr_update(q, r, u, v, False)
        # 计算 a1，更新后的矩阵 a
        a1 = a + np.outer(u, v.conj())
        # 调用 check_qr 函数检查更新后的 QR 分解结果的正确性
        check_qr(q1, r1, a1, self.rtol, self.atol, False)

    # 定义测试函数 test_Mx1_economic_rank_p，测试 Mx1 economic 类型且 rank p 的 QR 更新
    def test_Mx1_economic_rank_p(self):
        # 当 M 或 N 等于 1 时，只支持 rank 1 更新。虽然不是基本限制，但代码不支持此功能。
        # 调用 generate 方法生成矩阵 a 和向量 u, v，以及 q, r 矩阵，其中 p 被设置为 1
        a, q, r, u, v = self.generate('Mx1', 'economic', p=1)
        # 将 u 和 v 进行形状调整
        u = u.reshape(u.size, 1)
        v = v.reshape(v.size, 1)
        # 调用 qr_update 函数进行 QR 更新，返回更新后的 q1, r1
        q1, r1 = qr_update(q, r, u, v, False)
        # 计算 a1，更新后的矩阵 a
        a1 = a + np.dot(u, v.T.conj())
        # 调用 check_qr 函数检查更新后的 QR 分解结果的正确性
        check_qr(q1, r1, a1, self.rtol, self.atol, False)

    # 定义测试函数 test_1xN_rank_1，测试 1xN 类型且 rank 1 的 QR 更新
    def test_1xN_rank_1(self):
        # 调用 generate 方法生成矩阵 a 和向量 u, v，以及 q, r 矩阵
        a, q, r, u, v = self.generate('1xN')
        # 调用 qr_update 函数进行 QR 更新，返回更新后的 q1, r1
        q1, r1 = qr_update(q, r, u, v, False)
        # 计算 a1，更新后的矩阵 a
        a1 = a + np.outer(u, v.conj())
        # 调用 check_qr 函数检查更新后的 QR 分解结果的正确性
        check_qr(q1, r1, a1, self.rtol, self.atol)

    # 定义测试函数 test_1xN_rank_p，测试 1xN 类型且 rank p 的 QR 更新
    def test_1xN_rank_p(self):
        # 当 M 或 N 等于 1 时，只支持 rank 1 更新。虽然不是基本限制，但代码不支持此功能。
        # 调用 generate 方法生成矩阵 a 和向量 u, v，以及 q, r 矩阵，其中 p 被设置为 1
        a, q, r, u, v = self.generate('1xN', p=1)
        # 将 u 和 v 进行形状调整
        u = u.reshape(u.size, 1)
        v = v.reshape(v.size, 1)
        # 调用 qr_update 函数进行 QR 更新，返回更新后的 q1, r1
        q1, r1 = qr_update(q, r, u, v, False)
        # 计算 a1，更新后的矩阵 a
        a1 = a + np.dot(u, v.T.conj())
        # 调用 check_qr 函数检查更新后的 QR 分解结果的正确性
        check_qr(q1, r1, a1, self.rtol, self.atol)

    # 定义测试函数 test_1x1_rank_1，测试 1x1 类型且 rank 1 的 QR 更新
    def test_1x1_rank_1(self):
        # 调用 generate 方法生成矩阵 a 和向量 u, v，以及 q, r 矩阵
        a, q, r, u, v = self.generate('1x1')
        # 调用 qr_update 函数进行 QR 更新，返回更新后的 q1, r1
        q1, r1 = qr_update(q, r, u, v, False)
        # 计算 a1，更新后的矩阵 a
        a1 = a + np.outer(u, v.conj
    def test_1x1_rank_1_scalar(self):
        # 生成 '1x1' 类型的矩阵和相关参数
        a, q, r, u, v = self.generate('1x1')
        # 检查是否引发 ValueError 异常，因为 q[0, 0] 是标量，不支持 qr_update 函数
        assert_raises(ValueError, qr_update, q[0, 0], r, u, v)
        # 检查是否引发 ValueError 异常，因为 r[0, 0] 是标量，不支持 qr_update 函数
        assert_raises(ValueError, qr_update, q, r[0, 0], u, v)
        # 检查是否引发 ValueError 异常，因为 u[0] 是标量，不支持 qr_update 函数
        assert_raises(ValueError, qr_update, q, r, u[0], v)
        # 检查是否引发 ValueError 异常，因为 v[0] 是标量，不支持 qr_update 函数
        assert_raises(ValueError, qr_update, q, r, u, v[0])

    def test_non_unit_strides_rank_1(self):
        # 调用 base_non_simple_strides 方法测试非单位步幅，完整存储方式，等于测试一维数组
        self.base_non_simple_strides(make_strided, 'full', 1, True)

    def test_non_unit_strides_economic_rank_1(self):
        # 调用 base_non_simple_strides 方法测试非单位步幅，经济存储方式，等于测试一维数组
        self.base_non_simple_strides(make_strided, 'economic', 1, True)

    def test_non_unit_strides_rank_p(self):
        # 调用 base_non_simple_strides 方法测试非单位步幅，完整存储方式，测试多维数组
        self.base_non_simple_strides(make_strided, 'full', 3, False)

    def test_non_unit_strides_economic_rank_p(self):
        # 调用 base_non_simple_strides 方法测试非单位步幅，经济存储方式，测试多维数组
        self.base_non_simple_strides(make_strided, 'economic', 3, False)

    def test_neg_strides_rank_1(self):
        # 调用 base_non_simple_strides 方法测试负步幅，完整存储方式，测试一维数组
        self.base_non_simple_strides(negate_strides, 'full', 1, False)

    def test_neg_strides_economic_rank_1(self):
        # 调用 base_non_simple_strides 方法测试负步幅，经济存储方式，测试一维数组
        self.base_non_simple_strides(negate_strides, 'economic', 1, False)

    def test_neg_strides_rank_p(self):
        # 调用 base_non_simple_strides 方法测试负步幅，完整存储方式，测试多维数组
        self.base_non_simple_strides(negate_strides, 'full', 3, False)

    def test_neg_strides_economic_rank_p(self):
        # 调用 base_non_simple_strides 方法测试负步幅，经济存储方式，测试多维数组
        self.base_non_simple_strides(negate_strides, 'economic', 3, False)

    def test_non_itemsize_strides_rank_1(self):
        # 调用 base_non_simple_strides 方法测试非默认项大小的步幅，完整存储方式，测试一维数组
        self.base_non_simple_strides(nonitemsize_strides, 'full', 1, False)

    def test_non_itemsize_strides_economic_rank_1(self):
        # 调用 base_non_simple_strides 方法测试非默认项大小的步幅，经济存储方式，测试一维数组
        self.base_non_simple_strides(nonitemsize_strides, 'economic', 1, False)

    def test_non_itemsize_strides_rank_p(self):
        # 调用 base_non_simple_strides 方法测试非默认项大小的步幅，完整存储方式，测试多维数组
        self.base_non_simple_strides(nonitemsize_strides, 'full', 3, False)

    def test_non_itemsize_strides_economic_rank_p(self):
        # 调用 base_non_simple_strides 方法测试非默认项大小的步幅，经济存储方式，测试多维数组
        self.base_non_simple_strides(nonitemsize_strides, 'economic', 3, False)

    def test_non_native_byte_order_rank_1(self):
        # 调用 base_non_simple_strides 方法测试非本机字节顺序，完整存储方式，测试一维数组
        self.base_non_simple_strides(make_nonnative, 'full', 1, False)

    def test_non_native_byte_order_economic_rank_1(self):
        # 调用 base_non_simple_strides 方法测试非本机字节顺序，经济存储方式，测试一维数组
        self.base_non_simple_strides(make_nonnative, 'economic', 1, False)

    def test_non_native_byte_order_rank_p(self):
        # 调用 base_non_simple_strides 方法测试非本机字节顺序，完整存储方式，测试多维数组
        self.base_non_simple_strides(make_nonnative, 'full', 3, False)

    def test_non_native_byte_order_economic_rank_p(self):
        # 调用 base_non_simple_strides 方法测试非本机字节顺序，经济存储方式，测试多维数组
        self.base_non_simple_strides(make_nonnative, 'economic', 3, False)
    def test_overwrite_qruv_rank_1(self):
        # 定义测试函数，测试覆盖对 QR 分解的影响

        # 生成测试数据 'sqr'
        a, q0, r0, u0, v0 = self.generate('sqr')

        # 构造矩阵 a1 = a + u0 * v0^H
        a1 = a + np.outer(u0, v0.conj())

        # 深复制 q0, r0, u0, v0，并确保是 F 风格的数组
        q = q0.copy('F')
        r = r0.copy('F')
        u = u0.copy('F')
        v = v0.copy('F')

        # 不覆盖原始数据
        q1, r1 = qr_update(q, r, u, v, False)
        check_qr(q1, r1, a1, self.rtol, self.atol)
        check_qr(q, r, a, self.rtol, self.atol)

        # 覆盖原始数据
        q2, r2 = qr_update(q, r, u, v, True)
        check_qr(q2, r2, a1, self.rtol, self.atol)

        # 验证覆盖操作，无法直接检查 u 和 v
        assert_allclose(q2, q, rtol=self.rtol, atol=self.atol)
        assert_allclose(r2, r, rtol=self.rtol, atol=self.atol)

        # 切换为 C 风格的数组
        q = q0.copy('C')
        r = r0.copy('C')
        u = u0.copy('C')
        v = v0.copy('C')

        # 覆盖原始数据
        q3, r3 = qr_update(q, r, u, v, True)
        check_qr(q3, r3, a1, self.rtol, self.atol)

        # 验证覆盖操作
        assert_allclose(q3, q, rtol=self.rtol, atol=self.atol)
        assert_allclose(r3, r, rtol=self.rtol, atol=self.atol)

    def test_overwrite_qruv_rank_1_economic(self):
        # 定义测试函数，测试经济型 QR 分解的覆盖效果

        # 生成测试数据 'tall', 'economic'
        a, q0, r0, u0, v0 = self.generate('tall', 'economic')

        # 构造矩阵 a1 = a + u0 * v0^H
        a1 = a + np.outer(u0, v0.conj())

        # 深复制 q0, r0, u0, v0，并确保是 F 风格的数组
        q = q0.copy('F')
        r = r0.copy('F')
        u = u0.copy('F')
        v = v0.copy('F')

        # 不覆盖原始数据
        q1, r1 = qr_update(q, r, u, v, False)
        check_qr(q1, r1, a1, self.rtol, self.atol, False)

        # 切换为 C 风格的数组，不覆盖原始数据
        check_qr(q, r, a, self.rtol, self.atol, False)

        # 覆盖原始数据
        q2, r2 = qr_update(q, r, u, v, True)
        check_qr(q2, r2, a1, self.rtol, self.atol, False)

        # 验证覆盖操作，无法直接检查 u 和 v
        assert_allclose(q2, q, rtol=self.rtol, atol=self.atol)
        assert_allclose(r2, r, rtol=self.rtol, atol=self.atol)

        # 切换为 C 风格的数组
        q = q0.copy('C')
        r = r0.copy('C')
        u = u0.copy('C')
        v = v0.copy('C')

        # 覆盖原始数据
        q3, r3 = qr_update(q, r, u, v, True)
        check_qr(q3, r3, a1, self.rtol, self.atol, False)

        # 验证覆盖操作
        assert_allclose(q3, q, rtol=self.rtol, atol=self.atol)
        assert_allclose(r3, r, rtol=self.rtol, atol=self.atol)
    def test_overwrite_qruv_rank_p(self):
        # for rank p updates, q r must be F contiguous, v must be C (v.T --> F)
        # and u can be C or F, but is only overwritten if Q is C and complex
        # 从生成器生成相应的矩阵和向量
        a, q0, r0, u0, v0 = self.generate('sqr', p=3)
        # 构造新的矩阵 a1，作为 a 和 u0*v0.T.conj() 的和
        a1 = a + np.dot(u0, v0.T.conj())
        # 对 q0, r0, u0, v0 进行深复制，同时要求 q 和 r 是 F 连续的，u 是 F，v 是 C
        q = q0.copy('F')
        r = r0.copy('F')
        u = u0.copy('F')
        v = v0.copy('C')

        # 不进行覆盖操作
        q1, r1 = qr_update(q, r, u, v, False)
        # 检查不覆盖操作后的 qr 分解的准确性
        check_qr(q1, r1, a1, self.rtol, self.atol)
        # 检查 q 和 r 未被覆盖
        check_qr(q, r, a, self.rtol, self.atol)

        # 进行覆盖操作
        q2, r2 = qr_update(q, r, u, v, True)
        # 检查覆盖操作后的 qr 分解的准确性
        check_qr(q2, r2, a1, self.rtol, self.atol)
        # 验证 q 和 r 被正确覆盖，无法直接验证 u 和 v
        assert_allclose(q2, q, rtol=self.rtol, atol=self.atol)
        assert_allclose(r2, r, rtol=self.rtol, atol=self.atol)

    def test_empty_inputs(self):
        # 生成 'tall' 类型的矩阵和向量
        a, q, r, u, v = self.generate('tall')
        # 确保空输入会引发 ValueError
        assert_raises(ValueError, qr_update, np.array([]), r, u, v)
        assert_raises(ValueError, qr_update, q, np.array([]), u, v)
        assert_raises(ValueError, qr_update, q, r, np.array([]), v)
        assert_raises(ValueError, qr_update, q, r, u, np.array([]))

    def test_mismatched_shapes(self):
        # 生成 'tall' 类型的矩阵和向量
        a, q, r, u, v = self.generate('tall')
        # 确保形状不匹配会引发 ValueError
        assert_raises(ValueError, qr_update, q, r[1:], u, v)
        assert_raises(ValueError, qr_update, q[:-2], r, u, v)
        assert_raises(ValueError, qr_update, q, r, u[1:], v)
        assert_raises(ValueError, qr_update, q, r, u, v[1:])

    def test_unsupported_dtypes(self):
        # 不支持的数据类型列表
        dts = ['int8', 'int16', 'int32', 'int64',
               'uint8', 'uint16', 'uint32', 'uint64',
               'float16', 'longdouble', 'clongdouble',
               'bool']
        # 生成 'tall' 类型的矩阵和向量
        a, q0, r0, u0, v0 = self.generate('tall')
        # 遍历所有不支持的数据类型
        for dtype in dts:
            # 将相关矩阵和向量转换为当前不支持的数据类型
            q = q0.real.astype(dtype)
            with np.errstate(invalid="ignore"):
                r = r0.real.astype(dtype)
            u = u0.real.astype(dtype)
            v = v0.real.astype(dtype)
            # 确保使用不支持的数据类型会引发 ValueError
            assert_raises(ValueError, qr_update, q, r0, u0, v0)
            assert_raises(ValueError, qr_update, q0, r, u0, v0)
            assert_raises(ValueError, qr_update, q0, r0, u, v0)
            assert_raises(ValueError, qr_update, q0, r0, u0, v)

    def test_integer_input(self):
        # 创建一个整数类型的输入矩阵 q
        q = np.arange(16).reshape(4, 4)
        r = q.copy()  # 复制 q，实际值并不重要
        u = q[:, 0].copy()
        v = r[0, :].copy()
        # 确保整数输入会引发 ValueError
        assert_raises(ValueError, qr_update, q, r, u, v)
    def test_check_finite(self):
        # 使用 `generate` 方法生成 'tall' 模式下的测试数据 a0, q0, r0, u0, v0
        a0, q0, r0, u0, v0 = self.generate('tall', p=3)

        # 复制 q0 并转换为 'F' (Fortran) 风格的数组
        q = q0.copy('F')
        # 将 q 中的某个元素设为 NaN
        q[1,1] = np.nan
        # 断言调用 qr_update 函数会抛出 ValueError 异常
        assert_raises(ValueError, qr_update, q, r0, u0[:,0], v0[:,0])
        # 同上，但传入的 u0 和 v0 是整列
        assert_raises(ValueError, qr_update, q, r0, u0, v0)

        # 复制 r0 并转换为 'F' 风格的数组
        r = r0.copy('F')
        # 将 r 中的某个元素设为 NaN
        r[1,1] = np.nan
        # 断言调用 qr_update 函数会抛出 ValueError 异常
        assert_raises(ValueError, qr_update, q0, r, u0[:,0], v0[:,0])
        # 同上，但传入的 u0 和 v0 是整列
        assert_raises(ValueError, qr_update, q0, r, u0, v0)

        # 复制 u0 并转换为 'F' 风格的数组
        u = u0.copy('F')
        # 将 u 中的某个元素设为 NaN
        u[0,0] = np.nan
        # 断言调用 qr_update 函数会抛出 ValueError 异常
        assert_raises(ValueError, qr_update, q0, r0, u[:,0], v0[:,0])
        # 同上，但传入的 u 和 v0 是整列
        assert_raises(ValueError, qr_update, q0, r0, u, v0)

        # 复制 v0 并转换为 'F' 风格的数组
        v = v0.copy('F')
        # 将 v 中的某个元素设为 NaN
        v[0,0] = np.nan
        # 断言调用 qr_update 函数会抛出 ValueError 异常
        assert_raises(ValueError, qr_update, q0, r0, u[:,0], v[:,0])
        # 同上，但传入的 u 和 v 是整列
        assert_raises(ValueError, qr_update, q0, r0, u, v)

    def test_economic_check_finite(self):
        # 使用 `generate` 方法生成 'tall' 模式下的测试数据 a0, q0, r0, u0, v0
        a0, q0, r0, u0, v0 = self.generate('tall', mode='economic', p=3)

        # 复制 q0 并转换为 'F' 风格的数组
        q = q0.copy('F')
        # 将 q 中的某个元素设为 NaN
        q[1,1] = np.nan
        # 断言调用 qr_update 函数会抛出 ValueError 异常
        assert_raises(ValueError, qr_update, q, r0, u0[:,0], v0[:,0])
        # 同上，但传入的 u0 和 v0 是整列
        assert_raises(ValueError, qr_update, q, r0, u0, v0)

        # 复制 r0 并转换为 'F' 风格的数组
        r = r0.copy('F')
        # 将 r 中的某个元素设为 NaN
        r[1,1] = np.nan
        # 断言调用 qr_update 函数会抛出 ValueError 异常
        assert_raises(ValueError, qr_update, q0, r, u0[:,0], v0[:,0])
        # 同上，但传入的 u0 和 v0 是整列
        assert_raises(ValueError, qr_update, q0, r, u0, v0)

        # 复制 u0 并转换为 'F' 风格的数组
        u = u0.copy('F')
        # 将 u 中的某个元素设为 NaN
        u[0,0] = np.nan
        # 断言调用 qr_update 函数会抛出 ValueError 异常
        assert_raises(ValueError, qr_update, q0, r0, u[:,0], v0[:,0])
        # 同上，但传入的 u 和 v0 是整列
        assert_raises(ValueError, qr_update, q0, r0, u, v0)

        # 复制 v0 并转换为 'F' 风格的数组
        v = v0.copy('F')
        # 将 v 中的某个元素设为 NaN
        v[0,0] = np.nan
        # 断言调用 qr_update 函数会抛出 ValueError 异常
        assert_raises(ValueError, qr_update, q0, r0, u[:,0], v[:,0])
        # 同上，但传入的 u 和 v 是整列
        assert_raises(ValueError, qr_update, q0, r0, u, v)

    def test_u_exactly_in_span_q(self):
        # 创建一个特定的 numpy 数组 q, r, u, v
        q = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], self.dtype)
        r = np.array([[1, 0], [0, 1]], self.dtype)
        u = np.array([0, 0, 0, -1], self.dtype)
        v = np.array([1, 2], self.dtype)
        # 调用 qr_update 函数，返回 q1, r1
        q1, r1 = qr_update(q, r, u, v)
        # 计算矩阵乘积 q * r 和外积 u * v 的和，赋给 a1
        a1 = np.dot(q, r) + np.outer(u, v.conj())
        # 调用 check_qr 函数，检查 q1, r1, a1 的数值是否符合给定的公差要求
        check_qr(q1, r1, a1, self.rtol, self.atol, False)
class TestQRupdate_f(BaseQRupdate):
    dtype = np.dtype('f')

class TestQRupdate_F(BaseQRupdate):
    dtype = np.dtype('F')

class TestQRupdate_d(BaseQRupdate):
    dtype = np.dtype('d')

class TestQRupdate_D(BaseQRupdate):
    dtype = np.dtype('D')

def test_form_qTu():
    # 确保通过此函数的所有代码路径都被测试到。大部分情况应该在测试套件中被覆盖，
    # 但显式的测试可以清楚地表明正在测试什么。
    #
    # 此函数期望 Q 是 C 或 F 连续的方阵。经济模式分解（Q 是 (M, N)，M != N）不会通过此函数。
    # U 可以有任何正的步长。
    #
    # 一些测试是重复的，因为连续的一维数组既是 C 连续的也是 F 连续的。

    q_order = ['F', 'C']
    q_shape = [(8, 8), ]
    u_order = ['F', 'C', 'A']  # 这里的 A 表示不是 F 也不是 C
    u_shape = [1, 3]
    dtype = ['f', 'd', 'F', 'D']

    for qo, qs, uo, us, d in \
            itertools.product(q_order, q_shape, u_order, u_shape, dtype):
        if us == 1:
            check_form_qTu(qo, qs, uo, us, 1, d)
            check_form_qTu(qo, qs, uo, us, 2, d)
        else:
            check_form_qTu(qo, qs, uo, us, 2, d)

def check_form_qTu(q_order, q_shape, u_order, u_shape, u_ndim, dtype):
    # 设定随机数种子
    np.random.seed(47)
    
    if u_shape == 1 and u_ndim == 1:
        u_shape = (q_shape[0],)
    else:
        u_shape = (q_shape[0], u_shape)
    
    dtype = np.dtype(dtype)

    # 根据 dtype 类型生成 Q 和 U 矩阵
    if dtype.char in 'fd':
        q = np.random.random(q_shape)
        u = np.random.random(u_shape)
    elif dtype.char in 'FD':
        q = np.random.random(q_shape) + 1j*np.random.random(q_shape)
        u = np.random.random(u_shape) + 1j*np.random.random(u_shape)
    else:
        raise ValueError("form_qTu doesn't support this dtype")

    # 要求 Q 按照指定的顺序和 dtype 进行转换
    q = np.require(q, dtype, q_order)
    
    # 如果 U 的顺序不是 'A'（任意），则按照指定的顺序和 dtype 进行转换
    if u_order != 'A':
        u = np.require(u, dtype, u_order)
    else:
        # 否则，调用 make_strided 函数将 U 转换为 strided 数组
        u, = make_strided((u.astype(dtype),))

    # 设置相对误差和绝对误差的容差值
    rtol = 10.0 ** -(np.finfo(dtype).precision-2)
    atol = 2*np.finfo(dtype).eps

    # 计算预期结果，使用矩阵乘积计算 Q 的共轭转置与 U 的乘积
    expected = np.dot(q.T.conj(), u)
    
    # 调用 _form_qTu 函数计算实际结果，并使用 assert_allclose 断言检查结果
    res = _decomp_update._form_qTu(q, u)
    assert_allclose(res, expected, rtol=rtol, atol=atol)
```