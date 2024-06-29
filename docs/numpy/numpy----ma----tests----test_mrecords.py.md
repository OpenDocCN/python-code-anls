# `.\numpy\numpy\ma\tests\test_mrecords.py`

```
# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for mrecords.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu

"""
# 导入pickle模块，用于对象序列化和反序列化
import pickle

# 导入NumPy库及其模块
import numpy as np
import numpy.ma as ma

# 导入NumPy中的masked和nomask函数
from numpy.ma import masked, nomask

# 导入NumPy中的temppath函数，用于临时路径处理
from numpy.testing import temppath

# 导入NumPy中的recarray, recfromrecords, recfromarrays等记录相关函数
from numpy._core.records import (
    recarray, fromrecords as recfromrecords, fromarrays as recfromarrays
    )

# 导入NumPy中的MaskedRecords, mrecarray, fromarrays, fromtextfile, fromrecords, addfield等记录相关函数
from numpy.ma.mrecords import (
    MaskedRecords, mrecarray, fromarrays, fromtextfile, fromrecords, addfield
    )

# 导入NumPy测试工具函数
from numpy.ma.testutils import (
    assert_, assert_equal,
    assert_equal_records,
    )

# 定义测试类TestMRecords
class TestMRecords:

    # 初始化测试数据
    ilist = [1, 2, 3, 4, 5]
    flist = [1.1, 2.2, 3.3, 4.4, 5.5]
    slist = [b'one', b'two', b'three', b'four', b'five']
    ddtype = [('a', int), ('b', float), ('c', '|S8')]
    mask = [0, 1, 0, 0, 1]
    base = ma.array(list(zip(ilist, flist, slist)), mask=mask, dtype=ddtype)

    # 定义测试方法test_byview
    def test_byview(self):
        # Test creation by view
        base = self.base  # 获取测试数据
        mbase = base.view(mrecarray)  # 将base转换为mrecarray类型
        assert_equal(mbase.recordmask, base.recordmask)  # 断言recordmask相等
        assert_equal_records(mbase._mask, base._mask)  # 断言_mask属性相等
        assert_(isinstance(mbase._data, recarray))  # 断言_data属性是recarray类型
        assert_equal_records(mbase._data, base._data.view(recarray))  # 断言_data属性与base的data属性视图相等
        for field in ('a', 'b', 'c'):
            assert_equal(base[field], mbase[field])  # 断言字段a、b、c在base和mbase中的值相等
        assert_equal_records(mbase.view(mrecarray), mbase)  # 断言mbase视图为mrecarray后仍等于mbase本身
    # 定义一个测试方法 `test_get`
    def test_get(self):
        # 测试字段的检索
        base = self.base.copy()
        mbase = base.view(mrecarray)
        
        # 遍历字段 'a', 'b', 'c'
        for field in ('a', 'b', 'c'):
            # 断言获取属性与获取字段内容相等
            assert_equal(getattr(mbase, field), mbase[field])
            # 断言基本数据与视图数据中的字段内容相等
            assert_equal(base[field], mbase[field])
        
        # 获取第一个元素
        mbase_first = mbase[0]
        # 断言第一个元素是 `mrecarray` 类型
        assert_(isinstance(mbase_first, mrecarray))
        # 断言第一个元素的数据类型与整体数据类型相同
        assert_equal(mbase_first.dtype, mbase.dtype)
        # 断言第一个元素的转换为列表后内容正确
        assert_equal(mbase_first.tolist(), (1, 1.1, b'one'))
        # 断言第一个元素的记录掩码为 `nomask`
        assert_equal(mbase_first.recordmask, nomask)
        # 断言第一个元素的掩码为全 `False`
        assert_equal(mbase_first._mask.item(), (False, False, False))
        # 断言通过索引 'a' 获取的内容与 `mbase['a'][0]` 相等
        assert_equal(mbase_first['a'], mbase['a'][0])
        
        # 获取最后一个元素
        mbase_last = mbase[-1]
        # 断言最后一个元素是 `mrecarray` 类型
        assert_(isinstance(mbase_last, mrecarray))
        # 断言最后一个元素的数据类型与整体数据类型相同
        assert_equal(mbase_last.dtype, mbase.dtype)
        # 断言最后一个元素的转换为列表后内容正确
        assert_equal(mbase_last.tolist(), (None, None, None))
        # 断言最后一个元素的记录掩码为 `True`
        assert_equal(mbase_last.recordmask, True)
        # 断言最后一个元素的掩码为全 `True`
        assert_equal(mbase_last._mask.item(), (True, True, True))
        # 断言通过索引 'a' 获取的内容为 `masked`
        assert_((mbase_last['a'] is masked))
        
        # 获取切片
        mbase_sl = mbase[:2]
        # 断言切片是 `mrecarray` 类型
        assert_(isinstance(mbase_sl, mrecarray))
        # 断言切片的数据类型与整体数据类型相同
        assert_equal(mbase_sl.dtype, mbase.dtype)
        # 断言切片的记录掩码为 `[0, 1]`
        assert_equal(mbase_sl.recordmask, [0, 1])
        # 断言切片的掩码与给定的掩码数组相等
        assert_equal_records(mbase_sl.mask,
                             np.array([(False, False, False),
                                       (True, True, True)],
                                      dtype=mbase._mask.dtype))
        # 断言切片内容与基本数据的前两个元素视图相等
        assert_equal_records(mbase_sl, base[:2].view(mrecarray))
        
        # 再次遍历字段 'a', 'b', 'c'
        for field in ('a', 'b', 'c'):
            # 断言获取属性与获取基本数据前两个元素的字段内容相等
            assert_equal(getattr(mbase_sl, field), base[:2][field])
    def test_set_fields(self):
        # Tests setting fields.
        base = self.base.copy()  # 复制 self.base 到 base
        mbase = base.view(mrecarray)  # 将 base 转换为 mrecarray 视图，并赋给 mbase
        mbase = mbase.copy()  # 复制 mbase 自身，生成一个新的副本
        mbase.fill_value = (999999, 1e20, 'N/A')  # 设置 mbase 的填充值
        # Change the data, the mask should be conserved
        mbase.a._data[:] = 5  # 修改 mbase 中字段 'a' 的数据为 5
        assert_equal(mbase['a']._data, [5, 5, 5, 5, 5])  # 断言检查 'a' 字段的数据是否为 [5, 5, 5, 5, 5]
        assert_equal(mbase['a']._mask, [0, 1, 0, 0, 1])  # 断言检查 'a' 字段的掩码是否为 [0, 1, 0, 0, 1]
        # Change the elements, and the mask will follow
        mbase.a = 1  # 将 mbase 中字段 'a' 的所有元素设置为 1
        assert_equal(mbase['a']._data, [1]*5)  # 断言检查 'a' 字段的数据是否全部为 1
        assert_equal(ma.getmaskarray(mbase['a']), [0]*5)  # 断言检查 'a' 字段的掩码是否全部为 0
        # Use to be _mask, now it's recordmask
        assert_equal(mbase.recordmask, [False]*5)  # 断言检查 recordmask 是否全部为 False
        assert_equal(mbase._mask.tolist(),
                     np.array([(0, 0, 0),
                               (0, 1, 1),
                               (0, 0, 0),
                               (0, 0, 0),
                               (0, 1, 1)],
                              dtype=bool))  # 断言检查整体的掩码矩阵是否符合预期
        # Set a field to mask ........................
        mbase.c = masked  # 将 mbase 中字段 'c' 的所有元素设置为 masked（掩码状态）
        # Use to be mask, and now it's still mask !
        assert_equal(mbase.c.mask, [1]*5)  # 断言检查 'c' 字段的掩码是否全部为 1
        assert_equal(mbase.c.recordmask, [1]*5)  # 断言检查 'c' 字段的 recordmask 是否全部为 1
        assert_equal(ma.getmaskarray(mbase['c']), [1]*5)  # 断言检查 'c' 字段的掩码是否全部为 1
        assert_equal(ma.getdata(mbase['c']), [b'N/A']*5)  # 断言检查 'c' 字段的数据是否全部为 b'N/A'
        assert_equal(mbase._mask.tolist(),
                     np.array([(0, 0, 1),
                               (0, 1, 1),
                               (0, 0, 1),
                               (0, 0, 1),
                               (0, 1, 1)],
                              dtype=bool))  # 断言检查整体的掩码矩阵是否符合预期
        # Set fields by slices .......................
        mbase = base.view(mrecarray).copy()  # 将 base 转换为 mrecarray 视图，并复制一份给 mbase
        mbase.a[3:] = 5  # 将 mbase 中字段 'a' 的索引从 3 开始的元素设置为 5
        assert_equal(mbase.a, [1, 2, 3, 5, 5])  # 断言检查 'a' 字段的数据是否符合预期
        assert_equal(mbase.a._mask, [0, 1, 0, 0, 0])  # 断言检查 'a' 字段的掩码是否符合预期
        mbase.b[3:] = masked  # 将 mbase 中字段 'b' 的索引从 3 开始的元素设置为 masked（掩码状态）
        assert_equal(mbase.b, base['b'])  # 断言检查 'b' 字段的数据是否符合 base 中相应字段的预期
        assert_equal(mbase.b._mask, [0, 1, 0, 1, 1])  # 断言检查 'b' 字段的掩码是否符合预期
        # Set fields globally..........................
        ndtype = [('alpha', '|S1'), ('num', int)]  # 定义一个结构化数据类型
        data = ma.array([('a', 1), ('b', 2), ('c', 3)], dtype=ndtype)  # 创建一个结构化数组
        rdata = data.view(MaskedRecords)  # 将结构化数组转换为 MaskedRecords 视图
        val = ma.array([10, 20, 30], mask=[1, 0, 0])  # 创建一个带有掩码的数组

        rdata['num'] = val  # 将 rdata 中字段 'num' 的值设置为 val
        assert_equal(rdata.num, val)  # 断言检查 'num' 字段的数据是否符合预期
        assert_equal(rdata.num.mask, [1, 0, 0])  # 断言检查 'num' 字段的掩码是否符合预期

    def test_set_fields_mask(self):
        # Tests setting the mask of a field.
        base = self.base.copy()  # 复制 self.base 到 base
        # This one has already a mask....
        mbase = base.view(mrecarray)  # 将 base 转换为 mrecarray 视图，并赋给 mbase
        mbase['a'][-2] = masked  # 将 mbase 中字段 'a' 的倒数第二个元素设置为 masked（掩码状态）
        assert_equal(mbase.a, [1, 2, 3, 4, 5])  # 断言检查 'a' 字段的数据是否符合预期
        assert_equal(mbase.a._mask, [0, 1, 0, 1, 1])  # 断言检查 'a' 字段的掩码是否符合预期
        # This one has not yet
        mbase = fromarrays([np.arange(5), np.random.rand(5)],
                           dtype=[('a', int), ('b', float)])  # 创建一个新的 mrecarray
        mbase['a'][-2] = masked  # 将 mbase 中字段 'a' 的倒数第二个元素设置为 masked（掩码状态）
        assert_equal(mbase.a, [0, 1, 2, 3, 4])  # 断言检查 'a' 字段的数据是否符合预期
        assert_equal(mbase.a._mask, [0, 0, 0, 1, 0])  # 断言检查 'a' 字段的掩码是否符合预期
    def test_set_mask(self):
        base = self.base.copy()
        mbase = base.view(mrecarray)
        # 设置掩码为True
        mbase.mask = masked
        # 断言：确保mbase['b']的掩码为全1数组
        assert_equal(ma.getmaskarray(mbase['b']), [1]*5)
        # 断言：确保mbase['a']和mbase['b']的掩码相同
        assert_equal(mbase['a']._mask, mbase['b']._mask)
        # 断言：确保mbase['a']和mbase['c']的掩码相同
        assert_equal(mbase['a']._mask, mbase['c']._mask)
        # 断言：确保mbase整体的掩码为全1数组
        assert_equal(mbase._mask.tolist(),
                     np.array([(1, 1, 1)]*5, dtype=bool))
        # 删除掩码
        mbase.mask = nomask
        # 断言：确保mbase['c']的掩码为全0数组
        assert_equal(ma.getmaskarray(mbase['c']), [0]*5)
        # 断言：确保mbase整体的掩码为全0数组
        assert_equal(mbase._mask.tolist(),
                     np.array([(0, 0, 0)]*5, dtype=bool))

    def test_set_mask_fromarray(self):
        base = self.base.copy()
        mbase = base.view(mrecarray)
        # 使用数组设置掩码
        mbase.mask = [1, 0, 0, 0, 1]
        # 断言：确保mbase.a的掩码为指定数组
        assert_equal(mbase.a.mask, [1, 0, 0, 0, 1])
        # 断言：确保mbase.b的掩码为指定数组
        assert_equal(mbase.b.mask, [1, 0, 0, 0, 1])
        # 断言：确保mbase.c的掩码为指定数组
        assert_equal(mbase.c.mask, [1, 0, 0, 0, 1])
        # 再次设置掩码
        mbase.mask = [0, 0, 0, 0, 1]
        # 断言：确保mbase.a的掩码为指定数组
        assert_equal(mbase.a.mask, [0, 0, 0, 0, 1])
        # 断言：确保mbase.b的掩码为指定数组
        assert_equal(mbase.b.mask, [0, 0, 0, 0, 1])
        # 断言：确保mbase.c的掩码为指定数组
        assert_equal(mbase.c.mask, [0, 0, 0, 0, 1])

    def test_set_mask_fromfields(self):
        mbase = self.base.copy().view(mrecarray)

        nmask = np.array(
            [(0, 1, 0), (0, 1, 0), (1, 0, 1), (1, 0, 1), (0, 0, 0)],
            dtype=[('a', bool), ('b', bool), ('c', bool)])
        # 使用字段数组设置掩码
        mbase.mask = nmask
        # 断言：确保mbase.a的掩码为指定数组
        assert_equal(mbase.a.mask, [0, 0, 1, 1, 0])
        # 断言：确保mbase.b的掩码为指定数组
        assert_equal(mbase.b.mask, [1, 1, 0, 0, 0])
        # 断言：确保mbase.c的掩码为指定数组
        assert_equal(mbase.c.mask, [0, 0, 1, 1, 0])
        # 重新初始化并再次设置掩码
        mbase.mask = False
        mbase.fieldmask = nmask
        # 断言：确保mbase.a的掩码为指定数组
        assert_equal(mbase.a.mask, [0, 0, 1, 1, 0])
        # 断言：确保mbase.b的掩码为指定数组
        assert_equal(mbase.b.mask, [1, 1, 0, 0, 0])
        # 断言：确保mbase.c的掩码为指定数组
        assert_equal(mbase.c.mask, [0, 0, 1, 1, 0])
    def test_set_elements(self):
        base = self.base.copy()
        # 复制基础数据结构并转换为记录数组视图，再次复制以确保修改不影响原始数据
        mbase = base.view(mrecarray).copy()
        # 将倒数第二行设置为掩码（masked）
        mbase[-2] = masked
        assert_equal(
            mbase._mask.tolist(),
            np.array([(0, 0, 0), (1, 1, 1), (0, 0, 0), (1, 1, 1), (1, 1, 1)],
                     dtype=bool))
        # 检查记录掩码（recordmask），预期为 [0, 1, 0, 1, 1]
        assert_equal(mbase.recordmask, [0, 1, 0, 1, 1])
        
        # 再次复制基础数据结构并转换为记录数组视图，设置前两行为元组 (5, 5, 5)
        mbase = base.view(mrecarray).copy()
        mbase[:2] = (5, 5, 5)
        # 检查字段 a 的数据部分，预期为 [5, 5, 3, 4, 5]
        assert_equal(mbase.a._data, [5, 5, 3, 4, 5])
        # 检查字段 a 的掩码部分，预期为 [0, 0, 0, 0, 1]
        assert_equal(mbase.a._mask, [0, 0, 0, 0, 1])
        # 检查字段 b 的数据部分，预期为 [5., 5., 3.3, 4.4, 5.5]
        assert_equal(mbase.b._data, [5., 5., 3.3, 4.4, 5.5])
        # 检查字段 b 的掩码部分，预期为 [0, 0, 0, 0, 1]
        assert_equal(mbase.b._mask, [0, 0, 0, 0, 1])
        # 检查字段 c 的数据部分，预期为 [b'5', b'5', b'three', b'four', b'five']
        assert_equal(mbase.c._data,
                     [b'5', b'5', b'three', b'four', b'five'])
        # 检查字段 b 的掩码部分，预期为 [0, 0, 0, 0, 1]
        assert_equal(mbase.b._mask, [0, 0, 0, 0, 1])

        # 再次复制基础数据结构并转换为记录数组视图，设置前两行为掩码（masked）
        mbase = base.view(mrecarray).copy()
        mbase[:2] = masked
        # 检查字段 a 的数据部分，预期为 [1, 2, 3, 4, 5]
        assert_equal(mbase.a._data, [1, 2, 3, 4, 5])
        # 检查字段 a 的掩码部分，预期为 [1, 1, 0, 0, 1]
        assert_equal(mbase.a._mask, [1, 1, 0, 0, 1])
        # 检查字段 b 的数据部分，预期为 [1.1, 2.2, 3.3, 4.4, 5.5]
        assert_equal(mbase.b._data, [1.1, 2.2, 3.3, 4.4, 5.5])
        # 检查字段 b 的掩码部分，预期为 [1, 1, 0, 0, 1]
        assert_equal(mbase.b._mask, [1, 1, 0, 0, 1])
        # 检查字段 c 的数据部分，预期为 [b'one', b'two', b'three', b'four', b'five']
        assert_equal(mbase.c._data,
                     [b'one', b'two', b'three', b'four', b'five'])
        # 检查字段 b 的掩码部分，预期为 [1, 1, 0, 0, 1]
        assert_equal(mbase.b._mask, [1, 1, 0, 0, 1])

    def test_setslices_hardmask(self):
        # 测试使用硬掩码设置切片
        base = self.base.copy()
        mbase = base.view(mrecarray)
        # 硬化掩码
        mbase.harden_mask()
        try:
            # 尝试设置倒数第二行及之后为元组 (5, 5, 5)
            mbase[-2:] = (5, 5, 5)
            # 检查字段 a 的数据部分，预期为 [1, 2, 3, 5, 5]
            assert_equal(mbase.a._data, [1, 2, 3, 5, 5])
            # 检查字段 b 的数据部分，预期为 [1.1, 2.2, 3.3, 5, 5.5]
            assert_equal(mbase.b._data, [1.1, 2.2, 3.3, 5, 5.5])
            # 检查字段 c 的数据部分，预期为 [b'one', b'two', b'three', b'5', b'five']
            assert_equal(mbase.c._data,
                         [b'one', b'two', b'three', b'5', b'five'])
            # 检查字段 a 的掩码部分，预期为 [0, 1, 0, 0, 1]
            assert_equal(mbase.a._mask, [0, 1, 0, 0, 1])
            # 检查字段 b 的掩码部分，预期与字段 a 相同
            assert_equal(mbase.b._mask, mbase.a._mask)
            # 检查字段 b 的掩码部分，预期与字段 a 相同
            assert_equal(mbase.b._mask, mbase.c._mask)
        except NotImplementedError:
            # 如果抛出未实现错误，捕获并忽略
            pass
        except AssertionError:
            # 如果断言错误，重新抛出异常
            raise
        else:
            # 如果没有抛出异常，抛出自定义异常
            raise Exception("Flexible hard masks should be supported !")
        
        # 尝试设置倒数第二行及之后为单个值 3，预期抛出类型错误
        try:
            mbase[-2:] = 3
        except (NotImplementedError, TypeError):
            # 如果抛出未实现错误或类型错误，捕获并忽略
            pass
        else:
            # 如果没有抛出预期的异常类型错误，抛出类型错误异常
            raise TypeError("Should have expected a readable buffer object!")
    def test_hardmask(self):
        # Test hardmask
        # 复制基础数据，并将其视为结构化数组
        base = self.base.copy()
        mbase = base.view(mrecarray)
        # 设置硬屏蔽
        mbase.harden_mask()
        # 断言硬屏蔽已开启
        assert_(mbase._hardmask)
        # 清除屏蔽
        mbase.mask = nomask
        # 断言记录屏蔽与基础的屏蔽相等
        assert_equal_records(mbase._mask, base._mask)
        # 变为软屏蔽
        mbase.soften_mask()
        # 断言硬屏蔽已关闭
        assert_(not mbase._hardmask)
        # 清除屏蔽
        mbase.mask = nomask
        # 所以，字段的屏蔽不再设置为 nomask...
        # 断言记录的屏蔽与基础形状、数据类型的全部屏蔽相等
        assert_equal_records(mbase._mask,
                             ma.make_mask_none(base.shape, base.dtype))
        # 断言 mbase['b'] 的屏蔽为 nomask
        assert_(ma.make_mask(mbase['b']._mask) is nomask)
        # 断言 mbase['a'] 的屏蔽与 mbase['b'] 的屏蔽相等
        assert_equal(mbase['a']._mask, mbase['b']._mask)

    def test_pickling(self):
        # Test pickling
        # 复制基础数据
        base = self.base.copy()
        # 将基础数据视为结构化数组
        mrec = base.view(mrecarray)
        # 对于从 2 到 pickle.HIGHEST_PROTOCOL + 1 的每个协议
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            # 使用指定协议序列化 mrec
            _ = pickle.dumps(mrec, protocol=proto)
            # 反序列化为 mrec_
            mrec_ = pickle.loads(_)
            # 断言 mrec_ 的数据类型与 mrec 相等
            assert_equal(mrec_.dtype, mrec.dtype)
            # 断言 mrec_ 的数据记录与 mrec 的数据记录相等
            assert_equal_records(mrec_._data, mrec._data)
            # 断言 mrec_ 的屏蔽与 mrec 的屏蔽相等
            assert_equal(mrec_._mask, mrec._mask)
            # 断言 mrec_ 的数据记录的屏蔽与 mrec 的数据记录的屏蔽相等
            assert_equal_records(mrec_._mask, mrec._mask)

    def test_filled(self):
        # Test filling the array
        # 创建带屏蔽的数组 _a, _b, _c
        _a = ma.array([1, 2, 3], mask=[0, 0, 1], dtype=int)
        _b = ma.array([1.1, 2.2, 3.3], mask=[0, 0, 1], dtype=float)
        _c = ma.array(['one', 'two', 'three'], mask=[0, 0, 1], dtype='|S8')
        # 定义结构化数组的数据类型 ddtype
        ddtype = [('a', int), ('b', float), ('c', '|S8')]
        # 使用 fromarrays 创建结构化数组 mrec，并指定填充值
        mrec = fromarrays([_a, _b, _c], dtype=ddtype,
                          fill_value=(99999, 99999., 'N/A'))
        # 获取填充后的结构化数组 mrecfilled
        mrecfilled = mrec.filled()
        # 断言填充后的 mrecfilled['a'] 与预期相等
        assert_equal(mrecfilled['a'], np.array((1, 2, 99999), dtype=int))
        # 断言填充后的 mrecfilled['b'] 与预期相等
        assert_equal(mrecfilled['b'], np.array((1.1, 2.2, 99999.),
                                               dtype=float))
        # 断言填充后的 mrecfilled['c'] 与预期相等
        assert_equal(mrecfilled['c'], np.array(('one', 'two', 'N/A'),
                                               dtype='|S8'))

    def test_tolist(self):
        # Test tolist.
        # 创建带屏蔽的数组 _a, _b, _c
        _a = ma.array([1, 2, 3], mask=[0, 0, 1], dtype=int)
        _b = ma.array([1.1, 2.2, 3.3], mask=[0, 0, 1], dtype=float)
        _c = ma.array(['one', 'two', 'three'], mask=[1, 0, 0], dtype='|S8')
        # 定义结构化数组的数据类型 ddtype
        ddtype = [('a', int), ('b', float), ('c', '|S8')]
        # 使用 fromarrays 创建结构化数组 mrec，并指定填充值
        mrec = fromarrays([_a, _b, _c], dtype=ddtype,
                          fill_value=(99999, 99999., 'N/A'))

        # 断言 mrec 转换为列表后与预期相等
        assert_equal(mrec.tolist(),
                     [(1, 1.1, None), (2, 2.2, b'two'),
                      (None, None, b'three')])

    def test_withnames(self):
        # Test the creation w/ format and names
        # 使用给定格式和名称创建结构化数组 x
        x = mrecarray(1, formats=float, names='base')
        # 设置第一个元素的 'base' 字段为 10
        x[0]['base'] = 10
        # 断言 x 的 'base' 字段第一个元素与预期相等
        assert_equal(x['base'][0], 10)
    # 定义测试函数 test_exotic_formats(self)
    def test_exotic_formats(self):
        # 测试处理“奇特”格式是否正确
        # 创建一个具有指定数据类型的结构化数组，包含整数、字节字符串和浮点数字段
        easy = mrecarray(1, dtype=[('i', int), ('s', '|S8'), ('f', float)])
        # 将数组的第一个元素设置为掩码值
        easy[0] = masked
        # 断言填充掩码后的数组元素是否符合预期值
        assert_equal(easy.filled(1).item(), (1, b'1', 1.))

        # 创建一个具有指定数据类型的结构化数组，包含一个名为 'f0' 的二维浮点数组字段
        solo = mrecarray(1, dtype=[('f0', '<f8', (2, 2))])
        # 将数组的第一个元素设置为掩码值
        solo[0] = masked
        # 断言填充掩码后的数组元素是否符合预期值
        assert_equal(solo.filled(1).item(),
                     np.array((1,), dtype=solo.dtype).item())

        # 创建一个具有复杂数据类型的结构化数组，包含整数、二维浮点数组和单个浮点数字段
        mult = mrecarray(2, dtype="i4, (2,3)float, float")
        # 将数组的第一个元素设置为掩码值
        mult[0] = masked
        # 设置数组的第二个元素为指定的值元组
        mult[1] = (1, 1, 1)
        # 调用 filled 方法填充数组中的掩码值
        mult.filled(0)
        # 断言填充掩码后的数组是否与预期数组相等
        assert_equal_records(mult.filled(0),
                             np.array([(0, 0, 0), (1, 1, 1)],
                                      dtype=mult.dtype))
class TestView:

    def setup_method(self):
        # 创建包含整数和随机浮点数的 NumPy 数组
        (a, b) = (np.arange(10), np.random.rand(10))
        # 定义数组的数据类型
        ndtype = [('a', float), ('b', float)]
        # 使用数据类型创建结构化数组
        arr = np.array(list(zip(a, b)), dtype=ndtype)

        # 使用 fromarrays 函数创建一个 MaskedRecords 对象
        mrec = fromarrays([a, b], dtype=ndtype, fill_value=(-9., -99.))
        # 将第三个元素的第二个字段标记为 True
        mrec.mask[3] = (False, True)
        # 将数据保存在实例变量中
        self.data = (mrec, a, b, arr)

    def test_view_by_itself(self):
        (mrec, a, b, arr) = self.data
        # 创建 mrec 的视图对象
        test = mrec.view()
        # 断言 test 是 MaskedRecords 类型的对象
        assert_(isinstance(test, MaskedRecords))
        # 断言 test 与 mrec 相等
        assert_equal_records(test, mrec)
        # 断言 test 的掩码与 mrec 的掩码相等
        assert_equal_records(test._mask, mrec._mask)

    def test_view_simple_dtype(self):
        (mrec, a, b, arr) = self.data
        # 定义一个简单的数据类型
        ntype = (float, 2)
        # 创建 mrec 的视图对象，使用指定的数据类型
        test = mrec.view(ntype)
        # 断言 test 是 ma.MaskedArray 类型的对象
        assert_(isinstance(test, ma.MaskedArray))
        # 断言 test 与 arr 相等，转换为指定的浮点数数据类型
        assert_equal(test, np.array(list(zip(a, b)), dtype=float))
        # 断言 test 的第四行第二列是掩码值
        assert_(test[3, 1] is ma.masked)

    def test_view_flexible_type(self):
        (mrec, a, b, arr) = self.data
        # 定义一个灵活的数据类型
        alttype = [('A', float), ('B', float)]
        # 创建 mrec 的视图对象，使用灵活的数据类型
        test = mrec.view(alttype)
        # 断言 test 是 MaskedRecords 类型的对象
        assert_(isinstance(test, MaskedRecords))
        # 断言 test 与 arr 的视图对象相等，使用指定的灵活数据类型
        assert_equal_records(test, arr.view(alttype))
        # 断言 test 的'B'字段的第四个元素是掩码值
        assert_(test['B'][3] is masked)
        # 断言 test 的数据类型与指定的灵活数据类型相等
        assert_equal(test.dtype, np.dtype(alttype))
        # 断言 test 的填充值为 None
        assert_(test._fill_value is None)


##############################################################################
class TestMRecordsImport:

    # 创建带有掩码的 ma.array 对象
    _a = ma.array([1, 2, 3], mask=[0, 0, 1], dtype=int)
    _b = ma.array([1.1, 2.2, 3.3], mask=[0, 0, 1], dtype=float)
    _c = ma.array([b'one', b'two', b'three'],
                  mask=[0, 0, 1], dtype='|S8')
    # 定义一个结构化数据类型
    ddtype = [('a', int), ('b', float), ('c', '|S8')]
    # 使用 fromarrays 函数创建 MaskedRecords 对象
    mrec = fromarrays([_a, _b, _c], dtype=ddtype,
                      fill_value=(b'99999', b'99999.',
                                  b'N/A'))
    # 使用 recfromarrays 函数创建记录数组对象
    nrec = recfromarrays((_a._data, _b._data, _c._data), dtype=ddtype)
    # 将数据保存在实例变量中
    data = (mrec, nrec, ddtype)

    def test_fromarrays(self):
        # 创建带有掩码的 ma.array 对象
        _a = ma.array([1, 2, 3], mask=[0, 0, 1], dtype=int)
        _b = ma.array([1.1, 2.2, 3.3], mask=[0, 0, 1], dtype=float)
        _c = ma.array(['one', 'two', 'three'], mask=[0, 0, 1], dtype='|S8')
        # 解包数据元组
        (mrec, nrec, _) = self.data
        # 遍历字段和其对应的 ma.array 对象
        for (f, l) in zip(('a', 'b', 'c'), (_a, _b, _c)):
            # 断言 mrec 中字段的掩码与相应的 ma.array 对象的掩码相等
            assert_equal(getattr(mrec, f)._mask, l._mask)
        # 创建只有一个记录的 ma.array 对象
        _x = ma.array([1, 1.1, 'one'], mask=[1, 0, 0], dtype=object)
        # 断言使用 fromarrays 函数创建的对象与 mrec 的第一个记录相等
        assert_equal_records(fromarrays(_x, dtype=mrec.dtype), mrec[0])
    def test_fromrecords(self):
        # Test construction from records.

        (mrec, nrec, ddtype) = self.data
        # 解包数据元组，获取测试数据 mrec, nrec, ddtype

        palist = [(1, 'abc', 3.7000002861022949, 0),
                  (2, 'xy', 6.6999998092651367, 1),
                  (0, ' ', 0.40000000596046448, 0)]
        # 定义一个包含多个记录的列表 palist

        pa = recfromrecords(palist, names='c1, c2, c3, c4')
        # 使用 recfromrecords 函数从 palist 中创建结构化数组 pa，并指定字段名

        mpa = fromrecords(palist, names='c1, c2, c3, c4')
        # 使用 fromrecords 函数从 palist 中创建结构化数组 mpa，并指定字段名

        assert_equal_records(pa, mpa)
        # 断言结构化数组 pa 和 mpa 相等

        _mrec = fromrecords(nrec)
        # 使用 fromrecords 函数从 nrec 中创建结构化数组 _mrec

        assert_equal(_mrec.dtype, mrec.dtype)
        # 断言 _mrec 的数据类型与 mrec 的数据类型相等

        for field in _mrec.dtype.names:
            assert_equal(getattr(_mrec, field), getattr(mrec._data, field))
        # 遍历 _mrec 的字段名，断言每个字段的属性值与 mrec 数据对象中对应字段的属性值相等

        _mrec = fromrecords(nrec.tolist(), names='c1,c2,c3')
        # 使用 fromrecords 函数从 nrec 转换成列表后，指定字段名创建结构化数组 _mrec

        assert_equal(_mrec.dtype, [('c1', int), ('c2', float), ('c3', '|S5')])
        # 断言 _mrec 的数据类型为包含字段 'c1', 'c2', 'c3' 的元组数组，分别指定为 int, float 和固定字节字符串

        for (f, n) in zip(('c1', 'c2', 'c3'), ('a', 'b', 'c')):
            assert_equal(getattr(_mrec, f), getattr(mrec._data, n))
        # 使用 zip 函数并结合 getattr 函数，断言 _mrec 中的字段值与 mrec 数据对象中对应字段的值相等

        _mrec = fromrecords(mrec)
        # 使用 fromrecords 函数直接从 mrec 创建结构化数组 _mrec

        assert_equal(_mrec.dtype, mrec.dtype)
        # 断言 _mrec 的数据类型与 mrec 的数据类型相等

        assert_equal_records(_mrec._data, mrec.filled())
        # 断言 _mrec 的数据部分与 mrec 的填充后数据部分相等

        assert_equal_records(_mrec._mask, mrec._mask)
        # 断言 _mrec 的掩码部分与 mrec 的掩码部分相等

    def test_fromrecords_wmask(self):
        # Tests construction from records w/ mask.

        (mrec, nrec, ddtype) = self.data
        # 解包数据元组，获取测试数据 mrec, nrec, ddtype

        _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=[0, 1, 0,])
        # 使用 fromrecords 函数从 nrec 转换成列表后，指定数据类型 ddtype，并指定掩码创建结构化数组 _mrec

        assert_equal_records(_mrec._data, mrec._data)
        # 断言 _mrec 的数据部分与 mrec 的数据部分相等

        assert_equal(_mrec._mask.tolist(), [(0, 0, 0), (1, 1, 1), (0, 0, 0)])
        # 断言 _mrec 的掩码部分与指定的掩码列表相等

        _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=True)
        # 使用 fromrecords 函数从 nrec 转换成列表后，指定数据类型 ddtype，并使用 True 作为掩码创建结构化数组 _mrec

        assert_equal_records(_mrec._data, mrec._data)
        # 断言 _mrec 的数据部分与 mrec 的数据部分相等

        assert_equal(_mrec._mask.tolist(), [(1, 1, 1), (1, 1, 1), (1, 1, 1)])
        # 断言 _mrec 的掩码部分全为 True

        _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=mrec._mask)
        # 使用 fromrecords 函数从 nrec 转换成列表后，指定数据类型 ddtype，并使用 mrec 的掩码创建结构化数组 _mrec

        assert_equal_records(_mrec._data, mrec._data)
        # 断言 _mrec 的数据部分与 mrec 的数据部分相等

        assert_equal(_mrec._mask.tolist(), mrec._mask.tolist())
        # 断言 _mrec 的掩码部分与 mrec 的掩码部分相等

        _mrec = fromrecords(nrec.tolist(), dtype=ddtype,
                            mask=mrec._mask.tolist())
        # 使用 fromrecords 函数从 nrec 转换成列表后，指定数据类型 ddtype，并使用 mrec 的掩码列表创建结构化数组 _mrec

        assert_equal_records(_mrec._data, mrec._data)
        # 断言 _mrec 的数据部分与 mrec 的数据部分相等

        assert_equal(_mrec._mask.tolist(), mrec._mask.tolist())
        # 断言 _mrec 的掩码部分与 mrec 的掩码部分相等

    def test_fromtextfile(self):
        # Tests reading from a text file.
        
        fcontent = (
# 定义一个测试类 TestMaskedRecords，用于测试 MaskedRecords 相关功能
class TestMaskedRecords(TestCase):

    # 测试从文本文件读取并生成 MaskedRecords 对象的功能
    def test_fromtextfile(self):
        # 定义测试用的文本内容
        fcontent = (
            """#
            'One (S)','Two (I)','Three (F)','Four (M)','Five (-)','Six (C)'
            'strings',1,1.0,'mixed column',,1
            'with embedded "double quotes"',2,2.0,1.0,,1
            'strings',3,3.0E5,3,,1
            'strings',4,-1e-10,,,1
            """)
        # 创建临时文件路径，将文本内容写入文件
        with temppath() as path:
            with open(path, 'w') as f:
                f.write(fcontent)
            # 调用 fromtextfile 函数，从文件中读取数据并生成 MaskedRecords 对象
            mrectxt = fromtextfile(path, delimiter=',', varnames='ABCDEFG')
        # 断言 mrectxt 是 MaskedRecords 类型的对象
        assert_(isinstance(mrectxt, MaskedRecords))
        # 断言 mrectxt 中属性 F 的值与预期列表相等
        assert_equal(mrectxt.F, [1, 1, 1, 1])
        # 断言 mrectxt 中属性 E 的掩码（mask）与预期列表相等
        assert_equal(mrectxt.E._mask, [1, 1, 1, 1])
        # 断言 mrectxt 中属性 C 的值与预期列表相等
        assert_equal(mrectxt.C, [1, 2, 3.e+5, -1e-10])

    # 测试向 MaskedRecords 对象添加新字段的功能
    def test_addfield(self):
        # 获取测试数据
        (mrec, nrec, ddtype) = self.data
        (d, m) = ([100, 200, 300], [1, 0, 0])
        # 向 mrec 对象添加新字段，字段值为数组 d，掩码为数组 m
        mrec = addfield(mrec, ma.array(d, mask=m))
        # 断言 mrec 中新添加的字段 f3 的值与数组 d 相等
        assert_equal(mrec.f3, d)
        # 断言 mrec 中新添加的字段 f3 的掩码与数组 m 相等
        assert_equal(mrec.f3._mask, m)


def test_record_array_with_object_field():
    # Trac #1839
    # 创建一个具有对象字段的 MaskedArray 对象 y
    y = ma.masked_array(
        [(1, '2'), (3, '4')],
        mask=[(0, 0), (0, 1)],
        dtype=[('a', int), ('b', object)])
    # 尝试获取 y 的第二个元素，以前此操作会失败
    y[1]
```