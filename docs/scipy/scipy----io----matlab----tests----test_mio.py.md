# `D:\src\scipysrc\scipy\scipy\io\matlab\tests\test_mio.py`

```
''' Nose test generators

Need function load / save / roundtrip tests

'''
# 导入所需的库
import os
from collections import OrderedDict
from os.path import join as pjoin, dirname
from glob import glob
from io import BytesIO
import re
from tempfile import mkdtemp

import warnings
import shutil
import gzip

# 导入 numpy 的测试工具函数
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_equal, assert_, assert_warns, assert_allclose)
# 导入 pytest，并重命名 raises 函数为 assert_raises
import pytest
from pytest import raises as assert_raises

# 导入 numpy 和 scipy 相关模块
import numpy as np
from numpy import array
import scipy.sparse as SP

# 导入 scipy 的 mat 文件处理相关模块
import scipy.io
from scipy.io.matlab import MatlabOpaque, MatlabFunction, MatlabObject
import scipy.io.matlab._byteordercodes as boc
from scipy.io.matlab._miobase import (
    matdims, MatWriteError, MatReadError, matfile_version)
from scipy.io.matlab._mio import mat_reader_factory, loadmat, savemat, whosmat
from scipy.io.matlab._mio5 import (
    MatFile5Writer, MatFile5Reader, varmats_from_mat, to_writeable,
    EmptyStructMarker)
import scipy.io.matlab._mio5_params as mio5p
from scipy._lib._util import VisibleDeprecationWarning

# 定义测试数据的路径
test_data_path = pjoin(dirname(__file__), 'data')


def mlarr(*args, **kwargs):
    """Convenience function to return matlab-compatible 2-D array."""
    # 创建一个 numpy 数组，并根据输入参数调整其形状为 Matlab 兼容的二维数组
    arr = np.array(*args, **kwargs)
    arr.shape = matdims(arr)
    return arr


# 定义测试用例
theta = np.pi/4*np.arange(9,dtype=float).reshape(1,9)
case_table4 = [
    {'name': 'double',
     'classes': {'testdouble': 'double'},
     'expected': {'testdouble': theta}
     }]
case_table4.append(
    {'name': 'string',
     'classes': {'teststring': 'char'},
     'expected': {'teststring':
                  array(['"Do nine men interpret?" "Nine men," I nod.'])}
     })
case_table4.append(
    {'name': 'complex',
     'classes': {'testcomplex': 'double'},
     'expected': {'testcomplex': np.cos(theta) + 1j*np.sin(theta)}
     })
A = np.zeros((3,5))
A[0] = list(range(1,6))
A[:,0] = list(range(1,4))
case_table4.append(
    {'name': 'matrix',
     'classes': {'testmatrix': 'double'},
     'expected': {'testmatrix': A},
     })
case_table4.append(
    {'name': 'sparse',
     'classes': {'testsparse': 'sparse'},
     'expected': {'testsparse': SP.coo_matrix(A)},
     })
B = A.astype(complex)
B[0,0] += 1j
case_table4.append(
    {'name': 'sparsecomplex',
     'classes': {'testsparsecomplex': 'sparse'},
     'expected': {'testsparsecomplex': SP.coo_matrix(B)},
     })
case_table4.append(
    {'name': 'multi',
     'classes': {'theta': 'double', 'a': 'double'},
     'expected': {'theta': theta, 'a': A},
     })
case_table4.append(
    {'name': 'minus',
     'classes': {'testminus': 'double'},
     'expected': {'testminus': mlarr(-1)},
     })
case_table4.append(
    {'name': 'onechar',
     'classes': {'testonechar': 'char'},
     'expected': {'testonechar': array(['r'])},
     })
# Cell arrays stored as object arrays
CA = mlarr((  # tuple for object array creation
        [],
        mlarr([1]),
        mlarr([[1,2]]),
        mlarr([[1,2,3]])), dtype=object).reshape(1,-1)
# 创建一个元组 CA，其中包含不同形状的 mlarr 对象（多维列表），并指定数据类型为 object，然后将其形状重塑为 (1, -1)

CA[0,0] = array(
    ['This cell contains this string and 3 arrays of increasing length'])
# 在 CA 的第一个元素位置插入一个包含字符串和三个逐渐增长长度数组的数组对象

case_table5 = [
    {'name': 'cell',
     'classes': {'testcell': 'cell'},
     'expected': {'testcell': CA}}]
# 创建一个名为 case_table5 的列表，其中包含一个字典，用于测试 cell 类型，预期结果是一个包含 CA 的字典

CAE = mlarr((  # tuple for object array creation
    mlarr(1),
    mlarr(2),
    mlarr([]),
    mlarr([]),
    mlarr(3)), dtype=object).reshape(1,-1)
# 创建一个元组 CAE，其中包含不同形状的 mlarr 对象（多维列表），指定数据类型为 object，然后将其形状重塑为 (1, -1)

objarr = np.empty((1,1),dtype=object)
objarr[0,0] = mlarr(1)
# 创建一个形状为 (1, 1) 的空对象数组 objarr，将其第一个元素设置为一个 mlarr 对象，其内容为 1

case_table5.append(
    {'name': 'scalarcell',
     'classes': {'testscalarcell': 'cell'},
     'expected': {'testscalarcell': objarr}
     })
# 在 case_table5 列表末尾追加一个字典，用于测试 scalarcell 类型，预期结果是一个包含 objarr 的字典

case_table5.append(
    {'name': 'emptycell',
     'classes': {'testemptycell': 'cell'},
     'expected': {'testemptycell': CAE}})
# 在 case_table5 列表末尾追加一个字典，用于测试 emptycell 类型，预期结果是一个包含 CAE 的字典

case_table5.append(
    {'name': 'stringarray',
     'classes': {'teststringarray': 'char'},
     'expected': {'teststringarray': array(
    ['one  ', 'two  ', 'three'])},
     })
# 在 case_table5 列表末尾追加一个字典，用于测试 stringarray 类型，预期结果是一个包含字符串数组的字典

case_table5.append(
    {'name': '3dmatrix',
     'classes': {'test3dmatrix': 'double'},
     'expected': {
    'test3dmatrix': np.transpose(np.reshape(list(range(1,25)), (4,3,2)))}
     })
# 在 case_table5 列表末尾追加一个字典，用于测试 3dmatrix 类型，预期结果是一个包含 3D 矩阵的字典

st_sub_arr = array([np.sqrt(2),np.exp(1),np.pi]).reshape(1,3)
# 创建一个形状为 (1, 3) 的数组 st_sub_arr，其中包含三个数值，然后将其形状重塑为 (1, 3)

dtype = [(n, object) for n in ['stringfield', 'doublefield', 'complexfield']]
# 创建一个结构化数据类型 dtype，包含三个字段名和类型为 object

st1 = np.zeros((1,1), dtype)
# 创建一个形状为 (1, 1) 的零填充数组 st1，数据类型为 dtype

st1['stringfield'][0,0] = array(['Rats live on no evil star.'])
# 将 st1 的 stringfield 字段的第一个元素设置为包含字符串的数组对象

st1['doublefield'][0,0] = st_sub_arr
# 将 st1 的 doublefield 字段的第一个元素设置为之前创建的 st_sub_arr 数组对象

st1['complexfield'][0,0] = st_sub_arr * (1 + 1j)
# 将 st1 的 complexfield 字段的第一个元素设置为 st_sub_arr 数组对象乘以复数单位 (1 + 1j)

case_table5.append(
    {'name': 'struct',
     'classes': {'teststruct': 'struct'},
     'expected': {'teststruct': st1}
     })
# 在 case_table5 列表末尾追加一个字典，用于测试 struct 类型，预期结果是一个包含 st1 的字典

CN = np.zeros((1,2), dtype=object)
# 创建一个形状为 (1, 2) 的零填充数组 CN，数据类型为 object

CN[0,0] = mlarr(1)
# 将 CN 的第一个元素位置设置为包含整数 1 的 mlarr 对象

CN[0,1] = np.zeros((1,3), dtype=object)
# 将 CN 的第二个元素位置设置为形状为 (1, 3) 的零填充数组，数据类型为 object

CN[0,1][0,0] = mlarr(2, dtype=np.uint8)
# 将 CN 的第二个元素的第一个元素位置设置为包含整数 2 的 mlarr 对象，数据类型为 uint8

CN[0,1][0,1] = mlarr([[3]], dtype=np.uint8)
# 将 CN 的第二个元素的第二个元素位置设置为包含整数 3 的 mlarr 对象，数据类型为 uint8

CN[0,1][0,2] = np.zeros((1,2), dtype=object)
# 将 CN 的第二个元素的第三个元素位置设置为形状为 (1, 2) 的零填充数组，数据类型为 object

CN[0,1][0,2][0,0] = mlarr(4, dtype=np.uint8)
# 将 CN 的第二个元素的第三个元素的第一个元素位置设置为包含整数 4 的 mlarr 对象，数据类型为 uint8

CN[0,1][0,2][0,1] = mlarr(5, dtype=np.uint8)
# 将 CN 的第二个元素的第三个元素的第二个元素位置设置为包含整数 5 的 mlarr 对象，数据类型为 uint8

case_table5.append(
    {'name': 'cellnest',
     'classes': {'testcellnest': 'cell'},
     'expected': {'testcellnest': CN},
     })
# 在 case_table5 列表末尾追加一个字典，用于测试 cellnest 类型，预期结果是一个包含 CN 的字典

st2 = np.empty((1,1), dtype=[(n, object) for n in ['one', 'two']])
# 创建一个形状为 (1, 1) 的空对象数组 st2，其中包含两个字段名和类型为 object 的结构化数据类型

st2[0,0]['one'] = mlarr(1)
# 将 st2 的 one 字段的第一个元素设置为包含整数 1 的 mlarr 对象

st2[0,0]['two'] = np.empty((1,1), dtype=[('three', object)])
# 将 st2 的 two 字段的第一个元素设置为一个空对象数组，其数据类型为包含字段名 'three' 和类型为 object 的结构化数据类型

st2[0,0]['two'][0,0]['three'] = array(['number 3'])
# 将 st2 的 two 字段的第一个元素的 three 字段的第一个元素设置为包含字符串的数组对象

case_table5.append(
    {'name': 'structnest',
     'classes': {'teststructnest': 'struct'},
     'expected': {'teststructnest': st2}
     })
# 在 case_table5 列表末尾追加一个字典，用于测试 structnest 类型，预期结果是一个包含 st2 的字典

a = np.empty((1,2), dtype=[(n, object) for n in ['one', 'two']])
# 创建一个形状为 (1, 2) 的空对象数组 a，其中包含两个字段名和类型为 object 的结构化数据类型

a[0,0]['one'] = mlarr(
m0['expr'] = array(['x'])
# 将字符串数组 ['x'] 赋值给 m0 字典的键 'expr'

m0['inputExpr'] = array([' x = INLINE_INPUTS_{1};'])
# 将字符串数组 [' x = INLINE_INPUTS_{1};'] 赋值给 m0 字典的键 'inputExpr'

m0['args'] = array(['x'])
# 将字符串数组 ['x'] 赋值给 m0 字典的键 'args'

m0['isEmpty'] = mlarr(0)
# 调用 mlarr 函数创建一个表示空值的对象，将其赋值给 m0 字典的键 'isEmpty'

m0['numArgs'] = mlarr(1)
# 调用 mlarr 函数创建一个表示数值 1 的对象，将其赋值给 m0 字典的键 'numArgs'

m0['version'] = mlarr(1)
# 调用 mlarr 函数创建一个表示数值 1 的对象，将其赋值给 m0 字典的键 'version'

case_table5.append(
    {'name': 'object',
     'classes': {'testobject': 'object'},
     'expected': {'testobject': MO}
     })
# 向列表 case_table5 中添加一个字典，包含 'name' 键为 'object'，'classes' 键为 {'testobject': 'object'}，'expected' 键为 {'testobject': MO}

fp_u_str = open(pjoin(test_data_path, 'japanese_utf8.txt'), 'rb')
# 打开 'japanese_utf8.txt' 文件，以二进制读取模式，并赋值给变量 fp_u_str

u_str = fp_u_str.read().decode('utf-8')
# 读取 fp_u_str 中的内容，并使用 UTF-8 解码为字符串，赋值给变量 u_str

fp_u_str.close()
# 关闭文件对象 fp_u_str

case_table5.append(
    {'name': 'unicode',
     'classes': {'testunicode': 'char'},
     'expected': {'testunicode': array([u_str])}
     })
# 向列表 case_table5 中添加一个字典，包含 'name' 键为 'unicode'，'classes' 键为 {'testunicode': 'char'}，'expected' 键为 {'testunicode': array([u_str])}

case_table5.append(
    {'name': 'sparse',
     'classes': {'testsparse': 'sparse'},
     'expected': {'testsparse': SP.coo_matrix(A)},
     })
# 向列表 case_table5 中添加一个字典，包含 'name' 键为 'sparse'，'classes' 键为 {'testsparse': 'sparse'}，'expected' 键为 {'testsparse': SP.coo_matrix(A)}

case_table5.append(
    {'name': 'sparsecomplex',
     'classes': {'testsparsecomplex': 'sparse'},
     'expected': {'testsparsecomplex': SP.coo_matrix(B)},
     })
# 向列表 case_table5 中添加一个字典，包含 'name' 键为 'sparsecomplex'，'classes' 键为 {'testsparsecomplex': 'sparse'}，'expected' 键为 {'testsparsecomplex': SP.coo_matrix(B)}

case_table5.append(
    {'name': 'bool',
     'classes': {'testbools': 'logical'},
     'expected': {'testbools':
                  array([[True], [False]])},
     })
# 向列表 case_table5 中添加一个字典，包含 'name' 键为 'bool'，'classes' 键为 {'testbools': 'logical'}，'expected' 键为 {'testbools': array([[True], [False]])}

case_table5_rt = case_table5[:]
# 将 case_table5 列表复制给 case_table5_rt

case_table5_rt.append(
    {'name': 'objectarray',
     'classes': {'testobjectarray': 'object'},
     'expected': {'testobjectarray': np.repeat(MO, 2).reshape(1,2)}})
# 向列表 case_table5_rt 中添加一个字典，包含 'name' 键为 'objectarray'，'classes' 键为 {'testobjectarray': 'object'}，'expected' 键为 {'testobjectarray': np.repeat(MO, 2).reshape(1,2)}

def types_compatible(var1, var2):
    """Check if types are same or compatible.

    0-D numpy scalars are compatible with bare python scalars.
    """
    type1 = type(var1)
    type2 = type(var2)
    if type1 is type2:
        return True
    if type1 is np.ndarray and var1.shape == ():
        return type(var1.item()) is type2
    if type2 is np.ndarray and var2.shape == ():
        return type(var2.item()) is type1
    return False
# 定义函数 types_compatible，用于检查两个变量的类型是否相同或兼容

def _check_level(label, expected, actual):
    """ Check one level of a potentially nested array """
    if SP.issparse(expected):  # allow different types of sparse matrices
        assert_(SP.issparse(actual))
        assert_array_almost_equal(actual.toarray(),
                                  expected.toarray(),
                                  err_msg=label,
                                  decimal=5)
        return
    # Check types are as expected
    assert_(types_compatible(expected, actual),
            f"Expected type {type(expected)}, got {type(actual)} at {label}")
    # A field in a record array may not be an ndarray
    # A scalar from a record array will be type np.void
    if not isinstance(expected,
                      (np.void, np.ndarray, MatlabObject)):
        assert_equal(expected, actual)
        return
    # This is an ndarray-like thing
    assert_(expected.shape == actual.shape,
            msg=f'Expected shape {expected.shape}, got {actual.shape} at {label}')
    ex_dtype = expected.dtype
    # 如果期望数据类型中有对象类型：处理对象数组的情况
    if ex_dtype.hasobject:  # array of objects
        # 如果期望值是 MatlabObject 类型，断言其类名与实际值相等
        if isinstance(expected, MatlabObject):
            assert_equal(expected.classname, actual.classname)
        # 遍历期望值数组，逐个检查每个对象
        for i, ev in enumerate(expected):
            # 构建当前层级的标签，格式为 "label, [i], "
            level_label = "%s, [%d], " % (label, i)
            # 递归调用 _check_level 函数，比较期望值和实际值的当前层级
            _check_level(level_label, ev, actual[i])
        return
    # 如果期望数据类型中有字段（可能是记录数组）
    if ex_dtype.fields:  # probably recarray
        # 遍历期望数据类型中的字段名
        for fn in ex_dtype.fields:
            # 构建当前层级的标签，格式为 "label, field fn, "
            level_label = f"{label}, field {fn}, "
            # 递归调用 _check_level 函数，比较期望值字段和实际值字段的当前层级
            _check_level(level_label,
                         expected[fn], actual[fn])
        return
    # 如果期望数据类型是字符串或布尔类型
    if ex_dtype.type in (str,  # string or bool
                         np.str_,
                         np.bool_):
        # 断言实际值与期望值相等，使用给定的错误消息标签
        assert_equal(actual, expected, err_msg=label)
        return
    # 如果是数值类型的情况
    # 断言实际值数组与期望值数组几乎相等，使用给定的错误消息标签和小数位数（5 位）
    assert_array_almost_equal(actual, expected, err_msg=label, decimal=5)
# 定义一个函数，用于加载和检查特定测试用例中的数据
def _load_check_case(name, files, case):
    # 遍历传入的文件列表
    for file_name in files:
        # 使用`loadmat`函数加载MATLAB格式的文件数据，并转换成字典
        matdict = loadmat(file_name, struct_as_record=True)
        # 设置当前文件的标签，用于错误断言信息
        label = f"test {name}; file {file_name}"
        # 遍历传入的测试用例字典
        for k, expected in case.items():
            # 设置当前变量的标签，用于错误断言信息
            k_label = f"{label}, variable {k}"
            # 断言当前变量在加载的MATLAB数据中存在
            assert_(k in matdict, "Missing key at %s" % k_label)
            # 调用内部函数检查变量值的层次结构和期望值是否一致
            _check_level(k_label, expected, matdict[k])


# 定义一个函数，用于加载和检查特定测试用例中的whosmat结果
def _whos_check_case(name, files, case, classes):
    # 遍历传入的文件列表
    for file_name in files:
        # 设置当前文件的标签，用于错误断言信息
        label = f"test {name}; file {file_name}"
        # 使用`whosmat`函数获取MATLAB格式文件的变量信息列表
        whos = whosmat(file_name)
        # 生成期望的变量信息列表，包括变量名、形状和类别
        expected_whos = [
            (k, expected.shape, classes[k]) for k, expected in case.items()]
        # 对实际和期望的变量信息列表进行排序
        whos.sort()
        expected_whos.sort()
        # 断言实际的变量信息列表与期望的变量信息列表相等
        assert_equal(whos, expected_whos,
                     f"{label}: {whos!r} != {expected_whos!r}"
                     )


# 定义一个函数，用于执行往返测试（round trip tests）
def _rt_check_case(name, expected, format):
    # 创建一个内存中的MATLAB数据流对象
    mat_stream = BytesIO()
    # 将预期的数据保存到MATLAB数据流中，使用指定的格式
    savemat(mat_stream, expected, format=format)
    # 将数据流的指针位置重置到起始位置
    mat_stream.seek(0)
    # 调用_load_check_case函数来加载和检查数据流中的数据
    _load_check_case(name, [mat_stream], expected)


# 生成测试用例的生成器函数
def _cases(version, filt='test%(name)s_*.mat'):
    # 根据版本选择相应的测试用例集合
    if version == '4':
        cases = case_table4
    elif version == '5':
        cases = case_table5
    else:
        assert version == '5_rt'
        cases = case_table5_rt
    # 遍历选定版本的测试用例集合
    for case in cases:
        # 获取测试用例的名称、预期输出、文件筛选模式和类别信息
        name = case['name']
        expected = case['expected']
        # 根据筛选模式生成文件列表，如果筛选结果为空则报错
        if filt is None:
            files = None
        else:
            use_filt = pjoin(test_data_path, filt % dict(name=name))
            files = glob(use_filt)
            assert len(files) > 0, \
                f"No files for test {name} using filter {filt}"
        classes = case['classes']
        # 返回生成的测试用例的名称、文件列表、预期输出和类别信息
        yield name, files, expected, classes


# 使用pytest的参数化装饰器定义加载测试函数
@pytest.mark.parametrize('version', ('4', '5'))
def test_load(version):
    # 遍历生成的测试用例集合，并依次执行_load_check_case函数
    for case in _cases(version):
        _load_check_case(*case[:3])


# 使用pytest的参数化装饰器定义whosmat测试函数
@pytest.mark.parametrize('version', ('4', '5'))
def test_whos(version):
    # 遍历生成的测试用例集合，并依次执行_whos_check_case函数
    for case in _cases(version):
        _whos_check_case(*case)


# 使用pytest的参数化装饰器定义往返测试函数
@pytest.mark.parametrize('version, fmts', [
    ('4', ['4', '5']),
    ('5_rt', ['5']),
])
def test_round_trip(version, fmts):
    # 遍历生成的测试用例集合和指定的MATLAB格式，并依次执行_rt_check_case函数
    for case in _cases(version, filt=None):
        for fmt in fmts:
            _rt_check_case(case[0], case[2], fmt)


# 定义一个简单的GZIP压缩格式的测试函数
def test_gzip_simple():
    # 创建一个稀疏矩阵对象，并设置其中的非零元素值
    xdense = np.zeros((20,20))
    xdense[2,3] = 2.3
    xdense[4,5] = 4.5
    x = SP.csc_matrix(xdense)

    # 设置测试名称、预期输出和MATLAB格式
    name = 'gzip_test'
    expected = {'x':x}
    format = '4'

    # 创建一个临时目录用于保存测试文件
    tmpdir = mkdtemp()
    try:
        fname = pjoin(tmpdir,name)
        # 打开并写入GZIP格式的MATLAB数据流
        mat_stream = gzip.open(fname, mode='wb')
        savemat(mat_stream, expected, format=format)
        mat_stream.close()

        # 打开并读取GZIP格式的MATLAB数据流，并加载其中的数据
        mat_stream = gzip.open(fname, mode='rb')
        actual = loadmat(mat_stream, struct_as_record=True)
        mat_stream.close()
    finally:
        # 最终清理临时目录及其内容
        shutil.rmtree(tmpdir)
    # 断言：比较两个稀疏矩阵的数组表示，确保它们几乎相等。
    assert_array_almost_equal(
        # 获取实际输出中 'x' 对应的稀疏矩阵的数组表示，并转换为普通数组进行比较
        actual['x'].toarray(),
        # 获取期望输出中 'x' 对应的稀疏矩阵的数组表示，并转换为普通数组进行比较
        expected['x'].toarray(),
        # 如果断言失败，将实际输出的表示形式作为错误消息的一部分
        err_msg=repr(actual)
    )
# 测试多次打开文件是否会正确关闭，特别针对 Windows 平台下的问题，避免文件未正确关闭的情况发生
def test_multiple_open():
    # 创建一个临时目录用于测试
    tmpdir = mkdtemp()
    try:
        # 创建一个包含数据的字典
        x = dict(x=np.zeros((2, 2)))

        # 在临时目录下创建一个 mat 文件路径
        fname = pjoin(tmpdir, "a.mat")

        # 检查文件是否被正确关闭
        savemat(fname, x)  # 将数据保存到 mat 文件中
        os.unlink(fname)    # 删除 mat 文件
        savemat(fname, x)   # 再次保存数据到 mat 文件
        loadmat(fname)      # 加载 mat 文件
        os.unlink(fname)    # 再次删除 mat 文件

        # 检查流是否被正确关闭
        f = open(fname, 'wb')   # 以写入二进制模式打开文件
        savemat(f, x)           # 将数据保存到文件流中
        f.seek(0)               # 将文件指针移动到文件开头
        f.close()               # 关闭文件流

        f = open(fname, 'rb')   # 以读取二进制模式重新打开文件
        loadmat(f)              # 从文件流中加载数据
        f.seek(0)               # 将文件指针移动到文件开头
        f.close()               # 关闭文件流
    finally:
        shutil.rmtree(tmpdir)   # 最终删除临时目录及其内容


# 测试 mat73 格式文件是否会引发错误
def test_mat73():
    # 获取所有符合条件的 hdf5 文件名列表
    filenames = glob(
        pjoin(test_data_path, 'testhdf5*.mat'))
    assert_(len(filenames) > 0)  # 断言至少有一个文件符合条件
    for filename in filenames:
        fp = open(filename, 'rb')  # 以只读二进制模式打开文件
        assert_raises(NotImplementedError,
                      loadmat,
                      fp,
                      struct_as_record=True)  # 断言加载文件时会引发 NotImplementedError 异常
        fp.close()  # 关闭文件


# 测试警告行为
def test_warnings():
    # 这个测试反映了先前的行为，即如果用户在 Python 系统路径上搜索 mat 文件，则会发出警告
    fname = pjoin(test_data_path, 'testdouble_7.1_GLNX86.mat')
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # 设置警告为错误级别
        # 这里不应该生成警告
        loadmat(fname, struct_as_record=True)  # 加载 mat 文件，结构化记录设置为 True
        # 这里也不应该生成警告
        loadmat(fname, struct_as_record=False)  # 加载 mat 文件，结构化记录设置为 False


# 测试回归问题 #653
def test_regression_653():
    # 保存一个只有无效键的字典曾经会引发错误，现在将其保存为 Matlab 空间中的空结构体
    sio = BytesIO()
    savemat(sio, {'d':{1:2}}, format='5')  # 将数据保存为 mat 文件格式 5
    back = loadmat(sio)['d']  # 加载 mat 文件中的数据 'd'
    # 检查是否获得了等效的空结构体
    assert_equal(back.shape, (1,1))  # 断言结构体形状为 (1,1)
    assert_equal(back.dtype, np.dtype(object))  # 断言结构体数据类型为 object
    assert_(back[0,0] is None)  # 断言结构体中的元素为 None


# 测试结构体字段名长度限制
def test_structname_len():
    # 测试结构体字段名长度的限制
    lim = 31
    fldname = 'a' * lim
    st1 = np.zeros((1,1), dtype=[(fldname, object)])  # 创建一个结构体
    savemat(BytesIO(), {'longstruct': st1}, format='5')  # 将结构体保存为 mat 文件格式 5
    fldname = 'a' * (lim+1)
    st1 = np.zeros((1,1), dtype=[(fldname, object)])  # 创建一个超过长度限制的结构体字段名
    assert_raises(ValueError, savemat, BytesIO(),
                  {'longstruct': st1}, format='5')  # 断言保存超长字段名的结构体时会引发 ValueError 异常


# 测试长字段名选项是否支持
def test_4_and_long_field_names_incompatible():
    # 在格式 4 中不支持长字段名选项
    my_struct = np.zeros((1,1),dtype=[('my_fieldname',object)])
    assert_raises(ValueError, savemat, BytesIO(),
                  {'my_struct':my_struct}, format='4', long_field_names=True)  # 断言在格式 4 中使用长字段名选项会引发 ValueError 异常


# 测试长字段名的长度限制
def test_long_field_names():
    # 测试结构体字段名长度的限制
    lim = 63
    fldname = 'a' * lim
    st1 = np.zeros((1,1), dtype=[(fldname, object)])  # 创建一个结构体
    savemat(BytesIO(), {'longstruct': st1}, format='5', long_field_names=True)  # 将结构体保存为 mat 文件格式 5，使用长字段名选项
    fldname = 'a' * (lim+1)
    # 创建一个形状为 (1, 1) 的 NumPy 数组，元素类型是结构化数组，结构中有一个字段名为 fldname，每个元素是对象类型
    st1 = np.zeros((1, 1), dtype=[(fldname, object)])
    # 断言语句，验证 savemat 函数在以下情况下会引发 ValueError 异常：
    # 尝试将一个包含结构化数组 st1 的字典 {'longstruct': st1} 保存到一个空的字节流中，使用格式 '5'，并开启了长字段名选项
    assert_raises(ValueError, savemat, BytesIO(),
                  {'longstruct': st1}, format='5', long_field_names=True)
# 回归测试 - 如果将结构体嵌套在结构体中，则 long_field_names 会被擦除
def test_long_field_names_in_struct():
    lim = 63  # 定义一个边界值为 63
    fldname = 'a' * lim  # 创建一个长度为 lim 的字符串作为字段名
    cell = np.ndarray((1,2), dtype=object)  # 创建一个形状为 (1,2) 的对象数组
    st1 = np.zeros((1,1), dtype=[(fldname, object)])  # 创建一个 dtype 为 [(fldname, object)] 的全零数组
    cell[0,0] = st1  # 将 st1 存入 cell 的第一个元素
    cell[0,1] = st1  # 将 st1 存入 cell 的第二个元素
    savemat(BytesIO(), {'longstruct': cell}, format='5', long_field_names=True)
    #
    # 检查确保 long_field_names 关闭时会失败
    #
    assert_raises(ValueError, savemat, BytesIO(),
                  {'longstruct': cell}, format='5', long_field_names=False)


# 回归测试 - 创建一个 1 x 2 的 cell 数组并放入两个字符串，以前的版本可能无法正常工作
def test_cell_with_one_thing_in_it():
    cells = np.ndarray((1,2), dtype=object)  # 创建一个形状为 (1,2) 的对象数组
    cells[0,0] = 'Hello'  # 在第一个位置放入字符串 'Hello'
    cells[0,1] = 'World'  # 在第二个位置放入字符串 'World'
    savemat(BytesIO(), {'x': cells}, format='5')

    cells = np.ndarray((1,1), dtype=object)  # 创建一个形状为 (1,1) 的对象数组
    cells[0,0] = 'Hello, world'  # 在唯一的位置放入字符串 'Hello, world'
    savemat(BytesIO(), {'x': cells}, format='5')


# 测试写入器属性的设置和获取
def test_writer_properties():
    mfw = MatFile5Writer(BytesIO())  # 创建一个 MatFile5Writer 对象
    assert_equal(mfw.global_vars, [])  # 检查 global_vars 是否为空列表
    mfw.global_vars = ['avar']  # 设置 global_vars 属性为 ['avar']
    assert_equal(mfw.global_vars, ['avar'])  # 检查 global_vars 属性是否为 ['avar']
    assert_equal(mfw.unicode_strings, False)  # 检查 unicode_strings 属性是否为 False
    mfw.unicode_strings = True  # 设置 unicode_strings 属性为 True
    assert_equal(mfw.unicode_strings, True)  # 检查 unicode_strings 属性是否为 True
    assert_equal(mfw.long_field_names, False)  # 检查 long_field_names 属性是否为 False
    mfw.long_field_names = True  # 设置 long_field_names 属性为 True
    assert_equal(mfw.long_field_names, True)  # 检查 long_field_names 属性是否为 True


# 测试是否使用小数据元素
def test_use_small_element():
    sio = BytesIO()  # 创建一个 BytesIO 对象
    wtr = MatFile5Writer(sio)  # 创建一个 MatFile5Writer 对象，使用 sio 作为输出流
    # 首先检查无名称情况下的大小
    arr = np.zeros(10)  # 创建一个长度为 10 的全零数组
    wtr.put_variables({'aaaaa': arr})  # 将数组 arr 写入 MatFile5Writer 对象，使用 'aaaaa' 作为变量名
    w_sz = len(sio.getvalue())  # 获取当前流的大小
    # 检查小名称是否导致大小差异较大
    sio.truncate(0)  # 清空流
    sio.seek(0)  # 将流指针重置到起始位置
    wtr.put_variables({'aaaa': arr})  # 将数组 arr 写入 MatFile5Writer 对象，使用 'aaaa' 作为变量名
    assert_(w_sz - len(sio.getvalue()) > 4)  # 检查大小是否减少了大于4个字节
    # 增加名称大小会使差异变小
    sio.truncate(0)  # 清空流
    sio.seek(0)  # 将流指针重置到起始位置
    wtr.put_variables({'aaaaaa': arr})  # 将数组 arr 写入 MatFile5Writer 对象，使用 'aaaaaa' 作为变量名
    assert_(len(sio.getvalue()) - w_sz < 4)  # 检查大小是否增加了少于4个字节


# 测试保存字典
def test_save_dict():
    # 测试能够保存 dict 和 OrderedDict（作为 recarray），加载为 matstruct 并保持顺序
    ab_exp = np.array([[(1, 2)]], dtype=[('a', object), ('b', object)])  # 创建预期的数组 ab_exp
    for dict_type in (dict, OrderedDict):  # 循环遍历 dict 和 OrderedDict
        # 使用元组初始化以保持顺序
        d = dict_type([('a', 1), ('b', 2)])  # 创建一个字典 d
        stream = BytesIO()  # 创建一个 BytesIO 对象
        savemat(stream, {'dict': d})  # 将字典 d 保存到流中，使用 'dict' 作为变量名
        stream.seek(0)  # 将流指针重置到起始位置
        vals = loadmat(stream)['dict']  # 从流中加载 'dict' 变量的值到 vals
        assert_equal(vals.dtype.names, ('a', 'b'))  # 检查 vals 的 dtype 是否为 ('a', 'b')
        assert_array_equal(vals, ab_exp)  # 检查 vals 是否等于预期的 ab_exp 数组


# 新 5 版本的行为是 1D 数组会被处理为行向量
def test_1d_shape():
    arr = np.arange(5)  # 创建一个长度为 5 的数组
    for format in ('4', '5'):
        # 针对每个格式进行循环处理，格式为字符串 '4' 和 '5'
        
        # 使用 BytesIO 创建一个字节流对象，用于保存数据
        stream = BytesIO()
        
        # 将数组 arr 保存到 stream 中，使用指定的格式进行保存
        savemat(stream, {'oned': arr}, format=format)
        
        # 从 stream 中加载数据
        vals = loadmat(stream)
        
        # 断言加载的数据 'oned' 的形状为 (1, 5)
        assert_equal(vals['oned'].shape, (1, 5))
        
        # 可以显式地指定 'oned_as' 为 'column'
        stream = BytesIO()
        savemat(stream, {'oned': arr},
                format=format,
                oned_as='column')
        vals = loadmat(stream)
        
        # 断言加载的数据 'oned' 的形状为 (5, 1)
        assert_equal(vals['oned'].shape, (5, 1))
        
        # 但是与 'row' 显式指定不同
        stream = BytesIO()
        savemat(stream, {'oned': arr},
                format=format,
                oned_as='row')
        vals = loadmat(stream)
        
        # 断言加载的数据 'oned' 的形状为 (1, 5)
        assert_equal(vals['oned'].shape, (1, 5))
def test_compression():
    # 创建一个 100 元素的全零数组，并将其重塑为 5x20 的数组
    arr = np.zeros(100).reshape((5,20))
    # 将数组中索引为 (2,10) 的元素设为 1
    arr[2,10] = 1
    # 创建一个字节流对象
    stream = BytesIO()
    # 将数组保存到 MAT 文件格式的流中
    savemat(stream, {'arr':arr})
    # 计算未压缩时流的长度
    raw_len = len(stream.getvalue())
    # 从流中加载数据到 vals 中
    vals = loadmat(stream)
    # 断言加载的数据与原始数组 arr 相等
    assert_array_equal(vals['arr'], arr)
    
    # 创建一个新的字节流对象
    stream = BytesIO()
    # 将数组以压缩形式保存到 MAT 文件格式的流中
    savemat(stream, {'arr':arr}, do_compression=True)
    # 计算压缩后流的长度
    compressed_len = len(stream.getvalue())
    # 从流中加载数据到 vals 中
    vals = loadmat(stream)
    # 断言加载的数据与原始数组 arr 相等
    assert_array_equal(vals['arr'], arr)
    # 断言压缩后的长度比原始长度要小
    assert_(raw_len > compressed_len)
    
    # 复制 arr 数组，并修改复制后数组的第一个元素为 1
    arr2 = arr.copy()
    arr2[0,0] = 1
    # 创建一个新的字节流对象
    stream = BytesIO()
    # 将两个数组 arr 和 arr2 以不压缩形式保存到 MAT 文件格式的流中
    savemat(stream, {'arr':arr, 'arr2':arr2}, do_compression=False)
    # 从流中加载数据到 vals 中
    vals = loadmat(stream)
    # 断言加载的数据与 arr2 数组相等
    assert_array_equal(vals['arr2'], arr2)
    
    # 创建一个新的字节流对象
    stream = BytesIO()
    # 将两个数组 arr 和 arr2 以压缩形式保存到 MAT 文件格式的流中
    savemat(stream, {'arr':arr, 'arr2':arr2}, do_compression=True)
    # 从流中加载数据到 vals 中
    vals = loadmat(stream)
    # 断言加载的数据与 arr2 数组相等
    assert_array_equal(vals['arr2'], arr2)


def test_single_object():
    # 创建一个字节流对象
    stream = BytesIO()
    # 将包含一个整数对象的字典保存到 MAT 文件格式的流中
    savemat(stream, {'A':np.array(1, dtype=object)})


def test_skip_variable():
    # 测试跳过 MAT 文件中两个变量的第一个变量，使用 mat_reader_factory 和 put_variables 进行读取
    #
    # 这是一个回归测试，检查使用压缩文件读取器 seek 而非原始文件 I/O seek 在跳过压缩块时可能引起的问题。
    #
    # 当块很大时会出现问题：本文件包含一个 256x256 的随机（不可压缩）双精度数组。
    #
    filename = pjoin(test_data_path,'test_skip_variable.mat')
    #
    # 使用 loadmat 证明可以加载文件
    #
    d = loadmat(filename, struct_as_record=True)
    assert_('first' in d)
    assert_('second' in d)
    #
    # 创建工厂
    #
    factory, file_opened = mat_reader_factory(filename, struct_as_record=True)
    #
    # 这是工厂在 MatMatrixGetter.to_next 中出错的地方
    #
    d = factory.get_variables('second')
    assert_('second' in d)
    factory.mat_stream.close()


def test_empty_struct():
    # ticket 885
    filename = pjoin(test_data_path,'test_empty_struct.mat')
    # 在修复问题前，这会因为数据类型为空而导致 ValueError
    d = loadmat(filename, struct_as_record=True)
    a = d['a']
    assert_equal(a.shape, (1,1))
    assert_equal(a.dtype, np.dtype(object))
    assert_(a[0,0] is None)
    # 创建一个字节流对象
    stream = BytesIO()
    # 创建一个空的 Unicode 字符串数组，并保存到 MAT 文件格式的流中
    arr = np.array((), dtype='U')
    # 在修复问题前，这会导致数据类型不被理解的错误
    savemat(stream, {'arr':arr})
    d = loadmat(stream)
    a2 = d['arr']
    # 断言加载的数组与 arr 相等
    assert_array_equal(a2, arr)


def test_save_empty_dict():
    # 保存空字典也会得到空结构
    # 创建一个字节流对象
    stream = BytesIO()
    # 将空字典保存到 MAT 文件格式的流中
    savemat(stream, {'arr': {}})
    d = loadmat(stream)
    a = d['arr']
    assert_equal(a.shape, (1,1))
    assert_equal(a.dtype, np.dtype(object))
    assert_(a[0,0] is None)
    # 对于每个预期输出进行迭代检查
    for expected in alternatives:
        # 检查输出是否与当前预期输出数组相等
        if np.all(output == expected):
            # 如果找到匹配的预期输出，则设置标志为True并退出循环
            one_equal = True
            break
    # 使用断言确保至少找到一个匹配的预期输出
    assert_(one_equal)
# 定义一个测试函数，用于测试 to_writeable 函数
def test_to_writeable():
    # Test to_writeable function
    # 测试传入 np.array 对象时的返回结果
    res = to_writeable(np.array([1]))  # pass through ndarrays
    # 断言返回结果的形状为 (1,)
    assert_equal(res.shape, (1,))
    # 断言返回结果与预期值相等
    assert_array_equal(res, 1)
    
    # 测试传入字典对象时，字段的顺序可以是任意的
    expected1 = np.array([(1, 2)], dtype=[('a', '|O8'), ('b', '|O8')])
    expected2 = np.array([(2, 1)], dtype=[('b', '|O8'), ('a', '|O8')])
    alternatives = (expected1, expected2)
    assert_any_equal(to_writeable({'a':1,'b':2}), alternatives)
    
    # 测试传入字典对象时，忽略带下划线的字段
    assert_any_equal(to_writeable({'a':1,'b':2, '_c':3}), alternatives)
    
    # 测试传入字典对象时，忽略非字符串键的字段
    assert_any_equal(to_writeable({'a':1,'b':2, 100:3}), alternatives)
    
    # 测试传入字典对象时，忽略作为 Python 合法标识符的字符串字段
    assert_any_equal(to_writeable({'a':1,'b':2, '99':3}), alternatives)
    
    # 测试传入对象实例时，返回结果与预期值相等
    class klass:
        pass
    c = klass
    c.a = 1
    c.b = 2
    assert_any_equal(to_writeable(c), alternatives)
    
    # 测试传入空列表时，返回结果的形状为 (0,)
    res = to_writeable([])
    assert_equal(res.shape, (0,))
    # 断言返回结果的数据类型为 np.float64
    assert_equal(res.dtype.type, np.float64)
    
    # 测试传入空元组时，返回结果的形状为 (0,)
    res = to_writeable(())
    assert_equal(res.shape, (0,))
    # 断言返回结果的数据类型为 np.float64
    assert_equal(res.dtype.type, np.float64)
    
    # 测试传入 None 时，返回结果应为 None
    assert_(to_writeable(None) is None)
    
    # 测试传入字符串时，返回结果的数据类型应为 np.str_
    assert_equal(to_writeable('a string').dtype.type, np.str_)
    
    # 测试传入标量时，返回结果的形状为 ()
    res = to_writeable(1)
    assert_equal(res.shape, ())
    # 断言返回结果的数据类型与 np.array(1) 的数据类型相等
    assert_equal(res.dtype.type, np.array(1).dtype.type)
    # 断言返回结果与预期值相等
    assert_array_equal(res, 1)
    
    # 测试传入空字典时，返回结果应为 EmptyStructMarker
    assert_(to_writeable({}) is EmptyStructMarker)
    
    # 测试传入普通对象时，返回结果应为 None（因为普通对象没有 __dict__ 属性）
    assert_(to_writeable(object()) is None)
    
    # 测试传入自定义对象实例时，返回结果应为 EmptyStructMarker（因为自定义对象实例没有 __dict__ 属性）
    class C:
        pass
    assert_(to_writeable(C()) is EmptyStructMarker)
    
    # 测试传入包含合法字符键的字典时，返回结果中键 'a' 对应的形状为 (1,)
    res = to_writeable({'a': 1})['a']
    assert_equal(res.shape, (1,))
    # 断言返回结果的数据类型为 np.object_
    assert_equal(res.dtype.type, np.object_)
    
    # 测试传入字典中包含非法字符键时，返回结果应为 EmptyStructMarker
    assert_(to_writeable({'1':1}) is EmptyStructMarker)
    assert_(to_writeable({'_a':1}) is EmptyStructMarker)
    
    # 测试传入字典中包含合法字符键时，返回结果应为结构化数组
    assert_equal(to_writeable({'1':1, 'f': 2}),
                 np.array([(2,)], dtype=[('f', '|O8')]))


def test_recarray():
    # 检查结构化数组的往返操作
    dt = [('f1', 'f8'),
          ('f2', 'S10')]
    arr = np.zeros((2,), dtype=dt)
    arr[0]['f1'] = 0.5
    arr[0]['f2'] = 'python'
    arr[1]['f1'] = 99
    arr[1]['f2'] = 'not perl'
    stream = BytesIO()
    # 将结构化数组保存到字节流中
    savemat(stream, {'arr': arr})
    # 从字节流中加载数据，不将结构数组视为对象数组
    d = loadmat(stream, struct_as_record=False)
    a20 = d['arr'][0,0]
    # 断言加载后的结构化数组字段值符合预期
    assert_equal(a20.f1, 0.5)
    assert_equal(a20.f2, 'python')
    # 从字节流中加载数据，将结构数组视为对象数组
    d = loadmat(stream, struct_as_record=True)
    a20 = d['arr'][0,0]
    # 确保字典 `a20` 中键 'f1' 的值为 0.5
    assert_equal(a20['f1'], 0.5)
    # 确保字典 `a20` 中键 'f2' 的值为 'python'
    assert_equal(a20['f2'], 'python')
    
    # 结构体始终作为对象类型返回
    # 确保 `a20` 的数据类型是一个 NumPy 结构化数据类型，包含字段 'f1' 和 'f2'，类型为对象 ('O')
    assert_equal(a20.dtype, np.dtype([('f1', 'O'),
                                      ('f2', 'O')]))
    
    # 从字典 `d` 的数组 'arr' 的扁平化视图中获取第二个元素赋给 `a21`
    a21 = d['arr'].flat[1]
    # 确保字典 `a21` 中键 'f1' 的值为 99
    assert_equal(a21['f1'], 99)
    # 确保字典 `a21` 中键 'f2' 的值为 'not perl'
    assert_equal(a21['f2'], 'not perl')
# 定义一个函数用于测试保存对象的功能
def test_save_object():
    # 定义一个简单的类C
    class C:
        pass
    # 创建类C的实例c
    c = C()
    # 设置类C实例c的两个属性
    c.field1 = 1
    c.field2 = 'a string'
    # 创建一个字节流对象
    stream = BytesIO()
    # 将类C的实例c保存到字节流中
    savemat(stream, {'c': c})
    # 从字节流中加载数据，返回一个字典d
    d = loadmat(stream, struct_as_record=False)
    # 获取字典d中键为'c'的元素，并取其第[0,0]个元素，赋值给c2
    c2 = d['c'][0,0]
    # 断言c2的field1属性等于1
    assert_equal(c2.field1, 1)
    # 断言c2的field2属性等于'a string'
    assert_equal(c2.field2, 'a string')
    # 重新从字节流中加载数据，设置struct_as_record=True，返回一个新的字典d
    d = loadmat(stream, struct_as_record=True)
    # 获取字典d中键为'c'的元素，并取其第[0,0]个元素，赋值给c2
    c2 = d['c'][0,0]
    # 断言c2的'field1'键对应的值等于1
    assert_equal(c2['field1'], 1)
    # 断言c2的'field2'键对应的值等于'a string'
    assert_equal(c2['field2'], 'a string')


# 定义一个函数用于测试读取选项的功能
def test_read_opts():
    # 测试读取选项是否在初始化和初始化后有效
    # 创建一个形状为(1,6)的NumPy数组arr，内容为0到5
    arr = np.arange(6).reshape(1,6)
    # 创建一个字节流对象
    stream = BytesIO()
    # 将数组arr保存到字节流中，键为'a'
    savemat(stream, {'a': arr})
    # 创建MatFile5Reader对象rdr，从stream中读取数据
    rdr = MatFile5Reader(stream)
    # 获取rdr中的变量字典back_dict
    back_dict = rdr.get_variables()
    # 获取back_dict中键为'a'的数组rarr
    rarr = back_dict['a']
    # 断言rarr与arr相等
    assert_array_equal(rarr, arr)
    # 重新创建MatFile5Reader对象rdr，设置squeeze_me=True，从stream中读取数据
    rdr = MatFile5Reader(stream, squeeze_me=True)
    # 断言rdr中的变量字典中键为'a'的数组与arr重塑为(6,)后相等
    assert_array_equal(rdr.get_variables()['a'], arr.reshape((6,)))
    # 设置rdr.squeeze_me = False
    rdr.squeeze_me = False
    # 断言rarr与arr相等
    assert_array_equal(rarr, arr)
    # 重新创建MatFile5Reader对象rdr，设置byte_order=boc.native_code，从stream中读取数据
    rdr = MatFile5Reader(stream, byte_order=boc.native_code)
    # 断言rdr中的变量字典中键为'a'的数组与arr相等
    assert_array_equal(rdr.get_variables()['a'], arr)
    # 设置byte_order=boc.swapped_code会导致读取错误，预期抛出异常
    rdr = MatFile5Reader(stream, byte_order=boc.swapped_code)
    assert_raises(Exception, rdr.get_variables)
    # 设置rdr.byte_order = boc.native_code
    rdr.byte_order = boc.native_code
    # 断言rdr中的变量字典中键为'a'的数组与arr相等
    assert_array_equal(rdr.get_variables()['a'], arr)
    # 创建一个内容为['a string']的字符串数组arr
    arr = np.array(['a string'])
    # 清空并重置字节流对象stream
    stream.truncate(0)
    stream.seek(0)
    # 将数组arr保存到字节流中，键为'a'
    savemat(stream, {'a': arr})
    # 重新创建MatFile5Reader对象rdr，从stream中读取数据
    rdr = MatFile5Reader(stream)
    # 断言rdr中的变量字典中键为'a'的数组与arr相等
    assert_array_equal(rdr.get_variables()['a'], arr)
    # 重新创建MatFile5Reader对象rdr，设置chars_as_strings=False，从stream中读取数据
    rdr = MatFile5Reader(stream, chars_as_strings=False)
    # 创建一个至少二维的字符数组carr，内容与arr.item()列表相同，dtype为'U1'
    carr = np.atleast_2d(np.array(list(arr.item()), dtype='U1'))
    # 断言rdr中的变量字典中键为'a'的数组与carr相等
    assert_array_equal(rdr.get_variables()['a'], carr)
    # 设置rdr.chars_as_strings = True
    rdr.chars_as_strings = True
    # 断言rdr中的变量字典中键为'a'的数组与arr相等
    assert_array_equal(rdr.get_variables()['a'], arr)


# 定义一个函数用于测试空字符串的读取
def test_empty_string():
    # 确保读取空字符串不会引发错误
    # 定义单个空字符串文件名estring_fname
    estring_fname = pjoin(test_data_path, 'single_empty_string.mat')
    # 打开文件estring_fname，以二进制读取模式
    fp = open(estring_fname, 'rb')
    # 创建MatFile5Reader对象rdr，从fp中读取数据
    rdr = MatFile5Reader(fp)
    # 获取rdr中的变量字典d
    d = rdr.get_variables()
    # 关闭文件fp
    fp.close()
    # 断言d中键为'a'的数组与空的字符串数组np.array([], dtype='U1')相等
    assert_array_equal(d['a'], np.array([], dtype='U1'))
    # 空字符串往返测试。Matlab无法区分空的字符串数组和包含单个空字符串的字符串数组，
    # 因为它将字符串存储为字符数组。没有办法有一个非空的包含空字符串的字符数组。
    # 创建一个字节流对象stream
    stream = BytesIO()
    # 将包含单个空字符串的数组{'a': np.array([''])}保存到字节流中
    savemat(stream, {'a': np.array([''])})
    # 创建MatFile5Reader对象rdr，从stream中读取数据
    rdr = MatFile5Reader(stream)
    # 获取rdr中的变量字典d
    d = rdr.get_variables()
    # 断言d中键为'a'的数组与空的字符串数组np.array([], dtype='U1')相等
    assert_array_equal(d['a'], np.array([], dtype='U1'))
    # 清空并重置字节流对象stream
    stream.truncate(0)
    stream.seek(0)
    # 将空的字符串数组{'a': np.array([], dtype='U1')}保存到字节流中
    savemat(stream, {'a': np.array([], dtype='U1')})
    # 创建MatFile5Reader对象rdr，从stream中读取数据
    rdr = MatFile5Reader(stream)
    # 获取rdr中的变量字典d
    d = rdr.get_variables()
    # 断言d中键为'a'的数组与空的字符串数组np.array([], dtype='U1')相等
    assert_array_equal(d['a'], np.array([], dtype='U1'))
    # 关闭字节流对象stream
    stream.close()
    # 定义一个循环，对每对异常类型（exc）和文件名（fname）进行处理
    for exc, fname in [(ValueError, 'corrupted_zlib_data.mat'),
                       (zlib.error, 'corrupted_zlib_checksum.mat')]:
        # 打开指定路径下的文件（以二进制读取模式）
        with open(pjoin(test_data_path, fname), 'rb') as fp:
            # 使用 MatFile5Reader 类读取打开的文件对象 fp
            rdr = MatFile5Reader(fp)
            # 断言在读取过程中会引发异常 exc
            assert_raises(exc, rdr.get_variables)
def test_corrupted_data_check_can_be_disabled():
    # 打开名为 'corrupted_zlib_data.mat' 的测试数据文件，以二进制模式
    with open(pjoin(test_data_path, 'corrupted_zlib_data.mat'), 'rb') as fp:
        # 创建 MatFile5Reader 对象，禁用数据完整性验证
        rdr = MatFile5Reader(fp, verify_compressed_data_integrity=False)
        # 获取文件中的变量
        rdr.get_variables()


def test_read_both_endian():
    # 确保能正确读取大端和小端数据
    for fname in ('big_endian.mat', 'little_endian.mat'):
        # 打开测试数据文件，以二进制模式
        fp = open(pjoin(test_data_path, fname), 'rb')
        # 创建 MatFile5Reader 对象
        rdr = MatFile5Reader(fp)
        # 获取文件中的变量
        d = rdr.get_variables()
        fp.close()
        # 检查字符串数组是否相等
        assert_array_equal(d['strings'],
                           np.array([['hello'],
                                     ['world']], dtype=object))
        # 检查浮点数数组是否相等
        assert_array_equal(d['floats'],
                           np.array([[2., 3.],
                                     [3., 4.]], dtype=np.float32))


def test_write_opposite_endian():
    # 不支持写入不同字节序的 .mat 文件，但需要正确处理用户提供的异字节序 NumPy 数组
    float_arr = np.array([[2., 3.],
                          [3., 4.]])
    int_arr = np.arange(6).reshape((2, 3))
    uni_arr = np.array(['hello', 'world'], dtype='U')
    # 创建一个字节流对象
    stream = BytesIO()
    # 将数据保存到流中
    savemat(stream, {
        'floats': float_arr.byteswap().view(float_arr.dtype.newbyteorder()),
        'ints': int_arr.byteswap().view(int_arr.dtype.newbyteorder()),
        'uni_arr': uni_arr.byteswap().view(uni_arr.dtype.newbyteorder()),
    })
    # 创建 MatFile5Reader 对象
    rdr = MatFile5Reader(stream)
    # 获取文件中的变量
    d = rdr.get_variables()
    # 检查浮点数数组是否相等
    assert_array_equal(d['floats'], float_arr)
    # 检查整数数组是否相等
    assert_array_equal(d['ints'], int_arr)
    # 检查 Unicode 字符串数组是否相等
    assert_array_equal(d['uni_arr'], uni_arr)
    # 关闭流
    stream.close()


def test_logical_array():
    # 回路测试并不验证我们是否正确加载了布尔类型的数据
    with open(pjoin(test_data_path, 'testbool_8_WIN64.mat'), 'rb') as fobj:
        # 创建 MatFile5Reader 对象，将 Matlab 类型转换为布尔类型
        rdr = MatFile5Reader(fobj, mat_dtype=True)
        # 获取文件中的变量
        d = rdr.get_variables()
    # 创建一个布尔数组
    x = np.array([[True], [False]], dtype=np.bool_)
    # 检查数组是否相等
    assert_array_equal(d['testbools'], x)
    # 检查数组的数据类型是否正确
    assert_equal(d['testbools'].dtype, x.dtype)


def test_logical_out_type():
    # 确认布尔类型被写入为 uint8，其类别为 uint8 类
    stream = BytesIO()
    barr = np.array([False, True, False])
    # 将布尔数组保存到流中
    savemat(stream, {'barray': barr})
    stream.seek(0)
    # 创建 MatFile5Reader 对象
    reader = MatFile5Reader(stream)
    # 初始化读取
    reader.initialize_read()
    # 读取文件头部
    reader.read_file_header()
    # 读取变量头部信息
    hdr, _ = reader.read_var_header()
    # 检查变量类型
    assert_equal(hdr.mclass, mio5p.mxUINT8_CLASS)
    # 确认为逻辑数组
    assert_equal(hdr.is_logical, True)
    # 读取变量数组
    var = reader.read_var_array(hdr, False)
    # 检查数组的数据类型是否为 uint8
    assert_equal(var.dtype.type, np.uint8)


def test_roundtrip_zero_dimensions():
    # 测试零维数组的往返操作
    stream = BytesIO()
    # 保存一个空的 (10, 0) 数组到流中
    savemat(stream, {'d':np.empty((10, 0))})
    # 从流中加载数据
    d = loadmat(stream)
    # 检查数组的形状是否为 (10, 0)
    assert d['d'].shape == (10, 0)


def test_mat4_3d():
    # 测试将三维数组写入 Matlab 4 文件时的行为
    stream = BytesIO()
    # 创建一个 3D 数组
    arr = np.arange(24).reshape((2,3,4))
    # 使用断言（assert）来验证函数 savemat 在特定条件下会引发 ValueError 异常
    assert_raises(ValueError, savemat, stream, {'a': arr}, True, '4')
# 测试函数，用于读取测试数据中的特定文件，验证MatFile5Reader类的功能
def test_func_read():
    # 构建测试文件路径
    func_eg = pjoin(test_data_path, 'testfunc_7.4_GLNX86.mat')
    # 打开文件对象以二进制读取模式
    fp = open(func_eg, 'rb')
    # 创建MatFile5Reader对象，传入文件对象
    rdr = MatFile5Reader(fp)
    # 获取文件中的变量数据字典
    d = rdr.get_variables()
    # 关闭文件对象
    fp.close()
    # 断言'测试函数'变量在返回的数据字典中类型为MatlabFunction类对象
    assert isinstance(d['testfunc'], MatlabFunction)
    # 创建字节流对象
    stream = BytesIO()
    # 创建MatFile5Writer对象，传入字节流对象
    wtr = MatFile5Writer(stream)
    # 断言调用put_variables方法时抛出MatWriteError异常
    assert_raises(MatWriteError, wtr.put_variables, d)


# 测试MatFile5Reader类在不同数据类型设置下的功能
def test_mat_dtype():
    # 构建测试文件路径
    double_eg = pjoin(test_data_path, 'testmatrix_6.1_SOL2.mat')
    # 打开文件对象以二进制读取模式
    fp = open(double_eg, 'rb')
    # 创建MatFile5Reader对象，关闭Matlab数据类型检查
    rdr = MatFile5Reader(fp, mat_dtype=False)
    # 获取文件中的变量数据字典
    d = rdr.get_variables()
    # 关闭文件对象
    fp.close()
    # 断言'testmatrix'变量在返回的数据字典中dtype的kind为'u'（无符号整数类型）
    assert_equal(d['testmatrix'].dtype.kind, 'u')

    # 重新打开文件对象以二进制读取模式
    fp = open(double_eg, 'rb')
    # 创建MatFile5Reader对象，打开Matlab数据类型检查
    rdr = MatFile5Reader(fp, mat_dtype=True)
    # 获取文件中的变量数据字典
    d = rdr.get_variables()
    # 关闭文件对象
    fp.close()
    # 断言'testmatrix'变量在返回的数据字典中dtype的kind为'f'（浮点数类型）
    assert_equal(d['testmatrix'].dtype.kind, 'f')


# 测试在结构体中包含稀疏矩阵的功能
def test_sparse_in_struct():
    # 创建包含稀疏矩阵的结构体
    st = {'sparsefield': SP.coo_matrix(np.eye(4))}
    # 创建字节流对象
    stream = BytesIO()
    # 将结构体保存为.mat文件
    savemat(stream, {'a': st})
    # 从字节流中加载.mat文件，返回数据字典
    d = loadmat(stream, struct_as_record=True)
    # 断言从加载的数据中取出稀疏矩阵字段并转换为数组后与np.eye(4)相等
    assert_array_equal(d['a'][0, 0]['sparsefield'].toarray(), np.eye(4))


# 测试.mat结构体数据加载时的squeeze选项功能
def test_mat_struct_squeeze():
    # 创建字节流对象
    stream = BytesIO()
    # 定义包含结构体的输入数据字典
    in_d = {'st': {'one': 1, 'two': 2}}
    # 将输入数据字典保存为.mat文件
    savemat(stream, in_d)
    # 加载.mat文件，不使用squeeze选项，不应出现错误
    loadmat(stream, struct_as_record=False)
    # 加载.mat文件，使用squeeze选项，前面出现的错误应被修复
    loadmat(stream, struct_as_record=False, squeeze_me=True)


# 测试标量数据加载时的squeeze选项功能
def test_scalar_squeeze():
    # 创建字节流对象
    stream = BytesIO()
    # 定义包含标量、字符串和结构体的输入数据字典
    in_d = {'scalar': [[0.1]], 'string': 'my name', 'st': {'one': 1, 'two': 2}}
    # 将输入数据字典保存为.mat文件
    savemat(stream, in_d)
    # 加载.mat文件，使用squeeze选项，确保标量被正确解释为float类型
    out_d = loadmat(stream, squeeze_me=True)
    assert_(isinstance(out_d['scalar'], float))
    assert_(isinstance(out_d['string'], str))
    assert_(isinstance(out_d['st'], np.ndarray))


# 测试字符串数据在加载时的rounding功能
def test_str_round():
    # 创建字节流对象
    stream = BytesIO()
    # 定义输入字符串数组和期望输出字符串数组
    in_arr = np.array(['Hello', 'Foob'])
    out_arr = np.array(['Hello', 'Foob '])
    # 将输入字符串数组保存为.mat文件
    savemat(stream, dict(a=in_arr))
    # 加载.mat文件，检查加载结果是否与期望的输出字符串数组相等
    res = loadmat(stream)
    assert_array_equal(res['a'], out_arr)
    # 清空并定位字节流对象到起始位置
    stream.truncate(0)
    stream.seek(0)
    # 创建Fortran顺序的字符串版本
    in_str = in_arr.tobytes(order='F')
    in_from_str = np.ndarray(shape=in_arr.shape,
                             dtype=in_arr.dtype,
                             order='F',
                             buffer=in_str)
    # 将Fortran顺序的字符串版本保存为.mat文件
    savemat(stream, dict(a=in_from_str))
    # 加载.mat文件，检查加载结果是否与期望的输出字符串数组相等
    assert_array_equal(res['a'], out_arr)
    # 清空并定位字节流对象到起始位置
    stream.truncate(0)
    stream.seek(0)
    # 将输入字符串数组转换为Unicode类型，并将期望输出字符串数组也转换为Unicode类型
    in_arr_u = in_arr.astype('U')
    out_arr_u = out_arr.astype('U')
    # 将Unicode类型的输入字符串数组保存为.mat文件
    savemat(stream, {'a': in_arr_u})
    # 加载.mat文件，检查加载结果是否与期望的Unicode类型输出字符串数组相等
    res = loadmat(stream)
    assert_array_equal(res['a'], out_arr_u)


# 测试.mat文件中结构体的字段名
def test_fieldnames():
    # 创建字节流对象
    stream = BytesIO()
    # 将包含结构体的数据保存为.mat文件
    savemat(stream, {'a': {'a': 1, 'b': 2}})
    # 加载.mat文件，返回加载后的数据字典
    res = loadmat(stream)
    # 从变量 res 中获取 'a' 列的数据类型，并获取其字段名
    field_names = res['a'].dtype.names
    # 使用断言确保字段名集合与预期的 {'a', 'b'} 一致
    assert_equal(set(field_names), {'a', 'b'})
# 测试函数：测试从 mat 文件中使用 loadmat 只获取一个变量的功能
def test_loadmat_varnames():
    # 预定义 MAT 文件中的系统变量名列表
    mat5_sys_names = ['__globals__', '__header__', '__version__']
    # 遍历不同的 MAT 文件及其预期系统变量名
    for eg_file, sys_v_names in (
        (pjoin(test_data_path, 'testmulti_4.2c_SOL2.mat'), []),
        (pjoin(test_data_path, 'testmulti_7.4_GLNX86.mat'), mat5_sys_names)):
        
        # 使用 loadmat 加载 MAT 文件
        vars = loadmat(eg_file)
        # 断言加载的变量名集合与预期的变量名集合相同
        assert_equal(set(vars.keys()), set(['a', 'theta'] + sys_v_names))
        
        # 使用 loadmat 只加载 'a' 变量
        vars = loadmat(eg_file, variable_names='a')
        assert_equal(set(vars.keys()), set(['a'] + sys_v_names))
        
        # 使用 loadmat 只加载 ['a'] 变量
        vars = loadmat(eg_file, variable_names=['a'])
        assert_equal(set(vars.keys()), set(['a'] + sys_v_names))
        
        # 使用 loadmat 只加载 ['theta'] 变量
        vars = loadmat(eg_file, variable_names=['theta'])
        assert_equal(set(vars.keys()), set(['theta'] + sys_v_names))
        
        # 使用 loadmat 只加载 ('theta',) 变量
        vars = loadmat(eg_file, variable_names=('theta',))
        assert_equal(set(vars.keys()), set(['theta'] + sys_v_names))
        
        # 使用 loadmat 不加载任何变量
        vars = loadmat(eg_file, variable_names=[])
        assert_equal(set(vars.keys()), set(sys_v_names))
        
        # 使用 loadmat 只加载 ['theta'] 变量，并进行断言
        vnames = ['theta']
        vars = loadmat(eg_file, variable_names=vnames)
        assert_equal(vnames, ['theta'])


# 测试函数：检查保存和加载过程中大多数情况下保留 dtype
def test_round_types():
    # 创建一个 numpy 数组
    arr = np.arange(10)
    # 使用 BytesIO 创建字节流对象
    stream = BytesIO()
    # 遍历不同的 dtype 类型
    for dts in ('f8', 'f4', 'i8', 'i4', 'i2', 'i1',
                'u8', 'u4', 'u2', 'u1', 'c16', 'c8'):
        # 清空字节流并重置位置指针
        stream.truncate(0)
        stream.seek(0)  # 在 Python 3 中对 BytesIO 需要重置指针
        # 将数组 arr 保存到 stream 中，指定 dtype
        savemat(stream, {'arr': arr.astype(dts)})
        # 从 stream 中加载数据
        vars = loadmat(stream)
        # 断言加载后的数组 dtype 与原始 dtype 相同
        assert_equal(np.dtype(dts), vars['arr'].dtype)


# 测试函数：创建一个 mat 文件包含多个变量，并验证写入后再读取的正确性
def test_varmats_from_mat():
    # 定义多个变量及其名称
    names_vars = (('arr', mlarr(np.arange(10))),
                  ('mystr', mlarr('a string')),
                  ('mynum', mlarr(10)))

    # 类 C 类似字典，用于按定义顺序提供变量
    class C:
        def items(self):
            return names_vars
    
    # 使用 BytesIO 创建字节流对象
    stream = BytesIO()
    # 将类 C 的内容保存到 stream 中
    savemat(stream, C())
    # 从 stream 中读取变量数据
    varmats = varmats_from_mat(stream)
    # 断言 varmats 中变量的数量为 3
    assert_equal(len(varmats), 3)
    # 遍历 varmats 中的每个变量并进行断言
    for i in range(3):
        name, var_stream = varmats[i]
        exp_name, exp_res = names_vars[i]
        # 断言变量名称与预期名称相同
        assert_equal(name, exp_name)
        # 加载 var_stream 中的数据并断言与预期数据相同
        res = loadmat(var_stream)
        assert_array_equal(res[name], exp_res)


# 测试函数：测试读取 1x0 字符串的正确性
def test_one_by_zero():
    # 测试文件路径
    func_eg = pjoin(test_data_path, 'one_by_zero_char.mat')
    # 打开 MAT 文件
    fp = open(func_eg, 'rb')
    # 创建 MatFile5Reader 对象
    rdr = MatFile5Reader(fp)
    # 获取文件中的变量
    d = rdr.get_variables()
    # 关闭文件
    fp.close()
    # 断言变量 'var' 的形状为 (0,)
    assert_equal(d['var'].shape, (0,))


# 测试函数：测试在大端平台上读取小端 floa64 稠密矩阵的 byte order 正确性
def test_load_mat4_le():
    # 测试 MAT 文件路径
    mat4_fname = pjoin(test_data_path, 'test_mat4_le_floats.mat')
    # 加载 MAT 文件内容
    vars = loadmat(mat4_fname)
    # 断言加载的数组 'a' 与预期的数组相等
    assert_array_equal(vars['a'], [[0.1, 1.2]])
    # 创建一个空的字节流对象
    bio = BytesIO()
    # 定义一个包含 Unicode 字符串的字典
    var = {'second_cat': 'Schrödinger'}
    # 使用 savemat 函数将字典 var 保存到字节流 bio 中，使用格式 '4'（Mat4 格式）
    savemat(bio, var, format='4')
    # 使用 loadmat 函数从字节流 bio 中加载数据到 var_back 中
    var_back = loadmat(bio)
    # 断言：验证 var_back 中键 'second_cat' 的值与原始字典 var 中相同
    assert_equal(var_back['second_cat'], var['second_cat'])
def test_logical_sparse():
    # 测试能够读取以字节形式存储的 mat 文件中的逻辑稀疏数据。
    # 参考 https://github.com/scipy/scipy/issues/3539。
    # 在某些由 MATLAB 保存的文件中，稀疏数据元素（MATLAB 中的实部子元素）以明显的双精度类型（miDOUBLE）存储，但实际上是单字节。
    filename = pjoin(test_data_path,'logical_sparse.mat')
    # 在修复之前，以下代码会崩溃并报错：
    # ValueError: indices and data should have the same size
    d = loadmat(filename, struct_as_record=True)
    log_sp = d['sp_log_5_4']
    assert_(isinstance(log_sp, SP.csc_matrix))
    assert_equal(log_sp.dtype.type, np.bool_)
    assert_array_equal(log_sp.toarray(),
                       [[True, True, True, False],
                        [False, False, True, False],
                        [False, False, True, False],
                        [False, False, False, False],
                        [False, False, False, False]])


def test_empty_sparse():
    # 能够读取空的稀疏矩阵吗？
    sio = BytesIO()
    import scipy.sparse
    empty_sparse = scipy.sparse.csr_matrix([[0,0],[0,0]])
    savemat(sio, dict(x=empty_sparse))
    sio.seek(0)
    res = loadmat(sio)
    assert_array_equal(res['x'].shape, empty_sparse.shape)
    assert_array_equal(res['x'].toarray(), 0)
    # 空的稀疏矩阵是否以最大 nnz 为 1 写入？
    # 参考 https://github.com/scipy/scipy/issues/4208
    sio.seek(0)
    reader = MatFile5Reader(sio)
    reader.initialize_read()
    reader.read_file_header()
    hdr, _ = reader.read_var_header()
    assert_equal(hdr.nzmax, 1)


def test_empty_mat_error():
    # 测试对空 mat 文件获取特定警告信息
    sio = BytesIO()
    assert_raises(MatReadError, loadmat, sio)


def test_miuint32_compromise():
    # 读取器应该接受 miUINT32 作为 miINT32，但要检查符号
    # 包含 miUINT32 作为 miINT32 的 mat 文件，但值是合法的
    filename = pjoin(test_data_path, 'miuint32_for_miint32.mat')
    res = loadmat(filename)
    assert_equal(res['an_array'], np.arange(10)[None, :])
    # 包含 miUINT32 作为 miINT32 的 mat 文件，但有负值
    filename = pjoin(test_data_path, 'bad_miuint32.mat')
    with assert_raises(ValueError):
        loadmat(filename)


def test_miutf8_for_miint8_compromise():
    # 检查读取器是否接受 ASCII 作为 miUTF8 的数组名称
    filename = pjoin(test_data_path, 'miutf8_array_name.mat')
    res = loadmat(filename)
    assert_equal(res['array_name'], [[1]])
    # 包含非 ASCII UTF8 名称的 mat 文件会触发错误
    filename = pjoin(test_data_path, 'bad_miutf8_array_name.mat')
    with assert_raises(ValueError):
        loadmat(filename)


def test_bad_utf8():
    # 检查读取器是否使用 'replace' 选项读取损坏的 UTF8 数据
    filename = pjoin(test_data_path,'broken_utf8.mat')
    res = loadmat(filename)
    assert_equal(res['bad_string'],
                 b'\x80 am broken'.decode('utf8', 'replace'))


def test_save_unicode_field(tmpdir):
    # ...
    # 使用 os 模块的 join 函数将临时目录 tmpdir 和文件名 'test.mat' 组合成完整的文件路径
    filename = os.path.join(str(tmpdir), 'test.mat')
    # 定义一个测试用的字典，包含复杂结构和不同类型的数据
    test_dict = {'a': {'b': 1, 'c': 'test_str'}}
    # 使用 scipy 库的 savemat 函数将 test_dict 保存为 MATLAB 格式的文件，文件名为 filename
    savemat(filename, test_dict)
def test_save_custom_array_type(tmpdir):
    # 定义一个自定义类 CustomArray，该类重载了 __array__ 方法，返回一个二维的 numpy 数组
    class CustomArray:
        def __array__(self, dtype=None, copy=None):
            return np.arange(6.0).reshape(2, 3)
    
    # 创建一个 CustomArray 的实例
    a = CustomArray()
    
    # 指定保存文件的路径和文件名
    filename = os.path.join(str(tmpdir), 'test.mat')
    
    # 将自定义对象 a 保存为 MATLAB 格式的 .mat 文件
    savemat(filename, {'a': a})
    
    # 从保存的 .mat 文件中读取数据
    out = loadmat(filename)
    
    # 断言从 .mat 文件读取的数据与原始自定义对象 a 转换为 numpy 数组后的数据相等
    assert_array_equal(out['a'], np.array(a))


def test_filenotfound():
    # 检查当文件不存在时是否正确抛出 OSError 异常
    assert_raises(OSError, loadmat, "NotExistentFile00.mat")
    assert_raises(OSError, loadmat, "NotExistentFile00")


def test_simplify_cells():
    # 测试 simplify_cells=True 时的输出
    filename = pjoin(test_data_path, 'testsimplecell.mat')
    
    # 使用 simplify_cells=True 参数加载 .mat 文件
    res1 = loadmat(filename, simplify_cells=True)
    
    # 使用 simplify_cells=False 参数加载 .mat 文件
    res2 = loadmat(filename, simplify_cells=False)
    
    # 断言 simplify_cells=True 时结果中的 "s" 键对应一个字典
    assert_(isinstance(res1["s"], dict))
    
    # 断言 simplify_cells=False 时结果中的 "s" 键对应一个 numpy 数组
    assert_(isinstance(res2["s"], np.ndarray))
    
    # 断言简化后的单元格数组内容正确
    assert_array_equal(res1["s"]["mycell"], np.array(["a", "b", "c"]))


@pytest.mark.parametrize('version, filt, regex', [
    (0, '_4*_*', None),
    (1, '_5*_*', None),
    (1, '_6*_*', None),
    (1, '_7*_*', '^((?!hdf5).)*$'),  # 不包含 hdf5 的过滤条件
    (2, '_7*_*', '.*hdf5.*'),
    (1, '8*_*', None),
])
def test_matfile_version(version, filt, regex):
    # 构建匹配的文件过滤条件
    use_filt = pjoin(test_data_path, 'test*%s.mat' % filt)
    
    # 使用 glob 函数获取符合条件的文件列表
    files = glob(use_filt)
    
    # 如果 regex 参数不为空，则进一步筛选文件列表
    if regex is not None:
        files = [file for file in files if re.match(regex, file) is not None]
    
    # 断言至少有一个文件符合条件
    assert len(files) > 0, \
        f"No files for version {version} using filter {filt}"
    
    # 遍历文件列表，检查每个文件的版本是否符合预期
    for file in files:
        got_version = matfile_version(file)
        assert got_version[0] == version


def test_opaque():
    """Test that we can read a MatlabOpaque object."""
    # 加载包含 MatlabOpaque 对象的 .mat 文件，并验证类型
    data = loadmat(pjoin(test_data_path, 'parabola.mat'))
    assert isinstance(data['parabola'], MatlabFunction)
    assert isinstance(data['parabola'].item()[3].item()[3], MatlabOpaque)


def test_opaque_simplify():
    """Test that we can read a MatlabOpaque object when simplify_cells=True."""
    # 使用 simplify_cells=True 参数加载包含 MatlabOpaque 对象的 .mat 文件，并验证类型
    data = loadmat(pjoin(test_data_path, 'parabola.mat'), simplify_cells=True)
    assert isinstance(data['parabola'], MatlabFunction)


def test_deprecation():
    """Test that access to previous attributes still works."""
    # 检查访问已弃用属性是否仍然可用，并产生 DeprecationWarning 警告
    with assert_warns(DeprecationWarning):
        scipy.io.matlab.mio5_params.MatlabOpaque
    
    # 同样检查另一个已弃用属性
    with assert_warns(DeprecationWarning):
        from scipy.io.matlab.miobase import MatReadError  # noqa: F401


def test_gh_17992(tmp_path):
    # 生成随机数种子
    rng = np.random.default_rng(12345)
    
    # 指定保存路径和文件名
    outfile = tmp_path / "lists.mat"
    
    # 生成两个随机数组
    array_one = rng.random((5,3))
    array_two = rng.random((6,3))
    
    # 创建一个包含这两个数组的列表
    list_of_arrays = [array_one, array_two]
    
    # 对于低于 NumPy 1.24.0 版本，需要抑制警告信息
    # 使用 numpy.testing.suppress_warnings() 上下文管理器来忽略特定类型的警告
    with np.testing.suppress_warnings() as sup:
        # 设置警告过滤器，过滤掉 VisibleDeprecationWarning 类型的警告
        sup.filter(VisibleDeprecationWarning)
        # 将列表 list_of_arrays 中的数据保存到 outfile 指定的 MATLAB 格式文件中
        savemat(outfile,
                {'data': list_of_arrays},
                long_field_names=True,  # 使用长字段名称
                do_compression=True)    # 启用数据压缩

    # 回读检查
    new_dict = {}
    # 从 outfile 文件中加载数据到 new_dict 字典中
    loadmat(outfile,
            new_dict)
    # 使用 assert_allclose 检查 new_dict["data"][0][0] 是否接近于 array_one
    assert_allclose(new_dict["data"][0][0], array_one)
    # 使用 assert_allclose 检查 new_dict["data"][0][1] 是否接近于 array_two
    assert_allclose(new_dict["data"][0][1], array_two)
# 定义一个名为 test_gh_19659 的测试函数，接受一个临时路径 tmp_path 作为参数
def test_gh_19659(tmp_path):
    # 创建一个字典 d，包含两个条目：
    # - "char_array" 键对应一个包含两个字符串数组的 NumPy 数组，每个字符串数组由单个字符构成
    # - "string_array" 键对应一个包含两个字符串的 NumPy 数组
    d = {
        "char_array": np.array([list("char"), list("char")], dtype="U1"),
        "string_array": np.array(["string", "string"]),
    }
    # 使用临时路径 tmp_path 和文件名 "tmp.mat"，创建一个输出文件路径对象
    outfile = tmp_path / "tmp.mat"
    # 调用 savemat 函数，将字典 d 的内容保存到文件 outfile 中，使用 MAT 文件格式版本 4
    # 此处期望不会出现错误
    savemat(outfile, d, format="4")
```