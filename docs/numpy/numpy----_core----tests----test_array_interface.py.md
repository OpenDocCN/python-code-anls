# `.\numpy\numpy\_core\tests\test_array_interface.py`

```py
# 导入系统相关模块
import sys
# 导入 pytest 测试框架模块
import pytest
# 导入 numpy 模块并重命名为 np
import numpy as np
# 从 numpy.testing 模块导入 extbuild, IS_WASM, IS_EDITABLE
from numpy.testing import extbuild, IS_WASM, IS_EDITABLE

# 定义 pytest 的 fixture，用于生成数据和管理临时缓冲区，以便通过数组接口协议与 numpy 共享
@pytest.fixture
def get_module(tmp_path):
    """ Some codes to generate data and manage temporary buffers use when
    sharing with numpy via the array interface protocol.
    """

    # 如果不是在 Linux 平台上，跳过测试（在 cygwin 上链接失败）
    if not sys.platform.startswith('linux'):
        pytest.skip('link fails on cygwin')
    
    # 如果是在 WASM 环境中，跳过测试（无法在 WASM 中构建模块）
    if IS_WASM:
        pytest.skip("Can't build module inside Wasm")
    
    # 如果是可编辑安装，跳过测试（无法为可编辑安装构建模块）
    if IS_EDITABLE:
        pytest.skip("Can't build module for editable install")

    # 定义 prologue 变量，存储 C 语言代码片段，用于构建模块的初始化
    prologue = '''
        #include <Python.h>
        #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
        #include <numpy/arrayobject.h>
        #include <stdio.h>
        #include <math.h>

        NPY_NO_EXPORT
        void delete_array_struct(PyObject *cap) {

            /* get the array interface structure */
            // 获取数组接口结构体指针
            PyArrayInterface *inter = (PyArrayInterface*)
                PyCapsule_GetPointer(cap, NULL);

            /* get the buffer by which data was shared */
            // 获取共享数据的缓冲区指针
            double *ptr = (double*)PyCapsule_GetContext(cap);

            /* for the purposes of the regression test set the elements
               to nan */
            // 为回归测试目的，将元素设置为 nan
            for (npy_intp i = 0; i < inter->shape[0]; ++i)
                ptr[i] = nan("");

            /* free the shared buffer */
            // 释放共享缓冲区
            free(ptr);

            /* free the array interface structure */
            // 释放数组接口结构体
            free(inter->shape);
            free(inter);

            // 输出调试信息
            fprintf(stderr, "delete_array_struct\\ncap = %ld inter = %ld"
                " ptr = %ld\\n", (long)cap, (long)inter, (long)ptr);
        }
        '''
    functions = [
        ("new_array_struct", "METH_VARARGS", """
            # 定义一个新的 Python C 扩展函数 new_array_struct，接受变长参数
            long long n_elem = 0;  # 声明并初始化数组元素数量为0
            double value = 0.0;  # 声明并初始化数组元素的默认值为0.0

            # 解析传入的 Python 参数 args，期望参数为一个长整型和一个双精度浮点数
            if (!PyArg_ParseTuple(args, "Ld", &n_elem, &value)) {
                Py_RETURN_NONE;  # 如果解析失败，返回 None
            }

            /* 分配并初始化用于与 NumPy 共享的数据内存 */
            long long n_bytes = n_elem*sizeof(double);  // 计算需要分配的字节数
            double *data = (double*)malloc(n_bytes);  // 分配内存空间

            if (!data) {
                PyErr_Format(PyExc_MemoryError,
                    "Failed to malloc %lld bytes", n_bytes);  // 如果分配内存失败，抛出内存错误异常

                Py_RETURN_NONE;  // 返回 None
            }

            // 将数组初始化为指定值
            for (long long i = 0; i < n_elem; ++i) {
                data[i] = value;
            }

            /* 计算数组的形状和步幅 */
            int nd = 1;  // 数组维度为1

            npy_intp *ss = (npy_intp*)malloc(2*nd*sizeof(npy_intp));  // 分配存储形状和步幅的内存
            npy_intp *shape = ss;  // 指向形状数据的指针
            npy_intp *stride = ss + nd;  // 指向步幅数据的指针

            shape[0] = n_elem;  // 设置数组形状的第一个维度为元素数量
            stride[0] = sizeof(double);  // 设置数组步幅的第一个维度为双精度浮点数的大小

            /* 构建数组接口 */
            PyArrayInterface *inter = (PyArrayInterface*)
                malloc(sizeof(PyArrayInterface));  // 分配数组接口的内存空间

            memset(inter, 0, sizeof(PyArrayInterface));  // 将分配的内存清零

            inter->two = 2;  // 设置数组接口版本号
            inter->nd = nd;  // 设置数组接口的维度数
            inter->typekind = 'f';  // 设置数组接口的数据类型为浮点数
            inter->itemsize = sizeof(double);  // 设置数组接口的每个元素大小
            inter->shape = shape;  // 设置数组接口的形状数据指针
            inter->strides = stride;  // 设置数组接口的步幅数据指针
            inter->data = data;  // 设置数组接口的数据指针
            inter->flags = NPY_ARRAY_WRITEABLE | NPY_ARRAY_NOTSWAPPED |
                           NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS;  // 设置数组接口的标志位

            /* 封装为一个 Capsule 对象 */
            PyObject *cap = PyCapsule_New(inter, NULL, delete_array_struct);  // 创建一个 Capsule 对象
            PyCapsule_SetContext(cap, data);  // 在 Capsule 对象中保存数据指针

            // 打印调试信息到标准错误输出
            fprintf(stderr, "new_array_struct\\ncap = %ld inter = %ld"
                " ptr = %ld\\n", (long)cap, (long)inter, (long)data);

            // 返回创建的 Capsule 对象
            return cap;
        """)
        ]

    more_init = "import_array();"

    try:
        import array_interface_testing
        return array_interface_testing  # 尝试导入已存在的 array_interface_testing 模块并返回
    except ImportError:
        pass

    // 如果模块不存在，使用 extbuild 构建并导入扩展模块
    return extbuild.build_and_import_extension('array_interface_testing',
                                               functions,
                                               prologue=prologue,
                                               include_dirs=[np.get_include()],
                                               build_dir=tmp_path,
                                               more_init=more_init)
@pytest.mark.slow
def test_cstruct(get_module):
    """
    Test case for validating the behavior of the PyCapsule destructor
    when numpy releases its reference to shared data through the array
    interface protocol.
    """

    class data_source:
        """
        This class is for testing the timing of the PyCapsule destructor
        invoked when numpy release its reference to the shared data as part of
        the numpy array interface protocol. If the PyCapsule destructor is
        called early the shared data is freed and invalid memory accesses will
        occur.
        """

        def __init__(self, size, value):
            self.size = size
            self.value = value

        @property
        def __array_struct__(self):
            """
            Method returning a new array struct using size and value from
            the instance.
            """
            return get_module.new_array_struct(self.size, self.value)

    # write to the same stream as the C code
    stderr = sys.__stderr__

    # used to validate the shared data.
    expected_value = -3.1415
    multiplier = -10000.0

    # create some data to share with numpy via the array interface
    # assign the data an expected value.
    stderr.write(' ---- create an object to share data ---- \n')
    buf = data_source(256, expected_value)
    stderr.write(' ---- OK!\n\n')

    # share the data
    stderr.write(' ---- share data via the array interface protocol ---- \n')
    arr = np.array(buf, copy=False)
    stderr.write('arr.__array_interface___ = %s\n' % (
                 str(arr.__array_interface__)))
    stderr.write('arr.base = %s\n' % (str(arr.base)))
    stderr.write(' ---- OK!\n\n')

    # release the source of the shared data. this will not release the data
    # that was shared with numpy, that is done in the PyCapsule destructor.
    stderr.write(' ---- destroy the object that shared data ---- \n')
    buf = None
    stderr.write(' ---- OK!\n\n')

    # check that we got the expected data. If the PyCapsule destructor we
    # defined was prematurely called then this test will fail because our
    # destructor sets the elements of the array to NaN before free'ing the
    # buffer. Reading the values here may also cause a SEGV
    assert np.allclose(arr, expected_value)

    # read the data. If the PyCapsule destructor we defined was prematurely
    # called then reading the values here may cause a SEGV and will be reported
    # as invalid reads by valgrind
    stderr.write(' ---- read shared data ---- \n')
    stderr.write('arr = %s\n' % (str(arr)))
    stderr.write(' ---- OK!\n\n')

    # write to the shared buffer. If the shared data was prematurely deleted
    # this will may cause a SEGV and valgrind will report invalid writes
    stderr.write(' ---- modify shared data ---- \n')
    arr *= multiplier
    expected_value *= multiplier
    stderr.write('arr.__array_interface___ = %s\n' % (
                 str(arr.__array_interface__)))
    stderr.write('arr.base = %s\n' % (str(arr.base)))
    stderr.write(' ---- OK!\n\n')

    # read the data. If the shared data was prematurely deleted this
    # will may cause a SEGV and valgrind will report invalid reads
    stderr.write(' ---- read modified shared data ---- \n')
    # 输出调试信息，显示 arr 的值
    stderr.write('arr = %s\n' % (str(arr)))
    # 输出调试信息，表示测试通过
    stderr.write(' ---- OK!\n\n')

    # 检查是否获得了预期的数据。如果我们定义的 PyCapsule 析构函数被提前调用，
    # 这个测试将失败，因为我们的析构函数在释放缓冲区之前将数组元素设置为 NaN。
    # 在这里读取值可能会导致段错误（SEGV）。
    assert np.allclose(arr, expected_value)

    # 释放共享数据，这里应该运行 PyCapsule 的析构函数
    stderr.write(' ---- free shared data ---- \n')
    # 将 arr 设置为 None，释放共享数据
    arr = None
    # 输出调试信息，表示释放共享数据成功
    stderr.write(' ---- OK!\n\n')
```