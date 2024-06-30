# `D:\src\scipysrc\sympy\sympy\utilities\tests\test_autowrap.py`

```
# 导入必要的模块
import os
import tempfile
import shutil
from io import StringIO

# 导入 sympy 相关模块和函数
from sympy.core import symbols, Eq
from sympy.utilities.autowrap import (
    autowrap, binary_function, CythonCodeWrapper, UfuncifyCodeWrapper, CodeWrapper
)
from sympy.utilities.codegen import (
    CCodeGen, C99CodeGen, CodeGenArgumentListError, make_routine
)
from sympy.testing.pytest import raises
from sympy.testing.tmpfiles import TmpFileManager


def get_string(dump_fn, routines, prefix="file", **kwargs):
    """Wrapper for dump_fn. dump_fn writes its results to a stream object and
       this wrapper returns the contents of that stream as a string. This
       auxiliary function is used by many tests below.

       The header and the empty lines are not generator to facilitate the
       testing of the output.
    """
    # 创建一个 StringIO 对象，用于捕获 dump_fn 写入的输出
    output = StringIO()
    # 调用 dump_fn 将结果写入 output
    dump_fn(routines, output, prefix, **kwargs)
    # 从 StringIO 对象中获取写入的内容作为字符串
    source = output.getvalue()
    # 关闭 StringIO 对象
    output.close()
    return source


def test_cython_wrapper_scalar_function():
    # 定义符号变量
    x, y, z = symbols('x,y,z')
    # 定义表达式
    expr = (x + y)*z
    # 使用 make_routine 创建一个例程
    routine = make_routine("test", expr)
    # 使用 CCodeGen 生成 Cython 代码的包装器
    code_gen = CythonCodeWrapper(CCodeGen())
    # 调用 get_string 函数获取生成的代码字符串
    source = get_string(code_gen.dump_pyx, [routine])

    # 期望的生成代码字符串
    expected = (
        "cdef extern from 'file.h':\n"
        "    double test(double x, double y, double z)\n"
        "\n"
        "def test_c(double x, double y, double z):\n"
        "\n"
        "    return test(x, y, z)")
    # 断言生成的代码与期望的一致
    assert source == expected


def test_cython_wrapper_outarg():
    # 导入 Equality 类
    from sympy.core.relational import Equality
    # 定义符号变量
    x, y, z = symbols('x,y,z')
    # 使用 C99CodeGen 生成 Cython 代码的包装器
    code_gen = CythonCodeWrapper(C99CodeGen())

    # 使用 make_routine 创建一个例程
    routine = make_routine("test", Equality(z, x + y))
    # 调用 get_string 函数获取生成的代码字符串
    source = get_string(code_gen.dump_pyx, [routine])

    # 期望的生成代码字符串
    expected = (
        "cdef extern from 'file.h':\n"
        "    void test(double x, double y, double *z)\n"
        "\n"
        "def test_c(double x, double y):\n"
        "\n"
        "    cdef double z = 0\n"
        "    test(x, y, &z)\n"
        "    return z")
    # 断言生成的代码与期望的一致
    assert source == expected


def test_cython_wrapper_inoutarg():
    # 导入 Equality 类
    from sympy.core.relational import Equality
    # 定义符号变量
    x, y, z = symbols('x,y,z')
    # 使用 C99CodeGen 生成 Cython 代码的包装器
    code_gen = CythonCodeWrapper(C99CodeGen())
    # 使用 make_routine 创建一个例程
    routine = make_routine("test", Equality(z, x + y + z))
    # 调用 get_string 函数获取生成的代码字符串
    source = get_string(code_gen.dump_pyx, [routine])

    # 期望的生成代码字符串
    expected = (
        "cdef extern from 'file.h':\n"
        "    void test(double x, double y, double *z)\n"
        "\n"
        "def test_c(double x, double y, double z):\n"
        "\n"
        "    test(x, y, &z)\n"
        "    return z")
    # 断言生成的代码与期望的一致
    assert source == expected


def test_cython_wrapper_compile_flags():
    # 导入 Equality 类
    from sympy.core.relational import Equality
    # 定义符号变量
    x, y, z = symbols('x,y,z')
    # 使用 make_routine 创建一个例程
    routine = make_routine("test", Equality(z, x + y))

    # 使用 CCodeGen 生成 Cython 代码的包装器
    code_gen = CythonCodeWrapper(CCodeGen())

    # 期望的生成代码字符串
    expected = """\
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
# 定义 Cython 编译选项字典，设置语言级别为 Python 3
cy_opts = {'compiler_directives': {'language_level': '3'}}

# 定义 Cython 扩展模块列表，每个模块包含一个扩展的文件和一个包含的 C 代码文件
ext_mods = [Extension(
    'wrapper_module_%(num)s', ['wrapper_module_%(num)s.pyx', 'wrapped_code_%(num)s.c'],
    include_dirs=[],  # 包含的目录为空列表
    library_dirs=[],  # 库文件目录为空列表
    libraries=[],     # 库文件名为空列表
    extra_compile_args=['-std=c99'],  # 额外的编译参数，包含 C99 标准选项
    extra_link_args=[]  # 额外的链接参数为空列表
)]

# 使用 Cython 编译扩展模块，传入 Cython 编译选项
setup(ext_modules=cythonize(ext_mods, **cy_opts))

# 创建临时目录并设置为临时文件管理器的临时文件夹
temp_dir = tempfile.mkdtemp()
TmpFileManager.tmp_folder(temp_dir)

# 在临时目录中创建 setup.py 文件路径
setup_file_path = os.path.join(temp_dir, 'setup.py')

# 准备文件以供代码生成，将生成的 setup.py 文件内容读取到 setup_text 变量中
code_gen._prepare_files(routine, build_dir=temp_dir)
with open(setup_file_path) as f:
    setup_text = f.read()

# 断言读取到的 setup.py 文件内容与预期的内容相同
assert setup_text == expected

# 使用 CythonCodeWrapper 对象初始化代码生成器，设置包含目录、库目录、库文件、额外编译参数和链接参数
code_gen = CythonCodeWrapper(CCodeGen(),
                             include_dirs=['/usr/local/include', '/opt/booger/include'],
                             library_dirs=['/user/local/lib'],
                             libraries=['thelib', 'nilib'],
                             extra_compile_args=['-slow-math'],
                             extra_link_args=['-lswamp', '-ltrident'],
                             cythonize_options={'compiler_directives': {'boundscheck': False}}
                             )

# 将预期的 setup.py 内容重新赋值给 expected 变量
expected = """\
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
cy_opts = {'compiler_directives': {'boundscheck': False}}

ext_mods = [Extension(
    'wrapper_module_%(num)s', ['wrapper_module_%(num)s.pyx', 'wrapped_code_%(num)s.c'],
    include_dirs=['/usr/local/include', '/opt/booger/include'],
    library_dirs=['/user/local/lib'],
    libraries=['thelib', 'nilib'],
    extra_compile_args=['-slow-math', '-std=c99'],
    extra_link_args=['-lswamp', '-ltrident']
)]
setup(ext_modules=cythonize(ext_mods, **cy_opts))
""" % {'num': CodeWrapper._module_counter}

# 再次准备文件以供代码生成，读取 setup.py 文件内容并与预期值进行断言
code_gen._prepare_files(routine, build_dir=temp_dir)
with open(setup_file_path) as f:
    setup_text = f.read()
assert setup_text == expected

# 导入 NumPy 库并将其包含目录添加到扩展模块的 include_dirs 中
import numpy as np
expected = """\
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
cy_opts = {'compiler_directives': {'boundscheck': False}}
import numpy as np

ext_mods = [Extension(
    'wrapper_module_%(num)s', ['wrapper_module_%(num)s.pyx', 'wrapped_code_%(num)s.c'],
    include_dirs=['/usr/local/include', '/opt/booger/include', np.get_include()],
    library_dirs=['/user/local/lib'],
    libraries=['thelib', 'nilib'],
    extra_compile_args=['-slow-math', '-std=c99'],
    extra_link_args=['-lswamp', '-ltrident']
)]
setup(ext_modules=cythonize(ext_mods, **cy_opts))
""" % {'num': CodeWrapper._module_counter}

# 将需要 NumPy 支持的标志设置为 True，再次准备文件以供代码生成，并断言 setup.py 内容与预期值相符
code_gen._need_numpy = True
code_gen._prepare_files(routine, build_dir=temp_dir)
with open(setup_file_path) as f:
    setup_text = f.read()
assert setup_text == expected

# 清理临时文件夹
TmpFileManager.cleanup()
    # 导入符号操作需要的 Dummy 类
    from sympy.core.symbol import Dummy
    # 创建三个 Dummy 实例，分别代表符号变量 x, y, z
    x, y, z = Dummy('x'), Dummy('y'), Dummy('z')
    # 获取每个 Dummy 实例的索引，转换为字符串
    x_id, y_id, z_id = [str(d.dummy_index) for d in [x, y, z]]
    # 创建一个等式表达式 z = x + y
    expr = Equality(z, x + y)
    # 使用表达式创建一个名称为 "test" 的例程（routine）
    routine = make_routine("test", expr)
    # 创建一个 Cython 代码包装器，使用 C 代码生成器
    code_gen = CythonCodeWrapper(CCodeGen())
    # 调用 code_gen 的 dump_pyx 方法生成代码字符串，传入例程列表
    source = get_string(code_gen.dump_pyx, [routine])
    # 构建预期的模板字符串，定义了一个外部函数和一个包装函数
    expected_template = (
        "cdef extern from 'file.h':\n"
        "    void test(double x_{x_id}, double y_{y_id}, double *z_{z_id})\n"
        "\n"
        "def test_c(double x_{x_id}, double y_{y_id}):\n"
        "\n"
        "    cdef double z_{z_id} = 0\n"
        "    test(x_{x_id}, y_{y_id}, &z_{z_id})\n"
        "    return z_{z_id}")
    # 使用格式化方法替换模板中的占位符，得到最终预期的代码字符串
    expected = expected_template.format(x_id=x_id, y_id=y_id, z_id=z_id)
    # 断言生成的源代码字符串与预期字符串相等
    assert source == expected
def test_autowrap_dummy():
    x, y, z = symbols('x y z')

    # 使用 DummyWrapper 来测试代码生成是否按预期工作

    # 测试简单表达式 x + y 的 autowrap 函数
    f = autowrap(x + y, backend='dummy')
    assert f() == str(x + y)
    assert f.args == "x, y"
    assert f.returns == "nameless"

    # 测试等式 Eq(z, x + y) 的 autowrap 函数
    f = autowrap(Eq(z, x + y), backend='dummy')
    assert f() == str(x + y)
    assert f.args == "x, y"
    assert f.returns == "z"

    # 测试复杂表达式 Eq(z, x + y + z) 的 autowrap 函数
    f = autowrap(Eq(z, x + y + z), backend='dummy')
    assert f() == str(x + y + z)
    assert f.args == "x, y, z"
    assert f.returns == "z"


def test_autowrap_args():
    x, y, z = symbols('x y z')

    # 使用 DummyWrapper 测试 autowrap 函数，同时传递自定义的参数列表

    # 测试传递不正确参数列表时是否会抛出异常
    raises(CodeGenArgumentListError, lambda: autowrap(Eq(z, x + y),
           backend='dummy', args=[x]))

    # 测试传递正确参数列表 [y, x] 给 autowrap 函数
    f = autowrap(Eq(z, x + y), backend='dummy', args=[y, x])
    assert f() == str(x + y)
    assert f.args == "y, x"
    assert f.returns == "z"

    # 测试传递不正确参数列表时是否会抛出异常
    raises(CodeGenArgumentListError, lambda: autowrap(Eq(z, x + y + z),
           backend='dummy', args=[x, y]))

    # 测试传递正确参数列表 [y, x, z] 给 autowrap 函数
    f = autowrap(Eq(z, x + y + z), backend='dummy', args=[y, x, z])
    assert f() == str(x + y + z)
    assert f.args == "y, x, z"
    assert f.returns == "z"

    # 测试传递参数列表 (y, x, z) 给 autowrap 函数（元组形式）
    f = autowrap(Eq(z, x + y + z), backend='dummy', args=(y, x, z))
    assert f() == str(x + y + z)
    assert f.args == "y, x, z"
    assert f.returns == "z"


def test_autowrap_store_files():
    x, y = symbols('x y')

    # 使用 DummyWrapper 测试 autowrap 函数，并指定临时文件夹

    tmp = tempfile.mkdtemp()
    TmpFileManager.tmp_folder(tmp)

    # 测试在指定临时文件夹中生成代码的 autowrap 函数
    f = autowrap(x + y, backend='dummy', tempdir=tmp)
    assert f() == str(x + y)
    assert os.access(tmp, os.F_OK)

    TmpFileManager.cleanup()


def test_autowrap_store_files_issue_gh12939():
    x, y = symbols('x y')

    # 使用 DummyWrapper 测试 autowrap 函数，解决 GitHub 问题 #12939

    tmp = './tmp'
    saved_cwd = os.getcwd()
    temp_cwd = tempfile.mkdtemp()
    try:
        os.chdir(temp_cwd)

        # 测试在指定临时文件夹中生成代码的 autowrap 函数（解决路径问题）
        f = autowrap(x + y, backend='dummy', tempdir=tmp)
        assert f() == str(x + y)
        assert os.access(tmp, os.F_OK)
    finally:
        os.chdir(saved_cwd)
        shutil.rmtree(temp_cwd)


def test_binary_function():
    x, y = symbols('x y')

    # 测试 binary_function 函数的基本功能

    f = binary_function('f', x + y, backend='dummy')
    assert f._imp_() == str(x + y)


def test_ufuncify_source():
    x, y, z = symbols('x,y,z')

    # 测试 ufuncify 源码生成的相关功能

    code_wrapper = UfuncifyCodeWrapper(C99CodeGen("ufuncify"))
    routine = make_routine("test", x + y + z)

    # 获取生成的 C 代码字符串
    source = get_string(code_wrapper.dump_c, [routine])

    expected = """\
#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
#include "file.h"

static PyMethodDef wrapper_module_%(num)sMethods[] = {
        {NULL, NULL, 0, NULL}
};

#ifdef NPY_1_19_API_VERSION
static void test_ufunc(char **args, const npy_intp *dimensions, const npy_intp* steps, void* data)
#else
static void test_ufunc(char **args, npy_intp *dimensions, npy_intp* steps, void* data)
#endif
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
    char *in1 = args[1];
    char *in2 = args[2];
    char *out0 = args[3];
    npy_intp in0_step = steps[0];
"""
    # 从步长数组中获取第一个输入数组（in1）的步长
    npy_intp in1_step = steps[1];
    # 从步长数组中获取第二个输入数组（in2）的步长
    npy_intp in2_step = steps[2];
    # 从步长数组中获取输出数组（out0）的步长
    npy_intp out0_step = steps[3];
    # 遍历循环，执行 n 次迭代
    for (i = 0; i < n; i++) {
        # 将第一个输入数组（in0）的数据作为 double 类型，传递给 test 函数，并将结果存入输出数组（out0）
        *((double *)out0) = test(*(double *)in0, *(double *)in1, *(double *)in2);
        # 更新第一个输入数组（in0）的指针位置，移动一个步长
        in0 += in0_step;
        # 更新第二个输入数组（in1）的指针位置，移动一个步长
        in1 += in1_step;
        # 更新第三个输入数组（in2）的指针位置，移动一个步长
        in2 += in2_step;
        # 更新输出数组（out0）的指针位置，移动一个步长
        out0 += out0_step;
    }
// 定义静态变量 `test_funcs`，包含一个指向 `test_ufunc` 函数的指针数组
PyUFuncGenericFunction test_funcs[1] = {&test_ufunc};
// 定义静态字符数组 `test_types`，包含四个元素，分别代表四个 `NPY_DOUBLE` 类型
static char test_types[4] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
// 定义静态指针数组 `test_data`，包含一个空指针
static void *test_data[1] = {NULL};

// 如果 Python 版本大于等于 3.0
#if PY_VERSION_HEX >= 0x03000000
// 定义结构体 `moduledef`，用于描述 Python 模块
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, // 使用模块定义头初始化
    "wrapper_module_%(num)s", // 模块名称字符串
    NULL, // 模块文档字符串
    -1, // 模块状态
    wrapper_module_%(num)sMethods, // 模块方法列表
    NULL, // 模块的槽函数列表
    NULL, // 模块的内存分配器
    NULL, // 模块的释放器
    NULL  // 模块的执行器
};

// Python 模块初始化函数，返回一个 Python 模块对象
PyMODINIT_FUNC PyInit_wrapper_module_%(num)s(void)
{
    PyObject *m, *d; // 定义 Python 对象指针
    PyObject *ufunc0; // 定义 ufunc 对象指针
    m = PyModule_Create(&moduledef); // 创建 Python 模块对象
    if (!m) { // 如果模块创建失败
        return NULL; // 返回空指针
    }
    import_array(); // 导入 NumPy 数组模块
    import_umath(); // 导入 NumPy 数学函数模块
    d = PyModule_GetDict(m); // 获取模块的字典对象
    // 创建 ufunc 对象
    ufunc0 = PyUFunc_FromFuncAndData(test_funcs, test_data, test_types, 1, 3, 1,
            PyUFunc_None, "wrapper_module_%(num)s", "Created in SymPy with Ufuncify", 0);
    PyDict_SetItemString(d, "test", ufunc0); // 将 ufunc 对象添加到模块的字典中
    Py_DECREF(ufunc0); // 释放 ufunc 对象的引用
    return m; // 返回 Python 模块对象
}
// 如果 Python 版本小于 3.0
#else
// Python 2.x 的模块初始化函数
PyMODINIT_FUNC initwrapper_module_%(num)s(void)
{
    PyObject *m, *d; // 定义 Python 对象指针
    PyObject *ufunc0; // 定义 ufunc 对象指针
    m = Py_InitModule("wrapper_module_%(num)s", wrapper_module_%(num)sMethods); // 初始化 Python 模块对象
    if (m == NULL) { // 如果模块初始化失败
        return; // 直接返回
    }
    import_array(); // 导入 NumPy 数组模块
    import_umath(); // 导入 NumPy 数学函数模块
    d = PyModule_GetDict(m); // 获取模块的字典对象
    // 创建 ufunc 对象
    ufunc0 = PyUFunc_FromFuncAndData(test_funcs, test_data, test_types, 1, 3, 1,
            PyUFunc_None, "wrapper_module_%(num)s", "Created in SymPy with Ufuncify", 0);
    PyDict_SetItemString(d, "test", ufunc0); // 将 ufunc 对象添加到模块的字典中
    Py_DECREF(ufunc0); // 释放 ufunc 对象的引用
}
#endif""" % {'num': CodeWrapper._module_counter}
    assert source == expected
    // 循环从 i = 0 开始，执行直到 i < n 为止，每次迭代增加 i 的值
    for (i = 0; i < n; i++) {
        // 使用 func0 函数处理 in0、in1 和 in2 指向的 double 类型数据，并将结果写入 out0 指向的地址
        *((double *)out0) = func0(*(double *)in0, *(double *)in1, *(double *)in2);
        // 使用 func1 函数处理 in0、in1 和 in2 指向的 double 类型数据，并将结果写入 out1 指向的地址
        *((double *)out1) = func1(*(double *)in0, *(double *)in1, *(double *)in2);
        // 使用 func2 函数处理 in0、in1 和 in2 指向的 double 类型数据，并将结果写入 out2 指向的地址
        *((double *)out2) = func2(*(double *)in0, *(double *)in1, *(double *)in2);
        // 增加指针 in0 的位置，使其指向下一个 double 类型数据
        in0 += in0_step;
        // 增加指针 in1 的位置，使其指向下一个 double 类型数据
        in1 += in1_step;
        // 增加指针 in2 的位置，使其指向下一个 double 类型数据
        in2 += in2_step;
        // 增加指针 out0 的位置，使其指向下一个 double 类型数据位置，准备写入下一个结果
        out0 += out0_step;
        // 增加指针 out1 的位置，使其指向下一个 double 类型数据位置，准备写入下一个结果
        out1 += out1_step;
        // 增加指针 out2 的位置，使其指向下一个 double 类型数据位置，准备写入下一个结果
        out2 += out2_step;
    }
}
# 定义一个数组，包含一个指向 multitest_ufunc 函数的指针
PyUFuncGenericFunction multitest_funcs[1] = {&multitest_ufunc};
# 定义一个静态字符数组，表示多个数据类型
static char multitest_types[6] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
# 定义一个静态指针数组，包含一个空指针
static void *multitest_data[1] = {NULL};

# 如果 Python 版本大于等于 3.0
# 定义一个 PyModuleDef 结构体变量 moduledef
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,               # 初始化 Python 模块定义结构
    "wrapper_module_%(num)s",            # 模块名称字符串
    NULL,                               # 模块文档字符串（未使用）
    -1,                                 # 模块状态（-1 表示可选）
    wrapper_module_%(num)sMethods,       # 模块方法定义结构体指针
    NULL,                               # 模块初始化函数（未使用）
    NULL,                               # 模块清理函数（未使用）
    NULL,                               # 模块状态数据（未使用）
    NULL                                # 模块的额外数据（未使用）
};

# 定义 Python 模块初始化函数 PyInit_wrapper_module_%(num)s
PyMODINIT_FUNC PyInit_wrapper_module_%(num)s(void)
{
    PyObject *m, *d;
    PyObject *ufunc0;
    m = PyModule_Create(&moduledef);    # 创建 Python 模块对象
    if (!m) {
        return NULL;                    # 如果创建失败，则返回空指针
    }
    import_array();                     # 导入 NumPy 数组对象接口
    import_umath();                     # 导入 NumPy 数学函数接口
    d = PyModule_GetDict(m);            # 获取模块的字典对象
    # 创建并注册 ufunc 对象
    ufunc0 = PyUFunc_FromFuncAndData(multitest_funcs, multitest_data, multitest_types, 1, 3, 3,
            PyUFunc_None, "wrapper_module_%(num)s", "Created in SymPy with Ufuncify", 0);
    PyDict_SetItemString(d, "multitest", ufunc0);   # 将 ufunc 对象添加到模块字典中
    Py_DECREF(ufunc0);                  # 释放 ufunc 对象的引用
    return m;                           # 返回 Python 模块对象
}
# 如果 Python 版本小于 3.0
# 定义 Python 2.x 的模块初始化函数 initwrapper_module_%(num)s
PyMODINIT_FUNC initwrapper_module_%(num)s(void)
{
    PyObject *m, *d;
    PyObject *ufunc0;
    m = Py_InitModule("wrapper_module_%(num)s", wrapper_module_%(num)sMethods);   # 初始化 Python 模块对象
    if (m == NULL) {
        return;                         # 如果初始化失败，则直接返回
    }
    import_array();                     # 导入 NumPy 数组对象接口
    import_umath();                     # 导入 NumPy 数学函数接口
    d = PyModule_GetDict(m);            # 获取模块的字典对象
    # 创建并注册 ufunc 对象
    ufunc0 = PyUFunc_FromFuncAndData(multitest_funcs, multitest_data, multitest_types, 1, 3, 3,
            PyUFunc_None, "wrapper_module_%(num)s", "Created in SymPy with Ufuncify", 0);
    PyDict_SetItemString(d, "multitest", ufunc0);   # 将 ufunc 对象添加到模块字典中
    Py_DECREF(ufunc0);                  # 释放 ufunc 对象的引用
}
#endif""" % {'num': CodeWrapper._module_counter}
assert source == expected
```