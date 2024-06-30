# `D:\src\scipysrc\sympy\sympy\printing\tests\test_llvmjit.py`

```
# 从 sympy.external 中导入 import_module 函数
from sympy.external import import_module
# 从 sympy.testing.pytest 中导入 raises 函数
from sympy.testing.pytest import raises
# 导入 ctypes 模块，用于处理 C 数据类型和函数调用

# 如果 llvmlite 模块可以导入，则导入 sympy.printing.llvmjitcode as g
if import_module('llvmlite'):
    import sympy.printing.llvmjitcode as g
# 否则，设置 disabled 标志为 True
else:
    disabled = True

# 导入 sympy 库
import sympy
# 从 sympy.abc 中导入 a, b, n 作为符号变量

# copied from numpy.isclose documentation
# 定义函数 isclose，用于判断两个数是否在一定误差范围内相等
def isclose(a, b):
    rtol = 1e-5  # 相对误差容忍度
    atol = 1e-8  # 绝对误差容忍度
    return abs(a-b) <= atol + rtol*abs(b)

# 定义测试函数 test_simple_expr，测试简单的表达式
def test_simple_expr():
    e = a + 1.0  # 创建一个简单的表达式 e = a + 1.0
    f = g.llvm_callable([a], e)  # 使用 llvmjitcode 创建可调用的函数 f
    res = float(e.subs({a: 4.0}).evalf())  # 计算表达式 e 在 a=4.0 处的值
    jit_res = f(4.0)  # 调用 llvmjitcode 编译的函数 f，计算结果

    assert isclose(jit_res, res)  # 断言计算结果符合预期

# 定义测试函数 test_two_arg，测试带两个参数的表达式
def test_two_arg():
    e = 4.0*a + b + 3.0  # 创建表达式 e = 4.0*a + b + 3.0
    f = g.llvm_callable([a, b], e)  # 使用 llvmjitcode 创建带两个参数的可调用函数 f
    res = float(e.subs({a: 4.0, b: 3.0}).evalf())  # 计算表达式 e 在 a=4.0, b=3.0 处的值
    jit_res = f(4.0, 3.0)  # 调用 llvmjitcode 编译的函数 f，计算结果

    assert isclose(jit_res, res)  # 断言计算结果符合预期

# 定义测试函数 test_func，测试包含 sympy 函数的表达式
def test_func():
    e = 4.0*sympy.exp(-a)  # 创建表达式 e = 4.0*exp(-a)
    f = g.llvm_callable([a], e)  # 使用 llvmjitcode 创建可调用函数 f
    res = float(e.subs({a: 1.5}).evalf())  # 计算表达式 e 在 a=1.5 处的值
    jit_res = f(1.5)  # 调用 llvmjitcode 编译的函数 f，计算结果

    assert isclose(jit_res, res)  # 断言计算结果符合预期

# 定义测试函数 test_two_func，测试包含两个 sympy 函数的表达式
def test_two_func():
    e = 4.0*sympy.exp(-a) + sympy.exp(b)  # 创建表达式 e = 4.0*exp(-a) + exp(b)
    f = g.llvm_callable([a, b], e)  # 使用 llvmjitcode 创建带两个参数的可调用函数 f
    res = float(e.subs({a: 1.5, b: 2.0}).evalf())  # 计算表达式 e 在 a=1.5, b=2.0 处的值
    jit_res = f(1.5, 2.0)  # 调用 llvmjitcode 编译的函数 f，计算结果

    assert isclose(jit_res, res)  # 断言计算结果符合预期

# 定义测试函数 test_two_sqrt，测试包含两个平方根的表达式
def test_two_sqrt():
    e = 4.0*sympy.sqrt(a) + sympy.sqrt(b)  # 创建表达式 e = 4.0*sqrt(a) + sqrt(b)
    f = g.llvm_callable([a, b], e)  # 使用 llvmjitcode 创建带两个参数的可调用函数 f
    res = float(e.subs({a: 1.5, b: 2.0}).evalf())  # 计算表达式 e 在 a=1.5, b=2.0 处的值
    jit_res = f(1.5, 2.0)  # 调用 llvmjitcode 编译的函数 f，计算结果

    assert isclose(jit_res, res)  # 断言计算结果符合预期

# 定义测试函数 test_two_pow，测试包含两个幂的表达式
def test_two_pow():
    e = a**1.5 + b**7  # 创建表达式 e = a**1.5 + b**7
    f = g.llvm_callable([a, b], e)  # 使用 llvmjitcode 创建带两个参数的可调用函数 f
    res = float(e.subs({a: 1.5, b: 2.0}).evalf())  # 计算表达式 e 在 a=1.5, b=2.0 处的值
    jit_res = f(1.5, 2.0)  # 调用 llvmjitcode 编译的函数 f，计算结果

    assert isclose(jit_res, res)  # 断言计算结果符合预期

# 定义测试函数 test_callback，测试带回调函数类型的表达式
def test_callback():
    e = a + 1.2  # 创建表达式 e = a + 1.2
    f = g.llvm_callable([a], e, callback_type='scipy.integrate.test')  # 使用 llvmjitcode 创建带回调的可调用函数 f
    m = ctypes.c_int(1)  # 创建 ctypes 的整数类型对象 m
    array_type = ctypes.c_double * 1  # 创建包含一个 double 类型的数组类型
    inp = {a: 2.2}  # 创建输入参数字典
    array = array_type(inp[a])  # 创建输入参数数组
    jit_res = f(m, array)  # 调用 llvmjitcode 编译的函数 f，计算结果

    res = float(e.subs(inp).evalf())  # 计算表达式 e 在给定输入下的值

    assert isclose(jit_res, res)  # 断言计算结果符合预期

# 定义测试函数 test_callback_cubature，测试带回调函数类型 'cubature' 的表达式
def test_callback_cubature():
    e = a + 1.2  # 创建表达式 e = a + 1.2
    f = g.llvm_callable([a], e, callback_type='cubature')  # 使用 llvmjitcode 创建带回调 'cubature' 的可调用函数 f
    m = ctypes.c_int(1)  # 创建 ctypes 的整数类型对象 m
    array_type = ctypes.c_double * 1  # 创建包含一个 double 类型的数组类型
    inp = {a: 2.2}  # 创建输入参数字典
    array = array_type(inp[a])  # 创建输入参数数组
    out_array = array_type(0.0)  # 创建输出参数数组
    jit_ret = f(m, array, None, m, out_array)  # 调用 llvmjitcode 编译的函数 f，计算结果

    assert jit_ret == 0  # 断言计算结果符合预期

    res = float(e.subs(inp).evalf())  # 计算表达式 e 在给定输入下的值

    assert isclose(out_array[0], res)  # 断言计算结果符合预期

# 定义测试函数 test_callback_two，测试带两个参数的回调函数类型表达式
def test_callback_two():
    e = 3*a*b  # 创建
def test_multiple_statements():
    # 匹配从 CSE 返回的结果
    e = [[(b, 4.0*a)], [b + 5]]
    # 通过 g 对象调用 LLVM 可执行函数，传入参数 a，并传入表达式 e
    f = g.llvm_callable([a], e)
    # 计算 e[0][0][1] 中 a=1.5 的值
    b_val = e[0][0][1].subs({a: 1.5})
    # 计算表达式 e[1][0] 中 b=b_val 的值，并转换为浮点数
    res = float(e[1][0].subs({b: b_val}).evalf())
    # 调用 LLVM 可执行函数 f，传入参数 1.5，并返回结果
    jit_res = f(1.5)
    # 断言 jit_res 和 res 接近
    assert isclose(jit_res, res)

    # 使用 callback_type='scipy.integrate.test'，通过 g 对象创建 LLVM 可执行函数 f_callback
    f_callback = g.llvm_callable([a], e, callback_type='scipy.integrate.test')
    # 准备 ctypes.c_int 类型的参数 m，以及 ctypes.c_double 类型的数组 array
    m = ctypes.c_int(1)
    array_type = ctypes.c_double * 1
    array = array_type(1.5)
    # 调用 f_callback，传入 m 和 array，返回结果
    jit_callback_res = f_callback(m, array)
    # 断言 jit_callback_res 和 res 接近
    assert isclose(jit_callback_res, res)


def test_cse():
    # 定义表达式 e，包括 a*a + b*b + exp(-a*a - b*b)
    e = a*a + b*b + sympy.exp(-a*a - b*b)
    # 对表达式 e 进行公共子表达式提取，得到 e2
    e2 = sympy.cse(e)
    # 通过 g 对象创建 LLVM 可执行函数 f，传入参数 a 和 b，以及提取后的表达式 e2
    f = g.llvm_callable([a, b], e2)
    # 计算 e 在 a=2.3, b=0.1 时的精确值，并转换为浮点数
    res = float(e.subs({a: 2.3, b: 0.1}).evalf())
    # 调用 LLVM 可执行函数 f，传入参数 2.3 和 0.1，返回结果
    jit_res = f(2.3, 0.1)

    # 断言 jit_res 和 res 接近
    assert isclose(jit_res, res)


def eval_cse(e, sub_dict):
    # 初始化临时字典 tmp_dict
    tmp_dict = {}
    # 遍历 e[0] 中的临时名称和表达式
    for tmp_name, tmp_expr in e[0]:
        # 对 tmp_expr 应用 sub_dict，得到 e2
        e2 = tmp_expr.subs(sub_dict)
        # 对 e2 应用 tmp_dict，得到 e3
        e3 = e2.subs(tmp_dict)
        # 将 e3 存入 tmp_dict 中
        tmp_dict[tmp_name] = e3
    # 对 e[1] 中的每个表达式，应用 sub_dict 和 tmp_dict，并返回结果列表
    return [e.subs(sub_dict).subs(tmp_dict) for e in e[1]]


def test_cse_multiple():
    # 定义 e1 和 e2 的表达式
    e1 = a*a
    e2 = a*a + b*b
    # 对 [e1, e2] 进行公共子表达式提取，得到 e3
    e3 = sympy.cse([e1, e2])

    # 使用 callback_type='scipy.integrate' 尝试创建 LLVM 可执行函数，预期抛出 NotImplementedError
    raises(NotImplementedError, lambda: g.llvm_callable([a, b], e3, callback_type='scipy.integrate'))

    # 通过 g 对象创建 LLVM 可执行函数 f，传入参数 a 和 b，以及提取后的表达式 e3
    f = g.llvm_callable([a, b], e3)
    # 调用 f，传入参数 0.1 和 1.5，返回结果 jit_res
    jit_res = f(0.1, 1.5)
    # 断言 jit_res 的长度为 2
    assert len(jit_res) == 2
    # 计算 e3 在 a=0.1, b=1.5 时的精确值，并存入 res
    res = eval_cse(e3, {a: 0.1, b: 1.5})
    # 断言 jit_res 的第一个元素与 res 的第一个元素接近
    assert isclose(res[0], jit_res[0])
    # 断言 jit_res 的第二个元素与 res 的第二个元素接近
    assert isclose(res[1], jit_res[1])


def test_callback_cubature_multiple():
    # 定义 e1 和 e2 的表达式
    e1 = a*a
    e2 = a*a + b*b
    # 对 [e1, e2, 4*e2] 进行公共子表达式提取，得到 e3
    e3 = sympy.cse([e1, e2, 4*e2])
    # 通过 g 对象创建 LLVM 可执行函数 f，传入参数 a 和 b，以及提取后的表达式 e3，使用 callback_type='cubature'
    f = g.llvm_callable([a, b], e3, callback_type='cubature')

    # 定义输入变量的数量 ndim 和输出表达式值的数量 outdim
    ndim = 2
    outdim = 3

    # 创建 ctypes.c_int 类型的参数 m 和 fdim
    m = ctypes.c_int(ndim)
    fdim = ctypes.c_int(outdim)
    # 创建 ctypes.c_double 数组类型的 inp 和 out_array
    array_type = ctypes.c_double * ndim
    out_array_type = ctypes.c_double * outdim
    inp = {a: 0.2, b: 1.5}
    array = array_type(inp[a], inp[b])
    out_array = out_array_type()
    # 调用 f，传入 m、array、None、fdim 和 out_array，返回结果 jit_ret
    jit_ret = f(m, array, None, fdim, out_array)

    # 断言 jit_ret 等于 0
    assert jit_ret == 0

    # 计算 e3 在输入字典 inp={a: 0.2, b: 1.5} 下的精确值，并存入 res
    res = eval_cse(e3, inp)

    # 断言 out_array 的第一个元素与 res 的第一个元素接近
    assert isclose(out_array[0], res[0])
    # 断言 out_array 的第二个元素与 res 的第二个元素接近
    assert isclose(out_array[1], res[1])
    # 断言 out_array 的第三个元素与 res 的第三个元素接近
    assert isclose(out_array[2], res[2])


def test_symbol_not_found():
    # 定义表达式 e，包括 a*a + b
    e = a*a + b
    # 使用 g 对象尝试创建 LLVM 可执行函数，预期抛出 LookupError
    raises(LookupError, lambda: g.llvm_callable([a], e))


def test_bad_callback():
    # 定义表达式 e，包括 a
    e = a
    # 使用 g 对象尝试创建 LLVM 可执行函数，使用非法的 callback_type='bad_callback'，预期抛出 ValueError
    raises(ValueError, lambda: g.llvm_callable([a], e, callback_type='bad_callback'))
```