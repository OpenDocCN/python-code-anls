# `D:\src\scipysrc\sympy\sympy\printing\llvmjitcode.py`

```
'''
Use llvmlite to create executable functions from SymPy expressions

This module requires llvmlite (https://github.com/numba/llvmlite).
'''

# 导入 ctypes 库
import ctypes

# 从 sympy.external 模块导入 import_module 函数
from sympy.external import import_module
# 从 sympy.printing.printer 模块导入 Printer 类
from sympy.printing.printer import Printer
# 从 sympy.core.singleton 模块导入 S 对象
from sympy.core.singleton import S
# 从 sympy.tensor.indexed 模块导入 IndexedBase 类
from sympy.tensor.indexed import IndexedBase
# 从 sympy.utilities.decorator 模块导入 doctest_depends_on 装饰器
from sympy.utilities.decorator import doctest_depends_on

# 尝试导入 llvmlite 模块，如果失败则 llvmlite 为 None
llvmlite = import_module('llvmlite')
if llvmlite:
    # 导入 llvmlite.ir 中的 ir 模块
    ll = import_module('llvmlite.ir').ir
    # 导入 llvmlite.binding 中的 binding 模块
    llvm = import_module('llvmlite.binding').binding
    # 初始化 LLVM
    llvm.initialize()
    # 初始化本地目标
    llvm.initialize_native_target()
    # 初始化本地汇编打印器
    llvm.initialize_native_asmprinter()

# 指定 doctest 所需的依赖
__doctest_requires__ = {('llvm_callable'): ['llvmlite']}


class LLVMJitPrinter(Printer):
    '''Convert expressions to LLVM IR'''

    def __init__(self, module, builder, fn, *args, **kwargs):
        # 初始化函数参数映射字典
        self.func_arg_map = kwargs.pop("func_arg_map", {})
        # 如果 llvmlite 未导入，则抛出 ImportError 异常
        if not llvmlite:
            raise ImportError("llvmlite is required for LLVMJITPrinter")
        super().__init__(*args, **kwargs)
        # 设置浮点类型为 DoubleType
        self.fp_type = ll.DoubleType()
        # 设置模块、构建器、函数
        self.module = module
        self.builder = builder
        self.fn = fn
        # 用于保存外部函数的包装器
        self.ext_fn = {}
        # 用于临时变量的字典
        self.tmp_var = {}

    def _add_tmp_var(self, name, value):
        # 添加临时变量到字典中
        self.tmp_var[name] = value

    def _print_Number(self, n):
        # 将数字 n 打印为 LLVM 中的常量
        return ll.Constant(self.fp_type, float(n))

    def _print_Integer(self, expr):
        # 将整数表达式 expr 打印为 LLVM 中的常量
        return ll.Constant(self.fp_type, float(expr.p))

    def _print_Symbol(self, s):
        # 获取符号 s 对应的临时变量或函数参数值
        val = self.tmp_var.get(s)
        if not val:
            # 如果未找到临时变量，则尝试查找函数参数映射中的值
            val = self.func_arg_map.get(s)
        if not val:
            raise LookupError("Symbol not found: %s" % s)
        return val

    def _print_Pow(self, expr):
        # 打印指数表达式 expr 对应的 LLVM IR
        base0 = self._print(expr.base)
        if expr.exp == S.NegativeOne:
            return self.builder.fdiv(ll.Constant(self.fp_type, 1.0), base0)
        if expr.exp == S.Half:
            fn = self.ext_fn.get("sqrt")
            if not fn:
                fn_type = ll.FunctionType(self.fp_type, [self.fp_type])
                fn = ll.Function(self.module, fn_type, "sqrt")
                self.ext_fn["sqrt"] = fn
            return self.builder.call(fn, [base0], "sqrt")
        if expr.exp == 2:
            return self.builder.fmul(base0, base0)

        exp0 = self._print(expr.exp)
        fn = self.ext_fn.get("pow")
        if not fn:
            fn_type = ll.FunctionType(self.fp_type, [self.fp_type, self.fp_type])
            fn = ll.Function(self.module, fn_type, "pow")
            self.ext_fn["pow"] = fn
        return self.builder.call(fn, [base0, exp0], "pow")

    def _print_Mul(self, expr):
        # 打印乘法表达式 expr 对应的 LLVM IR
        nodes = [self._print(a) for a in expr.args]
        e = nodes[0]
        for node in nodes[1:]:
            e = self.builder.fmul(e, node)
        return e
    # 定义一个方法，用于将加法表达式打印为 LLVM IR 指令
    def _print_Add(self, expr):
        # 对表达式中的每个参数进行打印，得到节点列表
        nodes = [self._print(a) for a in expr.args]
        # 初始将第一个节点作为加法的左操作数
        e = nodes[0]
        # 遍历剩余节点，依次将它们与之前的结果进行加法操作
        for node in nodes[1:]:
            e = self.builder.fadd(e, node)
        # 返回最终的加法表达式结果
        return e

    # TODO - 假设所有调用的函数都接受一个双精度浮点数参数。
    #        应该有一个数学库函数列表来验证这一点。
    # 定义一个方法，用于将函数表达式打印为 LLVM IR 调用指令
    def _print_Function(self, expr):
        # 获取函数名称
        name = expr.func.__name__
        # 打印函数的第一个参数表达式
        e0 = self._print(expr.args[0])
        # 获取外部函数的定义
        fn = self.ext_fn.get(name)
        if not fn:
            # 如果未找到外部函数的定义，创建一个新的 LLVM 函数类型
            fn_type = ll.FunctionType(self.fp_type, [self.fp_type])
            # 在 LLVM 模块中定义一个新的函数
            fn = ll.Function(self.module, fn_type, name)
            # 将新定义的函数保存在外部函数字典中
            self.ext_fn[name] = fn
        # 返回对函数的调用指令
        return self.builder.call(fn, [e0], name)

    # 定义一个方法，用于处理不支持的表达式类型，抛出类型错误异常
    def emptyPrinter(self, expr):
        # 抛出类型错误异常，指示不支持的 LLVM JIT 转换类型
        raise TypeError("Unsupported type for LLVM JIT conversion: %s"
                        % type(expr))
# Used when parameters are passed by array.  Often used in callbacks to
# handle a variable number of parameters.
class LLVMJitCallbackPrinter(LLVMJitPrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _print_Indexed(self, expr):
        # Retrieve array and index from func_arg_map based on expr.base
        array, idx = self.func_arg_map[expr.base]
        # Evaluate the offset from expr.indices and convert to integer
        offset = int(expr.indices[0].evalf())
        # Get a pointer to the indexed element in the array
        array_ptr = self.builder.gep(array, [ll.Constant(ll.IntType(32), offset)])
        # Bitcast the array pointer to a pointer of floating-point type (fp_type)
        fp_array_ptr = self.builder.bitcast(array_ptr, ll.PointerType(self.fp_type))
        # Load the value from the floating-point array pointer
        value = self.builder.load(fp_array_ptr)
        return value

    def _print_Symbol(self, s):
        # Check if symbol 's' exists in tmp_var dictionary
        val = self.tmp_var.get(s)
        if val:
            return val

        # Retrieve array and index (default to None and 0 if not found)
        array, idx = self.func_arg_map.get(s, [None, 0])
        # Raise LookupError if symbol 's' is not found in func_arg_map
        if not array:
            raise LookupError("Symbol not found: %s" % s)
        # Get a pointer to the indexed element in the array
        array_ptr = self.builder.gep(array, [ll.Constant(ll.IntType(32), idx)])
        # Bitcast the array pointer to a pointer of floating-point type (fp_type)
        fp_array_ptr = self.builder.bitcast(array_ptr, ll.PointerType(self.fp_type))
        # Load the value from the floating-point array pointer
        value = self.builder.load(fp_array_ptr)
        return value


# ensure lifetime of the execution engine persists (else call to compiled
#   function will seg fault)
exe_engines = []

# ensure names for generated functions are unique
link_names = set()
current_link_suffix = 0


class LLVMJitCode:
    def __init__(self, signature):
        # Initialize LLVMJitCode object with given signature
        self.signature = signature
        self.fp_type = ll.DoubleType()
        self.module = ll.Module('mod1')
        self.fn = None
        self.llvm_arg_types = []
        self.llvm_ret_type = self.fp_type
        self.param_dict = {}  # map symbol name to LLVM function argument
        self.link_name = ''

    def _from_ctype(self, ctype):
        # Convert ctype to corresponding LLVM type
        if ctype == ctypes.c_int:
            return ll.IntType(32)
        if ctype == ctypes.c_double:
            return self.fp_type
        if ctype == ctypes.POINTER(ctypes.c_double):
            return ll.PointerType(self.fp_type)
        if ctype == ctypes.c_void_p:
            return ll.PointerType(ll.IntType(32))
        if ctype == ctypes.py_object:
            return ll.PointerType(ll.IntType(32))

        # Print a message for unhandled ctype
        print("Unhandled ctype = %s" % str(ctype))

    def _create_args(self, func_args):
        """Create types for function arguments"""
        # Convert each argument type in func_args to LLVM type
        self.llvm_ret_type = self._from_ctype(self.signature.ret_type)
        self.llvm_arg_types = [self._from_ctype(a) for a in self.signature.arg_ctypes]

    def _create_function_base(self):
        """Create function with name and type signature"""
        # Generate a unique link name for the function
        global link_names, current_link_suffix
        default_link_name = 'jit_func'
        current_link_suffix += 1
        self.link_name = default_link_name + str(current_link_suffix)
        link_names.add(self.link_name)

        # Define LLVM function type with return type and argument types
        fn_type = ll.FunctionType(self.llvm_ret_type, self.llvm_arg_types)
        # Create LLVM function object within the current LLVM module
        self.fn = ll.Function(self.module, fn_type, name=self.link_name)
    def _create_param_dict(self, func_args):
        """Mapping of symbolic values to function arguments"""
        # 遍历函数参数列表，为每个参数设置名称，并将参数对象映射到参数字典中
        for i, a in enumerate(func_args):
            self.fn.args[i].name = str(a)
            self.param_dict[a] = self.fn.args[i]

    def _create_function(self, expr):
        """Create function body and return LLVM IR"""
        # 在函数对象中附加一个基本块作为入口点，并创建IR构建器
        bb_entry = self.fn.append_basic_block('entry')
        builder = ll.IRBuilder(bb_entry)

        # 创建LLVMJitPrinter实例，用于打印LLVM IR，并传入函数参数映射
        lj = LLVMJitPrinter(self.module, builder, self.fn,
                            func_arg_map=self.param_dict)

        # 转换表达式并返回LLVM IR，然后将转换后的返回值包装
        ret = self._convert_expr(lj, expr)
        lj.builder.ret(self._wrap_return(lj, ret))

        # 将模块转换为字符串形式并返回
        strmod = str(self.module)
        return strmod

    def _wrap_return(self, lj, vals):
        # Return a single double if there is one return value,
        #  else return a tuple of doubles.

        # 如果只有一个返回值且返回类型为double，则直接返回该值
        if self.signature.ret_type == ctypes.c_double:
            return vals[0]

        # 创建一个PyObject*类型的指针
        void_ptr = ll.PointerType(ll.IntType(32))

        # 创建一个包装double值的PyObject*: PyObject* PyFloat_FromDouble(double v)
        wrap_type = ll.FunctionType(void_ptr, [self.fp_type])
        wrap_fn = ll.Function(lj.module, wrap_type, "PyFloat_FromDouble")

        # 对每个返回值调用包装函数，并根据返回值数量返回单个值或元组
        wrapped_vals = [lj.builder.call(wrap_fn, [v]) for v in vals]
        if len(vals) == 1:
            final_val = wrapped_vals[0]
        else:
            # 创建一个元组：PyObject* PyTuple_Pack(Py_ssize_t n, ...)

            # 定义元组参数类型，首个参数应为Py_ssize_t类型
            tuple_arg_types = [ll.IntType(32)]
            tuple_arg_types.extend([void_ptr]*len(vals))
            tuple_type = ll.FunctionType(void_ptr, tuple_arg_types)
            tuple_fn = ll.Function(lj.module, tuple_type, "PyTuple_Pack")

            # 创建元组参数列表，包括元素数量和所有元素的PyObject*指针
            tuple_args = [ll.Constant(ll.IntType(32), len(wrapped_vals))]
            tuple_args.extend(wrapped_vals)

            # 调用元组打包函数并返回最终值
            final_val = lj.builder.call(tuple_fn, tuple_args)

        return final_val

    def _convert_expr(self, lj, expr):
        try:
            # Match CSE return data structure.
            # 如果表达式长度为2，则处理临时表达式和最终表达式
            if len(expr) == 2:
                tmp_exprs = expr[0]
                final_exprs = expr[1]
                # 如果最终表达式数量不为1且返回类型为double，则抛出异常
                if len(final_exprs) != 1 and self.signature.ret_type == ctypes.c_double:
                    raise NotImplementedError("Return of multiple expressions not supported for this callback")
                # 将临时表达式中的名称和表达式打印，并添加到临时变量列表中
                for name, e in tmp_exprs:
                    val = lj._print(e)
                    lj._add_tmp_var(name, val)
        except TypeError:
            # 如果发生TypeError，则将整个表达式作为最终表达式处理
            final_exprs = [expr]

        # 打印所有最终表达式并返回其值
        vals = [lj._print(e) for e in final_exprs]

        return vals
    # 编译函数，接受一个 LLVM IR 字符串作为参数
    def _compile_function(self, strmod):
        # 声明全局变量，用于存储编译后的执行引擎
        global exe_engines
        # 解析 LLVM 汇编字符串为 LLVM 模块对象
        llmod = llvm.parse_assembly(strmod)

        # 创建优化级别为 2 的 Pass 管理器构建器
        pmb = llvm.create_pass_manager_builder()
        pmb.opt_level = 2
        # 创建模块级别的 Pass 管理器，并使用上述构建器配置
        pass_manager = llvm.create_module_pass_manager()
        pmb.populate(pass_manager)

        # 运行 Pass 管理器，对 LLVM 模块进行优化
        pass_manager.run(llmod)

        # 创建目标机器对象，使用默认三元组创建目标
        target_machine = \
            llvm.Target.from_default_triple().create_target_machine()
        # 创建 MCJIT 编译器，将优化后的 LLVM 模块传入
        exe_eng = llvm.create_mcjit_compiler(llmod, target_machine)
        # 最终化目标对象文件
        exe_eng.finalize_object()
        # 将编译后的执行引擎添加到全局引擎列表中
        exe_engines.append(exe_eng)

        # 如果条件满足（此处永远不会执行）
        if False:
            # 打印生成的汇编代码
            print("Assembly")
            print(target_machine.emit_assembly(llmod))

        # 获取函数指针地址，使用链接名从执行引擎中获取
        fptr = exe_eng.get_function_address(self.link_name)

        # 返回函数指针地址
        return fptr
class LLVMJitCodeCallback(LLVMJitCode):
    def __init__(self, signature):
        super().__init__(signature)

    def _create_param_dict(self, func_args):
        # 遍历函数参数列表，创建参数字典，用于跟踪参数与 LLVM 函数参数的对应关系
        for i, a in enumerate(func_args):
            if isinstance(a, IndexedBase):
                # 如果参数是 IndexedBase 类型，将其映射到 LLVM 函数参数及其索引，并命名参数
                self.param_dict[a] = (self.fn.args[i], i)
                self.fn.args[i].name = str(a)
            else:
                # 对于普通参数，将其映射到 LLVM 函数输入参数及其索引
                self.param_dict[a] = (self.fn.args[self.signature.input_arg], i)

    def _create_function(self, expr):
        """创建函数体并返回 LLVM IR"""
        # 在函数中创建基本块 'entry'
        bb_entry = self.fn.append_basic_block('entry')
        # 创建 IR 构建器
        builder = ll.IRBuilder(bb_entry)

        # 创建 LLVMJitCallbackPrinter 实例用于打印 LLVM IR，传入模块、构建器、函数和参数映射
        lj = LLVMJitCallbackPrinter(self.module, builder, self.fn, func_arg_map=self.param_dict)

        # 转换表达式为 LLVM IR
        ret = self._convert_expr(lj, expr)

        # 如果有返回参数，则处理返回值
        if self.signature.ret_arg:
            # 将返回参数的指针进行位转换为指向浮点类型的指针
            output_fp_ptr = builder.bitcast(self.fn.args[self.signature.ret_arg], ll.PointerType(self.fp_type))
            # 将返回值数组存储到指针中
            for i, val in enumerate(ret):
                index = ll.Constant(ll.IntType(32), i)
                output_array_ptr = builder.gep(output_fp_ptr, [index])
                builder.store(val, output_array_ptr)
            builder.ret(ll.Constant(ll.IntType(32), 0))  # 返回成功状态码
        else:
            # 否则直接返回包装后的返回值
            lj.builder.ret(self._wrap_return(lj, ret))

        # 将模块转换为字符串表示并返回
        strmod = str(self.module)
        return strmod


class CodeSignature:
    def __init__(self, ret_type):
        self.ret_type = ret_type
        self.arg_ctypes = []

        # 输入参数在数组中的元素索引
        self.input_arg = 0

        # 当输出值通过参数引用而不是返回值时使用
        self.ret_arg = None


def _llvm_jit_code(args, expr, signature, callback_type):
    """从 SymPy 表达式创建本地代码函数"""
    if callback_type is None:
        jit = LLVMJitCode(signature)
    else:
        jit = LLVMJitCodeCallback(signature)

    # 创建函数参数
    jit._create_args(args)
    # 创建函数的基本结构
    jit._create_function_base()
    # 创建参数字典
    jit._create_param_dict(args)
    # 创建函数体并返回 LLVM IR 的字符串表示
    strmod = jit._create_function(expr)
    if False:
        print("LLVM IR")
        print(strmod)
    # 编译函数并返回函数指针
    fptr = jit._compile_function(strmod)
    return fptr


@doctest_depends_on(modules=('llvmlite', 'scipy'))
def llvm_callable(args, expr, callback_type=None):
    '''从 SymPy 表达式编译函数

    表达式使用双精度浮点数进行评估。
    表达式支持一些单参数数学函数（exp、sin、cos 等）。

    参数
    ==========

    args : Symbol 的列表
        生成函数的参数。通常是表达式中的自由符号。
        目前假设每个符号都转换为双精度标量。
    expr : Expr，或者 'cse' 返回的 (Replacements, Expr)
        要编译的表达式。
    '''
    '''
    如果没有安装 llvmlite 库，则抛出 ImportError 异常
    '''
    if not llvmlite:
        raise ImportError("llvmlite is required for llvmjitcode")

    '''
    定义函数签名为单个返回值的代码签名
    '''
    signature = CodeSignature(ctypes.py_object)

    '''
    如果 callback_type 为 None，则假设参数列表中每个参数都是双精度浮点数
    '''
    arg_ctypes = []
    if callback_type is None:
        for _ in args:
            arg_ctype = ctypes.c_double
            arg_ctypes.append(arg_ctype)
    
    '''
    如果 callback_type 为 'scipy.integrate' 或 'scipy.integrate.test'，
    则设置返回类型为双精度浮点数，参数类型包括整数和双精度浮点数的指针
    '''
    elif callback_type in ('scipy.integrate', 'scipy.integrate.test'):
        signature.ret_type = ctypes.c_double
        arg_ctypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        arg_ctypes_formal = [ctypes.c_int, ctypes.c_double]
        signature.input_arg = 1
    # 如果回调类型是 'cubature'，设置对应的参数类型列表
    elif callback_type == 'cubature':
        arg_ctypes = [ctypes.c_int,                      # 第一个参数是 ctypes.c_int
                      ctypes.POINTER(ctypes.c_double),    # 第二个参数是 ctypes.POINTER(ctypes.c_double)
                      ctypes.c_void_p,                   # 第三个参数是 ctypes.c_void_p
                      ctypes.c_int,                      # 第四个参数是 ctypes.c_int
                      ctypes.POINTER(ctypes.c_double)     # 第五个参数是 ctypes.POINTER(ctypes.c_double)
                      ]
        # 设置签名的返回类型为 ctypes.c_int
        signature.ret_type = ctypes.c_int
        # 设置签名的输入参数索引为 1
        signature.input_arg = 1
        # 设置签名的返回值参数索引为 4
        signature.ret_arg = 4
    else:
        # 如果回调类型未知，则抛出 ValueError 异常
        raise ValueError("Unknown callback type: %s" % callback_type)

    # 将最终确定的参数类型列表赋值给签名对象
    signature.arg_ctypes = arg_ctypes

    # 调用 _llvm_jit_code 函数生成函数指针 fptr
    fptr = _llvm_jit_code(args, expr, signature, callback_type)

    # 如果回调类型为 'scipy.integrate'，将参数类型列表调整为形式参数的类型列表
    if callback_type and callback_type == 'scipy.integrate':
        arg_ctypes = arg_ctypes_formal

    # 根据签名的返回类型确定使用的函数类型
    # PYFUNCTYPE 在调用 PyFloat_FromDouble 时保持 GIL，防止在 Python 3.10 上调用时段错误。
    # 可能最好的方法是在返回 float 时使用 ctypes.c_double，而不是使用 ctypes.py_object 并在 jitted 函数内部返回 PyFloat（即让 ctypes 处理从 double 到 PyFloat 的转换）。
    if signature.ret_type == ctypes.py_object:
        FUNCTYPE = ctypes.PYFUNCTYPE
    else:
        FUNCTYPE = ctypes.CFUNCTYPE

    # 使用 FUNCTYPE 创建 ctypes 函数对象 cfunc，并传入函数指针 fptr 和参数类型列表
    cfunc = FUNCTYPE(signature.ret_type, *arg_ctypes)(fptr)
    # 返回 ctypes 函数对象 cfunc
    return cfunc
```