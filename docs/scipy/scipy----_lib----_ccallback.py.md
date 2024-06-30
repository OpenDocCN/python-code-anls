# `D:\src\scipysrc\scipy\scipy\_lib\_ccallback.py`

```
# 导入_ccallback_c模块
from . import _ccallback_c

# 导入ctypes模块，用于处理C函数指针和库的加载
import ctypes

# 定义PyCFuncPtr为ctypes.CFUNCTYPE(ctypes.c_void_p)的基类
PyCFuncPtr = ctypes.CFUNCTYPE(ctypes.c_void_p).__bases__[0]

# 初始化ffi为None
ffi = None

# 定义一个空的CData类
class CData:
    pass

# 定义_import_cffi函数，用于导入cffi模块并进行初始化设置
def _import_cffi():
    global ffi, CData

    # 如果ffi已经被初始化过，则直接返回
    if ffi is not None:
        return

    try:
        # 尝试导入cffi模块
        import cffi
        # 初始化ffi为cffi.FFI()实例
        ffi = cffi.FFI()
        # 设置CData类为ffi.CData
        CData = ffi.CData
    except ImportError:
        # 如果导入cffi失败，则将ffi设置为False
        ffi = False

# 定义LowLevelCallable类，继承自tuple
class LowLevelCallable(tuple):
    """
    Low-level callback function.

    Some functions in SciPy take as arguments callback functions, which
    can either be python callables or low-level compiled functions. Using
    compiled callback functions can improve performance somewhat by
    avoiding wrapping data in Python objects.

    Such low-level functions in SciPy are wrapped in `LowLevelCallable`
    objects, which can be constructed from function pointers obtained from
    ctypes, cffi, Cython, or contained in Python `PyCapsule` objects.

    .. seealso::

       Functions accepting low-level callables:

       `scipy.integrate.quad`, `scipy.ndimage.generic_filter`,
       `scipy.ndimage.generic_filter1d`, `scipy.ndimage.geometric_transform`

       Usage examples:

       :ref:`ndimage-ccallbacks`, :ref:`quad-callbacks`

    Parameters
    ----------
    function : {PyCapsule, ctypes function pointer, cffi function pointer}
        Low-level callback function.
    user_data : {PyCapsule, ctypes void pointer, cffi void pointer}
        User data to pass on to the callback function.
    signature : str, optional
        Signature of the function. If omitted, determined from *function*,
        if possible.

    Attributes
    ----------
    function
        Callback function given.
    user_data
        User data given.
    signature
        Signature of the function.

    Methods
    -------
    from_cython
        Class method for constructing callables from Cython C-exported
        functions.

    Notes
    -----
    The argument ``function`` can be one of:

    - PyCapsule, whose name contains the C function signature
    - ctypes function pointer
    - cffi function pointer

    The signature of the low-level callback must match one of those expected
    by the routine it is passed to.

    If constructing low-level functions from a PyCapsule, the name of the
    capsule must be the corresponding signature, in the format::

        return_type (arg1_type, arg2_type, ...)

    For example::

        "void (double)"
        "double (double, int *, void *)"

    The context of a PyCapsule passed in as ``function`` is used as ``user_data``,
    if an explicit value for ``user_data`` was not given.

    """

    # Make the class immutable
    __slots__ = ()

    def __new__(cls, function, user_data=None, signature=None):
        # We need to hold a reference to the function & user data,
        # to prevent them going out of scope
        # 解析callback函数、用户数据和签名，生成一个元组并返回
        item = cls._parse_callback(function, user_data, signature)
        return tuple.__new__(cls, (item, function, user_data))
    # 返回一个格式化字符串，表示 LowLevelCallable 对象的可打印形式，包括其函数和用户数据
    def __repr__(self):
        return f"LowLevelCallable({self.function!r}, {self.user_data!r})"

    # 获取 LowLevelCallable 对象中的函数属性
    @property
    def function(self):
        return tuple.__getitem__(self, 1)

    # 获取 LowLevelCallable 对象中的用户数据属性
    @property
    def user_data(self):
        return tuple.__getitem__(self, 2)

    # 获取 LowLevelCallable 对象中存储的函数签名，使用 tuple.__getitem__ 方法访问
    @property
    def signature(self):
        return _ccallback_c.get_capsule_signature(tuple.__getitem__(self, 0))

    # 抛出 ValueError 异常，表示不支持从 LowLevelCallable 对象中获取元素操作
    def __getitem__(self, idx):
        raise ValueError()

    # 从 Cython 模块中创建 LowLevelCallable 对象，使用给定的函数名、用户数据和函数签名（可选）
    @classmethod
    def from_cython(cls, module, name, user_data=None, signature=None):
        """
        Create a low-level callback function from an exported Cython function.

        Parameters
        ----------
        module : module
            Cython module where the exported function resides
        name : str
            Name of the exported function
        user_data : {PyCapsule, ctypes void pointer, cffi void pointer}, optional
            User data to pass on to the callback function.
        signature : str, optional
            Signature of the function. If omitted, determined from *function*.

        """
        try:
            # 获取 Cython 模块中的导出函数
            function = module.__pyx_capi__[name]
        except AttributeError as e:
            # 若模块不是 Cython 模块，则引发错误
            message = "Given module is not a Cython module with __pyx_capi__ attribute"
            raise ValueError(message) from e
        except KeyError as e:
            # 若导出函数在 Cython 模块中不存在，则引发错误
            message = f"No function {name!r} found in __pyx_capi__ of the module"
            raise ValueError(message) from e
        # 返回创建的 LowLevelCallable 对象
        return cls(function, user_data, signature)

    # 解析回调对象，根据其类型获取相应的函数和签名，并生成原始的底层回调对象
    @classmethod
    def _parse_callback(cls, obj, user_data=None, signature=None):
        _import_cffi()

        # 如果输入对象是 LowLevelCallable 类型，则获取其函数和签名
        if isinstance(obj, LowLevelCallable):
            func = tuple.__getitem__(obj, 0)
        # 如果输入对象是 PyCFuncPtr 类型，则使用 ctypes 获取函数和签名
        elif isinstance(obj, PyCFuncPtr):
            func, signature = _get_ctypes_func(obj, signature)
        # 如果输入对象是 CData 类型，则使用 cffi 获取函数和签名
        elif isinstance(obj, CData):
            func, signature = _get_cffi_func(obj, signature)
        # 如果输入对象是有效的 Python capsule，则直接使用作为函数
        elif _ccallback_c.check_capsule(obj):
            func = obj
        else:
            # 若输入对象不是有效的回调对象类型，则引发 ValueError
            raise ValueError("Given input is not a callable or a "
                             "low-level callable (pycapsule/ctypes/cffi)")

        # 根据用户数据类型进行相应处理，获取上下文信息
        if isinstance(user_data, ctypes.c_void_p):
            context = _get_ctypes_data(user_data)
        elif isinstance(user_data, CData):
            context = _get_cffi_data(user_data)
        elif user_data is None:
            context = 0
        elif _ccallback_c.check_capsule(user_data):
            context = user_data
        else:
            # 若用户数据不是有效的底层指针类型，则引发 ValueError
            raise ValueError("Given user data is not a valid "
                             "low-level void* pointer (pycapsule/ctypes/cffi)")

        # 返回原始的 Python capsule 对象，表示底层回调函数
        return _ccallback_c.get_raw_capsule(func, signature, context)
# ctypes 辅助函数: 获取 ctypes 函数的函数指针及其签名
def _get_ctypes_func(func, signature=None):
    # 将 func 转换为 void 指针，并获取其值作为函数指针
    func_ptr = ctypes.cast(func, ctypes.c_void_p).value

    # 构造函数签名
    if signature is None:
        signature = _typename_from_ctypes(func.restype) + " ("
        for j, arg in enumerate(func.argtypes):
            if j == 0:
                signature += _typename_from_ctypes(arg)
            else:
                signature += ", " + _typename_from_ctypes(arg)
        signature += ")"

    return func_ptr, signature


# 从 ctypes 类型中获取类型名称
def _typename_from_ctypes(item):
    if item is None:
        return "void"
    elif item is ctypes.c_void_p:
        return "void *"

    # 获取类型的名称
    name = item.__name__

    pointer_level = 0
    while name.startswith("LP_"):
        pointer_level += 1
        name = name[3:]

    if name.startswith('c_'):
        name = name[2:]

    # 添加指针级别信息
    if pointer_level > 0:
        name += " " + "*"*pointer_level

    return name


# 获取 ctypes 数据的 void 指针值
def _get_ctypes_data(data):
    # 将 data 转换为 void 指针，并获取其值
    return ctypes.cast(data, ctypes.c_void_p).value


#
# CFFI 辅助函数
#

# 获取 CFFI 函数的函数指针及其签名
def _get_cffi_func(func, signature=None):
    # 将 func 转换为 uintptr_t 类型的函数指针
    func_ptr = ffi.cast('uintptr_t', func)

    # 获取函数签名
    if signature is None:
        signature = ffi.getctype(ffi.typeof(func)).replace('(*)', ' ')

    return func_ptr, signature


# 获取 CFFI 数据的指针值
def _get_cffi_data(data):
    # 将 data 转换为 uintptr_t 类型的指针值
    return ffi.cast('uintptr_t', data)
```