# `D:\src\scipysrc\scipy\benchmarks\benchmarks\cython_special.py`

```
# 导入正则表达式模块
import re
# 导入 NumPy 库并用 np 别名表示
import numpy as np
# 从 SciPy 库中导入 special 模块
from scipy import special
# 从当前包中的 common 模块中导入 with_attributes 和 safe_import 函数
from .common import with_attributes, safe_import

# 使用 safe_import 上下文管理器，安全导入 cython_special 模块
with safe_import():
    # 从 scipy.special 中导入 cython_special 模块
    from scipy.special import cython_special

# FUNC_ARGS 字典，包含函数名到其参数元组的映射
FUNC_ARGS = {
    'airy_d': (1,),
    'airy_D': (1,),
    'beta_dd': (0.25, 0.75),
    'erf_d': (1,),
    'erf_D': (1+1j,),
    'exprel_d': (1e-6,),
    'gamma_d': (100,),
    'gamma_D': (100+100j,),
    'jv_dd': (1, 1),
    'jv_dD': (1, (1+1j)),
    'loggamma_D': (20,),
    'logit_d': (0.5,),
    'psi_d': (1,),
    'psi_D': (1,),
}

# _CythonSpecialMeta 类，用于创建 CythonSpecial 元类
class _CythonSpecialMeta(type):
    """
    Add time_* benchmarks corresponding to cython_special._bench_*_cy
    """

    # 定义 __new__ 方法，创建类的实例
    def __new__(cls, cls_name, bases, dct):
        # 定义参数列表和参数名称列表
        params = [(10, 100, 1000), ('python', 'numpy', 'cython')]
        param_names = ['N', 'api']

        # 定义获取时间函数的方法
        def get_time_func(name, args):
            # 嵌套函数，用 with_attributes 装饰器添加属性
            @with_attributes(params=[(name,), (args,)] + params,
                             param_names=['name', 'argument'] + param_names)
            # 定义具体的时间函数
            def func(self, name, args, N, api):
                # 根据 api 不同调用不同的函数进行时间测量
                if api == 'python':
                    self.py_func(N, *args)
                elif api == 'numpy':
                    self.np_func(*self.obj)
                else:
                    self.cy_func(N, *args)

            # 设置函数名
            func.__name__ = 'time_' + name
            return func

        # 遍历 FUNC_ARGS 中的函数名，为每个函数生成时间测量函数并加入 dct
        for name in FUNC_ARGS.keys():
            func = get_time_func(name, FUNC_ARGS[name])
            dct[func.__name__] = func

        # 返回元类创建的类
        return type.__new__(cls, cls_name, bases, dct)

# CythonSpecial 类，使用 _CythonSpecialMeta 元类创建
class CythonSpecial(metaclass=_CythonSpecialMeta):
    # 初始化方法，设置函数和对象
    def setup(self, name, args, N, api):
        # 获取对应的 Python、Cython 函数和 NumPy 函数
        self.py_func = getattr(cython_special, f'_bench_{name}_py')
        self.cy_func = getattr(cython_special, f'_bench_{name}_cy')
        m = re.match('^(.*)_[dDl]+$', name)
        self.np_func = getattr(special, m.group(1))

        # 根据参数列表 args 和 N 创建对象并赋值给 self.obj
        self.obj = []
        for arg in args:
            self.obj.append(arg*np.ones(N))
        self.obj = tuple(self.obj)
```