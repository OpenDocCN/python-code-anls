# `D:\src\scipysrc\scipy\scipy\signal\ltisys.py`

```
# 导入 _sub_module_deprecation 函数，用于处理子模块的弃用警告和提示
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 变量，指定公开的函数和类名列表，用于模块导入时的限定
__all__ = [
    'lti', 'dlti', 'TransferFunction', 'ZerosPolesGain', 'StateSpace',  # lti 系列函数和类
    'lsim', 'impulse', 'step', 'bode',  # 时域和频域响应函数
    'freqresp', 'place_poles', 'dlsim', 'dstep', 'dimpulse',  # 频率响应和状态空间函数
    'dfreqresp', 'dbode',  # 频率响应的离散域函数
    'tf2zpk', 'zpk2tf', 'normalize', 'freqs',  # 系统转换和频率响应函数
    'freqz', 'freqs_zpk', 'freqz_zpk', 'tf2ss', 'abcd_normalize',  # 频率响应和状态空间转换函数
    'ss2tf', 'zpk2ss', 'ss2zpk', 'cont2discrete',  # 状态空间和系统转换函数
]

# 定义 __dir__ 函数，返回当前模块的所有公开名称列表
def __dir__():
    return __all__

# 定义 __getattr__ 函数，处理对不存在的属性的访问
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，生成子模块弃用警告信息，指定模块和属性名
    return _sub_module_deprecation(sub_package="signal", module="ltisys",
                                   private_modules=["_ltisys"], all=__all__,
                                   attribute=name)
```