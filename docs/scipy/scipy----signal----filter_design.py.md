# `D:\src\scipysrc\scipy\scipy\signal\filter_design.py`

```
# 导入 scipy 库中的 _sub_module_deprecation 函数，用于处理子模块过时警告
from scipy._lib.deprecation import _sub_module_deprecation

# 设置 __all__ 变量，定义了模块中公开的所有函数和类名，用于控制导出的内容
__all__ = [  # noqa: F822
    'findfreqs', 'freqs', 'freqz', 'tf2zpk', 'zpk2tf', 'normalize',
    'lp2lp', 'lp2hp', 'lp2bp', 'lp2bs', 'bilinear', 'iirdesign',
    'iirfilter', 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel',
    'band_stop_obj', 'buttord', 'cheb1ord', 'cheb2ord', 'ellipord',
    'buttap', 'cheb1ap', 'cheb2ap', 'ellipap', 'besselap',
    'BadCoefficients', 'freqs_zpk', 'freqz_zpk',
    'tf2sos', 'sos2tf', 'zpk2sos', 'sos2zpk', 'group_delay',
    'sosfreqz', 'iirnotch', 'iirpeak', 'bilinear_zpk',
    'lp2lp_zpk', 'lp2hp_zpk', 'lp2bp_zpk', 'lp2bs_zpk',
    'gammatone', 'iircomb',
]

# 定义 __dir__() 函数，返回模块的所有公开成员列表，实现模块的自动补全
def __dir__():
    return __all__

# 定义 __getattr__() 函数，处理对模块中未定义的属性的访问
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，生成过时警告，指示用户使用 signal 子模块中的 filter_design 模块
    return _sub_module_deprecation(
        sub_package="signal",  # 子包名称
        module="filter_design",  # 模块名称
        private_modules=["_filter_design"],  # 私有模块列表
        all=__all__,  # 全部公开成员列表
        attribute=name  # 访问的属性名
    )
```