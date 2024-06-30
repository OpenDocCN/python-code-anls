# `D:\src\scipysrc\sympy\sympy\polys\polyconfig.py`

```
"""Configuration utilities for polynomial manipulation algorithms. """

# 导入上下文管理器
from contextlib import contextmanager

# 默认配置项
_default_config = {
    'USE_COLLINS_RESULTANT':      False,  # 是否使用柯林斯结果式
    'USE_SIMPLIFY_GCD':           True,   # 是否简化最大公因式
    'USE_HEU_GCD':                True,   # 是否使用启发式最大公因式

    'USE_IRREDUCIBLE_IN_FACTOR':  False,  # 在因式分解中是否使用不可约多项式
    'USE_CYCLOTOMIC_FACTOR':      True,   # 是否使用周期多项式进行因式分解

    'EEZ_RESTART_IF_NEEDED':      True,   # 如果需要，是否重新启动EEZ算法
    'EEZ_NUMBER_OF_CONFIGS':      3,      # EEZ算法的配置数量
    'EEZ_NUMBER_OF_TRIES':        5,      # EEZ算法的尝试次数
    'EEZ_MODULUS_STEP':           2,      # EEZ算法的模数步长

    'GF_IRRED_METHOD':            'rabin',    # 有限域中不可约多项式的计算方法
    'GF_FACTOR_METHOD':           'zassenhaus',  # 有限域中多项式的因式分解方法

    'GROEBNER':                   'buchberger',  # 格劳布纳基基础算法的选择
}

_current_config = {}  # 当前配置项

@contextmanager
def using(**kwargs):
    for k, v in kwargs.items():
        setup(k, v)

    yield

    for k in kwargs.keys():
        setup(k)

def setup(key, value=None):
    """Assign a value to (or reset) a configuration item. """
    key = key.upper()  # 将键名转换为大写

    if value is not None:
        _current_config[key] = value  # 设置配置项的值
    else:
        _current_config[key] = _default_config[key]  # 使用默认值重置配置项的值


def query(key):
    """Ask for a value of the given configuration item. """
    return _current_config.get(key.upper(), None)  # 获取给定配置项的值


def configure():
    """Initialized configuration of polys module. """
    from os import getenv

    for key, default in _default_config.items():
        value = getenv('SYMPY_' + key)  # 从环境变量中获取配置项值

        if value is not None:
            try:
                _current_config[key] = eval(value)  # 尝试将值转换为对应类型
            except NameError:
                _current_config[key] = value
        else:
            _current_config[key] = default  # 使用默认值初始化配置项

configure()
```