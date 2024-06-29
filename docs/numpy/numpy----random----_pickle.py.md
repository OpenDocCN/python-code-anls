# `.\numpy\numpy\random\_pickle.py`

```
# 从本地模块中导入不同的随机数生成器类和函数
from .bit_generator import BitGenerator
from .mtrand import RandomState
from ._philox import Philox
from ._pcg64 import PCG64, PCG64DXSM
from ._sfc64 import SFC64

# 定义一个字典，将随机数生成器的名称映射到相应的类
BitGenerators = {'MT19937': MT19937,
                 'PCG64': PCG64,
                 'PCG64DXSM': PCG64DXSM,
                 'Philox': Philox,
                 'SFC64': SFC64,
                 }

def __bit_generator_ctor(bit_generator: str | type[BitGenerator] = 'MT19937'):
    """
    用于反序列化的辅助函数，返回一个随机数生成器对象

    Parameters
    ----------
    bit_generator : type[BitGenerator] or str
        BitGenerator 类或包含 BitGenerator 名称的字符串

    Returns
    -------
    BitGenerator
        BitGenerator 的实例对象
    """
    # 如果 bit_generator 是类对象，则直接使用
    if isinstance(bit_generator, type):
        bit_gen_class = bit_generator
    # 如果 bit_generator 是字符串且存在于 BitGenerators 字典中，则使用对应的类
    elif bit_generator in BitGenerators:
        bit_gen_class = BitGenerators[bit_generator]
    # 否则，抛出 ValueError 异常
    else:
        raise ValueError(
            str(bit_generator) + ' is not a known BitGenerator module.'
        )

    return bit_gen_class()


def __generator_ctor(bit_generator_name="MT19937",
                     bit_generator_ctor=__bit_generator_ctor):
    """
    用于反序列化的辅助函数，返回一个 Generator 对象

    Parameters
    ----------
    bit_generator_name : str or BitGenerator
        包含核心 BitGenerator 名称的字符串或 BitGenerator 实例
    bit_generator_ctor : callable, optional
        接受 bit_generator_name 作为唯一参数并返回初始化的 bit generator 的可调用函数

    Returns
    -------
    rg : Generator
        使用指定核心 BitGenerator 的 Generator
    """
    # 如果 bit_generator_name 是 BitGenerator 的实例，则直接创建 Generator 对象
    if isinstance(bit_generator_name, BitGenerator):
        return Generator(bit_generator_name)
    
    # 使用遗留路径，使用 bit_generator_name 和 bit_generator_ctor 创建 Generator 对象
    return Generator(bit_generator_ctor(bit_generator_name))


def __randomstate_ctor(bit_generator_name="MT19937",
                       bit_generator_ctor=__bit_generator_ctor):
    """
    用于反序列化的辅助函数，返回一个类似于遗留 RandomState 的对象

    Parameters
    ----------
    bit_generator_name : str
        包含核心 BitGenerator 名称的字符串
    bit_generator_ctor : callable, optional
        接受 bit_generator_name 作为唯一参数并返回初始化的 bit generator 的可调用函数

    Returns
    -------
    rs : RandomState
        使用指定核心 BitGenerator 的遗留 RandomState
    """
    # 如果 bit_generator_name 是 BitGenerator 的实例，则直接创建 RandomState 对象
    if isinstance(bit_generator_name, BitGenerator):
        return RandomState(bit_generator_name)
    
    # 使用 bit_generator_name 和 bit_generator_ctor 创建 RandomState 对象
    return RandomState(bit_generator_ctor(bit_generator_name))
```