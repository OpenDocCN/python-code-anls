# `D:\src\scipysrc\matplotlib\lib\matplotlib\colors.py`

```
# 导入必要的模块和库
import base64  # 导入base64模块，用于处理base64编码解码操作
from collections.abc import Sized, Sequence, Mapping  # 导入集合抽象基类中的Sized, Sequence, Mapping，用于处理集合类的操作
import functools  # 导入functools模块，用于高阶函数的操作
import importlib  # 导入importlib模块，用于动态加载模块
import inspect  # 导入inspect模块，用于解析源码和检查对象
import io  # 导入io模块，用于核心Python IO功能
import itertools  # 导入itertools模块，用于创建和操作迭代器
from numbers import Real  # 从numbers模块导入Real，用于实数类型的操作
import re  # 导入re模块，用于正则表达式操作

from PIL import Image  # 从PIL库中导入Image模块，用于处理图像
from PIL.PngImagePlugin import PngInfo  # 从PIL.PngImagePlugin中导入PngInfo，用于PNG图像的信息处理

import matplotlib as mpl  # 导入matplotlib库并简写为mpl，用于绘图
import numpy as np  # 导入numpy库并简写为np，用于科学计算
from matplotlib import _api, _cm, cbook, scale  # 从matplotlib中导入特定模块和类
from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS  # 从_color_data模块导入特定颜色数据

class _ColorMapping(dict):
    def __init__(self, mapping):
        super().__init__(mapping)
        self.cache = {}  # 初始化缓存字典

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.cache.clear()  # 设置项时清空缓存

    def __delitem__(self, key):
        super().__delitem__(key)
        self.cache.clear()  # 删除项时清空缓存

_colors_full_map = {}
# 按优先级逆序设置颜色映射
_colors_full_map.update(XKCD_COLORS)  # 更新为XKCD颜色映射
_colors_full_map.update({k.replace('grey', 'gray'): v
                         for k, v in XKCD_COLORS.items()
                         if 'grey' in k})  # 更新包含'grey'的颜色映射，并替换为'gray'
_colors_full_map.update(CSS4_COLORS)  # 更新为CSS4颜色映射
_colors_full_map.update(TABLEAU_COLORS)  # 更新为TABLEAU颜色映射
_colors_full_map.update({k.replace('gray', 'grey'): v
                         for k, v in TABLEAU_COLORS.items()
                         if 'gray' in k})  # 更新包含'gray'的颜色映射，并替换为'grey'
_colors_full_map.update(BASE_COLORS)  # 更新为基础颜色映射
_colors_full_map = _ColorMapping(_colors_full_map)  # 将所有颜色映射包装成_ColorMapping对象

_REPR_PNG_SIZE = (512, 64)  # 设置PNG图片的默认大小

def get_named_colors_mapping():
    """返回名称到命名颜色的全局映射。"""
    return _colors_full_map  # 返回全局颜色映射
# 定义一个名为 ColorSequenceRegistry 的类，它继承自 Mapping 类（未显示代码中的实现细节）。
class ColorSequenceRegistry(Mapping):
    """
    Container for sequences of colors that are known to Matplotlib by name.

    The universal registry instance is `matplotlib.color_sequences`. There
    should be no need for users to instantiate `.ColorSequenceRegistry`
    themselves.

    Read access uses a dict-like interface mapping names to lists of colors::

        import matplotlib as mpl
        cmap = mpl.color_sequences['tab10']

    The returned lists are copies, so that their modification does not change
    the global definition of the color sequence.

    Additional color sequences can be added via
    `.ColorSequenceRegistry.register`::

        mpl.color_sequences.register('rgb', ['r', 'g', 'b'])
    """

    # 内置颜色序列的字典，键为颜色序列的名称，值为对应的颜色数据，从 _cm 模块中获取
    _BUILTIN_COLOR_SEQUENCES = {
        'tab10': _cm._tab10_data,
        'tab20': _cm._tab20_data,
        'tab20b': _cm._tab20b_data,
        'tab20c': _cm._tab20c_data,
        'Pastel1': _cm._Pastel1_data,
        'Pastel2': _cm._Pastel2_data,
        'Paired': _cm._Paired_data,
        'Accent': _cm._Accent_data,
        'Dark2': _cm._Dark2_data,
        'Set1': _cm._Set1_data,
        'Set2': _cm._Set2_data,
        'Set3': _cm._Set3_data,
    }

    # 构造函数，初始化 ColorSequenceRegistry 实例时调用
    def __init__(self):
        # 创建 _color_sequences 字典，复制内置颜色序列字典的内容
        self._color_sequences = {**self._BUILTIN_COLOR_SEQUENCES}

    # 实现字典索引操作，通过颜色序列的名称获取对应的颜色列表的副本
    def __getitem__(self, item):
        try:
            return list(self._color_sequences[item])
        except KeyError:
            raise KeyError(f"{item!r} is not a known color sequence name")

    # 实现迭代器接口，返回内置颜色序列字典的迭代器
    def __iter__(self):
        return iter(self._color_sequences)

    # 返回内置颜色序列字典的长度
    def __len__(self):
        return len(self._color_sequences)

    # 返回 ColorSequenceRegistry 的字符串表示形式，列出所有可用的颜色序列名称
    def __str__(self):
        return ('ColorSequenceRegistry; available colormaps:\n' +
                ', '.join(f"'{name}'" for name in self))
    def register(self, name, color_list):
        """
        Register a new color sequence.

        The color sequence registry stores a copy of the given *color_list*, so
        that future changes to the original list do not affect the registered
        color sequence. Think of this as the registry taking a snapshot
        of *color_list* at registration.

        Parameters
        ----------
        name : str
            The name for the color sequence.

        color_list : list of :mpltype:`color`
            An iterable returning valid Matplotlib colors when iterating over.
            Note however that the returned color sequence will always be a
            list regardless of the input type.

        """
        # Check if the provided name is reserved for built-in color sequences
        if name in self._BUILTIN_COLOR_SEQUENCES:
            raise ValueError(f"{name!r} is a reserved name for a builtin "
                             "color sequence")

        # Force a copy of the input color_list and ensure it's of type list
        color_list = list(color_list)  # force copy and coerce type to list
        
        # Validate each color in the color_list
        for color in color_list:
            try:
                to_rgba(color)  # Check if color can be converted to RGBA format
            except ValueError:
                raise ValueError(
                    f"{color!r} is not a valid color specification")

        # Store the validated color_list in the registry under the given name
        self._color_sequences[name] = color_list

    def unregister(self, name):
        """
        Remove a sequence from the registry.

        You cannot remove built-in color sequences.

        If the name is not registered, returns with no error.
        """
        # Check if the name corresponds to a built-in color sequence
        if name in self._BUILTIN_COLOR_SEQUENCES:
            raise ValueError(
                f"Cannot unregister builtin color sequence {name!r}")
        
        # Remove the color sequence associated with the provided name from the registry
        self._color_sequences.pop(name, None)
# 创建一个颜色序列注册表的实例
_color_sequences = ColorSequenceRegistry()


def _sanitize_extrema(ex):
    # 如果输入的ex为None，则直接返回None
    if ex is None:
        return ex
    try:
        # 尝试调用ex的item()方法，如果存在则返回其结果
        ret = ex.item()
    except AttributeError:
        # 如果ex没有item()方法，则将ex转换为float类型并返回
        ret = float(ex)
    return ret


# 正则表达式，用于匹配C后跟数字的字符串
_nth_color_re = re.compile(r"\AC[0-9]+\Z")


def _is_nth_color(c):
    """Return whether *c* can be interpreted as an item in the color cycle."""
    # 判断字符串c是否匹配_nth_color_re定义的正则表达式，如果匹配则返回True，否则返回False
    return isinstance(c, str) and _nth_color_re.match(c)


def is_color_like(c):
    """Return whether *c* can be interpreted as an RGB(A) color."""
    # 对于特殊情况的处理：判断c是否符合_nth_color_re定义的格式
    if _is_nth_color(c):
        return True
    try:
        # 尝试将c转换为RGBA颜色，如果能成功转换则返回True，否则返回False
        to_rgba(c)
    except (TypeError, ValueError):
        return False
    else:
        return True


def _has_alpha_channel(c):
    """Return whether *c* is a color with an alpha channel."""
    # 判断c是否不是字符串且长度为4，这种情况下认为c是一个带有alpha通道的颜色
    return not isinstance(c, str) and len(c) == 4


def _check_color_like(**kwargs):
    """
    For each *key, value* pair in *kwargs*, check that *value* is color-like.
    """
    # 遍历kwargs中的每对键值对，检查值是否为颜色格式
    for k, v in kwargs.items():
        if not is_color_like(v):
            # 如果值不是颜色格式，则抛出ValueError异常，说明不支持的输入格式
            raise ValueError(
                f"{v!r} is not a valid value for {k}: supported inputs are "
                f"(r, g, b) and (r, g, b, a) 0-1 float tuples; "
                f"'#rrggbb', '#rrggbbaa', '#rgb', '#rgba' strings; "
                f"named color strings; "
                f"string reprs of 0-1 floats for grayscale values; "
                f"'C0', 'C1', ... strings for colors of the color cycle; "
                f"and pairs combining one of the above with an alpha value")


def same_color(c1, c2):
    """
    Return whether the colors *c1* and *c2* are the same.

    *c1*, *c2* can be single colors or lists/arrays of colors.
    """
    # 将c1和c2转换为RGBA数组，确保可以进行颜色比较
    c1 = to_rgba_array(c1)
    c2 = to_rgba_array(c2)
    n1 = max(c1.shape[0], 1)  # 'none' results in shape (0, 4), but is 1-elem
    n2 = max(c2.shape[0], 1)  # 'none' results in shape (0, 4), but is 1-elem

    if n1 != n2:
        # 如果c1和c2的元素数量不相同，则抛出ValueError异常
        raise ValueError('Different number of elements passed.')
    # 进行形状和值的比较，确保c1和c2在RGBA空间中相同
    return c1.shape == c2.shape and (c1 == c2).all()


def to_rgba(c, alpha=None):
    """
    Convert *c* to an RGBA color.

    Parameters
    ----------
    c : Matplotlib color or ``np.ma.masked``

    alpha : float, optional
        If *alpha* is given, force the alpha value of the returned RGBA tuple
        to *alpha*.

        If None, the alpha value from *c* is used. If *c* does not have an
        alpha channel, then alpha defaults to 1.

        *alpha* is ignored for the color value ``"none"`` (case-insensitive),
        which always maps to ``(0, 0, 0, 0)``.

    Returns
    -------
    """
    # 如果参数 c 是元组且长度为 2
    if isinstance(c, tuple) and len(c) == 2:
        # 如果未提供 alpha 值，则从元组中获取并赋值给 alpha
        if alpha is None:
            c, alpha = c
        else:
            # 否则，只获取元组的第一个值赋给 c
            c = c[0]
    
    # 特殊处理第 n 个颜色的语法，因为它不应该被缓存
    if _is_nth_color(c):
        # 获取当前 Axes 的属性循环对象
        prop_cycler = mpl.rcParams['axes.prop_cycle']
        # 获取颜色列表，默认为 ['k'] (黑色)
        colors = prop_cycler.by_key().get('color', ['k'])
        # 根据第 n 个颜色的索引计算实际颜色值
        c = colors[int(c[1:]) % len(colors)]
    
    try:
        # 尝试从缓存中获取颜色 c 和 alpha 对应的 RGBA 值
        rgba = _colors_full_map.cache[c, alpha]
    except (KeyError, TypeError):  # 如果在缓存中找不到或者无法哈希化
        # 如果未能从缓存中获取到值，设为 None
        rgba = None
    
    if rgba is None:  # 如果 RGBA 值为 None
        # 调用函数 _to_rgba_no_colorcycle 计算颜色 c 和 alpha 对应的 RGBA 值
        rgba = _to_rgba_no_colorcycle(c, alpha)
        try:
            # 尝试将计算得到的 RGBA 值存入缓存
            _colors_full_map.cache[c, alpha] = rgba
        except TypeError:
            # 如果无法将值存入缓存（通常是因为 c 或 alpha 无法哈希化），忽略此错误
            pass
    
    # 返回计算得到的 RGBA 值
    return rgba
def _to_rgba_no_colorcycle(c, alpha=None):
    """
    Convert *c* to an RGBA color, with no support for color-cycle syntax.

    If *alpha* is given, force the alpha value of the returned RGBA tuple
    to *alpha*. Otherwise, the alpha value from *c* is used, if it has alpha
    information, or defaults to 1.

    *alpha* is ignored for the color value ``"none"`` (case-insensitive),
    which always maps to ``(0, 0, 0, 0)``.
    """
    # 如果 alpha 超出范围，则抛出 ValueError 异常
    if alpha is not None and not 0 <= alpha <= 1:
        raise ValueError("'alpha' must be between 0 and 1, inclusive")
    # 保存原始的颜色值
    orig_c = c
    # 如果 c 是 masked 数组，则返回全透明的 RGBA 值
    if c is np.ma.masked:
        return (0., 0., 0., 0.)
    # 如果 c 是字符串类型
    if isinstance(c, str):
        # 如果 c 是 "none"（不区分大小写），返回全透明的 RGBA 值
        if c.lower() == "none":
            return (0., 0., 0., 0.)
        # 尝试从全局颜色映射表中获取颜色值
        try:
            c = _colors_full_map[c]
        except KeyError:
            # 如果原始的 c 的长度不为 1，尝试以小写形式再次查找颜色映射
            if len(orig_c) != 1:
                try:
                    c = _colors_full_map[c.lower()]
                except KeyError:
                    pass
    # 再次检查 c 是否为字符串类型
    if isinstance(c, str):
        # 处理 #rrggbb 格式的十六进制颜色
        match = re.match(r"\A#[a-fA-F0-9]{6}\Z", c)
        if match:
            return (tuple(int(n, 16) / 255
                          for n in [c[1:3], c[3:5], c[5:7]])
                    + (alpha if alpha is not None else 1.,))
        # 处理 #rgb 格式的十六进制颜色，将其扩展为 #rrggbb 格式
        match = re.match(r"\A#[a-fA-F0-9]{3}\Z", c)
        if match:
            return (tuple(int(n, 16) / 255
                          for n in [c[1]*2, c[2]*2, c[3]*2])
                    + (alpha if alpha is not None else 1.,))
        # 处理 #rrggbbaa 格式的十六进制颜色，支持设置 alpha 值
        match = re.match(r"\A#[a-fA-F0-9]{8}\Z", c)
        if match:
            color = [int(n, 16) / 255
                     for n in [c[1:3], c[3:5], c[5:7], c[7:9]]]
            if alpha is not None:
                color[-1] = alpha
            return tuple(color)
        # 处理 #rgba 格式的十六进制颜色，将其扩展为 #rrggbbaa 格式
        match = re.match(r"\A#[a-fA-F0-9]{4}\Z", c)
        if match:
            color = [int(n, 16) / 255
                     for n in [c[1]*2, c[2]*2, c[3]*2, c[4]*2]]
            if alpha is not None:
                color[-1] = alpha
            return tuple(color)
        # 处理字符串形式的灰度值
        try:
            c = float(c)
        except ValueError:
            pass
        else:
            if not (0 <= c <= 1):
                raise ValueError(
                    f"Invalid string grayscale value {orig_c!r}. "
                    f"Value must be within 0-1 range")
            return c, c, c, alpha if alpha is not None else 1.
        # 如果以上都不匹配，则抛出异常
        raise ValueError(f"Invalid RGBA argument: {orig_c!r}")
    # 如果 c 是 numpy 的数组，将二维数组转换为一维数组
    if isinstance(c, np.ndarray):
        if c.ndim == 2 and c.shape[0] == 1:
            c = c.reshape(-1)
    # 返回处理后的颜色值
    # tuple 类型的颜色值
    # 如果输入的颜色 c 不是可迭代对象，则抛出 ValueError 异常，显示错误消息和原始参数值
    if not np.iterable(c):
        raise ValueError(f"Invalid RGBA argument: {orig_c!r}")
    # 如果颜色 c 的长度不是 3 或 4，则抛出 ValueError 异常
    if len(c) not in [3, 4]:
        raise ValueError("RGBA sequence should have length 3 or 4")
    # 如果颜色 c 中的所有元素不都是实数类型，则抛出 ValueError 异常，显示错误消息和原始参数值
    if not all(isinstance(x, Real) for x in c):
        # 某些转换方式如 `map(float, ...)`, `np.array(..., float)` 和 `np.array(...).astype(float)` 都会将 "0.5" 转换为 0.5，但这些检查方法都不适用。
        raise ValueError(f"Invalid RGBA argument: {orig_c!r}")
    # 将颜色 c 中的每个元素转换为 float 类型，并返回元组，以防止修改缓存的值
    c = tuple(map(float, c))
    # 如果颜色 c 的长度为 3 并且未指定 alpha 值，则将 alpha 设置为 1
    if len(c) == 3 and alpha is None:
        alpha = 1
    # 如果指定了 alpha 值，则将其添加到颜色 c 中
    if alpha is not None:
        c = c[:3] + (alpha,)
    # 如果颜色 c 中有任何元素小于 0 或大于 1，则抛出 ValueError 异常
    if any(elem < 0 or elem > 1 for elem in c):
        raise ValueError("RGBA values should be within 0-1 range")
    # 返回处理后的颜色 c
    return c
def to_rgba_array(c, alpha=None):
    """
    Convert *c* to a (n, 4) array of RGBA colors.

    Parameters
    ----------
    c : Matplotlib color or array of colors
        If *c* is a masked array, an `~numpy.ndarray` is returned with a
        (0, 0, 0, 0) row for each masked value or row in *c*.

    alpha : float or sequence of floats, optional
        If *alpha* is given, force the alpha value of the returned RGBA tuple
        to *alpha*.

        If None, the alpha value from *c* is used. If *c* does not have an
        alpha channel, then alpha defaults to 1.

        *alpha* is ignored for the color value ``"none"`` (case-insensitive),
        which always maps to ``(0, 0, 0, 0)``.

        If *alpha* is a sequence and *c* is a single color, *c* will be
        repeated to match the length of *alpha*.

    Returns
    -------
    array
        (n, 4) array of RGBA colors,  where each channel (red, green, blue,
        alpha) can assume values between 0 and 1.
    """
    # 如果 *c* 是一个长度为2的元组，且第二个元素是实数，则根据 *alpha* 设置透明度
    if isinstance(c, tuple) and len(c) == 2 and isinstance(c[1], Real):
        if alpha is None:
            c, alpha = c
        else:
            c = c[0]
    
    # 处理已经是数组的特殊情况，提高性能
    if np.iterable(alpha):
        alpha = np.asarray(alpha).ravel()
    
    # 如果 *c* 是一个二维数组且每行是三或四个通道的颜色值
    if (isinstance(c, np.ndarray) and c.dtype.kind in "if"
            and c.ndim == 2 and c.shape[1] in [3, 4]):
        # 处理掩码数组的情况
        mask = c.mask.any(axis=1) if np.ma.is_masked(c) else None
        c = np.ma.getdata(c)
        
        # 如果 *alpha* 是一个序列，且 *c* 是单个颜色，则重复 *c* 来匹配 *alpha* 的长度
        if np.iterable(alpha):
            if c.shape[0] == 1 and alpha.shape[0] > 1:
                c = np.tile(c, (alpha.shape[0], 1))
            elif c.shape[0] != alpha.shape[0]:
                raise ValueError("The number of colors must match the number"
                                 " of alpha values if there are more than one"
                                 " of each.")
        
        # 根据 *c* 的通道数来创建结果数组
        if c.shape[1] == 3:
            result = np.column_stack([c, np.zeros(len(c))])
            result[:, -1] = alpha if alpha is not None else 1.
        elif c.shape[1] == 4:
            result = c.copy()
            if alpha is not None:
                result[:, -1] = alpha
        
        # 处理掩码数组的值
        if mask is not None:
            result[mask] = 0
        
        # 检查结果数组是否在合法的范围内
        if np.any((result < 0) | (result > 1)):
            raise ValueError("RGBA values should be within 0-1 range")
        
        return result
    
    # 处理单个颜色值的情况
    if cbook._str_lower_equal(c, "none"):
        return np.zeros((0, 4), float)
    
    try:
        if np.iterable(alpha):
            return np.array([to_rgba(c, a) for a in alpha], float)
        else:
            return np.array([to_rgba(c, alpha)], float)
    # 如果捕获到 TypeError 异常，则忽略该异常继续执行
    except TypeError:
        pass
    # 如果捕获到 ValueError 异常，并且异常参数为 ("'alpha' must be between 0 and 1, inclusive", )
    # 则抛出该异常，该异常通常来自 _to_rgba_no_colorcycle() 函数
    except ValueError as e:
        if e.args == ("'alpha' must be between 0 and 1, inclusive", ):
            raise e

    # 如果 c 是字符串类型，则抛出 ValueError 异常，指出 c 不是有效的颜色值
    if isinstance(c, str):
        raise ValueError(f"{c!r} is not a valid color value.")

    # 如果 c 的长度为 0，则返回一个全为 0 的 numpy 数组，形状为 (0, 4)
    if len(c) == 0:
        return np.zeros((0, 4), float)

    # 如果 c 是 Sequence 类型，进入快速处理路径
    if isinstance(c, Sequence):
        # 计算 c 中每个元素的长度，并存储在集合 lens 中
        lens = {len(cc) if isinstance(cc, (list, tuple)) else -1 for cc in c}
        # 如果 lens 集合中只包含一个元素 3，则说明 c 中每个元素是 RGB 格式的颜色值
        if lens == {3}:
            rgba = np.column_stack([c, np.ones(len(c))])
        # 如果 lens 集合中只包含一个元素 4，则说明 c 中每个元素是 RGBA 格式的颜色值
        elif lens == {4}:
            rgba = np.array(c)
        # 否则，对 c 中每个元素调用 to_rgba() 函数，将结果存储在 numpy 数组中
        else:
            rgba = np.array([to_rgba(cc) for cc in c])
    else:
        # 如果 c 不是 Sequence 类型，则对 c 中每个元素调用 to_rgba() 函数，将结果存储在 numpy 数组中
        rgba = np.array([to_rgba(cc) for cc in c])

    # 如果 alpha 参数不为 None，则将 rgba 数组的最后一列（透明度值）设置为 alpha
    if alpha is not None:
        rgba[:, 3] = alpha
        # 如果 c 是 Sequence 类型，则确保显示指定的 alpha 不会覆盖 "none" 的完全透明
        none_mask = [cbook._str_equal(cc, "none") for cc in c]
        rgba[:, 3][none_mask] = 0

    # 返回处理后的 rgba 数组
    return rgba
def to_rgb(c):
    """Convert *c* to an RGB color, silently dropping the alpha channel."""
    # 调用 to_rgba 函数将颜色 c 转换为 RGBA 格式，并丢弃 alpha 通道，返回前三个值
    return to_rgba(c)[:3]


def to_hex(c, keep_alpha=False):
    """
    Convert *c* to a hex color.

    Parameters
    ----------
    c : :ref:`color <colors_def>` or `numpy.ma.masked`
        输入的颜色，可以是字符串描述或者 numpy.ma.masked 类型的对象

    keep_alpha : bool, default: False
        如果为 False，使用 `#rrggbb` 格式；否则使用 `#rrggbbaa` 格式

    Returns
    -------
    str
        `#rrggbb` 或者 `#rrggbbaa` 的十六进制颜色字符串
    """
    # 调用 to_rgba 函数将颜色 c 转换为 RGBA 格式
    c = to_rgba(c)
    # 如果 keep_alpha 为 False，则只保留 RGB 值，否则保留 RGBA 值
    if not keep_alpha:
        c = c[:3]
    # 将 RGBA 值转换为十六进制颜色字符串
    return "#" + "".join(format(round(val * 255), "02x") for val in c)


### Backwards-compatible color-conversion API


# 使用 CSS4_COLORS 中定义的颜色名称
cnames = CSS4_COLORS
# 匹配十六进制颜色值的正则表达式
hexColorPattern = re.compile(r"\A#[a-fA-F0-9]{6}\Z")
# rgb2hex 函数引用到 to_hex 函数
rgb2hex = to_hex
# hex2color 函数引用到 to_rgb 函数
hex2color = to_rgb


class ColorConverter:
    """
    A class only kept for backwards compatibility.

    Its functionality is entirely provided by module-level functions.
    """
    # 颜色映射表，由 _colors_full_map 提供
    colors = _colors_full_map
    # 缓存，与 _colors_full_map 相关
    cache = _colors_full_map.cache
    # 静态方法，通过 module-level 函数提供 to_rgb 转换功能
    to_rgb = staticmethod(to_rgb)
    # 静态方法，通过 module-level 函数提供 to_rgba 转换功能
    to_rgba = staticmethod(to_rgba)
    # 静态方法，通过 module-level 函数提供 to_rgba_array 转换功能
    to_rgba_array = staticmethod(to_rgba_array)


# ColorConverter 类的实例化对象
colorConverter = ColorConverter()


### End of backwards-compatible color-conversion API


def _create_lookup_table(N, data, gamma=1.0):
    r"""
    Create an *N* -element 1D lookup table.

    This assumes a mapping :math:`f : [0, 1] \rightarrow [0, 1]`. The returned
    data is an array of N values :math:`y = f(x)` where x is sampled from
    [0, 1].

    By default (*gamma* = 1) x is equidistantly sampled from [0, 1]. The
    *gamma* correction factor :math:`\gamma` distorts this equidistant
    sampling by :math:`x \rightarrow x^\gamma`.

    Parameters
    ----------
    N : int
        创建的查找表的元素个数，至少为1

    data : (M, 3) array-like or callable
        定义映射 :math:`f` 的数据结构或者函数

        如果是 (M, 3) 形式的数组或类似对象，其中每一行定义了 (x, y0, y1) 的值。x 的值必须从 x=0 开始，到 x=1 结束，并且所有 x 的值必须按递增顺序排列。

        如果是可调用的对象，必须接受并返回 numpy 数组：

           data(x : ndarray) -> ndarray

        并将 0 到 1 之间的值映射到 0 到 1 之间。

    gamma : float
        映射函数输入分布 x 的 gamma 修正因子

        参考 https://en.wikipedia.org/wiki/Gamma_correction

    Returns
    -------

    """
    # array
    #     表示查找表，其中 `lut[x * (N-1)]` 给出在 0 到 1 之间 x 值最接近的值
    
    Notes
    -----
    这个函数在 `.LinearSegmentedColormap` 内部使用。
    """
    
    if callable(data):
        # 如果 data 是可调用的，生成一个经过 gamma 校正的线性空间
        xind = np.linspace(0, 1, N) ** gamma
        # 调用 data 函数生成 lut，确保所有值在 0 到 1 之间
        lut = np.clip(np.array(data(xind), dtype=float), 0, 1)
        return lut
    
    try:
        # 将 data 转换为 numpy 数组
        adata = np.array(data)
    except Exception as err:
        # 如果无法转换，抛出类型错误
        raise TypeError("data must be convertible to an array") from err
    # 检查数据形状是否为 (None, 3)
    _api.check_shape((None, 3), data=adata)
    
    # 从 adata 中提取 x, y0, y1 列
    x = adata[:, 0]
    y0 = adata[:, 1]
    y1 = adata[:, 2]
    
    if x[0] != 0. or x[-1] != 1.0:
        # 如果映射点不是从 x=0 到 x=1，抛出数值错误
        raise ValueError(
            "data mapping points must start with x=0 and end with x=1")
    if (np.diff(x) < 0).any():
        # 如果 x 不是递增的，抛出数值错误
        raise ValueError("data mapping points must have x in increasing order")
    # 开始生成查找表
    if N == 1:
        # 如果 N 等于 1，使用 y0[-1] 作为查找表的唯一值
        lut = np.array(y0[-1])
    else:
        # 将 x 调整到 0 到 N-1 的范围
        x = x * (N - 1)
        # 生成 gamma 校正后的索引
        xind = (N - 1) * np.linspace(0, 1, N) ** gamma
        # 在 x 中搜索 xind 的索引，并排除边界点
        ind = np.searchsorted(x, xind)[1:-1]
    
        # 计算距离和插值
        distance = (xind[1:-1] - x[ind - 1]) / (x[ind] - x[ind - 1])
        lut = np.concatenate([
            [y1[0]],
            distance * (y0[ind] - y1[ind - 1]) + y1[ind - 1],
            [y0[-1]],
        ])
    # 确保 lut 的值在 0 到 1 之间
    return np.clip(lut, 0.0, 1.0)
    """
    Baseclass for all scalar to RGBA mappings.

    Typically, Colormap instances are used to convert data values (floats)
    from the interval ``[0, 1]`` to the RGBA color that the respective
    Colormap represents. For scaling of data into the ``[0, 1]`` interval see
    `matplotlib.colors.Normalize`. Subclasses of `matplotlib.cm.ScalarMappable`
    make heavy use of this ``data -> normalize -> map-to-color`` processing
    chain.
    """
class Colormap:
    def __init__(self, name, N=256):
        """
        Parameters
        ----------
        name : str
            The name of the colormap.
        N : int
            The number of RGB quantization levels.
        """
        # 初始化颜色映射对象的名称
        self.name = name
        # 确保 N 始终为整数
        self.N = int(N)  # ensure that N is always int
        # 如果数据无效，使用全透明的 RGBA 值
        self._rgba_bad = (0.0, 0.0, 0.0, 0.0)  # If bad, don't paint anything.
        self._rgba_under = None  # 如果数据低于范围，采用的 RGBA 值
        self._rgba_over = None  # 如果数据高于范围，采用的 RGBA 值
        self._i_under = self.N  # 用于标识低于范围的索引值
        self._i_over = self.N + 1  # 用于标识高于范围的索引值
        self._i_bad = self.N + 2  # 用于标识无效数据的索引值
        self._isinit = False  # 初始化标志，用于检查是否已初始化
        #: 当此颜色映射存在于标量映射对象和 colorbar_extend 不为 False 时，
        #: colorbar 创建将使用 colorbar_extend 作为 `extend` 关键字的默认值
        self.colorbar_extend = False
    def __call__(self, X, alpha=None, bytes=False):
        r"""
        Parameters
        ----------
        X : float or int, `~numpy.ndarray` or scalar
            The data value(s) to convert to RGBA.
            For floats, *X* should be in the interval ``[0.0, 1.0]`` to
            return the RGBA values ``X*100`` percent along the Colormap line.
            For integers, *X* should be in the interval ``[0, Colormap.N)`` to
            return RGBA values *indexed* from the Colormap with index ``X``.
        alpha : float or array-like or None
            Alpha must be a scalar between 0 and 1, a sequence of such
            floats with shape matching X, or None.
        bytes : bool
            If False (default), the returned RGBA values will be floats in the
            interval ``[0, 1]`` otherwise they will be `numpy.uint8`\s in the
            interval ``[0, 255]``.

        Returns
        -------
        Tuple of RGBA values if X is scalar, otherwise an array of
        RGBA values with a shape of ``X.shape + (4, )``.
        """
        # 如果对象未初始化，则进行初始化
        if not self._isinit:
            self._init()

        # 将输入数据 X 复制为 numpy 数组 xa
        xa = np.array(X, copy=True)

        # 如果 xa 的字节顺序不是本机字节顺序，则进行字节交换以提高性能
        if not xa.dtype.isnative:
            xa = xa.byteswap().view(xa.dtype.newbyteorder())

        # 如果 xa 的数据类型为浮点数，则将其乘以 colormap 的大小 N
        if xa.dtype.kind == "f":
            xa *= self.N
            # 对于 xa == N 的情况，将其设为 N-1，因为索引从 0 开始
            xa[xa == self.N] = self.N - 1

        # 预先计算掩码以处理溢出和负数的情况
        mask_under = xa < 0
        mask_over = xa >= self.N

        # 如果输入 X 是掩码数组，则从中获取坏掩码，否则从 xa 中检测 NaN
        mask_bad = X.mask if np.ma.is_masked(X) else np.isnan(xa)

        # 忽略无效值警告，将 xa 转换为整数类型，适用于无符号整数和浮点数
        with np.errstate(invalid="ignore"):
            xa = xa.astype(int)

        # 将溢出和负数的位置替换为预定义的索引值
        xa[mask_under] = self._i_under
        xa[mask_over] = self._i_over
        xa[mask_bad] = self._i_bad

        # 获取颜色映射表 lut
        lut = self._lut

        # 如果需要返回的是字节（0-255 范围内的整数），则将 lut 缩放到 0-255 范围内
        if bytes:
            lut = (lut * 255).astype(np.uint8)

        # 根据索引 xa 从 lut 中获取对应的 RGBA 值，使用 'clip' 模式防止索引越界
        rgba = lut.take(xa, axis=0, mode='clip')

        # 如果提供了 alpha 值，则对 RGBA 进行透明度处理
        if alpha is not None:
            alpha = np.clip(alpha, 0, 1)
            if bytes:
                alpha *= 255  # 将 alpha 缩放到 0-255 范围以便转换为 uint8
            # 如果 alpha 的形状不匹配 X 的形状，则抛出 ValueError
            if alpha.shape not in [(), xa.shape]:
                raise ValueError(
                    f"alpha is array-like but its shape {alpha.shape} does "
                    f"not match that of X {xa.shape}")
            rgba[..., -1] = alpha

            # 如果最后一个颜色值是全零（代表透明），则忽略输入的 alpha
            if (lut[-1] == 0).all():
                rgba[mask_bad] = (0, 0, 0, 0)

        # 如果 X 是标量，则返回单个 RGBA 元组，否则返回带有额外维度的 RGBA 数组
        if not np.iterable(X):
            rgba = tuple(rgba)
        return rgba
    # 定义一个特殊方法，用于复制 colormap 对象
    def __copy__(self):
        # 获取当前对象的类
        cls = self.__class__
        # 创建一个新的同类对象，但不调用初始化方法
        cmapobject = cls.__new__(cls)
        # 将当前对象的属性复制到新对象中
        cmapobject.__dict__.update(self.__dict__)
        # 如果当前对象已初始化，则复制查找表（lookup table）
        if self._isinit:
            cmapobject._lut = np.copy(self._lut)
        # 返回新对象
        return cmapobject

    # 定义一个特殊方法，用于判断两个 colormap 对象是否相等
    def __eq__(self, other):
        # 如果 other 不是 Colormap 类的实例，或者 colorbar_extend 属性不相等，则返回 False
        if (not isinstance(other, Colormap) or
                self.colorbar_extend != other.colorbar_extend):
            return False
        # 比较两个 colormap 的查找表，需要先确保两个对象都已初始化
        if not self._isinit:
            self._init()
        if not other._isinit:
            other._init()
        # 返回查找表是否相等的比较结果
        return np.array_equal(self._lut, other._lut)

    # 获取用于 masked 值的颜色
    def get_bad(self):
        # 如果对象尚未初始化，则进行初始化
        if not self._isinit:
            self._init()
        # 返回指定索引处的颜色值
        return np.array(self._lut[self._i_bad])

    # 设置 masked 值的颜色
    def set_bad(self, color='k', alpha=None):
        # 将输入的颜色和透明度转换为 RGBA 格式
        self._rgba_bad = to_rgba(color, alpha)
        # 如果对象已初始化，则重新设置极值
        if self._isinit:
            self._set_extremes()

    # 获取用于低于范围的颜色
    def get_under(self):
        # 如果对象尚未初始化，则进行初始化
        if not self._isinit:
            self._init()
        # 返回指定索引处的颜色值
        return np.array(self._lut[self._i_under])

    # 设置低于范围的颜色
    def set_under(self, color='k', alpha=None):
        # 将输入的颜色和透明度转换为 RGBA 格式
        self._rgba_under = to_rgba(color, alpha)
        # 如果对象已初始化，则重新设置极值
        if self._isinit:
            self._set_extremes()

    # 获取用于高于范围的颜色
    def get_over(self):
        # 如果对象尚未初始化，则进行初始化
        if not self._isinit:
            self._init()
        # 返回指定索引处的颜色值
        return np.array(self._lut[self._i_over])

    # 设置高于范围的颜色
    def set_over(self, color='k', alpha=None):
        # 将输入的颜色和透明度转换为 RGBA 格式
        self._rgba_over = to_rgba(color, alpha)
        # 如果对象已初始化，则重新设置极值
        if self._isinit:
            self._set_extremes()

    # 设置异常值（如 masked、低于范围和高于范围值）的颜色
    def set_extremes(self, *, bad=None, under=None, over=None):
        # 如果指定了 bad，则设置对应颜色
        if bad is not None:
            self.set_bad(bad)
        # 如果指定了 under，则设置对应颜色
        if under is not None:
            self.set_under(under)
        # 如果指定了 over，则设置对应颜色
        if over is not None:
            self.set_over(over)

    # 返回一个设置了异常值颜色的 colormap 的副本
    def with_extremes(self, *, bad=None, under=None, over=None):
        # 复制当前 colormap 对象
        new_cm = self.copy()
        # 设置副本对象的异常值颜色
        new_cm.set_extremes(bad=bad, under=under, over=over)
        # 返回设置了异常值颜色的 colormap 副本
        return new_cm
    def _set_extremes(self):
        # 如果定义了下限颜色，则将下限索引位置的颜色设置为定义的下限颜色，否则使用第一个颜色
        if self._rgba_under:
            self._lut[self._i_under] = self._rgba_under
        else:
            self._lut[self._i_under] = self._lut[0]
        
        # 如果定义了上限颜色，则将上限索引位置的颜色设置为定义的上限颜色，否则使用最后一个颜色
        if self._rgba_over:
            self._lut[self._i_over] = self._rgba_over
        else:
            self._lut[self._i_over] = self._lut[self.N - 1]
        
        # 将坏数据索引位置的颜色设置为定义的坏数据颜色
        self._lut[self._i_bad] = self._rgba_bad

    def _init(self):
        """Generate the lookup table, ``self._lut``."""
        # 抛出未实现错误，因为这是抽象类的方法，需要在子类中实现
        raise NotImplementedError("Abstract class only")

    def is_gray(self):
        """Return whether the colormap is grayscale."""
        # 如果未初始化，则调用初始化方法
        if not self._isinit:
            self._init()
        
        # 检查是否是灰度颜色映射
        return (np.all(self._lut[:, 0] == self._lut[:, 1]) and
                np.all(self._lut[:, 0] == self._lut[:, 2]))

    def resampled(self, lutsize):
        """Return a new colormap with *lutsize* entries."""
        # 如果存在_resample方法，则警告用户，因为现在可以公开使用重新采样颜色映射的能力
        if hasattr(self, '_resample'):
            _api.warn_external(
                "The ability to resample a color map is now public API "
                f"However the class {type(self)} still only implements "
                "the previous private _resample method.  Please update "
                "your class."
            )
            # 调用_resample方法并返回重新采样后的颜色映射
            return self._resample(lutsize)
        
        # 如果不存在_resample方法，则抛出未实现错误
        raise NotImplementedError()

    def reversed(self, name=None):
        """
        Return a reversed instance of the Colormap.

        .. note:: This function is not implemented for the base class.

        Parameters
        ----------
        name : str, optional
            The name for the reversed colormap. If None, the
            name is set to ``self.name + "_r"``.

        See Also
        --------
        LinearSegmentedColormap.reversed
        ListedColormap.reversed
        """
        # 抛出未实现错误，因为这个方法在基类中没有实现，只能在子类中实现
        raise NotImplementedError()

    def _repr_png_(self):
        """Generate a PNG representation of the Colormap."""
        # 创建一个线性分段的颜色映射的像素数据，并将其保存为 PNG 格式的字节流
        X = np.tile(np.linspace(0, 1, _REPR_PNG_SIZE[0]),
                    (_REPR_PNG_SIZE[1], 1))
        pixels = self(X, bytes=True)
        png_bytes = io.BytesIO()
        title = self.name + ' colormap'
        author = f'Matplotlib v{mpl.__version__}, https://matplotlib.org'
        pnginfo = PngInfo()
        pnginfo.add_text('Title', title)
        pnginfo.add_text('Description', title)
        pnginfo.add_text('Author', author)
        pnginfo.add_text('Software', author)
        Image.fromarray(pixels).save(png_bytes, format='png', pnginfo=pnginfo)
        return png_bytes.getvalue()
    # 生成用于显示颜色映射的 HTML 表示
    def _repr_html_(self):
        """Generate an HTML representation of the Colormap."""
        # 生成表示 colormap 的 PNG 图片的字节流
        png_bytes = self._repr_png_()
        # 将 PNG 图片的字节流转换为 base64 编码的 ASCII 字符串
        png_base64 = base64.b64encode(png_bytes).decode('ascii')
        
        # 定义生成每种颜色块的函数
        def color_block(color):
            # 将颜色转换为十六进制表示，并保留 alpha 通道
            hex_color = to_hex(color, keep_alpha=True)
            # 生成包含颜色块的 HTML，包括颜色的十六进制表示和样式
            return (f'<div title="{hex_color}" '
                    'style="display: inline-block; '
                    'width: 1em; height: 1em; '
                    'margin: 0; '
                    'vertical-align: middle; '
                    'border: 1px solid #555; '
                    f'background-color: {hex_color};"></div>')
        
        # 返回 HTML 结构，显示 colormap 的名称、图片、颜色块等内容
        return ('<div style="vertical-align: middle;">'
                f'<strong>{self.name}</strong> '
                '</div>'
                '<div class="cmap"><img '
                f'alt="{self.name} colormap" '
                f'title="{self.name}" '
                'style="border: 1px solid #555;" '
                f'src="data:image/png;base64,{png_base64}"></div>'
                '<div style="vertical-align: middle; '
                f'max-width: {_REPR_PNG_SIZE[0]+2}px; '
                'display: flex; justify-content: space-between;">'
                '<div style="float: left;">'
                f'{color_block(self.get_under())} under'
                '</div>'
                '<div style="margin: 0 auto; display: inline-block;">'
                f'bad {color_block(self.get_bad())}'
                '</div>'
                '<div style="float: right;">'
                f'over {color_block(self.get_over())}'
                '</div>')

    # 返回颜色映射的副本
    def copy(self):
        """Return a copy of the colormap."""
        return self.__copy__()
class LinearSegmentedColormap(Colormap):
    """
    Colormap objects based on lookup tables using linear segments.

    The lookup table is generated using linear interpolation for each
    primary color, with the 0-1 domain divided into any number of
    segments.
    """

    def __init__(self, name, segmentdata, N=256, gamma=1.0):
        """
        Create colormap from linear mapping segments

        segmentdata argument is a dictionary with a red, green and blue
        entries. Each entry should be a list of *x*, *y0*, *y1* tuples,
        forming rows in a table. Entries for alpha are optional.

        Example: suppose you want red to increase from 0 to 1 over
        the bottom half, green to do the same over the middle half,
        and blue over the top half.  Then you would use::

            cdict = {'red':   [(0.0,  0.0, 0.0),
                               (0.5,  1.0, 1.0),
                               (1.0,  1.0, 1.0)],

                     'green': [(0.0,  0.0, 0.0),
                               (0.25, 0.0, 0.0),
                               (0.75, 1.0, 1.0),
                               (1.0,  1.0, 1.0)],

                     'blue':  [(0.0,  0.0, 0.0),
                               (0.5,  0.0, 0.0),
                               (1.0,  1.0, 1.0)]}

        Each row in the table for a given color is a sequence of
        *x*, *y0*, *y1* tuples.  In each sequence, *x* must increase
        monotonically from 0 to 1.  For any input value *z* falling
        between *x[i]* and *x[i+1]*, the output value of a given color
        will be linearly interpolated between *y1[i]* and *y0[i+1]*::

            row i:   x  y0  y1
                           /
                          /
            row i+1: x  y0  y1

        Hence y0 in the first row and y1 in the last row are never used.

        See Also
        --------
        LinearSegmentedColormap.from_list
            Static method; factory function for generating a smoothly-varying
            LinearSegmentedColormap.
        """
        # True only if all colors in map are identical; needed for contouring.
        self.monochrome = False
        # Initialize the superclass Colormap with the given name and number of colors
        super().__init__(name, N)
        # Store segmentdata and gamma value as attributes of the colormap
        self._segmentdata = segmentdata
        self._gamma = gamma

    def _init(self):
        # Initialize the lookup table with ones and allocate space for color values
        self._lut = np.ones((self.N + 3, 4), float)
        # Populate the lookup table with interpolated color values for red, green, and blue channels
        self._lut[:-3, 0] = _create_lookup_table(
            self.N, self._segmentdata['red'], self._gamma)
        self._lut[:-3, 1] = _create_lookup_table(
            self.N, self._segmentdata['green'], self._gamma)
        self._lut[:-3, 2] = _create_lookup_table(
            self.N, self._segmentdata['blue'], self._gamma)
        # If alpha values are specified in segmentdata, populate the alpha channel in the lookup table
        if 'alpha' in self._segmentdata:
            self._lut[:-3, 3] = _create_lookup_table(
                self.N, self._segmentdata['alpha'], 1)
        # Mark initialization as complete
        self._isinit = True
        # Set the extremes of the colormap
        self._set_extremes()
    # 设置新的 gamma 值并重新生成颜色映射
    def set_gamma(self, gamma):
        self._gamma = gamma  # 设置对象的 gamma 属性为给定值
        self._init()  # 调用 _init 方法重新初始化对象

    @staticmethod
    def from_list(name, colors, N=256, gamma=1.0):
        """
        从颜色列表创建一个线性分段的颜色映射。

        Parameters
        ----------
        name : str
            颜色映射的名称。
        colors : list of :mpltype:`color` or list of (value, color)
            如果只提供颜色列表，则它们将等距映射到范围 [0, 1]；
            即 0 映射到 colors[0]，1 映射到 colors[-1]。
            如果提供了 (value, color) 对，则映射是从 value 到 color。
            这可以用来不均匀地划分范围。
        N : int
            RGB 量化级别的数量。
        gamma : float
            Gamma 校正值。
        """
        if not np.iterable(colors):
            raise ValueError('colors must be iterable')

        if (isinstance(colors[0], Sized) and len(colors[0]) == 2
                and not isinstance(colors[0], str)):
            # 如果 colors 是值、颜色对的列表
            vals, colors = zip(*colors)
        else:
            vals = np.linspace(0, 1, len(colors))

        r, g, b, a = to_rgba_array(colors).T
        # 创建颜色字典 cdict
        cdict = {
            "red": np.column_stack([vals, r, r]),
            "green": np.column_stack([vals, g, g]),
            "blue": np.column_stack([vals, b, b]),
            "alpha": np.column_stack([vals, a, a]),
        }

        return LinearSegmentedColormap(name, cdict, N, gamma)

    # 返回具有新 lutsize 条目的新颜色映射
    def resampled(self, lutsize):
        new_cmap = LinearSegmentedColormap(self.name, self._segmentdata,
                                           lutsize)
        new_cmap._rgba_over = self._rgba_over
        new_cmap._rgba_under = self._rgba_under
        new_cmap._rgba_bad = self._rgba_bad
        return new_cmap

    # 辅助函数，确保反转的颜色映射的可序列化
    @staticmethod
    def _reverser(func, x):
        return func(1 - x)
    def reversed(self, name=None):
        """
        Return a reversed instance of the Colormap.

        Parameters
        ----------
        name : str, optional
            The name for the reversed colormap. If None, the
            name is set to ``self.name + "_r"``.

        Returns
        -------
        LinearSegmentedColormap
            The reversed colormap.
        """
        if name is None:
            name = self.name + "_r"

        # 使用部分对象使得该颜色映射可序列化。
        # 对每个分段数据进行反转处理，如果数据可调用，则使用functools.partial包装反转函数；
        # 否则，对数据内部的每组三元组(x, y0, y1)进行反转操作。
        data_r = {key: (functools.partial(self._reverser, data)
                        if callable(data) else
                        [(1.0 - x, y1, y0) for x, y0, y1 in reversed(data)])
                  for key, data in self._segmentdata.items()}

        # 创建一个新的线性分段颜色映射对象，用反转后的数据data_r和原对象的其他属性初始化。
        new_cmap = LinearSegmentedColormap(name, data_r, self.N, self._gamma)
        # 将原对象的上/下限颜色值进行反转处理。
        new_cmap._rgba_over = self._rgba_under
        new_cmap._rgba_under = self._rgba_over
        new_cmap._rgba_bad = self._rgba_bad
        # 返回新的反转后的颜色映射对象。
        return new_cmap
# 定义一个从颜色列表生成的颜色映射对象
class ListedColormap(Colormap):
    """
    Colormap object generated from a list of colors.

    This may be most useful when indexing directly into a colormap,
    but it can also be used to generate special colormaps for ordinary
    mapping.

    Parameters
    ----------
    colors : list, array
        Sequence of Matplotlib color specifications (color names or RGB(A)
        values).
    name : str, optional
        String to identify the colormap.
    N : int, optional
        Number of entries in the map. The default is *None*, in which case
        there is one colormap entry for each element in the list of colors.
        If ::
    
            N < len(colors)
    
        the list will be truncated at *N*. If ::
    
            N > len(colors)
    
        the list will be extended by repetition.
    """
    
    # 初始化方法，接受颜色列表和可选的名称和条目数
    def __init__(self, colors, name='from_list', N=None):
        # 是否为单色映射？（用于 contour.py）
        self.monochrome = False
        # 如果条目数 N 为 None，则直接使用给定的颜色列表
        if N is None:
            self.colors = colors
            N = len(colors)
        else:
            # 如果颜色是字符串，则复制 N 次形成列表，并标记为单色映射
            if isinstance(colors, str):
                self.colors = [colors] * N
                self.monochrome = True
            # 如果颜色是可迭代对象，则循环使用颜色列表形成 N 个颜色
            elif np.iterable(colors):
                if len(colors) == 1:
                    self.monochrome = True
                # 使用 itertools.cycle 循环迭代颜色列表形成 N 个颜色
                self.colors = list(
                    itertools.islice(itertools.cycle(colors), N))
            else:
                # 如果颜色可以转换为浮点数，则使用该灰度值形成 N 个颜色
                try:
                    gray = float(colors)
                except TypeError:
                    pass
                else:
                    self.colors = [gray] * N
                self.monochrome = True
        # 调用父类的初始化方法，设置名称和条目数
        super().__init__(name, N)

    # 初始化私有方法，初始化颜色查找表
    def _init(self):
        self._lut = np.zeros((self.N + 3, 4), float)
        # 将颜色列表转换为 RGBA 数组并存储到查找表中
        self._lut[:-3] = to_rgba_array(self.colors)
        self._isinit = True
        self._set_extremes()

    # 返回一个包含 lutsize 个条目的新颜色映射对象
    def resampled(self, lutsize):
        """Return a new colormap with *lutsize* entries."""
        # 使用当前颜色映射对象生成等间隔的 lutsize 个颜色
        colors = self(np.linspace(0, 1, lutsize))
        # 创建新的 ListedColormap 对象，保留原始的上下限值
        new_cmap = ListedColormap(colors, name=self.name)
        new_cmap._rgba_over = self._rgba_over
        new_cmap._rgba_under = self._rgba_under
        new_cmap._rgba_bad = self._rgba_bad
        return new_cmap
    # 定义一个方法 `reversed`，用于生成当前颜色映射的反转实例
    def reversed(self, name=None):
        """
        Return a reversed instance of the Colormap.

        Parameters
        ----------
        name : str, optional
            The name for the reversed colormap. If None, the
            name is set to ``self.name + "_r"``.

        Returns
        -------
        ListedColormap
            A reversed instance of the colormap.
        """
        # 如果未提供名称，则使用原颜色映射名称加上后缀 "_r"
        if name is None:
            name = self.name + "_r"

        # 将颜色列表反转
        colors_r = list(reversed(self.colors))
        # 创建一个新的 ListedColormap 实例，使用反转后的颜色列表和指定的名称和 N 值
        new_cmap = ListedColormap(colors_r, name=name, N=self.N)
        
        # 反转 over/under 值
        new_cmap._rgba_over = self._rgba_under
        new_cmap._rgba_under = self._rgba_over
        new_cmap._rgba_bad = self._rgba_bad
        
        # 返回新的颜色映射实例
        return new_cmap
    """
    A class which, when called, maps values within the interval
    ``[vmin, vmax]`` linearly to the interval ``[0.0, 1.0]``. The mapping of
    values outside ``[vmin, vmax]`` depends on *clip*.

    Examples
    --------
    ::

        x = [-2, -1, 0, 1, 2]

        norm = mpl.colors.Normalize(vmin=-1, vmax=1, clip=False)
        norm(x)  # [-0.5, 0., 0.5, 1., 1.5]
        norm = mpl.colors.Normalize(vmin=-1, vmax=1, clip=True)
        norm(x)  # [0., 0., 0.5, 1., 1.]

    See Also
    --------
    :ref:`colormapnorms`
    """
    
    def __init__(self, vmin=None, vmax=None, clip=False):
        """
        Parameters
        ----------
        vmin, vmax : float or None
            Values within the range ``[vmin, vmax]`` from the input data will be
            linearly mapped to ``[0, 1]``. If either *vmin* or *vmax* is not
            provided, they default to the minimum and maximum values of the input,
            respectively.

        clip : bool, default: False
            Determines the behavior for mapping values outside the range
            ``[vmin, vmax]``.

            If clipping is off, values outside the range ``[vmin, vmax]`` are
            also transformed, resulting in values outside ``[0, 1]``.  This
            behavior is usually desirable, as colormaps can mark these *under*
            and *over* values with specific colors.

            If clipping is on, values below *vmin* are mapped to 0 and values
            above *vmax* are mapped to 1. Such values become indistinguishable
            from regular boundary values, which may cause misinterpretation of
            the data.

        Notes
        -----
        If ``vmin == vmax``, input data will be mapped to 0.
        """
        # 对输入的极值进行清理和赋值
        self._vmin = _sanitize_extrema(vmin)
        self._vmax = _sanitize_extrema(vmax)
        self._clip = clip
        self._scale = None
        # 定义回调函数，用于处理“changed”信号的监听器
        self.callbacks = cbook.CallbackRegistry(signals=["changed"])

    @property
    def vmin(self):
        # 返回当前的 vmin 值
        return self._vmin

    @vmin.setter
    def vmin(self, value):
        # 设置 vmin 值，并在值发生变化时调用回调函数
        value = _sanitize_extrema(value)
        if value != self._vmin:
            self._vmin = value
            self._changed()

    @property
    def vmax(self):
        # 返回当前的 vmax 值
        return self._vmax

    @vmax.setter
    def vmax(self, value):
        # 设置 vmax 值，并在值发生变化时调用回调函数
        value = _sanitize_extrema(value)
        if value != self._vmax:
            self._vmax = value
            self._changed()

    @property
    def clip(self):
        # 返回当前的 clip 值
        return self._clip

    @clip.setter
    def clip(self, value):
        # 设置 clip 值，并在值发生变化时调用回调函数
        if value != self._clip:
            self._clip = value
            self._changed()

    def _changed(self):
        """
        Call this whenever the norm is changed to notify all the
        callback listeners to the 'changed' signal.
        """
        # 调用此方法以通知所有监听器“changed”信号的变化
        self.callbacks.process('changed')

    @staticmethod
    # 定义函数 process_value，用于对输入的值进行归一化处理和规范化

    """
    Homogenize the input *value* for easy and efficient normalization.

    *value* can be a scalar or sequence.

    Parameters
    ----------
    value
        Data to normalize.

    Returns
    -------
    result : masked array
        Masked array with the same shape as *value*.
    is_scalar : bool
        Whether *value* is a scalar.

    Notes
    -----
    Float dtypes are preserved; integer types with two bytes or smaller are
    converted to np.float32, and larger types are converted to np.float64.
    Preserving float32 when possible, and using in-place operations,
    greatly improves speed for large arrays.
    """

    # 检查 value 是否为标量（非可迭代对象）
    is_scalar = not np.iterable(value)
    if is_scalar:
        # 如果是标量，则转换为单元素列表
        value = [value]

    # 确定 value 中元素的最小标量类型
    dtype = np.min_scalar_type(value)

    # 根据数据类型进行处理：bool_/int8/int16 转换为 float32；int32/int64 转换为 float64
    if np.issubdtype(dtype, np.integer) or dtype.type is np.bool_:
        dtype = np.promote_types(dtype, np.float32)

    # 确保将数据作为 ndarray 子类解释，以解决问题 #6622
    mask = np.ma.getmask(value)
    data = np.asarray(value)

    # 创建一个带有屏蔽值的 masked array，使用与 value 类型相同的 dtype
    result = np.ma.array(data, mask=mask, dtype=dtype, copy=True)
    
    # 返回结果数组和是否为标量的布尔值
    return result, is_scalar
    def __call__(self, value, clip=None):
        """
        Normalize the data and return the normalized data.

        Parameters
        ----------
        value
            Data to normalize.
        clip : bool, optional
            See the description of the parameter *clip* in `.Normalize`.

            If ``None``, defaults to ``self.clip`` (which defaults to
            ``False``).

        Notes
        -----
        If not already initialized, ``self.vmin`` and ``self.vmax`` are
        initialized using ``self.autoscale_None(value)``.
        """
        # 如果 clip 参数为 None，则使用 self.clip 的默认值
        if clip is None:
            clip = self.clip

        # 对输入的 value 进行处理，获取处理后的结果和是否是标量值的信息
        result, is_scalar = self.process_value(value)

        # 如果 self.vmin 或 self.vmax 任一为 None，则使用 autoscale_None 方法初始化它们
        if self.vmin is None or self.vmax is None:
            self.autoscale_None(result)

        # 将 self.vmin 和 self.vmax 转换为 float 类型，确保不丢失精度
        (vmin,), _ = self.process_value(self.vmin)
        (vmax,), _ = self.process_value(self.vmax)

        # 如果 self.vmin 等于 self.vmax，则将 result 填充为 0
        if vmin == vmax:
            result.fill(0)  # Or should it be all masked?  Or 0.5?
        # 如果 self.vmin 大于 self.vmax，则抛出 ValueError
        elif vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        else:
            # 如果 clip 为 True，则对 result 进行剪裁操作
            if clip:
                mask = np.ma.getmask(result)
                result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                     mask=mask)
            # 进行数据归一化计算
            resdat = result.data
            resdat -= vmin
            resdat /= (vmax - vmin)
            result = np.ma.array(resdat, mask=result.mask, copy=False)

        # 如果 is_scalar 为 True，则将 result 转换为标量值返回
        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        """
        Maps the normalized value (i.e., index in the colormap) back to image
        data value.

        Parameters
        ----------
        value
            Normalized value.
        """
        # 如果 vmin 或 vmax 未设置，则抛出 ValueError
        if not self.scaled():
            raise ValueError("Not invertible until both vmin and vmax are set")

        # 获取 self.vmin 和 self.vmax 的数值
        (vmin,), _ = self.process_value(self.vmin)
        (vmax,), _ = self.process_value(self.vmax)

        # 如果 value 是可迭代对象，则对每个元素进行反归一化计算
        if np.iterable(value):
            val = np.ma.asarray(value)
            return vmin + val * (vmax - vmin)
        else:
            # 对单个值进行反归一化计算
            return vmin + value * (vmax - vmin)

    def autoscale(self, A):
        """Set *vmin*, *vmax* to min, max of *A*."""
        # 阻止回调以便在更新时只获得单个更新信号
        with self.callbacks.blocked():
            self.vmin = self.vmax = None
            # 使用 autoscale_None 方法设置 self.vmin 和 self.vmax
            self.autoscale_None(A)
        # 标记对象状态已更改
        self._changed()
    # 如果 *vmin* 或 *vmax* 没有设置，使用 *A* 的最小值和最大值来设置它们
    def autoscale_None(self, A):
        """If *vmin* or *vmax* are not set, use the min/max of *A* to set them."""
        A = np.asanyarray(A)  # 将输入转换为 NumPy 数组的任意类型表示形式

        if isinstance(A, np.ma.MaskedArray):
            # 对于掩码数组，需要区分数组本身、False 和 np.bool_(False)
            if A.mask is False or not A.mask.shape:
                A = A.data  # 如果掩码为 False 或者没有形状，则使用实际数据

        if self.vmin is None and A.size:
            self.vmin = A.min()  # 如果 self.vmin 未设置且 A 非空，则设置为 A 的最小值
        if self.vmax is None and A.size:
            self.vmax = A.max()  # 如果 self.vmax 未设置且 A 非空，则设置为 A 的最大值

    # 返回 *vmin* 和 *vmax* 是否都已设置
    def scaled(self):
        """Return whether *vmin* and *vmax* are both set."""
        return self.vmin is not None and self.vmax is not None
class TwoSlopeNorm(Normalize):
    def __init__(self, vcenter, vmin=None, vmax=None):
        """
        Normalize data with a set center.

        Useful when mapping data with an unequal rates of change around a
        conceptual center, e.g., data that range from -2 to 4, with 0 as
        the midpoint.

        Parameters
        ----------
        vcenter : float
            The data value that defines ``0.5`` in the normalization.
        vmin : float, optional
            The data value that defines ``0.0`` in the normalization.
            Defaults to the min value of the dataset.
        vmax : float, optional
            The data value that defines ``1.0`` in the normalization.
            Defaults to the max value of the dataset.

        Examples
        --------
        This maps data value -4000 to 0., 0 to 0.5, and +10000 to 1.0; data
        between is linearly interpolated::

            >>> import matplotlib.colors as mcolors
            >>> offset = mcolors.TwoSlopeNorm(vmin=-4000.,
            ...                               vcenter=0., vmax=10000)
            >>> data = [-4000., -2000., 0., 2500., 5000., 7500., 10000.]
            >>> offset(data)
            array([0., 0.25, 0.5, 0.625, 0.75, 0.875, 1.0])
        """

        # 调用父类的初始化方法，设置 vmin 和 vmax
        super().__init__(vmin=vmin, vmax=vmax)
        # 设置中心值 vcenter
        self._vcenter = vcenter
        # 如果设置了 vcenter、vmax，并且 vcenter 大于等于 vmax，抛出异常
        if vcenter is not None and vmax is not None and vcenter >= vmax:
            raise ValueError('vmin, vcenter, and vmax must be in '
                             'ascending order')
        # 如果设置了 vcenter、vmin，并且 vcenter 小于等于 vmin，抛出异常
        if vcenter is not None and vmin is not None and vcenter <= vmin:
            raise ValueError('vmin, vcenter, and vmax must be in '
                             'ascending order')

    @property
    def vcenter(self):
        # 返回中心值 vcenter
        return self._vcenter

    @vcenter.setter
    def vcenter(self, value):
        # 如果传入的值与当前的中心值不同，则更新中心值并触发 _changed 方法
        if value != self._vcenter:
            self._vcenter = value
            self._changed()

    def autoscale_None(self, A):
        """
        Get vmin and vmax.

        If vcenter isn't in the range [vmin, vmax], either vmin or vmax
        is expanded so that vcenter lies in the middle of the modified range
        [vmin, vmax].
        """
        # 调用父类的 autoscale_None 方法，获取 vmin 和 vmax
        super().autoscale_None(A)
        # 如果当前的 vmin 大于等于 vcenter，则将 vmin 扩展到 vcenter 的左侧
        if self.vmin >= self.vcenter:
            self.vmin = self.vcenter - (self.vmax - self.vcenter)
        # 如果当前的 vmax 小于等于 vcenter，则将 vmax 扩展到 vcenter 的右侧
        if self.vmax <= self.vcenter:
            self.vmax = self.vcenter + (self.vcenter - self.vmin)
    def __call__(self, value, clip=None):
        """
        将值映射到区间 [0, 1]。*clip* 参数未使用。
        """
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)  # 如果 vmin 和 vmax 为 None，则设置它们

        # 检查 vmin <= vcenter <= vmax 是否成立
        if not self.vmin <= self.vcenter <= self.vmax:
            raise ValueError("vmin, vcenter, vmax 必须单调增加")

        # 在计算 tick locators 时需要注意我们必须外推:
        # 使用 np.interp 对 result 进行插值，将其映射到 [0, 0.5, 1] 区间，并进行遮蔽处理
        result = np.ma.masked_array(
            np.interp(result, [self.vmin, self.vcenter, self.vmax],
                      [0, 0.5, 1], left=-np.inf, right=np.inf),
            mask=np.ma.getmask(result))

        # 如果是标量，则将 result 转换为至少一维数组
        if is_scalar:
            result = np.atleast_1d(result)[0]

        # 返回映射后的结果
        return result

    def inverse(self, value):
        """
        反向映射：将 [0, 1] 区间内的值反向映射到原始数据空间。
        如果 vmin 和 vmax 未设置，则抛出 ValueError。
        """
        if not self.scaled():
            raise ValueError("只有在设置了 vmin 和 vmax 后才能进行反向映射")

        # 分别处理 vmin, vmax, vcenter 的值，确保它们是标量
        (vmin,), _ = self.process_value(self.vmin)
        (vmax,), _ = self.process_value(self.vmax)
        (vcenter,), _ = self.process_value(self.vcenter)

        # 使用 np.interp 对 value 在 [0, 0.5, 1] 区间内进行插值，反向映射到原始数据空间
        result = np.interp(value, [0, 0.5, 1], [vmin, vcenter, vmax],
                           left=-np.inf, right=np.inf)
        
        # 返回反向映射后的结果
        return result
class CenteredNorm(Normalize):
    def __init__(self, vcenter=0, halfrange=None, clip=False):
        """
        Normalize symmetrical data around a center (0 by default).

        Unlike `TwoSlopeNorm`, `CenteredNorm` applies an equal rate of change
        around the center.

        Useful when mapping symmetrical data around a conceptual center
        e.g., data that range from -2 to 4, with 0 as the midpoint, and
        with equal rates of change around that midpoint.

        Parameters
        ----------
        vcenter : float, default: 0
            The data value that defines ``0.5`` in the normalization.
        halfrange : float, optional
            The range of data values that defines a range of ``0.5`` in the
            normalization, so that *vcenter* - *halfrange* is ``0.0`` and
            *vcenter* + *halfrange* is ``1.0`` in the normalization.
            Defaults to the largest absolute difference to *vcenter* for
            the values in the dataset.
        clip : bool, default: False
            Determines the behavior for mapping values outside the range
            ``[vmin, vmax]``.

            If clipping is off, values outside the range ``[vmin, vmax]`` are
            also transformed, resulting in values outside ``[0, 1]``.  This
            behavior is usually desirable, as colormaps can mark these *under*
            and *over* values with specific colors.

            If clipping is on, values below *vmin* are mapped to 0 and values
            above *vmax* are mapped to 1. Such values become indistinguishable
            from regular boundary values, which may cause misinterpretation of
            the data.

        Examples
        --------
        This maps data values -2 to 0.25, 0 to 0.5, and 4 to 1.0
        (assuming equal rates of change above and below 0.0):

            >>> import matplotlib.colors as mcolors
            >>> norm = mcolors.CenteredNorm(halfrange=4.0)
            >>> data = [-2., 0., 4.]
            >>> norm(data)
            array([0.25, 0.5 , 1.  ])
        """
        super().__init__(vmin=None, vmax=None, clip=clip)
        self._vcenter = vcenter
        # 设置halfrange属性，调用setter方法设置vmin和vmax
        self.halfrange = halfrange

    def autoscale(self, A):
        """
        Set *halfrange* to ``max(abs(A-vcenter))``, then set *vmin* and *vmax*.
        """
        A = np.asanyarray(A)
        # 计算halfrange，使其等于A中值到vcenter的最大绝对差值，然后设置vmin和vmax
        self.halfrange = max(self._vcenter - A.min(),
                             A.max() - self._vcenter)

    def autoscale_None(self, A):
        """Set *vmin* and *vmax*."""
        A = np.asanyarray(A)
        # 如果halfrange为None且A非空，则调用autoscale方法设置vmin和vmax
        if self.halfrange is None and A.size:
            self.autoscale(A)

    @property
    def vmin(self):
        return self._vmin

    @vmin.setter
    def vmin(self, value):
        # 对vmin进行规范化处理，调用_sanitize_extrema方法
        value = _sanitize_extrema(value)
        # 如果设置的值与当前_vmin不同，则更新_vmin，并重新计算_vmax，然后触发_changed方法
        if value != self._vmin:
            self._vmin = value
            self._vmax = 2 * self.vcenter - value
            self._changed()

    @property
    def vmax(self):
        return self._vmax
    # 返回私有属性 _vmax 的值
    def vmax(self):
        return self._vmax

    # 设置私有属性 _vmax 的值，并更新相关属性
    @vmax.setter
    def vmax(self, value):
        # 对输入的值进行范围清理和调整
        value = _sanitize_extrema(value)
        # 如果新值与当前值不同，则更新 _vmax，并重新计算 _vmin 和触发变更事件
        if value != self._vmax:
            self._vmax = value
            self._vmin = 2*self.vcenter - value
            self._changed()

    # 返回私有属性 _vcenter 的值
    @property
    def vcenter(self):
        return self._vcenter

    # 设置私有属性 _vcenter 的值，并触发相关更新
    @vcenter.setter
    def vcenter(self, vcenter):
        # 如果新值与当前值不同，则更新 _vcenter，并触发 vmin/vmax 的更新和变更事件
        if vcenter != self._vcenter:
            self._vcenter = vcenter
            # 通过设置 halfrange 属性来触发 vmin/vmax 的更新
            self.halfrange = self.halfrange
            self._changed()

    # 返回当前数据范围的一半值（半范围）
    @property
    def halfrange(self):
        # 如果 vmin 或 vmax 有任意一个为 None，则返回 None
        if self.vmin is None or self.vmax is None:
            return None
        # 计算并返回当前数据范围的一半值
        return (self.vmax - self.vmin) / 2

    # 设置数据范围的一半值（半范围）
    @halfrange.setter
    def halfrange(self, halfrange):
        # 如果输入值为 None，则将 vmin 和 vmax 都设为 None
        if halfrange is None:
            self.vmin = None
            self.vmax = None
        else:
            # 根据输入的半范围值重新计算并设置 vmin 和 vmax
            self.vmin = self.vcenter - abs(halfrange)
            self.vmax = self.vcenter + abs(halfrange)
# 从规模类构建`.Normalize`子类的装饰器，基于`~.scale.ScaleBase`子类。
def make_norm_from_scale(scale_cls, base_norm_cls=None, *, init=None):
    """
    Decorator for building a `.Normalize` subclass from a `~.scale.ScaleBase`
    subclass.

    After ::

        @make_norm_from_scale(scale_cls)
        class norm_cls(Normalize):
            ...

    *norm_cls* is filled with methods so that normalization computations are
    forwarded to *scale_cls* (i.e., *scale_cls* is the scale that would be used
    for the colorbar of a mappable normalized with *norm_cls*).

    If *init* is not passed, then the constructor signature of *norm_cls*
    will be ``norm_cls(vmin=None, vmax=None, clip=False)``; these three
    parameters will be forwarded to the base class (``Normalize.__init__``),
    and a *scale_cls* object will be initialized with no arguments (other than
    a dummy axis).

    If the *scale_cls* constructor takes additional parameters, then *init*
    should be passed to `make_norm_from_scale`.  It is a callable which is
    *only* used for its signature.  First, this signature will become the
    signature of *norm_cls*.  Second, the *norm_cls* constructor will bind the
    parameters passed to it using this signature, extract the bound *vmin*,
    *vmax*, and *clip* values, pass those to ``Normalize.__init__``, and
    forward the remaining bound values (including any defaults defined by the
    signature) to the *scale_cls* constructor.
    """

    # 如果未提供基本规范类，则返回一个部分应用的`make_norm_from_scale`函数
    if base_norm_cls is None:
        return functools.partial(make_norm_from_scale, scale_cls, init=init)

    # 如果`scale_cls`是`functools.partial`的实例，则解包其参数
    if isinstance(scale_cls, functools.partial):
        scale_args = scale_cls.args
        scale_kwargs_items = tuple(scale_cls.keywords.items())
        scale_cls = scale_cls.func
    else:
        scale_args = scale_kwargs_items = ()

    # 如果未提供`init`，则定义一个空函数`init`，用于占位
    if init is None:
        def init(vmin=None, vmax=None, clip=False): pass

    # 调用辅助函数`_make_norm_from_scale`，返回其结果
    return _make_norm_from_scale(
        scale_cls, scale_args, scale_kwargs_items,
        base_norm_cls, inspect.signature(init))


# 使用`functools.cache`装饰器缓存结果的辅助函数`_make_norm_from_scale`
@functools.cache
def _make_norm_from_scale(
    scale_cls, scale_args, scale_kwargs_items,
    base_norm_cls, bound_init_signature,
):
    """
    Helper for `make_norm_from_scale`.

    This function is split out to enable caching (in particular so that
    different unpickles reuse the same class).  In order to do so,

    - ``functools.partial`` *scale_cls* is expanded into ``func, args, kwargs``
      to allow memoizing returned norms (partial instances always compare
      unequal, but we can check identity based on ``func, args, kwargs``;
    - *init* is replaced by *init_signature*, as signatures are picklable,
      unlike to arbitrary lambdas.
    """

    # 如果基本规范类是`Normalize`，则设置`Norm`类的名称和文档字符串
    if base_norm_cls is Normalize:
        Norm.__name__ = f"{scale_cls.__name__}Norm"
        Norm.__qualname__ = f"{scale_cls.__qualname__}Norm"
    else:
        Norm.__name__ = base_norm_cls.__name__
        Norm.__qualname__ = base_norm_cls.__qualname__
    Norm.__module__ = base_norm_cls.__module__
    Norm.__doc__ = base_norm_cls.__doc__
    # 返回变量 Norm 的值作为函数的返回结果
    return Norm
# 创建一个属于指定类的空对象实例，并返回
def _create_empty_object_of_class(cls):
    return cls.__new__(cls)


# 根据指定参数创建一个可序列化的规范化对象
def _picklable_norm_constructor(*args):
    # 使用_make_norm_from_scale函数创建一个空对象实例
    return _create_empty_object_of_class(_make_norm_from_scale(*args))


# 使用make_norm_from_scale装饰器创建一个新的类FuncNorm，继承自Normalize
@make_norm_from_scale(
    scale.FuncScale,
    init=lambda functions, vmin=None, vmax=None, clip=False: None)
class FuncNorm(Normalize):
    """
    Arbitrary normalization using functions for the forward and inverse.

    Parameters
    ----------
    functions : (callable, callable)
        two-tuple of the forward and inverse functions for the normalization.
        The forward function must be monotonic.

        Both functions must have the signature ::

           def forward(values: array-like) -> array-like

    vmin, vmax : float or None
        If *vmin* and/or *vmax* is not given, they are initialized from the
        minimum and maximum value, respectively, of the first input
        processed; i.e., ``__call__(A)`` calls ``autoscale_None(A)``.

    clip : bool, default: False
        Determines the behavior for mapping values outside the range
        ``[vmin, vmax]``.

        If clipping is off, values outside the range ``[vmin, vmax]`` are also
        transformed by the function, resulting in values outside ``[0, 1]``.
        This behavior is usually desirable, as colormaps can mark these *under*
        and *over* values with specific colors.

        If clipping is on, values below *vmin* are mapped to 0 and values above
        *vmax* are mapped to 1. Such values become indistinguishable from
        regular boundary values, which may cause misinterpretation of the data.
    """


# 创建一个LogNorm类，使用make_norm_from_scale装饰器装饰Normalize类，并初始化
LogNorm = make_norm_from_scale(
    functools.partial(scale.LogScale, nonpositive="mask"))(Normalize)
LogNorm.__name__ = LogNorm.__qualname__ = "LogNorm"
LogNorm.__doc__ = "Normalize a given value to the 0-1 range on a log scale."


# 使用make_norm_from_scale装饰器创建一个新的类SymLogNorm，继承自Normalize
@make_norm_from_scale(
    scale.SymmetricalLogScale,
    init=lambda linthresh, linscale=1., vmin=None, vmax=None, clip=False, *,
                base=10: None)
class SymLogNorm(Normalize):
    """
    The symmetrical logarithmic scale is logarithmic in both the
    positive and negative directions from the origin.

    Since the values close to zero tend toward infinity, there is a
    need to have a range around zero that is linear.  The parameter
    *linthresh* allows the user to specify the size of this range
    (-*linthresh*, *linthresh*).

    Parameters
    ----------
    linthresh : float
        The range within which the plot is linear (to avoid having the plot
        go to infinity around zero).
    """
    linscale : float, default: 1
        This allows the linear range (-*linthresh* to *linthresh*) to be
        stretched relative to the logarithmic range. Its value is the
        number of decades to use for each half of the linear range. For
        example, when *linscale* == 1.0 (the default), the space used for
        the positive and negative halves of the linear range will be equal
        to one decade in the logarithmic range.
    base : float, default: 10
    """
    # 定义属性 linthresh，用于访问 _scale 对象的 linthresh 属性值
    @property
    def linthresh(self):
        return self._scale.linthresh
    
    # 定义 linthresh 的 setter 方法，用于设置 _scale 对象的 linthresh 属性值
    @linthresh.setter
    def linthresh(self, value):
        self._scale.linthresh = value
# 使用装饰器将下面的类定义与指定的 AsinhScale 进行标准化
@make_norm_from_scale(
    scale.AsinhScale,
    # 定义初始化函数，设置默认参数及其说明
    init=lambda linear_width=1, vmin=None, vmax=None, clip=False: None)
class AsinhNorm(Normalize):
    """
    The inverse hyperbolic sine scale is approximately linear near
    the origin, but becomes logarithmic for larger positive
    or negative values. Unlike the `SymLogNorm`, the transition between
    these linear and logarithmic regions is smooth, which may reduce
    the risk of visual artifacts.

    .. note::

       This API is provisional and may be revised in the future
       based on early user feedback.

    Parameters
    ----------
    linear_width : float, default: 1
        The effective width of the linear region, beyond which
        the transformation becomes asymptotically logarithmic
    """

    @property
    def linear_width(self):
        # 返回当前对象所使用的线性宽度参数值
        return self._scale.linear_width

    @linear_width.setter
    def linear_width(self, value):
        # 设置当前对象所使用的线性宽度参数值
        self._scale.linear_width = value


class PowerNorm(Normalize):
    r"""
    Linearly map a given value to the 0-1 range and then apply
    a power-law normalization over that range.

    Parameters
    ----------
    gamma : float
        Power law exponent.
    vmin, vmax : float or None
        If *vmin* and/or *vmax* is not given, they are initialized from the
        minimum and maximum value, respectively, of the first input
        processed; i.e., ``__call__(A)`` calls ``autoscale_None(A)``.
    clip : bool, default: False
        Determines the behavior for mapping values outside the range
        ``[vmin, vmax]``.

        If clipping is off, values above *vmax* are transformed by the power
        function, resulting in values above 1, and values below *vmin* are linearly
        transformed resulting in values below 0. This behavior is usually desirable, as
        colormaps can mark these *under* and *over* values with specific colors.

        If clipping is on, values below *vmin* are mapped to 0 and values above
        *vmax* are mapped to 1. Such values become indistinguishable from
        regular boundary values, which may cause misinterpretation of the data.

    Notes
    -----
    The normalization formula is

    .. math::

        \left ( \frac{x - v_{min}}{v_{max}  - v_{min}} \right )^{\gamma}

    For input values below *vmin*, gamma is set to one.
    """
    def __init__(self, gamma, vmin=None, vmax=None, clip=False):
        # 调用父类的初始化方法，设置 vmin, vmax, clip 参数
        super().__init__(vmin, vmax, clip)
        # 设置当前对象所使用的 gamma 参数值
        self.gamma = gamma
    # 实现调用对象的方法，接受一个值和一个可选的剪裁参数
    def __call__(self, value, clip=None):
        # 如果剪裁参数为 None，则使用对象自身的剪裁属性
        if clip is None:
            clip = self.clip

        # 处理输入值，获取处理后的结果和是否为标量的信息
        result, is_scalar = self.process_value(value)

        # 自动调整缩放范围
        self.autoscale_None(result)
        
        # 获取对象的 gamma 属性
        gamma = self.gamma
        # 获取对象的最小值和最大值
        vmin, vmax = self.vmin, self.vmax
        
        # 检查最小值和最大值的关系
        if vmin > vmax:
            # 如果最小值大于最大值，抛出数值错误异常
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin == vmax:
            # 如果最小值等于最大值，将结果数组填充为 0
            result.fill(0)
        else:
            # 如果需要剪裁处理
            if clip:
                # 获取结果数组的掩码
                mask = np.ma.getmask(result)
                # 将结果数组填充到最大值并剪裁到指定范围内
                result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                     mask=mask)
            # 获取结果数组的实际数据部分
            resdat = result.data
            # 将数据归一化到 [0, 1] 范围
            resdat -= vmin
            resdat /= (vmax - vmin)
            # 对正数部分应用 gamma 幂函数变换
            resdat[resdat > 0] = np.power(resdat[resdat > 0], gamma)

            # 将处理后的数据重新封装成掩码数组
            result = np.ma.array(resdat, mask=result.mask, copy=False)
        
        # 如果处理结果是标量，则返回其第一个元素
        if is_scalar:
            result = result[0]
        
        # 返回处理后的结果数组或标量
        return result

    # 实现反向转换方法，接受一个值并返回反向转换后的结果
    def inverse(self, value):
        # 如果尚未进行缩放，则抛出数值错误异常
        if not self.scaled():
            raise ValueError("Not invertible until scaled")

        # 处理输入值，获取处理后的结果和是否为标量的信息
        result, is_scalar = self.process_value(value)

        # 获取对象的 gamma 属性
        gamma = self.gamma
        # 获取对象的最小值和最大值
        vmin, vmax = self.vmin, self.vmax

        # 获取结果数组的实际数据部分
        resdat = result.data
        # 对数据应用反向 gamma 幂函数变换
        resdat[resdat > 0] = np.power(resdat[resdat > 0], 1 / gamma)
        # 将数据反归一化到原始范围内
        resdat *= (vmax - vmin)
        resdat += vmin

        # 将处理后的数据重新封装成掩码数组
        result = np.ma.array(resdat, mask=result.mask, copy=False)
        
        # 如果处理结果是标量，则返回其第一个元素
        if is_scalar:
            result = result[0]
        
        # 返回反向转换后的结果数组或标量
        return result
class BoundaryNorm(Normalize):
    """
    Generate a colormap index based on discrete intervals.

    Unlike `Normalize` or `LogNorm`, `BoundaryNorm` maps values to integers
    instead of to the interval 0-1.
    """

    # 通过继承 `Normalize` 类，定义了一个新的归一化类 `BoundaryNorm`
    # 它基于离散间隔生成颜色映射索引

    # 尽管可以通过分段线性插值将值映射到0-1区间，
    # 但使用整数似乎更简单，并减少了整数和浮点数之间的来回转换。
    def __init__(self, boundaries, ncolors, clip=False, *, extend='neither'):
        """
        Parameters
        ----------
        boundaries : array-like
            Monotonically increasing sequence of at least 2 bin edges:  data
            falling in the n-th bin will be mapped to the n-th color.

        ncolors : int
            Number of colors in the colormap to be used.

        clip : bool, optional
            If clip is ``True``, out of range values are mapped to 0 if they
            are below ``boundaries[0]`` or mapped to ``ncolors - 1`` if they
            are above ``boundaries[-1]``.

            If clip is ``False``, out of range values are mapped to -1 if
            they are below ``boundaries[0]`` or mapped to *ncolors* if they are
            above ``boundaries[-1]``. These are then converted to valid indices
            by `Colormap.__call__`.

        extend : {'neither', 'both', 'min', 'max'}, default: 'neither'
            Extend the number of bins to include one or both of the
            regions beyond the boundaries.  For example, if ``extend``
            is 'min', then the color to which the region between the first
            pair of boundaries is mapped will be distinct from the first
            color in the colormap, and by default a
            `~matplotlib.colorbar.Colorbar` will be drawn with
            the triangle extension on the left or lower end.

        Notes
        -----
        If there are fewer bins (including extensions) than colors, then the
        color index is chosen by linearly interpolating the ``[0, nbins - 1]``
        range onto the ``[0, ncolors - 1]`` range, effectively skipping some
        colors in the middle of the colormap.
        """
        # 检查 'clip=True' 与 'extend' 参数的兼容性
        if clip and extend != 'neither':
            raise ValueError("'clip=True' is not compatible with 'extend'")
        
        # 调用父类的初始化方法，设置颜色映射的最小值和最大值，并设置是否裁剪边界
        super().__init__(vmin=boundaries[0], vmax=boundaries[-1], clip=clip)
        
        # 将边界转换为 NumPy 数组，计算边界数量
        self.boundaries = np.asarray(boundaries)
        self.N = len(self.boundaries)
        
        # 如果提供的边界少于两个，则引发 ValueError
        if self.N < 2:
            raise ValueError("You must provide at least 2 boundaries "
                             f"(1 region) but you passed in {boundaries!r}")
        
        # 设置颜色映射的颜色数量和扩展方式
        self.Ncmap = ncolors
        self.extend = extend

        # 不使用默认的比例尺
        self._scale = None

        # 计算需要的区域数和偏移量，根据 extend 参数增加区域数和偏移量
        self._n_regions = self.N - 1  # 需要的颜色区域数
        self._offset = 0
        if extend in ('min', 'both'):
            self._n_regions += 1
            self._offset = 1
        if extend in ('max', 'both'):
            self._n_regions += 1
        
        # 如果需要的区域数大于颜色数量，则引发 ValueError
        if self._n_regions > self.Ncmap:
            raise ValueError(f"There are {self._n_regions} color bins "
                             "including extensions, but ncolors = "
                             f"{ncolors}; ncolors must equal or exceed the "
                             "number of bins")
    def __call__(self, value, clip=None):
        """
        This method behaves similarly to `.Normalize.__call__`, except that it
        returns integers or arrays of int16.
        """
        # 如果没有指定 clip 参数，则使用对象自身的 clip 属性
        if clip is None:
            clip = self.clip

        # 处理输入值，返回处理后的 xx 数组和是否为标量的标志 is_scalar
        xx, is_scalar = self.process_value(value)
        
        # 获取 xx 的掩码数组
        mask = np.ma.getmaskarray(xx)
        
        # 将掩码值填充为 vmax + 1，确保填充后的 xx 至少为一维数组
        xx = np.atleast_1d(xx.filled(self.vmax + 1))
        
        # 如果 clip 为真，则将 xx 数组裁剪到 [self.vmin, self.vmax] 范围内，并将最大列数设为 self.Ncmap - 1
        if clip:
            np.clip(xx, self.vmin, self.vmax, out=xx)
            max_col = self.Ncmap - 1
        else:
            # 否则，最大列数设为 self.Ncmap
            max_col = self.Ncmap
        
        # 将 xx 数组的每个元素映射到 self.boundaries 所定义的区间，得到对应的区间索引 iret
        iret = np.digitize(xx, self.boundaries) - 1 + self._offset
        
        # 如果颜色数 self.Ncmap 大于区间数 self._n_regions，则进行以下处理
        if self.Ncmap > self._n_regions:
            if self._n_regions == 1:
                # 当只有一个区间时，特殊处理为选择中间颜色
                iret[iret == 0] = (self.Ncmap - 1) // 2
            else:
                # 否则，线性地将区间索引映射到颜色索引空间
                iret = (self.Ncmap - 1) / (self._n_regions - 1) * iret
        
        # 将 iret 数组转换为 int16 类型
        iret = iret.astype(np.int16)
        
        # 将小于 self.vmin 的 xx 对应的 iret 设为 -1
        iret[xx < self.vmin] = -1
        
        # 将大于等于 self.vmax 的 xx 对应的 iret 设为 max_col
        iret[xx >= self.vmax] = max_col
        
        # 创建一个带掩码的 numpy 数组 ret，掩码为 mask，数据为 iret
        ret = np.ma.array(iret, mask=mask)
        
        # 如果 is_scalar 为真，则将 ret 转换为整数，假设为 Python 标量
        if is_scalar:
            ret = int(ret[0])  # 假设为 Python 标量
        
        # 返回处理后的 ret 结果
        return ret

    def inverse(self, value):
        """
        Raises
        ------
        ValueError
            BoundaryNorm is not invertible, so calling this method will always
            raise an error
        """
        # 抛出 ValueError 异常，因为 BoundaryNorm 不可逆转
        raise ValueError("BoundaryNorm is not invertible")
class NoNorm(Normalize):
    """
    Dummy replacement for `Normalize`, for the case where we want to use
    indices directly in a `~matplotlib.cm.ScalarMappable`.
    """

    def __call__(self, value, clip=None):
        # 如果 value 是可迭代的，则返回一个掩码数组
        if np.iterable(value):
            return np.ma.array(value)
        return value

    def inverse(self, value):
        # 如果 value 是可迭代的，则返回一个掩码数组
        if np.iterable(value):
            return np.ma.array(value)
        return value


def rgb_to_hsv(arr):
    """
    Convert an array of float RGB values (in the range [0, 1]) to HSV values.

    Parameters
    ----------
    arr : (..., 3) array-like
       All values must be in the range [0, 1]

    Returns
    -------
    (..., 3) `~numpy.ndarray`
       Colors converted to HSV values in range [0, 1]
    """
    arr = np.asarray(arr)

    # 检查最后一个维度的长度，应该是 RGB 颜色通道
    if arr.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         f"shape {arr.shape} was found.")

    in_shape = arr.shape
    arr = np.array(
        arr, copy=False,
        dtype=np.promote_types(arr.dtype, np.float32),  # 不处理整数类型
        ndmin=2,  # 如果输入是1维，则升级为2维
    )
    out = np.zeros_like(arr)
    arr_max = arr.max(-1)
    ipos = arr_max > 0
    delta = np.ptp(arr, -1)
    s = np.zeros_like(delta)
    s[ipos] = delta[ipos] / arr_max[ipos]
    ipos = delta > 0
    # 红色分量最大
    idx = (arr[..., 0] == arr_max) & ipos
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]
    # 绿色分量最大
    idx = (arr[..., 1] == arr_max) & ipos
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0]) / delta[idx]
    # 蓝色分量最大
    idx = (arr[..., 2] == arr_max) & ipos
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1]) / delta[idx]

    out[..., 0] = (out[..., 0] / 6.0) % 1.0
    out[..., 1] = s
    out[..., 2] = arr_max

    return out.reshape(in_shape)


def hsv_to_rgb(hsv):
    """
    Convert HSV values to RGB.

    Parameters
    ----------
    hsv : (..., 3) array-like
       All values assumed to be in range [0, 1]

    Returns
    -------
    (..., 3) `~numpy.ndarray`
       Colors converted to RGB values in range [0, 1]
    """
    hsv = np.asarray(hsv)

    # 检查最后一个维度的长度，应该是 HSV 颜色通道
    if hsv.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         f"shape {hsv.shape} was found.")

    in_shape = hsv.shape
    hsv = np.array(
        hsv, copy=False,
        dtype=np.promote_types(hsv.dtype, np.float32),  # 不处理整数类型
        ndmin=2,  # 如果输入是1维，则升级为2维
    )

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = i % 6 == 0
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]
    # 检查 i 是否等于 1，返回布尔数组 idx
    idx = i == 1
    # 根据 idx 更新 r 数组中对应位置的值
    r[idx] = q[idx]
    # 根据 idx 更新 g 数组中对应位置的值
    g[idx] = v[idx]
    # 根据 idx 更新 b 数组中对应位置的值
    b[idx] = p[idx]

    # 检查 i 是否等于 2，返回布尔数组 idx
    idx = i == 2
    # 根据 idx 更新 r 数组中对应位置的值
    r[idx] = p[idx]
    # 根据 idx 更新 g 数组中对应位置的值
    g[idx] = v[idx]
    # 根据 idx 更新 b 数组中对应位置的值
    b[idx] = t[idx]

    # 检查 i 是否等于 3，返回布尔数组 idx
    idx = i == 3
    # 根据 idx 更新 r 数组中对应位置的值
    r[idx] = p[idx]
    # 根据 idx 更新 g 数组中对应位置的值
    g[idx] = q[idx]
    # 根据 idx 更新 b 数组中对应位置的值
    b[idx] = v[idx]

    # 检查 i 是否等于 4，返回布尔数组 idx
    idx = i == 4
    # 根据 idx 更新 r 数组中对应位置的值
    r[idx] = t[idx]
    # 根据 idx 更新 g 数组中对应位置的值
    g[idx] = p[idx]
    # 根据 idx 更新 b 数组中对应位置的值
    b[idx] = v[idx]

    # 检查 i 是否等于 5，返回布尔数组 idx
    idx = i == 5
    # 根据 idx 更新 r 数组中对应位置的值
    r[idx] = v[idx]
    # 根据 idx 更新 g 数组中对应位置的值
    g[idx] = p[idx]
    # 根据 idx 更新 b 数组中对应位置的值
    b[idx] = q[idx]

    # 检查 s 是否等于 0，返回布尔数组 idx
    idx = s == 0
    # 根据 idx 更新 r、g、b 数组中对应位置的值为 v[idx]
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    # 将 r、g、b 数组沿最后一个轴(axis=-1)堆叠成一个 RGB 图像数组
    rgb = np.stack([r, g, b], axis=-1)

    # 将形状重塑为输入形状 in_shape 的数组，并返回结果
    return rgb.reshape(in_shape)
def _vector_magnitude(arr):
    # 计算向量的大小（即模长）
    # 这里不能使用 np.linalg.norm，因为它会从 ma.array 中移除掩码
    # 也不能使用 np.sum，除非整个向量都被掩码覆盖

    # 初始化总和的平方
    sum_sq = 0
    # 遍历数组中的最后一个维度
    for i in range(arr.shape[-1]):
        # 将每个元素平方并加入总和
        sum_sq += arr[..., i, np.newaxis] ** 2
    # 返回总和的平方根，即向量的大小
    return np.sqrt(sum_sq)


class LightSource:
    """
    创建一个光源，根据指定的方位角和高度角。
    角度单位为度，方位角从北方向顺时针测量，高度角从表面的零平面向上测量。

    `shade` 方法用于为数据数组生成“阴影”RGB值。
    `shade_rgb` 方法可用于将RGB图像与高程图结合。
    `hillshade` 方法生成表面的照明图。
    """

    def __init__(self, azdeg=315, altdeg=45, hsv_min_val=0, hsv_max_val=1,
                 hsv_min_sat=1, hsv_max_sat=0):
        """
        指定光源的方位角（顺时针从南方测量）和高度角（从水平面向上测量）。

        Parameters
        ----------
        azdeg : float, 默认值：315度（从西北方向）
            光源的方位角（0-360，从北方顺时针测量的度数）。
        altdeg : float, 默认值：45度
            光源的高度角（0-90，从水平方向向上测量的度数）。
        hsv_min_val : number, 默认值：0
            *intensity* 地图能够将输出图像的最小值（“v”在“hsv”中）移动的最小值。
        hsv_max_val : number, 默认值：1
            *intensity* 地图能够将输出图像的最大值（“v”在“hsv”中）移动的最大值。
        hsv_min_sat : number, 默认值：1
            *intensity* 地图能够将输出图像的最小饱和度值移动到的最小饱和度值。
        hsv_max_sat : number, 默认值：0
            *intensity* 地图能够将输出图像的最大饱和度值移动到的最大饱和度值。

        Notes
        -----
        为了向后兼容，可以在初始化时也提供参数 *hsv_min_val*、*hsv_max_val*、
        *hsv_min_sat* 和 *hsv_max_sat*。但是，只有当 `blend_mode='hsv'` 被传递到 `shade` 或 `shade_rgb` 中时，
        才会使用这些参数。详情请参阅 `blend_hsv` 的文档。
        """
        self.azdeg = azdeg
        self.altdeg = altdeg
        self.hsv_min_val = hsv_min_val
        self.hsv_max_val = hsv_max_val
        self.hsv_min_sat = hsv_min_sat
        self.hsv_max_sat = hsv_max_sat

    @property
    def direction(self):
        """
        The unit vector direction towards the light source.

        Calculates the unit vector pointing from the surface towards the light
        source using azimuth and altitude angles.

        Returns
        -------
        numpy.ndarray
            A 1D array representing the direction vector in Cartesian coordinates.
        """
        # Azimuth angle in radians counterclockwise from East (mathematical notation)
        az = np.radians(90 - self.azdeg)
        # Altitude angle in radians
        alt = np.radians(self.altdeg)
        # Calculate the direction vector in Cartesian coordinates
        return np.array([
            np.cos(az) * np.cos(alt),
            np.sin(az) * np.cos(alt),
            np.sin(alt)
        ])

    def hillshade(self, elevation, vert_exag=1, dx=1, dy=1, fraction=1.):
        """
        Calculate the illumination intensity for a surface using the defined
        azimuth and elevation for the light source.

        This computes the normal vectors for the surface, and then passes them
        on to `shade_normals`

        Parameters
        ----------
        elevation : 2D array-like
            The height values used to generate an illumination map
        vert_exag : number, optional
            The amount to exaggerate the elevation values by when calculating
            illumination. This can be used either to correct for differences in
            units between the x-y coordinate system and the elevation
            coordinate system (e.g. decimal degrees vs. meters) or to
            exaggerate or de-emphasize topographic effects.
        dx : number, optional
            The x-spacing (columns) of the input *elevation* grid.
        dy : number, optional
            The y-spacing (rows) of the input *elevation* grid.
        fraction : number, optional
            Increases or decreases the contrast of the hillshade. Values
            greater than one will cause intermediate values to move closer to
            full illumination or shadow (and clipping any values that move
            beyond 0 or 1). Note that this is not visually or mathematically
            the same as vertical exaggeration.

        Returns
        -------
        `~numpy.ndarray`
            A 2D array of illumination values between 0-1, where 0 is
            completely in shadow and 1 is completely illuminated.
        """

        # Adjust dy to account for the implicit negative direction in image raster
        dy = -dy

        # Compute the partial derivatives of the exaggerated elevation grid
        e_dy, e_dx = np.gradient(vert_exag * elevation, dy, dx)

        # Initialize an empty array for normal vectors with shape (rows, cols, 3)
        normal = np.empty(elevation.shape + (3,)).view(type(elevation))
        # Calculate normal vectors from the partial derivatives
        normal[..., 0] = -e_dx
        normal[..., 1] = -e_dy
        normal[..., 2] = 1
        # Normalize the normal vectors
        normal /= _vector_magnitude(normal)

        # Compute and return the shaded relief using the calculated normals
        return self.shade_normals(normal, fraction)
    def shade_normals(self, normals, fraction=1.):
        """
        Calculate the illumination intensity for the normal vectors of a
        surface using the defined azimuth and elevation for the light source.

        Imagine an artificial sun placed at infinity in some azimuth and
        elevation position illuminating our surface. The parts of the surface
        that slope toward the sun should brighten while those sides facing away
        should become darker.

        Parameters
        ----------
        fraction : number, optional
            Increases or decreases the contrast of the hillshade.  Values
            greater than one will cause intermediate values to move closer to
            full illumination or shadow (and clipping any values that move
            beyond 0 or 1). Note that this is not visually or mathematically
            the same as vertical exaggeration.

        Returns
        -------
        `~numpy.ndarray`
            A 2D array of illumination values between 0-1, where 0 is
            completely in shadow and 1 is completely illuminated.
        """

        # Calculate dot product of normal vectors with light direction
        intensity = normals.dot(self.direction)

        # Apply contrast stretch
        imin, imax = intensity.min(), intensity.max()
        intensity *= fraction

        # Rescale to 0-1, keeping range before contrast stretch
        # If constant slope, keep relative scaling (i.e. flat should be 0.5,
        # fully occluded 0, etc.)
        if (imax - imin) > 1e-6:
            # Rescale intensity to [0, 1] preserving relative differences
            intensity -= imin
            intensity /= (imax - imin)
        
        # Clip values to ensure they are within valid range [0, 1]
        intensity = np.clip(intensity, 0, 1)

        return intensity

    def blend_soft_light(self, rgb, intensity):
        """
        Combine an RGB image with an intensity map using "soft light" blending,
        using the "pegtop" formula.

        Parameters
        ----------
        rgb : `~numpy.ndarray`
            An (M, N, 3) RGB array of floats ranging from 0 to 1 (color image).
        intensity : `~numpy.ndarray`
            An (M, N, 1) array of floats ranging from 0 to 1 (grayscale image).

        Returns
        -------
        `~numpy.ndarray`
            An (M, N, 3) RGB array representing the combined images.
        """
        # Apply soft light blending formula to combine RGB and intensity map
        return 2 * intensity * rgb + (1 - 2 * intensity) * rgb**2
    # 定义一个方法，用于将 RGB 图像和强度映射使用"叠加"混合方式进行组合

    Parameters
    ----------
    rgb : `~numpy.ndarray`
        一个 (M, N, 3) 的 RGB 浮点数组，范围在 0 到 1 之间（彩色图像）。
    intensity : `~numpy.ndarray`
        一个 (M, N, 1) 的浮点数组，范围在 0 到 1 之间（灰度图像）。

    Returns
    -------
    ndarray
        一个 (M, N, 3) 的 RGB 数组，表示组合后的图像。

    # 计算强度低时的叠加混合结果
    low = 2 * intensity * rgb
    # 计算强度高时的叠加混合结果
    high = 1 - 2 * (1 - intensity) * (1 - rgb)
    # 根据 RGB 图像的像素值进行条件判断，选择合适的混合结果
    return np.where(rgb <= 0.5, low, high)
def from_levels_and_colors(levels, colors, extend='neither'):
    """
    A helper routine to generate a cmap and a norm instance which
    behave similar to contourf's levels and colors arguments.

    Parameters
    ----------
    levels : sequence of numbers
        The quantization levels used to construct the `BoundaryNorm`.
        Value ``v`` is quantized to level ``i`` if ``lev[i] <= v < lev[i+1]``.
    colors : sequence of colors
        The fill color to use for each level. If *extend* is "neither" there
        must be ``n_level - 1`` colors. For an *extend* of "min" or "max" add
        one extra color, and for an *extend* of "both" add two colors.
    extend : {'neither', 'min', 'max', 'both'}, optional
        The behaviour when a value falls out of range of the given levels.
        See `~.Axes.contourf` for details.

    Returns
    -------
    cmap : `~matplotlib.colors.Colormap`
        The generated colormap object based on the provided levels and colors.
    norm : `~matplotlib.colors.Normalize`
        The normalization object based on the provided levels.
    """
    # 定义一个映射字典，根据 extend 参数选择合适的切片范围
    slice_map = {
        'both': slice(1, -1),
        'min': slice(1, None),
        'max': slice(0, -1),
        'neither': slice(0, None),
    }
    # 检查 extend 参数是否在预定义的切片范围内
    _api.check_in_list(slice_map, extend=extend)
    # 根据 extend 参数选择颜色切片
    color_slice = slice_map[extend]

    # 计算数据颜色的数量
    n_data_colors = len(levels) - 1
    # 计算预期的颜色数量
    n_expected = n_data_colors + color_slice.start - (color_slice.stop or 0)
    # 检查颜色列表长度是否符合预期
    if len(colors) != n_expected:
        raise ValueError(
            f'With extend == {extend!r} and {len(levels)} levels, '
            f'expected {n_expected} colors, but got {len(colors)}')

    # 根据颜色切片创建 ListedColormap 对象
    cmap = ListedColormap(colors[color_slice], N=n_data_colors)

    # 设置 colormap 的下限颜色
    if extend in ['min', 'both']:
        cmap.set_under(colors[0])
    else:
        cmap.set_under('none')

    # 设置 colormap 的上限颜色
    if extend in ['max', 'both']:
        cmap.set_over(colors[-1])
    else:
        cmap.set_over('none')

    # 设置 colormap 的颜色扩展属性
    cmap.colorbar_extend = extend

    # 创建并返回 BoundaryNorm 对象
    norm = BoundaryNorm(levels, ncolors=n_data_colors)
    return cmap, norm
```