# `.\numpy\numpy\lib\_version.py`

```py
"""Utility to compare (NumPy) version strings.

The NumpyVersion class allows properly comparing numpy version strings.
The LooseVersion and StrictVersion classes that distutils provides don't
work; they don't recognize anything like alpha/beta/rc/dev versions.

"""
import re  # 导入正则表达式模块


__all__ = ['NumpyVersion']  # 声明公开的类名列表


class NumpyVersion():
    """Parse and compare numpy version strings.

    NumPy has the following versioning scheme (numbers given are examples; they
    can be > 9 in principle):

    - Released version: '1.8.0', '1.8.1', etc.
    - Alpha: '1.8.0a1', '1.8.0a2', etc.
    - Beta: '1.8.0b1', '1.8.0b2', etc.
    - Release candidates: '1.8.0rc1', '1.8.0rc2', etc.
    - Development versions: '1.8.0.dev-f1234afa' (git commit hash appended)
    - Development versions after a1: '1.8.0a1.dev-f1234afa',
                                     '1.8.0b2.dev-f1234afa',
                                     '1.8.1rc1.dev-f1234afa', etc.
    - Development versions (no git hash available): '1.8.0.dev-Unknown'

    Comparing needs to be done against a valid version string or other
    `NumpyVersion` instance. Note that all development versions of the same
    (pre-)release compare equal.

    .. versionadded:: 1.9.0

    Parameters
    ----------
    vstring : str
        NumPy version string (``np.__version__``).

    Examples
    --------
    >>> from numpy.lib import NumpyVersion
    >>> if NumpyVersion(np.__version__) < '1.7.0':
    ...     print('skip')
    >>> # skip

    >>> NumpyVersion('1.7')  # raises ValueError, add ".0"
    Traceback (most recent call last):
        ...
    ValueError: Not a valid numpy version string

    """

    def __init__(self, vstring):
        self.vstring = vstring  # 初始化版本字符串属性
        ver_main = re.match(r'\d+\.\d+\.\d+', vstring)  # 使用正则表达式匹配主要版本号
        if not ver_main:
            raise ValueError("Not a valid numpy version string")

        self.version = ver_main.group()  # 提取主要版本号
        self.major, self.minor, self.bugfix = [int(x) for x in
            self.version.split('.')]  # 拆分版本号为主、次、修订版本号
        if len(vstring) == ver_main.end():
            self.pre_release = 'final'  # 检查是否为最终版本
        else:
            alpha = re.match(r'a\d', vstring[ver_main.end():])  # 匹配alpha版本号
            beta = re.match(r'b\d', vstring[ver_main.end():])  # 匹配beta版本号
            rc = re.match(r'rc\d', vstring[ver_main.end():])  # 匹配rc版本号
            pre_rel = [m for m in [alpha, beta, rc] if m is not None]  # 获取预发布版本信息
            if pre_rel:
                self.pre_release = pre_rel[0].group()  # 获取预发布版本
            else:
                self.pre_release = ''  # 如果没有预发布版本，则为空字符串

        self.is_devversion = bool(re.search(r'.dev', vstring))  # 检查是否为开发版本
    def _compare_version(self, other):
        """比较主版本号.次版本号.修订版本号"""
        if self.major == other.major:
            if self.minor == other.minor:
                if self.bugfix == other.bugfix:
                    vercmp = 0  # 版本相同
                elif self.bugfix > other.bugfix:
                    vercmp = 1  # self版本号修订版本号较大
                else:
                    vercmp = -1  # other版本号修订版本号较大
            elif self.minor > other.minor:
                vercmp = 1  # self版本号次版本号较大
            else:
                vercmp = -1  # other版本号次版本号较大
        elif self.major > other.major:
            vercmp = 1  # self版本号主版本号较大
        else:
            vercmp = -1  # other版本号主版本号较大

        return vercmp

    def _compare_pre_release(self, other):
        """比较预发行版本：alpha/beta/rc/final"""
        if self.pre_release == other.pre_release:
            vercmp = 0  # 预发行版本相同
        elif self.pre_release == 'final':
            vercmp = 1  # self为正式版，比other版本号大
        elif other.pre_release == 'final':
            vercmp = -1  # other为正式版，比self版本号大
        elif self.pre_release > other.pre_release:
            vercmp = 1  # self预发行版本较大
        else:
            vercmp = -1  # other预发行版本较大

        return vercmp

    def _compare(self, other):
        """比较两个版本号对象"""
        if not isinstance(other, (str, NumpyVersion)):
            raise ValueError("Invalid object to compare with NumpyVersion.")  # 抛出异常，无效的比较对象

        if isinstance(other, str):
            other = NumpyVersion(other)  # 将字符串转换为NumpyVersion对象

        vercmp = self._compare_version(other)  # 比较版本号
        if vercmp == 0:
            # 如果版本号相同，则检查预发行版本
            vercmp = self._compare_pre_release(other)
            if vercmp == 0:
                # 如果版本号和预发行版本都相同，则检查是否为开发版本
                if self.is_devversion is other.is_devversion:
                    vercmp = 0  # 同为开发版本
                elif self.is_devversion:
                    vercmp = -1  # self为开发版本，比other版本小
                else:
                    vercmp = 1  # other为开发版本，比self版本小

        return vercmp

    def __lt__(self, other):
        """小于运算符重载"""
        return self._compare(other) < 0  # 判断self版本号是否小于other版本号

    def __le__(self, other):
        """小于等于运算符重载"""
        return self._compare(other) <= 0  # 判断self版本号是否小于等于other版本号

    def __eq__(self, other):
        """等于运算符重载"""
        return self._compare(other) == 0  # 判断self版本号是否等于other版本号

    def __ne__(self, other):
        """不等于运算符重载"""
        return self._compare(other) != 0  # 判断self版本号是否不等于other版本号

    def __gt__(self, other):
        """大于运算符重载"""
        return self._compare(other) > 0  # 判断self版本号是否大于other版本号

    def __ge__(self, other):
        """大于等于运算符重载"""
        return self._compare(other) >= 0  # 判断self版本号是否大于等于other版本号

    def __repr__(self):
        """对象的字符串表示"""
        return "NumpyVersion(%s)" % self.vstring  # 返回包含版本号字符串的表示形式
```