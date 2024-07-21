# `.\pytorch\torch\_vendor\packaging\version.py`

```py
# 导入模块 itertools、re 和类型提示 Any、Callable、NamedTuple、Optional、SupportsInt、Tuple、Union
import itertools
import re
from typing import Any, Callable, NamedTuple, Optional, SupportsInt, Tuple, Union

# 从 _structures 模块导入 Infinity、InfinityType、NegativeInfinity 和 NegativeInfinityType
from ._structures import Infinity, InfinityType, NegativeInfinity, NegativeInfinityType

# __all__ 列表，指定模块中公开的符号
__all__ = ["VERSION_PATTERN", "parse", "Version", "InvalidVersion"]

# LocalType 类型别名，表示一个元组，元素可以是整数或字符串
LocalType = Tuple[Union[int, str], ...]

# CmpPrePostDevType 类型别名，可以是 InfinityType、NegativeInfinityType 或一个由字符串和整数组成的元组
CmpPrePostDevType = Union[InfinityType, NegativeInfinityType, Tuple[str, int]]

# CmpLocalType 类型别名，可以是 NegativeInfinityType 或一个元组，元素可以是整数和字符串的组合
CmpLocalType = Union[
    NegativeInfinityType,
    Tuple[Union[Tuple[int, str], Tuple[NegativeInfinityType, Union[int, str]]], ...],
]

# CmpKey 类型别名，定义了版本比较的关键属性
CmpKey = Tuple[
    int,                     # epoch（纪元）
    Tuple[int, ...],         # release（发布版本号）
    CmpPrePostDevType,       # dev（开发版本）
    CmpPrePostDevType,       # pre（预发行版本）
    CmpPrePostDevType,       # post（发布后版本）
    CmpLocalType             # local（本地版本标识）
]

# VersionComparisonMethod 类型别名，定义了版本比较方法的类型
VersionComparisonMethod = Callable[[CmpKey, CmpKey], bool]


class _Version(NamedTuple):
    epoch: int                       # 纪元
    release: Tuple[int, ...]         # 发布版本号
    dev: Optional[Tuple[str, int]]   # 开发版本
    pre: Optional[Tuple[str, int]]   # 预发行版本
    post: Optional[Tuple[str, int]]  # 发布后版本
    local: Optional[LocalType]       # 本地版本标识


def parse(version: str) -> "Version":
    """Parse the given version string.

    >>> parse('1.0.dev1')
    <Version('1.0.dev1')>

    :param version: The version string to parse.
    :raises InvalidVersion: When the version string is not a valid version.
    """
    return Version(version)


class InvalidVersion(ValueError):
    """Raised when a version string is not a valid version.

    >>> Version("invalid")
    Traceback (most recent call last):
        ...
    packaging.version.InvalidVersion: Invalid version: 'invalid'
    """


class _BaseVersion:
    _key: Tuple[Any, ...]  # _key 属性，用于基础版本类的比较关键

    def __hash__(self) -> int:
        return hash(self._key)

    # __lt__ 方法，定义小于比较操作
    def __lt__(self, other: "_BaseVersion") -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key < other._key

    # __le__ 方法，定义小于等于比较操作
    def __le__(self, other: "_BaseVersion") -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key <= other._key

    # __eq__ 方法，定义等于比较操作
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key == other._key

    # __ge__ 方法，定义大于等于比较操作
    def __ge__(self, other: "_BaseVersion") -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key >= other._key

    # __gt__ 方法，定义大于比较操作
    def __gt__(self, other: "_BaseVersion") -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key > other._key
    # 定义一个特殊方法 __ne__，用于检查当前对象和另一个对象是否不相等
    def __ne__(self, other: object) -> bool:
        # 如果另一个对象不是 _BaseVersion 类型，则返回 NotImplemented
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        # 返回当前对象的 _key 属性与另一个对象的 _key 属性比较的结果
        return self._key != other._key
# 定义一个用于匹配版本号的正则表达式模式字符串
_VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""

# 将版本号正则表达式模式字符串赋值给变量VERSION_PATTERN，用于匹配有效版本号
VERSION_PATTERN = _VERSION_PATTERN
"""
A string containing the regular expression used to match a valid version.

The pattern is not anchored at either end, and is intended for embedding in larger
expressions (for example, matching a version number as part of a file name). The
regular expression should be compiled with the ``re.VERBOSE`` and ``re.IGNORECASE``
flags set.

:meta hide-value:
"""


class Version(_BaseVersion):
    """This class abstracts handling of a project's versions.

    A :class:`Version` instance is comparison aware and can be compared and
    sorted using the standard Python interfaces.

    >>> v1 = Version("1.0a5")
    >>> v2 = Version("1.0")
    >>> v1
    <Version('1.0a5')>
    >>> v2
    <Version('1.0')>
    >>> v1 < v2
    True
    >>> v1 == v2
    False
    >>> v1 > v2
    False
    >>> v1 >= v2
    False
    >>> v1 <= v2
    True
    """

    # 使用预定义的版本号正则表达式模式字符串创建一个正则表达式对象，用于匹配版本号字符串
    _regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)
    _key: CmpKey
    def __init__(self, version: str) -> None:
        """Initialize a Version object.

        :param version:
            The string representation of a version which will be parsed and normalized
            before use.
        :raises InvalidVersion:
            If the ``version`` does not conform to PEP 440 in any way then this
            exception will be raised.
        """

        # Validate the version and parse it into pieces
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion(f"Invalid version: '{version}'")

        # Store the parsed out pieces of the version
        self._version = _Version(
            epoch=int(match.group("epoch")) if match.group("epoch") else 0,
            release=tuple(int(i) for i in match.group("release").split(".")),
            pre=_parse_letter_version(match.group("pre_l"), match.group("pre_n")),
            post=_parse_letter_version(
                match.group("post_l"), match.group("post_n1") or match.group("post_n2")
            ),
            dev=_parse_letter_version(match.group("dev_l"), match.group("dev_n")),
            local=_parse_local_version(match.group("local")),
        )

        # Generate a key which will be used for sorting
        self._key = _cmpkey(
            self._version.epoch,
            self._version.release,
            self._version.pre,
            self._version.post,
            self._version.dev,
            self._version.local,
        )

    def __repr__(self) -> str:
        """A representation of the Version that shows all internal state.

        >>> Version('1.0.0')
        <Version('1.0.0')>
        """
        return f"<Version('{self}')>"

    def __str__(self) -> str:
        """A string representation of the version that can be rounded-tripped.

        >>> str(Version("1.0a5"))
        '1.0a5'
        """
        parts = []

        # Epoch
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release segment
        parts.append(".".join(str(x) for x in self.release))

        # Pre-release
        if self.pre is not None:
            parts.append("".join(str(x) for x in self.pre))

        # Post-release
        if self.post is not None:
            parts.append(f".post{self.post}")

        # Development release
        if self.dev is not None:
            parts.append(f".dev{self.dev}")

        # Local version segment
        if self.local is not None:
            parts.append(f"+{self.local}")

        return "".join(parts)

    @property
    def epoch(self) -> int:
        """The epoch of the version.

        >>> Version("2.0.0").epoch
        0
        >>> Version("1!2.0.0").epoch
        1
        """
        return self._version.epoch

    @property
    def release(self) -> Tuple[int, ...]:
        """The release segment of the version.

        >>> Version("1.0.0").release
        (1, 0, 0)
        """
        return self._version.release
    def release(self) -> Tuple[int, ...]:
        """获取版本号中的"release"段。

        >>> Version("1.2.3").release
        (1, 2, 3)
        >>> Version("2.0.0").release
        (2, 0, 0)
        >>> Version("1!2.0.0.post0").release
        (2, 0, 0)

        包括尾随的零，但不包括纪元或任何预发布/开发/后发布的后缀。
        """
        return self._version.release

    @property
    def pre(self) -> Optional[Tuple[str, int]]:
        """获取版本号中的预发布段。

        >>> print(Version("1.2.3").pre)
        None
        >>> Version("1.2.3a1").pre
        ('a', 1)
        >>> Version("1.2.3b1").pre
        ('b', 1)
        >>> Version("1.2.3rc1").pre
        ('rc', 1)
        """
        return self._version.pre

    @property
    def post(self) -> Optional[int]:
        """获取版本号中的后发布号。

        >>> print(Version("1.2.3").post)
        None
        >>> Version("1.2.3.post1").post
        1
        """
        return self._version.post[1] if self._version.post else None

    @property
    def dev(self) -> Optional[int]:
        """获取版本号中的开发号。

        >>> print(Version("1.2.3").dev)
        None
        >>> Version("1.2.3.dev1").dev
        1
        """
        return self._version.dev[1] if self._version.dev else None

    @property
    def local(self) -> Optional[str]:
        """获取版本号中的本地版本段。

        >>> print(Version("1.2.3").local)
        None
        >>> Version("1.2.3+abc").local
        'abc'
        """
        if self._version.local:
            return ".".join(str(x) for x in self._version.local)
        else:
            return None

    @property
    def public(self) -> str:
        """获取版本号的公共部分。

        >>> Version("1.2.3").public
        '1.2.3'
        >>> Version("1.2.3+abc").public
        '1.2.3'
        >>> Version("1.2.3+abc.dev1").public
        '1.2.3'
        """
        return str(self).split("+", 1)[0]

    @property
    def base_version(self) -> str:
        """获取版本号的“基本版本”。

        >>> Version("1.2.3").base_version
        '1.2.3'
        >>> Version("1.2.3+abc").base_version
        '1.2.3'
        >>> Version("1!1.2.3+abc.dev1").base_version
        '1!1.2.3'

        “基本版本”是项目的公共版本，不包含任何预发布或后发布标记。
        """
        parts = []

        # 纪元
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # 发布段
        parts.append(".".join(str(x) for x in self.release))

        return "".join(parts)
    # 返回此版本是否为预发布版本的布尔值。
    def is_prerelease(self) -> bool:
        """Whether this version is a pre-release.

        >>> Version("1.2.3").is_prerelease
        False
        >>> Version("1.2.3a1").is_prerelease
        True
        >>> Version("1.2.3b1").is_prerelease
        True
        >>> Version("1.2.3rc1").is_prerelease
        True
        >>> Version("1.2.3dev1").is_prerelease
        True
        """
        # 返回 True 如果版本有开发版本号或预发布版本号
        return self.dev is not None or self.pre is not None

    @property
    # 返回此版本是否为后发布版本的布尔值。
    def is_postrelease(self) -> bool:
        """Whether this version is a post-release.

        >>> Version("1.2.3").is_postrelease
        False
        >>> Version("1.2.3.post1").is_postrelease
        True
        """
        # 返回 True 如果版本有后发布版本号
        return self.post is not None

    @property
    # 返回此版本是否为开发版本的布尔值。
    def is_devrelease(self) -> bool:
        """Whether this version is a development release.

        >>> Version("1.2.3").is_devrelease
        False
        >>> Version("1.2.3.dev1").is_devrelease
        True
        """
        # 返回 True 如果版本有开发版本号
        return self.dev is not None

    @property
    # 返回版本号的主要部分，如果不可用则返回 0。
    def major(self) -> int:
        """The first item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").major
        1
        """
        # 返回版本号列表中的第一个元素，如果不存在则返回 0
        return self.release[0] if len(self.release) >= 1 else 0

    @property
    # 返回版本号的次要部分，如果不可用则返回 0。
    def minor(self) -> int:
        """The second item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").minor
        2
        >>> Version("1").minor
        0
        """
        # 返回版本号列表中的第二个元素，如果不存在则返回 0
        return self.release[1] if len(self.release) >= 2 else 0

    @property
    # 返回版本号的微小部分，如果不可用则返回 0。
    def micro(self) -> int:
        """The third item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").micro
        3
        >>> Version("1").micro
        0
        """
        # 返回版本号列表中的第三个元素，如果不存在则返回 0
        return self.release[2] if len(self.release) >= 3 else 0
def _parse_letter_version(
    letter: Optional[str], number: Union[str, bytes, SupportsInt, None]
) -> Optional[Tuple[str, int]]:
    # 如果存在字母部分
    if letter:
        # 如果没有与之关联的数字，则默认为0
        if number is None:
            number = 0

        # 将字母部分规范化为小写形式
        letter = letter.lower()

        # 将某些单词视为其他单词的替代拼写，并将其规范化为首选拼写形式
        if letter == "alpha":
            letter = "a"
        elif letter == "beta":
            letter = "b"
        elif letter in ["c", "pre", "preview"]:
            letter = "rc"
        elif letter in ["rev", "r"]:
            letter = "post"

        # 返回规范化后的字母部分和数字部分组成的元组
        return letter, int(number)
    
    # 如果没有字母部分但有数字部分
    if not letter and number:
        # 假设如果给定数字但未给定字母，则使用隐式的后发布语法（例如 1.0-1）
        letter = "post"

        # 返回后发布的字母部分和数字部分组成的元组
        return letter, int(number)

    # 如果既没有字母部分也没有数字部分，则返回None
    return None


_local_version_separators = re.compile(r"[\._-]")


def _parse_local_version(local: Optional[str]) -> Optional[LocalType]:
    """
    Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
    """
    # 如果传入的本地版本字符串不为空
    if local is not None:
        # 使用正则表达式分隔符分割字符串，并将各部分转换为小写形式或整数形式（如果是数字）
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_separators.split(local)
        )
    # 如果本地版本字符串为空，则返回None
    return None


def _cmpkey(
    epoch: int,
    release: Tuple[int, ...],
    pre: Optional[Tuple[str, int]],
    post: Optional[Tuple[str, int]],
    dev: Optional[Tuple[str, int]],
    local: Optional[LocalType],
) -> CmpKey:
    # 当比较发布版本时，我们希望移除所有末尾的零以便比较。因此，我们将反转列表，删除所有前导零，
    # 直到遇到非零值，然后将其再次反转并转换为元组，用作排序键。
    _release = tuple(
        reversed(list(itertools.dropwhile(lambda x: x == 0, reversed(release))))
    )

    # 我们需要“欺骗”排序算法，使得 1.0.dev0 在 1.0a0 之前。我们通过滥用pre段来实现这一点，
    # 但是只有在没有pre或post段时才这样做。如果有pre或post段，则正常的排序规则将正确处理这种情况。
    if pre is None and post is None and dev is not None:
        _pre: CmpPrePostDevType = NegativeInfinity
    # 没有预发布版本（除非如上所述）的版本应该在具有预发布版本的版本之后排序。
    elif pre is None:
        _pre = Infinity
    else:
        _pre = pre

    # 没有后发布段的版本应该在具有后发布段的版本之前排序。
    if post is None:
        _post: CmpPrePostDevType = NegativeInfinity
    else:
        _post = post
    # 如果版本号没有开发段落（dev），将其排序在带有开发段落的版本号之后。
    if dev is None:
        # 设置开发段落为无穷大，表示该版本在排序时排在有开发段落的版本之后。
        _dev: CmpPrePostDevType = Infinity
    else:
        # 否则，使用给定的开发段落进行排序。
        _dev = dev

    if local is None:
        # 如果版本号没有本地段落（local），将其排序在带有本地段落的版本号之前。
        _local: CmpLocalType = NegativeInfinity
    else:
        # 否则，需要解析本地段落以实现 PEP440 中的排序规则。
        # - 字母数字段在数字段之前排序
        # - 字母数字段按字典顺序排序
        # - 数字段按数字顺序排序
        # - 当前缀完全匹配时，较短的版本号在较长的版本号之前排序
        _local = tuple(
            # 将本地段落转换为元组，以便排序
            (i, "") if isinstance(i, int) else (NegativeInfinity, i) for i in local
        )

    # 返回版本号的各个部分
    return epoch, _release, _pre, _post, _dev, _local
```