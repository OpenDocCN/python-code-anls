# `D:\src\scipysrc\scikit-learn\sklearn\externals\_packaging\version.py`

```
"""
Vendoered from
https://github.com/pypa/packaging/blob/main/packaging/version.py
"""
# 导入必要的模块和库
import collections
import itertools
import re
import warnings
from typing import Callable, Iterator, List, Optional, SupportsInt, Tuple, Union

# 导入本地模块中的结构体
from ._structures import Infinity, InfinityType, NegativeInfinity, NegativeInfinityType

# 仅对外公开以下标识符
__all__ = ["parse", "Version", "LegacyVersion", "InvalidVersion", "VERSION_PATTERN"]

# 定义用于表示无限的类型
InfiniteTypes = Union[InfinityType, NegativeInfinityType]

# 定义用于表示版本中的预发布、发布、开发版本等类型
PrePostDevType = Union[InfiniteTypes, Tuple[str, int]]

# 定义用于表示子版本号的类型
SubLocalType = Union[InfiniteTypes, int, str]

# 定义用于表示本地版本号的类型
LocalType = Union[
    NegativeInfinityType,
    Tuple[
        Union[
            SubLocalType,
            Tuple[SubLocalType, str],
            Tuple[NegativeInfinityType, SubLocalType],
        ],
        ...,
    ],
]

# 定义用于版本比较的关键元组类型
CmpKey = Tuple[
    int, Tuple[int, ...], PrePostDevType, PrePostDevType, PrePostDevType, LocalType
]

# 定义用于旧版本比较的关键元组类型
LegacyCmpKey = Tuple[int, Tuple[str, ...]]

# 定义版本比较方法的类型
VersionComparisonMethod = Callable[
    [Union[CmpKey, LegacyCmpKey], Union[CmpKey, LegacyCmpKey]], bool
]

# 定义版本结构体，用于存储版本信息
_Version = collections.namedtuple(
    "_Version", ["epoch", "release", "dev", "pre", "post", "local"]
)

def parse(version: str) -> Union["LegacyVersion", "Version"]:
    """Parse the given version from a string to an appropriate class.

    Parameters
    ----------
    version : str
        Version in a string format, eg. "0.9.1" or "1.2.dev0".

    Returns
    -------
    version : :class:`Version` object or a :class:`LegacyVersion` object
        Returned class depends on the given version: if is a valid
        PEP 440 version or a legacy version.
    """
    # 尝试使用 version 创建 Version 对象，并返回该对象
    try:
        return Version(version)
    # 如果 version 不合法，捕获 InvalidVersion 异常
    except InvalidVersion:
        # 返回一个 LegacyVersion 对象，传入 version 作为参数
        return LegacyVersion(version)
class InvalidVersion(ValueError):
    """
    An invalid version was found, users should refer to PEP 440.
    """


class _BaseVersion:
    _key: Union[CmpKey, LegacyCmpKey]

    def __hash__(self) -> int:
        # 返回基于 _key 的哈希值
        return hash(self._key)

    # Please keep the duplicated `isinstance` check
    # in the six comparisons hereunder
    # unless you find a way to avoid adding overhead function calls.
    def __lt__(self, other: "_BaseVersion") -> bool:
        # 如果 other 不是 _BaseVersion 的实例，则返回 NotImplemented
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        # 比较 self 和 other 的 _key，返回比较结果
        return self._key < other._key

    def __le__(self, other: "_BaseVersion") -> bool:
        # 如果 other 不是 _BaseVersion 的实例，则返回 NotImplemented
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        # 比较 self 和 other 的 _key，返回比较结果
        return self._key <= other._key

    def __eq__(self, other: object) -> bool:
        # 如果 other 不是 _BaseVersion 的实例，则返回 NotImplemented
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        # 比较 self 和 other 的 _key，返回比较结果
        return self._key == other._key

    def __ge__(self, other: "_BaseVersion") -> bool:
        # 如果 other 不是 _BaseVersion 的实例，则返回 NotImplemented
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        # 比较 self 和 other 的 _key，返回比较结果
        return self._key >= other._key

    def __gt__(self, other: "_BaseVersion") -> bool:
        # 如果 other 不是 _BaseVersion 的实例，则返回 NotImplemented
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        # 比较 self 和 other 的 _key，返回比较结果
        return self._key > other._key

    def __ne__(self, other: object) -> bool:
        # 如果 other 不是 _BaseVersion 的实例，则返回 NotImplemented
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        # 比较 self 和 other 的 _key，返回比较结果
        return self._key != other._key


class LegacyVersion(_BaseVersion):
    def __init__(self, version: str) -> None:
        self._version = str(version)
        # 根据 version 创建 _legacy_cmpkey，并赋值给 _key
        self._key = _legacy_cmpkey(self._version)

        # 发出警告，表示创建 LegacyVersion 已被弃用，并将在下一个主要版本中移除
        warnings.warn(
            "Creating a LegacyVersion has been deprecated and will be "
            "removed in the next major release",
            DeprecationWarning,
        )

    def __str__(self) -> str:
        # 返回 LegacyVersion 的字符串表示形式
        return self._version

    def __repr__(self) -> str:
        # 返回 LegacyVersion 的详细字符串表示形式
        return f"<LegacyVersion('{self}')>"

    @property
    def public(self) -> str:
        # 返回 LegacyVersion 的公开版本号
        return self._version

    @property
    def base_version(self) -> str:
        # 返回 LegacyVersion 的基本版本号
        return self._version

    @property
    def epoch(self) -> int:
        # 返回 LegacyVersion 的时代（epoch），默认为 -1
        return -1

    @property
    def release(self) -> None:
        # 返回 LegacyVersion 的发行版信息，默认为 None
        return None

    @property
    def pre(self) -> None:
        # 返回 LegacyVersion 的预发行版信息，默认为 None
        return None

    @property
    def post(self) -> None:
        # 返回 LegacyVersion 的后续版本信息，默认为 None
        return None

    @property
    def dev(self) -> None:
        # 返回 LegacyVersion 的开发版信息，默认为 None
        return None

    @property
    def local(self) -> None:
        # 返回 LegacyVersion 的本地版本信息，默认为 None
        return None

    @property
    def is_prerelease(self) -> bool:
        # 检查 LegacyVersion 是否为预发布版本，始终返回 False
        return False

    @property
    def is_postrelease(self) -> bool:
        # 检查 LegacyVersion 是否为后续版本，始终返回 False
        return False

    @property
    def is_devrelease(self) -> bool:
        # 检查 LegacyVersion 是否为开发版本，始终返回 False
        return False


_legacy_version_component_re = re.compile(r"(\d+ | [a-z]+ | \.| -)", re.VERBOSE)

_legacy_version_replacement_map = {
    "pre": "c",
    "preview": "c",
    "-": "final-",
    "rc": "c",
    "dev": "@",
}


def _parse_version_parts(s: str) -> Iterator[str]:
    # 将字符串 s 拆分为版本部分的迭代器，根据正则表达式匹配的规则
    # 使用正则表达式 `_legacy_version_component_re` 分割字符串 `s`，返回每个部分
    for part in _legacy_version_component_re.split(s):
        # 如果 part 在 `_legacy_version_replacement_map` 中存在替换，则替换为对应的值，否则保持原样
        part = _legacy_version_replacement_map.get(part, part)

        # 如果 part 为空或者为点号 "."，则跳过当前循环，继续下一个部分的处理
        if not part or part == ".":
            continue

        # 如果 part 的第一个字符是数字，则在左侧补零，使其具备数字比较的标准格式
        if part[:1] in "0123456789":
            # 生成带前导零的字符串，长度为 8
            yield part.zfill(8)
        else:
            # 如果 part 的第一个字符不是数字，则在其前面加上 "*"
            yield "*" + part

    # 确保生成的结果中 alpha/beta/candidate 排在 *final 的前面
    yield "*final"
# 定义一个函数，用于生成 LegacyCmpKey 对象，参数是版本号字符串，返回 LegacyCmpKey
def _legacy_cmpkey(version: str) -> LegacyCmpKey:

    # 在这里将 epoch 硬编码为 -1。PEP 440 规范的版本号的 epoch 必须大于等于 0。
    # 这样做将 LegacyVersion，它使用 setuptools 最初实现的事实标准，放在所有 PEP 440 版本之前。
    epoch = -1

    # 这个方案来自于 pkg_resources.parse_version，它是 setuptools 在其采用 packaging 库之前使用的方案。
    parts: List[str] = []
    # 对版本号字符串进行分析并处理
    for part in _parse_version_parts(version.lower()):
        if part.startswith("*"):
            # 如果版本号中包含 '*final'，则去除前面的 '-'
            if part < "*final":
                while parts and parts[-1] == "*final-":
                    parts.pop()

            # 去除每个数值部分末尾的零
            while parts and parts[-1] == "00000000":
                parts.pop()

        parts.append(part)

    # 返回 epoch 和 parts 组成的元组作为 LegacyCmpKey 对象
    return epoch, tuple(parts)


# 版本号模式的正则表达式模式字符串，用于匹配版本号字符串
VERSION_PATTERN = r"""
    v?                                                   # 可选的 'v' 字符
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))  # pre-release 标签
            [-_\.]?
            (?P<pre_n>[0-9]+)?                            # pre-release 版本号
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))                      # post-release 版本号
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)                    # post-release 标签
                [-_\.]?
                (?P<post_n2>[0-9]+)?                      # post-release 版本号
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)                                # dev-release 标签
            [-_\.]?
            (?P<dev_n>[0-9]+)?                            # dev-release 版本号
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""


# 表示版本号的类，继承自 _BaseVersion 类
class Version(_BaseVersion):

    # 正则表达式对象，用于匹配版本号字符串
    _regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)
    def __init__(self, version: str) -> None:
        # Validate the version and parse it into pieces
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion(f"Invalid version: '{version}'")

        # Store the parsed out pieces of the version
        self._version = _Version(
            epoch=int(match.group("epoch")) if match.group("epoch") else 0,  # Extract and convert epoch part of the version
            release=tuple(int(i) for i in match.group("release").split(".")),  # Extract and convert release segment into tuple of integers
            pre=_parse_letter_version(match.group("pre_l"), match.group("pre_n")),  # Parse pre-release information
            post=_parse_letter_version(
                match.group("post_l"), match.group("post_n1") or match.group("post_n2")  # Parse post-release information
            ),
            dev=_parse_letter_version(match.group("dev_l"), match.group("dev_n")),  # Parse development release information
            local=_parse_local_version(match.group("local")),  # Parse local version segment information
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
        # Return a string representation of the object
        return f"<Version('{self}')>"

    def __str__(self) -> str:
        parts = []

        # Epoch
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")  # Append epoch with '!' if not zero

        # Release segment
        parts.append(".".join(str(x) for x in self.release))  # Append release segment as a dot-separated string

        # Pre-release
        if self.pre is not None:
            parts.append("".join(str(x) for x in self.pre))  # Append pre-release information as a concatenated string

        # Post-release
        if self.post is not None:
            parts.append(f".post{self.post}")  # Append post-release version with '.post'

        # Development release
        if self.dev is not None:
            parts.append(f".dev{self.dev}")  # Append development release version with '.dev'

        # Local version segment
        if self.local is not None:
            parts.append(f"+{self.local}")  # Append local version segment prefixed with '+'

        return "".join(parts)  # Concatenate all parts into a single string and return it

    @property
    def epoch(self) -> int:
        _epoch: int = self._version.epoch  # Get epoch value from internal version object
        return _epoch  # Return epoch value

    @property
    def release(self) -> Tuple[int, ...]:
        _release: Tuple[int, ...] = self._version.release  # Get release segment tuple from internal version object
        return _release  # Return release segment tuple

    @property
    def pre(self) -> Optional[Tuple[str, int]]:
        _pre: Optional[Tuple[str, int]] = self._version.pre  # Get pre-release information from internal version object
        return _pre  # Return pre-release information

    @property
    def post(self) -> Optional[int]:
        return self._version.post[1] if self._version.post else None  # Return the second element of post-release information or None if not available

    @property
    def dev(self) -> Optional[int]:
        return self._version.dev[1] if self._version.dev else None  # Return the second element of development release information or None if not available

    @property
    def local(self) -> Optional[str]:
        if self._version.local:
            return ".".join(str(x) for x in self._version.local)  # Join local version segment into a dot-separated string if it exists
        else:
            return None  # Return None if no local version segment is present

    @property
    def public(self) -> str:
        return str(self).split("+", 1)[0]  # Return the public version string by splitting at the first '+' character
    # 返回基本版本号的字符串表示形式，包括 epoch 和 release 部分
    def base_version(self) -> str:
        parts = []

        # 如果 epoch 不为 0，则添加到版本号部分
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # 添加 release 部分，将其转换为点分隔的字符串形式
        parts.append(".".join(str(x) for x in self.release))

        # 将所有部分拼接成最终的版本号字符串
        return "".join(parts)

    @property
    # 检查当前版本是否为预发布版本，通过判断 dev 或 pre 属性是否存在来确定
    def is_prerelease(self) -> bool:
        return self.dev is not None or self.pre is not None

    @property
    # 检查当前版本是否为后发布版本，通过判断 post 属性是否存在来确定
    def is_postrelease(self) -> bool:
        return self.post is not None

    @property
    # 检查当前版本是否为开发发布版本，通过判断 dev 属性是否存在来确定
    def is_devrelease(self) -> bool:
        return self.dev is not None

    @property
    # 返回版本号的主要版本号部分，即 release 列表的第一个元素，若不存在则默认为 0
    def major(self) -> int:
        return self.release[0] if len(self.release) >= 1 else 0

    @property
    # 返回版本号的次要版本号部分，即 release 列表的第二个元素，若不存在则默认为 0
    def minor(self) -> int:
        return self.release[1] if len(self.release) >= 2 else 0

    @property
    # 返回版本号的微小版本号部分，即 release 列表的第三个元素，若不存在则默认为 0
    def micro(self) -> int:
        return self.release[2] if len(self.release) >= 3 else 0
def _parse_letter_version(
    letter: str, number: Union[str, bytes, SupportsInt]
) -> Optional[Tuple[str, int]]:
    # 如果存在字母部分
    if letter:
        # 如果没有与之相关的数字，我们假设预发布版本中隐含数字 0
        if number is None:
            number = 0

        # 将字母部分转换为小写形式
        letter = letter.lower()

        # 标准化一些单词作为其他单词的替代拼写，在这些情况下我们希望将拼写标准化为我们的首选拼写
        if letter == "alpha":
            letter = "a"
        elif letter == "beta":
            letter = "b"
        elif letter in ["c", "pre", "preview"]:
            letter = "rc"
        elif letter in ["rev", "r"]:
            letter = "post"

        # 返回标准化后的字母和对应的数字
        return letter, int(number)
    
    # 如果没有字母部分但有数字部分
    if not letter and number:
        # 我们假设如果给定数字但没有字母，则这是使用隐含的后发布版本语法（例如 1.0-1）
        letter = "post"
        return letter, int(number)

    # 如果既没有字母部分也没有数字部分，则返回 None
    return None


_local_version_separators = re.compile(r"[\._-]")


def _parse_local_version(local: str) -> Optional[LocalType]:
    """
    Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
    """
    # 如果输入的本地版本字符串不为空
    if local is not None:
        # 使用正则表达式分割本地版本字符串，并转换成小写形式（除非是数字）
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_separators.split(local)
        )
    # 如果本地版本字符串为空，则返回 None
    return None


def _cmpkey(
    epoch: int,
    release: Tuple[int, ...],
    pre: Optional[Tuple[str, int]],
    post: Optional[Tuple[str, int]],
    dev: Optional[Tuple[str, int]],
    local: Optional[Tuple[SubLocalType]],
) -> CmpKey:

    # 当比较发布版本时，我们希望删除所有末尾的零。因此，我们将反转列表，丢弃所有现在成为前导零的元素，直到遇到非零元素，然后将其再次反转为正确的顺序，并转换为元组，用作排序键。
    _release = tuple(
        reversed(list(itertools.dropwhile(lambda x: x == 0, reversed(release))))
    )

    # 我们需要“欺骗”排序算法，以使 1.0.dev0 在 1.0a0 之前。我们将利用预发布段来实现这一点，但是仅当没有预发布或后发布段时才这样做。如果有这些段中的任何一个，则通常的排序规则将正确处理此案例。
    if pre is None and post is None and dev is not None:
        _pre: PrePostDevType = NegativeInfinity
    # 没有预发布版本的版本（除非如上所述）应该在具有预发布版本的版本之后排序。
    elif pre is None:
        _pre = Infinity
    else:
        _pre = pre

    # 没有后发布段的版本应该在具有后发布段的版本之前排序。
    if post is None:
        _post: PrePostDevType = NegativeInfinity
    else:
        _post = post
    # 如果版本没有开发段（dev），则将 _dev 设置为无穷大，以确保这些版本在排序时排在有开发段的版本之后。
    if dev is None:
        _dev: PrePostDevType = Infinity

    else:
        # 如果版本有开发段，则将 _dev 设置为给定的开发段值。
        _dev = dev

    if local is None:
        # 如果版本没有本地段（local），则将 _local 设置为负无穷大，以确保这些版本在排序时排在有本地段的版本之前。
        _local: LocalType = NegativeInfinity
    else:
        # 如果版本有本地段，需要解析该段以实现 PEP440 中的排序规则。
        # - 字母数字段在数字段之前排序
        # - 字母数字段按字典顺序排序
        # - 数字段按数值排序
        # - 当前缀完全匹配时，较短的版本号排在较长的版本号之前
        _local = tuple(
            # 将本地段中的每个部分转换为元组，以便按照规则排序
            (i, "") if isinstance(i, int) else (NegativeInfinity, i) for i in local
        )

    # 返回解析后的版本信息，按照 epoch, release, pre, post, dev, local 的顺序返回
    return epoch, _release, _pre, _post, _dev, _local
```