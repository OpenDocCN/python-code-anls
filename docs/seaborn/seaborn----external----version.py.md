# `D:\src\scipysrc\seaborn\seaborn\external\version.py`

```
"""
Extract reference documentation from the pypa/packaging source tree.

In the process of copying, some unused methods / classes were removed.
These include:

- parse()
- anything involving LegacyVersion

This software is made available under the terms of *either* of the licenses
found in LICENSE.APACHE or LICENSE.BSD. Contributions to this software is made
under the terms of *both* these licenses.

Vendored from:
- https://github.com/pypa/packaging/
- commit ba07d8287b4554754ac7178d177033ea3f75d489 (09/09/2021)
"""


# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.


import collections
import itertools
import re
from typing import Callable, Optional, SupportsInt, Tuple, Union

__all__ = ["Version", "InvalidVersion", "VERSION_PATTERN"]


# Vendored from https://github.com/pypa/packaging/blob/main/packaging/_structures.py

# 定义一个表示正无穷大的类型
class InfinityType:
    def __repr__(self) -> str:
        return "Infinity"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        return False

    def __le__(self, other: object) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __ne__(self, other: object) -> bool:
        return not isinstance(other, self.__class__)

    def __gt__(self, other: object) -> bool:
        return True

    def __ge__(self, other: object) -> bool:
        return True

    def __neg__(self: object) -> "NegativeInfinityType":
        return NegativeInfinity


# 定义一个表示负无穷大的类型
Infinity = InfinityType()


class NegativeInfinityType:
    def __repr__(self) -> str:
        return "-Infinity"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        return True

    def __le__(self, other: object) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __ne__(self, other: object) -> bool:
        return not isinstance(other, self.__class__)

    def __gt__(self, other: object) -> bool:
        return False

    def __ge__(self, other: object) -> bool:
        return False

    def __neg__(self: object) -> InfinityType:
        return Infinity


# 定义一个表示负无穷大的类型
NegativeInfinity = NegativeInfinityType()


# Vendored from https://github.com/pypa/packaging/blob/main/packaging/version.py

# 定义一些类型别名
InfiniteTypes = Union[InfinityType, NegativeInfinityType]
PrePostDevType = Union[InfiniteTypes, Tuple[str, int]]
SubLocalType = Union[InfiniteTypes, int, str]
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
CmpKey = Tuple[
    int, Tuple[int, ...], PrePostDevType, PrePostDevType, PrePostDevType, LocalType
]
# 定义一个类型别名，LegacyCmpKey 是一个元组，包含一个整数和一个字符串元组
LegacyCmpKey = Tuple[int, Tuple[str, ...]]

# 定义一个类型别名，VersionComparisonMethod 是一个可调用对象，接受两个参数为 CmpKey 或 LegacyCmpKey，返回布尔值
VersionComparisonMethod = Callable[[Union[CmpKey, LegacyCmpKey], Union[CmpKey, LegacyCmpKey]], bool]

# 定义一个命名元组 _Version，包含 epoch、release、dev、pre、post 和 local 六个字段
_Version = collections.namedtuple("_Version", ["epoch", "release", "dev", "pre", "post", "local"])


class InvalidVersion(ValueError):
    """
    未找到有效的版本，用户应参考 PEP 440。
    """
    # 异常类，表示找到了无效的版本


class _BaseVersion:
    _key: Union[CmpKey, LegacyCmpKey]

    def __hash__(self) -> int:
        # 返回对象的哈希值
        return hash(self._key)

    # __lt__ 方法，实现小于号比较操作符
    def __lt__(self, other: "_BaseVersion") -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key < other._key

    # __le__ 方法，实现小于等于号比较操作符
    def __le__(self, other: "_BaseVersion") -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key <= other._key

    # __eq__ 方法，实现等于号比较操作符
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key == other._key

    # __ge__ 方法，实现大于等于号比较操作符
    def __ge__(self, other: "_BaseVersion") -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key >= other._key

    # __gt__ 方法，实现大于号比较操作符
    def __gt__(self, other: "_BaseVersion") -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key > other._key

    # __ne__ 方法，实现不等于号比较操作符
    def __ne__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key != other._key


# 版本模式的正则表达式模式字符串，用于解析版本号
# 此正则表达式不强制以字符串的开始和结束锚定，以便第三方代码可以更轻松地重用
VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
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


class Version(_BaseVersion):
    # 版本类继承自 _BaseVersion

    # 编译的正则表达式对象，用于验证版本号字符串的格式
    _regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)
    def __init__(self, version: str) -> None:
        # 验证版本并解析成各个部分

        # 使用正则表达式匹配版本号
        match = self._regex.search(version)
        if not match:
            # 如果匹配失败，抛出无效版本异常
            raise InvalidVersion(f"Invalid version: '{version}'")

        # 解析版本号的各个部分并存储
        self._version = _Version(
            epoch=int(match.group("epoch")) if match.group("epoch") else 0,  # 解析 epoch 部分
            release=tuple(int(i) for i in match.group("release").split(".")),  # 解析 release 部分
            pre=_parse_letter_version(match.group("pre_l"), match.group("pre_n")),  # 解析 pre-release 部分
            post=_parse_letter_version(
                match.group("post_l"), match.group("post_n1") or match.group("post_n2")  # 解析 post-release 部分
            ),
            dev=_parse_letter_version(match.group("dev_l"), match.group("dev_n")),  # 解析 dev-release 部分
            local=_parse_local_version(match.group("local")),  # 解析 local version 部分
        )

        # 生成用于排序的键
        self._key = _cmpkey(
            self._version.epoch,
            self._version.release,
            self._version.pre,
            self._version.post,
            self._version.dev,
            self._version.local,
        )

    def __repr__(self) -> str:
        # 返回对象的字符串表示形式，供开发者查看调试使用
        return f"<Version('{self}')>"

    def __str__(self) -> str:
        parts = []

        # Epoch 部分
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release 部分
        parts.append(".".join(str(x) for x in self.release))

        # Pre-release 部分
        if self.pre is not None:
            parts.append("".join(str(x) for x in self.pre))

        # Post-release 部分
        if self.post is not None:
            parts.append(f".post{self.post}")

        # Development release 部分
        if self.dev is not None:
            parts.append(f".dev{self.dev}")

        # Local version 部分
        if self.local is not None:
            parts.append(f"+{self.local}")

        return "".join(parts)

    @property
    def epoch(self) -> int:
        # 返回版本号的 epoch 部分
        _epoch: int = self._version.epoch
        return _epoch

    @property
    def release(self) -> Tuple[int, ...]:
        # 返回版本号的 release 部分
        _release: Tuple[int, ...] = self._version.release
        return _release

    @property
    def pre(self) -> Optional[Tuple[str, int]]:
        # 返回版本号的 pre-release 部分
        _pre: Optional[Tuple[str, int]] = self._version.pre
        return _pre

    @property
    def post(self) -> Optional[int]:
        # 返回版本号的 post-release 部分
        return self._version.post[1] if self._version.post else None

    @property
    def dev(self) -> Optional[int]:
        # 返回版本号的 dev-release 部分
        return self._version.dev[1] if self._version.dev else None

    @property
    def local(self) -> Optional[str]:
        # 返回版本号的 local version 部分
        if self._version.local:
            return ".".join(str(x) for x in self._version.local)
        else:
            return None

    @property
    def public(self) -> str:
        # 返回版本号的公共部分（不包括 local version）
        return str(self).split("+", 1)[0]
    # 返回基本版本号的字符串表示形式
    def base_version(self) -> str:
        parts = []

        # 如果存在 epoch，添加到版本号的部分中
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # 添加 release 段的字符串表示形式，用点号连接各个部分
        parts.append(".".join(str(x) for x in self.release))

        # 将所有部分连接成一个字符串并返回
        return "".join(parts)

    # 检查版本是否为预发布版本
    @property
    def is_prerelease(self) -> bool:
        return self.dev is not None or self.pre is not None

    # 检查版本是否为后发布版本
    @property
    def is_postrelease(self) -> bool:
        return self.post is not None

    # 检查版本是否为开发版本
    @property
    def is_devrelease(self) -> bool:
        return self.dev is not None

    # 返回主版本号，如果 release 中至少包含一个元素
    @property
    def major(self) -> int:
        return self.release[0] if len(self.release) >= 1 else 0

    # 返回次版本号，如果 release 中至少包含两个元素
    @property
    def minor(self) -> int:
        return self.release[1] if len(self.release) >= 2 else 0

    # 返回微版本号，如果 release 中至少包含三个元素
    @property
    def micro(self) -> int:
        return self.release[2] if len(self.release) >= 3 else 0
# 解析字母版本号，返回一个元组，包含规范化后的字母和对应的数字
def _parse_letter_version(
    letter: str, number: Union[str, bytes, SupportsInt]
) -> Optional[Tuple[str, int]]:

    if letter:
        # 如果没有与之关联的数字，我们认为在预发行版本中有一个隐含的 0
        if number is None:
            number = 0

        # 将字母规范化为小写形式
        letter = letter.lower()

        # 将一些单词视为另一些单词的替代拼写，在这些情况下，我们希望将拼写规范化为我们首选的拼写方式。
        if letter == "alpha":
            letter = "a"
        elif letter == "beta":
            letter = "b"
        elif letter in ["c", "pre", "preview"]:
            letter = "rc"
        elif letter in ["rev", "r"]:
            letter = "post"

        return letter, int(number)
    
    # 如果没有字母但有数字，则假设这是使用隐含的后发行版本语法（例如 1.0-1）
    if not letter and number:
        letter = "post"
        return letter, int(number)

    # 如果既没有字母也没有数字，则返回空
    return None


# 正则表达式，用于分隔本地版本字符串的分隔符
_local_version_separators = re.compile(r"[\._-]")


# 解析本地版本字符串，返回一个元组，包含规范化后的本地版本信息
def _parse_local_version(local: str) -> Optional[LocalType]:
    """
    将类似 abc.1.twelve 的字符串转换为 ("abc", 1, "twelve") 的元组形式。
    """
    if local is not None:
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_separators.split(local)
        )
    return None


# 构建用于版本比较的键值，返回一个排序键
def _cmpkey(
    epoch: int,
    release: Tuple[int, ...],
    pre: Optional[Tuple[str, int]],
    post: Optional[Tuple[str, int]],
    dev: Optional[Tuple[str, int]],
    local: Optional[Tuple[SubLocalType]],
) -> CmpKey:

    # 比较发行版本时，我们希望删除所有尾随的零。因此，我们将反转列表，丢弃所有前导零直到找到非零元素，然后再次反转列表，转换为元组作为排序键。
    _release = tuple(
        reversed(list(itertools.dropwhile(lambda x: x == 0, reversed(release))))
    )

    # 如果没有预发行版本和后发行版本，但有开发版本，我们需要“欺骗”排序算法，以便将 1.0.dev0 排在 1.0a0 之前。
    # 我们将滥用预发行段，但仅当没有预发行或后发行段时才这样做。如果存在这些段，则通常的排序规则将正确处理此情况。
    if pre is None and post is None and dev is not None:
        _pre: PrePostDevType = NegativeInfinity
    # 没有预发行版本的版本应在具有预发行版本的版本之后排序。
    elif pre is None:
        _pre = Infinity
    else:
        _pre = pre

    # 没有后发行段的版本应在具有后发行段的版本之前排序。
    if post is None:
        _post: PrePostDevType = NegativeInfinity
    else:
        _post = post
    #`
    # 如果没有开发版本号段，则将 _dev 设置为 Infinity，以确保这些版本号在有开发版本号段的版本号之后排序。
    if dev is None:
        _dev: PrePostDevType = Infinity

    else:
        # 否则，使用提供的开发版本号。
        _dev = dev

    if local is None:
        # 如果没有本地版本号段，则将 _local 设置为 NegativeInfinity，以确保这些版本号在有本地版本号段的版本号之前排序。
        _local: LocalType = NegativeInfinity
    else:
        # 如果有本地版本号段，则需要解析该段以实现 PEP440 中的排序规则。
        # - 字母数字段在数字段之前排序
        # - 字母数字段按字典顺序排序
        # - 数字段按数字顺序排序
        # - 当前缀完全匹配时，较短的版本号段在较长的版本号段之前排序
        _local = tuple(
            (i, "") if isinstance(i, int) else (NegativeInfinity, i) for i in local
        )

    # 返回各个版本号段的结果
    return epoch, _release, _pre, _post, _dev, _local
```