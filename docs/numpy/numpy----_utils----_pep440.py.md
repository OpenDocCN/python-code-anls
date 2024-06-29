# `.\numpy\numpy\_utils\_pep440.py`

```py
"""Utility to compare pep440 compatible version strings.

The LooseVersion and StrictVersion classes that distutils provides don't
work; they don't recognize anything like alpha/beta/rc/dev versions.
"""

# Copyright (c) Donald Stufft and individual contributors.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.

#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import collections
import itertools
import re

# 定义模块导出的公共接口
__all__ = [
    "parse", "Version", "LegacyVersion", "InvalidVersion", "VERSION_PATTERN",
]

# BEGIN packaging/_structures.py

# 定义无穷大的类
class Infinity:
    def __repr__(self):
        return "Infinity"

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return False

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __ne__(self, other):
        return not isinstance(other, self.__class__)

    def __gt__(self, other):
        return True

    def __ge__(self, other):
        return True

    def __neg__(self):
        return NegativeInfinity

# 定义负无穷大的类
Infinity = Infinity()

# 定义负无穷大的类
class NegativeInfinity:
    def __repr__(self):
        return "-Infinity"

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        return True

    def __le__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __ne__(self, other):
        return not isinstance(other, self.__class__)

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return False

    def __neg__(self):
        return Infinity

# BEGIN packaging/version.py

# 载入负无穷大类的别名
NegativeInfinity = NegativeInfinity()

# 使用 collections 模块的 namedtuple 定义版本元组结构
_Version = collections.namedtuple(
    "_Version",
    # 字符串 "_Version"，可能用作某种标识或关键字
    ["epoch", "release", "dev", "pre", "post", "local"],
    # 包含了多个字符串的列表，这些字符串可能代表软件版本的不同部分或阶段
def parse(version):
    """
    Parse the given version string and return either a :class:`Version` object
    or a :class:`LegacyVersion` object depending on if the given version is
    a valid PEP 440 version or a legacy version.
    """
    try:
        # 尝试使用给定的版本字符串创建一个 Version 对象
        return Version(version)
    except InvalidVersion:
        # 如果版本字符串无效，则创建一个 LegacyVersion 对象
        return LegacyVersion(version)


class InvalidVersion(ValueError):
    """
    An invalid version was found, users should refer to PEP 440.
    """


class _BaseVersion:

    def __hash__(self):
        # 返回对象的哈希值，使用 _key 属性
        return hash(self._key)

    def __lt__(self, other):
        # 比较当前对象是否小于另一个对象，使用 _compare 方法进行比较
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other):
        # 比较当前对象是否小于等于另一个对象，使用 _compare 方法进行比较
        return self._compare(other, lambda s, o: s <= o)

    def __eq__(self, other):
        # 比较当前对象是否等于另一个对象，使用 _compare 方法进行比较
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other):
        # 比较当前对象是否大于等于另一个对象，使用 _compare 方法进行比较
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other):
        # 比较当前对象是否大于另一个对象，使用 _compare 方法进行比较
        return self._compare(other, lambda s, o: s > o)

    def __ne__(self, other):
        # 比较当前对象是否不等于另一个对象，使用 _compare 方法进行比较
        return self._compare(other, lambda s, o: s != o)

    def _compare(self, other, method):
        # 通用比较方法，接受另一个对象和一个比较函数作为参数，用于比较 _key 属性
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return method(self._key, other._key)


class LegacyVersion(_BaseVersion):

    def __init__(self, version):
        # 初始化 LegacyVersion 对象，接受版本字符串作为参数，并生成 _key 属性
        self._version = str(version)
        self._key = _legacy_cmpkey(self._version)

    def __str__(self):
        # 返回 LegacyVersion 对象的字符串表示形式，即版本字符串
        return self._version

    def __repr__(self):
        # 返回 LegacyVersion 对象的详细字符串表示形式，用于调试和显示
        return "<LegacyVersion({0})>".format(repr(str(self)))

    @property
    def public(self):
        # 返回 LegacyVersion 对象的公共版本号（即版本字符串）
        return self._version

    @property
    def base_version(self):
        # 返回 LegacyVersion 对象的基础版本号（即版本字符串）
        return self._version

    @property
    def local(self):
        # 返回 LegacyVersion 对象的本地版本号，对于 LegacyVersion 永远为 None
        return None

    @property
    def is_prerelease(self):
        # 返回 LegacyVersion 对象是否为预发布版本，对于 LegacyVersion 永远为 False
        return False

    @property
    def is_postrelease(self):
        # 返回 LegacyVersion 对象是否为发布后版本，对于 LegacyVersion 永远为 False
        return False


_legacy_version_component_re = re.compile(
    r"(\d+ | [a-z]+ | \.| -)", re.VERBOSE,
)

_legacy_version_replacement_map = {
    "pre": "c", "preview": "c", "-": "final-", "rc": "c", "dev": "@",
}


def _parse_version_parts(s):
    # 解析版本字符串，返回版本组件的生成器
    for part in _legacy_version_component_re.split(s):
        part = _legacy_version_replacement_map.get(part, part)

        if not part or part == ".":
            continue

        if part[:1] in "0123456789":
            # 对于数字开头的部分，进行填充以便进行数字比较
            yield part.zfill(8)
        else:
            # 对于非数字开头的部分，添加通配符以确保正确排序
            yield "*" + part

    # 确保 alpha/beta/candidate 在 final 之前排序
    yield "*final"


def _legacy_cmpkey(version):
    # 在这里固定使用 -1 作为 epoch。PEP 440 的版本只能使用大于等于 0 的 epoch。
    # 这将有效地将 LegacyVersion，使用 setuptools 最初实现的标准，放在所有 PEP 440 版本之前。
    epoch = -1

    # 此方案来自 pkg_resources.parse_version，在其采用 packaging 库之前，使用的是这种方案。
    parts = []
    # 对版本号进行分割并转换为小写，返回每个部分的迭代器
    for part in _parse_version_parts(version.lower()):
        # 如果部分以 "*" 开头
        if part.startswith("*"):
            # 如果该部分小于 "*final"，则移除在预发布标签之前的 "-"
            if part < "*final":
                # 移除最后一个元素为 "*final-" 的所有元素
                while parts and parts[-1] == "*final-":
                    parts.pop()

            # 移除每个数字部分系列末尾的零
            while parts and parts[-1] == "00000000":
                parts.pop()

        # 将处理过的部分添加到列表中
        parts.append(part)
    
    # 将列表转换为元组
    parts = tuple(parts)

    # 返回版本号的 epoch 和处理后的 parts
    return epoch, parts
# 定义版本号匹配模式，用于解析和验证版本号字符串的格式
VERSION_PATTERN = r"""
    v?                                                  # 可选的 'v' 前缀
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?                                        # 可选的分隔符
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))   # pre-release 类型
            [-_\.]?                                        # 可选的分隔符
            (?P<pre_n>[0-9]+)?                             # 可选的 pre-release 版本号
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))                       # post-release 版本号1
            |                                              # 或者
            (?:
                [-_\.]?                                    # 可选的分隔符
                (?P<post_l>post|rev|r)                      # post-release 类型
                [-_\.]?                                    # 可选的分隔符
                (?P<post_n2>[0-9]+)?                        # 可选的 post-release 版本号2
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?                                        # 可选的分隔符
            (?P<dev_l>dev)                                 # dev-release 类型
            [-_\.]?                                        # 可选的分隔符
            (?P<dev_n>[0-9]+)?                             # 可选的 dev-release 版本号
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""

class Version(_BaseVersion):
    
    # 编译版本号正则表达式模式，忽略大小写和允许多行注释模式
    _regex = re.compile(
        r"^\s*" + VERSION_PATTERN + r"\s*$",
        re.VERBOSE | re.IGNORECASE,
    )

    def __init__(self, version):
        # 验证并解析给定的版本号字符串
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion("Invalid version: '{0}'".format(version))

        # 解析版本号各个部分并存储
        self._version = _Version(
            epoch=int(match.group("epoch")) if match.group("epoch") else 0,  # 解析 epoch
            release=tuple(int(i) for i in match.group("release").split(".")),  # 解析 release
            pre=_parse_letter_version(
                match.group("pre_l"),
                match.group("pre_n"),
            ),  # 解析 pre-release
            post=_parse_letter_version(
                match.group("post_l"),
                match.group("post_n1") or match.group("post_n2"),
            ),  # 解析 post-release
            dev=_parse_letter_version(
                match.group("dev_l"),
                match.group("dev_n"),
            ),  # 解析 dev-release
            local=_parse_local_version(match.group("local")),  # 解析 local version
        )

        # 生成用于排序的关键字
        self._key = _cmpkey(
            self._version.epoch,
            self._version.release,
            self._version.pre,
            self._version.post,
            self._version.dev,
            self._version.local,
        )

    def __repr__(self):
        return "<Version({0})>".format(repr(str(self)))
    # 返回对象的字符串表示形式
    def __str__(self):
        # 初始化一个空列表，用于存储版本号的各个部分
        parts = []

        # 如果版本中的 epoch 不为 0，则添加到 parts 列表中
        if self._version.epoch != 0:
            parts.append("{0}!".format(self._version.epoch))

        # 添加版本号的 release 段，转换为字符串后添加到 parts 列表中
        parts.append(".".join(str(x) for x in self._version.release))

        # 如果存在预发行版本（pre-release），则将其转换为字符串添加到 parts 列表中
        if self._version.pre is not None:
            parts.append("".join(str(x) for x in self._version.pre))

        # 如果存在后发布版本（post-release），则添加以 ".post" 开头的版本号到 parts 列表中
        if self._version.post is not None:
            parts.append(".post{0}".format(self._version.post[1]))

        # 如果存在开发中版本（development release），则添加以 ".dev" 开头的版本号到 parts 列表中
        if self._version.dev is not None:
            parts.append(".dev{0}".format(self._version.dev[1]))

        # 如果存在本地版本段（local version segment），则添加以 "+" 开头的版本号到 parts 列表中
        if self._version.local is not None:
            parts.append(
                "+{0}".format(".".join(str(x) for x in self._version.local))
            )

        # 将 parts 列表中的所有部分连接成一个字符串并返回
        return "".join(parts)

    # 返回公共版本号（去除本地版本信息后的版本号）
    @property
    def public(self):
        # 将版本字符串按 "+" 分割，并取第一个部分作为公共版本号
        return str(self).split("+", 1)[0]

    # 返回基本版本号（去除开发版本信息和本地版本信息后的版本号）
    @property
    def base_version(self):
        # 初始化一个空列表，用于存储版本号的各个部分
        parts = []

        # 如果版本中的 epoch 不为 0，则添加到 parts 列表中
        if self._version.epoch != 0:
            parts.append("{0}!".format(self._version.epoch))

        # 添加版本号的 release 段，转换为字符串后添加到 parts 列表中
        parts.append(".".join(str(x) for x in self._version.release))

        # 将 parts 列表中的所有部分连接成一个字符串并返回
        return "".join(parts)

    # 返回本地版本信息（若存在）
    @property
    def local(self):
        # 获取版本号的字符串表示形式
        version_string = str(self)
        # 如果版本号中包含 "+"
        if "+" in version_string:
            # 将版本号按 "+" 分割，并取第二部分作为本地版本信息
            return version_string.split("+", 1)[1]

    # 返回是否为预发布版本（包含开发版本或预发布版本）
    @property
    def is_prerelease(self):
        # 如果存在开发版本（dev）或预发布版本（pre），返回 True，否则返回 False
        return bool(self._version.dev or self._version.pre)

    # 返回是否为后发布版本
    @property
    def is_postrelease(self):
        # 如果存在后发布版本（post），返回 True，否则返回 False
        return bool(self._version.post)
def _parse_letter_version(letter, number):
    if letter:
        # 如果存在 letter，则假设在预发布版本中没有数字与之关联时默认为 0
        if number is None:
            number = 0

        # 将 letter 规范化为小写形式
        letter = letter.lower()

        # 将一些单词视为其他单词的替代拼写，在这些情况下将其规范化为首选拼写
        if letter == "alpha":
            letter = "a"
        elif letter == "beta":
            letter = "b"
        elif letter in ["c", "pre", "preview"]:
            letter = "rc"
        elif letter in ["rev", "r"]:
            letter = "post"

        # 返回规范化后的 letter 和转换为整数的 number
        return letter, int(number)
    
    if not letter and number:
        # 如果没有 letter 但有 number，则假设这是使用隐含的 post 发布语法（例如，1.0-1）
        letter = "post"

        # 返回 letter 和转换为整数的 number
        return letter, int(number)


_local_version_seperators = re.compile(r"[\._-]")


def _parse_local_version(local):
    """
    Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
    """
    if local is not None:
        # 将 local 字符串根据分隔符（`.`、`_`、`-`）分割并转换为小写或整数形式
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_seperators.split(local)
        )


def _cmpkey(epoch, release, pre, post, dev, local):
    # 当比较发布版本时，我们希望去掉所有尾随的零。因此，将列表反转，删除所有前导的零，
    # 直到遇到非零元素，然后将其再次反转为正确的顺序，并转换为元组作为排序键。
    release = tuple(
        reversed(list(
            itertools.dropwhile(
                lambda x: x == 0,
                reversed(release),
            )
        ))
    )

    # 我们需要“欺骗”排序算法，以便在 1.0.dev0 之前放置 1.0a0。我们将通过滥用 pre 段来做到这一点，
    # 但仅当没有 pre 或 post 段时才这样做。如果有这些段中的任何一个，正常的排序规则将正确处理这种情况。
    if pre is None and post is None and dev is not None:
        pre = -Infinity
    # 没有预发布版本的版本（除非如上所述）应该在具有预发布版本的版本之后排序。
    elif pre is None:
        pre = Infinity

    # 没有 post 段的版本应该在具有 post 段的版本之前排序。
    if post is None:
        post = -Infinity

    # 没有开发段的版本应该在具有开发段的版本之后排序。
    if dev is None:
        dev = Infinity

    if local is None:
        # 没有本地段的版本应该在具有本地段的版本之前排序。
        local = -Infinity
    else:
        # 如果版本有本地段（local segment），需要解析该段以实现 PEP440 中的排序规则。
        # - 字母数字段在数字段之前排序
        # - 字母数字段按词典顺序排序
        # - 数字段按数值排序
        # - 当前缀完全匹配时，较短的版本在较长的版本之前排序
        # 将本地段转换为元组，处理为(i, "")如果i是整数，否则处理为(-Infinity, i)
        local = tuple(
            (i, "") if isinstance(i, int) else (-Infinity, i)
            for i in local
        )

    # 返回解析后的各个版本部分的元组
    return epoch, release, pre, post, dev, local
```