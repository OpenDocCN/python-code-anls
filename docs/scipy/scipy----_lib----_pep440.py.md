# `D:\src\scipysrc\scipy\scipy\_lib\_pep440.py`

```
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


__all__ = [
    "parse", "Version", "LegacyVersion", "InvalidVersion", "VERSION_PATTERN",
]

# BEGIN packaging/_structures.py

# 定义一个表示正无穷大的类
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

# 定义一个表示负无穷大的类
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

# 导入负无穷大类，确保在后续使用中可用
NegativeInfinity = NegativeInfinity()

# 定义一个 namedtuple 用于表示版本号信息
_Version = collections.namedtuple(
    "_Version",
    ["epoch", "release", "dev", "pre", "post", "local"],


# 字符串 "_Version" 和列表 ["epoch", "release", "dev", "pre", "post", "local"] 是用作数据结构或者配置中的标识符
# 它们可能用于指示版本信息的特定部分或者其他相关的标识内容
# 解析给定的版本字符串，根据其是否符合 PEP 440 规范返回相应的 Version 对象或 LegacyVersion 对象
def parse(version):
    try:
        # 尝试创建 Version 对象
        return Version(version)
    except InvalidVersion:
        # 如果版本无效，则创建 LegacyVersion 对象
        return LegacyVersion(version)


# 定义一个自定义异常类，表示版本号无效
class InvalidVersion(ValueError):
    """
    An invalid version was found, users should refer to PEP 440.
    """


# 定义一个基础版本类 _BaseVersion
class _BaseVersion:

    # 实现 __hash__ 方法，用于哈希对象
    def __hash__(self):
        return hash(self._key)

    # 实现比较操作 __lt__ 方法，用于小于比较
    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)

    # 实现比较操作 __le__ 方法，用于小于等于比较
    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)

    # 实现比较操作 __eq__ 方法，用于等于比较
    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)

    # 实现比较操作 __ge__ 方法，用于大于等于比较
    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)

    # 实现比较操作 __gt__ 方法，用于大于比较
    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)

    # 实现比较操作 __ne__ 方法，用于不等于比较
    def __ne__(self, other):
        return self._compare(other, lambda s, o: s != o)

    # 内部比较方法 _compare，根据传入的方法和键进行比较
    def _compare(self, other, method):
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return method(self._key, other._key)


# 定义一个 LegacyVersion 类，继承自 _BaseVersion
class LegacyVersion(_BaseVersion):

    # 初始化方法，接受一个版本号字符串并初始化
    def __init__(self, version):
        self._version = str(version)
        # 使用 _legacy_cmpkey 方法生成比较键
        self._key = _legacy_cmpkey(self._version)

    # 返回版本号字符串表示形式
    def __str__(self):
        return self._version

    # 返回版本号的规范表示形式
    def __repr__(self):
        return f"<LegacyVersion({repr(str(self))})>"

    # 返回公共版本号，即原始版本号字符串
    @property
    def public(self):
        return self._version

    # 返回基础版本号，即原始版本号字符串
    @property
    def base_version(self):
        return self._version

    # 返回本地版本信息，对于 LegacyVersion 总是返回 None
    @property
    def local(self):
        return None

    # 判断是否为预发布版本，对于 LegacyVersion 总是返回 False
    @property
    def is_prerelease(self):
        return False

    # 判断是否为后发布版本，对于 LegacyVersion 总是返回 False
    @property
    def is_postrelease(self):
        return False


# 定义正则表达式模式，用于解析 LegacyVersion 版本号的组成部分
_legacy_version_component_re = re.compile(
    r"(\d+ | [a-z]+ | \.| -)", re.VERBOSE,
)

# 定义替换映射表，用于将 LegacyVersion 版本号中的特定部分替换成标准化表示
_legacy_version_replacement_map = {
    "pre": "c", "preview": "c", "-": "final-", "rc": "c", "dev": "@",
}


# 定义函数 _parse_version_parts，用于解析 LegacyVersion 版本号的各个部分
def _parse_version_parts(s):
    for part in _legacy_version_component_re.split(s):
        part = _legacy_version_replacement_map.get(part, part)

        if not part or part == ".":
            continue

        if part[:1] in "0123456789":
            # 对数字部分进行填充，以便进行数值比较
            yield part.zfill(8)
        else:
            yield "*" + part

    # 确保 alpha/beta/candidate 出现在 final 之前
    yield "*final"


# 定义函数 _legacy_cmpkey，用于生成 LegacyVersion 对象的比较键
def _legacy_cmpkey(version):
    # 在此固定使用 -1 作为 epoch，因为 PEP 440 版本号的 epoch 必须 >= 0
    epoch = -1

    # 该方案源自 pkg_resources.parse_version，即 setuptools 在采用 packaging 库之前的实现
    parts = []
    # 对给定版本号进行小写化并解析成各个部分
    for part in _parse_version_parts(version.lower()):
        # 如果版本号部分以 "*" 开头
        if part.startswith("*"):
            # 在预发布标签之前移除 "-" 符号
            if part < "*final":
                # 移除最后一个元素为 "*final-" 的部分
                while parts and parts[-1] == "*final-":
                    parts.pop()

            # 移除每个数字部分系列末尾的零
            while parts and parts[-1] == "00000000":
                parts.pop()

        # 将处理后的部分添加到列表中
        parts.append(part)

    # 将列表转换为元组，以便返回不可变的部分信息
    parts = tuple(parts)

    # 返回版本的 epoch 和处理后的部分元组
    return epoch, parts
# 定义版本号的正则表达式模式，用于解析版本号的各个部分
VERSION_PATTERN = r"""
    v?                                               # 可能以 'v' 开头
    (?:
        (?:(?P<epoch>[0-9]+)!)?                      # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)             # release segment
        (?P<pre>                                     # pre-release
            [-_\.]?                                  # 可选的分隔符
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))  # pre-release 标签
            [-_\.]?                                  # 可选的分隔符
            (?P<pre_n>[0-9]+)?                       # pre-release 版本号
        )?
        (?P<post>                                    # post release
            (?:-(?P<post_n1>[0-9]+))                 # post-release 版本号
            |
            (?:
                [-_\.]?                              # 可选的分隔符
                (?P<post_l>post|rev|r)                # post-release 标签
                [-_\.]?                              # 可选的分隔符
                (?P<post_n2>[0-9]+)?                  # post-release 版本号
            )
        )?
        (?P<dev>                                     # dev release
            [-_\.]?                                  # 可选的分隔符
            (?P<dev_l>dev)                           # dev-release 标签
            [-_\.]?                                  # 可选的分隔符
            (?P<dev_n>[0-9]+)?                       # dev-release 版本号
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?  # local version
"""

class Version(_BaseVersion):
    # 编译正则表达式，用于验证和解析版本号
    _regex = re.compile(
        r"^\s*" + VERSION_PATTERN + r"\s*$",         # 完整的版本号匹配模式
        re.VERBOSE | re.IGNORECASE,                  # 使用 VERBOSE 和 IGNORECASE 标志
    )

    def __init__(self, version):
        # 验证版本号并解析成各个部分
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion(f"Invalid version: '{version}'")

        # 存储解析后的版本号部分
        self._version = _Version(
            epoch=int(match.group("epoch")) if match.group("epoch") else 0,  # 解析 epoch，如果不存在则默认为 0
            release=tuple(int(i) for i in match.group("release").split(".")),  # 解析 release 部分
            pre=_parse_letter_version(
                match.group("pre_l"),
                match.group("pre_n"),
            ),  # 解析 pre-release 部分
            post=_parse_letter_version(
                match.group("post_l"),
                match.group("post_n1") or match.group("post_n2"),
            ),  # 解析 post-release 部分
            dev=_parse_letter_version(
                match.group("dev_l"),
                match.group("dev_n"),
            ),  # 解析 dev-release 部分
            local=_parse_local_version(match.group("local")),  # 解析 local 版本号部分
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
        return f"<Version({repr(str(self))})>"  # 返回 Version 对象的字符串表示形式
    def __str__(self):
        parts = []

        # Epoch
        if self._version.epoch != 0:
            parts.append(f"{self._version.epoch}!")  # 如果版本的 epoch 不为 0，则添加到 parts 列表中

        # Release segment
        parts.append(".".join(str(x) for x in self._version.release))  # 将版本的 release 段转换为字符串并加入 parts 列表

        # Pre-release
        if self._version.pre is not None:
            parts.append("".join(str(x) for x in self._version.pre))  # 如果存在 pre-release 版本，则将其加入 parts 列表

        # Post-release
        if self._version.post is not None:
            parts.append(f".post{self._version.post[1]}")  # 如果存在 post-release 版本，则将其加入 parts 列表

        # Development release
        if self._version.dev is not None:
            parts.append(f".dev{self._version.dev[1]}")  # 如果存在 development release 版本，则将其加入 parts 列表

        # Local version segment
        if self._version.local is not None:
            parts.append(
                "+{}".format(".".join(str(x) for x in self._version.local))
            )  # 如果存在 local 版本段，则将其加入 parts 列表

        return "".join(parts)  # 返回所有 parts 列表中的元素组成的字符串

    @property
    def public(self):
        return str(self).split("+", 1)[0]  # 返回版本字符串中第一个加号（+）前的部分作为公共版本号

    @property
    def base_version(self):
        parts = []

        # Epoch
        if self._version.epoch != 0:
            parts.append(f"{self._version.epoch}!")  # 如果版本的 epoch 不为 0，则添加到 parts 列表中

        # Release segment
        parts.append(".".join(str(x) for x in self._version.release))  # 将版本的 release 段转换为字符串并加入 parts 列表

        return "".join(parts)  # 返回所有 parts 列表中的元素组成的字符串作为基础版本号

    @property
    def local(self):
        version_string = str(self)
        if "+" in version_string:
            return version_string.split("+", 1)[1]  # 如果版本字符串中存在加号（+），则返回加号后面的部分作为 local 版本信息

    @property
    def is_prerelease(self):
        return bool(self._version.dev or self._version.pre)  # 如果存在开发版本（dev）或者预发布版本（pre），则返回 True，否则返回 False

    @property
    def is_postrelease(self):
        return bool(self._version.post)  # 如果存在后发布版本（post），则返回 True，否则返回 False
# 解析字母版本号，将字母和数字进行解析和规范化
def _parse_letter_version(letter, number):
    if letter:
        # 如果没有与字母关联的数字，则假设预发行版本中有一个隐式的 0
        if number is None:
            number = 0

        # 将字母转换为小写形式
        letter = letter.lower()

        # 将某些单词视为其他单词的替代拼写，将其规范化为首选拼写
        if letter == "alpha":
            letter = "a"
        elif letter == "beta":
            letter = "b"
        elif letter in ["c", "pre", "preview"]:
            letter = "rc"
        elif letter in ["rev", "r"]:
            letter = "post"

        # 返回规范化后的字母和整数化的数字
        return letter, int(number)
    if not letter and number:
        # 如果有数字但没有字母，则假设这是使用隐式后发布语法（例如，1.0-1）
        letter = "post"

        # 返回字母和整数化的数字
        return letter, int(number)


# 用于分割本地版本号的正则表达式，可以将类似于 abc.1.twelve 的字符串转换为 ("abc", 1, "twelve") 形式
_local_version_seperators = re.compile(r"[\._-]")


def _parse_local_version(local):
    """
    接受类似 abc.1.twelve 的字符串，将其转换为 ("abc", 1, "twelve") 形式。
    """
    if local is not None:
        # 使用正则表达式分割本地版本号字符串，将每部分转换为小写形式或整数（如果是数字）
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_seperators.split(local)
        )


def _cmpkey(epoch, release, pre, post, dev, local):
    # 比较发布版本时，希望移除所有末尾的零。因此，我们将列表反转，丢弃所有前导零，直到找到非零项，然后取剩余部分，
    # 再将其重新反转为正确的顺序，并将其作为排序键使用。
    release = tuple(
        reversed(list(
            itertools.dropwhile(
                lambda x: x == 0,
                reversed(release),
            )
        ))
    )

    # 我们需要“欺骗”排序算法，使得 1.0.dev0 在 1.0a0 之前。我们将通过滥用预发行段来实现这一点，
    # 但是只有在没有预发行或后发布段时才这样做。如果有这些段，通常的排序规则会正确处理此案例。
    if pre is None and post is None and dev is not None:
        pre = -Infinity
    # 没有预发行版本的版本（除非如上所述）应该在具有预发行版本之后排序。
    elif pre is None:
        pre = Infinity

    # 没有后发布段的版本应该在具有后发布段之前排序。
    if post is None:
        post = -Infinity

    # 没有开发版本段的版本应该在具有开发版本段之后排序。
    if dev is None:
        dev = Infinity

    if local is None:
        # 没有本地段的版本应该在具有本地段之后排序。
        local = -Infinity
    else:
        # 如果版本有本地段，则需要解析该段以实现 PEP440 中的排序规则。
        # - 字母数字段在数字段之前排序
        # - 字母数字段按字典顺序排序
        # - 数字段按数值大小排序
        # - 当前缀完全匹配时，较短的版本比较长的版本排序靠前
        # 将本地段转换为元组，处理成 (i, "") 如果 i 是整数，否则处理成 (-Infinity, i)
        local = tuple(
            (i, "") if isinstance(i, int) else (-Infinity, i)
            for i in local
        )

    # 返回解析后的版本元组
    return epoch, release, pre, post, dev, local
```