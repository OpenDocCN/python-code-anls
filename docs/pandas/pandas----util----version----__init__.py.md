# `D:\src\scipysrc\pandas\pandas\util\version\__init__.py`

```
# 导入collections模块中的namedtuple函数，用于创建命名元组_Version
_Version = collections.namedtuple(
    "_Version", ["epoch", "release", "dev", "pre", "post", "local"]
)

# 定义函数parse，接受一个版本号字符串，返回LegacyVersion或Version对象
def parse(version: str) -> LegacyVersion | Version:
    """
    Parse the given version string and return either a :class:`Version` object
    or a :class:`LegacyVersion` object.
    """
    # 尝试创建一个 Version 对象，如果 version 是一个有效的 PEP 440 版本，则创建 Version 对象
    # 如果 version 不是一个有效的 PEP 440 版本，则抛出 InvalidVersion 异常，然后创建一个 LegacyVersion 对象
    try:
        return Version(version)
    except InvalidVersion:
        return LegacyVersion(version)
class InvalidVersion(ValueError):
    """
    An invalid version was found, users should refer to PEP 440.

    Examples
    --------
    >>> pd.util.version.Version("1.")
    Traceback (most recent call last):
    InvalidVersion: Invalid version: '1.'
    """


class _BaseVersion:
    _key: CmpKey | LegacyCmpKey

    def __hash__(self) -> int:
        return hash(self._key)

    # Please keep the duplicated `isinstance` check
    # in the six comparisons hereunder
    # unless you find a way to avoid adding overhead function calls.
    def __lt__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key < other._key

    def __le__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key <= other._key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key == other._key

    def __ge__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key >= other._key

    def __gt__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key > other._key

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key != other._key


class LegacyVersion(_BaseVersion):
    def __init__(self, version: str) -> None:
        self._version = str(version)  # 将输入的版本号转换为字符串并存储在实例变量中
        self._key = _legacy_cmpkey(self._version)  # 使用_legacy_cmpkey函数生成版本的比较键

        warnings.warn(
            "Creating a LegacyVersion has been deprecated and will be "
            "removed in the next major release.",
            DeprecationWarning,
        )

    def __str__(self) -> str:
        return self._version  # 返回版本号的字符串表示形式

    def __repr__(self) -> str:
        return f"<LegacyVersion('{self}')>"  # 返回版本对象的可打印表示形式

    @property
    def public(self) -> str:
        return self._version  # 返回版本号的公共版本字符串表示形式

    @property
    def base_version(self) -> str:
        return self._version  # 返回版本号的基本版本字符串表示形式

    @property
    def epoch(self) -> int:
        return -1  # 返回版本号的时代（epoch），这里固定为-1

    @property
    def release(self) -> None:
        return None  # 返回版本号的发布属性，这里为None

    @property
    def pre(self) -> None:
        return None  # 返回版本号的预发布属性，这里为None

    @property
    def post(self) -> None:
        return None  # 返回版本号的后发布属性，这里为None

    @property
    def dev(self) -> None:
        return None  # 返回版本号的开发版本属性，这里为None

    @property
    def local(self) -> None:
        return None  # 返回版本号的本地版本属性，这里为None

    @property
    def is_prerelease(self) -> bool:
        return False  # 返回是否为预发布版本的布尔值，这里固定为False

    @property
    def is_postrelease(self) -> bool:
        return False  # 返回是否为后发布版本的布尔值，这里固定为False

    @property
    def is_devrelease(self) -> bool:
        return False  # 返回是否为开发版本的布尔值，这里固定为False


_legacy_version_component_re = re.compile(r"(\d+ | [a-z]+ | \.| -)", re.VERBOSE)
# 编译正则表达式，用于匹配版本号中的组件（数字、字母、点和破折号）

_legacy_version_replacement_map = {
    "pre": "c",  # 将版本号中的"pre"替换为"c"
    "preview": "c",  # 将版本号中的"preview"替换为"c"
    "-": "final-",  # 将版本号中的"-"替换为"final-"
    # 定义一个包含字符串键值对的字典，键 "rc" 对应值 "c"，键 "dev" 对应值 "@"
    "rc": "c",
    "dev": "@",
}


def _parse_version_parts(s: str) -> Iterator[str]:
    # 使用正则表达式 _legacy_version_component_re 拆分输入字符串 s
    for part in _legacy_version_component_re.split(s):
        # 获取映射后的部分，如果没有映射或者为 "."，则跳过
        mapped_part = _legacy_version_replacement_map.get(part, part)

        # 如果映射后的部分为空或者为 "."，则继续下一轮循环
        if not mapped_part or mapped_part == ".":
            continue

        # 如果映射后的部分以数字开头，则填充0使其达到8位，用于数值比较
        if mapped_part[:1] in "0123456789":
            # pad for numeric comparison
            yield mapped_part.zfill(8)
        else:
            # 否则在映射后的部分前加上 "*"，表示非数字的部分
            yield "*" + mapped_part

    # 确保 alpha/beta/candidate 出现在 final 之前
    yield "*final"


def _legacy_cmpkey(version: str) -> LegacyCmpKey:
    # 在这里硬编码 epoch 为 -1。根据 PEP 440 的规定，版本号的 epoch 只能大于等于 0。
    # 这将有效地将 LegacyVersion（使用 setuptools 最初实现的事实标准）放在所有 PEP 440 版本之前。
    epoch = -1

    # 这个方案取自于 pkg_resources.parse_version，在其使用 packaging 库之前的 setuptools 实现。
    parts: list[str] = []
    for part in _parse_version_parts(version.lower()):
        if part.startswith("*"):
            # 在预发行标签之前删除 "-"
            if part < "*final":
                while parts and parts[-1] == "*final-":
                    parts.pop()

            # 移除每个数值部分系列的尾随零
            while parts and parts[-1] == "00000000":
                parts.pop()

        parts.append(part)

    return epoch, tuple(parts)


# 未锚定到字符串的开头和结尾，以便第三方代码更容易重用
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
    # 定义版本号对象 Version，使用预编译的正则表达式来匹配版本号模式
    _regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)
    def __init__(self, version: str) -> None:
        # Validate the version and parse it into pieces
        match = self._regex.search(version)
        # 如果没有找到匹配的版本号，抛出 InvalidVersion 异常
        if not match:
            raise InvalidVersion(f"Invalid version: '{version}'")

        # Store the parsed out pieces of the version
        # 解析版本号的不同部分并存储
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
        # 生成一个用于排序的关键字
        self._key = _cmpkey(
            self._version.epoch,
            self._version.release,
            self._version.pre,
            self._version.post,
            self._version.dev,
            self._version.local,
        )

    def __repr__(self) -> str:
        # 返回对象的字符串表示形式，用于调试和日志记录
        return f"<Version('{self}')>"

    def __str__(self) -> str:
        parts = []

        # Epoch
        # 如果 epoch 不为 0，添加到版本字符串中
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release segment
        # 添加版本发布段到版本字符串中
        parts.append(".".join([str(x) for x in self.release]))

        # Pre-release
        # 如果存在 pre-release，添加到版本字符串中
        if self.pre is not None:
            parts.append("".join([str(x) for x in self.pre]))

        # Post-release
        # 如果存在 post-release，添加到版本字符串中
        if self.post is not None:
            parts.append(f".post{self.post}")

        # Development release
        # 如果存在 development release，添加到版本字符串中
        if self.dev is not None:
            parts.append(f".dev{self.dev}")

        # Local version segment
        # 如果存在 local version segment，添加到版本字符串中
        if self.local is not None:
            parts.append(f"+{self.local}")

        # 返回构建好的版本字符串
        return "".join(parts)

    @property
    def epoch(self) -> int:
        # 获取版本的 epoch 属性
        _epoch: int = self._version.epoch
        return _epoch

    @property
    def release(self) -> tuple[int, ...]:
        # 获取版本的 release 属性
        _release: tuple[int, ...] = self._version.release
        return _release

    @property
    def pre(self) -> tuple[str, int] | None:
        # 获取版本的 pre-release 属性
        _pre: tuple[str, int] | None = self._version.pre
        return _pre

    @property
    def post(self) -> int | None:
        # 获取版本的 post-release 属性的第二个元素
        return self._version.post[1] if self._version.post else None

    @property
    def dev(self) -> int | None:
        # 获取版本的 development release 属性的第二个元素
        return self._version.dev[1] if self._version.dev else None

    @property
    def local(self) -> str | None:
        # 获取版本的 local version segment 属性
        if self._version.local:
            return ".".join([str(x) for x in self._version.local])
        else:
            return None

    @property
    def public(self) -> str:
        # 获取版本的 public 部分，即不包含 local version segment 的部分
        return str(self).split("+", 1)[0]
    # 返回基础版本号的字符串表示
    def base_version(self) -> str:
        parts = []

        # 如果存在 Epoch，添加 Epoch 到版本号中
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # 添加主版本号、次版本号和修订号到版本号中
        parts.append(".".join([str(x) for x in self.release]))

        # 将所有部分组合成一个完整的版本号字符串
        return "".join(parts)

    # 检查当前版本是否为预发布版本
    @property
    def is_prerelease(self) -> bool:
        return self.dev is not None or self.pre is not None

    # 检查当前版本是否为发布后版本
    @property
    def is_postrelease(self) -> bool:
        return self.post is not None

    # 检查当前版本是否为开发中版本
    @property
    def is_devrelease(self) -> bool:
        return self.dev is not None

    # 返回主版本号
    @property
    def major(self) -> int:
        return self.release[0] if len(self.release) >= 1 else 0

    # 返回次版本号
    @property
    def minor(self) -> int:
        return self.release[1] if len(self.release) >= 2 else 0

    # 返回修订号
    @property
    def micro(self) -> int:
        return self.release[2] if len(self.release) >= 3 else 0
def _parse_letter_version(
    letter: str, number: str | bytes | SupportsInt
) -> tuple[str, int] | None:
    if letter:
        # 如果 letter 不为空，则执行以下逻辑
        # 如果 number 为 None，则假设在预发布版本中有一个隐含的 0
        if number is None:
            number = 0

        # 将 letter 规范化为小写形式
        letter = letter.lower()

        # 规范化某些单词为其他单词的替代拼写，将其归一化为首选拼写
        if letter == "alpha":
            letter = "a"
        elif letter == "beta":
            letter = "b"
        elif letter in ["c", "pre", "preview"]:
            letter = "rc"
        elif letter in ["rev", "r"]:
            letter = "post"

        # 返回 letter 和 number 的元组形式
        return letter, int(number)
    if not letter and number:
        # 如果 letter 为空且 number 不为空，则假设这是隐式的后发布版本语法（例如 1.0-1）
        letter = "post"

        # 返回 letter 和 number 的元组形式
        return letter, int(number)

    # 如果 letter 和 number 都为空，则返回 None
    return None


_local_version_separators = re.compile(r"[\._-]")


def _parse_local_version(local: str) -> LocalType | None:
    """
    Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
    """
    if local is not None:
        # 如果 local 不为空，则执行以下逻辑
        # 使用正则表达式分隔符将 local 字符串分割，并转换为小写形式或整数
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_separators.split(local)
        )
    # 如果 local 为空，则返回 None
    return None


def _cmpkey(
    epoch: int,
    release: tuple[int, ...],
    pre: tuple[str, int] | None,
    post: tuple[str, int] | None,
    dev: tuple[str, int] | None,
    local: tuple[SubLocalType] | None,
) -> CmpKey:
    # 当比较发布版本时，我们希望移除所有尾部的零来进行比较。
    # 将 release 列表逆转，并丢弃所有前导零，直到遇到非零值，
    # 然后将其重新逆转为正确顺序，并转换为元组形式，作为排序关键字使用。
    _release = tuple(
        reversed(list(itertools.dropwhile(lambda x: x == 0, reversed(release))))
    )

    # 我们需要“欺骗”排序算法，使得 1.0.dev0 在 1.0a0 之前。
    # 我们将滥用 pre 段，但仅当没有 pre 或 post 段时才这样做。
    # 如果有 pre 或 post 段，则正常的排序规则将正确处理此情况。
    if pre is None and post is None and dev is not None:
        _pre: PrePostDevType = NegativeInfinity
    # 没有预发布版本的版本（除非如上所述）应在具有预发布版本的版本之后排序。
    elif pre is None:
        _pre = Infinity
    else:
        _pre = pre

    # 没有后发布段的版本应在具有后发布段的版本之前排序。
    if post is None:
        _post: PrePostDevType = NegativeInfinity
    else:
        _post = post
    # 如果开发版本号（dev）为None，则将_dev设置为正无穷，这样这种版本就会在带有开发版本号的版本之后排序。
    if dev is None:
        _dev: PrePostDevType = Infinity

    else:
        # 否则，使用提供的开发版本号（dev）进行设置。
        _dev = dev

    if local is None:
        # 如果本地版本号（local）为None，则将_local设置为负无穷，这样这种版本就会在带有本地版本号的版本之前排序。
        _local: LocalType = NegativeInfinity
    else:
        # 否则，需要解析本地版本号（local）以实现PEP440规范中的排序规则。
        # - 字母数字段在数字段之前排序
        # - 字母数字段按字典顺序排序
        # - 数字段按数值大小排序
        # - 当前缀完全匹配时，较短的版本号在较长的版本号之前排序
        _local = tuple(
            (i, "") if isinstance(i, int) else (NegativeInfinity, i) for i in local
        )

    # 返回版本号中的各个部分：epoch, _release, _pre, _post, _dev, _local
    return epoch, _release, _pre, _post, _dev, _local
```