# `.\pytorch\tools\testing\test_run.py`

```py
# 引入 future 模块的 annotations 功能，支持在类定义中使用注解
from __future__ import annotations

# 从 copy 模块中导入 copy 函数
from copy import copy
# 从 functools 模块中导入 total_ordering 装饰器
from functools import total_ordering
# 导入 typing 模块中的 Any 和 Iterable 类型
from typing import Any, Iterable

# 定义 TestRun 类，表示一组应在单个 pytest 调用中一起运行的测试
class TestRun:
    """
    TestRun defines the set of tests that should be run together in a single pytest invocation.
    It'll either be a whole test file or a subset of a test file.

    This class assumes that we won't always know the full set of TestClasses in a test file.
    So it's designed to include or exclude explicitly requested TestClasses, while having accepting
    that there will be an ambiguous set of "unknown" test classes that are not expliclty called out.
    Those manifest as tests that haven't been explicitly excluded.
    """

    # 测试文件名
    test_file: str
    # 要从此测试运行中排除的测试集合（不可变集合）
    _excluded: frozenset[str]
    # 如果非空，则仅在此测试运行中运行这些测试（不可变集合）
    _included: frozenset[str]

    def __init__(
        self,
        name: str,
        excluded: Iterable[str] | None = None,
        included: Iterable[str] | None = None,
    ) -> None:
        # 如果同时指定了 included 和 excluded，则引发 ValueError
        if excluded and included:
            raise ValueError("Can't specify both included and excluded")

        # 初始化 included 和 excluded 的集合
        ins = set(included or [])
        exs = set(excluded or [])

        # 如果 name 中包含 "::"，则将其解析为测试文件名和测试类名
        if "::" in name:
            # 确保在指定文件名中的测试类时，不能同时指定 included 或 excluded
            assert (
                not included and not excluded
            ), "Can't specify included or excluded tests when specifying a test class in the file name"
            self.test_file, test_class = name.split("::")
            ins.add(test_class)
        else:
            self.test_file = name

        # 将集合转换为不可变集合并赋值给 _excluded 和 _included
        self._excluded = frozenset(exs)
        self._included = frozenset(ins)

    @staticmethod
    def empty() -> TestRun:
        # 返回一个表示空运行的 TestRun 实例
        return TestRun("")

    def is_empty(self) -> bool:
        # 如果没有指定测试文件名，则表示是一个空运行
        return not self.test_file

    def is_full_file(self) -> bool:
        # 如果 _included 和 _excluded 都为空，则表示要运行整个测试文件
        return not self._included and not self._excluded

    def included(self) -> frozenset[str]:
        # 返回包含要运行的测试集合的不可变集合 _included
        return self._included

    def excluded(self) -> frozenset[str]:
        # 返回包含要从运行中排除的测试集合的不可变集合 _excluded
        return self._excluded

    def get_pytest_filter(self) -> str:
        # 根据 _included 和 _excluded 生成 pytest 的过滤器表达式
        if self._included:
            return " or ".join(sorted(self._included))
        elif self._excluded:
            return f"not ({' or '.join(sorted(self._excluded))})"
        else:
            return ""
    # 判断当前对象是否包含给定的测试运行对象
    def contains(self, test: TestRun) -> bool:
        # 检查测试文件路径是否相同，如果不同直接返回 False
        if self.test_file != test.test_file:
            return False

        # 如果当前对象包含所有测试（即没有排除任何测试），则返回 True
        if self.is_full_file():
            return True  # self contains all tests

        # 如果测试运行对象包含所有测试（即没有排除任何测试），但当前对象不包含所有测试，则返回 False
        if test.is_full_file():
            return False  # test contains all tests, but self doesn't

        # 如果测试运行对象有排除列表，检查当前对象的排除列表是否包含测试对象的排除列表
        # 如果是，则当前对象包含测试对象，返回 True
        if test._excluded:
            return test._excluded.issubset(self._excluded)

        # 如果测试运行对象有包含列表，检查当前对象的包含列表是否包含测试对象的包含列表
        # 如果是，则当前对象包含测试对象，返回 True
        if self._included:
            return test._included.issubset(self._included)

        # 如果执行到这里，说明测试对象包含一些内容，而当前对象排除一些内容
        # 检查当前对象的排除列表是否与测试对象的包含列表有交集
        # 如果没有交集，说明当前对象不排除测试对象包含的所有内容，返回 True
        return not self._excluded.intersection(test._included)

    # 返回当前对象的浅拷贝，包括测试文件路径和排除与包含列表
    def __copy__(self) -> TestRun:
        return TestRun(self.test_file, excluded=self._excluded, included=self._included)

    # 定义对象的布尔值，如果当前对象不为空，则返回 True
    def __bool__(self) -> bool:
        return not self.is_empty()

    # 返回对象的字符串表示形式，如果对象为空则返回 "Empty"，否则返回测试文件路径和 pytest 过滤条件
    def __str__(self) -> str:
        if self.is_empty():
            return "Empty"

        pytest_filter = self.get_pytest_filter()
        if pytest_filter:
            return self.test_file + ", " + pytest_filter
        return self.test_file

    # 返回对象的详细字符串表示形式，包括测试文件路径以及包含与排除列表（如果有）
    def __repr__(self) -> str:
        r: str = f"RunTest({self.test_file}"
        r += f", included: {self._included}" if self._included else ""
        r += f", excluded: {self._excluded}" if self._excluded else ""
        r += ")"
        return r

    # 检查两个 TestRun 对象是否相等，包括测试文件路径以及包含与排除列表的比较
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TestRun):
            return False

        ret = self.test_file == other.test_file
        ret = ret and self._included == other._included
        ret = ret and self._excluded == other._excluded
        return ret

    # 返回对象的哈希值，用于集合等需要哈希值的数据结构
    def __hash__(self) -> int:
        return hash((self.test_file, self._included, self._excluded))
    # 定义一个特殊方法，实现两个测试运行的 OR/Union 操作
    def __or__(self, other: TestRun) -> TestRun:
        """
        To OR/Union test runs means to run all the tests that either of the two runs specify.
        """

        # 检查是否有任何一个文件是空的
        if self.is_empty():
            # 如果当前对象是空的，则返回另一个对象
            return other
        if other.is_empty():
            # 如果另一个对象是空的，则返回当前对象的副本
            return copy(self)

        # 确保两个对象操作的是同一个测试文件
        assert (
            self.test_file == other.test_file
        ), f"Can't exclude {other} from {self} because they're not the same test file"

        # 4种可能的情况:

        # 1. 任何一个对象都是完整文件，那么合并后就是整个文件
        if self.is_full_file() or other.is_full_file():
            # 合并后的对象是整个测试文件
            return TestRun(self.test_file)

        # 2. 两个对象都只运行 _included 集合中的测试，合并后即为两个集合的并集
        if self._included and other._included:
            return TestRun(
                self.test_file, included=self._included.union(other._included)
            )

        # 3. 两个对象都只排除 _excluded 集合中的测试，合并后即为两个集合的交集
        if self._excluded and other._excluded:
            return TestRun(
                self.test_file, excluded=self._excluded.intersection(other._excluded)
            )

        # 4. 一个对象包含测试，另一个对象排除测试，合并后即为 _excluded 减去 _included 的结果
        included = self._included | other._included
        excluded = self._excluded | other._excluded
        return TestRun(self.test_file, excluded=excluded - included)

    # 定义一个特殊方法，实现两个测试运行的减法操作
    def __sub__(self, other: TestRun) -> TestRun:
        """
        To subtract test runs means to run all the tests in the first run except for what the second run specifies.
        """

        # 检查是否有任何一个文件是空的
        if self.is_empty():
            # 如果当前对象是空的，则返回一个空的 TestRun 对象
            return TestRun.empty()
        if other.is_empty():
            # 如果另一个对象是空的，则返回当前对象的副本
            return copy(self)

        # 如果试图从当前测试运行中减去不包含的测试
        if self.test_file != other.test_file:
            # 返回当前对象的副本
            return copy(self)

        # 如果试图减去的是整个文件
        if other.is_full_file():
            # 返回一个空的 TestRun 对象
            return TestRun.empty()

        def return_inclusions_or_empty(inclusions: frozenset[str]) -> TestRun:
            if inclusions:
                return TestRun(self.test_file, included=inclusions)
            return TestRun.empty()

        if other._included:
            if self._included:
                # 返回当前对象中减去另一个对象 _included 后的结果
                return return_inclusions_or_empty(self._included - other._included)
            else:
                # 返回当前对象中排除另一个对象 _included 后的结果
                return TestRun(
                    self.test_file, excluded=self._excluded | other._included
                )
        else:
            if self._included:
                # 返回当前对象中减去另一个对象 _excluded 后的结果
                return return_inclusions_or_empty(self._included & other._excluded)
            else:
                # 返回另一个对象中减去当前对象 _excluded 后的结果
                return return_inclusions_or_empty(other._excluded - self._excluded)
    # 定义与运算符重载方法，返回与操作后的测试运行对象
    def __and__(self, other: TestRun) -> TestRun:
        # 如果当前测试运行对象的测试文件与另一个对象的测试文件不同
        if self.test_file != other.test_file:
            # 返回一个空的测试运行对象
            return TestRun.empty()

        # 返回两个对象的并集减去各自的差集的结果
        return (self | other) - (self - other) - (other - self)

    # 将对象转换为 JSON 格式的字典
    def to_json(self) -> dict[str, Any]:
        r: dict[str, Any] = {
            "test_file": self.test_file,
        }
        # 如果包含了被包含的测试集合
        if self._included:
            r["included"] = list(self._included)
        # 如果包含了被排除的测试集合
        if self._excluded:
            r["excluded"] = list(self._excluded)
        # 返回 JSON 字典
        return r

    # 从 JSON 格式的字典中构造测试运行对象
    @staticmethod
    def from_json(json: dict[str, Any]) -> TestRun:
        # 从 JSON 字典中提取测试文件名、包含的测试和排除的测试列表，构造测试运行对象并返回
        return TestRun(
            json["test_file"],
            included=json.get("included", []),
            excluded=json.get("excluded", []),
        )
# 使用 total_ordering 装饰器为类提供比较运算符的默认实现（__eq__ 和 __lt__）
@total_ordering
class ShardedTest:
    # 类型注解，test 属性为 TestRun 类型的对象
    test: TestRun
    # 整数类型属性，表示当前分片的编号
    shard: int
    # 整数类型属性，表示总共的分片数
    num_shards: int
    # 可选的浮点数类型属性，表示测试运行的时间（单位为秒）
    time: float | None  # In seconds

    # 初始化方法，接受测试对象（TestRun 类型或字符串）、分片编号、总分片数和时间参数（默认为 None）
    def __init__(
        self,
        test: TestRun | str,
        shard: int,
        num_shards: int,
        time: float | None = None,
    ) -> None:
        # 如果传入的测试对象是字符串，则创建 TestRun 对象
        if isinstance(test, str):
            test = TestRun(test)
        # 设置对象的 test、shard、num_shards 和 time 属性
        self.test = test
        self.shard = shard
        self.num_shards = num_shards
        self.time = time

    # 使用 @property 装饰器，返回测试对象的测试文件名
    @property
    def name(self) -> str:
        return self.test.test_file

    # 实现相等比较运算符，比较当前对象与另一个对象是否相等
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShardedTest):
            return False
        return (
            self.test == other.test
            and self.shard == other.shard
            and self.num_shards == other.num_shards
            and self.time == other.time
        )

    # 实现小于比较运算符，定义对象的排序规则
    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ShardedTest):
            raise NotImplementedError

        # 按照测试文件名、分片编号、总分片数、时间的顺序比较对象
        if self.name != other.name:
            return self.name < other.name
        if self.shard != other.shard:
            return self.shard < other.shard
        if self.num_shards != other.num_shards:
            return self.num_shards < other.num_shards

        # None 值被认为是最小的
        if self.time is None:
            return True
        if other.time is None:
            return False
        return self.time < other.time

    # 返回对象的字符串表示形式，包括测试对象、分片信息和可选的时间信息
    def __repr__(self) -> str:
        ret = f"{self.test} {self.shard}/{self.num_shards}"
        if self.time:
            ret += f" ({self.time}s)"
        return ret

    # 返回对象的字符串表示形式，仅包括测试对象和分片信息
    def __str__(self) -> str:
        return f"{self.test} {self.shard}/{self.num_shards}"

    # 返回对象的时间信息，如果时间为 None 则返回默认值
    def get_time(self, default: float = 0) -> float:
        return self.time if self.time is not None else default

    # 返回用于运行 pytest 的参数列表，如果有测试过滤器则返回 ['-k', 过滤器]
    def get_pytest_args(self) -> list[str]:
        filter = self.test.get_pytest_filter()
        if filter:
            return ["-k", self.test.get_pytest_filter()]
        return []
```