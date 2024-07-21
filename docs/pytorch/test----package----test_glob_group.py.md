# `.\pytorch\test\package\test_glob_group.py`

```py
# Owner(s): ["oncall: package/deploy"]

# 引入类型提示模块
from typing import Iterable

# 引入 torch.package 模块中的 GlobGroup 类和 torch.testing._internal.common_utils 模块中的 run_tests 函数
from torch.package import GlobGroup
from torch.testing._internal.common_utils import run_tests

try:
    # 尝试从当前目录的 common 模块中导入 PackageTestCase 类
    from .common import PackageTestCase
except ImportError:
    # 如果导入失败，支持直接运行该文件的情况，从 common 模块中导入 PackageTestCase 类
    from common import PackageTestCase

# 定义一个测试类 TestGlobGroup，继承自 PackageTestCase 类
class TestGlobGroup(PackageTestCase):

    # 定义一个断言方法 assertMatchesGlob，用于验证 GlobGroup 对象匹配给定候选项列表
    def assertMatchesGlob(self, glob: GlobGroup, candidates: Iterable[str]):
        for candidate in candidates:
            self.assertTrue(glob.matches(candidate))

    # 定义一个断言方法 assertNotMatchesGlob，用于验证 GlobGroup 对象不匹配给定候选项列表
    def assertNotMatchesGlob(self, glob: GlobGroup, candidates: Iterable[str]):
        for candidate in candidates:
            self.assertFalse(glob.matches(candidate))

    # 定义一个测试方法 test_one_star，测试单个星号通配符的 GlobGroup 匹配行为
    def test_one_star(self):
        glob_group = GlobGroup("torch.*")
        self.assertMatchesGlob(glob_group, ["torch.foo", "torch.bar"])
        self.assertNotMatchesGlob(glob_group, ["tor.foo", "torch.foo.bar", "torch"])

    # 定义一个测试方法 test_one_star_middle，测试中间位置单个星号通配符的 GlobGroup 匹配行为
    def test_one_star_middle(self):
        glob_group = GlobGroup("foo.*.bar")
        self.assertMatchesGlob(glob_group, ["foo.q.bar", "foo.foo.bar"])
        self.assertNotMatchesGlob(
            glob_group,
            [
                "foo.bar",
                "foo.foo",
                "outer.foo.inner.bar",
                "foo.q.bar.more",
                "foo.one.two.bar",
            ],
        )

    # 定义一个测试方法 test_one_star_partial，测试部分匹配的单个星号通配符的 GlobGroup 匹配行为
    def test_one_star_partial(self):
        glob_group = GlobGroup("fo*.bar")
        self.assertMatchesGlob(glob_group, ["fo.bar", "foo.bar", "foobar.bar"])
        self.assertNotMatchesGlob(glob_group, ["oij.bar", "f.bar", "foo"])

    # 定义一个测试方法 test_one_star_multiple_in_component，测试组件中多个单个星号通配符的 GlobGroup 匹配行为
    def test_one_star_multiple_in_component(self):
        glob_group = GlobGroup("foo/a*.htm*", separator="/")
        self.assertMatchesGlob(glob_group, ["foo/a.html", "foo/a.htm", "foo/abc.html"])

    # 定义一个测试方法 test_one_star_partial_extension，测试部分扩展名的单个星号通配符的 GlobGroup 匹配行为
    def test_one_star_partial_extension(self):
        glob_group = GlobGroup("foo/*.txt", separator="/")
        self.assertMatchesGlob(
            glob_group, ["foo/hello.txt", "foo/goodbye.txt", "foo/.txt"]
        )
        self.assertNotMatchesGlob(
            glob_group, ["foo/bar/hello.txt", "bar/foo/hello.txt"]
        )

    # 定义一个测试方法 test_two_star，测试双星号通配符的 GlobGroup 匹配行为
    def test_two_star(self):
        glob_group = GlobGroup("torch.**")
        self.assertMatchesGlob(
            glob_group, ["torch.foo", "torch.bar", "torch.foo.bar", "torch"]
        )
        self.assertNotMatchesGlob(glob_group, ["what.torch", "torchvision"])

    # 定义一个测试方法 test_two_star_end，测试结尾位置双星号通配符的 GlobGroup 匹配行为
    def test_two_star_end(self):
        glob_group = GlobGroup("**.torch")
        self.assertMatchesGlob(glob_group, ["torch", "bar.torch"])
        self.assertNotMatchesGlob(glob_group, ["visiontorch"])

    # 定义一个测试方法 test_two_star_middle，测试中间位置双星号通配符的 GlobGroup 匹配行为
    def test_two_star_middle(self):
        glob_group = GlobGroup("foo.**.baz")
        self.assertMatchesGlob(
            glob_group, ["foo.baz", "foo.bar.baz", "foo.bar1.bar2.baz"]
        )
        self.assertNotMatchesGlob(glob_group, ["foobaz", "foo.bar.baz.z"])
    # 定义一个测试方法，用于测试 GlobGroup 类处理 '**/bar/**/*.txt' 格式的全局匹配情况
    def test_two_star_multiple(self):
        # 创建 GlobGroup 实例，使用 '/' 作为路径分隔符
        glob_group = GlobGroup("**/bar/**/*.txt", separator="/")
        # 断言匹配 GlobGroup 的文件列表
        self.assertMatchesGlob(
            glob_group, ["bar/baz.txt", "a/bar/b.txt", "bar/foo/c.txt"]
        )
        # 断言不匹配 GlobGroup 的文件列表
        self.assertNotMatchesGlob(glob_group, ["baz.txt", "a/b.txt"])

    # 定义一个测试方法，用于测试 GlobGroup 类处理 '**' 格式的全局匹配情况
    def test_raw_two_star(self):
        # 创建 GlobGroup 实例，匹配所有路径和文件
        glob_group = GlobGroup("**")
        # 断言匹配 GlobGroup 的文件列表
        self.assertMatchesGlob(glob_group, ["bar", "foo.bar", "ab.c.d.e"])
        # 断言不匹配 GlobGroup 的文件列表
        self.assertNotMatchesGlob(glob_group, [""])

    # 定义一个测试方法，测试 GlobGroup 类处理无效的 glob 格式情况
    def test_invalid_raw(self):
        # 使用 GlobGroup 初始化一个无效 glob 格式字符串时应抛出 ValueError 异常
        with self.assertRaises(ValueError):
            GlobGroup("a.**b")

    # 定义一个测试方法，测试 GlobGroup 类处理带排除项的 glob 格式
    def test_exclude(self):
        # 创建 GlobGroup 实例，匹配以 'torch.' 开头但不匹配以 'torch.**.foo' 结尾的路径
        glob_group = GlobGroup("torch.**", exclude=["torch.**.foo"])
        # 断言匹配 GlobGroup 的文件列表
        self.assertMatchesGlob(
            glob_group,
            ["torch", "torch.bar", "torch.barfoo"],
        )
        # 断言不匹配 GlobGroup 的文件列表
        self.assertNotMatchesGlob(
            glob_group,
            ["torch.foo", "torch.some.foo"],
        )

    # 定义一个测试方法，测试 GlobGroup 类处理全局匹配但排除指定路径的情况
    def test_exclude_from_all(self):
        # 创建 GlobGroup 实例，匹配所有路径但不匹配以 'foo.' 或 'bar.' 开头的路径
        glob_group = GlobGroup("**", exclude=["foo.**", "bar.**"])
        # 断言匹配 GlobGroup 的文件列表
        self.assertMatchesGlob(glob_group, ["a", "hello", "anything.really"])
        # 断言不匹配 GlobGroup 的文件列表
        self.assertNotMatchesGlob(glob_group, ["foo.bar", "foo.bar.baz"])

    # 定义一个测试方法，测试 GlobGroup 类处理列表形式的包含和排除情况
    def test_list_include_exclude(self):
        # 创建 GlobGroup 实例，匹配 'foo' 和以 'bar.' 开头但不匹配 'bar.baz' 或 'bar.qux' 的路径
        glob_group = GlobGroup(["foo", "bar.**"], exclude=["bar.baz", "bar.qux"])
        # 断言匹配 GlobGroup 的文件列表
        self.assertMatchesGlob(glob_group, ["foo", "bar.other", "bar.bazother"])
        # 断言不匹配 GlobGroup 的文件列表
        self.assertNotMatchesGlob(glob_group, ["bar.baz", "bar.qux"])
# 如果当前脚本被直接执行（而非被导入为模块），则运行 run_tests 函数
if __name__ == "__main__":
    run_tests()
```