# `.\pytorch\test\package\test_mangling.py`

```py
# Owner(s): ["oncall: package/deploy"]

# 导入所需的模块和函数
from io import BytesIO

from torch.package import PackageExporter, PackageImporter
from torch.package._mangling import (
    demangle,
    get_mangle_prefix,
    is_mangled,
    PackageMangler,
)
from torch.testing._internal.common_utils import run_tests

try:
    # 尝试从当前目录导入 PackageTestCase 类
    from .common import PackageTestCase
except ImportError:
    # 如果失败，则从 common 模块导入 PackageTestCase 类
    from common import PackageTestCase

# 测试类，继承自 PackageTestCase
class TestMangling(PackageTestCase):
    
    def test_unique_manglers(self):
        """
        Each mangler instance should generate a unique mangled name for a given input.
        """
        # 创建两个 PackageMangler 实例 a 和 b
        a = PackageMangler()
        b = PackageMangler()
        # 断言两个不同的 mangler 对相同输入 "foo.bar" 进行 mangling 后结果不相同
        self.assertNotEqual(a.mangle("foo.bar"), b.mangle("foo.bar"))

    def test_mangler_is_consistent(self):
        """
        Mangling the same name twice should produce the same result.
        """
        # 创建 PackageMangler 实例 a
        a = PackageMangler()
        # 断言对相同输入 "abc.def" 进行两次 mangling 后结果相同
        self.assertEqual(a.mangle("abc.def"), a.mangle("abc.def"))

    def test_roundtrip_mangling(self):
        # 创建 PackageMangler 实例 a
        a = PackageMangler()
        # 断言对 "foo" 进行 mangling 然后再进行 demangling 后结果应该还是 "foo"
        self.assertEqual("foo", demangle(a.mangle("foo")))

    def test_is_mangled(self):
        # 创建两个 PackageMangler 实例 a 和 b
        a = PackageMangler()
        b = PackageMangler()
        # 断言对 mangling 后的结果应该返回 True
        self.assertTrue(is_mangled(a.mangle("foo.bar")))
        self.assertTrue(is_mangled(b.mangle("foo.bar")))

        # 断言对未 mangling 的字符串应该返回 False
        self.assertFalse(is_mangled("foo.bar"))
        # 断言对 demangling 后的结果应该返回 False
        self.assertFalse(is_mangled(demangle(a.mangle("foo.bar"))))

    def test_demangler_multiple_manglers(self):
        """
        PackageDemangler should be able to demangle name generated by any PackageMangler.
        """
        # 创建两个 PackageMangler 实例 a 和 b
        a = PackageMangler()
        b = PackageMangler()

        # 断言 demangling 后的结果应该与原始输入相同
        self.assertEqual("foo.bar", demangle(a.mangle("foo.bar")))
        self.assertEqual("bar.foo", demangle(b.mangle("bar.foo")))

    def test_mangle_empty_errors(self):
        # 创建 PackageMangler 实例 a
        a = PackageMangler()
        # 断言对空字符串进行 mangling 应该抛出 AssertionError 异常
        with self.assertRaises(AssertionError):
            a.mangle("")

    def test_demangle_base(self):
        """
        Demangling a mangle parent directly should currently return an empty string.
        """
        # 创建 PackageMangler 实例 a
        a = PackageMangler()
        # 对 "foo" 进行 mangling
        mangled = a.mangle("foo")
        # 取 mangling 后的字符串的父级部分（第一个点之前的部分）
        mangle_parent = mangled.partition(".")[0]
        # 断言对父级部分进行 demangling 应该返回空字符串
        self.assertEqual("", demangle(mangle_parent))

    def test_mangle_prefix(self):
        # 创建 PackageMangler 实例 a
        a = PackageMangler()
        # 对 "foo.bar" 进行 mangling
        mangled = a.mangle("foo.bar")
        # 获取 mangling 后的前缀部分
        mangle_prefix = get_mangle_prefix(mangled)
        # 断言获取的前缀加上原始输入应该等于 mangling 后的结果
        self.assertEqual(mangle_prefix + "." + "foo.bar", mangled)
    def test_unique_module_names(self):
        # 导入 package_a.subpackage 模块
        import package_a.subpackage

        # 创建 package_a.subpackage 下的对象
        obj = package_a.subpackage.PackageASubpackageObject()
        # 使用 package_a 下的对象 obj 创建一个新对象 obj2
        obj2 = package_a.PackageAObject(obj)
        # 创建一个字节流对象 f1
        f1 = BytesIO()
        # 使用 PackageExporter 类来操作 f1
        with PackageExporter(f1) as pe:
            # 对所有名称进行内部化处理
            pe.intern("**")
            # 将 obj2 保存为 obj.pkl 文件
            pe.save_pickle("obj", "obj.pkl", obj2)
        # 将 f1 的指针移到开头
        f1.seek(0)
        # 使用 PackageImporter 类来导入 f1
        importer1 = PackageImporter(f1)
        # 从 f1 中加载 obj.pkl 文件内容到 loaded1
        loaded1 = importer1.load_pickle("obj", "obj.pkl")
        # 将 f1 的指针移到开头
        f1.seek(0)
        # 再次使用 PackageImporter 类来导入 f1
        importer2 = PackageImporter(f1)
        # 从 f1 中再次加载 obj.pkl 文件内容到 loaded2
        loaded2 = importer2.load_pickle("obj", "obj.pkl")

        # 断言：加载的包中的模块名不应该与原始模块名相同
        # 更多信息请参考 mangling.md
        self.assertNotEqual(type(obj2).__module__, type(loaded1).__module__)
        self.assertNotEqual(type(loaded1).__module__, type(loaded2).__module__)

    def test_package_mangler(self):
        # 创建 PackageMangler 类的实例 a 和 b
        a = PackageMangler()
        b = PackageMangler()
        # 对字符串 "foo.bar" 进行混淆处理，并赋给 a_mangled
        a_mangled = a.mangle("foo.bar")
        # 由于 a 对此字符串进行了混淆处理，应该能正确地解混淆回 "foo.bar"
        self.assertEqual(a.demangle(a_mangled), "foo.bar")
        # 由于 b 没有对此字符串进行混淆处理，解混淆应该不改变它
        self.assertEqual(b.demangle(a_mangled), a_mangled)
# 如果当前脚本作为主程序执行（而不是被导入到其他脚本中执行），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```