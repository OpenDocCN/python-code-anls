# `.\numpy\numpy\distutils\tests\test_ccompiler_opt_conf.py`

```
import unittest                         # 导入单元测试框架 unittest
from os import sys, path                # 导入 os 模块中的 sys 和 path 函数

is_standalone = __name__ == '__main__' and __package__ is None  # 检查是否独立运行
if is_standalone:
    sys.path.append(path.abspath(path.join(path.dirname(__file__), "..")))  # 将上级目录添加到系统路径中
    from ccompiler_opt import CCompilerOpt  # 如果是独立运行，则从上级目录导入 ccompiler_opt 模块
else:
    from numpy.distutils.ccompiler_opt import CCompilerOpt  # 否则从 numpy.distutils.ccompiler_opt 导入 CCompilerOpt 类

arch_compilers = dict(                   # 定义一个字典，存储不同架构对应的编译器列表
    x86 = ("gcc", "clang", "icc", "iccw", "msvc"),
    x64 = ("gcc", "clang", "icc", "iccw", "msvc"),
    ppc64 = ("gcc", "clang"),
    ppc64le = ("gcc", "clang"),
    armhf = ("gcc", "clang"),
    aarch64 = ("gcc", "clang"),
    narch = ("gcc",)
)

class FakeCCompilerOpt(CCompilerOpt):   # 定义一个名为 FakeCCompilerOpt 的类，继承自 CCompilerOpt 类
    fake_info = ("arch", "compiler", "extra_args")  # 定义类属性 fake_info，包含三个元素

    def __init__(self, *args, **kwargs):  # 类的初始化方法
        CCompilerOpt.__init__(self, None, **kwargs)  # 调用父类 CCompilerOpt 的初始化方法

    def dist_compile(self, sources, flags, **kwargs):  # 定义一个方法 dist_compile
        return sources                   # 返回传入的 sources 参数

    def dist_info(self):                 # 定义一个方法 dist_info
        return FakeCCompilerOpt.fake_info  # 返回类属性 fake_info

    @staticmethod
    def dist_log(*args, stderr=False):   # 定义一个静态方法 dist_log，接受任意参数但不做任何操作
        pass                             # 空操作

class _TestConfFeatures(FakeCCompilerOpt):  # 定义一个名为 _TestConfFeatures 的类，继承自 FakeCCompilerOpt 类
    """A hook to check the sanity of configured features
-   before it called by the abstract class '_Feature'
    """

    def conf_features_partial(self):     # 定义一个方法 conf_features_partial
        conf_all = self.conf_features   # 获取实例的 conf_features 属性
        for feature_name, feature in conf_all.items():  # 遍历 conf_all 字典的键值对
            self.test_feature(           # 调用实例方法 test_feature 进行测试
                "attribute conf_features",
                conf_all, feature_name, feature
            )

        conf_partial = FakeCCompilerOpt.conf_features_partial(self)  # 调用父类的 conf_features_partial 方法
        for feature_name, feature in conf_partial.items():  # 遍历 conf_partial 字典的键值对
            self.test_feature(           # 再次调用实例方法 test_feature 进行测试
                "conf_features_partial()",
                conf_partial, feature_name, feature
            )
        return conf_partial             # 返回 conf_partial 字典

    def test_feature(self, log, search_in, feature_name, feature_dict):  # 定义一个方法 test_feature
        error_msg = (                   # 定义一个错误信息字符串
            "during validate '{}' within feature '{}', "
            "march '{}' and compiler '{}'\n>> "
        ).format(log, feature_name, self.cc_march, self.cc_name)

        if not feature_name.isupper():  # 如果 feature_name 不全为大写字母，则抛出断言错误
            raise AssertionError(error_msg + "feature name must be in uppercase")

        for option, val in feature_dict.items():  # 遍历 feature_dict 字典的键值对
            self.test_option_types(error_msg, option, val)  # 调用实例方法 test_option_types 进行类型测试
            self.test_duplicates(error_msg, option, val)  # 调用实例方法 test_duplicates 进行重复项测试

        self.test_implies(error_msg, search_in, feature_name, feature_dict)  # 调用实例方法 test_implies 进行条件逻辑测试
        self.test_group(error_msg, search_in, feature_name, feature_dict)    # 调用实例方法 test_group 进行组逻辑测试
        self.test_extra_checks(error_msg, search_in, feature_name, feature_dict)  # 调用实例方法 test_extra_checks 进行额外检查
    # 检查选项的类型是否符合预期，并抛出断言错误信息
    def test_option_types(self, error_msg, option, val):
        # 遍历不同类型和其可用选项的元组
        for tp, available in (
            # 字符串或列表类型对应的选项
            ((str, list), (
                "implies", "headers", "flags", "group", "detect", "extra_checks"
            )),
            # 字符串类型对应的选项
            ((str,),  ("disable",)),
            # 整数类型对应的选项
            ((int,),  ("interest",)),
            # 布尔类型或空值类型对应的选项
            ((bool,), ("implies_detect",)),
            ((bool, type(None)), ("autovec",)),
        ) :
            # 检查当前选项是否在可用选项中
            found_it = option in available
            if not found_it:
                continue
            # 检查值的类型是否在预期的类型范围内
            if not isinstance(val, tp):
                # 准备错误信息，指明预期的类型和当前值的实际类型不匹配
                error_tp = [t.__name__ for t in (*tp,)]
                error_tp = ' or '.join(error_tp)
                raise AssertionError(error_msg +
                    "expected '%s' type for option '%s' not '%s'" % (
                     error_tp, option, type(val).__name__
                ))
            break

        # 如果未找到匹配的选项，则抛出错误信息，说明选项名称无效
        if not found_it:
            raise AssertionError(error_msg + "invalid option name '%s'" % option)

    # 检查选项值中是否存在重复，并抛出断言错误信息
    def test_duplicates(self, error_msg, option, val):
        # 如果选项不在需要检查重复的选项列表中，则直接返回
        if option not in (
            "implies", "headers", "flags", "group", "detect", "extra_checks"
        ) : return

        # 如果值是字符串，则转换成列表
        if isinstance(val, str):
            val = val.split()

        # 检查列表中是否存在重复值，如果存在则抛出错误信息
        if len(val) != len(set(val)):
            raise AssertionError(error_msg + "duplicated values in option '%s'" % option)

    # 检查特性是否有"implies"选项，并进行相应检查
    def test_implies(self, error_msg, search_in, feature_name, feature_dict):
        # 如果特性已经被禁用，则直接返回
        if feature_dict.get("disabled") is not None:
            return
        # 获取特性的"implies"选项，如果为空则直接返回
        implies = feature_dict.get("implies", "")
        if not implies:
            return
        # 如果"implies"是字符串，则转换成列表
        if isinstance(implies, str):
            implies = implies.split()

        # 检查特性是否暗示了自身，如果是则抛出错误信息
        if feature_name in implies:
            raise AssertionError(error_msg + "feature implies itself")

        # 遍历每个暗示的特性，检查其是否存在于搜索范围中，如果不存在则抛出错误信息
        for impl in implies:
            impl_dict = search_in.get(impl)
            if impl_dict is not None:
                # 如果该暗示特性已被禁用，则抛出错误信息
                if "disable" in impl_dict:
                    raise AssertionError(error_msg + "implies disabled feature '%s'" % impl)
                continue
            raise AssertionError(error_msg + "implies non-exist feature '%s'" % impl)

    # 检查特性是否有"group"选项，并进行相应检查
    def test_group(self, error_msg, search_in, feature_name, feature_dict):
        # 如果特性已经被禁用，则直接返回
        if feature_dict.get("disabled") is not None:
            return
        # 获取特性的"group"选项，如果为空则直接返回
        group = feature_dict.get("group", "")
        if not group:
            return
        # 如果"group"是字符串，则转换成列表
        if isinstance(group, str):
            group = group.split()

        # 遍历"group"中的每个特性，检查其是否已存在于搜索范围中，如果已存在则抛出错误信息
        for f in group:
            impl_dict = search_in.get(f)
            if not impl_dict or "disable" in impl_dict:
                continue
            raise AssertionError(error_msg +
                "in option 'group', '%s' already exists as a feature name" % f
            )
    # 测试额外检查项的有效性，如果特性字典中包含 "disabled" 键，则直接返回，不进行额外检查
    def test_extra_checks(self, error_msg, search_in, feature_name, feature_dict):
        # 检查特性字典中是否包含 "disabled" 键，如果有则退出函数
        if feature_dict.get("disabled") is not None:
            return
        
        # 获取特性字典中的额外检查项，如果没有额外检查项则退出函数
        extra_checks = feature_dict.get("extra_checks", "")
        if not extra_checks:
            return
        
        # 如果额外检查项是字符串类型，则按空格分割为列表
        if isinstance(extra_checks, str):
            extra_checks = extra_checks.split()

        # 遍历额外检查项列表
        for f in extra_checks:
            # 在搜索对象中查找额外检查项对应的实现字典
            impl_dict = search_in.get(f)
            # 如果实现字典不存在或者包含 "disable" 键，则继续下一次循环
            if not impl_dict or "disable" in impl_dict:
                continue
            # 抛出断言错误，指明额外检查项已存在作为特性名称的测试用例
            raise AssertionError(error_msg +
                "in option 'extra_checks', extra test case '%s' already exists as a feature name" % f
            )
# 定义一个名为 TestConfFeatures 的测试类，继承自 unittest.TestCase
class TestConfFeatures(unittest.TestCase):
    
    # 初始化方法，用于设置测试方法的名称，默认为 "runTest"
    def __init__(self, methodName="runTest"):
        # 调用父类的初始化方法
        unittest.TestCase.__init__(self, methodName)
        # 调用 _setup 方法进行额外的设置
        self._setup()

    # 辅助方法，用于设置 FakeCCompilerOpt 的 conf_nocache 属性为 True
    def _setup(self):
        FakeCCompilerOpt.conf_nocache = True

    # 测试方法，测试特性
    def test_features(self):
        # 遍历 arch_compilers 字典，arch 是键，compilers 是值（列表）
        for arch, compilers in arch_compilers.items():
            # 对于每个 arch，遍历其对应的 compilers 列表
            for cc in compilers:
                # 设置 FakeCCompilerOpt 的 fake_info 属性为 (arch, cc, "")
                FakeCCompilerOpt.fake_info = (arch, cc, "")
                # 调用 _TestConfFeatures 类的实例化，实际上是调用 _TestConfFeatures() 函数
                _TestConfFeatures()

# 如果 is_standalone 为真（即为 True），则执行 unittest.main()，启动测试
if is_standalone:
    unittest.main()
```