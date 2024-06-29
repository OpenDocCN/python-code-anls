# `.\numpy\numpy\_core\tests\test_cpu_features.py`

```
# 导入必要的模块：sys, platform, re, pytest
# 从 numpy._core._multiarray_umath 中导入特定的 CPU 相关功能
import sys, platform, re, pytest
from numpy._core._multiarray_umath import (
    __cpu_features__,
    __cpu_baseline__,
    __cpu_dispatch__,
)
# 导入 numpy 库并重命名为 np
import numpy as np
# 导入 subprocess 模块用于执行外部命令
import subprocess
# 导入 pathlib 模块用于处理文件路径
import pathlib
# 导入 os 模块，提供了一种与操作系统进行交互的方法
import os
# 再次导入 re 模块，可能是为了与之前的 re 模块有所区分

# 定义一个函数 assert_features_equal，用于比较实际特性和期望特性是否相等
def assert_features_equal(actual, desired, fname):
    __tracebackhide__ = True  # 在 pytest 中隐藏回溯信息
    # 将 actual 和 desired 转换为字符串形式
    actual, desired = str(actual), str(desired)
    # 如果 actual 等于 desired，直接返回，说明相等
    if actual == desired:
        return
    # 将 __cpu_features__ 转换为字符串，并去除单引号
    detected = str(__cpu_features__).replace("'", "")
    # 尝试打开 /proc/cpuinfo 文件，并读取其内容，最多读取 2048 字节
    try:
        with open("/proc/cpuinfo") as fd:
            cpuinfo = fd.read(2048)
    except Exception as err:
        cpuinfo = str(err)

    # 尝试使用 subprocess 模块执行 '/bin/true' 命令，获取其输出
    try:
        import subprocess
        auxv = subprocess.check_output(['/bin/true'], env=dict(LD_SHOW_AUXV="1"))
        auxv = auxv.decode()
    except Exception as err:
        auxv = str(err)

    # 导入 textwrap 模块，用于格式化输出
    import textwrap
    # 构建错误报告，包含检测到的特性、cpuinfo 内容和 auxv 输出
    error_report = textwrap.indent(
"""
###########################################
### Extra debugging information
###########################################
-------------------------------------------
--- NumPy Detections
-------------------------------------------
%s
-------------------------------------------
--- SYS / CPUINFO
-------------------------------------------
%s....
-------------------------------------------
--- SYS / AUXV
-------------------------------------------
%s
""" % (detected, cpuinfo, auxv), prefix='\r')

    # 抛出 AssertionError 异常，显示详细的错误信息和报告
    raise AssertionError((
        "Failure Detection\n"
        " NAME: '%s'\n"
        " ACTUAL: %s\n"
        " DESIRED: %s\n"
        "%s"
    ) % (fname, actual, desired, error_report))

# 定义一个函数 _text_to_list，用于将文本转换为列表形式
def _text_to_list(txt):
    # 去除文本中的 '['、']'、'\n' 和单引号，并按 ', ' 分割为列表
    out = txt.strip("][\n").replace("'", "").split(', ')
    # 如果列表的第一个元素为空字符串，则返回 None，否则返回列表
    return None if out[0] == "" else out

# 定义一个抽象类 AbstractTest
class AbstractTest:
    # 类变量初始化
    features = []
    features_groups = {}
    features_map = {}
    features_flags = set()

    # 定义一个方法 load_flags，作为钩子函数，不做任何操作
    def load_flags(self):
        # a hook
        pass
    
    # 定义一个测试方法 test_features，用于测试 CPU 特性
    def test_features(self):
        # 调用 load_flags 方法加载特性标志
        self.load_flags()
        # 遍历特性分组字典 features_groups
        for gname, features in self.features_groups.items():
            # 对于每个特性分组，获取特性是否存在的列表
            test_features = [self.cpu_have(f) for f in features]
            # 检查 __cpu_features__ 中对应分组的特性是否与测试结果相等
            assert_features_equal(__cpu_features__.get(gname), all(test_features), gname)

        # 遍历单独的特性列表 features
        for feature_name in self.features:
            # 检查 CPU 是否支持该特性
            cpu_have = self.cpu_have(feature_name)
            # 获取 NumPy 是否检测到该特性
            npy_have = __cpu_features__.get(feature_name)
            # 检查 NumPy 检测结果与实际 CPU 支持结果是否相等
            assert_features_equal(npy_have, cpu_have, feature_name)

    # 定义一个方法 cpu_have，用于检查 CPU 是否支持特定特性
    def cpu_have(self, feature_name):
        # 获取特性名称在 features_map 中的映射名称
        map_names = self.features_map.get(feature_name, feature_name)
        # 如果映射名称是字符串，则直接检查是否在 features_flags 集合中
        if isinstance(map_names, str):
            return map_names in self.features_flags
        # 如果映射名称是列表，则遍历列表，检查是否有特性在 features_flags 集合中
        for f in map_names:
            if f in self.features_flags:
                return True
        return False

    # 定义一个方法 load_flags_cpuinfo，用于加载 CPU 相关信息的标志
    def load_flags_cpuinfo(self, magic_key):
        # 调用 get_cpuinfo_item 方法获取 CPU 相关信息的标志，并设置 features_flags
        self.features_flags = self.get_cpuinfo_item(magic_key)
    # 获取指定的 CPU 信息项目
    def get_cpuinfo_item(self, magic_key):
        # 初始化一个空集合来存储数值
        values = set()
        # 打开 CPU 信息文件 '/proc/cpuinfo'
        with open('/proc/cpuinfo') as fd:
            # 逐行读取文件内容
            for line in fd:
                # 如果当前行不以 magic_key 开头，则继续下一行
                if not line.startswith(magic_key):
                    continue
                # 将当前行按 ':' 分割成键和值，并去除两边的空格
                flags_value = [s.strip() for s in line.split(':', 1)]
                # 如果成功分割为键值对，将值转换为大写后分割成集合，并与已有集合合并
                if len(flags_value) == 2:
                    values = values.union(flags_value[1].upper().split())
        # 返回最终得到的数值集合
        return values

    # 加载辅助的标志寄存器信息
    def load_flags_auxv(self):
        # 使用 subprocess 模块执行命令 '/bin/true' 并设置环境变量 LD_SHOW_AUXV="1"，获取输出
        auxv = subprocess.check_output(['/bin/true'], env=dict(LD_SHOW_AUXV="1"))
        # 逐行遍历输出内容
        for at in auxv.split(b'\n'):
            # 如果当前行不以 "AT_HWCAP" 开头，则继续下一行
            if not at.startswith(b"AT_HWCAP"):
                continue
            # 将当前行按 ':' 分割成键和值，并去除两边的空格
            hwcap_value = [s.strip() for s in at.split(b':', 1)]
            # 如果成功分割为键值对，将值转换为大写后分割成列表，并与类属性 features_flags 合并
            if len(hwcap_value) == 2:
                self.features_flags = self.features_flags.union(
                    hwcap_value[1].upper().decode().split()
                )
# 如果运行平台是 emscripten，跳过这个测试类
@pytest.mark.skipif(
    sys.platform == 'emscripten',
    reason=(
        "The subprocess module is not available on WASM platforms and"
        " therefore this test class cannot be properly executed."
    ),
)
class TestEnvPrivation:
    # 获取当前文件的父目录的绝对路径
    cwd = pathlib.Path(__file__).parent.resolve()
    # 复制当前环境变量
    env = os.environ.copy()
    # 弹出环境变量中的 'NPY_ENABLE_CPU_FEATURES'，并保存到 _enable
    _enable = os.environ.pop('NPY_ENABLE_CPU_FEATURES', None)
    # 弹出环境变量中的 'NPY_DISABLE_CPU_FEATURES'，并保存到 _disable
    _disable = os.environ.pop('NPY_DISABLE_CPU_FEATURES', None)
    # 定义子进程参数的字典
    SUBPROCESS_ARGS = dict(cwd=cwd, capture_output=True, text=True, check=True)
    # 在 __cpu_dispatch__ 中遍历，找出不可用的 CPU 特性
    unavailable_feats = [
        feat for feat in __cpu_dispatch__ if not __cpu_features__[feat]
    ]
    # 第一个不可用的 CPU 特性，若没有则为 None
    UNAVAILABLE_FEAT = (
        None if len(unavailable_feats) == 0
        else unavailable_feats[0]
    )
    # 第一个基准 CPU 特性，若没有则为 None
    BASELINE_FEAT = None if len(__cpu_baseline__) == 0 else __cpu_baseline__[0]
    # 测试脚本的内容
    SCRIPT = """
def main():
    from numpy._core._multiarray_umath import (
        __cpu_features__, 
        __cpu_dispatch__
    )

    detected = [feat for feat in __cpu_dispatch__ if __cpu_features__[feat]]
    print(detected)

if __name__ == "__main__":
    main()
    """

    @pytest.fixture(autouse=True)
    def setup_class(self, tmp_path_factory):
        # 创建临时目录来存放运行时测试脚本
        file = tmp_path_factory.mktemp("runtime_test_script")
        file /= "_runtime_detect.py"
        # 将测试脚本内容写入文件
        file.write_text(self.SCRIPT)
        self.file = file
        return

    def _run(self):
        # 运行子进程，执行测试脚本
        return subprocess.run(
            [sys.executable, self.file],
            env=self.env,
            **self.SUBPROCESS_ARGS,
            )

    # 辅助函数，模拟 pytest.raises 对子进程调用的期望错误
    def _expect_error(
        self,
        msg,
        err_type,
        no_error_msg="Failed to generate error"
    ):
        try:
            self._run()
        except subprocess.CalledProcessError as e:
            assertion_message = f"Expected: {msg}\nGot: {e.stderr}"
            assert re.search(msg, e.stderr), assertion_message

            assertion_message = (
                f"Expected error of type: {err_type}; see full "
                f"error:\n{e.stderr}"
            )
            assert re.search(err_type, e.stderr), assertion_message
        else:
            assert False, no_error_msg

    def setup_method(self):
        """确保环境变量被重置"""
        self.env = os.environ.copy()
        return
    def test_runtime_feature_selection(self):
        """
        Ensure that when selecting `NPY_ENABLE_CPU_FEATURES`, only the
        features exactly specified are dispatched.
        """

        # 调用 _run 方法，捕获运行时启用的特性信息
        out = self._run()
        # 解析 stdout 中的非基准特性列表
        non_baseline_features = _text_to_list(out.stdout)

        if non_baseline_features is None:
            # 如果没有检测到除基准外的可调度特性，则跳过测试
            pytest.skip(
                "No dispatchable features outside of baseline detected."
            )
        # 从非基准特性列表中选择一个特性进行测试
        feature = non_baseline_features[0]

        # 设置环境变量 `NPY_ENABLE_CPU_FEATURES` 为选定的特性
        self.env['NPY_ENABLE_CPU_FEATURES'] = feature
        # 再次运行测试，捕获此时启用的特性信息
        out = self._run()
        # 解析 stdout 中的已启用特性列表
        enabled_features = _text_to_list(out.stdout)

        # 确保只有一个特性被启用，并且正是由 `NPY_ENABLE_CPU_FEATURES` 指定的特性
        assert set(enabled_features) == {feature}

        if len(non_baseline_features) < 2:
            # 如果只检测到一个非基准特性，则跳过测试
            pytest.skip("Only one non-baseline feature detected.")
        # 设置环境变量 `NPY_ENABLE_CPU_FEATURES` 为所有非基准特性列表的字符串形式
        self.env['NPY_ENABLE_CPU_FEATURES'] = ",".join(non_baseline_features)
        # 再次运行测试，捕获此时启用的特性信息
        out = self._run()
        # 解析 stdout 中的已启用特性列表
        enabled_features = _text_to_list(out.stdout)

        # 确保所有指定的特性都被正确启用
        assert set(enabled_features) == set(non_baseline_features)
        return

    @pytest.mark.parametrize("enabled, disabled",
    [
        ("feature", "feature"),
        ("feature", "same"),
    ])
    def test_both_enable_disable_set(self, enabled, disabled):
        """
        Ensure that when both environment variables are set then an
        ImportError is thrown
        """
        # 设置环境变量 `NPY_ENABLE_CPU_FEATURES` 和 `NPY_DISABLE_CPU_FEATURES`
        self.env['NPY_ENABLE_CPU_FEATURES'] = enabled
        self.env['NPY_DISABLE_CPU_FEATURES'] = disabled
        # 准备错误消息和错误类型
        msg = "Both NPY_DISABLE_CPU_FEATURES and NPY_ENABLE_CPU_FEATURES"
        err_type = "ImportError"
        # 断言预期的错误发生
        self._expect_error(msg, err_type)

    @pytest.mark.skipif(
        not __cpu_dispatch__,
        reason=(
            "NPY_*_CPU_FEATURES only parsed if "
            "`__cpu_dispatch__` is non-empty"
        )
    )
    @pytest.mark.parametrize("action", ["ENABLE", "DISABLE"])
    def test_variable_too_long(self, action):
        """
        Test that an error is thrown if the environment variables are too long
        to be processed. Current limit is 1024, but this may change later.
        """
        MAX_VAR_LENGTH = 1024
        # 设置环境变量 `NPY_{action}_CPU_FEATURES` 超出处理限制的长度
        self.env[f'NPY_{action}_CPU_FEATURES'] = "t" * MAX_VAR_LENGTH
        # 准备错误消息和错误类型
        msg = (
            f"Length of environment variable 'NPY_{action}_CPU_FEATURES' is "
            f"{MAX_VAR_LENGTH + 1}, only {MAX_VAR_LENGTH} accepted"
        )
        err_type = "RuntimeError"
        # 断言预期的错误发生
        self._expect_error(msg, err_type)
    @pytest.mark.skipif(
        not __cpu_dispatch__,
        reason=(
            "NPY_*_CPU_FEATURES only parsed if "
            "`__cpu_dispatch__` is non-empty"
        )
    )
    # 标记测试用例为跳过状态，条件是如果 `__cpu_dispatch__` 为空，则跳过执行
    def test_impossible_feature_disable(self):
        """
        Test that a RuntimeError is thrown if an impossible feature-disabling
        request is made. This includes disabling a baseline feature.
        """

        if self.BASELINE_FEAT is None:
            # 如果没有基准特性可用，则跳过测试
            pytest.skip("There are no unavailable features to test with")
        
        bad_feature = self.BASELINE_FEAT
        # 将环境变量设置为要禁用的基准特性
        self.env['NPY_DISABLE_CPU_FEATURES'] = bad_feature
        
        msg = (
            f"You cannot disable CPU feature '{bad_feature}', since it is "
            "part of the baseline optimizations"
        )
        err_type = "RuntimeError"
        # 预期出现特定错误消息和错误类型
        self._expect_error(msg, err_type)

    def test_impossible_feature_enable(self):
        """
        Test that a RuntimeError is thrown if an impossible feature-enabling
        request is made. This includes enabling a feature not supported by the
        machine, or disabling a baseline optimization.
        """

        if self.UNAVAILABLE_FEAT is None:
            # 如果没有不可用特性可用，则跳过测试
            pytest.skip("There are no unavailable features to test with")
        
        bad_feature = self.UNAVAILABLE_FEAT
        # 将环境变量设置为要启用的不支持的 CPU 特性
        self.env['NPY_ENABLE_CPU_FEATURES'] = bad_feature
        
        msg = (
            f"You cannot enable CPU features \\({bad_feature}\\), since "
            "they are not supported by your machine."
        )
        err_type = "RuntimeError"
        # 预期出现特定错误消息和错误类型
        self._expect_error(msg, err_type)

        # 确保即使提供了额外的垃圾数据，测试依然会失败
        feats = f"{bad_feature}, Foobar"
        self.env['NPY_ENABLE_CPU_FEATURES'] = feats
        msg = (
            f"You cannot enable CPU features \\({bad_feature}\\), since they "
            "are not supported by your machine."
        )
        self._expect_error(msg, err_type)

        if self.BASELINE_FEAT is not None:
            # 确保只有错误的特性会被报告
            feats = f"{bad_feature}, {self.BASELINE_FEAT}"
            self.env['NPY_ENABLE_CPU_FEATURES'] = feats
            msg = (
                f"You cannot enable CPU features \\({bad_feature}\\), since "
                "they are not supported by your machine."
            )
            self._expect_error(msg, err_type)
# 检查当前操作系统是否为Linux
is_linux = sys.platform.startswith('linux')
# 检查当前操作系统是否为Cygwin
is_cygwin = sys.platform.startswith('cygwin')
# 获取当前机器的架构信息
machine = platform.machine()
# 使用正则表达式匹配机器架构，判断是否为x86架构
is_x86 = re.match("^(amd64|x86|i386|i686)", machine, re.IGNORECASE)

# 根据条件标记测试类为跳过状态，仅当操作系统为Linux且机器架构为x86时执行
@pytest.mark.skipif(
    not (is_linux or is_cygwin) or not is_x86, reason="Only for Linux and x86"
)
class Test_X86_Features(AbstractTest):
    # 定义x86架构支持的特性列表
    features = [
        "MMX", "SSE", "SSE2", "SSE3", "SSSE3", "SSE41", "POPCNT", "SSE42",
        "AVX", "F16C", "XOP", "FMA4", "FMA3", "AVX2", "AVX512F", "AVX512CD",
        "AVX512ER", "AVX512PF", "AVX5124FMAPS", "AVX5124VNNIW", "AVX512VPOPCNTDQ",
        "AVX512VL", "AVX512BW", "AVX512DQ", "AVX512VNNI", "AVX512IFMA",
        "AVX512VBMI", "AVX512VBMI2", "AVX512BITALG", "AVX512FP16",
    ]
    # 定义x86架构特性的分组
    features_groups = dict(
        AVX512_KNL=["AVX512F", "AVX512CD", "AVX512ER", "AVX512PF"],
        AVX512_KNM=["AVX512F", "AVX512CD", "AVX512ER", "AVX512PF", "AVX5124FMAPS",
                    "AVX5124VNNIW", "AVX512VPOPCNTDQ"],
        AVX512_SKX=["AVX512F", "AVX512CD", "AVX512BW", "AVX512DQ", "AVX512VL"],
        AVX512_CLX=["AVX512F", "AVX512CD", "AVX512BW", "AVX512DQ", "AVX512VL", "AVX512VNNI"],
        AVX512_CNL=["AVX512F", "AVX512CD", "AVX512BW", "AVX512DQ", "AVX512VL", "AVX512IFMA",
                    "AVX512VBMI"],
        AVX512_ICL=["AVX512F", "AVX512CD", "AVX512BW", "AVX512DQ", "AVX512VL", "AVX512IFMA",
                    "AVX512VBMI", "AVX512VNNI", "AVX512VBMI2", "AVX512BITALG", "AVX512VPOPCNTDQ"],
        AVX512_SPR=["AVX512F", "AVX512CD", "AVX512BW", "AVX512DQ",
                    "AVX512VL", "AVX512IFMA", "AVX512VBMI", "AVX512VNNI",
                    "AVX512VBMI2", "AVX512BITALG", "AVX512VPOPCNTDQ",
                    "AVX512FP16"],
    )
    # 定义x86架构特性的映射表
    features_map = dict(
        SSE3="PNI", SSE41="SSE4_1", SSE42="SSE4_2", FMA3="FMA",
        AVX512VNNI="AVX512_VNNI", AVX512BITALG="AVX512_BITALG", AVX512VBMI2="AVX512_VBMI2",
        AVX5124FMAPS="AVX512_4FMAPS", AVX5124VNNIW="AVX512_4VNNIW", AVX512VPOPCNTDQ="AVX512_VPOPCNTDQ",
        AVX512FP16="AVX512_FP16",
    )

    # 加载x86架构特性标志的方法
    def load_flags(self):
        self.load_flags_cpuinfo("flags")

# 使用正则表达式匹配机器架构，判断是否为PowerPC架构
is_power = re.match("^(powerpc|ppc)64", machine, re.IGNORECASE)

# 根据条件标记测试类为跳过状态，仅当操作系统为Linux且机器架构为PowerPC时执行
@pytest.mark.skipif(not is_linux or not is_power, reason="Only for Linux and Power")
class Test_POWER_Features(AbstractTest):
    # 定义PowerPC架构支持的特性列表
    features = ["VSX", "VSX2", "VSX3", "VSX4"]
    # 定义PowerPC架构特性的映射表
    features_map = dict(VSX2="ARCH_2_07", VSX3="ARCH_3_00", VSX4="ARCH_3_1")

    # 加载PowerPC架构特性标志的方法
    def load_flags(self):
        self.load_flags_auxv()

# 使用正则表达式匹配机器架构，判断是否为IBM Z架构
is_zarch = re.match("^(s390x)", machine, re.IGNORECASE)

# 根据条件标记测试类为跳过状态，仅当操作系统为Linux且机器架构为IBM Z时执行
@pytest.mark.skipif(not is_linux or not is_zarch,
                    reason="Only for Linux and IBM Z")
class Test_ZARCH_Features(AbstractTest):
    # 定义IBM Z架构支持的特性列表
    features = ["VX", "VXE", "VXE2"]

    # 加载IBM Z架构特性标志的方法
    def load_flags(self):
        self.load_flags_auxv()

# 使用正则表达式匹配机器架构，判断是否为ARM架构
is_arm = re.match("^(arm|aarch64)", machine, re.IGNORECASE)

# 根据条件标记测试类为跳过状态，仅当操作系统为Linux且机器架构为ARM时执行
@pytest.mark.skipif(not is_linux or not is_arm, reason="Only for Linux and ARM")
class Test_ARM_Features(AbstractTest):
    # 空白，待补充的测试类，未提供具体特性和特性映射
    # 定义一个包含处理器功能名称的列表
    features = [
        "SVE", "NEON", "ASIMD", "FPHP", "ASIMDHP", "ASIMDDP", "ASIMDFHM"
    ]
    # 定义一个包含功能组名称和其成员列表的字典
    features_groups = dict(
        NEON_FP16  = ["NEON", "HALF"],
        NEON_VFPV4 = ["NEON", "VFPV4"],
    )
    
    # 定义一个方法用于加载处理器功能标志
    def load_flags(self):
        # 调用一个方法加载 CPU 信息中的特性信息
        self.load_flags_cpuinfo("Features")
        # 获取 CPU 架构信息
        arch = self.get_cpuinfo_item("CPU architecture")
        # 如果是在挂载的虚拟文件系统中运行的 aarch64 内核
        # 判断是否为 aarch64 架构并设置 is_rootfs_v8 标志
        is_rootfs_v8 = int('0'+next(iter(arch))) > 7 if arch else 0
        # 如果系统匹配 aarch64 或 is_rootfs_v8 则进行以下判断
        if  re.match("^(aarch64|AARCH64)", machine) or is_rootfs_v8:
            # 设置特性映射字典，匹配 NEON 到 ASIMD，HALF 到 ASIMD，VFPV4 到 ASIMD
            self.features_map = dict(
                NEON="ASIMD", HALF="ASIMD", VFPV4="ASIMD"
            )
        else:
            # 否则设置特性映射字典，假设 ASIMD 被支持，因为 Linux 内核 (armv8 aarch32) 
            # 在 ELF 辅助向量和 /proc/cpuinfo 上未提供有关 ASIMD 的信息
            self.features_map = dict(
                ASIMD=("AES", "SHA1", "SHA2", "PMULL", "CRC32")
            )
```