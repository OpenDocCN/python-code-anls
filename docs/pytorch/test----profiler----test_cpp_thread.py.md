# `.\pytorch\test\profiler\test_cpp_thread.py`

```py
# Owner(s): ["oncall: profiler"]

# 引入必要的标准库和第三方库
import os
import shutil
import subprocess

# 引入PyTorch及其相关扩展模块
import torch
import torch.utils.cpp_extension
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase

# 移除默认的构建路径
def remove_build_path():
    # 获取默认的构建根目录
    default_build_root = torch.utils.cpp_extension.get_default_build_root()
    # 如果构建根目录存在，则尝试删除它
    if os.path.exists(default_build_root):
        if IS_WINDOWS:
            # 在Windows上，使用subprocess调用rm命令删除目录（解决权限问题）
            subprocess.run(["rm", "-rf", default_build_root], stdout=subprocess.PIPE)
        else:
            # 在其他系统上，直接使用shutil.rmtree递归删除目录
            shutil.rmtree(default_build_root)

# 判断当前环境是否为Facebook的开发环境
def is_fbcode():
    return not hasattr(torch.version, "git_version")

# 根据环境加载不同的C++库模块
if is_fbcode():
    # 如果是在Facebook的环境下，则导入特定的C++测试库模块
    import caffe2.test.profiler_test_cpp_thread_lib as cpp
else:
    # 如果不是在Facebook环境下，则加载本地的C++扩展模块
    # 需要将工作目录切换到当前脚本所在目录的上级目录
    old_working_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 加载名为"profiler_test_cpp_thread_lib"的C++扩展模块，指定源文件为"test_cpp_thread.cpp"
    cpp = torch.utils.cpp_extension.load(
        name="profiler_test_cpp_thread_lib",
        sources=[
            "test_cpp_thread.cpp",
        ],
        verbose=True,
    )
    
    # 加载完成后恢复原工作目录
    os.chdir(old_working_dir)

# 设置全局变量和常量
KinetoProfiler = None
IterationCount = 5
ActivateIteration = 2

# 定义打印蓝图效果的函数
def blueprint(text):
    print(f"\33[34m{text}\33[0m")

# 定义Python版本的Profiler事件处理器类，继承自C++库中的ProfilerEventHandler类
class PythonProfilerEventHandler(cpp.ProfilerEventHandler):
    def onIterationStart(self, iteration: int) -> None:
        global KinetoProfiler, IterationCount
        # 在同一个线程上调用step()时启动Profiler非常重要
        # onIterationStart()始终在同一个线程上调用
        if iteration == 0:
            # 当iteration为0时启动Profiler
            KinetoProfiler.start()
            blueprint("starting kineto profiler")
        elif iteration == IterationCount - 1:
            # 当iteration达到设定的迭代次数减1时停止Profiler
            KinetoProfiler.stop()
            blueprint("stopping kineto profiler")
        else:
            # 在其他迭代次数时调用Profiler的step()方法
            blueprint("stepping kineto profiler")
            KinetoProfiler.step()

    def emulateTraining(self, iteration: int, thread_id: int) -> None:
        # 模拟训练过程，打印训练迭代和线程ID
        # blueprint(f"training iteration {iteration} in thread {thread_id}")
        device = torch.device("cuda")
        # 使用torch.autograd.profiler.record_function记录"user_function"函数的性能
        with torch.autograd.profiler.record_function("user_function"):
            a = torch.ones(1, device=device)
            b = torch.ones(1, device=device)
            torch.add(a, b).cpu()
            torch.cuda.synchronize()

# 定义CppThreadTest类，继承自TestCase
class CppThreadTest(TestCase):
    ThreadCount = 20  # 设置线程数为20（调试时可修改为2）
    EventHandler = None
    TraceObject = None

    @classmethod
    # 设置测试类的类级别初始化方法
    def setUpClass(cls) -> None:
        # 调用父类的 setUpClass 方法，准备测试类的环境
        super(TestCase, cls).setUpClass()
        # 创建 PythonProfilerEventHandler 实例，并设置为 CppThreadTest.EventHandler
        CppThreadTest.EventHandler = PythonProfilerEventHandler()
        # 将事件处理程序注册到 cpp.ProfilerEventHandler 中
        cpp.ProfilerEventHandler.Register(CppThreadTest.EventHandler)

    # 设置测试类的类级别清理方法
    @classmethod
    def tearDownClass(cls):
        # 如果不是在 fbcode 环境下，则移除构建路径
        if not is_fbcode():
            remove_build_path()

    # 设置测试方法的初始化方法
    def setUp(self) -> None:
        # 如果没有 CUDA 可用，则跳过测试
        if not torch.cuda.is_available():
            self.skipTest("Test machine does not have cuda")

        # 清除初始化过程中的事件
        self.start_profiler(False)
        # 启动一个线程进行测试，设置迭代次数为 IterationCount
        cpp.start_threads(1, IterationCount, False)

    # 启动性能分析器的方法
    def start_profiler(self, profile_memory):
        # 声明全局变量 KinetoProfiler
        global KinetoProfiler
        # 使用 torch.profiler.profile 方法创建性能分析器
        KinetoProfiler = torch.profiler.profile(
            # 设置性能分析器的调度策略
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=ActivateIteration, repeat=1
            ),
            # 设置跟踪准备好时的回调函数为 self.set_trace
            on_trace_ready=self.set_trace,
            # 开启堆栈跟踪
            with_stack=True,
            # 设置是否进行内存分析
            profile_memory=profile_memory,
            # 记录张量形状信息
            record_shapes=True,
        )

    # 设置跟踪对象的方法
    def set_trace(self, trace_obj) -> None:
        # 将 trace_obj 设置为 CppThreadTest.TraceObject
        CppThreadTest.TraceObject = trace_obj

    # 断言文本输出的方法
    def assert_text(self, condition, text, msg):
        # 如果条件为真，以绿色打印文本，否则以红色打印文本
        if condition:
            print(f"\33[32m{text}\33[0m")
        else:
            print(f"\33[31m{text}\33[0m")
        # 断言条件为真，输出 msg 作为错误信息
        self.assertTrue(condition, msg)
    # 验证追踪信息，检查预期事件是否发生
    def check_trace(self, expected, mem=False) -> None:
        # 输出信息，表示正在验证追踪信息
        blueprint("verifying trace")
        # 获取CppThreadTest.TraceObject中记录的事件列表
        event_list = CppThreadTest.TraceObject.events()
        # 遍历期望的事件字典
        for key, values in expected.items():
            # 获取预期事件发生的次数
            count = values[0]
            # 计算至少应该出现的事件次数
            min_count = count * (ActivateIteration - 1)
            # 获取预期事件所在的设备类型
            device = values[1]
            # 使用过滤器筛选事件列表中符合条件的事件
            filtered = filter(
                lambda ev: ev.name == key
                and str(ev.device_type) == f"DeviceType.{device}",
                event_list,
            )

            # 如果需要验证内存使用情况
            if mem:
                actual = 0
                # 统计符合条件的事件中cuda_memory_usage设置的数量
                for ev in filtered:
                    sev = str(ev)
                    has_cuda_memory_usage = (
                        sev.find("cuda_memory_usage=0 ") < 0
                        and sev.find("cuda_memory_usage=") > 0
                    )
                    if has_cuda_memory_usage:
                        actual += 1
                # 断言实际cuda_memory_usage设置的事件数量大于等于最小次数要求
                self.assert_text(
                    actual >= min_count,
                    f"{key}: {actual} >= {min_count}",
                    "not enough event with cuda_memory_usage set",
                )
            else:
                # 统计符合条件的事件数量
                actual = len(list(filtered))
                # 如果预期事件次数为1，乘以迭代次数
                if count == 1:  # test_without
                    count *= ActivateIteration
                    # 断言实际事件数量等于计算出的预期次数
                    self.assert_text(
                        actual == count,
                        f"{key}: {actual} == {count}",
                        "baseline event count incorrect",
                    )
                else:
                    # 断言实际事件数量大于等于最小次数要求
                    self.assert_text(
                        actual >= min_count,
                        f"{key}: {actual} >= {min_count}",
                        "not enough event recorded",
                    )

    # 测试在子线程中启用性能分析器的情况
    def test_with_enable_profiler_in_child_thread(self) -> None:
        # 启动性能分析器，并开始线程
        self.start_profiler(False)
        cpp.start_threads(self.ThreadCount, IterationCount, True)
        # 调用check_trace方法验证预期事件的发生情况
        self.check_trace(
            {
                "aten::add": [self.ThreadCount, "CPU"],
                "user_function": [self.ThreadCount, "CUDA"],
            }
        )

    # 测试在子线程中不启用性能分析器的情况
    def test_without_enable_profiler_in_child_thread(self) -> None:
        # 不启动性能分析器，并开始线程
        self.start_profiler(False)
        cpp.start_threads(self.ThreadCount, IterationCount, False)
        # 调用check_trace方法验证预期事件的发生情况
        self.check_trace(
            {
                "aten::add": [1, "CPU"],
                "user_function": [1, "CUDA"],
            }
        )

    # 测试在子线程中启用内存分析的情况
    def test_profile_memory(self) -> None:
        # 启动内存分析器，并开始线程
        self.start_profiler(True)
        cpp.start_threads(self.ThreadCount, IterationCount, True)
        # 调用check_trace方法验证预期事件的发生情况，同时验证内存使用情况
        self.check_trace(
            {
                "aten::add": [self.ThreadCount, "CPU"],
            },
            mem=True,
        )
# 如果这个脚本作为主程序执行（而不是作为模块被导入），则运行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```