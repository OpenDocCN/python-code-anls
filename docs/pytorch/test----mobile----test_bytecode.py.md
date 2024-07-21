# `.\pytorch\test\mobile\test_bytecode.py`

```
# Owner(s): ["oncall: mobile"]

# 导入所需的库和模块
import fnmatch  # 导入用于文件名匹配的模块
import io  # 导入用于处理流的模块
import shutil  # 导入用于文件操作的模块
import tempfile  # 导入用于创建临时文件和目录的模块
from pathlib import Path  # 导入用于处理文件路径的模块

import torch  # 导入 PyTorch 深度学习库
import torch.utils.show_pickle  # 导入用于展示 pickle 文件内容的模块

# 下面的代码已经被注释掉，这是导入 mobile_optimizer 模块的部分
# from torch.utils.mobile_optimizer import optimize_for_mobile

# 导入 torch.jit.mobile 中的函数和类
from torch.jit.mobile import (
    _backport_for_mobile,
    _backport_for_mobile_to_buffer,
    _get_mobile_model_contained_types,
    _get_model_bytecode_version,
    _get_model_ops_and_info,
    _load_for_lite_interpreter,
)
from torch.testing._internal.common_utils import run_tests, TestCase  # 导入用于测试的辅助函数和类

# 获取当前脚本文件的上级目录路径
pytorch_test_dir = Path(__file__).resolve().parents[1]

# 下面是脚本模块 v4 和 v5 的字节码，用三重引号表示长字符串
# 用于描述 TestModule 类的 forward 方法的字节码，版本为 v4
SCRIPT_MODULE_V4_BYTECODE_PKL = """
(4,
 ('__torch__.*.TestModule.forward',
  (('instructions',
    (('STOREN', 1, 2),
     ('DROPR', 1, 0),
     ('LOADC', 0, 0),
     ('LOADC', 1, 0),
     ('MOVE', 2, 0),
     ('OP', 0, 0),
     ('LOADC', 1, 0),
     ('OP', 1, 0),
     ('RET', 0, 0))),
   ('operators', (('aten::add', 'int'), ('aten::add', 'Scalar'))),
   ('constants',
    (torch._utils._rebuild_tensor_v2(pers.obj(('storage', torch.DoubleStorage, '0', 'cpu', 8),),
       0,
       (2, 4),
       (4, 1),
       False,
       collections.OrderedDict()),
     1)),
   ('types', ()),
   ('register_size', 2)),
  (('arguments',
    ((('name', 'self'),
      ('type', '__torch__.*.TestModule'),
      ('default_value', None)),
     (('name', 'y'), ('type', 'int'), ('default_value', None)))),
   ('returns',
    ((('name', ''), ('type', 'Tensor'), ('default_value', None)),)))))
"""

# 用于描述 TestModule 类的 forward 方法的字节码，版本为 v5
SCRIPT_MODULE_V5_BYTECODE_PKL = """
(5,
 ('__torch__.*.TestModule.forward',
  (('instructions',
    (('STOREN', 1, 2),
     ('DROPR', 1, 0),
     ('LOADC', 0, 0),
     ('LOADC', 1, 0),
     ('MOVE', 2, 0),
     ('OP', 0, 0),
     ('LOADC', 1, 0),
     ('OP', 1, 0),
     ('RET', 0, 0))),
   ('operators', (('aten::add', 'int'), ('aten::add', 'Scalar'))),
   ('constants',
    (torch._utils._rebuild_tensor_v2(pers.obj(('storage', torch.DoubleStorage, 'constants/0', 'cpu', 8),),
       0,
       (2, 4),
       (4, 1),
       False,
       collections.OrderedDict()),
     1)),
   ('types', ()),
   ('register_size', 2)),
  (('arguments',
    ((('name', 'self'),
      ('type', '__torch__.*.TestModule'),
      ('default_value', None)),
     (('name', 'y'), ('type', 'int'), ('default_value', None)))),
   ('returns',
    ((('name', ''), ('type', 'Tensor'), ('default_value', None)),)))))
"""
# 定义一个多行字符串常量，存储模块的字节码信息
SCRIPT_MODULE_V6_BYTECODE_PKL = """
(6,
 ('__torch__.*.TestModule.forward',
  (('instructions',
    (('STOREN', 1, 2),          # 存储变量
     ('DROPR', 1, 0),           # 删除变量
     ('LOADC', 0, 0),           # 载入常量
     ('LOADC', 1, 0),           # 载入常量
     ('MOVE', 2, 0),            # 移动变量
     ('OP', 0, 0),              # 运算
     ('OP', 1, 0),              # 运算
     ('RET', 0, 0))),           # 返回
   ('operators', (('aten::add', 'int', 2), ('aten::add', 'Scalar', 2))),  # 操作符列表
   ('constants',
    (torch._utils._rebuild_tensor_v2(pers.obj(('storage', torch.DoubleStorage, '0', 'cpu', 8),),
       0,
       (2, 4),
       (4, 1),
       False,
       collections.OrderedDict()),  # 常量列表
     1)),
   ('types', ()),                 # 类型信息
   ('register_size', 2)),         # 寄存器大小
  (('arguments',
    ((('name', 'self'),          # 参数：self
      ('type', '__torch__.*.TestModule'),  # 类型：__torch__.*.TestModule
      ('default_value', None)),
     (('name', 'y'), ('type', 'int'), ('default_value', None)))),  # 参数：y
   ('returns',
    ((('name', ''), ('type', 'Tensor'), ('default_value', None)),))))  # 返回值：Tensor
    """

# 定义字典，存储不同模型版本的字节码信息和模型名称
SCRIPT_MODULE_BYTECODE_PKL = {
    4: {
        "bytecode_pkl": SCRIPT_MODULE_V4_BYTECODE_PKL,  # 模型字节码信息
        "model_name": "script_module_v4.ptl",           # 模型文件名
    },
}

# 最低可以回退到的模型版本号
# 当某个字节码版本完全淘汰时需要更新此版本号
MINIMUM_TO_VERSION = 4

# 测试不同模型版本的类
class testVariousModelVersions(TestCase):
    # 测试获取模型字节码版本的方法
    def test_get_model_bytecode_version(self):
        # 检查模型版本号是否符合预期
        def check_model_version(model_path, expect_version):
            actual_version = _get_model_bytecode_version(model_path)
            assert actual_version == expect_version

        # 遍历不同的模型版本和相关信息
        for version, model_info in SCRIPT_MODULE_BYTECODE_PKL.items():
            # 构建模型路径
            model_path = pytorch_test_dir / "cpp" / "jit" / model_info["model_name"]
            # 检查模型版本号
            check_model_version(model_path, version)
    def test_bytecode_values_for_all_backport_functions(self):
        # 检查所有回溯函数的字节码数值
        # 找到已检查模型的最大版本，从最小支持版本开始回溯，并比较字节码.pkl内容。
        # 无法合并到测试`test_all_backport_functions`中，因为优化是动态的，
        # 当 optimize 函数更改时内容可能会改变。该测试专注于字节码.pkl内容的验证。
        # 对于内容验证，不是逐字节检查，而是正则表达式匹配。通配符可用于跳过特定内容比较。
        maximum_checked_in_model_version = max(SCRIPT_MODULE_BYTECODE_PKL.keys())
        current_from_version = maximum_checked_in_model_version

        with tempfile.TemporaryDirectory() as tmpdirname:
            while current_from_version > MINIMUM_TO_VERSION:
                # 加载模型v5并运行前向方法
                model_name = SCRIPT_MODULE_BYTECODE_PKL[current_from_version][
                    "model_name"
                ]
                input_model_path = pytorch_test_dir / "cpp" / "jit" / model_name

                # 临时模型文件将被导出到此路径，并通过字节码.pkl内容检查。
                tmp_output_model_path_backport = Path(
                    tmpdirname, "tmp_script_module_backport.ptl"
                )

                current_to_version = current_from_version - 1
                # 进行模型回溯到较低版本
                backport_success = _backport_for_mobile(
                    input_model_path, tmp_output_model_path_backport, current_to_version
                )
                assert backport_success

                # 期望的字节码.pkl内容
                expect_bytecode_pkl = SCRIPT_MODULE_BYTECODE_PKL[current_to_version][
                    "bytecode_pkl"
                ]

                # 使用torch.utils.show_pickle.main显示字节码.pkl文件内容
                buf = io.StringIO()
                torch.utils.show_pickle.main(
                    [
                        "",
                        tmpdirname
                        + "/"
                        + tmp_output_model_path_backport.name
                        + "@*/bytecode.pkl",
                    ],
                    output_stream=buf,
                )
                output = buf.getvalue()

                # 清理实际输出和期望输出中的空白字符，进行匹配检查
                acutal_result_clean = "".join(output.split())
                expect_result_clean = "".join(expect_bytecode_pkl.split())
                isMatch = fnmatch.fnmatch(acutal_result_clean, expect_result_clean)
                assert isMatch

                current_from_version -= 1
            # 清理临时目录
            shutil.rmtree(tmpdirname)

    # 在进行回溯时，请手动运行此测试。
    # 该测试在OSS中通过，但在内部测试中失败，可能是由于构建中的某些步骤缺失。
    # def test_all_backport_functions(self):
    #     # 从最新的字节码版本回溯到最小支持版本
    #     # 加载、运行回溯模型，并检查版本
    #     class TestModule(torch.nn.Module):
    # 定义一个测试模块类 TestModule，接受一个值作为参数并初始化
    class TestModule(torch.nn.Module):
        def __init__(self, v):
            # 调用父类的初始化方法
            super().__init__()
            # 设置实例变量 x 为输入的值 v
            self.x = v

        # 定义前向传播方法，接受一个整数 y 作为输入
        def forward(self, y: int):
            # 创建一个形状为 [2, 4] 的浮点数张量，元素都为 1
            increment = torch.ones([2, 4], dtype=torch.float64)
            # 返回 self.x + y + increment 的结果
            return self.x + y + increment

    # 定义模块的输入值为 1
    module_input = 1
    # 预期的移动模块结果为形状为 [2, 4] 的张量，所有元素为 3
    expected_mobile_module_result = 3 * torch.ones([2, 4], dtype=torch.float64)

    # 使用临时文件夹进行上下文管理，创建临时文件夹
    with tempfile.TemporaryDirectory() as tmpdirname:
        # 创建临时输入模型文件路径，命名为 "tmp_script_module.ptl"
        tmp_input_model_path = Path(tmpdirname, "tmp_script_module.ptl")
        # 使用 TestModule 类创建一个 TorchScript 模块
        script_module = torch.jit.script(TestModule(1))
        # 对 TorchScript 模块进行移动优化
        optimized_scripted_module = optimize_for_mobile(script_module)
        # 导出优化后的 TorchScript 模块为 Lite 解释器使用的格式，并保存到临时输入模型路径
        exported_optimized_scripted_module = optimized_scripted_module._save_for_lite_interpreter(str(tmp_input_model_path))

        # 获取当前模型字节码版本号
        current_from_version = _get_model_bytecode_version(tmp_input_model_path)
        # 计算目标模型字节码版本号，比当前版本低 1
        current_to_version = current_from_version - 1
        # 创建临时输出模型文件路径，命名为 "tmp_script_module_backport.ptl"
        tmp_output_model_path = Path(tmpdirname, "tmp_script_module_backport.ptl")

        # 循环直到达到最低兼容版本 MINIMUM_TO_VERSION
        while current_to_version >= MINIMUM_TO_VERSION:
            # 将最新版本的模型回退到 current_to_version，并保存到临时输出模型文件中
            backport_success = _backport_for_mobile(tmp_input_model_path, tmp_output_model_path, current_to_version)
            # 断言回退成功
            assert(backport_success)

            # 检查回退后模型的版本号是否为当前目标版本号
            backport_version = _get_model_bytecode_version(tmp_output_model_path)
            assert(backport_version == current_to_version)

            # 加载 Lite 解释器格式的模型，并运行 forward 方法得到结果 mobile_module_result
            mobile_module = _load_for_lite_interpreter(str(tmp_input_model_path))
            mobile_module_result = mobile_module(module_input)
            # 断言 Lite 解释器模型运行结果与预期的移动模块结果相似
            torch.testing.assert_close(mobile_module_result, expected_mobile_module_result)
            # 减少目标版本号，继续回退操作
            current_to_version -= 1

        # 检查回退失败的情况
        backport_success = _backport_for_mobile(tmp_input_model_path, tmp_output_model_path, MINIMUM_TO_VERSION - 1)
        assert(not backport_success)
        # 在临时文件夹关闭之前需要清理文件夹，否则会出现 Git 未清理错误
        shutil.rmtree(tmpdirname)

    # 仅检查 test_backport_bytecode_from_file_to_file 机制，而不是函数实现细节
    # 定义一个测试方法，用于测试从文件到文件的字节码回溯功能
    def test_backport_bytecode_from_file_to_file(self):
        # 获取已检查的模型版本中的最大值
        maximum_checked_in_model_version = max(SCRIPT_MODULE_BYTECODE_PKL.keys())
        # 构建脚本模型v5的路径，根据最大版本的模型名称确定
        script_module_v5_path = (
            pytorch_test_dir
            / "cpp"
            / "jit"
            / SCRIPT_MODULE_BYTECODE_PKL[maximum_checked_in_model_version]["model_name"]
        )

        # 检查是否需要回溯到更早版本
        if maximum_checked_in_model_version > MINIMUM_TO_VERSION:
            # 创建临时目录
            with tempfile.TemporaryDirectory() as tmpdirname:
                # 设置临时的回溯模型路径为v4版本
                tmp_backport_model_path = Path(
                    tmpdirname, "tmp_script_module_v5_backported_to_v4.ptl"
                )
                # 调用_backport_for_mobile函数进行模型回溯
                success = _backport_for_mobile(
                    script_module_v5_path,
                    tmp_backport_model_path,
                    maximum_checked_in_model_version - 1,
                )
                # 确认回溯成功
                assert success

                # 创建一个字符串IO缓冲区
                buf = io.StringIO()
                # 运行torch.utils.show_pickle.main函数，显示回溯后模型的字节码信息
                torch.utils.show_pickle.main(
                    [
                        "",
                        tmpdirname
                        + "/"
                        + tmp_backport_model_path.name
                        + "@*/bytecode.pkl",
                    ],
                    output_stream=buf,
                )
                # 获取显示结果字符串
                output = buf.getvalue()

                # 期望的结果为SCRIPT_MODULE_V4_BYTECODE_PKL
                expected_result = SCRIPT_MODULE_V4_BYTECODE_PKL
                # 清理实际结果和期望结果的空白字符，以便比较
                acutal_result_clean = "".join(output.split())
                expect_result_clean = "".join(expected_result.split())
                # 使用fnmatch模块检查实际结果是否匹配期望结果的模式
                isMatch = fnmatch.fnmatch(acutal_result_clean, expect_result_clean)
                # 断言实际结果与期望结果匹配
                assert isMatch

                # 加载v4版本模型，并运行其前向方法
                mobile_module = _load_for_lite_interpreter(str(tmp_backport_model_path))
                module_input = 1
                mobile_module_result = mobile_module(module_input)
                # 期望移动模块的结果为全为3的torch张量
                expected_mobile_module_result = 3 * torch.ones(
                    [2, 4], dtype=torch.float64
                )
                # 使用torch.testing.assert_close函数检查移动模块的结果是否与期望结果接近
                torch.testing.assert_close(
                    mobile_module_result, expected_mobile_module_result
                )
                # 递归删除临时目录及其内容
                shutil.rmtree(tmpdirname)

    # 检查_backport_for_mobile_to_buffer机制，但不涉及函数实现细节
    def test_backport_bytecode_from_file_to_buffer(self):
        # 获取当前已检查模型版本的最大值
        maximum_checked_in_model_version = max(SCRIPT_MODULE_BYTECODE_PKL.keys())
        # 构建脚本模块 v5 的路径
        script_module_v5_path = (
            pytorch_test_dir
            / "cpp"
            / "jit"
            / SCRIPT_MODULE_BYTECODE_PKL[maximum_checked_in_model_version]["model_name"]
        )

        # 如果当前模型版本大于最低支持版本 MINIMUM_TO_VERSION
        if maximum_checked_in_model_version > MINIMUM_TO_VERSION:
            # 将模型回溯到 v4 版本
            script_module_v4_buffer = _backport_for_mobile_to_buffer(
                script_module_v5_path, maximum_checked_in_model_version - 1
            )
            buf = io.StringIO()

            # 获取回溯后的模型 v4 的版本信息
            bytesio = io.BytesIO(script_module_v4_buffer)
            backport_version = _get_model_bytecode_version(bytesio)
            assert backport_version == maximum_checked_in_model_version - 1

            # 加载回溯后的模型 v4，并运行其前向方法
            bytesio = io.BytesIO(script_module_v4_buffer)
            mobile_module = _load_for_lite_interpreter(bytesio)
            module_input = 1
            mobile_module_result = mobile_module(module_input)
            expected_mobile_module_result = 3 * torch.ones([2, 4], dtype=torch.float64)
            torch.testing.assert_close(
                mobile_module_result, expected_mobile_module_result
            )

    def test_get_model_ops_and_info(self):
        # TODO update this to be more in the style of the above tests after a backport from 6 -> 5 exists
        # 获取脚本模块 v6 的路径
        script_module_v6 = pytorch_test_dir / "cpp" / "jit" / "script_module_v6.ptl"
        # 获取模型 v6 的操作和信息
        ops_v6 = _get_model_ops_and_info(script_module_v6)
        # 断言模型 v6 中 add.int 操作的参数数目为 2
        assert ops_v6["aten::add.int"].num_schema_args == 2
        # 断言模型 v6 中 add.Scalar 操作的参数数目为 2
        assert ops_v6["aten::add.Scalar"].num_schema_args == 2

    def test_get_mobile_model_contained_types(self):
        class MyTestModule(torch.nn.Module):
            def forward(self, x):
                return x + 10

        sample_input = torch.tensor([1])

        # 通过 TorchScript 将自定义模块 MyTestModule 脚本化
        script_module = torch.jit.script(MyTestModule())
        # 运行脚本化模块的前向方法并获取结果
        script_module_result = script_module(sample_input)

        # 将模型保存为字节流并获取其包含的类型信息列表
        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        type_list = _get_mobile_model_contained_types(buffer)
        # 断言类型列表的长度至少为 0
        assert len(type_list) >= 0
# 如果当前脚本作为主程序运行（而不是被导入到其他模块中），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```