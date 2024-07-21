# `.\pytorch\test\test_model_dump.py`

```py
# 指定 Python 解释器位置，用于运行脚本
#!/usr/bin/env python3

# 文件所有者信息，用于指明负责人或责任人员
# Owner(s): ["oncall: mobile"]

# 导入标准库中的模块
import os  # 导入操作系统相关功能的模块
import io  # 导入用于处理流的模块
import functools  # 导入用于高阶函数操作的模块
import tempfile  # 导入用于创建临时文件和目录的模块
import urllib  # 导入用于解析 URL 的模块
import unittest  # 导入用于编写和运行单元测试的模块

# 导入 PyTorch 库中的相关模块和函数
import torch  # 导入 PyTorch 深度学习框架
import torch.backends.xnnpack  # 导入 PyTorch 的 XNNPACK 后端支持
import torch.utils.model_dump  # 导入 PyTorch 的模型保存和加载工具
import torch.utils.mobile_optimizer  # 导入 PyTorch 移动端模型优化工具
from torch.testing._internal.common_utils import TestCase, run_tests, IS_WINDOWS, skipIfNoXNNPACK
from torch.testing._internal.common_quantized import supported_qengines  # 导入用于量化测试的相关功能

# 定义一个简单的神经网络模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(16, 64)  # 第一层线性变换，输入维度 16，输出维度 64
        self.relu1 = torch.nn.ReLU()  # 第一层激活函数 ReLU
        self.layer2 = torch.nn.Linear(64, 8)  # 第二层线性变换，输入维度 64，输出维度 8
        self.relu2 = torch.nn.ReLU()  # 第二层激活函数 ReLU

    def forward(self, features):
        act = features  # 将输入 features 赋值给变量 act
        act = self.layer1(act)  # 应用第一层线性变换
        act = self.relu1(act)  # 应用第一层 ReLU 激活函数
        act = self.layer2(act)  # 应用第二层线性变换
        act = self.relu2(act)  # 应用第二层 ReLU 激活函数
        return act  # 返回网络输出结果

# 定义一个量化后的模型
class QuantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()  # 定义量化的模型输入
        self.dequant = torch.ao.quantization.DeQuantStub()  # 定义量化的模型输出
        self.core = SimpleModel()  # 定义一个简单的神经网络模型作为核心

    def forward(self, x):
        x = self.quant(x)  # 对输入 x 进行量化
        x = self.core(x)  # 使用核心网络进行前向传播
        x = self.dequant(x)  # 对输出结果进行反量化
        return x  # 返回量化后的输出结果

# 定义一个包含列表的模型
class ModelWithLists(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rt = [torch.zeros(1)]  # 初始化包含一个元素的列表 rt
        self.ot = [torch.zeros(1), None]  # 初始化包含两个元素的列表 ot，其中第二个元素为 None

    def forward(self, arg):
        arg = arg + self.rt[0]  # 将输入 arg 与列表 rt 的第一个元素相加
        o = self.ot[0]  # 获取列表 ot 的第一个元素
        if o is not None:
            arg = arg + o  # 如果 o 不为 None，则将其与 arg 相加
        return arg  # 返回最终结果

# 用于包装 Webdriver 测试的装饰器函数
def webdriver_test(testfunc):
    @functools.wraps(testfunc)
    def wrapper(self, *args, **kwds):
        self.needs_resources()  # 执行需要的资源准备操作

        if os.environ.get("RUN_WEBDRIVER") != "1":  # 检查环境变量是否设置为运行 Webdriver
            self.skipTest("Webdriver not requested")  # 如果未请求运行 Webdriver，则跳过测试
        from selenium import webdriver  # 导入 Selenium 的 webdriver 接口

        for driver in [
                "Firefox",
                "Chrome",
        ]:
            with self.subTest(driver=driver):  # 创建一个子测试环境，用于不同的浏览器驱动
                wd = getattr(webdriver, driver)()  # 根据驱动名称创建相应的 webdriver
                testfunc(self, wd, *args, **kwds)  # 执行测试函数
                wd.close()  # 关闭 webdriver

    return wrapper  # 返回包装后的测试函数

# 测试模型保存工具的单元测试类
class TestModelDump(TestCase):
    def needs_resources(self):
        pass  # 占位函数，不执行任何操作

    def test_inline_skeleton(self):
        self.needs_resources()  # 执行需要的资源准备操作
        skel = torch.utils.model_dump.get_inline_skeleton()  # 获取内联骨架信息
        assert "unpkg.org" not in skel  # 断言确保内联骨架中不包含 "unpkg.org"
        assert "src=" not in skel  # 断言确保内联骨架中不包含 "src="

    def do_dump_model(self, model, extra_files=None):
        # 仅检查是否能够成功运行
        buf = io.BytesIO()  # 创建一个字节流缓冲区
        torch.jit.save(model, buf, _extra_files=extra_files)  # 将模型保存到字节流中，包括额外文件
        info = torch.utils.model_dump.get_model_info(buf)  # 获取模型信息
        assert info is not None  # 断言确保成功获取模型信息

    def open_html_model(self, wd, model, extra_files=None):
        buf = io.BytesIO()  # 创建一个字节流缓冲区
        torch.jit.save(model, buf, _extra_files=extra_files)  # 将模型保存到字节流中，包括额外文件
        page = torch.utils.model_dump.get_info_and_burn_skeleton(buf)  # 获取信息并生成骨架页面
        wd.get("data:text/html;charset=utf-8," + urllib.parse.quote(page))  # 打开 HTML 页面，并加载生成的骨架信息
    # 在指定的 Web 元素中查找具有特定 data-hider-title 属性值的容器
    container = wd.find_element_by_xpath(f"//div[@data-hider-title='{name}']")
    # 在找到的容器中查找具有特定类名 "caret" 的元素，用于控制展开或收起状态
    caret = container.find_element_by_class_name("caret")
    # 如果容器未显示（data-shown 属性不为 "true"），则点击 caret 元素展开内容
    if container.get_attribute("data-shown") != "true":
        caret.click()
    # 在容器中找到具体的内容元素，并返回该元素
    content = container.find_element_by_tag_name("div")
    # 返回获取到的内容元素
    return content

    # 创建一个使用 torch.jit.script 将 SimpleModel 脚本化后的模型，并调用 do_dump_model 方法进行处理
    model = torch.jit.script(SimpleModel())
    self.do_dump_model(model)

    # 创建一个使用 torch.jit.trace 将 SimpleModel 追踪化后的模型，并调用 do_dump_model 方法进行处理
    model = torch.jit.trace(SimpleModel(), torch.zeros(2, 16))
    self.do_dump_model(model)

    # 执行主测试函数，确保资源可用性，如果在 Windows 环境下，则跳过测试
    self.needs_resources()
    if IS_WINDOWS:
        self.skipTest("Disabled on Windows.")

    # 使用临时文件 tf 保存 torch.jit.script 将 SimpleModel 脚本化后的模型，并将其保存到磁盘
    with tempfile.NamedTemporaryFile() as tf:
        torch.jit.save(torch.jit.script(SimpleModel()), tf)
        # 刷新文件以确保内容写入磁盘
        tf.flush()

        # 创建一个 StringIO 对象 stdout，并调用 torch.utils.model_dump.main 函数将模型信息以 JSON 格式输出到 stdout
        stdout = io.StringIO()
        torch.utils.model_dump.main(
            [
                None,
                "--style=json",
                tf.name,
            ],
            stdout=stdout)
        # 使用正则表达式确保输出内容符合预期格式
        self.assertRegex(stdout.getvalue(), r'\A{.*SimpleModel')

        # 创建另一个 StringIO 对象 stdout，并调用 torch.utils.model_dump.main 函数将模型信息以 HTML 格式输出到 stdout
        stdout = io.StringIO()
        torch.utils.model_dump.main(
            [
                None,
                "--style=html",
                tf.name,
            ],
            stdout=stdout)
        # 使用正则表达式确保输出内容符合预期格式
        self.assertRegex(
            stdout.getvalue().replace("\n", " "),
            r'\A<!DOCTYPE.*SimpleModel.*componentDidMount')

    # 获取量化后的模型并返回
    fmodel = QuantModel().eval()
    fmodel = torch.ao.quantization.fuse_modules(fmodel, [
        ["core.layer1", "core.relu1"],
        ["core.layer2", "core.relu2"],
    ])
    fmodel.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
    prepped = torch.ao.quantization.prepare(fmodel)
    prepped(torch.randn(2, 16))
    qmodel = torch.ao.quantization.convert(prepped)
    return qmodel

    # 除非支持的引擎中包含 "qnnpack"，否则跳过此测试
    @unittest.skipUnless("qnnpack" in supported_qengines, "QNNPACK not available")
    def test_quantized_model(self):
        # 获取量化后的模型并调用 do_dump_model 方法进行处理
        qmodel = self.get_quant_model()
        self.do_dump_model(torch.jit.script(qmodel))

    # 除非支持的引擎中包含 "qnnpack"，否则跳过此测试
    @skipIfNoXNNPACK
    @unittest.skipUnless("qnnpack" in supported_qengines, "QNNPACK not available")
    def test_optimized_quantized_model(self):
        # 获取量化后的模型并追踪化，然后对其进行优化处理，并调用 do_dump_model 方法进行处理
        qmodel = self.get_quant_model()
        smodel = torch.jit.trace(qmodel, torch.zeros(2, 16))
        omodel = torch.utils.mobile_optimizer.optimize_for_mobile(smodel)
        self.do_dump_model(omodel)

    # 创建一个使用 torch.jit.script 将 ModelWithLists 脚本化后的模型，并调用 do_dump_model 方法进行处理
    model = torch.jit.script(ModelWithLists())
    self.do_dump_model(model)
    # 定义一个测试无效 JSON 的方法
    def test_invalid_json(self):
        # 使用 SimpleModel() 创建一个 TorchScript 模型，并将其转储，同时传入一个额外的文件 {"foo.json": "{"}
        model = torch.jit.script(SimpleModel())
        self.do_dump_model(model, extra_files={"foo.json": "{"})

    # 使用 WebDriver 进行测试的装饰器，测试内存计算功能
    @webdriver_test
    def test_memory_computation(self, wd):
        # 定义一个检查内存使用的嵌套函数，接受模型和预期内存用量作为参数
        def check_memory(model, expected):
            # 打开 HTML 模型，并在 WebDriver 中加载模型
            self.open_html_model(wd, model)
            # 打开“Tensor Memory”部分并获取其内容
            memory_table = self.open_section_and_get_body(wd, "Tensor Memory")
            # 查找并获取设备信息，预期为 CPU
            device = memory_table.find_element_by_xpath("//table/tbody/tr[1]/td[1]").text
            self.assertEqual("cpu", device)
            # 获取内存使用量的字符串表示
            memory_usage_str = memory_table.find_element_by_xpath("//table/tbody/tr[1]/td[2]").text
            # 断言预期的内存使用量与实际获取的内存使用量一致
            self.assertEqual(expected, int(memory_usage_str))

        # 计算简单模型的内存使用量
        simple_model_memory = (
            # 第一层，包括偏置
            64 * (16 + 1) +
            # 第二层，包括偏置
            8 * (64 + 1)
            # 32 位浮点数
        ) * 4

        # 检查简单模型的内存使用量
        check_memory(torch.jit.script(SimpleModel()), simple_model_memory)

        # 创建一个 SimpleModel 的实例，并检查其内存使用量
        a_simple_model = SimpleModel()
        # 将两个相同的 SimpleModel 实例放入序列中，确保张量共享，避免双重计数
        check_memory(
            torch.jit.script(
                torch.nn.Sequential(a_simple_model, a_simple_model)),
            simple_model_memory)

        # 冻结过程将权重和偏置从数据移至常量中，确保仍然计数
        check_memory(
            torch.jit.freeze(torch.jit.script(SimpleModel()).eval()),
            simple_model_memory)

        # 确保能够处理同时包含常量和数据张量的模型
        class ComposedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = torch.zeros(1, 2)
                self.w2 = torch.ones(2, 2)

            def forward(self, arg):
                return arg * self.w2 + self.w1

        # 冻结具有常量和数据张量的模型，并保留属性 "w1" 进行检查内存使用量
        check_memory(
            torch.jit.freeze(
                torch.jit.script(ComposedModule()).eval(),
                preserved_attrs=["w1"]),
            4 * (2 + 4))
# 如果当前脚本作为主程序执行（而不是被导入为模块），则执行以下代码块
if __name__ == '__main__':
    # 调用名为 run_tests 的函数，用于执行测试代码或功能
    run_tests()
```