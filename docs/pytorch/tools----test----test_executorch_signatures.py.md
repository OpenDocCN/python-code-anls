# `.\pytorch\tools\test\test_executorch_signatures.py`

```
# 导入单元测试模块
import unittest

# 从特定路径导入相关模块和类
from torchgen.executorch.api.types import ExecutorchCppSignature
from torchgen.local import parametrize
from torchgen.model import Location, NativeFunction

# 从 YAML 格式加载默认的本地函数定义，获取 DEFAULT_NATIVE_FUNCTION 变量
DEFAULT_NATIVE_FUNCTION, _ = NativeFunction.from_yaml(
    {"func": "foo.out(Tensor input, *, Tensor(a!) out) -> Tensor(a!)"},
    loc=Location(__file__, 1),  # 设置位置信息为当前文件的第一行
    valid_tags=set(),  # 使用空集合作为有效标签的占位符
)

# 定义单元测试类 ExecutorchCppSignatureTest，继承自 unittest.TestCase
class ExecutorchCppSignatureTest(unittest.TestCase):
    
    # 设置测试环境的初始化方法
    def setUp(self) -> None:
        self.sig = ExecutorchCppSignature.from_native_function(DEFAULT_NATIVE_FUNCTION)
    
    # 测试 runtime 签名包含运行时上下文的情况
    def test_runtime_signature_contains_runtime_context(self) -> None:
        with parametrize(
            use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
        ):
            # 获取包含上下文的参数列表
            args = self.sig.arguments(include_context=True)
            self.assertEqual(len(args), 3)  # 断言参数个数为 3
            self.assertTrue(any(a.name == "context" for a in args))  # 断言参数中包含名为 "context" 的参数
    
    # 测试 runtime 签名不包含运行时上下文的情况
    def test_runtime_signature_does_not_contain_runtime_context(self) -> None:
        with parametrize(
            use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
        ):
            # 获取不包含上下文的参数列表
            args = self.sig.arguments(include_context=False)
            self.assertEqual(len(args), 2)  # 断言参数个数为 2
            self.assertFalse(any(a.name == "context" for a in args))  # 断言参数中不包含名为 "context" 的参数
    
    # 测试 runtime 签名声明的正确性
    def test_runtime_signature_declaration_correct(self) -> None:
        with parametrize(
            use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
        ):
            # 获取包含上下文的签名声明
            decl = self.sig.decl(include_context=True)
            self.assertEqual(
                decl,
                (
                    "torch::executor::Tensor & foo_outf("
                    "torch::executor::KernelRuntimeContext & context, "  # 函数声明部分
                    "const torch::executor::Tensor & input, "  # 函数参数部分
                    "torch::executor::Tensor & out)"  # 函数返回部分
                ),
            )
            # 获取不包含上下文的签名声明
            no_context_decl = self.sig.decl(include_context=False)
            self.assertEqual(
                no_context_decl,
                (
                    "torch::executor::Tensor & foo_outf("
                    "const torch::executor::Tensor & input, "  # 函数参数部分
                    "torch::executor::Tensor & out)"  # 函数返回部分
                ),
            )
```