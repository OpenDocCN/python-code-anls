# `.\pytorch\test\dynamo\test_model_output.py`

```
# Owner(s): ["module: dynamo"]

# 导入必要的模块和类
import dataclasses
import unittest.mock

import torch

# 导入自定义的测试模块和函数
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import same

# 尝试导入transformers相关模块，处理ImportError异常
try:
    from transformers import modeling_outputs
    from transformers.configuration_utils import PretrainedConfig
    from transformers.file_utils import ModelOutput
    from transformers.modeling_outputs import (
        BaseModelOutput,
        BaseModelOutputWithPastAndCrossAttentions,
        BaseModelOutputWithPoolingAndCrossAttentions,
        CausalLMOutputWithPast,
    )
except ImportError:
    modeling_outputs = None


# 定义一个装饰器函数maybe_skip，根据transformers模块的可用性决定是否跳过测试
def maybe_skip(fn):
    if modeling_outputs is None:
        return unittest.skip("requires HuggingFace")(fn)
    return fn


# 定义一个测试类TestHFPretrained，继承自torch._dynamo.test_case.TestCase
class TestHFPretrained(torch._dynamo.test_case.TestCase):
    
    # 使用maybe_skip装饰器修饰test_pretrained方法，根据transformers模块的可用性决定是否跳过测试
    @maybe_skip
    def test_pretrained(self):
        # 定义一个内部函数fn，根据tmp对象的属性进行不同的计算
        def fn(a, tmp):
            if hasattr(tmp, "somekey"):
                a = a + 1
            if tmp.return_dict:
                return a + torch.ones(2) * tmp.max_length
            return a
        
        # 生成一个随机张量x
        x = torch.randn(2)
        # 创建一个PretrainedConfig对象tmp
        tmp = PretrainedConfig(return_dict=True, max_length=20)
        # 调用fn函数得到ref变量
        ref = fn(x, tmp)
        # 使用torch._dynamo.optimize函数优化fn函数得到opt_fn函数
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 调用优化后的函数opt_fn得到res变量
        res = opt_fn(x, tmp)
        # 使用assertTrue断言ref和res变量相同
        self.assertTrue(same(ref, res))


# 定义一个测试类TestModelOutput，继承自torch._dynamo.test_case.TestCase
class TestModelOutput(torch._dynamo.test_case.TestCase):
    
    # 使用maybe_skip装饰器修饰test_mo_create方法，根据transformers模块的可用性决定是否跳过测试
    @maybe_skip
    def test_mo_create(self):
        # 定义一个内部函数fn，创建一个BaseModelOutput对象tmp并返回
        def fn(a, b):
            tmp = BaseModelOutput(a + 1, attentions=b + 3)
            return tmp
        
        # 调用torch._dynamo.testing.standard_test函数进行标准化测试
        torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=2)
    
    # 使用maybe_skip装饰器修饰test_mo_assign方法，根据transformers模块的可用性决定是否跳过测试
    @maybe_skip
    def test_mo_assign(self):
        # 定义一个内部函数fn，创建一个BaseModelOutput对象tmp并设置其属性值，最后返回tmp对象
        def fn(a, b):
            tmp = BaseModelOutput(last_hidden_state=b + 3)
            tmp.hidden_states = a + 7
            tmp["attentions"] = a + b + 6
            return tmp
        
        # 创建一个参数列表args，包含两个随机张量
        args = [torch.randn(10), torch.randn(10)]
        # 调用fn函数得到obj1对象
        obj1 = fn(*args)
        
        # 创建一个CompileCounter对象cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用torch._dynamo.optimize_assert函数优化fn函数得到opt_fn函数
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        # 调用优化后的函数opt_fn得到obj2对象
        obj2 = opt_fn(*args)
        
        # 使用assertTrue断言obj1和obj2的特定属性相同
        self.assertTrue(same(obj1.last_hidden_state, obj2.last_hidden_state))
        self.assertTrue(same(obj1.hidden_states, obj2.hidden_states))
        self.assertTrue(same(obj1.attentions, obj2.attentions))
        # 使用assertEqual断言cnts对象的frame_count属性为1，op_count属性为4
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    # 定义一个内部方法_common，用于测试通用情况下的函数fn
    def _common(self, fn, op_count):
        # 创建一个参数列表args，包含一个BaseModelOutput对象
        args = [
            BaseModelOutput(
                last_hidden_state=torch.randn(10), attentions=torch.randn(10)
            )
        ]
        # 调用fn函数得到obj1对象
        obj1 = fn(*args)
        
        # 创建一个CompileCounter对象cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用torch._dynamo.optimize_assert函数优化fn函数得到opt_fn函数
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        # 调用优化后的函数opt_fn得到obj2对象
        obj2 = opt_fn(*args)
        
        # 使用assertTrue断言obj1和obj2相同
        self.assertTrue(same(obj1, obj2))
        # 使用assertEqual断言cnts对象的frame_count属性为1，op_count属性为op_count参数的值
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, op_count)

    # 使用maybe_skip装饰器修饰测试方法
    @maybe_skip
    def test_mo_pwc(self):
        pass  # Placeholder for future tests
    def test_mo_getattr(self):
        # 定义一个函数 fn，接受一个 BaseModelOutput 对象作为参数
        def fn(obj: BaseModelOutput):
            # 计算 obj.last_hidden_state 的值乘以 10
            x = obj.last_hidden_state * 10
            # 如果 obj.hidden_states 不为 None，则将其加到 x 上
            if obj.hidden_states is not None:
                x += obj.hidden_states
            # 如果 obj.attentions 不为 None，则将其加到 x 上
            if obj.attentions is not None:
                x += obj.attentions
            # 返回计算后的结果 x
            return x

        # 调用 self._common 方法，传入 fn 函数和参数 2
        self._common(fn, 2)

    @maybe_skip
    def test_mo_getattr_missing(self):
        # 定义一个函数 fn，接受一个 BaseModelOutput 对象作为参数
        def fn(obj: BaseModelOutput):
            # 如果 obj 中存在名为 "asdf" 的属性，则将其加 1
            if getattr(obj, "asdf", None) is not None:
                obj.asdf += 1
            # 返回 obj.attentions 的值加 1
            return obj.attentions + 1

        # 调用 self._common 方法，传入 fn 函数和参数 1
        self._common(fn, 1)

    @maybe_skip
    def test_mo_getitem(self):
        # 定义一个函数 fn，接受一个 BaseModelOutput 对象作为参数
        def fn(obj: BaseModelOutput):
            # 计算 obj["last_hidden_state"] 的值乘以 10
            x = obj["last_hidden_state"] * 10
            # 如果 obj 中存在键名为 "hidden_stats" 的项，则将其加到 x 上
            if "hidden_stats" in obj:
                x += obj["hidden_states"]
            # 如果 obj 中存在键名为 "attentions" 的项，则将其加到 x 上
            if "attentions" in obj:
                x += obj["attentions"]
            # 返回计算后的结果 x
            return x

        # 调用 self._common 方法，传入 fn 函数和参数 2
        self._common(fn, 2)

    @maybe_skip
    def test_mo_tuple(self):
        # 定义一个函数 fn，接受一个 BaseModelOutput 对象作为参数
        def fn(obj: BaseModelOutput):
            # 调用 obj.to_tuple() 方法，将返回的元组解包为变量 a 和 b
            a, b = obj.to_tuple()
            # 返回 a 加上 b 乘以 10 的结果
            return a + b * 10

        # 调用 self._common 方法，传入 fn 函数和参数 2
        self._common(fn, 2)

    @maybe_skip
    def test_mo_index(self):
        # 定义一个函数 fn，接受一个 BaseModelOutput 对象作为参数
        def fn(obj: BaseModelOutput):
            # 返回 obj 中索引为 0 的元素乘以 10 加上索引为 1 的元素的值
            return obj[0] * 10 + obj[1]

        # 调用 self._common 方法，传入 fn 函数和参数 2
        self._common(fn, 2)

    @maybe_skip
    def test_mo_init(self):
        # 定义一个数据类 MyDataClass，继承自 ModelOutput
        @dataclasses.dataclass
        class MyDataClass(ModelOutput):
            # 声明五个属性，类型为 torch.Tensor，其中 b 到 e 的默认值为 None
            a: torch.Tensor
            b: torch.Tensor = None
            c: torch.Tensor = None
            d: torch.Tensor = None
            e: torch.Tensor = None

        # 定义一个函数 fn，接受一个 MyDataClass 对象作为参数
        def fn(obj):
            # 获取 MyDataClass 类的所有字段
            class_fields = dataclasses.fields(obj)
            # 断言字段数量大于 0
            assert len(class_fields)
            # 断言除了第一个字段外，其它字段的默认值均为 None
            assert all(field.default is None for field in class_fields[1:])
            # 判断除了第一个字段外，其它字段是否全部为 None
            other_fields_are_none = all(
                getattr(obj, field.name) is None for field in class_fields[1:]
            )
            # 断言其它字段不全为 None
            assert not other_fields_are_none

            # 初始化 total 为第一个字段的值
            total = getattr(obj, class_fields[0].name)
            # 遍历除第一个字段外的每个字段，将其值加到 total 上
            for field in class_fields[1:]:
                v = getattr(obj, field.name)
                if v is not None:
                    total += v

            # 返回计算后的 total
            return total

        # 生成三个随机的 torch.Tensor 对象的列表
        tensors = [torch.randn(10), torch.randn(10), torch.randn(10)]
        # 使用 tensors 创建一个 MyDataClass 对象 obj1
        obj1 = MyDataClass(*tensors)
        # 调用 fn 函数，计算 obj1 的结果
        correct1 = fn(obj1)

        # 使用 tensors 再次创建一个 MyDataClass 对象 obj2
        obj2 = MyDataClass(*tensors)
        # 使用 torch._dynamo.testing.CompileCounter 对 fn 进行优化
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        # 断言优化后的函数 opt_fn 对 obj2 的计算结果与 correct1 相同
        self.assertTrue(same(opt_fn(obj2), correct1))
        # 断言 cnts 的 frame_count 为 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言 cnts 的 op_count 为 2
        self.assertEqual(cnts.op_count, 2)
    def test_mo_init2(self):
        # 定义测试函数，验证自定义的 ModelOutput 子类在不同 __post_init__ 路径下的行为
        @dataclasses.dataclass
        class MyDataClass(ModelOutput):
            x: torch.FloatTensor = None

        # 定义测试函数 fn，创建 MyDataClass 实例并返回
        def fn(x):
            obj = MyDataClass(x=x)
            return obj

        # 生成输入数据 inp
        inp = torch.randn(3, 3)
        # 使用 torch._dynamo.optimize 进行优化
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 断言未优化和优化后的结果相等
        self.assertEqual(fn(inp).x, opt_fn(inp).x)

    @maybe_skip
    def test_mo_init_with_disable(self):
        # 可能导致 "non-function or method super: <slot wrapper '__setattr__' of 'object' objects>" 的图形破裂
        # 最小重现 https://github.com/pytorch/pytorch/issues/126028 的示例
        @dataclasses.dataclass
        class MyDataClass(ModelOutput):
            x: torch.FloatTensor = None

        # 使用 torch._dynamo.disable 禁用某些优化
        @torch._dynamo.disable(recursive=False)
        def fn(x):
            return MyDataClass(x=x)

        # 生成输入数据 inp
        inp = torch.randn(3, 3)
        # 使用 torch._dynamo.optimize 进行优化
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 断言未优化和优化后的结果相等
        self.assertEqual(fn(inp).x, opt_fn(inp).x)

    @maybe_skip
    def test_mo_newkey(self):
        # 创建 BaseModelOutput 实例
        obj = BaseModelOutput()

        # 定义测试函数 fn，返回对象中特定键的值加一
        def fn(obj):
            return obj["wwww"] + 1

        # 生成输入数据 inp
        inp = torch.randn(3, 3)
        # 向对象中添加键值对 "wwww": inp
        obj["wwww"] = inp
        # 使用 torch._dynamo.optimize 进行优化
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 断言未优化和优化后的结果相等
        self.assertEqual(fn(obj), opt_fn(obj))

    @maybe_skip
    def test_mo_from_outside(self):
        # 定义测试函数 fn，返回对象中的 attentions 属性加一
        def fn(obj):
            return obj.attentions + 1

        # 创建 BaseModelOutput 实例，初始化 attentions 属性
        obj = BaseModelOutput(attentions=torch.randn(3, 3))
        # 使用 torch._dynamo.optimize 进行优化
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 断言未优化和优化后的结果相等
        self.assertEqual(fn(obj), opt_fn(obj))

    @maybe_skip
    def test_mo_reconstruct_bytecode(self):
        # 定义测试函数 fn，返回对象中 attentions 属性加一后的结果
        def fn(inp):
            return BaseModelOutput(attentions=inp + 1)

        # 生成输入数据 inp
        inp = torch.randn(3, 3)
        # 使用 torch._dynamo.optimize 进行优化
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 断言未优化和优化后的结果相等
        self.assertEqual(fn(inp).attentions, opt_fn(inp).attentions)

    @maybe_skip
    @maybe_skip
    def test_none(self):
        # 定义模型类 Model，实现 forward 方法
        class Model(torch.nn.Module):
            def forward(self, x):
                # 对输入 x 加一并返回结果
                x = x + 1
                # 创建 CausalLMOutputWithPast 实例，loss 设置为 None，logits 设置为 x
                return CausalLMOutputWithPast(loss=None, logits=x)[0]

        # 创建 Model 实例
        model = Model()
        # 使用 torch.compile 进行模型编译优化
        opt_model = torch.compile(model, backend="eager", fullgraph=True)
        # 生成输入数据 x
        x = torch.randn(1, 1, 1, 1)
        # 断言未优化和优化后的结果相同
        self.assertTrue(same(model(x), opt_model(x)))

    @maybe_skip
    def test_reconstruction(self):
        # 定义模型类 Model，实现 forward 方法
        class Model(torch.nn.Module):
            def forward(self, x):
                # 对输入 x 加一并返回结果
                x = x + 1
                # 创建 CausalLMOutputWithPast 实例，loss 设置为 x，logits 设置为 None
                return CausalLMOutputWithPast(loss=x, logits=None)

        # 创建 Model 实例
        model = Model()
        # 生成输入数据 x
        x = torch.randn(1, 1, 1, 1)
        # 使用 torch._dynamo.export 导出模型的计算图
        eo = torch._dynamo.export(Model(), aten_graph=True)(x)
        # 断言模型的输出结果和导出后的图模块对输入 x 的计算结果相同
        self.assertTrue(same(model(x), eo.graph_module(x)))
if __name__ == "__main__":
    # 检查当前模块是否作为主程序运行
    from torch._dynamo.test_case import run_tests
    # 导入测试用例运行函数

    # 执行测试用例
    run_tests()
```