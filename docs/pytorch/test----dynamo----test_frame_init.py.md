# `.\pytorch\test\dynamo\test_frame_init.py`

```
# Owner(s): ["module: dynamo"]

# 导入 torch 库
import torch
# 导入 dynamo 模块下的测试用例模块
import torch._dynamo.test_case

# 使用位置参数、位置-关键字参数、关键字参数和可变关键字参数定义的函数
def target_with_varkwargs(arg1, /, positional_only_arg, *, keyword_only_arg, **kwargs):
    # 定义一个本地变量 local 并赋值为 1
    local = 1
    # 返回包含各参数值和本地变量值的字典
    return {
        "local": local,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "keyword_only_arg": keyword_only_arg,
        "kwargs": kwargs,
    }


# 使用位置参数、位置-关键字参数、关键字参数和可变关键字参数定义的函数，移除了本地变量 local 的赋值
def varkwargs_code1(arg1, /, positional_only_arg, *, keyword_only_arg, **kwargs):
    # 返回包含各参数值和直接赋值的本地变量 1 的字典
    return {
        "local": 1,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "keyword_only_arg": keyword_only_arg,
        "kwargs": kwargs,
    }


# 使用位置参数、位置-关键字参数、关键字参数和可变关键字参数定义的函数，引入了两个本地变量
def varkwargs_code2(arg1, /, positional_only_arg, *, keyword_only_arg, **kwargs):
    # 定义两个本地变量 local1 和 local2，并分别赋值为 0 和 1
    local1 = 0
    local2 = 1
    # 返回包含各参数值和本地变量之和的字典
    return {
        "local": local1 + local2,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "keyword_only_arg": keyword_only_arg,
        "kwargs": kwargs,
    }


# 使用位置参数、位置-关键字参数、关键字参数和可变位置参数定义的函数
def target_with_varargs(arg1, /, positional_only_arg, *varargs, **kwargs):
    # 定义一个本地变量 local 并赋值为 1
    local = 1
    # 返回包含各参数值、可变位置参数和可变关键字参数的字典
    return {
        "local": local,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "varargs": varargs,
        "kwargs": kwargs,
    }


# 使用位置参数、位置-关键字参数、关键字参数和可变位置参数定义的函数，移除了本地变量 local 的赋值
def varargs_code1(arg1, /, positional_only_arg, *varargs, **kwargs):
    # 返回包含各参数值、可变位置参数和可变关键字参数，以及直接赋值的本地变量 1 的字典
    return {
        "local": 1,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "varargs": varargs,
        "kwargs": kwargs,
    }


# 使用位置参数、位置-关键字参数、关键字参数和可变位置参数定义的函数，引入了两个本地变量
def varargs_code2(arg1, /, positional_only_arg, *varargs, **kwargs):
    # 定义两个本地变量 local1 和 local2，并分别赋值为 0 和 1
    local1 = 0
    local2 = 1
    # 返回包含各参数值、可变位置参数、可变关键字参数和本地变量之和的字典
    return {
        "local": local1 + local2,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "varargs": varargs,
        "kwargs": kwargs,
    }


# 定义一个测试类 FrameInitTests，继承自 torch._dynamo.test_case.TestCase
class FrameInitTests(torch._dynamo.test_case.TestCase):
    # 定义测试函数 test_frame_init，用于测试帧初始化
    def test_frame_init(self):
        # 定义两个字典，映射到函数代码对象的转换后的代码对象
        code_map1 = {
            target_with_varargs.__code__: varargs_code1.__code__,
            target_with_varkwargs.__code__: varkwargs_code1.__code__,
        }
        code_map2 = {
            target_with_varargs.__code__: varargs_code2.__code__,
            target_with_varkwargs.__code__: varkwargs_code2.__code__,
        }

        # 定义回调函数 callback1，根据帧的代码对象在 code_map1 中查找匹配项，并返回转换后的代码对象
        def callback1(frame, cache_entry, frame_state):
            if frame.f_code in code_map1:
                transformed_code = code_map1[frame.f_code]
                # 返回转换后的代码对象和一个始终为真的 lambda 函数
                return torch._dynamo.types.GuardedCode(
                    transformed_code, lambda f_locals: True
                )
            # 如果没有匹配项，返回 None
            return None

        # 定义回调函数 callback2，根据帧的代码对象在 code_map2 中查找匹配项，并返回转换后的代码对象
        def callback2(frame, cache_entry, frame_state):
            if frame.f_code in code_map2:
                transformed_code = code_map2[frame.f_code]
                # 返回转换后的代码对象和一个始终为真的 lambda 函数
                return torch._dynamo.types.GuardedCode(
                    transformed_code, lambda f_locals: True
                )
            # 如果没有匹配项，返回 None
            return None

        # 对每个回调函数进行循环测试
        for callback in [callback1, callback2]:
            # 重置 torch._dynamo 的状态
            torch._dynamo.reset()
            
            # 测试带可变位置参数的目标函数，获取预期输出
            expected_varargs_output = target_with_varargs(
                1, 2, 3, 4, name1=1, name2=2, name3=3
            )
            # 测试带可变关键字参数的目标函数，获取预期输出
            expected_kwargs_output = target_with_varkwargs(
                1, 2, keyword_only_arg=1, name2=2, name3=3
            )
            
            # 设置当前回调函数为评估帧的回调函数，并保存原始设置
            original = torch._dynamo.eval_frame.set_eval_frame(callback1)
            
            # 执行带可变位置参数的目标函数，获取实际输出
            real_varargs_output = target_with_varargs(
                1, 2, 3, 4, name1=1, name2=2, name3=3
            )
            # 执行带可变关键字参数的目标函数，获取实际输出
            real_kwargs_output = target_with_varkwargs(
                1, 2, keyword_only_arg=1, name2=2, name3=3
            )
            
            # 使用断言检查实际输出与预期输出是否一致
            self.assertEqual(real_varargs_output, expected_varargs_output)
            self.assertEqual(real_kwargs_output, expected_kwargs_output)
            
            # 恢复原始的帧评估设置
            torch._dynamo.eval_frame.set_eval_frame(original)
# 如果当前脚本作为主程序执行（而不是被导入），则执行以下代码块
if __name__ == "__main__":
    # 导入 torch._dynamo.test_case 模块中的 run_tests 函数
    from torch._dynamo.test_case import run_tests

    # 运行测试用例
    run_tests()
```