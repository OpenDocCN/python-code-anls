# `.\AutoGPT\benchmark\agbenchmark\challenges\deprecated\code\2_write\artifacts_out\test.py`

```py
# 从 sample_code 模块中导入 multiply_int 函数
from sample_code import multiply_int

# 定义一个测试函数，用于测试 multiply_int 函数的输出是否符合预期
def test_multiply_int(num: int, expected_result: int) -> None:
    # 调用 multiply_int 函数，传入参数 num
    result = multiply_int(num)
    # 打印结果
    print(result)
    # 使用断言检查结果是否符合预期
    assert (
        result == expected_result
    ), f"AssertionError: Expected the output to be {expected_result}"

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 测试一个简单的情况
    num = 4
    expected_result = 8
    # 调用测试函数，传入参数 num 和 expected_result
    test_multiply_int(num, expected_result)
```