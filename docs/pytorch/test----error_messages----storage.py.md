# `.\pytorch\test\error_messages\storage.py`

```py
# 导入 PyTorch 库
import torch


# 检查错误的函数，用于验证特定情况下是否会引发异常
def check_error(desc, fn, *required_substrings):
    try:
        # 调用传入的函数 fn()，期望它会抛出异常
        fn()
    except Exception as e:
        # 捕获异常并提取异常消息
        error_message = e.args[0]
        # 打印分隔线和描述信息
        print("=" * 80)
        print(desc)
        print("-" * 80)
        # 打印异常消息
        print(error_message)
        print("")
        # 验证异常消息中是否包含所有必需的子字符串
        for sub in required_substrings:
            assert sub in error_message
        return
    # 如果没有抛出异常，则抛出断言错误
    raise AssertionError(f"given function ({desc}) didn't raise an error")


# 检查错误：错误的参数类型，传入 torch.FloatStorage 的对象类型不正确
check_error("Wrong argument types", lambda: torch.FloatStorage(object()), "object")

# 检查错误：未知的关键字参数传递给 torch.FloatStorage
check_error(
    "Unknown keyword argument", lambda: torch.FloatStorage(content=1234.0), "keyword"
)

# 检查错误：在序列内传递了无效类型给 torch.FloatStorage
check_error(
    "Invalid types inside a sequence",
    lambda: torch.FloatStorage(["a", "b"]),
    "list",
    "str",
)

# 检查错误：传递给 torch.FloatStorage 的大小参数类型不正确
check_error("Invalid size type", lambda: torch.FloatStorage(1.5), "float")

# 检查错误：传递给 torch.FloatStorage 的偏移量参数不正确
check_error(
    "Invalid offset", lambda: torch.FloatStorage(torch.FloatStorage(2), 4), "2", "4"
)

# 检查错误：传递给 torch.FloatStorage 的偏移量为负数
check_error(
    "Negative offset", lambda: torch.FloatStorage(torch.FloatStorage(2), -1), "2", "-1"
)

# 检查错误：传递给 torch.FloatStorage 的大小参数不正确
check_error(
    "Invalid size",
    lambda: torch.FloatStorage(torch.FloatStorage(3), 1, 5),
    "2",
    "1",
    "5",
)

# 检查错误：传递给 torch.FloatStorage 的大小为负数
check_error(
    "Negative size",
    lambda: torch.FloatStorage(torch.FloatStorage(3), 1, -5),
    "2",
    "1",
    "-5",
)

# 检查错误：对 torch.FloatStorage 进行索引时使用了无效的索引类型
check_error("Invalid index type", lambda: torch.FloatStorage(10)["first item"], "str")


# 尝试对 torch.FloatStorage 进行切片赋值操作，值类型不正确
def assign():
    torch.FloatStorage(10)[1:-1] = "1"


check_error("Invalid value type", assign, "str")

# 检查错误：尝试对 torch.FloatStorage 调用 resize_ 方法，大小参数类型不正确
check_error(
    "resize_ with invalid type", lambda: torch.FloatStorage(10).resize_(1.5), "float"
)

# 检查错误：尝试对 torch.IntStorage 调用 fill_ 方法，填充值类型不正确
check_error(
    "fill_ with invalid type", lambda: torch.IntStorage(10).fill_("asdf"), "str"
)

# TODO: frombuffer
```