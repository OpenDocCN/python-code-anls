# `D:\src\scipysrc\sympy\sympy\assumptions\tests\test_context.py`

```
# 从 sympy.assumptions 模块导入 ask、Q
# 导入 sympy.assumptions.assume 模块中的 assuming、global_assumptions
# 从 sympy.abc 模块导入 x、y 符号变量

# 定义测试函数 test_assuming，用于测试 assuming 上下文管理器的功能
def test_assuming():
    # 在假设 x 是整数的情况下进行测试
    with assuming(Q.integer(x)):
        # 断言 x 是否被认定为整数
        assert ask(Q.integer(x))
    # 断言 x 是否被认定为整数（这里应为不是整数，因为上下文管理器已结束）
    assert not ask(Q.integer(x))

# 定义测试函数 test_assuming_nested，用于测试嵌套的 assuming 上下文管理器功能
def test_assuming_nested():
    # 断言 x 和 y 都不是整数
    assert not ask(Q.integer(x))
    assert not ask(Q.integer(y))
    # 在假设 x 是整数的情况下进行测试
    with assuming(Q.integer(x)):
        # 断言 x 被认定为整数
        assert ask(Q.integer(x))
        # 断言 y 不是整数（因为上下文管理器仅应用于 x）
        assert not ask(Q.integer(y))
        # 在假设 y 是整数的情况下进行测试
        with assuming(Q.integer(y)):
            # 断言 x 和 y 都被认定为整数
            assert ask(Q.integer(x))
            assert ask(Q.integer(y))
        # 断言 x 仍然被认定为整数，但是 y 不是整数（因为上一个上下文管理器已结束）
        assert ask(Q.integer(x))
        assert not ask(Q.integer(y))
    # 断言 x 和 y 都不是整数
    assert not ask(Q.integer(x))
    assert not ask(Q.integer(y))

# 定义测试函数 test_finally，用于测试 finally 块在异常处理中的行为
def test_finally():
    try:
        # 在假设 x 是整数的情况下，尝试除以 0（会引发 ZeroDivisionError 异常）
        with assuming(Q.integer(x)):
            1/0
    except ZeroDivisionError:
        pass
    # 断言 x 不被认定为整数，因为异常处理中断了上下文管理器
    assert not ask(Q.integer(x))

# 定义测试函数 test_remove_safe，用于测试全局假设的添加和移除
def test_remove_safe():
    # 将全局假设添加为 x 是整数
    global_assumptions.add(Q.integer(x))
    with assuming():
        # 在空的假设下，断言 x 是整数
        assert ask(Q.integer(x))
        # 移除全局假设中的 x 是整数的断言
        global_assumptions.remove(Q.integer(x))
        # 断言 x 不是整数
        assert not ask(Q.integer(x))
    # 再次断言 x 是整数，因为全局假设恢复为默认状态
    assert ask(Q.integer(x))
    # 清空全局假设，以便其他测试的受益
    global_assumptions.clear()
```