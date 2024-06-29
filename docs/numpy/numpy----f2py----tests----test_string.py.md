# `.\numpy\numpy\f2py\tests\test_string.py`

```
import os
import pytest
import textwrap
import numpy as np
from . import util  # 导入自定义的 util 模块

class TestString(util.F2PyTest):
    sources = [util.getpath("tests", "src", "string", "char.f90")]  # 定义测试源文件路径列表

    @pytest.mark.slow
    def test_char(self):
        strings = np.array(["ab", "cd", "ef"], dtype="c").T  # 创建包含字符数组的 numpy 数组
        inp, out = self.module.char_test.change_strings(  # 调用模块方法，修改输入和输出字符串
            strings, strings.shape[1])
        assert inp == pytest.approx(strings)  # 断言输入是否与预期相符
        expected = strings.copy()  # 复制输入字符串数组作为预期输出的基础
        expected[1, :] = "AAA"  # 修改预期输出的第二行为 "AAA"
        assert out == pytest.approx(expected)  # 断言输出是否与预期相符

class TestDocStringArguments(util.F2PyTest):
    sources = [util.getpath("tests", "src", "string", "string.f")]  # 定义测试源文件路径列表

    def test_example(self):
        a = np.array(b"123\0\0")  # 创建包含字节串的 numpy 数组 a
        b = np.array(b"123\0\0")  # 创建包含字节串的 numpy 数组 b
        c = np.array(b"123")  # 创建包含字节串的 numpy 数组 c
        d = np.array(b"123")  # 创建包含字节串的 numpy 数组 d

        self.module.foo(a, b, c, d)  # 调用模块方法 foo 处理这些数组

        assert a.tobytes() == b"123\0\0"  # 断言数组 a 是否被正确处理
        assert b.tobytes() == b"B23\0\0"  # 断言数组 b 是否被正确处理
        assert c.tobytes() == b"123"  # 断言数组 c 是否被正确处理
        assert d.tobytes() == b"D23"  # 断言数组 d 是否被正确处理

class TestFixedString(util.F2PyTest):
    sources = [util.getpath("tests", "src", "string", "fixed_string.f90")]  # 定义测试源文件路径列表

    @staticmethod
    def _sint(s, start=0, end=None):
        """Return the content of a string buffer as integer value.

        For example:
          _sint('1234') -> 4321
          _sint('123A') -> 17321
        """
        if isinstance(s, np.ndarray):  # 如果 s 是 numpy 数组
            s = s.tobytes()  # 将其转换为字节串
        elif isinstance(s, str):  # 如果 s 是字符串
            s = s.encode()  # 将其编码为字节串
        assert isinstance(s, bytes)  # 断言 s 确实是字节串
        if end is None:
            end = len(s)
        i = 0
        for j in range(start, min(end, len(s))):  # 遍历字节串的一部分
            i += s[j] * 10**j  # 将字节串的每个字符乘以相应的权值并求和，构成整数
        return i  # 返回整数结果

    def _get_input(self, intent="in"):
        if intent in ["in"]:  # 如果意图是输入
            yield ""  # 返回空字符串
            yield "1"  # 返回字符串 "1"
            yield "1234"  # 返回字符串 "1234"
            yield "12345"  # 返回字符串 "12345"
            yield b""  # 返回空字节串
            yield b"\0"  # 返回含有一个空字符的字节串
            yield b"1"  # 返回含有字符 '1' 的字节串
            yield b"\01"  # 返回含有字符 '\x01' 的字节串
            yield b"1\0"  # 返回含有字符 '1' 和一个空字符的字节串
            yield b"1234"  # 返回字节串 "1234"
            yield b"12345"  # 返回字节串 "12345"
        yield np.ndarray((), np.bytes_, buffer=b"")  # 返回一个空的 numpy 字节串数组
        yield np.array(b"")  # 返回一个空的 numpy 字节串数组
        yield np.array(b"\0")  # 返回一个含有一个空字符的 numpy 字节串数组
        yield np.array(b"1")  # 返回一个含有字符 '1' 的 numpy 字节串数组
        yield np.array(b"1\0")  # 返回一个含有字符 '1' 和一个空字符的 numpy 字节串数组
        yield np.array(b"\01")  # 返回一个含有字符 '\x01' 的 numpy 字节串数组
        yield np.array(b"1234")  # 返回一个含有字节串 "1234" 的 numpy 数组
        yield np.array(b"123\0")  # 返回一个含有字节串 "123\0" 的 numpy 数组
        yield np.array(b"12345")  # 返回一个含有字节串 "12345" 的 numpy 数组

    def test_intent_in(self):
        for s in self._get_input():
            r = self.module.test_in_bytes4(s)  # 调用模块方法，处理输入的字节串 s
            # also checks that s is not changed inplace
            expected = self._sint(s, end=4)  # 计算预期的整数值
            assert r == expected, s  # 断言处理结果与预期整数值相符

    def test_intent_inout(self):
        for s in self._get_input(intent="inout"):
            rest = self._sint(s, start=4)  # 获取字节串 s 的后四个字符的整数值
            r = self.module.test_inout_bytes4(s)  # 调用模块方法，处理输入的字节串 s
            expected = self._sint(s, end=4)  # 计算预期的整数值
            assert r == expected  # 断言处理结果与预期整数值相符

            # check that the rest of input string is preserved
            assert rest == self._sint(s, start=4)  # 断言字节串 s 的后四个字符的整数值与之前一致
```