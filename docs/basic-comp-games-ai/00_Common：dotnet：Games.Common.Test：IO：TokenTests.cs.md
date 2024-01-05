# `d:/src/tocomm/basic-computer-games\00_Common\dotnet\Games.Common.Test\IO\TokenTests.cs`

```
    # 使用 FluentAssertions 库进行断言
    using FluentAssertions;
    # 使用 Xunit 进行单元测试
    using Xunit;

    # 声明 TokenTests 类
    namespace Games.Common.IO;

    public class TokenTests
    {
        # 使用 Theory 属性进行参数化测试
        [Theory]
        # 使用 MemberData 属性引用 TokenTestCases 方法提供的测试数据
        [MemberData(nameof(TokenTestCases))]
        # 测试 Token 类的构造函数是否正确填充属性
        public void Ctor_PopulatesProperties(string value, bool isNumber, float number)
        {
            # 创建期望的匿名对象
            var expected = new { String = value, IsNumber = isNumber, Number = number };

            # 调用 Token 类的构造函数
            var token = new Token(value);

            # 使用 FluentAssertions 库进行断言，判断 token 对象是否与期望值相等
            token.Should().BeEquivalentTo(expected);
        }

        # 提供 Token 类构造函数测试数据的方法
        public static TheoryData<string, bool, float> TokenTestCases() => new()
        {
```
```python
        # 在这里添加测试数据
        }
    }
```
```python
```

这段代码是一个 C# 的单元测试代码，使用了 FluentAssertions 和 Xunit 库进行断言和单元测试。其中包括了一个 TokenTests 类，以及一个测试 Token 类构造函数的方法。在方法中使用了 Theory 属性进行参数化测试，并引用了 TokenTestCases 方法提供的测试数据。在 TokenTestCases 方法中提供了测试数据。
        { "", false, float.NaN }, // 空字符串，不是有效的数字，返回 false 和 NaN
        { "abcde", false, float.NaN }, // 字母字符串，不是有效的数字，返回 false 和 NaN
        { "123  ", true, 123 }, // 去除空格后是有效的整数，返回 true 和 123
        { "+42  ", true, 42 }, // 去除空格后是有效的整数，返回 true 和 42
        { "-42  ", true, -42 }, // 去除空格后是有效的整数，返回 true 和 -42
        { "+3.14159  ", true, 3.14159F }, // 去除空格后是有效的浮点数，返回 true 和 3.14159
        { "-3.14159  ", true, -3.14159F }, // 去除空格后是有效的浮点数，返回 true 和 -3.14159
        { "   123", false, float.NaN }, // 开头有空格，不是有效的数字，返回 false 和 NaN
        { "1.2e4", true, 12000 }, // 科学计数法表示的有效数字，返回 true 和 12000
        { "2.3e-5", true, 0.000023F }, // 科学计数法表示的有效数字，返回 true 和 0.000023
        { "1e100", true, float.MaxValue }, // 超出浮点数范围的有效数字，返回 true 和 float.MaxValue
        { "-1E100", true, float.MinValue }, // 超出浮点数范围的有效数字，返回 true 和 float.MinValue
        { "1E-100", true, 0 }, // 科学计数法表示的有效数字，返回 true 和 0
        { "-1e-100", true, 0 }, // 科学计数法表示的有效数字，返回 true 和 0
        { "100abc", true, 100 }, // 数字后面有字母，返回 true 和 100
        { "1,2,3", true, 1 }, // 逗号分隔的数字，返回 true 和 1
        { "42,a,b", true, 42 }, // 逗号分隔的数字，返回 true 和 42
        { "1.2.3", true, 1.2F }, // 有多个小数点，返回 true 和 1.2
        { "12e.5", false, float.NaN }, // 科学计数法格式错误，返回 false 和 NaN
        { "12e0.5", true, 12 } // 科学计数法表示的有效数字，返回 true 和 12
抱歉，这段代码看起来不完整，缺少了一些关键信息，无法为其添加注释。
```