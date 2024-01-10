# `basic-computer-games\00_Common\dotnet\Games.Common.Test\IO\TokenTests.cs`

```
// 引入 FluentAssertions 和 Xunit 库
using FluentAssertions;
using Xunit;

// 定义 TokenTests 类
namespace Games.Common.IO
{
    // 定义 Ctor_PopulatesProperties 方法，用于测试 Token 构造函数
    public class TokenTests
    {
        // 使用 Theory 属性进行参数化测试，调用 TokenTestCases 方法
        [Theory]
        [MemberData(nameof(TokenTestCases))]
        public void Ctor_PopulatesProperties(string value, bool isNumber, float number)
        {
            // 创建期望的匿名对象
            var expected = new { String = value, IsNumber = isNumber, Number = number };

            // 调用 Token 构造函数
            var token = new Token(value);

            // 使用 FluentAssertions 验证 token 是否等价于 expected
            token.Should().BeEquivalentTo(expected);
        }

        // 定义 TokenTestCases 方法，返回 TheoryData 对象
        public static TheoryData<string, bool, float> TokenTestCases() => new()
        {
            // 测试用例
            { "", false, float.NaN },
            { "abcde", false, float.NaN },
            { "123  ", true, 123 },
            { "+42  ", true, 42 },
            // 更多测试用例...
        };
    }
}
```