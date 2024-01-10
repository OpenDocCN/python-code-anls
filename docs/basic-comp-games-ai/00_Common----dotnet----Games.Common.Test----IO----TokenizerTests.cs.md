# `basic-computer-games\00_Common\dotnet\Games.Common.Test\IO\TokenizerTests.cs`

```
// 引入 LINQ 和 FluentAssertions 库
using System.Linq;
using FluentAssertions;
using Xunit;

// 命名空间 Games.Common.IO
namespace Games.Common.IO
{
    // TokenizerTests 类
    public class TokenizerTests
    {
        // 使用 Theory 属性的数据成员来测试 Tokenizer 的不同情况
        [Theory]
        [MemberData(nameof(TokenizerTestCases))]
        // 测试方法：将输入字符串分割成预期的标记
        public void ParseTokens_SplitsStringIntoExpectedTokens(string input, string[] expected)
        {
            // 调用 Tokenizer 的 ParseTokens 方法
            var result = Tokenizer.ParseTokens(input);

            // 使用 LINQ 选择每个标记并转换为字符串，然后使用 FluentAssertions 库进行断言
            result.Select(t => t.ToString()).Should().BeEquivalentTo(expected);
        }

        // 静态方法：返回 Tokenizer 的测试用例数据
        public static TheoryData<string, string[]> TokenizerTestCases() => new()
        {
            { "", new[] { "" } },
            { "aBc", new[] { "aBc" } },
            { "  Foo   ", new[] { "Foo" } },
            { "  \" Foo  \"  ", new[] { " Foo  " } },
            { "  \" Foo    ", new[] { " Foo    " } },
            { "\"\"abc", new[] { "" } },
            { "a\"\"bc", new[] { "a\"\"bc" } },
            { "\"\"", new[] { "" } },
            { ",", new[] { "", "" } },
            { " foo  ,bar", new[] { "foo", "bar" } },
            { "\"a\"bc,de", new[] { "a" } },
            { "a\"b,\" c,d\", f ,,g", new[] { "a\"b", " c,d", "f", "", "g" } }
        };
    }
}
```