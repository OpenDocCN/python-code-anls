# `basic-computer-games\00_Common\dotnet\Games.Common.Test\IO\TokenizerTests.cs`

```

// 引入所需的命名空间
using System.Linq;
using FluentAssertions;
using Xunit;

// 创建 TokenizerTests 类
namespace Games.Common.IO;

public class TokenizerTests
{
    // 使用 Theory 属性进行参数化测试
    [Theory]
    [MemberData(nameof(TokenizerTestCases))]
    // 测试方法，将输入字符串分割成预期的标记
    public void ParseTokens_SplitsStringIntoExpectedTokens(string input, string[] expected)
    {
        // 调用 Tokenizer 类的 ParseTokens 方法
        var result = Tokenizer.ParseTokens(input);

        // 使用 FluentAssertions 库进行断言
        result.Select(t => t.ToString()).Should().BeEquivalentTo(expected);
    }

    // 创建 TokenizerTestCases 方法，返回测试数据
    public static TheoryData<string, string[]> TokenizerTestCases() => new()
    {
        // 测试数据
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

```