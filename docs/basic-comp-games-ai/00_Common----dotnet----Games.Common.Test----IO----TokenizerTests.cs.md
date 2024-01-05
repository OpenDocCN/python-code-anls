# `00_Common\dotnet\Games.Common.Test\IO\TokenizerTests.cs`

```
    using System.Linq;  # 导入 System.Linq 模块，用于对集合进行查询和操作
    using FluentAssertions;  # 导入 FluentAssertions 模块，用于编写更具表现力的断言
    using Xunit;  # 导入 Xunit 模块，用于编写单元测试

    namespace Games.Common.IO;  # 定义命名空间 Games.Common.IO

    public class TokenizerTests  # 定义 TokenizerTests 类
    {
        [Theory]  # 标记测试方法为理论测试
        [MemberData(nameof(TokenizerTestCases))]  # 使用 TokenizerTestCases 方法提供的数据作为测试参数
        public void ParseTokens_SplitsStringIntoExpectedTokens(string input, string[] expected)  # 定义测试方法，测试解析字符串是否得到预期的标记
        {
            var result = Tokenizer.ParseTokens(input);  # 调用 Tokenizer 类的 ParseTokens 方法解析输入字符串

            result.Select(t => t.ToString()).Should().BeEquivalentTo(expected);  # 对解析结果进行查询和断言，判断是否与预期结果相等
        }

        public static TheoryData<string, string[]> TokenizerTestCases() => new()  # 定义提供测试数据的方法 TokenizerTestCases
        {
            { "", new[] { "" } },  # 提供空字符串和空字符串数组作为测试数据
```
```python
        # 继续添加其他测试数据
        # ...
    }
}
        { "aBc", new[] { "aBc" } }, // 创建一个键为"aBc"，值为包含"aBc"的数组的条目
        { "  Foo   ", new[] { "Foo" } }, // 创建一个键为"  Foo   "，值为包含"Foo"的数组的条目
        { "  \" Foo  \"  ", new[] { " Foo  " } }, // 创建一个键为"  \" Foo  \"  "，值为包含" Foo  "的数组的条目
        { "  \" Foo    ", new[] { " Foo    " } }, // 创建一个键为"  \" Foo    "，值为包含" Foo    "的数组的条目
        { "\"\"abc", new[] { "" } }, // 创建一个键为"\"\"abc"，值为包含""的数组的条目
        { "a\"\"bc", new[] { "a\"\"bc" } }, // 创建一个键为"a\"\"bc"，值为包含"a\"\"bc"的数组的条目
        { "\"\"", new[] { "" } }, // 创建一个键为"\"\""，值为包含""的数组的条目
        { ",", new[] { "", "" } }, // 创建一个键为","，值为包含""和""的数组的条目
        { " foo  ,bar", new[] { "foo", "bar" } }, // 创建一个键为" foo  ,bar"，值为包含"foo"和"bar"的数组的条目
        { "\"a\"bc,de", new[] { "a" } }, // 创建一个键为"\"a\"bc,de"，值为包含"a"的数组的条目
        { "a\"b,\" c,d\", f ,,g", new[] { "a\"b", " c,d", "f", "", "g" } } // 创建一个键为"a\"b,\" c,d\", f ,,g"，值为包含"a\"b", " c,d", "f", "", "g"的数组的条目
    };
}
```