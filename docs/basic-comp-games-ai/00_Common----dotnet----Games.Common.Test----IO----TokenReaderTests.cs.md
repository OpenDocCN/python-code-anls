# `basic-computer-games\00_Common\dotnet\Games.Common.Test\IO\TokenReaderTests.cs`

```
// 引入所需的命名空间
using System;
using System.IO;
using System.Linq;
using FluentAssertions;
using FluentAssertions.Execution;
using Xunit;

// 使用静态导入简化代码
using static System.Environment;
using static Games.Common.IO.Strings;

// 声明命名空间
namespace Games.Common.IO
{
    // 定义 TokenReaderTests 类
    public class TokenReaderTests
    {
        // 声明私有字段 _outputWriter
        private readonly StringWriter _outputWriter;

        // TokenReaderTests 类的构造函数
        public TokenReaderTests()
        {
            // 初始化 _outputWriter 字段
            _outputWriter = new StringWriter();
        }

        // 定义测试方法 ReadTokens_QuantityNeededZero_ThrowsArgumentException
        [Fact]
        public void ReadTokens_QuantityNeededZero_ThrowsArgumentException()
        {
            // 创建 TokenReader 对象
            var sut = TokenReader.ForStrings(new TextIO(new StringReader(""), _outputWriter));

            // 定义并执行读取令牌的操作
            Action readTokens = () => sut.ReadTokens("", 0);

            // 验证读取操作是否抛出指定的异常
            readTokens.Should().Throw<ArgumentOutOfRangeException>()
                .WithMessage("'quantityNeeded' must be greater than zero.*")
                .WithParameterName("quantityNeeded");
        }

        // 定义测试方法 ReadTokens_ReadingValuesHasExpectedPromptsAndResults
        [Theory]
        [MemberData(nameof(ReadTokensTestCases))]
        public void ReadTokens_ReadingValuesHasExpectedPromptsAndResults(
            string prompt,
            uint tokenCount,
            string input,
            string expectedOutput,
            string[] expectedResult)
        {
            // 创建 TokenReader 对象
            var sut = TokenReader.ForStrings(new TextIO(new StringReader(input + NewLine), _outputWriter));

            // 执行读取令牌的操作
            var result = sut.ReadTokens(prompt, tokenCount);
            var output = _outputWriter.ToString();

            // 使用 AssertionScope 进行断言
            using var _ = new AssertionScope();
            output.Should().Be(expectedOutput);
            result.Select(t => t.String).Should().BeEquivalentTo(expectedResult);
        }

        // 定义测试方法 ReadTokens_Numeric_ReadingValuesHasExpectedPromptsAndResults
        [Theory]
        [MemberData(nameof(ReadNumericTokensTestCases))]
        public void ReadTokens_Numeric_ReadingValuesHasExpectedPromptsAndResults(
            string prompt,
            uint tokenCount,
            string input,
            string expectedOutput,
            float[] expectedResult)
    {
        // 创建 TokenReader 对象，用于读取数字类型的输入
        var sut = TokenReader.ForNumbers(new TextIO(new StringReader(input + NewLine), _outputWriter));
    
        // 读取指定数量的 token，并将输出写入 output 变量
        var result = sut.ReadTokens(prompt, tokenCount);
        var output = _outputWriter.ToString();
    
        // 使用 AssertionScope 进行断言
        using var _ = new AssertionScope();
        // 断言输出应该等于期望的输出
        output.Should().Be(expectedOutput);
        // 断言结果中的数字应该等同于期望的结果
        result.Select(t => t.Number).Should().BeEquivalentTo(expectedResult);
    }
    
    // 生成测试用例数据，包括输入、期望的输出和结果
    public static TheoryData<string, uint, string, string, string[]> ReadTokensTestCases()
    {
        return new()
        {
            { "Name", 1, "Bill", "Name? ", new[] { "Bill" } },
            { "Names", 2, " Bill , Bloggs ", "Names? ", new[] { "Bill", "Bloggs" } },
            { "Names", 2, $" Bill{NewLine}Bloggs ", "Names? ?? ", new[] { "Bill", "Bloggs" } },
            {
                "Foo",
                6,
                $"1,2{NewLine}\" a,b \"{NewLine},\"\"c,d{NewLine}d\"x,e,f",
                $"Foo? ?? ?? ?? {ExtraInput}{NewLine}",
                new[] { "1", "2", " a,b ", "", "", "d\"x" }
            }
        };
    }
    
    // 生成读取数字类型 token 的测试用例数据
    public static TheoryData<string, uint, string, string, float[]> ReadNumericTokensTestCases()
    {
        return new()
        {
            { "Age", 1, "23", "Age? ", new[] { 23F } },
            { "Constants", 2, " 3.141 , 2.71 ", "Constants? ", new[] { 3.141F, 2.71F } },
            { "Answer", 1, $"Forty-two{NewLine}42 ", $"Answer? {NumberExpected}{NewLine}? ", new[] { 42F } },
            {
                "Foo",
                6,
                $"1,2{NewLine}\" a,b \"{NewLine}3, 4  {NewLine}5.6,7,a, b",
                $"Foo? ?? {NumberExpected}{NewLine}? ?? {ExtraInput}{NewLine}",
                new[] { 1, 2, 3, 4, 5.6F, 7 }
            }
        };
    }
# 闭合前面的函数定义
```