# `basic-computer-games\00_Common\dotnet\Games.Common.Test\IO\TextIOTests\ReadMethodTests.cs`

```
// 引入所需的命名空间
using System;
using System.Collections.Generic;
using System.IO;
using FluentAssertions;
using FluentAssertions.Execution;
using Xunit;

// 定义别名
using TwoStrings = System.ValueTuple<string, string>;
using TwoNumbers = System.ValueTuple<float, float>;
using ThreeNumbers = System.ValueTuple<float, float, float>;
using FourNumbers = System.ValueTuple<float, float, float, float>;

// 使用静态类的静态成员
using static System.Environment;
using static Games.Common.IO.Strings;

// 定义命名空间和测试类
namespace Games.Common.IO.TextIOTests;

public class ReadMethodTests
{
    // 测试方法，使用不同的测试数据
    [Theory]
    [MemberData(nameof(ReadStringTestCases))]
    [MemberData(nameof(Read2StringsTestCases))]
    [MemberData(nameof(ReadNumberTestCases))]
    [MemberData(nameof(Read2NumbersTestCases))]
    [MemberData(nameof(Read3NumbersTestCases))]
    [MemberData(nameof(Read4NumbersTestCases))]
    [MemberData(nameof(ReadNumbersTestCases))]
    public void ReadingValuesHasExpectedPromptsAndResults<T>(
        Func<IReadWrite, T> read,
        string input,
        string expectedOutput,
        T expectedResult)
    {
        // 创建输入和输出的读写器
        var inputReader = new StringReader(input + Environment.NewLine);
        var outputWriter = new StringWriter();
        var io = new TextIO(inputReader, outputWriter);

        // 调用读取方法
        var result = read.Invoke(io);
        var output = outputWriter.ToString();

        // 使用 AssertionScope 进行断言
        using var _ = new AssertionScope();
        output.Should().Be(expectedOutput);
        result.Should().BeEquivalentTo(expectedResult);
    }

    // 测试读取数字时，当数组为空时抛出异常
    [Fact]
    public void ReadNumbers_ArrayEmpty_ThrowsArgumentException()
    {
        var io = new TextIO(new StringReader(""), new StringWriter());

        // 调用读取数字方法并断言抛出异常
        Action readNumbers = () => io.ReadNumbers("foo", Array.Empty<float>());

        readNumbers.Should().Throw<ArgumentException>()
            .WithMessage("'values' must have a non-zero length.*")
            .WithParameterName("values");
    }

    // 定义测试数据
    public static TheoryData<Func<IReadWrite, string>, string, string, string> ReadStringTestCases()
    {
        // 定义一个静态方法，用于读取字符串，返回一个接受 IReadWrite 参数的函数
        static Func<IReadWrite, string> ReadString(string prompt) => io => io.ReadString(prompt);
    
        // 返回一个新的 TheoryData 对象
        return new()
        {
            // 调用 ReadString 方法，传入参数 "Name"，并设置默认值为 ""，提示为 "Name? "，默认输入为 ""
            { ReadString("Name"), "", "Name? ", "" },
            // 调用 ReadString 方法，传入参数 "prompt"，并设置默认值为 " foo  ,bar"，提示为 "prompt? "，默认输入为 "foo"
            { ReadString("prompt"), " foo  ,bar", $"prompt? {ExtraInput}{NewLine}", "foo" }
        };
    }
    
    // 定义一个静态方法，用于生成读取两个字符串的测试用例数据
    public static TheoryData<Func<IReadWrite, TwoStrings>, string, string, TwoStrings> Read2StringsTestCases()
    {
        // 定义一个静态方法，用于读取两个字符串，返回一个接受 IReadWrite 参数的函数
        static Func<IReadWrite, TwoStrings> Read2Strings(string prompt) => io => io.Read2Strings(prompt);
    
        // 返回一个新的 TheoryData 对象
        return new()
        {
            // 调用 Read2Strings 方法，传入参数 "2 strings"，分隔符为 ","，提示为 "2 strings? "，默认输入为 ("", "")
            { Read2Strings("2 strings"), ",", "2 strings? ", ("", "") },
            {
                // 调用 Read2Strings 方法，传入参数 "Input please"，设置默认输入为 ("", "x")
                Read2Strings("Input please"),
                // 设置额外的输入为换行符和 "x,y"
                $"{NewLine}x,y",
                // 设置提示为 "Input please? ?? "，默认输入为 ("", "x")
                $"Input please? ?? {ExtraInput}{NewLine}",
                ("", "x")
            }
        };
    }
    
    // 定义一个静态方法，用于生成读取数字的测试用例数据
    public static TheoryData<Func<IReadWrite, float>, string, string, float> ReadNumberTestCases()
    {
        // 定义一个静态方法，用于读取数字，返回一个接受 IReadWrite 参数的函数
        static Func<IReadWrite, float> ReadNumber(string prompt) => io => io.ReadNumber(prompt);
    
        // 返回一个新的 TheoryData 对象
        return new()
        {
            // 调用 ReadNumber 方法，传入参数 "Age"，设置默认输入为 42
            { ReadNumber("Age"), $"{NewLine}42,", $"Age? {NumberExpected}{NewLine}? {ExtraInput}{NewLine}", 42 },
            // 调用 ReadNumber 方法，传入参数 "Guess"，设置默认输入为 3
            { ReadNumber("Guess"), "3,4,5", $"Guess? {ExtraInput}{NewLine}", 3 }
        };
    }
    
    // 定义一个静态方法，用于生成读取两个数字的测试用例数据
    public static TheoryData<Func<IReadWrite, TwoNumbers>, string, string, TwoNumbers> Read2NumbersTestCases()
    {
        // 定义一个静态方法，用于读取两个数字，返回一个接受 IReadWrite 参数的函数
        static Func<IReadWrite, TwoNumbers> Read2Numbers(string prompt) => io => io.Read2Numbers(prompt);
    
        // 返回一个新的 TheoryData 对象
        return new()
        {
            // 调用 Read2Numbers 方法，传入参数 "Point"，设置默认输入为 (3, 4)
            { Read2Numbers("Point"), "3,4,5", $"Point? {ExtraInput}{NewLine}", (3, 4) },
            {
                // 调用 Read2Numbers 方法，传入参数 "Foo"，设置默认输入为 (4, 5)
                Read2Numbers("Foo"),
                // 设置额外的输入为换行符和 "x,4,5"，"4,5,x"
                $"x,4,5{NewLine}4,5,x",
                // 设置提示为 "Foo? "，默认输入为 (4, 5)
                $"Foo? {NumberExpected}{NewLine}? {ExtraInput}{NewLine}",
                (4, 5)
            }
        };
    }
    
    // 定义一个静态方法，用于生成读取三个数字的测试用例数据
    public static TheoryData<Func<IReadWrite, ThreeNumbers>, string, string, ThreeNumbers> Read3NumbersTestCases()
    {
        // 定义一个静态方法，用于读取三个数字，返回一个函数
        static Func<IReadWrite, ThreeNumbers> Read3Numbers(string prompt) => io => io.Read3Numbers(prompt);
    
        // 返回一个新的 TheoryData 对象
        return new()
        {
            // 调用 Read3Numbers 方法，传入参数和预期结果，构成一个测试用例
            { Read3Numbers("Point"), "3.2, 4.3, 5.4, 6.5", $"Point? {ExtraInput}{NewLine}", (3.2F, 4.3F, 5.4F) },
            {
                // 调用 Read3Numbers 方法，传入参数和预期结果，构成另一个测试用例
                Read3Numbers("Bar"),
                $"x,4,5{NewLine}4,5,x{NewLine}6,7,8,y",
                $"Bar? {NumberExpected}{NewLine}? {NumberExpected}{NewLine}? {ExtraInput}{NewLine}",
                (6, 7, 8)
            }
        };
    }
    
    // 定义一个静态方法，用于生成包含四个数字的测试用例
    public static TheoryData<Func<IReadWrite, FourNumbers>, string, string, FourNumbers> Read4NumbersTestCases()
    {
        // 定义一个静态方法，用于读取四个数字，返回一个函数
        static Func<IReadWrite, FourNumbers> Read4Numbers(string prompt) => io => io.Read4Numbers(prompt);
    
        // 返回一个新的 TheoryData 对象
        return new()
        {
            // 调用 Read4Numbers 方法，传入参数和预期结果，构成一个测试用例
            { Read4Numbers("Point"), "3,4,5,6,7", $"Point? {ExtraInput}{NewLine}", (3, 4, 5, 6) },
            {
                // 调用 Read4Numbers 方法，传入参数和预期结果，构成另一个测试用例
                Read4Numbers("Baz"),
                $"x,4,5,6{NewLine} 4, 5 , 6,7  ,x",
                $"Baz? {NumberExpected}{NewLine}? {ExtraInput}{NewLine}",
                (4, 5, 6, 7)
            }
        };
    }
    
    // 定义一个静态方法，用于生成包含一组浮点数的测试用例
    public static TheoryData<Func<IReadWrite, IReadOnlyList<float>>, string, string, float[]> ReadNumbersTestCases()
    {
        // 定义一个静态方法，用于读取一组浮点数，返回一个函数
        static Func<IReadWrite, IReadOnlyList<float>> ReadNumbers(string prompt) =>
            io =>
            {
                // 创建一个包含六个浮点数的数组
                var numbers = new float[6];
                // 调用 io.ReadNumbers 方法，将输入的浮点数存入数组中
                io.ReadNumbers(prompt, numbers);
                return numbers;
            };
    
        // 返回一个新的 TheoryData 对象
        return new()
        {
            // 调用 ReadNumbers 方法，传入参数和预期结果，构成一个测试用例
            { ReadNumbers("Primes"), "2, 3, 5, 7, 11, 13", $"Primes? ", new float[] { 2, 3, 5, 7, 11, 13 } },
            {
                // 调用 ReadNumbers 方法，传入参数和预期结果，构成另一个测试用例
                ReadNumbers("Qux"),
                $"42{NewLine}3.141, 2.718{NewLine}3.0e8, 6.02e23{NewLine}9.11E-28",
                $"Qux? ?? ?? ?? ",
                new[] { 42, 3.141F, 2.718F, 3.0e8F, 6.02e23F, 9.11E-28F }
            }
        };
    }
# 闭合前面的函数定义
```