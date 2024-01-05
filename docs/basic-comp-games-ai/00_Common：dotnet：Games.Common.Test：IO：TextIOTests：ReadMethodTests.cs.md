# `d:/src/tocomm/basic-computer-games\00_Common\dotnet\Games.Common.Test\IO\TextIOTests\ReadMethodTests.cs`

```
using System;  # 导入 System 模块
using System.Collections.Generic;  # 导入集合类相关模块
using System.IO;  # 导入文件操作相关模块
using FluentAssertions;  # 导入 FluentAssertions 模块
using FluentAssertions.Execution;  # 导入 FluentAssertions.Execution 模块
using Xunit;  # 导入 Xunit 模块

using TwoStrings = System.ValueTuple<string, string>;  # 定义别名 TwoStrings 为包含两个字符串的元组
using TwoNumbers = System.ValueTuple<float, float>;  # 定义别名 TwoNumbers 为包含两个浮点数的元组
using ThreeNumbers = System.ValueTuple<float, float, float>;  # 定义别名 ThreeNumbers 为包含三个浮点数的元组
using FourNumbers = System.ValueTuple<float, float, float, float>;  # 定义别名 FourNumbers 为包含四个浮点数的元组

using static System.Environment;  # 导入 System.Environment 模块的所有静态成员
using static Games.Common.IO.Strings;  # 导入 Games.Common.IO.Strings 模块的所有静态成员

namespace Games.Common.IO.TextIOTests;  # 定义命名空间 Games.Common.IO.TextIOTests

public class ReadMethodTests  # 定义类 ReadMethodTests
{
    [Theory]  # 标记下面的方法为测试理论的方法
    # 使用 MemberData 属性指定测试用例数据源 ReadStringTestCases
    # 使用 MemberData 属性指定测试用例数据源 Read2StringsTestCases
    # 使用 MemberData 属性指定测试用例数据源 ReadNumberTestCases
    # 使用 MemberData 属性指定测试用例数据源 Read2NumbersTestCases
    # 使用 MemberData 属性指定测试用例数据源 Read3NumbersTestCases
    # 使用 MemberData 属性指定测试用例数据源 Read4NumbersTestCases
    # 使用 MemberData 属性指定测试用例数据源 ReadNumbersTestCases
    # 定义一个名为 ReadingValuesHasExpectedPromptsAndResults 的测试方法，接受一个泛型参数 T
    # 该方法接受一个读取函数 read、输入字符串 input、期望输出字符串 expectedOutput、期望结果 expectedResult
    def ReadingValuesHasExpectedPromptsAndResults<T>(
        Func<IReadWrite, T> read,
        string input,
        string expectedOutput,
        T expectedResult)
    {
        # 创建一个 StringReader 对象，用于读取输入字符串
        var inputReader = new StringReader(input + Environment.NewLine);
        # 创建一个 StringWriter 对象，用于写入输出字符串
        var outputWriter = new StringWriter();
        # 创建一个 TextIO 对象，用于输入输出操作
        var io = new TextIO(inputReader, outputWriter);

        # 调用 read 函数，传入 io 对象，获取读取结果
        var result = read.Invoke(io);
        # 获取输出字符串
        var output = outputWriter.ToString();
    }
        using var _ = new AssertionScope();  # 使用 var _ = new AssertionScope() 创建一个断言作用域，用于管理断言的范围
        output.Should().Be(expectedOutput);  # 断言输出应该等于期望的输出
        result.Should().BeEquivalentTo(expectedResult);  # 断言结果应该等价于期望的结果
    }

    [Fact]  # 标记测试方法，表示这是一个测试方法
    public void ReadNumbers_ArrayEmpty_ThrowsArgumentException()  # 测试读取数字时，当数组为空时应该抛出参数异常
    {
        var io = new TextIO(new StringReader(""), new StringWriter());  # 创建一个 TextIO 对象，用于读取和写入文本

        Action readNumbers = () => io.ReadNumbers("foo", Array.Empty<float>());  # 定义一个读取数字的动作，传入一个空的数组

        readNumbers.Should().Throw<ArgumentException>()  # 断言 readNumbers 动作应该抛出参数异常
            .WithMessage("'values' must have a non-zero length.*")  # 断言异常消息应该包含指定的内容
            .WithParameterName("values");  # 断言异常的参数名应该是 "values"
    }

    public static TheoryData<Func<IReadWrite, string>, string, string, string> ReadStringTestCases()  # 定义一个静态方法，返回用于测试读取字符串的测试数据
    {
        static Func<IReadWrite, string> ReadString(string prompt) => io => io.ReadString(prompt);  # 定义一个静态方法，用于创建读取字符串的函数
        return new()
        {
            { ReadString("Name"), "", "Name? ", "" },  // 创建一个包含四个元素的元组，元素分别为 ReadString("Name") 的返回值、空字符串、"Name? "、空字符串
            { ReadString("prompt"), " foo  ,bar", $"prompt? {ExtraInput}{NewLine}", "foo" }  // 创建一个包含四个元素的元组，元素分别为 ReadString("prompt") 的返回值、" foo  ,bar"、$"prompt? {ExtraInput}{NewLine}"、"foo"
        };
    }

    public static TheoryData<Func<IReadWrite, TwoStrings>, string, string, TwoStrings> Read2StringsTestCases()
    {
        static Func<IReadWrite, TwoStrings> Read2Strings(string prompt) => io => io.Read2Strings(prompt);  // 定义一个名为 Read2Strings 的静态方法，接受一个字符串参数 prompt，返回一个 Func<IReadWrite, TwoStrings> 类型的函数

        return new()
        {
            { Read2Strings("2 strings"), ",", "2 strings? ", ("", "") },  // 创建一个包含四个元素的元组，元素分别为 Read2Strings("2 strings") 的返回值、","、"2 strings? "、一个包含两个空字符串的元组
            {
                Read2Strings("Input please"),
                $"{NewLine}x,y",
                $"Input please? ?? {ExtraInput}{NewLine}",
                ("", "x")  // 创建一个包含四个元素的元组，元素分别为 Read2Strings("Input please") 的返回值、一个包含换行符和"x,y"的字符串、$"Input please? ?? {ExtraInput}{NewLine}"、一个包含一个空字符串和"x"的元组
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
        {
            { Read2Numbers("Point"), "3,4,5", $"Point? {ExtraInput}{NewLine}", (3, 4) }, 
            # 创建一个包含测试用例的数据集，每个测试用例包括一个函数、输入字符串、期望输出字符串和期望结果
            {
                Read2Numbers("Foo"), 
                $"x,4,5{NewLine}4,5,x", 
                $"Foo? {NumberExpected}{NewLine}? {ExtraInput}{NewLine}", 
                (4, 5)
            }
        };
    }

    # 定义一个静态方法，返回一个包含测试用例的数据集
    public static TheoryData<Func<IReadWrite, ThreeNumbers>, string, string, ThreeNumbers> Read3NumbersTestCases()
    {
        # 定义一个内部静态方法，用于创建一个函数，该函数接受一个 IReadWrite 对象并调用其 Read3Numbers 方法
        static Func<IReadWrite, ThreeNumbers> Read3Numbers(string prompt) => io => io.Read3Numbers(prompt);

        # 返回一个包含测试用例的数据集
        return new()
        {
            { Read3Numbers("Point"), "3.2, 4.3, 5.4, 6.5", $"Point? {ExtraInput}{NewLine}", (3.2F, 4.3F, 5.4F) },
            {
                Read3Numbers("Bar"),
```
在这段代码中，我们定义了两个方法，一个是 `Read2NumbersTestCases`，另一个是 `Read3NumbersTestCases`。这两个方法返回一个包含测试用例的数据集，每个测试用例包括一个函数、输入字符串、期望输出字符串和期望结果。这些测试用例将用于测试程序中的某些功能。
                $"x,4,5{NewLine}4,5,x{NewLine}6,7,8,y", // 创建包含换行符的字符串
                $"Bar? {NumberExpected}{NewLine}? {NumberExpected}{NewLine}? {ExtraInput}{NewLine}", // 创建包含占位符的字符串
                (6, 7, 8) // 创建包含三个整数的元组
            }
        };
    }

    public static TheoryData<Func<IReadWrite, FourNumbers>, string, string, FourNumbers> Read4NumbersTestCases()
    {
        static Func<IReadWrite, FourNumbers> Read4Numbers(string prompt) => io => io.Read4Numbers(prompt); // 创建一个函数，接受 IReadWrite 接口和 FourNumbers 类型的参数

        return new()
        {
            { Read4Numbers("Point"), "3,4,5,6,7", $"Point? {ExtraInput}{NewLine}", (3, 4, 5, 6) }, // 创建包含函数、字符串、字符串和元组的元组
            {
                Read4Numbers("Baz"),
                $"x,4,5,6{NewLine} 4, 5 , 6,7  ,x", // 创建包含换行符的字符串
                $"Baz? {NumberExpected}{NewLine}? {ExtraInput}{NewLine}", // 创建包含占位符的字符串
                (4, 5, 6, 7) // 创建包含四个整数的元组
            }
        };
    }
```

这部分代码是一个静态方法，用于生成测试用例数据。它返回一个TheoryData对象，其中包含了多组测试用例数据。

```
    public static TheoryData<Func<IReadWrite, IReadOnlyList<float>>, string, string, float[]> ReadNumbersTestCases()
```

这是一个静态方法的声明，返回类型为TheoryData。它接受一个IReadWrite接口类型的函数、两个字符串和一个float数组作为参数。

```
        static Func<IReadWrite, IReadOnlyList<float>> ReadNumbers(string prompt) =>
            io =>
            {
                var numbers = new float[6];
                io.ReadNumbers(prompt, numbers);
                return numbers;
            };
```

这是一个嵌套的静态方法，用于生成一个接受IReadWrite类型参数并返回IReadOnlyList<float>类型的函数。它接受一个字符串作为参数，返回一个lambda表达式。

```
        return new()
        {
            { ReadNumbers("Primes"), "2, 3, 5, 7, 11, 13", $"Primes? ", new float[] { 2, 3, 5, 7, 11, 13 } },
            {
                ReadNumbers("Qux"),
                $"42{NewLine}3.141, 2.718{NewLine}3.0e8, 6.02e23{NewLine}9.11E-28",
                $"Qux? ?? ?? ?? ",
```

这部分代码创建了一个新的TheoryData对象，并添加了多组测试用例数据。每组数据包含了一个函数、两个字符串和一个float数组。
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，并封装成字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```