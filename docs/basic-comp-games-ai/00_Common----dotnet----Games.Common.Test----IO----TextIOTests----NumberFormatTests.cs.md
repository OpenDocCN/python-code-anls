# `basic-computer-games\00_Common\dotnet\Games.Common.Test\IO\TextIOTests\NumberFormatTests.cs`

```py
// 引入必要的命名空间
using System;
using System.IO;
using FluentAssertions;
using Xunit;

// 定义 NumberFormatTests 类
namespace Games.Common.IO.TextIOTests
{
    public class NumberFormatTests
    {
        // 定义测试用例，验证 Write 方法是否能正确格式化浮点数
        [Theory]
        [MemberData(nameof(WriteFloatTestCases))]
        public void Write_Float_FormatsNumberSameAsBasic(float value, string basicString)
        {
            // 创建 StringWriter 对象
            var outputWriter = new StringWriter();
            // 创建 TextIO 对象
            var io = new TextIO(new StringReader(""), outputWriter);

            // 调用 Write 方法
            io.Write(value);

            // 验证输出结果是否与基本字符串相等
            outputWriter.ToString().Should().BeEquivalentTo(basicString);
        }

        // 定义测试用例，验证 WriteLine 方法是否能正确格式化浮点数
        [Theory]
        [MemberData(nameof(WriteFloatTestCases))]
        public void WriteLine_Float_FormatsNumberSameAsBasic(float value, string basicString)
        {
            // 创建 StringWriter 对象
            var outputWriter = new StringWriter();
            // 创建 TextIO 对象
            var io = new TextIO(new StringReader(""), outputWriter);

            // 调用 WriteLine 方法
            io.WriteLine(value);

            // 验证输出结果是否与基本字符串加上换行符相等
            outputWriter.ToString().Should().BeEquivalentTo(basicString + Environment.NewLine);
        }

        // 定义测试用例数据，包含浮点数和对应的基本字符串
        public static TheoryData<float, string> WriteFloatTestCases()
            => new()
            {
                { 1000F, " 1000 " },
                { 3.1415927F, " 3.1415927 " },
                { 1F, " 1 " },
                { 0F, " 0 " },
                { -1F, "-1 " },
                { -3.1415927F, "-3.1415927 " },
                { -1000F, "-1000 " },
            };

        // 定义测试用例，验证 Write 方法是否能正确格式化整数
        [Theory]
        [MemberData(nameof(WriteIntTestCases))]
        public void Write_Int_FormatsNumberSameAsBasic(int value, string basicString)
        {
            // 创建 StringWriter 对象
            var outputWriter = new StringWriter();
            // 创建 TextIO 对象
            var io = new TextIO(new StringReader(""), outputWriter);

            // 调用 Write 方法
            io.Write(value);

            // 验证输出结果是否与基本字符串相等
            outputWriter.ToString().Should().BeEquivalentTo(basicString);
        }

        // 定义测试用例，验证 WriteLine 方法是否能正确格式化整数
        [Theory]
        [MemberData(nameof(WriteIntTestCases))]
        public void WriteLine_Int_FormatsNumberSameAsBasic(int value, string basicString)
        {
            // 创建 StringWriter 对象
            var outputWriter = new StringWriter();
            // 创建 TextIO 对象
            var io = new TextIO(new StringReader(""), outputWriter);

            // 调用 WriteLine 方法
            io.WriteLine(value);

            // 验证输出结果是否与基本字符串加上换行符相等
            outputWriter.ToString().Should().BeEquivalentTo(basicString + Environment.NewLine);
        }
    // 创建一个包含整数和对应字符串表示的测试数据集合
    public static TheoryData<int, string> WriteIntTestCases()
        => new()
        {
            // 添加整数 1000 和对应的字符串 " 1000 "
            { 1000, " 1000 " },
            // 添加整数 1 和对应的字符串 " 1 "
            { 1, " 1 " },
            // 添加整数 0 和对应的字符串 " 0 "
            { 0, " 0 " },
            // 添加整数 -1 和对应的字符串 "-1 "
            { -1, "-1 " },
            // 添加整数 -1000 和对应的字符串 "-1000 "
            { -1000, "-1000 " },
        };
# 闭合前面的函数定义
```