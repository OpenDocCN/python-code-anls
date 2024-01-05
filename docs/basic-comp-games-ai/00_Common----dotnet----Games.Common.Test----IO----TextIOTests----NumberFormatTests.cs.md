# `00_Common\dotnet\Games.Common.Test\IO\TextIOTests\NumberFormatTests.cs`

```
# 导入所需的模块
import System
import IO
import FluentAssertions
import Xunit

# 定义测试类
class NumberFormatTests:
    # 使用理论测试数据
    @Theory
    @MemberData(nameof(WriteFloatTestCases))
    # 测试写入浮点数时的格式化是否与基本格式相同
    def Write_Float_FormatsNumberSameAsBasic(value, basicString):
        # 创建一个字符串写入器
        outputWriter = StringWriter()
        # 创建一个 TextIO 对象，使用空字符串作为输入，outputWriter 作为输出
        io = TextIO(StringReader(""), outputWriter)

        # 调用 TextIO 对象的 Write 方法，将浮点数写入
        io.Write(value)

        # 断言输出的字符串应该等同于基本字符串
        outputWriter.ToString().Should().BeEquivalentTo(basicString)
    [Theory]  # 标记下面的方法为测试理论
    [MemberData(nameof(WriteFloatTestCases))]  # 使用 WriteFloatTestCases 方法提供的数据作为测试用例
    public void WriteLine_Float_FormatsNumberSameAsBasic(float value, string basicString)  # 定义一个测试方法，接受一个浮点数和一个基本字符串作为参数
    {
        var outputWriter = new StringWriter();  # 创建一个字符串写入器
        var io = new TextIO(new StringReader(""), outputWriter);  # 创建一个 TextIO 对象，用于输入输出

        io.WriteLine(value);  # 调用 TextIO 对象的 WriteLine 方法写入浮点数

        outputWriter.ToString().Should().BeEquivalentTo(basicString + Environment.NewLine);  # 断言输出的字符串应该等同于基本字符串加上换行符
    }

    public static TheoryData<float, string> WriteFloatTestCases()  # 定义一个静态方法，返回测试用例数据
        => new()  # 创建一个新的 TheoryData 对象
        {
            { 1000F, " 1000 " },  # 测试用例1：输入 1000F，期望输出 " 1000 "
            { 3.1415927F, " 3.1415927 " },  # 测试用例2：输入 3.1415927F，期望输出 " 3.1415927 "
            { 1F, " 1 " },  # 测试用例3：输入 1F，期望输出 " 1 "
            { 0F, " 0 " },  # 测试用例4：输入 0F，期望输出 " 0 "
            { -1F, "-1 " }, // 创建一个包含浮点数-1和对应字符串"-1 "的元组
            { -3.1415927F, "-3.1415927 " }, // 创建一个包含浮点数-3.1415927和对应字符串"-3.1415927 "的元组
            { -1000F, "-1000 " }, // 创建一个包含浮点数-1000和对应字符串"-1000 "的元组
        };

    [Theory]
    [MemberData(nameof(WriteIntTestCases))] // 使用WriteIntTestCases作为测试数据源
    public void Write_Int_FormatsNumberSameAsBasic(int value, string basicString) // 测试方法，测试写入整数的格式是否与基本字符串相同
    {
        var outputWriter = new StringWriter(); // 创建一个StringWriter对象用于输出
        var io = new TextIO(new StringReader(""), outputWriter); // 创建一个TextIO对象，用于读取和写入文本

        io.Write(value); // 将整数值写入输出

        outputWriter.ToString().Should().BeEquivalentTo(basicString); // 断言输出的字符串应该等同于基本字符串
    }

    [Theory]
    [MemberData(nameof(WriteIntTestCases))] // 使用WriteIntTestCases作为测试数据源
    public void WriteLine_Int_FormatsNumberSameAsBasic(int value, string basicString) // 测试方法，测试写入整数的格式是否与基本字符串相同
    {
        // 创建一个 StringWriter 对象，用于将文本写入字符串
        var outputWriter = new StringWriter();
        // 创建一个 TextIO 对象，使用一个 StringReader 作为输入，outputWriter 作为输出
        var io = new TextIO(new StringReader(""), outputWriter);

        // 将给定的 value 写入到 io 对象中
        io.WriteLine(value);

        // 检查 outputWriter 中的内容是否等同于 basicString 加上一个换行符的字符串
        outputWriter.ToString().Should().BeEquivalentTo(basicString + Environment.NewLine);
    }

    // 定义一个静态方法，返回一个 TheoryData<int, string> 对象
    public static TheoryData<int, string> WriteIntTestCases()
        => new()
        {
            // 添加测试用例：输入 1000，期望输出 " 1000 "
            { 1000, " 1000 " },
            // 添加测试用例：输入 1，期望输出 " 1 "
            { 1, " 1 " },
            // 添加测试用例：输入 0，期望输出 " 0 "
            { 0, " 0 " },
            // 添加测试用例：输入 -1，期望输出 "-1 "
            { -1, "-1 " },
            // 添加测试用例：输入 -1000，期望输出 "-1000 "
            { -1000, "-1000 " },
        };
}
```