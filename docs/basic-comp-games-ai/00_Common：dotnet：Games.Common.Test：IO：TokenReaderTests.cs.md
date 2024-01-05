# `d:/src/tocomm/basic-computer-games\00_Common\dotnet\Games.Common.Test\IO\TokenReaderTests.cs`

```
# 导入所需的模块
import zipfile  # 用于处理 ZIP 文件
from io import BytesIO  # 用于创建字节流
```
```python
# 定义一个函数，根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
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
    [Fact] // 标记该方法为单元测试方法
    public void ReadTokens_QuantityNeededZero_ThrowsArgumentException() // 测试读取令牌数量为零时是否抛出参数异常
    {
        var sut = TokenReader.ForStrings(new TextIO(new StringReader(""), _outputWriter)); // 创建 TokenReader 对象

        Action readTokens = () => sut.ReadTokens("", 0); // 定义一个委托，用于调用 sut 的 ReadTokens 方法

        readTokens.Should().Throw<ArgumentOutOfRangeException>() // 断言 readTokens 调用时会抛出 ArgumentOutOfRangeException 异常
            .WithMessage("'quantityNeeded' must be greater than zero.*") // 验证异常消息是否符合预期
            .WithParameterName("quantityNeeded"); // 验证异常参数名称是否符合预期
    }

    [Theory] // 标记该方法为理论测试方法
    [MemberData(nameof(ReadTokensTestCases))] // 使用 ReadTokensTestCases 方法提供的数据进行测试
    public void ReadTokens_ReadingValuesHasExpectedPromptsAndResults( // 测试读取值时是否具有预期的提示和结果
        string prompt, // 输入提示
        uint tokenCount, // 令牌数量
        string input, // 输入值
        string expectedOutput,  // 期望的输出结果
        string[] expectedResult)  // 期望的结果数组

    {
        var sut = TokenReader.ForStrings(new TextIO(new StringReader(input + NewLine), _outputWriter));  // 创建一个 TokenReader 对象，用于读取输入的字符串

        var result = sut.ReadTokens(prompt, tokenCount);  // 使用 TokenReader 对象读取指定数量的 token
        var output = _outputWriter.ToString();  // 获取输出结果

        using var _ = new AssertionScope();  // 使用 AssertionScope 进行断言

        output.Should().Be(expectedOutput);  // 断言输出结果应该等于期望的输出结果
        result.Select(t => t.String).Should().BeEquivalentTo(expectedResult);  // 断言读取的 token 应该等于期望的结果数组
    }

    [Theory]
    [MemberData(nameof(ReadNumericTokensTestCases))]  // 使用 ReadNumericTokensTestCases 提供的测试数据
    public void ReadTokens_Numeric_ReadingValuesHasExpectedPromptsAndResults(
        string prompt,  // 输入的提示信息
        uint tokenCount,  // 读取的 token 数量
        string input,  // 输入的字符串
        string expectedOutput,  // 期望的输出结果
        float[] expectedResult)
    {
        # 创建 TokenReader 对象，用于读取数字
        var sut = TokenReader.ForNumbers(new TextIO(new StringReader(input + NewLine), _outputWriter));

        # 读取指定数量的 tokens
        var result = sut.ReadTokens(prompt, tokenCount);
        # 获取输出结果
        var output = _outputWriter.ToString();

        # 使用 AssertionScope 进行断言
        using var _ = new AssertionScope();
        # 断言输出结果应该等于期望的输出
        output.Should().Be(expectedOutput);
        # 断言读取的 tokens 中的数字应该等同于期望的结果
        result.Select(t => t.Number).Should().BeEquivalentTo(expectedResult);
    }

    # 定义测试用例数据
    public static TheoryData<string, uint, string, string, string[]> ReadTokensTestCases()
    {
        return new()
        {
            # 测试用例1
            { "Name", 1, "Bill", "Name? ", new[] { "Bill" } },
            # 测试用例2
            { "Names", 2, " Bill , Bloggs ", "Names? ", new[] { "Bill", "Bloggs" } },
            # 测试用例3
            { "Names", 2, $" Bill{NewLine}Bloggs ", "Names? ?? ", new[] { "Bill", "Bloggs" } },
            {
                "Foo", // 第一个参数：字符串 "Foo"
                6, // 第二个参数：整数 6
                $"1,2{NewLine}\" a,b \"{NewLine},\"\"c,d{NewLine}d\"x,e,f", // 第三个参数：包含特定格式的字符串
                $"Foo? ?? ?? ?? {ExtraInput}{NewLine}", // 第四个参数：包含特定格式的字符串
                new[] { "1", "2", " a,b ", "", "", "d\"x" } // 第五个参数：字符串数组
            }
        };
    }

    public static TheoryData<string, uint, string, string, float[]> ReadNumericTokensTestCases()
    {
        return new()
        {
            { "Age", 1, "23", "Age? ", new[] { 23F } }, // 第一个测试用例：包含字符串、整数、字符串、字符串、浮点数数组
            { "Constants", 2, " 3.141 , 2.71 ", "Constants? ", new[] { 3.141F, 2.71F } }, // 第二个测试用例：包含字符串、整数、字符串、字符串、浮点数数组
            { "Answer", 1, $"Forty-two{NewLine}42 ", $"Answer? {NumberExpected}{NewLine}? ", new[] { 42F } }, // 第三个测试用例：包含字符串、整数、特定格式的字符串、特定格式的字符串、浮点数数组
            {
                "Foo", // 第四个测试用例：字符串 "Foo"
                6, // 整数 6
                $"1,2{NewLine}\" a,b \"{NewLine}3, 4  {NewLine}5.6,7,a, b", // 包含特定格式的字符串
# 创建一个包含字符串和数组的对象
var data = new
{
    # 字符串属性，包含占位符和变量
    message = $"Foo? ?? {NumberExpected}{NewLine}? ?? {ExtraInput}{NewLine}",
    # 数组属性，包含整数和浮点数
    numbers = new[] { 1, 2, 3, 4, 5.6F, 7 }
};
```