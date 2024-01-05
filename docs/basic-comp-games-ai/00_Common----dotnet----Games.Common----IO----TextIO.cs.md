# `00_Common\dotnet\Games.Common\IO\TextIO.cs`

```
using Games.Common.Numbers; // 导入 Games.Common.Numbers 命名空间

namespace Games.Common.IO; // 声明 Games.Common.IO 命名空间

/// <inheritdoc /> // 继承自父类的注释
/// <summary> // 摘要注释，解释类的作用
/// Implements <see cref="IReadWrite" /> with input read from a <see cref="TextReader" /> and output written to a
/// <see cref="TextWriter" />.
/// </summary>
/// <remarks> // 备注注释，解释类的实现细节
/// This implementation reproduces the Vintage BASIC input experience, prompting multiple times when partial input
/// supplied, rejecting non-numeric input as needed, warning about extra input being ignored, etc.
/// </remarks>
public class TextIO : IReadWrite // 声明 TextIO 类，实现 IReadWrite 接口
{
    private readonly TextReader _input; // 声明私有的 TextReader 类型变量 _input
    private readonly TextWriter _output; // 声明私有的 TextWriter 类型变量 _output
    private readonly TokenReader _stringTokenReader; // 声明私有的 TokenReader 类型变量 _stringTokenReader
    private readonly TokenReader _numberTokenReader; // 声明私有的 TokenReader 类型变量 _numberTokenReader
    public TextIO(TextReader input, TextWriter output)
    {
        _input = input ?? throw new ArgumentNullException(nameof(input));  // 初始化输入流，如果输入流为空则抛出异常
        _output = output ?? throw new ArgumentNullException(nameof(output));  // 初始化输出流，如果输出流为空则抛出异常
        _stringTokenReader = TokenReader.ForStrings(this);  // 使用当前对象创建字符串标记读取器
        _numberTokenReader = TokenReader.ForNumbers(this);  // 使用当前对象创建数字标记读取器
    }

    public virtual char ReadCharacter()
    {
        while(true)
        {
            var ch = _input.Read();  // 读取输入流中的字符
            if (ch != -1) { return (char)ch; }  // 如果读取到的字符不是结束符，则返回该字符
        }
    }

    public float ReadNumber(string prompt) => ReadNumbers(prompt, 1)[0];  // 读取一个数字并返回

    public (float, float) Read2Numbers(string prompt)  // 读取两个数字
    {
        # 从用户输入中读取两个数字
        var numbers = ReadNumbers(prompt, 2);
        # 返回这两个数字
        return (numbers[0], numbers[1]);
    }

    # 读取三个数字并返回一个包含这三个数字的元组
    public (float, float, float) Read3Numbers(string prompt)
    {
        # 从用户输入中读取三个数字
        var numbers = ReadNumbers(prompt, 3);
        # 返回这三个数字
        return (numbers[0], numbers[1], numbers[2]);
    }

    # 读取四个数字并返回一个包含这四个数字的元组
    public (float, float, float, float) Read4Numbers(string prompt)
    {
        # 从用户输入中读取四个数字
        var numbers = ReadNumbers(prompt, 4);
        # 返回这四个数字
        return (numbers[0], numbers[1], numbers[2], numbers[3]);
    }

    # 从用户输入中读取指定数量的数字
    public void ReadNumbers(string prompt, float[] values)
    {
        # 如果传入的数组长度为0，则执行以下操作
        if (values.Length == 0)
        {
            throw new ArgumentException($"'{nameof(values)}' must have a non-zero length.", nameof(values));
        }
        # 如果传入的 values 数组长度为零，则抛出参数异常

        var numbers = _numberTokenReader.ReadTokens(prompt, (uint)values.Length).Select(t => t.Number).ToArray();
        numbers.CopyTo(values.AsSpan());
        # 使用 _numberTokenReader 从输入中读取指定数量的数字，并将其复制到传入的 values 数组中

    }

    private IReadOnlyList<float> ReadNumbers(string prompt, uint quantity) =>
        (quantity > 0)
            ? _numberTokenReader.ReadTokens(prompt, quantity).Select(t => t.Number).ToList()
            : throw new ArgumentOutOfRangeException(
                nameof(quantity),
                $"'{nameof(quantity)}' must be greater than zero.");
        # 如果传入的数量大于零，则使用 _numberTokenReader 从输入中读取指定数量的数字并返回列表，否则抛出参数异常

    public string ReadString(string prompt)
    {
        return ReadStrings(prompt, 1)[0];
    }
    # 从输入中读取一个字符串并返回
    public (string, string) Read2Strings(string prompt)
    {
        // 调用ReadStrings方法，获取两个字符串值，并以元组的形式返回
        var values = ReadStrings(prompt, 2);
        return (values[0], values[1]);
    }

    private IReadOnlyList<string> ReadStrings(string prompt, uint quantityRequired) =>
        // 调用_stringTokenReader的ReadTokens方法，根据指定的数量要求读取字符串，并转换为不可变的字符串列表
        _stringTokenReader.ReadTokens(prompt, quantityRequired).Select(t => t.String).ToList();

    internal string ReadLine(string prompt)
    {
        // 输出提示信息，并读取用户输入的字符串
        Write(prompt + "? ");
        return _input.ReadLine() ?? throw new InsufficientInputException();
    }

    public void Write(string value) => _output.Write(value); // 将指定的字符串值写入输出

    public void WriteLine(string value = "") => _output.WriteLine(value); // 将指定的字符串值写入输出并换行

    public void Write(Number value) => _output.Write(value.ToString()); // 将指定的数字值转换为字符串并写入输出
# 将数字值写入输出
public void WriteLine(Number value) => _output.WriteLine(value.ToString());

# 将对象值写入输出
public void Write(object value) => _output.Write(value.ToString());

# 将对象值写入输出并换行
public void WriteLine(object value) => _output.WriteLine(value.ToString());

# 使用指定格式和参数将字符串写入输出
public void Write(string format, params object[] values) => _output.Write(format, values);

# 使用指定格式和参数将字符串写入输出并换行
public void WriteLine(string format, params object[] values) => _output.WriteLine(format, values);

# 将流中的内容逐行写入输出
public void Write(Stream stream, bool keepOpen = false)
{
    # 使用流创建一个StreamReader对象
    using var reader = new StreamReader(stream);
    # 循环读取流中的内容并写入输出
    while (!reader.EndOfStream)
    {
        _output.WriteLine(reader.ReadLine());
    }

    # 如果不需要保持流打开状态，则关闭流
    if (!keepOpen) { stream?.Dispose(); }
}
    }  # 结束一个代码块

    private string GetString(float value) => value < 0 ? $"{value} " : $" {value} ";  # 定义一个私有方法，根据传入的浮点数值返回对应的字符串
```