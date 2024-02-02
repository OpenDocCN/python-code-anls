# `basic-computer-games\00_Common\dotnet\Games.Common\IO\TextIO.cs`

```py
using Games.Common.Numbers;  // 导入 Games.Common.Numbers 命名空间

namespace Games.Common.IO;  // 定义 Games.Common.IO 命名空间

/// <inheritdoc />  // 实现接口的注释
/// <summary>
/// Implements <see cref="IReadWrite" /> with input read from a <see cref="TextReader" /> and output written to a
/// <see cref="TextWriter" />.
/// </summary>
/// <remarks>
/// This implementation reproduces the Vintage BASIC input experience, prompting multiple times when partial input
/// supplied, rejecting non-numeric input as needed, warning about extra input being ignored, etc.
/// </remarks>
public class TextIO : IReadWrite  // 定义 TextIO 类并实现 IReadWrite 接口
{
    private readonly TextReader _input;  // 声明私有只读字段 _input，类型为 TextReader
    private readonly TextWriter _output;  // 声明私有只读字段 _output，类型为 TextWriter
    private readonly TokenReader _stringTokenReader;  // 声明私有只读字段 _stringTokenReader，类型为 TokenReader
    private readonly TokenReader _numberTokenReader;  // 声明私有只读字段 _numberTokenReader，类型为 TokenReader

    public TextIO(TextReader input, TextWriter output)  // TextIO 类的构造函数，接受 TextReader 和 TextWriter 作为参数
    {
        _input = input ?? throw new ArgumentNullException(nameof(input));  // 如果 input 为 null，则抛出 ArgumentNullException
        _output = output ?? throw new ArgumentNullException(nameof(output));  // 如果 output 为 null，则抛出 ArgumentNullException
        _stringTokenReader = TokenReader.ForStrings(this);  // 使用 TokenReader.ForStrings 方法初始化 _stringTokenReader
        _numberTokenReader = TokenReader.ForNumbers(this);  // 使用 TokenReader.ForNumbers 方法初始化 _numberTokenReader
    }

    public virtual char ReadCharacter()  // 定义公共虚方法 ReadCharacter，返回类型为 char
    {
        while(true)  // 进入无限循环
        {
            var ch = _input.Read();  // 从 _input 中读取一个字符的 ASCII 值
            if (ch != -1) { return (char)ch; }  // 如果读取的字符不是 -1（即还有字符可读），则将其转换为 char 类型并返回
        }
    }

    public float ReadNumber(string prompt) => ReadNumbers(prompt, 1)[0];  // 定义公共方法 ReadNumber，接受 prompt 作为参数，调用 ReadNumbers 方法读取一个数字并返回

    public (float, float) Read2Numbers(string prompt)  // 定义公共方法 Read2Numbers，接受 prompt 作为参数，返回类型为元组 (float, float)
    {
        var numbers = ReadNumbers(prompt, 2);  // 调用 ReadNumbers 方法读取两个数字
        return (numbers[0], numbers[1]);  // 返回读取到的两个数字组成的元组
    }

    public (float, float, float) Read3Numbers(string prompt)  // 定义公共方法 Read3Numbers，接受 prompt 作为参数，返回类型为元组 (float, float, float)
    {
        var numbers = ReadNumbers(prompt, 3);  // 调用 ReadNumbers 方法读取三个数字
        return (numbers[0], numbers[1], numbers[2]);  // 返回读取到的三个数字组成的元组
    }

    public (float, float, float, float) Read4Numbers(string prompt)  // 定义公共方法 Read4Numbers，接受 prompt 作为参数，返回类型为元组 (float, float, float, float)
    {
        var numbers = ReadNumbers(prompt, 4);  // 调用 ReadNumbers 方法读取四个数字
        return (numbers[0], numbers[1], numbers[2], numbers[3]);  // 返回读取到的四个数字组成的元组
    }

    public void ReadNumbers(string prompt, float[] values)  // 定义公共方法 ReadNumbers，接受 prompt 和 values 作为参数
    {
        // 检查传入的数组是否为空，如果是则抛出参数异常
        if (values.Length == 0)
        {
            throw new ArgumentException($"'{nameof(values)}' must have a non-zero length.", nameof(values));
        }
    
        // 从输入中读取指定数量的数字，并转换为数组
        var numbers = _numberTokenReader.ReadTokens(prompt, (uint)values.Length).Select(t => t.Number).ToArray();
        // 将读取到的数字复制到传入的数组中
        numbers.CopyTo(values.AsSpan());
    }
    
    // 读取指定数量的数字并返回一个只读列表
    private IReadOnlyList<float> ReadNumbers(string prompt, uint quantity) =>
        (quantity > 0)
            ? _numberTokenReader.ReadTokens(prompt, quantity).Select(t => t.Number).ToList()
            : throw new ArgumentOutOfRangeException(
                nameof(quantity),
                $"'{nameof(quantity)}' must be greater than zero.");
    
    // 读取单个字符串
    public string ReadString(string prompt)
    {
        return ReadStrings(prompt, 1)[0];
    }
    
    // 读取两个字符串并以元组形式返回
    public (string, string) Read2Strings(string prompt)
    {
        var values = ReadStrings(prompt, 2);
        return (values[0], values[1]);
    }
    
    // 读取指定数量的字符串并返回一个只读列表
    private IReadOnlyList<string> ReadStrings(string prompt, uint quantityRequired) =>
        _stringTokenReader.ReadTokens(prompt, quantityRequired).Select(t => t.String).ToList();
    
    // 读取一行输入
    internal string ReadLine(string prompt)
    {
        Write(prompt + "? ");
        return _input.ReadLine() ?? throw new InsufficientInputException();
    }
    
    // 输出字符串
    public void Write(string value) => _output.Write(value);
    
    // 输出字符串并换行
    public void WriteLine(string value = "") => _output.WriteLine(value);
    
    // 输出数字
    public void Write(Number value) => _output.Write(value.ToString());
    
    // 输出数字并换行
    public void WriteLine(Number value) => _output.WriteLine(value.ToString());
    
    // 输出对象
    public void Write(object value) => _output.Write(value.ToString());
    
    // 输出对象并换行
    public void WriteLine(object value) => _output.WriteLine(value.ToString());
    
    // 使用指定格式输出字符串
    public void Write(string format, params object[] values) => _output.Write(format, values);
    
    // 使用指定格式输出字符串并换行
    public void WriteLine(string format, params object[] values) => _output.WriteLine(format, values);
    
    // 输出流
    public void Write(Stream stream, bool keepOpen = false)
    {
        // 使用 var 声明一个 StreamReader 对象，用于读取流中的数据
        using var reader = new StreamReader(stream);
        // 循环读取流中的每一行数据，直到流结束
        while (!reader.EndOfStream)
        {
            // 将读取的每一行数据写入输出
            _output.WriteLine(reader.ReadLine());
        }
    
        // 如果不需要保持流打开状态，则释放流资源
        if (!keepOpen) { stream?.Dispose(); }
    }
    
    // 定义一个私有方法，用于将浮点数转换为字符串
    private string GetString(float value) => value < 0 ? $"{value} " : $" {value} ";
# 闭合前面的函数定义
```