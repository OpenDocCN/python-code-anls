# `basic-computer-games\00_Common\dotnet\Games.Common\IO\IReadWrite.cs`

```py
using Games.Common.Numbers;  // 导入 Games.Common.Numbers 命名空间

namespace Games.Common.IO;  // 声明 Games.Common.IO 命名空间

/// <summary>
/// 提供字符串和数字的输入输出。
/// </summary>
public interface IReadWrite  // 定义 IReadWrite 接口
{
    /// <summary>
    /// 从输入中读取一个字符。
    /// </summary>
    /// <returns>读取到的字符。</returns>
    char ReadCharacter();

    /// <summary>
    /// 从输入中读取一个 <see cref="float" /> 值。
    /// </summary>
    /// <param name="prompt">显示提示值的文本。</param>
    /// <returns><see cref="float" /> 值，即输入的值。</returns>
    float ReadNumber(string prompt);

    /// <summary>
    /// 从输入中读取 2 个 <see cref="float" /> 值。
    /// </summary>
    /// <param name="prompt">显示提示值的文本。</param>
    /// <returns><see cref="ValueTuple{float, float}" />，即输入的值。</returns>
    (float, float) Read2Numbers(string prompt);

    /// <summary>
    /// 从输入中读取 3 个 <see cref="float" /> 值。
    /// </summary>
    /// <param name="prompt">显示提示值的文本。</param>
    /// <returns><see cref="ValueTuple{float, float, float}" />，即输入的值。</returns>
    (float, float, float) Read3Numbers(string prompt);

    /// <summary>
    /// 从输入中读取 4 个 <see cref="float" /> 值。
    /// </summary>
    /// <param name="prompt">显示提示值的文本。</param>
    /// <returns><see cref="ValueTuple{float, float, float, float}" />，即输入的值。</returns>
    (float, float, float, float) Read4Numbers(string prompt);

    /// <summary>
    /// 从输入中读取数字，填充数组。
    /// </summary>
    /// <param name="prompt">显示提示值的文本。</param>
    /// <param name="values">要填充输入值的 <see cref="float[]" />。</param>
    void ReadNumbers(string prompt, float[] values);
}
    // 从输入中读取一个字符串值
    string ReadString(string prompt);
    
    // 从输入中读取两个字符串值
    (string, string) Read2Strings(string prompt);
    
    // 将一个字符串写入输出
    void Write(string message);
    
    // 将一个字符串写入输出，然后换行
    void WriteLine(string message = "");
    
    // 将一个数字写入输出
    void Write(Number value);
    
    // 将一个数字写入输出，然后换行
    void WriteLine(Number value);
    
    // 将一个对象写入输出
    void Write(object value);
    
    // 将一个对象写入输出，然后换行
    void WriteLine(object value);
    
    // 将一个格式化的字符串写入输出
    void WriteFormattedString(string format);
    // 写入格式化字符串到输出，可以插入多个值
    void Write(string format, params object[] values);
    
    // 写入格式化字符串到输出，然后换行
    // format: 要写入的格式化字符串
    // values: 要插入格式化字符串的值
    void WriteLine(string format, params object[] values);
    
    // 将流的内容写入输出
    // stream: 要写入的流
    // keepOpen: 是否保持流打开状态，默认为false
    void Write(Stream stream, bool keepOpen = false);
# 闭合前面的函数定义
```