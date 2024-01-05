# `00_Common\dotnet\Games.Common\IO\IReadWrite.cs`

```
using Games.Common.Numbers; // 导入 Games.Common.Numbers 命名空间
namespace Games.Common.IO; // 定义 Games.Common.IO 命名空间

/// <summary>
/// 提供字符串和数字的输入输出。
/// </summary>
public interface IReadWrite // 定义 IReadWrite 接口
{
    /// <summary>
    /// 从输入中读取一个字符。
    /// </summary>
    /// <returns>读取到的字符。</returns>
    char ReadCharacter(); // 定义 ReadCharacter 方法

    /// <summary>
    /// 从输入中读取一个 <see cref="float" /> 值。
    /// </summary>
    /// <param name="prompt">显示提示信息以获取值。</param>
    /// <returns><see cref="float" /> 值，即输入的值。</returns>
    // 声明一个函数，用于读取一个浮点数
    float ReadNumber(string prompt);

    /// <summary>
    /// 从输入中读取2个 <see cref="float" /> 值。
    /// </summary>
    /// <param name="prompt">显示提示值的文本。</param>
    /// <returns>一个 <see cref="ValueTuple{float, float}" />，包含输入的值。</returns>
    (float, float) Read2Numbers(string prompt);

    /// <summary>
    /// 从输入中读取3个 <see cref="float" /> 值。
    /// </summary>
    /// <param name="prompt">显示提示值的文本。</param>
    /// <returns>一个 <see cref="ValueTuple{float, float, float}" />，包含输入的值。</returns>
    (float, float, float) Read3Numbers(string prompt);

    /// <summary>
    /// 从输入中读取4个 <see cref="float" /> 值。
    /// </summary>
    /// <param name="prompt">显示提示值的文本。</param>
    // 从输入中读取4个数字，并以元组的形式返回这4个数字
    (float, float, float, float) Read4Numbers(string prompt);

    // 从输入中读取数字，填充到一个数组中
    // 参数 prompt：显示在提示值时要显示的文本
    // 参数 values：要用输入值填充的 float 数组
    void ReadNumbers(string prompt, float[] values);

    // 从输入中读取一个字符串值
    // 参数 prompt：显示在提示值时要显示的文本
    // 返回输入的字符串值
    string ReadString(string prompt);

    // 从输入中读取2个字符串值
    // 读取两个字符串值，并返回一个包含这两个值的元组
    (string, string) Read2Strings(string prompt);

    // 将字符串写入输出
    void Write(string message);

    // 将字符串写入输出，然后换行
    void WriteLine(string message = "");

    // 将数字写入输出
    void WriteNumber(Number value);
    # 将一个数字写入输出
    void Write(Number value);

    /// <summary>
    /// 将一个 <see cref="Number" /> 写入输出。
    /// </summary>
    /// <param name="value">要写入的 <see cref="Number" />。</param>
    void WriteLine(Number value);

    /// <summary>
    /// 将一个 <see cref="object" /> 写入输出。
    /// </summary>
    /// <param name="value">要写入的 <see cref="object" />。</param>
    void Write(object value);

    /// <summary>
    /// 将一个 <see cref="object" /> 写入输出。
    /// </summary>
    /// <param name="value">要写入的 <see cref="object" />。</param>
    void WriteLine(object value);
// 写入格式化的字符串到输出
void Write(string format, params object[] values);

// 写入格式化的字符串到输出，并在末尾添加换行符
void WriteLine(string format, params object[] values);

// 将流的内容写入输出
void Write(Stream stream, bool keepOpen = false);
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制数据，并封装成字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，'r'表示以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```