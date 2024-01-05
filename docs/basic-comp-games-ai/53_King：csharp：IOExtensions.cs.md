# `d:/src/tocomm/basic-computer-games\53_King\csharp\IOExtensions.cs`

```
# 导入所需的模块
import zipfile
from io import BytesIO

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
        return false;  # 返回 false，表示读取数值失败

    }

    internal static bool TryReadValue(this IReadWrite io, string prompt, out float value, params ValidityTest[] tests)
    {
        while (true)
        {
            var response = value = io.ReadNumber(prompt);  # 从输入流中读取数值，并将其赋值给 response 和 value
            if (response == 0) { return false; }  # 如果 response 为 0，则返回 false
            if (tests.All(test => test.IsValid(response, io))) { return true; }  # 使用传入的测试函数检验 response 是否有效，如果全部通过则返回 true
        } 
    }

    internal static bool TryReadValue(this IReadWrite io, string prompt, out float value)
        => io.TryReadValue(prompt, _ => true, "", out value);  # 调用重载的 TryReadValue 方法，默认使用 true 作为测试函数

    internal static bool TryReadValue(
        this IReadWrite io,
        string prompt,
        Predicate<float> isValid,  # 传入的测试函数，用于检验读取的数值是否有效
        string error,  # 定义一个字符串变量 error
        out float value)  # 定义一个浮点数变量 value，用于存储读取的值
        => io.TryReadValue(prompt, isValid, () => error, out value);  # 调用 TryReadValue 方法，尝试读取值并验证

    internal static bool TryReadValue(  # 定义一个静态方法 TryReadValue，用于尝试读取值
        this IReadWrite io,  # 使用 this 关键字表示该方法是一个扩展方法，作用于 IReadWrite 接口类型的对象
        string prompt,  # 提示用户输入的字符串
        Predicate<float> isValid,  # 用于验证输入值是否有效的委托
        Func<string> getError,  # 获取错误信息的委托
        out float value)  # 用于存储读取的值
    {
        while (true)  # 进入循环，持续读取值
        {
            value = io.ReadNumber(prompt);  # 调用 ReadNumber 方法，读取用户输入的值
            if (value < 0) { return false; }  # 如果值小于 0，则返回 false
            if (isValid(value)) { return true; }  # 如果值有效，则返回 true
            
            io.Write(getError());  # 调用 Write 方法，输出错误信息
        }
    }
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```