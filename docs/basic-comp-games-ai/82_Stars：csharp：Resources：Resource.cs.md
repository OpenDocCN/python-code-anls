# `d:/src/tocomm/basic-computer-games\82_Stars\csharp\Resources\Resource.cs`

```
# 使用 System.IO 命名空间中的类
using System.IO;
# 使用 System.Reflection 命名空间中的类
using System.Reflection;
# 使用 System.Runtime.CompilerServices 命名空间中的类
using System.Runtime.CompilerServices;

# 定义名为 Resource 的内部静态类
namespace Stars.Resources;

internal static class Resource
{
    # 定义名为 Streams 的内部静态类
    internal static class Streams
    {
        # 定义名为 Title 的公共静态属性，返回一个流对象
        public static Stream Title => GetStream();
    }

    # 定义名为 Formats 的内部静态类
    internal static class Formats
    {
        # 定义名为 Instructions 的公共静态属性，返回一个字符串
        public static string Instructions => GetString();
    }

    # 定义一个私有的静态方法，返回一个字符串
    private static string GetString([CallerMemberName] string name = null)
    {
        // 使用给定的文件名获取字节流
        using var stream = GetStream(name);
        // 使用字节流创建一个读取器
        using var reader = new StreamReader(stream);
        // 读取并返回读取器中的所有内容
        return reader.ReadToEnd();
    }

    // 根据调用者的成员名获取嵌入资源的字节流
    private static Stream GetStream([CallerMemberName] string name = null)
        => Assembly.GetExecutingAssembly().GetManifestResourceStream($"Stars.Resources.{name}.txt");
}
```