# `56_Life_for_Two\csharp\Resources\Resource.cs`

```
# 导入所需的模块
import System.Reflection
import System.Runtime.CompilerServices

# 声明命名空间
namespace LifeforTwo.Resources;

# 声明内部静态类 Resource
internal static class Resource
{
    # 声明内部静态类 Streams
    internal static class Streams
    {
        # 声明静态属性 Title，返回一个流对象
        public static Stream Title => GetStream();
        # 声明静态属性 IllegalCoords，返回一个流对象
        public static Stream IllegalCoords => GetStream();
        # 声明静态属性 SameCoords，返回一个流对象
        public static Stream SameCoords => GetStream();
    }

    # 声明内部静态类 Formats
    internal static class Formats
    {
        # 声明静态属性 InitialPieces，返回一个字符串
        public static string InitialPieces => GetString();
        # 声明静态属性 Player，返回一个字符串
        public static string Player => GetString();
        # 声明静态属性 Winner，返回一个字符串
        public static string Winner => GetString();
    }
}
    internal static class Strings
    {
        // 定义一个静态属性 Draw，其值为调用 GetString 方法的结果
        public static string Draw => GetString();
    }

    // 定义一个私有的静态方法 GetString，接受一个可选的参数 name，类型为 string，使用 CallerMemberName 特性
    private static string GetString([CallerMemberName] string? name = null)
    {
        // 使用 using 声明一个变量 stream，调用 GetStream 方法获取流
        using var stream = GetStream(name);
        // 使用 using 声明一个变量 reader，使用 stream 初始化 StreamReader
        using var reader = new StreamReader(stream);
        // 返回 reader 读取的所有内容
        return reader.ReadToEnd();
    }

    // 定义一个私有的静态方法 GetStream，接受一个可选的参数 name，类型为 string，使用 CallerMemberName 特性
    private static Stream GetStream([CallerMemberName] string? name = null) =>
        // 使用 Assembly.GetExecutingAssembly().GetManifestResourceStream 方法获取嵌入资源流
        Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
            // 如果获取的流为空，则抛出异常
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
}
```