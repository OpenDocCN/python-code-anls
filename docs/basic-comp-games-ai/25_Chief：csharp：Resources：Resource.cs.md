# `d:/src/tocomm/basic-computer-games\25_Chief\csharp\Resources\Resource.cs`

```
# 使用 System.Reflection 和 System.Runtime.CompilerServices 模块
using System.Reflection;
using System.Runtime.CompilerServices;

# 命名空间 Chief.Resources
namespace Chief.Resources;

# 内部静态类 Resource
internal static class Resource
{
    # 内部静态类 Streams
    internal static class Streams
    {
        # 公共静态属性 Bye，返回 GetStream() 方法的结果
        public static Stream Bye => GetStream();
        # 公共静态属性 Instructions，返回 GetStream() 方法的结果
        public static Stream Instructions => GetStream();
        # 公共静态属性 Lightning，返回 GetStream() 方法的结果
        public static Stream Lightning => GetStream();
        # 公共静态属性 ShutUp，返回 GetStream() 方法的结果
        public static Stream ShutUp => GetStream();
        # 公共静态属性 Title，返回 GetStream() 方法的结果
        public static Stream Title => GetStream();
    }

    # 内部静态类 Formats
    internal static class Formats
    {
        # 公共静态属性 Bet，返回 GetString() 方法的结果
        public static string Bet => GetString();
        # 公共静态属性 Working，返回 GetString() 方法的结果
        public static string Working => GetString();
```

这段代码是一个 C# 的命名空间和类定义，其中包含了一些静态属性和方法。每个静态属性都返回一个流（Stream）或字符串（string）的结果。
    }

    internal static class Prompts
    {
        // 定义静态属性 Answer，返回调用 GetString 方法的结果
        public static string Answer => GetString();
        // 定义静态属性 Believe，返回调用 GetString 方法的结果
        public static string Believe => GetString();
        // 定义静态属性 Original，返回调用 GetString 方法的结果
        public static string Original => GetString();
        // 定义静态属性 Ready，返回调用 GetString 方法的结果
        public static string Ready => GetString();
    }

    // 定义私有方法 GetString，参数为调用者的成员名，返回读取的文本内容
    private static string GetString([CallerMemberName] string? name = null)
    {
        // 使用 GetStream 方法获取资源文件的流
        using var stream = GetStream(name);
        // 使用 StreamReader 读取流中的文本内容
        using var reader = new StreamReader(stream);
        // 返回读取的文本内容
        return reader.ReadToEnd();
    }

    // 定义私有方法 GetStream，参数为调用者的成员名，返回资源文件的流
    private static Stream GetStream([CallerMemberName] string? name = null)
        // 使用 Assembly.GetExecutingAssembly().GetManifestResourceStream 获取资源文件的流
        => Assembly.GetExecutingAssembly().GetManifestResourceStream($"Chief.Resources.{name}.txt")
            // 如果资源文件流不存在，则抛出参数异常
            ?? throw new ArgumentException($"Resource stream {name} does not exist", nameof(name));
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源
```