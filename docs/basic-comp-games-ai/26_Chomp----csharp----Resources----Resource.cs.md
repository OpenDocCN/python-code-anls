# `26_Chomp\csharp\Resources\Resource.cs`

```
# 导入必要的模块
import System.Reflection
import System.Runtime.CompilerServices

# 定义命名空间 Chomp.Resources
namespace Chomp.Resources;

# 定义内部静态类 Resource
internal static class Resource
{
    # 定义内部静态类 Streams
    internal static class Streams
    {
        # 定义静态属性 HereWeGo，返回一个流对象
        public static Stream HereWeGo => GetStream();
        # 定义静态属性 Introduction，返回一个流对象
        public static Stream Introduction => GetStream();
        # 定义静态属性 Rules，返回一个流对象
        public static Stream Rules => GetStream();
        # 定义静态属性 NoFair，返回一个流对象
        public static Stream NoFair => GetStream();
    }

    # 定义内部静态类 Formats
    internal static class Formats
    {
        # 定义静态属性 Player，返回一个字符串
        public static string Player => GetString();
        # 定义静态属性 YouLose，返回一个字符串
        public static string YouLose => GetString();
    }
}
    internal static class Prompts
    {
        public static string Coordinates => GetString();  # 定义一个静态属性，返回调用 GetString() 方法的结果
        public static string HowManyPlayers => GetString();  # 定义一个静态属性，返回调用 GetString() 方法的结果
        public static string HowManyRows => GetString();  # 定义一个静态属性，返回调用 GetString() 方法的结果
        public static string HowManyColumns => GetString();  # 定义一个静态属性，返回调用 GetString() 方法的结果
        public static string TooManyColumns => GetString();  # 定义一个静态属性，返回调用 GetString() 方法的结果
    }

    internal static class Strings
    {
        public static string TooManyColumns => GetString();  # 定义一个静态属性，返回调用 GetString() 方法的结果
        public static string TooManyRows => GetString();  # 定义一个静态属性，返回调用 GetString() 方法的结果
    }

    private static string GetString([CallerMemberName] string? name = null)  # 定义一个私有方法，使用了 CallerMemberName 特性，返回值为字符串
    {
        using var stream = GetStream(name);  # 使用 GetStream 方法获取一个流对象，并使用 using 语句进行资源管理
        using var reader = new StreamReader(stream);  # 使用流对象创建一个 StreamReader 对象，并使用 using 语句进行资源管理
        return reader.ReadToEnd();
```
这行代码的作用是读取流中的所有内容并返回。

```
    private static Stream GetStream([CallerMemberName] string? name = null) =>
        Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
```
这段代码定义了一个私有的静态方法GetStream，它接受一个可选的参数name，并使用该参数来获取嵌入资源的流。如果找不到对应的资源流，则抛出一个异常。
```