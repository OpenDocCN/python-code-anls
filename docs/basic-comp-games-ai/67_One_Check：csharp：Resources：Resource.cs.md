# `67_One_Check\csharp\Resources\Resource.cs`

```
# 导入所需的模块
import System.Reflection
import System.Runtime.CompilerServices

# 定义命名空间 OneCheck.Resources
namespace OneCheck.Resources;

# 定义内部静态类 Resource
internal static class Resource
{
    # 定义内部静态类 Streams
    internal static class Streams
    {
        # 定义静态属性 Introduction，返回 GetStream() 方法的结果
        public static Stream Introduction => GetStream()
        # 定义静态属性 IllegalMove，返回 GetStream() 方法的结果
        public static Stream IllegalMove => GetStream()
        # 定义静态属性 YesOrNo，返回 GetStream() 方法的结果
        public static Stream YesOrNo => GetStream()
        # 定义静态属性 Bye，返回 GetStream() 方法的结果
        public static Stream Bye => GetStream()
    }

    # 定义内部静态类 Formats
    internal static class Formats
    {
        # 定义静态属性 Results，返回 GetString() 方法的结果
        public static string Results => GetString()
    }
}
    # 内部静态类 Prompts，包含 From、To 和 TryAgain 三个属性，分别表示提示信息的来源、目标和重试
    internal static class Prompts
    {
        # 获取 From 属性的值
        public static string From => GetString();
        # 获取 To 属性的值
        public static string To => GetString();
        # 获取 TryAgain 属性的值
        public static string TryAgain => GetString();
    }

    # 内部静态类 Strings，包含 TooManyColumns 和 TooManyRows 两个属性，分别表示列数过多和行数过多的提示信息
    internal static class Strings
    {
        # 获取 TooManyColumns 属性的值
        public static string TooManyColumns => GetString();
        # 获取 TooManyRows 属性的值
        public static string TooManyRows => GetString();
    }

    # 私有方法 GetString，用于获取指定名称的资源字符串
    private static string GetString([CallerMemberName] string? name = null)
    {
        # 使用指定名称获取资源流
        using var stream = GetStream(name);
        # 使用资源流创建 StreamReader 对象
        using var reader = new StreamReader(stream);
        # 读取并返回资源字符串
        return reader.ReadToEnd();
    }
# 使用私有方法获取嵌入资源的流
private static Stream GetStream([CallerMemberName] string? name = null) =>
    # 获取当前执行程序集的嵌入资源流
    Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
        # 如果找不到嵌入资源流，则抛出异常
        ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
```