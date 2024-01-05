# `d:/src/tocomm/basic-computer-games\41_Guess\csharp\Resources\Resource.cs`

```
# 使用 System.Reflection 和 System.Runtime.CompilerServices 模块
using System.Reflection;
using System.Runtime.CompilerServices;

# 命名空间 Guess.Resources
namespace Guess.Resources;

# 内部静态类 Resource
internal static class Resource
{
    # 内部静态类 Streams
    internal static class Streams
    {
        # 静态属性 Introduction 返回 GetStream() 方法的结果
        public static Stream Introduction => GetStream();
        # 静态属性 TooLow 返回 GetStream() 方法的结果
        public static Stream TooLow => GetStream();
        # 静态属性 TooHigh 返回 GetStream() 方法的结果
        public static Stream TooHigh => GetStream();
        # 静态属性 BlankLines 返回 GetStream() 方法的结果
        public static Stream BlankLines => GetStream();
    }

    # 内部静态类 Formats
    internal static class Formats
    {
        # 静态属性 Thinking 返回 GetString() 方法的结果
        public static string Thinking => GetString();
        # 静态属性 ThatsIt 返回 GetString() 方法的结果
        public static string ThatsIt => GetString();
        # 静态属性 ShouldHave 返回 GetString() 方法的结果
        public static string ShouldHave => GetString();
```

这段代码是一个 C# 的命名空间和类定义，其中包含了一些静态属性和方法。每个静态属性都返回一个方法的结果，但是在给定的代码中并没有给出这些方法的具体实现，因此无法准确解释它们的作用。
    }

    internal static class Prompts
    {
        public static string Limit => GetString();  # 定义一个静态属性 Limit，返回调用 GetString() 方法的结果
    }

    internal static class Strings
    {
        public static string Good => GetString();  # 定义一个静态属性 Good，返回调用 GetString() 方法的结果
        public static string VeryGood => GetString();  # 定义一个静态属性 VeryGood，返回调用 GetString() 方法的结果
    }

    private static string GetString([CallerMemberName] string? name = null)  # 定义一个私有的静态方法 GetString，接受一个可选的参数 name，默认值为 null
    {
        using var stream = GetStream(name);  # 使用关键字 using 创建一个 stream 对象，调用 GetStream 方法并传入 name 参数
        using var reader = new StreamReader(stream);  # 使用关键字 using 创建一个 reader 对象，传入 stream 对象
        return reader.ReadToEnd();  # 返回 reader 对象的全部内容
    }
# 从调用者的成员名获取流
def GetStream(name=None):
    # 获取当前执行程序集的嵌入资源流
    stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(f"{typeof(Resource).Namespace}.{name}.txt")
    # 如果找不到嵌入资源流，则抛出异常
    if stream is None:
        raise Exception(f"Could not find embedded resource stream '{name}'.")
    return stream
```