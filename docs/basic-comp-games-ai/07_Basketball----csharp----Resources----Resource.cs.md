# `basic-computer-games\07_Basketball\csharp\Resources\Resource.cs`

```py
// 使用 System.Reflection 和 System.Runtime.CompilerServices 命名空间
using System.Reflection;
using System.Runtime.CompilerServices;

// 声明命名空间 Basketball.Resources
namespace Basketball.Resources
{
    // 声明内部静态类 Resource
    internal static class Resource
    {
        // 声明内部静态类 Streams
        internal static class Streams
        {
            // 声明静态属性 Introduction，返回 GetStream() 方法的结果
            public static Stream Introduction => GetStream();
            // 声明静态属性 TwoMinutesLeft，返回 GetStream() 方法的结果
            public static Stream TwoMinutesLeft => GetStream();
        }

        // 声明内部静态类 Formats
        internal static class Formats
        {
            // 声明静态属性 EndOfFirstHalf，返回 GetString() 方法的结果
            public static string EndOfFirstHalf => GetString();
            // 声明静态属性 EndOfGame，返回 GetString() 方法的结果
            public static string EndOfGame => GetString();
            // 声明静态属性 EndOfSecondHalf，返回 GetString() 方法的结果
            public static string EndOfSecondHalf => GetString();
            // 声明静态属性 Score，返回 GetString() 方法的结果
            public static string Score => GetString();
        }

        // 声明私有静态方法 GetString，参数为 [CallerMemberName] string? name = null
        private static string GetString([CallerMemberName] string? name = null)
        {
            // 使用 GetStream(name) 方法获取流
            using var stream = GetStream(name);
            // 使用流创建 StreamReader 对象
            using var reader = new StreamReader(stream);
            // 返回读取的文本内容
            return reader.ReadToEnd();
        }

        // 声明私有静态方法 GetStream，参数为 [CallerMemberName] string? name = null
        private static Stream GetStream([CallerMemberName] string? name = null) =>
            // 获取当前执行程序集的嵌入资源流
            Assembly.GetExecutingAssembly().GetManifestResourceStream($"Basketball.Resources.{name}.txt")
                // 如果找不到资源流，则抛出异常
                ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
    }
}
```