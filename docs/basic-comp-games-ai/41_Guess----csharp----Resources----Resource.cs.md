# `basic-computer-games\41_Guess\csharp\Resources\Resource.cs`

```
// 使用 System.Reflection 和 System.Runtime.CompilerServices 命名空间
using System.Reflection;
using System.Runtime.CompilerServices;

// 声明 Guess.Resources 命名空间
namespace Guess.Resources
{
    // 声明 Resource 类
    internal static class Resource
    {
        // 声明 Streams 类
        internal static class Streams
        {
            // 获取 Introduction 的流
            public static Stream Introduction => GetStream();
            // 获取 TooLow 的流
            public static Stream TooLow => GetStream();
            // 获取 TooHigh 的流
            public static Stream TooHigh => GetStream();
            // 获取 BlankLines 的流
            public static Stream BlankLines => GetStream();
        }

        // 声明 Formats 类
        internal static class Formats
        {
            // 获取 Thinking 的字符串
            public static string Thinking => GetString();
            // 获取 ThatsIt 的字符串
            public static string ThatsIt => GetString();
            // 获取 ShouldHave 的字符串
            public static string ShouldHave => GetString();
        }

        // 声明 Prompts 类
        internal static class Prompts
        {
            // 获取 Limit 的字符串
            public static string Limit => GetString();
        }

        // 声明 Strings 类
        internal static class Strings
        {
            // 获取 Good 的字符串
            public static string Good => GetString();
            // 获取 VeryGood 的字符串
            public static string VeryGood => GetString();
        }

        // 根据成员名获取字符串
        private static string GetString([CallerMemberName] string? name = null)
        {
            // 使用成员名获取流
            using var stream = GetStream(name);
            // 使用流创建读取器
            using var reader = new StreamReader(stream);
            // 读取并返回字符串
            return reader.ReadToEnd();
        }

        // 根据成员名获取流
        private static Stream GetStream([CallerMemberName] string? name = null) =>
            // 获取嵌入资源流
            Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
                // 如果找不到资源流，则抛出异常
                ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
    }
}
```