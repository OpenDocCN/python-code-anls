# `basic-computer-games\25_Chief\csharp\Resources\Resource.cs`

```

// 使用 System.Reflection 和 System.Runtime.CompilerServices 命名空间
using System.Reflection;
using System.Runtime.CompilerServices;

// 定义 Chief.Resources 命名空间
namespace Chief.Resources
{
    // 定义 Resource 类
    internal static class Resource
    {
        // 定义 Streams 类
        internal static class Streams
        {
            // 获取 Bye 流
            public static Stream Bye => GetStream();
            // 获取 Instructions 流
            public static Stream Instructions => GetStream();
            // 获取 Lightning 流
            public static Stream Lightning => GetStream();
            // 获取 ShutUp 流
            public static Stream ShutUp => GetStream();
            // 获取 Title 流
            public static Stream Title => GetStream();
        }

        // 定义 Formats 类
        internal static class Formats
        {
            // 获取 Bet 字符串
            public static string Bet => GetString();
            // 获取 Working 字符串
            public static string Working => GetString();
        }

        // 定义 Prompts 类
        internal static class Prompts
        {
            // 获取 Answer 字符串
            public static string Answer => GetString();
            // 获取 Believe 字符串
            public static string Believe => GetString();
            // 获取 Original 字符串
            public static string Original => GetString();
            // 获取 Ready 字符串
            public static string Ready => GetString();
        }

        // 根据调用成员的名称获取字符串
        private static string GetString([CallerMemberName] string? name = null)
        {
            // 获取对应名称的流
            using var stream = GetStream(name);
            // 使用流创建读取器
            using var reader = new StreamReader(stream);
            // 读取流中的内容并返回
            return reader.ReadToEnd();
        }

        // 根据调用成员的名称获取流
        private static Stream GetStream([CallerMemberName] string? name = null)
            // 从当前执行的程序集中获取资源流
            => Assembly.GetExecutingAssembly().GetManifestResourceStream($"Chief.Resources.{name}.txt")
                // 如果资源流不存在则抛出异常
                ?? throw new ArgumentException($"Resource stream {name} does not exist", nameof(name));
    }
}

```