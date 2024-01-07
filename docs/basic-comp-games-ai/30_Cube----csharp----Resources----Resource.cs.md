# `basic-computer-games\30_Cube\csharp\Resources\Resource.cs`

```

// 使用 System.Reflection 和 System.Runtime.CompilerServices 命名空间
using System.Reflection;
using System.Runtime.CompilerServices;

// 声明 Cube.Resources 命名空间
namespace Cube.Resources
{
    // 声明 Resource 类
    internal static class Resource
    {
        // 声明 Streams 类
        internal static class Streams
        {
            // 声明并初始化静态属性 Introduction，Instructions，Wager 等，它们都返回 GetStream() 方法的结果
            public static Stream Introduction => GetStream();
            public static Stream Instructions => GetStream();
            public static Stream Wager => GetStream();
            public static Stream IllegalMove => GetStream();
            public static Stream Bang => GetStream();
            public static Stream Bust => GetStream();
            public static Stream Congratulations => GetStream();
            public static Stream Goodbye => GetStream();
        }

        // 声明 Prompts 类
        internal static class Prompts
        {
            // 声明并初始化静态属性 HowMuch，BetAgain，YourMove 等，它们都返回 GetString() 方法的结果
            public static string HowMuch => GetString();
            public static string BetAgain => GetString();
            public static string YourMove => GetString();
            public static string NextMove => GetString();
            public static string TryAgain => GetString();
        }

        // 声明 Formats 类
        internal static class Formats
        {
            // 声明并初始化静态属性 Balance，它返回 GetString() 方法的结果
            public static string Balance => GetString();
        }

        // 声明私有的 GetString 方法，使用 [CallerMemberName] 特性获取调用者的成员名
        private static string GetString([CallerMemberName] string? name = null)
        {
            // 使用 GetStream 方法获取流
            using var stream = GetStream(name);
            // 使用 StreamReader 读取流内容并返回
            using var reader = new StreamReader(stream);
            return reader.ReadToEnd();
        }

        // 声明私有的 GetStream 方法，使用 [CallerMemberName] 特性获取调用者的成员名
        private static Stream GetStream([CallerMemberName] string? name = null) =>
            // 使用 Assembly.GetExecutingAssembly().GetManifestResourceStream 获取嵌入资源流
            Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
                // 如果找不到资源流，则抛出异常
                ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
    }
}

```