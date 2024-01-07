# `basic-computer-games\34_Digits\csharp\Resources\Resource.cs`

```

// 使用 System.Reflection 和 System.Runtime.CompilerServices 命名空间
using System.Reflection;
using System.Runtime.CompilerServices;

// 声明 Digits.Resources 命名空间
namespace Digits.Resources
{
    // 声明 Resource 类
    internal static class Resource
    {
        // 声明 Streams 类
        internal static class Streams
        {
            // 声明 Introduction 属性，返回流
            public static Stream Introduction => GetStream();
            // 声明 Instructions 属性，返回流
            public static Stream Instructions => GetStream();
            // 声明 TryAgain 属性，返回流
            public static Stream TryAgain => GetStream();
            // 声明 ItsATie 属性，返回流
            public static Stream ItsATie => GetStream();
            // 声明 IWin 属性，返回流
            public static Stream IWin => GetStream();
            // 声明 YouWin 属性，返回流
            public static Stream YouWin => GetStream();
            // 声明 Thanks 属性，返回流
            public static Stream Thanks => GetStream();
            // 声明 Headings 属性，返回流
            public static Stream Headings => GetStream();
        }

        // 声明 Prompts 类
        internal static class Prompts
        {
            // 声明 ForInstructions 属性，返回字符串
            public static string ForInstructions => GetString();
            // 声明 TenNumbers 属性，返回字符串
            public static string TenNumbers => GetString();
            // 声明 WantToTryAgain 属性，返回字符串
            public static string WantToTryAgain => GetString();
        }

        // 声明 Formats 类
        internal static class Formats
        {
            // 声明 GuessResult 属性，返回字符串
            public static string GuessResult => GetString();
        }

        // 声明 GetString 方法，返回字符串，使用 CallerMemberName 特性
        private static string GetString([CallerMemberName] string? name = null)
        {
            // 获取流
            using var stream = GetStream(name);
            // 使用流创建读取器
            using var reader = new StreamReader(stream);
            // 读取并返回流内容
            return reader.ReadToEnd();
        }

        // 声明 GetStream 方法，返回流，使用 CallerMemberName 特性
        private static Stream GetStream([CallerMemberName] string? name = null) =>
            // 获取当前程序集的嵌入资源流
            Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
                // 如果找不到资源流，抛出异常
                ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
    }
}

```