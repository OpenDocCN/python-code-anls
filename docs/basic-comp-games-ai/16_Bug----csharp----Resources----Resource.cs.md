# `basic-computer-games\16_Bug\csharp\Resources\Resource.cs`

```

// 使用 System.Reflection 和 System.Runtime.CompilerServices 命名空间
using System.Reflection;
using System.Runtime.CompilerServices;

// 声明 BugGame.Resources 命名空间
namespace BugGame.Resources
{
    // 声明 Resource 类
    internal static class Resource
    {
        // 声明 Streams 类
        internal static class Streams
        {
            // 声明 Introduction 属性，返回 GetStream() 方法的结果
            public static Stream Introduction => GetStream();
            // 声明 Instructions 属性，返回 GetStream() 方法的结果
            public static Stream Instructions => GetStream();
            // 声明 PlayAgain 属性，返回 GetStream() 方法的结果
            public static Stream PlayAgain => GetStream();
        }

        // 声明 GetStream 方法，使用 CallerMemberName 特性获取方法名，默认为 null
        private static Stream GetStream([CallerMemberName] string? name = null) =>
            // 获取当前执行程序集，获取嵌入资源流
            Assembly.GetExecutingAssembly()
                .GetManifestResourceStream($"Bug.Resources.{name}.txt")
                // 如果找不到资源流，抛出异常
                ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
    }
}

```