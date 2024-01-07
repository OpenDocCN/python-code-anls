# `basic-computer-games\13_Bounce\csharp\Resources\Resource.cs`

```

// 使用 System.Reflection 和 System.Runtime.CompilerServices 命名空间
using System.Reflection;
using System.Runtime.CompilerServices;

// 声明 Bounce.Resources 命名空间
namespace Bounce.Resources
{
    // 声明 Resource 类
    internal static class Resource
    {
        // 声明 Streams 类
        internal static class Streams
        {
            // 声明 Instructions 属性，返回 GetStream() 方法的结果
            public static Stream Instructions => GetStream();
            // 声明 Title 属性，返回 GetStream() 方法的结果
            public static Stream Title => GetStream();
        }

        // 声明 GetStream 方法，使用 CallerMemberName 特性获取调用方法的名称，默认为 null
        private static Stream GetStream([CallerMemberName] string? name = null)
            // 获取当前程序集的嵌入资源流，资源名称为 "Bounce.Resources.{name}.txt"
            => Assembly.GetExecutingAssembly().GetManifestResourceStream($"Bounce.Resources.{name}.txt")
                // 如果资源流不存在，则抛出 ArgumentException 异常
                ?? throw new ArgumentException($"Resource stream {name} does not exist", nameof(name));
    }
}

```