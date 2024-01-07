# `basic-computer-games\71_Poker\csharp\Resources\Resource.cs`

```

// 使用 System.Reflection 和 System.Runtime.CompilerServices 命名空间
using System.Reflection;
using System.Runtime.CompilerServices;

// 声明名为 Resource 的内部静态类
namespace Poker.Resources;

internal static class Resource
{
    // 声明名为 Streams 的内部静态类
    internal static class Streams
    {
        // 声明名为 Instructions 的公共静态属性，返回 GetStream() 方法的结果
        public static Stream Instructions => GetStream();
        // 声明名为 Title 的公共静态属性，返回 GetStream() 方法的结果
        public static Stream Title => GetStream();
    }

    // 声明名为 GetStream 的私有静态方法，使用 CallerMemberName 特性获取调用者的名称，默认为 null
    private static Stream GetStream([CallerMemberName] string? name = null)
        // 使用 Assembly.GetExecutingAssembly().GetManifestResourceStream() 方法获取资源流
        => Assembly.GetExecutingAssembly().GetManifestResourceStream($"Poker.Resources.{name}.txt")
            // 如果资源流不存在，则抛出 ArgumentException 异常
            ?? throw new ArgumentException($"Resource stream {name} does not exist", nameof(name));
}

```