# `basic-computer-games\56_Life_for_Two\csharp\Resources\Resource.cs`

```

// 使用 System.Reflection 和 System.Runtime.CompilerServices 命名空间
using System.Reflection;
using System.Runtime.CompilerServices;

// LifeforTwo.Resources 命名空间
namespace LifeforTwo.Resources
{
    // Resource 类
    internal static class Resource
    {
        // Streams 类
        internal static class Streams
        {
            // 获取标题流
            public static Stream Title => GetStream();
            // 获取非法坐标流
            public static Stream IllegalCoords => GetStream();
            // 获取相同坐标流
            public static Stream SameCoords => GetStream();
        }

        // Formats 类
        internal static class Formats
        {
            // 获取初始棋子格式
            public static string InitialPieces => GetString();
            // 获取玩家格式
            public static string Player => GetString();
            // 获取获胜者格式
            public static string Winner => GetString();
        }

        // Strings 类
        internal static class Strings
        {
            // 获取平局字符串
            public static string Draw => GetString();
        }

        // 根据成员名获取字符串
        private static string GetString([CallerMemberName] string? name = null)
        {
            // 获取流
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