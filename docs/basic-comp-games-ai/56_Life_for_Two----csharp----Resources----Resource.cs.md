# `basic-computer-games\56_Life_for_Two\csharp\Resources\Resource.cs`

```
// 使用 System.Reflection 和 System.Runtime.CompilerServices 命名空间
using System.Reflection;
using System.Runtime.CompilerServices;

// 声明 LifeforTwo.Resources 命名空间
namespace LifeforTwo.Resources
{
    // 声明 Resource 类
    internal static class Resource
    {
        // 声明 Streams 类
        internal static class Streams
        {
            // 声明 Title 属性，返回流
            public static Stream Title => GetStream();
            // 声明 IllegalCoords 属性，返回流
            public static Stream IllegalCoords => GetStream();
            // 声明 SameCoords 属性，返回流
            public static Stream SameCoords => GetStream();
        }

        // 声明 Formats 类
        internal static class Formats
        {
            // 声明 InitialPieces 属性，返回字符串
            public static string InitialPieces => GetString();
            // 声明 Player 属性，返回字符串
            public static string Player => GetString();
            // 声明 Winner 属性，返回字符串
            public static string Winner => GetString();
        }

        // 声明 Strings 类
        internal static class Strings
        {
            // 声明 Draw 属性，返回字符串
            public static string Draw => GetString();
        }

        // 声明 GetString 方法，返回字符串，参数为调用者成员名
        private static string GetString([CallerMemberName] string? name = null)
        {
            // 使用调用者成员名获取流
            using var stream = GetStream(name);
            // 使用流创建读取器
            using var reader = new StreamReader(stream);
            // 读取并返回读取器的所有内容
            return reader.ReadToEnd();
        }

        // 声明 GetStream 方法，返回流，参数为调用者成员名
        private static Stream GetStream([CallerMemberName] string? name = null) =>
            // 获取当前执行程序集，并根据命名空间和成员名获取嵌入资源流
            Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
                // 如果找不到资源流，则抛出异常
                ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
    }
}
```