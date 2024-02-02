# `basic-computer-games\32_Diamond\csharp\Resources\Resource.cs`

```py
// 使用 System.Reflection 和 System.Runtime.CompilerServices 命名空间
using System.Reflection;
using System.Runtime.CompilerServices;

// 声明 Diamond.Resources 命名空间
namespace Diamond.Resources
{
    // 声明 Resource 类
    internal static class Resource
    {
        // 声明 Streams 类
        internal static class Streams
        {
            // 声明 Introduction 属性，返回流对象
            public static Stream Introduction => GetStream();
        }

        // 声明 Prompts 类
        internal static class Prompts
        {
            // 声明 TypeNumber 属性，返回字符串
            public static string TypeNumber => GetString();
        }

        // 声明 GetString 方法，返回字符串，使用 CallerMemberName 特性
        private static string GetString([CallerMemberName] string? name = null)
        {
            // 使用 GetStream 方法获取流对象
            using var stream = GetStream(name);
            // 使用流对象创建 StreamReader 对象
            using var reader = new StreamReader(stream);
            // 读取流中的所有内容并返回
            return reader.ReadToEnd();
        }

        // 声明 GetStream 方法，返回流对象，使用 CallerMemberName 特性
        private static Stream GetStream([CallerMemberName] string? name = null) =>
            // 获取当前执行程序集的嵌入资源流
            Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
                // 如果找不到资源流，则抛出异常
                ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
    }
}
```