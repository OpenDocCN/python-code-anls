# `basic-computer-games\07_Basketball\csharp\Resources\Resource.cs`

```

// 使用 System.Reflection 和 System.Runtime.CompilerServices 命名空间
using System.Reflection;
using System.Runtime.CompilerServices;

// 声明命名空间 Basketball.Resources
namespace Basketball.Resources
{
    // 声明内部静态类 Resource
    internal static class Resource
    {
        // 声明内部静态类 Streams
        internal static class Streams
        {
            // 声明公共静态属性 Introduction，返回流对象
            public static Stream Introduction => GetStream();
            // 声明公共静态属性 TwoMinutesLeft，返回流对象
            public static Stream TwoMinutesLeft => GetStream();
        }

        // 声明内部静态类 Formats
        internal static class Formats
        {
            // 声明公共静态属性 EndOfFirstHalf，返回字符串
            public static string EndOfFirstHalf => GetString();
            // 声明公共静态属性 EndOfGame，返回字符串
            public static string EndOfGame => GetString();
            // 声明公共静态属性 EndOfSecondHalf，返回字符串
            public static string EndOfSecondHalf => GetString();
            // 声明公共静态属性 Score，返回字符串
            public static string Score => GetString();
        }

        // 声明私有静态方法 GetString，返回字符串，使用 CallerMemberName 特性获取方法名
        private static string GetString([CallerMemberName] string? name = null)
        {
            // 使用 GetStream 方法获取流对象
            using var stream = GetStream(name);
            // 使用流对象创建 StreamReader 对象
            using var reader = new StreamReader(stream);
            // 读取流中的所有内容并返回
            return reader.ReadToEnd();
        }

        // 声明私有静态方法 GetStream，返回流对象，使用 CallerMemberName 特性获取方法名
        private static Stream GetStream([CallerMemberName] string? name = null) =>
            // 获取当前程序集中嵌入的资源流对象
            Assembly.GetExecutingAssembly().GetManifestResourceStream($"Basketball.Resources.{name}.txt")
                // 如果资源流对象为空，则抛出异常
                ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
    }
}

```