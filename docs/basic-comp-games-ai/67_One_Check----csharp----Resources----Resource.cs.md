# `basic-computer-games\67_One_Check\csharp\Resources\Resource.cs`

```py
// 使用 System.Reflection 和 System.Runtime.CompilerServices 命名空间
using System.Reflection;
using System.Runtime.CompilerServices;

// 声明命名空间 OneCheck.Resources
namespace OneCheck.Resources
{
    // 声明内部静态类 Resource
    internal static class Resource
    {
        // 声明内部静态类 Streams
        internal static class Streams
        {
            // 声明公共静态属性 Introduction，返回 GetStream() 方法的结果
            public static Stream Introduction => GetStream();
            // 声明公共静态属性 IllegalMove，返回 GetStream() 方法的结果
            public static Stream IllegalMove => GetStream();
            // 声明公共静态属性 YesOrNo，返回 GetStream() 方法的结果
            public static Stream YesOrNo => GetStream();
            // 声明公共静态属性 Bye，返回 GetStream() 方法的结果
            public static Stream Bye => GetStream();
        }

        // 声明内部静态类 Formats
        internal static class Formats
        {
            // 声明公共静态属性 Results，返回 GetString() 方法的结果
            public static string Results => GetString();
        }

        // 声明内部静态类 Prompts
        internal static class Prompts
        {
            // 声明公共静态属性 From，返回 GetString() 方法的结果
            public static string From => GetString();
            // 声明公共静态属性 To，返回 GetString() 方法的结果
            public static string To => GetString();
            // 声明公共静态属性 TryAgain，返回 GetString() 方法的结果
            public static string TryAgain => GetString();
        }

        // 声明内部静态类 Strings
        internal static class Strings
        {
            // 声明公共静态属性 TooManyColumns，返回 GetString() 方法的结果
            public static string TooManyColumns => GetString();
            // 声明公共静态属性 TooManyRows，返回 GetString() 方法的结果
            public static string TooManyRows => GetString();
        }

        // 声明私有静态方法 GetString，参数 name 默认为 null，使用 CallerMemberName 特性
        private static string GetString([CallerMemberName] string? name = null)
        {
            // 使用 GetStream 方法获取流
            using var stream = GetStream(name);
            // 使用流创建 StreamReader 对象
            using var reader = new StreamReader(stream);
            // 读取流中的所有内容并返回
            return reader.ReadToEnd();
        }

        // 声明私有静态方法 GetStream，参数 name 默认为 null，使用 CallerMemberName 特性
        private static Stream GetStream([CallerMemberName] string? name = null) =>
            // 获取当前执行程序集中嵌入的资源流
            Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
                // 如果资源流为空，则抛出异常
                ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
    }
}
```