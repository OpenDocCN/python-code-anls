# `basic-computer-games\30_Cube\csharp\Resources\Resource.cs`

```
// 声明命名空间 Cube.Resources
namespace Cube.Resources
{
    // 声明内部静态类 Resource
    internal static class Resource
    {
        // 声明内部静态类 Streams
        internal static class Streams
        {
            // 声明静态属性 Introduction，返回流对象
            public static Stream Introduction => GetStream();
            // 声明静态属性 Instructions，返回流对象
            public static Stream Instructions => GetStream();
            // 声明静态属性 Wager，返回流对象
            public static Stream Wager => GetStream();
            // 声明静态属性 IllegalMove，返回流对象
            public static Stream IllegalMove => GetStream();
            // 声明静态属性 Bang，返回流对象
            public static Stream Bang => GetStream();
            // 声明静态属性 Bust，返回流对象
            public static Stream Bust => GetStream();
            // 声明静态属性 Congratulations，返回流对象
            public static Stream Congratulations => GetStream();
            // 声明静态属性 Goodbye，返回流对象
            public static Stream Goodbye => GetStream();
        }

        // 声明内部静态类 Prompts
        internal static class Prompts
        {
            // 声明静态属性 HowMuch，返回字符串
            public static string HowMuch => GetString();
            // 声明静态属性 BetAgain，返回字符串
            public static string BetAgain => GetString();
            // 声明静态属性 YourMove，返回字符串
            public static string YourMove => GetString();
            // 声明静态属性 NextMove，返回字符串
            public static string NextMove => GetString();
            // 声明静态属性 TryAgain，返回字符串
            public static string TryAgain => GetString();
        }

        // 声明内部静态类 Formats
        internal static class Formats
        {
            // 声明静态属性 Balance，返回字符串
            public static string Balance => GetString();
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
            // 获取当前执行程序集，根据资源名称获取嵌入资源流对象
            Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
                // 如果找不到资源流，则抛出异常
                ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
    }
}
```