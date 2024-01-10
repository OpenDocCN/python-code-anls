# `basic-computer-games\25_Chief\csharp\Resources\Resource.cs`

```
// 声明命名空间 Chief.Resources
namespace Chief.Resources
{
    // 声明内部静态类 Resource
    internal static class Resource
    {
        // 声明内部静态类 Streams
        internal static class Streams
        {
            // 声明 Bye 属性，返回流对象
            public static Stream Bye => GetStream();
            // 声明 Instructions 属性，返回流对象
            public static Stream Instructions => GetStream();
            // 声明 Lightning 属性，返回流对象
            public static Stream Lightning => GetStream();
            // 声明 ShutUp 属性，返回流对象
            public static Stream ShutUp => GetStream();
            // 声明 Title 属性，返回流对象
            public static Stream Title => GetStream();
        }

        // 声明内部静态类 Formats
        internal static class Formats
        {
            // 声明 Bet 属性，返回字符串
            public static string Bet => GetString();
            // 声明 Working 属性，返回字符串
            public static string Working => GetString();
        }

        // 声明内部静态类 Prompts
        internal static class Prompts
        {
            // 声明 Answer 属性，返回字符串
            public static string Answer => GetString();
            // 声明 Believe 属性，返回字符串
            public static string Believe => GetString();
            // 声明 Original 属性，返回字符串
            public static string Original => GetString();
            // 声明 Ready 属性，返回字符串
            public static string Ready => GetString();
        }

        // 声明私有方法 GetString，返回字符串，参数为调用者成员名
        private static string GetString([CallerMemberName] string? name = null)
        {
            // 使用 GetStream 方法获取流对象
            using var stream = GetStream(name);
            // 使用流对象创建 StreamReader 对象
            using var reader = new StreamReader(stream);
            // 读取流中的所有内容并返回
            return reader.ReadToEnd();
        }

        // 声明私有方法 GetStream，返回流对象，参数为调用者成员名
        private static Stream GetStream([CallerMemberName] string? name = null)
            // 获取当前程序集的嵌入资源流对象，资源路径为 Chief.Resources.{name}.txt
            => Assembly.GetExecutingAssembly().GetManifestResourceStream($"Chief.Resources.{name}.txt")
                // 如果资源流对象不存在，则抛出参数异常
                ?? throw new ArgumentException($"Resource stream {name} does not exist", nameof(name));
    }
}
```