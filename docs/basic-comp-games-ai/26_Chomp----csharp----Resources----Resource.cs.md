# `basic-computer-games\26_Chomp\csharp\Resources\Resource.cs`

```
// 声明命名空间 Chomp.Resources
namespace Chomp.Resources
{
    // 声明内部静态类 Resource
    internal static class Resource
    {
        // 声明内部静态类 Streams
        internal static class Streams
        {
            // 声明公共静态属性 HereWeGo，返回流对象
            public static Stream HereWeGo => GetStream();
            // 声明公共静态属性 Introduction，返回流对象
            public static Stream Introduction => GetStream();
            // 声明公共静态属性 Rules，返回流对象
            public static Stream Rules => GetStream();
            // 声明公共静态属性 NoFair，返回流对象
            public static Stream NoFair => GetStream();
        }

        // 声明内部静态类 Formats
        internal static class Formats
        {
            // 声明公共静态属性 Player，返回字符串
            public static string Player => GetString();
            // 声明公共静态属性 YouLose，返回字符串
            public static string YouLose => GetString();
        }

        // 声明内部静态类 Prompts
        internal static class Prompts
        {
            // 声明公共静态属性 Coordinates，返回字符串
            public static string Coordinates => GetString();
            // 声明公共静态属性 HowManyPlayers，返回字符串
            public static string HowManyPlayers => GetString();
            // 声明公共静态属性 HowManyRows，返回字符串
            public static string HowManyRows => GetString();
            // 声明公共静态属性 HowManyColumns，返回字符串
            public static string HowManyColumns => GetString();
            // 声明公共静态属性 TooManyColumns，返回字符串
            public static string TooManyColumns => GetString();
        }

        // 声明内部静态类 Strings
        internal static class Strings
        {
            // 声明公共静态属性 TooManyColumns，返回字符串
            public static string TooManyColumns => GetString();
            // 声明公共静态属性 TooManyRows，返回字符串
            public static string TooManyRows => GetString();
        }

        // 声明私有静态方法 GetString，返回字符串，参数为调用者成员名
        private static string GetString([CallerMemberName] string? name = null)
        {
            // 使用 GetStream 方法获取流对象
            using var stream = GetStream(name);
            // 使用流对象创建 StreamReader 对象
            using var reader = new StreamReader(stream);
            // 读取流中的所有内容并返回
            return reader.ReadToEnd();
        }

        // 声明私有静态方法 GetStream，返回流对象，参数为调用者成员名
        private static Stream GetStream([CallerMemberName] string? name = null) =>
            // 获取当前程序集的嵌入资源流对象，资源路径为命名空间.成员名.txt
            Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
                // 如果资源流对象为空，则抛出异常
                ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
    }
}
```