# `basic-computer-games\34_Digits\csharp\Resources\Resource.cs`

```py
// 声明命名空间 Digits.Resources
namespace Digits.Resources
{
    // 声明内部静态类 Resource
    internal static class Resource
    {
        // 声明内部静态类 Streams
        internal static class Streams
        {
            // 声明静态属性 Introduction，返回 GetStream() 方法的结果
            public static Stream Introduction => GetStream();
            // 声明静态属性 Instructions，返回 GetStream() 方法的结果
            public static Stream Instructions => GetStream();
            // 声明静态属性 TryAgain，返回 GetStream() 方法的结果
            public static Stream TryAgain => GetStream();
            // 声明静态属性 ItsATie，返回 GetStream() 方法的结果
            public static Stream ItsATie => GetStream();
            // 声明静态属性 IWin，返回 GetStream() 方法的结果
            public static Stream IWin => GetStream();
            // 声明静态属性 YouWin，返回 GetStream() 方法的结果
            public static Stream YouWin => GetStream();
            // 声明静态属性 Thanks，返回 GetStream() 方法的结果
            public static Stream Thanks => GetStream();
            // 声明静态属性 Headings，返回 GetStream() 方法的结果
            public static Stream Headings => GetStream();
        }

        // 声明内部静态类 Prompts
        internal static class Prompts
        {
            // 声明静态属性 ForInstructions，返回 GetString() 方法的结果
            public static string ForInstructions => GetString();
            // 声明静态属性 TenNumbers，返回 GetString() 方法的结果
            public static string TenNumbers => GetString();
            // 声明静态属性 WantToTryAgain，返回 GetString() 方法的结果
            public static string WantToTryAgain => GetString();
        }

        // 声明内部静态类 Formats
        internal static class Formats
        {
            // 声明静态属性 GuessResult，返回 GetString() 方法的结果
            public static string GuessResult => GetString();
        }

        // 声明私有静态方法 GetString，参数为调用者成员名，返回字符串
        private static string GetString([CallerMemberName] string? name = null)
        {
            // 使用 GetStream 方法获取流
            using var stream = GetStream(name);
            // 使用流创建读取器
            using var reader = new StreamReader(stream);
            // 返回读取器读取的所有内容
            return reader.ReadToEnd();
        }

        // 声明私有静态方法 GetStream，参数为调用者成员名，返回流
        private static Stream GetStream([CallerMemberName] string? name = null) =>
            // 获取当前执行程序集的嵌入资源流
            Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
                // 如果找不到资源流，则抛出异常
                ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
    }
}
```