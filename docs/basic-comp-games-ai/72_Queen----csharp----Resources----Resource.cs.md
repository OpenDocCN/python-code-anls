# `basic-computer-games\72_Queen\csharp\Resources\Resource.cs`

```
// 声明命名空间 Queen.Resources
namespace Queen.Resources
{
    // 声明内部静态类 Resource
    internal static class Resource
    {
        // 声明内部静态类 Streams
        internal static class Streams
        {
            // 声明静态属性 Title，返回 GetStream() 方法的结果
            public static Stream Title => GetStream();
            // 声明静态属性 Instructions，返回 GetStream() 方法的结果
            public static Stream Instructions => GetStream();
            // 声明静态属性 YesOrNo，返回 GetStream() 方法的结果
            public static Stream YesOrNo => GetStream();
            // 声明静态属性 Board，返回 GetStream() 方法的结果
            public static Stream Board => GetStream();
            // 声明静态属性 IllegalStart，返回 GetStream() 方法的结果
            public static Stream IllegalStart => GetStream();
            // 声明静态属性 IllegalMove，返回 GetStream() 方法的结果
            public static Stream IllegalMove => GetStream();
            // 声明静态属性 Forfeit，返回 GetStream() 方法的结果
            public static Stream Forfeit => GetStream();
            // 声明静态属性 IWin，返回 GetStream() 方法的结果
            public static Stream IWin => GetStream();
            // 声明静态属性 Congratulations，返回 GetStream() 方法的结果
            public static Stream Congratulations => GetStream();
            // 声明静态属性 Thanks，返回 GetStream() 方法的结果
            public static Stream Thanks => GetStream();
        }

        // 声明内部静态类 Prompts
        internal static class Prompts
        {
            // 声明静态属性 Instructions，返回 GetPrompt() 方法的结果
            public static string Instructions => GetPrompt();
            // 声明静态属性 Start，返回 GetPrompt() 方法的结果
            public static string Start => GetPrompt();
            // 声明静态属性 Move，返回 GetPrompt() 方法的结果
            public static string Move => GetPrompt();
            // 声明静态属性 Anyone，返回 GetPrompt() 方法的结果
            public static string Anyone => GetPrompt();
        }

        // 声明内部静态类 Strings
        internal static class Strings
        {
            // 声明静态方法 ComputerMove，接受 Position 参数，返回 GetString() 方法的结果
            public static string ComputerMove(Position position) => string.Format(GetString(), position);
        }

        // 声明私有静态方法 GetPrompt，接受可空的 string 类型参数 name，默认值为 null，返回 GetString() 方法的结果
        private static string GetPrompt([CallerMemberName] string? name = null) => GetString($"{name}Prompt");

        // 声明私有静态方法 GetString，接受可空的 string 类型参数 name，默认值为 null，返回读取嵌入资源流的结果
        private static string GetString([CallerMemberName] string? name = null)
        {
            // 使用嵌入资源流创建 StreamReader 对象
            using var stream = GetStream(name);
            using var reader = new StreamReader(stream);
            // 返回读取的文本内容
            return reader.ReadToEnd();
        }

        // 声明私有静态方法 GetStream，接受可空的 string 类型参数 name，默认值为 null，返回嵌入资源流
        private static Stream GetStream([CallerMemberName] string? name = null) =>
            // 获取当前执行程序集的嵌入资源流
            Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
                // 如果找不到嵌入资源流，则抛出异常
                ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
    }
}
```