# `basic-computer-games\72_Queen\csharp\Resources\Resource.cs`

```

// 使用 System.Reflection 和 System.Runtime.CompilerServices 命名空间
using System.Reflection;
using System.Runtime.CompilerServices;

// Queen.Resources 命名空间
namespace Queen.Resources
{
    // Resource 类
    internal static class Resource
    {
        // Streams 类
        internal static class Streams
        {
            // 获取标题流
            public static Stream Title => GetStream();
            // 获取说明流
            public static Stream Instructions => GetStream();
            // 获取YesOrNo流
            public static Stream YesOrNo => GetStream();
            // 获取棋盘流
            public static Stream Board => GetStream();
            // 获取非法开始流
            public static Stream IllegalStart => GetStream();
            // 获取非法移动流
            public static Stream IllegalMove => GetStream();
            // 获取放弃流
            public static Stream Forfeit => GetStream();
            // 获取我赢了流
            public static Stream IWin => GetStream();
            // 获取祝贺流
            public static Stream Congratulations => GetStream();
            // 获取感谢流
            public static Stream Thanks => GetStream();
        }

        // Prompts 类
        internal static class Prompts
        {
            // 获取说明提示
            public static string Instructions => GetPrompt();
            // 获取开始提示
            public static string Start => GetPrompt();
            // 获取移动提示
            public static string Move => GetPrompt();
            // 获取任何人提示
            public static string Anyone => GetPrompt();
        }

        // Strings 类
        internal static class Strings
        {
            // 获取计算机移动字符串
            public static string ComputerMove(Position position) => string.Format(GetString(), position);
        }

        // 获取提示
        private static string GetPrompt([CallerMemberName] string? name = null) => GetString($"{name}Prompt");

        // 获取字符串
        private static string GetString([CallerMemberName] string? name = null)
        {
            using var stream = GetStream(name);
            using var reader = new StreamReader(stream);
            return reader.ReadToEnd();
        }

        // 获取流
        private static Stream GetStream([CallerMemberName] string? name = null) =>
            Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
                ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
    }
}

```