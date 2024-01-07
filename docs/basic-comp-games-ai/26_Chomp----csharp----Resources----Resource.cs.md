# `basic-computer-games\26_Chomp\csharp\Resources\Resource.cs`

```

// 使用 System.Reflection 和 System.Runtime.CompilerServices 命名空间
using System.Reflection;
using System.Runtime.CompilerServices;

// 声明 Chomp.Resources 命名空间
namespace Chomp.Resources;

// 声明 Resource 类
internal static class Resource
{
    // 声明 Streams 类
    internal static class Streams
    {
        // 声明 HereWeGo 属性，返回一个流
        public static Stream HereWeGo => GetStream();
        // 声明 Introduction 属性，返回一个流
        public static Stream Introduction => GetStream();
        // 声明 Rules 属性，返回一个流
        public static Stream Rules => GetStream();
        // 声明 NoFair 属性，返回一个流
        public static Stream NoFair => GetStream();
    }

    // 声明 Formats 类
    internal static class Formats
    {
        // 声明 Player 属性，返回一个字符串
        public static string Player => GetString();
        // 声明 YouLose 属性，返回一个字符串
        public static string YouLose => GetString();
    }

    // 声明 Prompts 类
    internal static class Prompts
    {
        // 声明 Coordinates 属性，返回一个字符串
        public static string Coordinates => GetString();
        // 声明 HowManyPlayers 属性，返回一个字符串
        public static string HowManyPlayers => GetString();
        // 声明 HowManyRows 属性，返回一个字符串
        public static string HowManyRows => GetString();
        // 声明 HowManyColumns 属性，返回一个字符串
        public static string HowManyColumns => GetString();
        // 声明 TooManyColumns 属性，返回一个字符串
        public static string TooManyColumns => GetString();
    }

    // 声明 Strings 类
    internal static class Strings
    {
        // 声明 TooManyColumns 属性，返回一个字符串
        public static string TooManyColumns => GetString();
        // 声明 TooManyRows 属性，返回一个字符串
        public static string TooManyRows => GetString();
    }

    // 声明 GetString 方法，返回一个字符串，使用 CallerMemberName 特性
    private static string GetString([CallerMemberName] string? name = null)
    {
        // 使用 GetStream 方法获取流
        using var stream = GetStream(name);
        // 使用 StreamReader 读取流内容并返回
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    // 声明 GetStream 方法，返回一个流，使用 CallerMemberName 特性
    private static Stream GetStream([CallerMemberName] string? name = null) =>
        // 使用 Assembly.GetExecutingAssembly().GetManifestResourceStream 获取嵌入资源流
        Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
            // 如果找不到资源流，则抛出异常
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
}

```