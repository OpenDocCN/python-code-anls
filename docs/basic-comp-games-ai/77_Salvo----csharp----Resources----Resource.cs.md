# `basic-computer-games\77_Salvo\csharp\Resources\Resource.cs`

```

// 使用 System.Reflection 和 System.Runtime.CompilerServices 命名空间
using System.Reflection;
using System.Runtime.CompilerServices;

// 声明内部静态类 Resource
namespace Salvo.Resources;
internal static class Resource
{
    // 声明内部静态类 Streams
    internal static class Streams
    {
        // 声明静态属性 Title，返回一个流
        public static Stream Title => GetStream();
        // 声明静态属性 YouHaveMoreShotsThanSquares，返回一个流
        public static Stream YouHaveMoreShotsThanSquares => GetStream();
        // 声明静态属性 YouWon，返回一个流
        public static Stream YouWon => GetStream();
        // 声明静态属性 IHaveMoreShotsThanSquares，返回一个流
        public static Stream IHaveMoreShotsThanSquares => GetStream();
        // 声明静态属性 IWon，返回一个流
        public static Stream IWon => GetStream();
        // 声明静态属性 Illegal，返回一个流
        public static Stream Illegal => GetStream();
    }

    // 声明内部静态类 Strings
    internal static class Strings
    {
        // 声明静态方法 WhereAreYourShips，返回一个字符串
        public static string WhereAreYourShips => GetString();
        // 声明静态方法 YouHaveShots，返回一个格式化后的字符串
        public static string YouHaveShots(int number) => Format(number);
        // 声明静态方法 IHaveShots，返回一个格式化后的字符串
        public static string IHaveShots(int number) => Format(number);
        // 声明静态方法 YouHit，返回一个格式化后的字符串
        public static string YouHit(string shipName) => Format(shipName);
        // 声明静态方法 IHit，返回一个格式化后的字符串
        public static string IHit(string shipName) => Format(shipName);
        // 声明静态方法 ShotBefore，返回一个格式化后的字符串
        public static string ShotBefore(int turnNumber) => Format(turnNumber);
        // 声明静态方法 Turn，返回一个格式化后的字符串
        public static string Turn(int number) => Format(number);
    }

    // 声明内部静态类 Prompts
    internal static class Prompts
    {
        // 声明静态属性 Coordinates，返回一个字符串
        public static string Coordinates => GetString();
        // 声明静态属性 Start，返回一个字符串
        public static string Start => GetString();
        // 声明静态属性 SeeShots，返回一个字符串
        public static string SeeShots => GetString();
    }

    // 声明私有静态方法 Format，返回一个格式化后的字符串
    private static string Format<T>(T value, [CallerMemberName] string? name = null) 
        => string.Format(GetString(name), value);

    // 声明私有静态方法 GetString，返回一个字符串
    private static string GetString([CallerMemberName] string? name = null)
    {
        // 使用资源流创建一个读取器，返回读取的内容
        using var stream = GetStream(name);
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    // 声明私有静态方法 GetStream，返回一个流
    private static Stream GetStream([CallerMemberName] string? name = null) =>
        // 获取当前程序集的嵌入资源流，如果不存在则抛出异常
        Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
}

```