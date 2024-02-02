# `basic-computer-games\77_Salvo\csharp\Resources\Resource.cs`

```py
// 声明命名空间 Salvo.Resources
namespace Salvo.Resources
{
    // 声明内部静态类 Resource
    internal static class Resource
    {
        // 声明内部静态类 Streams
        internal static class Streams
        {
            // 声明静态属性 Title，返回 GetStream() 方法的结果
            public static Stream Title => GetStream();
            // 声明静态属性 YouHaveMoreShotsThanSquares，返回 GetStream() 方法的结果
            public static Stream YouHaveMoreShotsThanSquares => GetStream();
            // 声明静态属性 YouWon，返回 GetStream() 方法的结果
            public static Stream YouWon => GetStream();
            // 声明静态属性 IHaveMoreShotsThanSquares，返回 GetStream() 方法的结果
            public static Stream IHaveMoreShotsThanSquares => GetStream();
            // 声明静态属性 IWon，返回 GetStream() 方法的结果
            public static Stream IWon => GetStream();
            // 声明静态属性 Illegal，返回 GetStream() 方法的结果
            public static Stream Illegal => GetStream();
        }

        // 声明内部静态类 Strings
        internal static class Strings
        {
            // 声明静态方法 WhereAreYourShips，返回 GetString() 方法的结果
            public static string WhereAreYourShips => GetString();
            // 声明静态方法 YouHaveShots，接受一个整数参数，返回 Format() 方法的结果
            public static string YouHaveShots(int number) => Format(number);
            // 声明静态方法 IHaveShots，接受一个整数参数，返回 Format() 方法的结果
            public static string IHaveShots(int number) => Format(number);
            // 声明静态方法 YouHit，接受一个字符串参数，返回 Format() 方法的结果
            public static string YouHit(string shipName) => Format(shipName);
            // 声明静态方法 IHit，接受一个字符串参数，返回 Format() 方法的结果
            public static string IHit(string shipName) => Format(shipName);
            // 声明静态方法 ShotBefore，接受一个整数参数，返回 Format() 方法的结果
            public static string ShotBefore(int turnNumber) => Format(turnNumber);
            // 声明静态方法 Turn，接受一个整数参数，返回 Format() 方法的结果
            public static string Turn(int number) => Format(number);
        }

        // 声明内部静态类 Prompts
        internal static class Prompts
        {
            // 声明静态属性 Coordinates，返回 GetString() 方法的结果
            public static string Coordinates => GetString();
            // 声明静态属性 Start，返回 GetString() 方法的结果
            public static string Start => GetString();
            // 声明静态属性 SeeShots，返回 GetString() 方法的结果
            public static string SeeShots => GetString();
        }

        // 声明私有静态方法 Format，接受一个泛型参数和一个可选的调用者成员名称参数，返回格式化后的字符串
        private static string Format<T>(T value, [CallerMemberName] string? name = null) 
            => string.Format(GetString(name), value);

        // 声明私有静态方法 GetString，接受一个可选的调用者成员名称参数，返回从资源文件中读取的字符串
        private static string GetString([CallerMemberName] string? name = null)
        {
            // 使用资源文件中的流创建 StreamReader 对象，读取流中的内容并返回
            using var stream = GetStream(name);
            using var reader = new StreamReader(stream);
            return reader.ReadToEnd();
        }

        // 声明私有静态方法 GetStream，接受一个可选的调用者成员名称参数，返回从资源文件中获取的流
        private static Stream GetStream([CallerMemberName] string? name = null) =>
            // 从当前程序集中获取嵌入资源的流，如果不存在则抛出异常
            Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
                ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
    }
}
```