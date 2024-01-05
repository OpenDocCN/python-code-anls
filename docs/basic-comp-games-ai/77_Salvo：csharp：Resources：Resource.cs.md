# `77_Salvo\csharp\Resources\Resource.cs`

```
# 导入系统反射和运行时编译服务
import System.Reflection
import System.Runtime.CompilerServices

# 命名空间 Salvo.Resources
namespace Salvo.Resources;

# 内部静态类 Resource
internal static class Resource
{
    # 内部静态类 Streams
    internal static class Streams
    {
        # 公共静态属性 Title，返回流对象
        public static Stream Title => GetStream()
        # 公共静态属性 YouHaveMoreShotsThanSquares，返回流对象
        public static Stream YouHaveMoreShotsThanSquares => GetStream()
        # 公共静态属性 YouWon，返回流对象
        public static Stream YouWon => GetStream()
        # 公共静态属性 IHaveMoreShotsThanSquares，返回流对象
        public static Stream IHaveMoreShotsThanSquares => GetStream()
        # 公共静态属性 IWon，返回流对象
        public static Stream IWon => GetStream()
        # 公共静态属性 Illegal，返回流对象
        public static Stream Illegal => GetStream()
    }

    # 内部静态类 Strings
    internal static class Strings
    {
        # 公共静态属性 WhereAreYourShips，返回字符串对象
        public static string WhereAreYourShips => GetString()
        # 定义一个静态方法，用于返回对方射击的信息
        public static string YouHaveShots(int number) => Format(number);
        # 定义一个静态方法，用于返回自己射击的信息
        public static string IHaveShots(int number) => Format(number);
        # 定义一个静态方法，用于返回对方击中自己的船只的信息
        public static string YouHit(string shipName) => Format(shipName);
        # 定义一个静态方法，用于返回自己击中对方的船只的信息
        public static string IHit(string shipName) => Format(shipName);
        # 定义一个静态方法，用于返回之前的射击回合数的信息
        public static string ShotBefore(int turnNumber) => Format(turnNumber);
        # 定义一个静态方法，用于返回当前回合数的信息
        public static string Turn(int number) => Format(number);
    }

    # 定义一个内部静态类，用于返回游戏中的提示信息
    internal static class Prompts
    {
        # 返回坐标信息的提示
        public static string Coordinates => GetString();
        # 返回游戏开始的提示
        public static string Start => GetString();
        # 返回查看射击信息的提示
        public static string SeeShots => GetString();
    }

    # 定义一个私有静态方法，用于格式化信息
    private static string Format<T>(T value, [CallerMemberName] string? name = null) 
        => string.Format(GetString(name), value);

    # 定义一个私有静态方法，用于获取特定信息的字符串
    private static string GetString([CallerMemberName] string? name = null)
    {
        // 使用 var 关键字声明一个变量 stream，并调用 GetStream 方法，传入参数 name
        using var stream = GetStream(name);
        // 使用 var 关键字声明一个变量 reader，并实例化 StreamReader 对象，传入参数 stream
        using var reader = new StreamReader(stream);
        // 返回 reader 对象的全部内容
        return reader.ReadToEnd();
    }

    // 声明一个私有的静态方法 GetStream，参数为 name，默认值为 null，使用 [CallerMemberName] 特性
    private static Stream GetStream([CallerMemberName] string? name = null) =>
        // 调用 Assembly.GetExecutingAssembly() 方法获取当前执行的程序集，再调用 GetManifestResourceStream 方法获取嵌入资源的流
        Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
            // 如果获取的流为空，则抛出异常
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
```