# `d:/src/tocomm/basic-computer-games\75_Roulette\csharp\Resources\Resource.cs`

```
# 导入所需的模块
import System.Reflection
import System.Runtime.CompilerServices
import Games.Common.Randomness

# 命名空间 Roulette.Resources
namespace Roulette.Resources;

# 内部静态类 Resource
internal static class Resource
{
    # 内部静态类 Streams
    internal static class Streams
    {
        # 公共静态属性 Title，返回 GetStream() 方法的结果
        public static Stream Title => GetStream();
        # 公共静态属性 Instructions，返回 GetStream() 方法的结果
        public static Stream Instructions => GetStream();
        # 公共静态属性 BetAlready，返回 GetStream() 方法的结果
        public static Stream BetAlready => GetStream();
        # 公共静态属性 Spinning，返回 GetStream() 方法的结果
        public static Stream Spinning => GetStream();
        # 公共静态属性 LastDollar，返回 GetStream() 方法的结果
        public static Stream LastDollar => GetStream();
        # 公共静态属性 BrokeHouse，返回 GetStream() 方法的结果
        public static Stream BrokeHouse => GetStream();
        # 公共静态属性 Thanks，返回 GetStream() 方法的结果
        public static Stream Thanks => GetStream();
    }

    # 内部静态类 Strings
    internal static class Strings
public static string Black(int number) => Slot(number);
// 定义一个公共静态方法，接受一个整数参数，调用 Slot 方法并返回结果

public static string Red(int number) => Slot(number);
// 定义一个公共静态方法，接受一个整数参数，调用 Slot 方法并返回结果

private static string Slot(int number, [CallerMemberName] string? colour = null)
// 定义一个私有静态方法，接受一个整数参数和一个可选的字符串参数，使用 CallerMemberName 特性获取调用者的成员名称，默认值为 null
    => string.Format(GetString(), number, colour);
// 使用 GetString 方法格式化字符串，包括传入的整数参数和颜色参数

public static string Lose(Bet bet) => Outcome(bet.Wager, bet.Number);
// 定义一个公共静态方法，接受一个 Bet 对象参数，调用 Outcome 方法并返回结果

public static string Win(Bet bet) => Outcome(bet.Payout, bet.Number);
// 定义一个公共静态方法，接受一个 Bet 对象参数，调用 Outcome 方法并返回结果

private static string Outcome(int amount, int number, [CallerMemberName] string? winlose = null)
// 定义一个私有静态方法，接受一个整数参数和一个可选的字符串参数，使用 CallerMemberName 特性获取调用者的成员名称，默认值为 null
    => string.Format(GetString(), winlose, amount, number);
// 使用 GetString 方法格式化字符串，包括传入的 winlose 参数、整数参数和数字参数

public static string Totals(int me, int you) => string.Format(GetString(), me, you);
// 定义一个公共静态方法，接受两个整数参数，使用 GetString 方法格式化字符串，包括传入的两个整数参数

public static string Check(IRandom random, string payee, int amount)
// 定义一个公共静态方法，接受一个 IRandom 对象参数、一个字符串参数和一个整数参数
    => string.Format(GetString(), random.Next(100), DateTime.Now, payee, amount);
// 使用 GetString 方法格式化字符串，包括随机数、当前时间、收款人和金额参数

internal static class Prompts
// 定义一个内部静态类 Prompts

public static string Instructions => GetPrompt();
// 定义一个公共静态属性 Instructions，返回 GetPrompt 方法的结果
        public static string HowManyBets => GetPrompt();  # 定义一个静态属性，返回调用 GetPrompt() 方法的结果
        public static string Bet(int number) => string.Format(GetPrompt(), number);  # 定义一个静态方法，返回调用 GetPrompt() 方法并格式化传入的数字的结果
        public static string Again => GetPrompt();  # 定义一个静态属性，返回调用 GetPrompt() 方法的结果
        public static string Check => GetPrompt();  # 定义一个静态属性，返回调用 GetPrompt() 方法的结果
    }

    private static string GetPrompt([CallerMemberName] string? name = null) => GetString($"{name}Prompt");  # 定义一个私有静态方法，根据调用者的成员名获取相应的提示信息

    private static string GetString([CallerMemberName] string? name = null)  # 定义一个私有静态方法，根据调用者的成员名获取相应的文本内容
    {
        using var stream = GetStream(name);  # 使用 GetStream 方法获取相应的文本内容流
        using var reader = new StreamReader(stream);  # 使用 StreamReader 读取文本内容流
        return reader.ReadToEnd();  # 返回读取的文本内容
    }

    private static Stream GetStream([CallerMemberName] string? name = null) =>  # 定义一个私有静态方法，根据调用者的成员名获取相应的资源流
        Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")  # 获取当前执行程序集中嵌入资源的流
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");  # 如果找不到资源流，则抛出异常
```