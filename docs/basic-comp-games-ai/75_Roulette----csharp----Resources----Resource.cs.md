# `basic-computer-games\75_Roulette\csharp\Resources\Resource.cs`

```py
using System.Reflection;
using System.Runtime.CompilerServices;
using Games.Common.Randomness;

namespace Roulette.Resources;
// 声明一个命名空间 Roulette.Resources

internal static class Resource
{
    // 声明一个内部静态类 Resource

    internal static class Streams
    {
        // 声明一个内部静态类 Streams

        public static Stream Title => GetStream();
        // 声明一个公共静态属性 Title，返回一个流对象

        public static Stream Instructions => GetStream();
        // 声明一个公共静态属性 Instructions，返回一个流对象

        public static Stream BetAlready => GetStream();
        // 声明一个公共静态属性 BetAlready，返回一个流对象

        public static Stream Spinning => GetStream();
        // 声明一个公共静态属性 Spinning，返回一个流对象

        public static Stream LastDollar => GetStream();
        // 声明一个公共静态属性 LastDollar，返回一个流对象

        public static Stream BrokeHouse => GetStream();
        // 声明一个公共静态属性 BrokeHouse，返回一个流对象

        public static Stream Thanks => GetStream();
        // 声明一个公共静态属性 Thanks，返回一个流对象
    }

    internal static class Strings
    {
        // 声明一个内部静态类 Strings

        public static string Black(int number) => Slot(number);
        // 声明一个公共静态方法 Black，接受一个整数参数，返回一个字符串

        public static string Red(int number) => Slot(number);
        // 声明一个公共静态方法 Red，接受一个整数参数，返回一个字符串

        private static string Slot(int number, [CallerMemberName] string? colour = null)
            => string.Format(GetString(), number, colour);
        // 声明一个私有静态方法 Slot，接受一个整数参数和一个可选的字符串参数，返回一个格式化后的字符串

        public static string Lose(Bet bet) => Outcome(bet.Wager, bet.Number);
        // 声明一个公共静态方法 Lose，接受一个 Bet 类型的参数，返回一个字符串

        public static string Win(Bet bet) => Outcome(bet.Payout, bet.Number);
        // 声明一个公共静态方法 Win，接受一个 Bet 类型的参数，返回一个字符串

        private static string Outcome(int amount, int number, [CallerMemberName] string? winlose = null)
            => string.Format(GetString(), winlose, amount, number);
        // 声明一个私有静态方法 Outcome，接受两个整数参数和一个可选的字符串参数，返回一个格式化后的字符串

        public static string Totals(int me, int you) => string.Format(GetString(), me, you);
        // 声明一个公共静态方法 Totals，接受两个整数参数，返回一个格式化后的字符串

        public static string Check(IRandom random, string payee, int amount)
            => string.Format(GetString(), random.Next(100), DateTime.Now, payee, amount);
        // 声明一个公共静态方法 Check，接受一个 IRandom 类型的参数和两个字符串参数，返回一个格式化后的字符串
    }

    internal static class Prompts
    {
        // 声明一个内部静态类 Prompts

        public static string Instructions => GetPrompt();
        // 声明一个公共静态属性 Instructions，返回一个字符串

        public static string HowManyBets => GetPrompt();
        // 声明一个公共静态属性 HowManyBets，返回一个字符串

        public static string Bet(int number) => string.Format(GetPrompt(), number);
        // 声明一个公共静态方法 Bet，接受一个整数参数，返回一个格式化后的字符串

        public static string Again => GetPrompt();
        // 声明一个公共静态属性 Again，返回一个字符串

        public static string Check => GetPrompt();
        // 声明一个公共静态属性 Check，返回一个字符串
    }

    private static string GetPrompt([CallerMemberName] string? name = null) => GetString($"{name}Prompt");
    // 声明一个私有静态方法 GetPrompt，接受一个可选的字符串参数，返回一个格式化后的字符串
}
    # 获取调用该方法的成员名称，如果没有传入名称则默认为null
    private static string GetString([CallerMemberName] string? name = null)
    {
        # 使用获取到的名称获取对应的流
        using var stream = GetStream(name);
        # 使用流创建一个读取器
        using var reader = new StreamReader(stream);
        # 读取读取器中的所有内容并返回
        return reader.ReadToEnd();
    }
    
    # 根据传入的名称获取嵌入资源的流
    private static Stream GetStream([CallerMemberName] string? name = null) =>
        # 使用程序集获取嵌入资源的流，如果找不到则抛出异常
        Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
# 闭合大括号，表示代码块的结束
```