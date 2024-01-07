# `basic-computer-games\75_Roulette\csharp\Resources\Resource.cs`

```

// 使用 System.Reflection 命名空间，获取程序集信息
using System.Reflection;
// 使用 System.Runtime.CompilerServices 命名空间，调用属性名称
using System.Runtime.CompilerServices;
// 使用 Games.Common.Randomness 命名空间
using Games.Common.Randomness;

// 命名空间 Roulette.Resources
namespace Roulette.Resources
{
    // 内部静态类 Resource
    internal static class Resource
    {
        // 内部静态类 Streams
        internal static class Streams
        {
            // 获取标题流
            public static Stream Title => GetStream();
            // 获取说明流
            public static Stream Instructions => GetStream();
            // 获取已下注流
            public static Stream BetAlready => GetStream();
            // 获取旋转中流
            public static Stream Spinning => GetStream();
            // 获取上一次赢得赌注流
            public static Stream LastDollar => GetStream();
            // 获取输光庄家赌注流
            public static Stream BrokeHouse => GetStream();
            // 获取感谢流
            public static Stream Thanks => GetStream();
        }

        // 内部静态类 Strings
        internal static class Strings
        {
            // 返回黑色赌注字符串
            public static string Black(int number) => Slot(number);
            // 返回红色赌注字符串
            public static string Red(int number) => Slot(number);
            // 返回赌注字符串
            private static string Slot(int number, [CallerMemberName] string? colour = null)
                => string.Format(GetString(), number, colour);
            // 返回输掉赌注字符串
            public static string Lose(Bet bet) => Outcome(bet.Wager, bet.Number);
            // 返回赢得赌注字符串
            public static string Win(Bet bet) => Outcome(bet.Payout, bet.Number);
            // 返回结果字符串
            private static string Outcome(int amount, int number, [CallerMemberName] string? winlose = null)
                => string.Format(GetString(), winlose, amount, number);
            // 返回总数字符串
            public static string Totals(int me, int you) => string.Format(GetString(), me, you);
            // 返回检查字符串
            public static string Check(IRandom random, string payee, int amount)
                => string.Format(GetString(), random.Next(100), DateTime.Now, payee, amount);
        }

        // 内部静态类 Prompts
        internal static class Prompts
        {
            // 返回说明提示字符串
            public static string Instructions => GetPrompt();
            // 返回下注数量提示字符串
            public static string HowManyBets => GetPrompt();
            // 返回赌注提示字符串
            public static string Bet(int number) => string.Format(GetPrompt(), number);
            // 返回再次提示字符串
            public static string Again => GetPrompt();
            // 返回检查提示字符串
            public static string Check => GetPrompt();
        }

        // 获取提示字符串
        private static string GetPrompt([CallerMemberName] string? name = null) => GetString($"{name}Prompt");
        // 获取字符串
        private static string GetString([CallerMemberName] string? name = null)
        {
            // 获取流
            using var stream = GetStream(name);
            // 使用流创建读取器
            using var reader = new StreamReader(stream);
            // 返回读取器的全部内容
            return reader.ReadToEnd();
        }

        // 获取流
        private static Stream GetStream([CallerMemberName] string? name = null) =>
            // 获取当前执行的程序集，获取嵌入的资源流
            Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
                // 如果找不到资源流，抛出异常
                ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
    }
}

```