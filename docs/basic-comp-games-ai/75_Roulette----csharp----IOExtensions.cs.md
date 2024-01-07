# `basic-computer-games\75_Roulette\csharp\IOExtensions.cs`

```

// 命名空间 Roulette 下的内部静态类 IOExtensions
namespace Roulette;

internal static class IOExtensions
{
    // 从 IReadWrite 接口中读取下注数量
    internal static int ReadBetCount(this IReadWrite io)
    {
        // 循环直到输入有效的下注数量
        while (true)
        {
            // 从用户输入中读取下注数量
            var betCount = io.ReadNumber(Prompts.HowManyBets);
            // 如果下注数量是有效的整数，返回该数量
            if (betCount.IsValidInt(1)) { return (int)betCount; }
        }
    }

    // 从 IReadWrite 接口中读取下注
    internal static Bet ReadBet(this IReadWrite io, int number)
    {
        // 循环直到输入有效的下注类型和金额
        while (true)
        {
            // 从用户输入中读取下注类型和金额
            var (type, amount) = io.Read2Numbers(Prompts.Bet(number));

            // 如果下注类型和金额都是有效的整数，创建并返回下注对象
            if (type.IsValidInt(1, 50) && amount.IsValidInt(5, 500))
            {
                return new()
                {
                    Type = (int)type, 
                    Number = number, 
                    Wager = (int)amount
                };
            }
        }
    }

    // 检查浮点数值是否是有效的整数，并且在指定范围内
    internal static bool IsValidInt(this float value, int minValue, int maxValue = int.MaxValue)
        => value == (int)value && value >= minValue && value <= maxValue;
}

```