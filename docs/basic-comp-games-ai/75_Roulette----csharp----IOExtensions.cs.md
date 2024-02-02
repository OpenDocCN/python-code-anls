# `basic-computer-games\75_Roulette\csharp\IOExtensions.cs`

```py
# 命名空间 Roulette 下的内部静态类 IOExtensions
internal static class IOExtensions
{
    # 从 IReadWrite 接口中读取下注数量
    internal static int ReadBetCount(this IReadWrite io)
    {
        # 循环直到条件满足
        while (true)
        {
            # 从输入中读取下注数量，并显示提示信息
            var betCount = io.ReadNumber(Prompts.HowManyBets);
            # 如果下注数量是有效的整数，则返回该数量
            if (betCount.IsValidInt(1)) { return (int)betCount; }
        }
    }

    # 从 IReadWrite 接口中读取下注信息
    internal static Bet ReadBet(this IReadWrite io, int number)
    {
        # 循环直到条件满足
        while (true)
        {
            # 从输入中读取下注类型和金额，并显示相应的提示信息
            var (type, amount) = io.Read2Numbers(Prompts.Bet(number));

            # 如果下注类型和金额都是有效的整数，则返回一个新的 Bet 对象
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

    # 判断浮点数是否是有效的整数，并且在指定范围内
    internal static bool IsValidInt(this float value, int minValue, int maxValue = int.MaxValue)
        => value == (int)value && value >= minValue && value <= maxValue;
}
```