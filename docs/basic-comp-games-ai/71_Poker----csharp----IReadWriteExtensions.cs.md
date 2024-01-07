# `basic-computer-games\71_Poker\csharp\IReadWriteExtensions.cs`

```

# 导入Poker.Strategies命名空间下的所有内容
using Poker.Strategies;
# 导入System命名空间下的StringComparison静态类
using static System.StringComparison;

# 定义Poker命名空间
namespace Poker;

# 定义一个内部静态类IReadWriteExtensions
internal static class IReadWriteExtensions
{
    # 定义一个扩展方法，用于从输入输出流中读取Yes或No的回答
    internal static bool ReadYesNo(this IReadWrite io, string prompt)
    {
        # 循环直到得到有效的回答
        while (true)
        {
            # 从输入输出流中读取字符串作为回答
            var response = io.ReadString(prompt);
            # 如果回答是"YES"，则返回true
            if (response.Equals("YES", InvariantCultureIgnoreCase)) { return true; }
            # 如果回答是"NO"，则返回false
            if (response.Equals("NO", InvariantCultureIgnoreCase)) { return false; }
            # 如果回答既不是"YES"也不是"NO"，则提示用户重新回答
            io.WriteLine("Answer Yes or No, please.");
        }
    }

    # 定义一个扩展方法，用于从输入输出流中读取数字
    internal static float ReadNumber(this IReadWrite io) => io.ReadNumber("");

    # 定义一个扩展方法，用于从输入输出流中读取数字，并限制最大值
    internal static int ReadNumber(this IReadWrite io, string prompt, int max, string maxPrompt)
    {
        # 输出提示信息
        io.Write(prompt);
        # 循环直到得到有效的数字
        while (true)
        {
            # 从输入输出流中读取数字作为回答
            var response = io.ReadNumber();
            # 如果回答小于等于最大值，则返回该数字
            if (response <= max) { return (int)response; }
            # 如果回答大于最大值，则提示用户重新输入
            io.WriteLine(maxPrompt);
        }
    }

    # 定义一个扩展方法，用于从输入输出流中读取人类玩家的策略
    internal static Strategy ReadHumanStrategy(this IReadWrite io, bool noCurrentBets)
    {
        # 循环直到得到有效的策略
        while(true)
        {
            # 输出空行
            io.WriteLine();
            # 从输入输出流中读取数字作为下注数
            var bet = io.ReadNumber("What is your bet");
            # 如果下注数不是整数
            if (bet != (int)bet)
            {
                # 如果允许的最小下注数为0.5，并且下注数为0.5，则返回Check策略
                if (noCurrentBets && bet == .5) { return Strategy.Check; }
                # 如果下注数不是整数且不符合条件，则提示用户重新输入
                io.WriteLine("No small change, please.");
                continue;
            }
            # 如果下注数为0，则返回Fold策略
            if (bet == 0) { return Strategy.Fold; }
            # 如果下注数大于0，则返回Bet策略，并将下注数作为参数
            return Strategy.Bet(bet);
        }
    }
}

```