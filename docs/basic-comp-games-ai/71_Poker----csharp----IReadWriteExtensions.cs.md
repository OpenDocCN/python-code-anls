# `basic-computer-games\71_Poker\csharp\IReadWriteExtensions.cs`

```
// 使用扑克策略命名空间中的策略
using Poker.Strategies;
// 使用不区分大小写的字符串比较
using static System.StringComparison;

// 定义扑克命名空间下的内部静态类 IReadWriteExtensions
namespace Poker;
internal static class IReadWriteExtensions
{
    // 从输入输出对象中读取 Yes 或 No
    internal static bool ReadYesNo(this IReadWrite io, string prompt)
    {
        // 循环直到得到有效的回答
        while (true)
        {
            // 读取用户输入的响应
            var response = io.ReadString(prompt);
            // 如果响应是 "YES"，则返回 true
            if (response.Equals("YES", InvariantCultureIgnoreCase)) { return true; }
            // 如果响应是 "NO"，则返回 false
            if (response.Equals("NO", InvariantCultureIgnoreCase)) { return false; }
            // 如果响应既不是 "YES" 也不是 "NO"，则提示用户重新输入
            io.WriteLine("Answer Yes or No, please.");
        }
    }

    // 从输入输出对象中读取数字
    internal static float ReadNumber(this IReadWrite io) => io.ReadNumber("");

    // 从输入输出对象中读取数字，并限制最大值
    internal static int ReadNumber(this IReadWrite io, string prompt, int max, string maxPrompt)
    {
        // 输出提示信息
        io.Write(prompt);
        // 循环直到得到有效的数字
        while (true)
        {
            // 读取用户输入的数字
            var response = io.ReadNumber();
            // 如果输入的数字小于等于最大值，则返回该数字
            if (response <= max) { return (int)response; }
            // 如果输入的数字大于最大值，则提示用户重新输入
            io.WriteLine(maxPrompt);
        }
    }

    // 从输入输出对象中读取人类策略
    internal static Strategy ReadHumanStrategy(this IReadWrite io, bool noCurrentBets)
    {
        // 循环直到得到有效的策略
        while(true)
        {
            // 输出空行
            io.WriteLine();
            // 询问用户的赌注
            var bet = io.ReadNumber("What is your bet");
            // 如果赌注不是整数
            if (bet != (int)bet)
            {
                // 如果没有当前的赌注且赌注为 0.5，则返回 Check 策略
                if (noCurrentBets && bet == .5) { return Strategy.Check; }
                // 如果赌注不是整数且不符合条件，则提示用户重新输入
                io.WriteLine("No small change, please.");
                continue;
            }
            // 如果赌注为 0，则返回 Fold 策略
            if (bet == 0) { return Strategy.Fold; }
            // 如果赌注为整数且不为 0.5，则返回 Bet 策略
            return Strategy.Bet(bet);
        }
    }
}
```