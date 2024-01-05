# `71_Poker\csharp\IReadWriteExtensions.cs`

```
using Poker.Strategies;  # 导入Poker.Strategies模块
using static System.StringComparison;  # 导入System.StringComparison模块的所有静态成员

namespace Poker;  # 命名空间Poker

internal static class IReadWriteExtensions  # 定义一个内部静态类IReadWriteExtensions
{
    internal static bool ReadYesNo(this IReadWrite io, string prompt)  # 定义一个静态方法ReadYesNo，接受IReadWrite类型的参数io和string类型的参数prompt
    {
        while (true):  # 进入无限循环
            var response = io.ReadString(prompt);  # 调用io的ReadString方法，将结果赋值给response
            if (response.Equals("YES", InvariantCultureIgnoreCase)) { return true; }  # 如果response等于"YES"（不区分大小写），返回true
            if (response.Equals("NO", InvariantCultureIgnoreCase)) { return false; }  # 如果response等于"NO"（不区分大小写），返回false
            io.WriteLine("Answer Yes or No, please.");  # 调用io的WriteLine方法，输出提示信息
    }

    internal static float ReadNumber(this IReadWrite io) => io.ReadNumber("");  # 定义一个静态方法ReadNumber，接受IReadWrite类型的参数io，调用io的ReadNumber方法并返回结果
        // 写入提示信息
        io.Write(prompt);
        // 循环读取用户输入的数字
        while (true)
        {
            // 从输入流中读取数字
            var response = io.ReadNumber();
            // 如果读取的数字小于等于最大值，则返回该数字
            if (response <= max) { return (int)response; }
            // 否则，输出最大值提示信息
            io.WriteLine(maxPrompt);
        }
    }

    internal static Strategy ReadHumanStrategy(this IReadWrite io, bool noCurrentBets)
    {
        // 循环读取用户策略
        while(true)
        {
            // 输出空行
            io.WriteLine();
            // 读取用户的赌注
            var bet = io.ReadNumber("What is your bet");
            // 如果赌注不是整数
            if (bet != (int)bet)
            {
                // 如果没有当前的赌注，并且赌注为0.5，则返回策略为Check
                if (noCurrentBets && bet == .5) { return Strategy.Check; }
                io.WriteLine("No small change, please.");  # 输出消息到控制台，提示不要进行小额更改
                continue;  # 继续执行下一次循环
            }
            if (bet == 0) { return Strategy.Fold; }  # 如果下注金额为0，则返回放弃策略
            return Strategy.Bet(bet);  # 返回下注策略，并传入下注金额
        }
    }
}
```