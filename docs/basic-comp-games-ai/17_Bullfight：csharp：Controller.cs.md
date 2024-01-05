# `d:/src/tocomm/basic-computer-games\17_Bullfight\csharp\Controller.cs`

```
// 使用 System 命名空间
using System;

namespace Game
{
    /// <summary>
    /// 包含从用户获取输入的函数。
    /// </summary>
    public static class Controller
    {
        /// <summary>
        /// 处理与玩家的初始交互。
        /// </summary>
        public static void StartGame()
        {
            // 显示游戏横幅
            View.ShowBanner();
            // 提示显示游戏说明
            View.PromptShowInstructions();

            // 从控制台读取用户输入
            var input = Console.ReadLine();
            // 如果输入为空，则退出程序
            if (input is null)
                Environment.Exit(0);
            // 如果输入不是 "NO"，则显示游戏指南
            if (input.ToUpperInvariant() != "NO")
                View.ShowInstructions();

            // 显示分隔符
            View.ShowSeparator();
        }

        /// <summary>
        /// 获取玩家当前回合的行动
        /// </summary>
        /// <param name="passNumber">
        /// 当前回合数
        /// </param>
        public static (Action action, RiskLevel riskLevel) GetPlayerIntention(int passNumber)
        {
            // 如果回合数小于3，提示玩家杀死公牛
            if (passNumber < 3)
                View.PromptKillBull();
            // 否则，简要提示玩家杀死公牛
            else
                View.PromptKillBullBrief();
            # 获取用户是否尝试杀死程序的输入
            var attemptToKill = GetYesOrNo();

            # 如果用户尝试杀死程序
            if (attemptToKill)
            {
                # 提示用户选择杀死程序的方法
                View.PromptKillMethod();

                # 获取用户输入
                var input = Console.ReadLine();
                # 如果用户输入为空，则退出程序
                if (input is null)
                    Environment.Exit(0);

                # 根据用户输入选择执行不同的操作
                return input switch
                {
                    "4" => (Action.Kill,  RiskLevel.High),  # 如果输入为"4"，执行杀死程序的高风险操作
                    "5" => (Action.Kill,  RiskLevel.Low),   # 如果输入为"5"，执行杀死程序的低风险操作
                    _   => (Action.Panic, default(RiskLevel))  # 其他输入，执行紧急操作
                };
            }
            # 如果用户不尝试杀死程序
            else
            {
                # 如果尝试次数小于2
                if (passNumber < 2)
                    View.PromptCapeMove();  # 调用 View 类的 PromptCapeMove 方法，提示用户进行动作选择
                else
                    View.PromptCapeMoveBrief();  # 调用 View 类的 PromptCapeMoveBrief 方法，提示用户进行简要动作选择

                var action = Action.Panic;  # 声明并初始化变量 action 为 Action.Panic
                var riskLevel = default(RiskLevel);  # 声明并初始化变量 riskLevel 为 RiskLevel 类型的默认值

                while (action == Action.Panic)  # 当 action 等于 Action.Panic 时循环执行以下代码
                {
                    var input = Console.ReadLine();  # 从控制台读取用户输入并赋值给变量 input
                    if (input is null)  # 如果输入为 null
                        Environment.Exit(0);  # 退出程序

                    (action, riskLevel) = input switch  # 根据用户输入的不同情况进行匹配
                    {
                        "0" => (Action.Dodge, RiskLevel.High),  # 当输入为 "0" 时，将 action 赋值为 Action.Dodge，riskLevel 赋值为 RiskLevel.High
                        "1" => (Action.Dodge, RiskLevel.Medium),  # 当输入为 "1" 时，将 action 赋值为 Action.Dodge，riskLevel 赋值为 RiskLevel.Medium
                        "2" => (Action.Dodge, RiskLevel.Low),  # 当输入为 "2" 时，将 action 赋值为 Action.Dodge，riskLevel 赋值为 RiskLevel.Low
                        _   => (Action.Panic, default(RiskLevel))  # 其他情况下，将 action 赋值为 Action.Panic，riskLevel 赋值为 RiskLevel 类型的默认值
                    };
                    if (action == Action.Panic)  # 如果动作是恐慌
                        View.PromptDontPanic();  # 提示不要恐慌
                }

                return (action, riskLevel);  # 返回动作和风险级别
            }
        }

        /// <summary>
        /// Gets the player's intention to flee (or not).
        /// </summary>
        /// <returns>
        /// True if the player flees; otherwise, false.
        /// </returns>
        public static bool GetPlayerRunsFromRing()  # 获取玩家是否逃离的意图
        {
            View.PromptRunFromRing();  # 提示逃离戒指

            var playerFlees = GetYesOrNo();  # 获取玩家的是或否选择
            // 如果玩家没有逃跑
            if (!playerFlees)
                // 显示玩家鲁莽的行为
                View.ShowPlayerFoolhardy();

            // 返回玩家是否逃跑的布尔值
            return playerFlees;
        }

        /// <summary>
        /// 从玩家获取是或否的回答。
        /// </summary>
        /// <returns>
        /// 如果玩家回答是，则返回 true；否则返回 false。
        /// </returns>
        public static bool GetYesOrNo()
        {
            // 循环直到获取到有效的输入
            while (true)
            {
                // 从控制台获取输入
                var input = Console.ReadLine();
                // 如果输入为空，则退出程序
                if (input is null)
                    Environment.Exit(0);
# 将用户输入转换为大写形式，然后根据不同的情况进行处理
switch (input.ToUpperInvariant())
{
    # 如果用户输入为"YES"，则返回 true
    case "YES":
        return true;
    # 如果用户输入为"NO"，则返回 false
    case "NO":
        return false;
    # 如果用户输入不是"YES"或"NO"，则打印错误提示信息
    default:
        Console.WriteLine("INCORRECT ANSWER - - PLEASE TYPE 'YES' OR 'NO'.");
        # 跳出当前的 case 分支
        break;
}
# 结束 switch 语句
}
# 结束方法
}
# 结束类
}
```