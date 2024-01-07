# `basic-computer-games\29_Craps\csharp\Craps\Program.cs`

```

// 引入 System.Diagnostics 命名空间
using System.Diagnostics;

// 命名空间 Craps
namespace Craps
{
    // 程序类
    class Program
    {
        // 主函数
        static void Main(string[] args)
        {
            // 创建用户界面对象
            var ui = new UserInterface();
            // 创建 Craps 游戏对象，并传入用户界面对象的引用
            var game = new CrapsGame(ref ui);
            // 初始化赢得的奖金为 0
            int winnings = 0;

            // 显示游戏介绍
            ui.Intro();

            // 游戏循环
            do
            {
                // 用户下注
                var bet = ui.PlaceBet();
                // 进行游戏，并返回游戏结果和骰子点数
                var result = game.Play(out int diceRoll);

                // 根据游戏结果进行奖金计算
                switch (result)
                {
                    case Result.naturalWin:
                        winnings += bet;
                        break;

                    case Result.naturalLoss:
                    case Result.snakeEyesLoss:
                    case Result.pointLoss:
                        winnings -= bet;
                        break;

                    case Result.pointWin:
                        winnings += (2 * bet);
                        break;

                    // 包含默认情况，以便在枚举值发生变化时提醒我们添加新值的处理代码
                    default:
                        Debug.Assert(false); // 我们不应该到达这里
                        break;
                }

                // 显示游戏结果
                ui.ShowResult(result, diceRoll, bet);
            } while (ui.PlayAgain(winnings)); // 判断是否再玩一次

            // 显示结束语
            ui.GoodBye(winnings);
        }
    }
}

```