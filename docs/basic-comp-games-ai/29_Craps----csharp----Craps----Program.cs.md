# `basic-computer-games\29_Craps\csharp\Craps\Program.cs`

```py
using System.Diagnostics; // 导入系统诊断命名空间

namespace Craps
{
    class Program
    {
        static void Main(string[] args)
        {
            var ui = new UserInterface(); // 创建用户界面对象
            var game = new CrapsGame(ref ui); // 创建 Craps 游戏对象，并传入用户界面对象的引用
            int winnings = 0; // 初始化赢得的奖金为 0

            ui.Intro(); // 调用用户界面对象的 Intro 方法，显示游戏介绍

            do
            {
                var bet = ui.PlaceBet(); // 调用用户界面对象的 PlaceBet 方法，获取玩家下注金额
                var result = game.Play(out int diceRoll); // 调用游戏对象的 Play 方法，获取游戏结果和骰子点数

                switch (result) // 根据游戏结果进行不同的处理
                {
                    case Result.naturalWin: // 如果是自然胜利
                        winnings += bet; // 奖金增加下注金额
                        break;

                    case Result.naturalLoss: // 如果是自然失败
                    case Result.snakeEyesLoss: // 如果是蛇眼失败
                    case Result.pointLoss: // 如果是点数失败
                        winnings -= bet; // 奖金减少下注金额
                        break;

                    case Result.pointWin: // 如果是点数胜利
                        winnings += (2 * bet); // 奖金增加下注金额的两倍
                        break;

                    // 包含一个默认情况，以便在枚举值发生变化时收到警告，并且忘记添加处理新值的代码
                    default:
                        Debug.Assert(false); // 断言，如果程序执行到这里，会触发断言错误
                        break;
                }

                ui.ShowResult(result, diceRoll, bet); // 调用用户界面对象的 ShowResult 方法，显示游戏结果、骰子点数和下注金额
            } while (ui.PlayAgain(winnings)); // 调用用户界面对象的 PlayAgain 方法，判断是否继续游戏

            ui.GoodBye(winnings); // 调用用户界面对象的 GoodBye 方法，显示结束语并展示最终奖金
        }
    }
}
```