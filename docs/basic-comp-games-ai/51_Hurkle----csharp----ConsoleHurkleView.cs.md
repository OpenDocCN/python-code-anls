# `basic-computer-games\51_Hurkle\csharp\ConsoleHurkleView.cs`

```

// 引入 System 命名空间
using System;

// 定义 ConsoleHurkleView 类，实现 IHurkleView 接口
namespace hurkle
{
    internal class ConsoleHurkleView : IHurkleView
    {
        // 获取玩家猜测的坐标
        public GamePoint GetGuess(GuessViewModel guessViewModel)
        {
            // 输出当前猜测的序号
            Console.WriteLine($"GUESS #{guessViewModel.CurrentGuessNumber}");
            // 读取玩家输入的坐标
            var inputLine = Console.ReadLine();
            // 根据逗号分隔输入的坐标，并去除空格
            var seperateStrings = inputLine.Split(',', 2, StringSplitOptions.TrimEntries);
            // 创建 GamePoint 对象，保存玩家猜测的坐标
            var guessPoint = new GamePoint{
                X = int.Parse(seperateStrings[0]),
                Y = int.Parse(seperateStrings[1])
            };

            return guessPoint; // 返回玩家猜测的坐标
        }

        // 显示猜测失败时的方向
        public void ShowDirection(FailedGuessViewModel failedGuessViewModel)
        {
            Console.Write("GO ");
            // 根据失败猜测的方向输出相应信息
            switch(failedGuessViewModel.Direction)
            {
                case CardinalDirection.East:
                    Console.WriteLine("EAST");
                    break;
                case CardinalDirection.North:
                    Console.WriteLine("NORTH");
                    break;
                // 其他方向的处理类似
            }

            Console.WriteLine(); // 输出空行
        }

        // 显示游戏失败信息
        public void ShowLoss(LossViewModel lossViewModel)
        {
            Console.WriteLine(); // 输出空行
            Console.WriteLine($"SORRY, THAT'S {lossViewModel.MaxGuesses} GUESSES"); // 输出失败信息和最大猜测次数
            Console.WriteLine($"THE HURKLE IS AT {lossViewModel.HurkleLocation.X},{lossViewModel.HurkleLocation.Y}"); // 输出 Hurkle 的位置
        }

        // 显示游戏胜利信息
        public void ShowVictory(VictoryViewModel victoryViewModel)
        {
            Console.WriteLine(); // 输出空行
            Console.WriteLine($"YOU FOUND HIM IN {victoryViewModel.CurrentGuessNumber} GUESSES!"); // 输出胜利信息和猜测次数
        }
    }
}

```