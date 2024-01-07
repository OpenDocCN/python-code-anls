# `basic-computer-games\33_Dice\csharp\Game.cs`

```

// 引入命名空间
using System;
using System.Linq;

// 创建名为Game的类
namespace BasicComputerGames.Dice
{
    public class Game
    {
        // 创建RollGenerator对象
        private readonly RollGenerator _roller = new RollGenerator();

        // 游戏循环
        public void GameLoop()
        {
            // 显示游戏介绍文本
            DisplayIntroText();

            // 游戏循环
            do
            {
                // 获取输入的次数
                int numRolls = GetInput();
                // 计算掷骰子的结果
                var counter = CountRolls(numRolls);
                // 显示掷骰子的结果
                DisplayCounts(counter);
            } while (TryAgain());
        }

        // 显示游戏介绍文本
        private void DisplayIntroText()
        {
            // 设置控制台前景色为黄色
            Console.ForegroundColor = ConsoleColor.Yellow;
            // 输出游戏信息
            Console.WriteLine("Dice");
            Console.WriteLine("Creating Computing, Morristown, New Jersey."); Console.WriteLine();

            // 设置控制台前景色为深绿色
            Console.ForegroundColor = ConsoleColor.DarkGreen;
            // 输出游戏信息
            Console.WriteLine("Original code by Danny Freidus.");
            Console.WriteLine("Originally published in 1978 in the book 'Basic Computer Games' by David Ahl.");
            Console.WriteLine("Modernized and converted to C# in 2021 by James Curran (noveltheory.com).");
            Console.WriteLine();

            // 设置控制台前景色为灰色
            Console.ForegroundColor = ConsoleColor.Gray;
            // 输出游戏信息
            Console.WriteLine("This program simulates the rolling of a pair of dice.");
            Console.WriteLine("You enter the number of times you want the computer to");
            Console.WriteLine("'roll' the dice. Watch out, very large numbers take");
            Console.WriteLine("a long time. In particular, numbers over 10 million.");
            Console.WriteLine();

            // 设置控制台前景色为黄色
            Console.ForegroundColor = ConsoleColor.Yellow;
            // 提示玩家按任意键开始游戏
            Console.WriteLine("Press any key start the game.");
            Console.ReadKey(true);
        }

        // 获取输入的次数
        private int GetInput()
        {
            int num = -1;
            Console.WriteLine();
            do
            {
                Console.WriteLine();
                Console.Write("How many rolls? ");
            } while (!Int32.TryParse(Console.ReadLine(), out num));

            return num;
        }

        // 显示掷骰子的结果
        private void DisplayCounts(int[] counter)
        {
            Console.WriteLine();
            Console.WriteLine($"\tTotal\tTotal Number");
            Console.WriteLine($"\tSpots\tof Times");
            Console.WriteLine($"\t===\t=========");
            for (var n = 1; n < counter.Length; ++n)
            {
                Console.WriteLine($"\t{n + 1,2}\t{counter[n],9:#,0}");
            }
            Console.WriteLine();
        }

        // 计算掷骰子的结果
        private int[] CountRolls(int x)
        {
            var counter = _roller.Rolls().Take(x).Aggregate(new int[12], (cntr, r) =>
            {
                cntr[r.die1 + r.die2 - 1]++;
                return cntr;
            });
            return counter;
        }

        // 提示玩家是否再次尝试，等待他们按Y或N
        private bool TryAgain()
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("Would you like to try again? (Press 'Y' for yes or 'N' for no)");

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write("> ");

            char pressedKey;
            // 循环直到获得一个被识别的输入
            do
            {
                // 读取一个键，不在屏幕上显示
                ConsoleKeyInfo key = Console.ReadKey(true);
                // 转换为大写，这样我们就不需要关心大小写
                pressedKey = Char.ToUpper(key.KeyChar);
                // 这是我们认识的键吗？如果不是，继续循环
            } while (pressedKey != 'Y' && pressedKey != 'N');
            // 在屏幕上显示结果
            Console.WriteLine(pressedKey);

            // 如果玩家按下'Y'，返回true，否则返回false
            return (pressedKey == 'Y');
        }
    }
}

```