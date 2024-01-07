# `basic-computer-games\76_Russian_Roulette\csharp\Program.cs`

```

// 导入 System 命名空间
using System;

// 定义 RussianRoulette 类
namespace RussianRoulette
{
    // 定义 Program 类
    public class Program
    {
        // 程序入口
        public static void Main(string[] args)
        {
            // 打印游戏标题
            PrintTitle();

            // 初始化 includeRevolver 变量为 true
            var includeRevolver = true;
            // 循环进行游戏
            while (true)
            {
                // 打印游戏说明
                PrintInstructions(includeRevolver);
                // 根据游戏结果进行不同的操作
                switch (PlayGame())
                {
                    case GameResult.Win:
                        includeRevolver = true;
                        break;
                    case GameResult.Chicken:
                    case GameResult.Dead:
                        includeRevolver = false;
                        break;
                }
            }
        }

        // 打印游戏标题
        private static void PrintTitle()
        {
            Console.WriteLine("           Russian Roulette");
            Console.WriteLine("Creative Computing  Morristown, New Jersey");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("This is a game of >>>>>>>>>>Russian Roulette.");
        }

        // 打印游戏说明
        private static void PrintInstructions(bool includeRevolver)
        {
            Console.WriteLine();
            if (includeRevolver)
            {
                Console.WriteLine("Here is a revolver.");
            }
            else
            {
                Console.WriteLine();
                Console.WriteLine();
                Console.WriteLine("...Next Victim...");
            }
            Console.WriteLine("Type '1' to spin chamber and pull trigger.");
            Console.WriteLine("Type '2' to give up.");
        }

        // 进行游戏并返回游戏结果
        private static GameResult PlayGame()
        {
            // 创建随机数生成器
            var rnd = new Random();
            // 初始化游戏轮数为 0
            var round = 0;
            // 循环进行游戏
            while (true)
            {
                round++;
                Console.Write("Go: ");
                var input = Console.ReadKey().KeyChar;
                Console.WriteLine();
                if (input != '2')
                {
                    // 生成随机数，模拟扳机扣动
                    if (rnd.Next(1, 7) == 6)
                    {
                        Console.WriteLine("     Bang!!!!!   You're dead!");
                        Console.WriteLine("Condolences will be sent to your relatives.");
                        return GameResult.Dead;
                    }
                    else
                    {
                        if (round > 10)
                        {
                            Console.WriteLine("You win!!!!!");
                            Console.WriteLine("Let someone else blow their brains out.");
                            return GameResult.Win;
                        }
                        else
                        {
                            Console.WriteLine("- CLICK -");
                            Console.WriteLine();
                        }
                    }
                }
                else
                {
                    Console.WriteLine("     CHICKEN!!!!!");
                    return GameResult.Chicken;
                }
            }
        }

        // 定义游戏结果枚举
        private enum GameResult
        {
            Win,
            Chicken,
            Dead
        }
    }
}

```