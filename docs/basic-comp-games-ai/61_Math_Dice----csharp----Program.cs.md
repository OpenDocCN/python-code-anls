# `basic-computer-games\61_Math_Dice\csharp\Program.cs`

```

// 引入 System 命名空间
using System;

// 定义 MathDice 类
namespace MathDice
{
    // 定义 Program 类
    public static class Program
    {
        // 创建 Random 对象
        readonly static Random random = new Random();

        // 定义两个骰子的点数
        static int DieOne = 0;
        static int DieTwo = 0;

        // 定义骰子点数对应的图案
        private const string NoPips = "I     I";
        private const string LeftPip = "I *   I";
        private const string CentrePip = "I  *  I";
        private const string RightPip = "I   * I";
        private const string TwoPips = "I * * I";
        private const string Edge = " ----- ";

        // 主函数
        static void Main(string[] args)
        {
            int answer;

            // 初始化游戏状态
            GameState gameState = GameState.FirstAttempt;

            // 输出游戏标题
            Console.WriteLine("MATH DICE".CentreAlign());
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".CentreAlign());
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("THIS PROGRAM GENERATES SUCCESSIVE PICTURES OF TWO DICE.");
            Console.WriteLine("WHEN TWO DICE AND AN EQUAL SIGN FOLLOWED BY A QUESTION");
            Console.WriteLine("MARK HAVE BEEN PRINTED, TYPE YOUR ANSWER AND THE RETURN KEY.");
            Console.WriteLine("TO CONCLUDE THE LESSON, TYPE CONTROL-C AS YOUR ANSWER.");
            Console.WriteLine();
            Console.WriteLine();

            // 游戏循环
            while (true)
            {
                // 如果是第一次尝试
                if (gameState == GameState.FirstAttempt)
                {
                    // 掷骰子
                    Roll(ref DieOne);
                    Roll(ref DieTwo);

                    // 绘制骰子图案
                    DrawDie(DieOne);
                    Console.WriteLine("   +");
                    DrawDie(DieTwo);
                }

                // 获取玩家答案
                answer = GetAnswer();

                // 判断答案是否正确
                if (answer == DieOne + DieTwo)
                {
                    Console.WriteLine("RIGHT!");
                    Console.WriteLine();
                    Console.WriteLine("THE DICE ROLL AGAIN...");

                    gameState = GameState.FirstAttempt;
                }
                else
                {
                    if (gameState == GameState.FirstAttempt)
                    {
                        Console.WriteLine("NO, COUNT THE SPOTS AND GIVE ANOTHER ANSWER.");
                        gameState = GameState.SecondAttempt;
                    }
                    else
                    {
                        Console.WriteLine($"NO, THE ANSWER IS{DieOne + DieTwo}");
                        Console.WriteLine();
                        Console.WriteLine("THE DICE ROLL AGAIN...");
                        gameState = GameState.FirstAttempt;
                    }
                }
            }
        }

        // 获取玩家答案
        private static int GetAnswer()
        {
            int answer;

            Console.Write("      =?");
            var input = Console.ReadLine();

            int.TryParse(input, out answer);

            return answer;
        }

        // 绘制骰子图案
        private static void DrawDie(int pips)
        {
            Console.WriteLine(Edge);
            Console.WriteLine(OuterRow(pips, true));
            Console.WriteLine(CentreRow(pips));
            Console.WriteLine(OuterRow(pips, false));
            Console.WriteLine(Edge);
            Console.WriteLine();
        }

        // 掷骰子
        private static void Roll(ref int die) => die = random.Next(1, 7);

        // 根据骰子点数返回对应的上下行图案
        private static string OuterRow(int pips, bool top)
        {
            return pips switch
            {
                1 => NoPips,
                var x when x == 2 || x == 3 => top ? LeftPip : RightPip,
                _ => TwoPips
            };
        }

        // 根据骰子点数返回对应的中间行图案
        private static string CentreRow(int pips)
        {
            return pips switch
            {
                var x when x == 2 || x == 4 => NoPips,
                6 => TwoPips,
                _ => CentrePip
            };
        }
    }
}

```