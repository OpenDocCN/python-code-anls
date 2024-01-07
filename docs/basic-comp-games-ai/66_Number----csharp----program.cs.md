# `basic-computer-games\66_Number\csharp\program.cs`

```

// 使用 System.Text 命名空间
using System.Text;

// 定义 Number 类
namespace Number
{
    // Number 类
    class Number
    {
        // 显示游戏介绍
        private void DisplayIntro()
        {
            // 输出游戏介绍
            Console.WriteLine();
            Console.WriteLine("NUMBER".PadLeft(23));
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("You have 100 points.  By guessing numbers from 1 to 5, you");
            Console.WriteLine("can gain or lose points depending upon how close you get to");
            Console.WriteLine("a random number selected by the computer.");
            Console.WriteLine();
            Console.WriteLine("You occaisionally will get a jackpot which will double(!)");
            Console.WriteLine("your point count.  You win when you get 500 points.");
            Console.WriteLine();

        }

        // 提示用户猜测数字
        private int PromptForGuess()
        {
            // 初始化变量
            bool Success = false;
            int Guess = 0;

            // 循环直到用户输入有效数字
            while (!Success)
            {
                Console.Write("Guess a number from 1 to 5? ");
                string LineInput = Console.ReadLine().Trim().ToLower();

                // 尝试将输入转换为整数
                if (int.TryParse(LineInput, out Guess))
                {
                    // 判断输入是否在1到5之间
                    if (Guess >= 0 && Guess <= 5)
                        Success = true;
                }
                else
                    Console.WriteLine("Please enter a number between 1 and 5.");
            }

            return Guess;
        }

        // 生成随机数
        private void GetRandomNumbers(out int Random1, out int Random2, out int Random3, out int Random4, out int Random5)
        {
            Random rand = new Random();

            // 生成不重复的随机数
            Random1 = (int)(5 * rand.NextDouble() + 1);
            do
            {
                Random2 = (int)(5 * rand.NextDouble() + 1);
            } while (Random2 == Random1);
            do
            {
                Random3 = (int)(5 * rand.NextDouble() + 1);
            } while (Random3 == Random1 || Random3 == Random2);
            do
            {
                Random4 = (int)(5 * rand.NextDouble() + 1);
            } while (Random4 == Random1 || Random4 == Random2 || Random4 == Random3);
            do
            {
                Random5 = (int)(5 * rand.NextDouble() + 1);
            } while (Random5 == Random1 || Random5 == Random2 || Random5 == Random3 || Random5 == Random4);

        }

        // 游戏主逻辑
        private void Play()
        {
            // 初始化变量
            int Points = 100;
            bool Win = false;
            int Random1, Random2, Random3, Random4, Random5;
            int Guess = 0;

            // 生成随机数
            GetRandomNumbers(out Random1, out Random2, out Random3, out Random4, out Random5);

            // 游戏循环
            while (!Win)
            {
                // 提示用户猜测数字
                Guess = PromptForGuess();

                // 根据猜测结果更新分数
                if (Guess == Random1)
                    Points -= 5;
                else if (Guess == Random2)
                    Points += 5;
                else if (Guess == Random3)
                {
                    Points += Points;
                    Console.WriteLine("You hit the jackpot!!!");
                }
                else if (Guess == Random4)
                    Points += 1;
                else if (Guess == Random5)
                    Points -= (int)(Points * 0.5);

                // 判断是否获胜
                if (Points > 500)
                {
                    Console.WriteLine("!!!!You Win!!!! with {0} points.", Points);
                    Win = true;
                }
                else
                    Console.WriteLine("You have {0} points.", Points);
            }
        }

        // 开始游戏
        public void PlayTheGame()
        {
            // 显示游戏介绍
            DisplayIntro();

            // 开始游戏
            Play();
        }
    }

    // 程序入口
    class Program
    {
        static void Main(string[] args)
        {
            // 创建 Number 对象并开始游戏
            new Number().PlayTheGame();
        }
    }
}

```