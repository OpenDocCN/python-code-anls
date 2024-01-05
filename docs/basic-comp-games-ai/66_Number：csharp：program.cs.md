# `d:/src/tocomm/basic-computer-games\66_Number\csharp\program.cs`

```
using System.Text;  // 导入 System.Text 命名空间

namespace Number  // 命名空间 Number
{
    class Number  // 类 Number
    {
        private void DisplayIntro()  // 方法 DisplayIntro
        {
            Console.WriteLine();  // 输出空行
            Console.WriteLine("NUMBER".PadLeft(23));  // 输出 "NUMBER" 并左对齐到第 23 列
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 输出 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
            Console.WriteLine();  // 输出空行
            Console.WriteLine();  // 输出空行
            Console.WriteLine();  // 输出空行
            Console.WriteLine("You have 100 points.  By guessing numbers from 1 to 5, you");  // 输出提示信息
            Console.WriteLine("can gain or lose points depending upon how close you get to");  // 输出提示信息
            Console.WriteLine("a random number selected by the computer.");  // 输出提示信息
            Console.WriteLine();  // 输出空行
            Console.WriteLine("You occaisionally will get a jackpot which will double(!)");  // 输出提示信息
            Console.WriteLine("your point count.  You win when you get 500 points.");  // 输出提示信息
            Console.WriteLine();  # 输出空行

        }  # 结束方法

        private int PromptForGuess()  # 定义一个返回整数的私有方法
        {
            bool Success = false;  # 定义一个布尔变量 Success，初始值为 false
            int Guess = 0;  # 定义一个整数变量 Guess，初始值为 0

            while (!Success)  # 当 Success 为 false 时执行循环
            {
                Console.Write("Guess a number from 1 to 5? ");  # 输出提示信息
                string LineInput = Console.ReadLine().Trim().ToLower();  # 读取用户输入并去除首尾空格，转换为小写

                if (int.TryParse(LineInput, out Guess))  # 尝试将用户输入转换为整数，如果成功则将转换后的值赋给 Guess
                {
                    if (Guess >= 0 && Guess <= 5)  # 如果 Guess 在 0 到 5 之间
                        Success = true;  # 将 Success 设置为 true
                }
                else
                    Console.WriteLine("Please enter a number between 1 and 5.");  # 输出错误提示信息
        }

        return Guess;
    }

    // 生成五个不重复的随机数
    private void GetRandomNumbers(out int Random1, out int Random2, out int Random3, out int Random4, out int Random5)
    {
        // 创建一个随机数生成器对象
        Random rand = new Random();

        // 生成一个介于1和5之间的唯一随机数
        // 我假设这是原始BASIC程序 FNR(X)=INT(5*RND(1)+1) 的功能
        Random1 = (int)(5 * rand.NextDouble() + 1);
        // 生成一个介于1和5之间的与Random1不同的随机数
        do
        {
            Random2 = (int)(5 * rand.NextDouble() + 1);
        } while (Random2 == Random1);
        // 生成一个介于1和5之间的与Random1和Random2都不同的随机数
        do
        {
            Random3 = (int)(5 * rand.NextDouble() + 1);
        } while (Random3 == Random1 || Random3 == Random2);
            // 生成一个不重复的随机数，范围为1到5
            do
            {
                Random4 = (int)(5 * rand.NextDouble() + 1);
            } while (Random4 == Random1 || Random4 == Random2 || Random4 == Random3);
            // 生成另一个不重复的随机数，范围为1到5
            do
            {
                Random5 = (int)(5 * rand.NextDouble() + 1);
            } while (Random5 == Random1 || Random5 == Random2 || Random5 == Random3 || Random5 == Random4);

        }
        private void Play()
        {
            // 初始化游戏得分
            int Points = 100;
            // 初始化游戏胜利状态
            bool Win = false;
            // 初始化五个随机数
            int Random1, Random2, Random3, Random4, Random5;
            // 初始化用户猜测的数字
            int Guess = 0;

            // 生成五个不重复的随机数
            GetRandomNumbers(out Random1, out Random2, out Random3, out Random4, out Random5);

            # 当游戏未结束时循环执行以下代码
            while (!Win):
                # 从用户输入中获取猜测值
                Guess = PromptForGuess();

                # 如果猜测值等于 Random1，则扣除5分
                if (Guess == Random1)
                    Points -= 5;
                # 如果猜测值等于 Random2，则增加5分
                else if (Guess == Random2)
                    Points += 5;
                # 如果猜测值等于 Random3，则加倍当前分数并输出"你中了大奖！！！"
                else if (Guess == Random3)
                    Points += Points;
                    Console.WriteLine("You hit the jackpot!!!");
                # 如果猜测值等于 Random4，则增加1分
                else if (Guess == Random4)
                    Points += 1;
                # 如果猜测值等于 Random5，则扣除当前分数的50%
                else if (Guess == Random5)
                    Points -= (int)(Points * 0.5);

                # 如果当前分数大于500，则结束游戏
                if (Points > 500)
                {
                    Console.WriteLine("!!!!You Win!!!! with {0} points.", Points);  // 打印出玩家获胜的消息，显示玩家得分
                    Win = true;  // 将Win标记为true，表示玩家获胜
                }
                else
                    Console.WriteLine("You have {0} points.", Points);  // 打印出玩家得分
            }
        }

        public void PlayTheGame()
        {
            DisplayIntro();  // 调用DisplayIntro方法，显示游戏介绍

            Play();  // 调用Play方法，开始游戏
        }
    }
    class Program
    {
        static void Main(string[] args)
        {
# 创建一个新的Number对象，并调用其PlayTheGame方法
new Number().PlayTheGame();
```