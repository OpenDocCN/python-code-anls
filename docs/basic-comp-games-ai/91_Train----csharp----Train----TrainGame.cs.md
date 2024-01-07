# `basic-computer-games\91_Train\csharp\Train\TrainGame.cs`

```

// 引入命名空间
using System;
using System.Linq;

// 定义 TrainGame 类
namespace Train
{
    public class TrainGame
    {
        // 创建 Random 对象
        private Random Rnd { get; } = new Random();
        // 定义允许的百分比差异
        private readonly int ALLOWED_PERCENTAGE_DIFFERENCE = 5;

        // 主函数
        static void Main()
        {
            // 创建 TrainGame 对象
            TrainGame train = new TrainGame();
            // 调用 GameLoop 方法
            train.GameLoop();
        }

        // 游戏循环
        public void GameLoop()
        {
            // 显示游戏介绍文本
            DisplayIntroText();

            do
            {
                // 进行游戏
                PlayGame();
            } while (TryAgain());
        }

        // 进行游戏
        private void PlayGame()
        {
            // 生成随机的车速、时间差和火车速度
            int carSpeed = (int)GenerateRandomNumber(40, 25);
            int timeDifference = (int)GenerateRandomNumber(5, 15);
            int trainSpeed = (int)GenerateRandomNumber(20, 19);

            // 输出问题
            Console.WriteLine($"A CAR TRAVELING {carSpeed} MPH CAN MAKE A CERTAIN TRIP IN");
            Console.WriteLine($"{timeDifference} HOURS LESS THAN A TRAIN TRAVELING AT {trainSpeed} MPH");
            Console.WriteLine("HOW LONG DOES THE TRIP TAKE BY CAR?");

            // 获取用户输入
            double userInputCarJourneyDuration = double.Parse(Console.ReadLine());
            // 计算实际的车程时间
            double actualCarJourneyDuration = CalculateCarJourneyDuration(carSpeed, timeDifference, trainSpeed);
            // 计算百分比差异
            int percentageDifference = CalculatePercentageDifference(userInputCarJourneyDuration, actualCarJourneyDuration);

            // 判断是否在允许的差异范围内
            if (IsWithinAllowedDifference(percentageDifference, ALLOWED_PERCENTAGE_DIFFERENCE))
            {
                Console.WriteLine($"GOOD! ANSWER WITHIN {percentageDifference} PERCENT.");
            }
            else
            {
                Console.WriteLine($"SORRY.  YOU WERE OFF BY {percentageDifference} PERCENT.");
            }
            Console.WriteLine($"CORRECT ANSWER IS {actualCarJourneyDuration} HOURS.");
        }

        // 判断百分比差异是否在允许范围内
        public static bool IsWithinAllowedDifference(int percentageDifference, int allowedDifference)
        {
            return percentageDifference <= allowedDifference;
        }

        // 计算百分比差异
        private static int CalculatePercentageDifference(double userInputCarJourneyDuration, double carJourneyDuration)
        {
            return (int)(Math.Abs((carJourneyDuration - userInputCarJourneyDuration) * 100 / userInputCarJourneyDuration) + .5);
        }

        // 计算车程时间
        public static double CalculateCarJourneyDuration(double carSpeed, double timeDifference, double trainSpeed)
        {
            return timeDifference * trainSpeed / (carSpeed - trainSpeed);
        }

        // 生成随机数
        public double GenerateRandomNumber(int baseSpeed, int multiplier)
        {
            return multiplier * Rnd.NextDouble() + baseSpeed;
        }

        // 再次尝试
        private bool TryAgain()
        {
            Console.WriteLine("ANOTHER PROBLEM (YES OR NO)? ");
            return IsInputYes(Console.ReadLine());
        }

        // 判断输入是否为 Yes
        public static bool IsInputYes(string consoleInput)
        {
            var options = new string[] { "Y", "YES" };
            return options.Any(o => o.Equals(consoleInput, StringComparison.CurrentCultureIgnoreCase));
        }

        // 显示游戏介绍文本
        private void DisplayIntroText()
        {
            Console.WriteLine("TRAIN");
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine();
            Console.WriteLine("TIME - SPEED DISTANCE EXERCISE");
            Console.WriteLine();
        }
    }
}

```