# `basic-computer-games\31_Depth_Charge\csharp\Program.cs`

```

// 引入 System 命名空间
using System;

// 定义 DepthCharge 类
namespace DepthCharge
{
    // 定义 Program 类
    class Program
    {
        // 程序入口
        static void Main(string[] args)
        {
            // 创建随机数生成器对象
            var random = new Random();

            // 显示游戏横幅
            View.ShowBanner();

            // 输入游戏维度
            var dimension = Controller.InputDimension();
            // 计算最大猜测次数
            var maximumGuesses = CalculateMaximumGuesses();

            // 显示游戏说明
            View.ShowInstructions(maximumGuesses);

            // 游戏循环
            do
            {
                // 显示游戏开始信息
                View.ShowStartGame();

                // 放置潜艇的坐标
                var submarineCoordinates = PlaceSubmarine();
                var trailNumber = 1;
                var guess = (0, 0, 0);

                // 猜测循环
                do
                {
                    // 输入猜测坐标
                    guess = Controller.InputCoordinates(trailNumber);
                    // 如果猜测错误，显示猜测结果
                    if (guess != submarineCoordinates)
                        View.ShowGuessPlacement(submarineCoordinates, guess);
                }
                while (guess != submarineCoordinates && trailNumber++ < maximumGuesses);

                // 显示游戏结果
                View.ShowGameResult(submarineCoordinates, guess, trailNumber);
            }
            while (Controller.InputPlayAgain());

            // 显示结束语
            View.ShowFarewell();

            // 计算最大猜测次数
            int CalculateMaximumGuesses() =>
                (int)Math.Log2(dimension) + 1;

            // 随机放置潜艇的坐标
            (int x, int y, int depth) PlaceSubmarine() =>
                (random.Next(dimension), random.Next(dimension), random.Next(dimension));
        }
    }
}

```