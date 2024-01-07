# `basic-computer-games\51_Hurkle\csharp\HurkleGame.cs`

```

// 引入 System 命名空间
using System;

// 定义 hurkle 命名空间
namespace hurkle
{
    // 定义 HurkleGame 类
    internal class HurkleGame
    {
        // 声明私有的 Random 对象 _random
        private readonly Random _random = new Random();
        // 声明私有的 IHurkleView 对象 _view
        private readonly IHurkleView _view;
        // 声明私有的整型变量 guesses
        private readonly int guesses;
        // 声明私有的整型变量 gridSize
        private readonly int gridSize;

        // 定义 HurkleGame 类的构造函数
        public HurkleGame(int guesses, int gridSize, IHurkleView view)
        {
            // 初始化 _view
            _view = view;
            // 初始化 this.guesses
            this.guesses = guesses;
            // 初始化 this.gridSize
            this.gridSize = gridSize;
        }

        // 定义 PlayGame 方法
        public void PlayGame()
        {
            // 生成随机的 hurklePoint 对象
            var hurklePoint = new GamePoint{
                X = _random.Next(0, gridSize),
                Y = _random.Next(0, gridSize)
            };

            // 循环进行猜测
            for(var K=1;K<=guesses;K++)
            {
                // 获取玩家的猜测点
                var guessPoint = _view.GetGuess(new GuessViewModel{CurrentGuessNumber = K});

                // 获取猜测点到 hurklePoint 的方向
                var direction = guessPoint.GetDirectionTo(hurklePoint);
                // 根据方向进行不同的处理
                switch(direction)
                {
                    // 如果方向为 None，显示胜利信息并返回
                    case CardinalDirection.None:
                        _view.ShowVictory(new VictoryViewModel{CurrentGuessNumber = K});
                        return;
                    // 否则显示方向信息并继续下一轮猜测
                    default:
                        _view.ShowDirection(new FailedGuessViewModel{Direction = direction});
                        continue;
                }
            }

            // 显示失败信息
            _view.ShowLoss(new LossViewModel{MaxGuesses = guesses, HurkleLocation = hurklePoint } );
        }
    }
}

```