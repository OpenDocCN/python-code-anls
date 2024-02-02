# `basic-computer-games\51_Hurkle\csharp\HurkleGame.cs`

```py
using System;

namespace hurkle
{
    internal class HurkleGame
    {
        private readonly Random _random = new Random();  // 创建一个随机数生成器对象
        private readonly IHurkleView _view;  // 创建一个 IHurkleView 接口类型的私有变量
        private readonly int guesses;  // 创建一个整型的只读变量 guesses
        private readonly int gridSize;  // 创建一个整型的只读变量 gridSize

        public HurkleGame(int guesses, int gridSize, IHurkleView view)
        {
            _view = view;  // 将传入的 view 参数赋值给私有变量 _view
            this.guesses = guesses;  // 将传入的 guesses 参数赋值给只读变量 guesses
            this.gridSize = gridSize;  // 将传入的 gridSize 参数赋值给只读变量 gridSize
        }

        public void PlayGame()
        {
            // BASIC program was generating a float between 0 and 1
            // then multiplying by the size of the grid to to a number
            // between 1 and 10. C# allows you to do that directly.
            var hurklePoint = new GamePoint{  // 创建一个 GamePoint 对象，并初始化 X 和 Y 属性
                X = _random.Next(0, gridSize),  // 生成一个 0 到 gridSize 之间的随机整数，并赋值给 X 属性
                Y = _random.Next(0, gridSize)  // 生成一个 0 到 gridSize 之间的随机整数，并赋值给 Y 属性
            };

            for(var K=1;K<=guesses;K++)  // 循环执行猜测的次数
            {
                var guessPoint = _view.GetGuess(new GuessViewModel{CurrentGuessNumber = K});  // 通过视图对象获取玩家的猜测点

                var direction = guessPoint.GetDirectionTo(hurklePoint);  // 获取玩家猜测点与隐藏点之间的方向
                switch(direction)  // 根据方向进行判断
                {
                    case CardinalDirection.None:  // 如果方向为 None
                        _view.ShowVictory(new VictoryViewModel{CurrentGuessNumber = K});  // 在视图上显示胜利信息
                        return;  // 结束游戏
                    default:  // 如果方向不为 None
                        _view.ShowDirection(new FailedGuessViewModel{Direction = direction});  // 在视图上显示猜测方向
                        continue;  // 继续下一次循环
                }
            }

            _view.ShowLoss(new LossViewModel{MaxGuesses = guesses, HurkleLocation = hurklePoint } );  // 在视图上显示失败信息和隐藏点的位置
        }
    }
}
```