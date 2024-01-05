# `d:/src/tocomm/basic-computer-games\51_Hurkle\csharp\HurkleGame.cs`

```
using System;  # 导入系统命名空间

namespace hurkle  # 定义名为 hurkle 的命名空间
{
    internal class HurkleGame  # 定义名为 HurkleGame 的内部类
    {
        private readonly Random _random = new Random();  # 创建名为 _random 的只读随机数生成器对象
        private readonly IHurkleView _view;  # 创建名为 _view 的只读 IHurkleView 接口对象
        private readonly int guesses;  # 创建名为 guesses 的只读整数变量
        private readonly int gridSize;  # 创建名为 gridSize 的只读整数变量

        public HurkleGame(int guesses, int gridSize, IHurkleView view)  # 定义名为 HurkleGame 的构造函数，接受 guesses、gridSize 和 view 三个参数
        {
            _view = view;  # 将 view 参数赋值给 _view 变量
            this.guesses = guesses;  # 将 guesses 参数赋值给 this.guesses 变量
            this.gridSize = gridSize;  # 将 gridSize 参数赋值给 this.gridSize 变量
        }

        public void PlayGame()  # 定义名为 PlayGame 的公共方法
        {
            // 生成一个介于0和1之间的浮点数，然后乘以网格的大小，得到一个介于1和10之间的数字。C#允许直接这样做。
            var hurklePoint = new GamePoint{
                X = _random.Next(0, gridSize),  // 生成一个介于0和网格大小之间的随机X坐标
                Y = _random.Next(0, gridSize)   // 生成一个介于0和网格大小之间的随机Y坐标
            };

            for(var K=1;K<=guesses;K++)
            {
                var guessPoint = _view.GetGuess(new GuessViewModel{CurrentGuessNumber = K});  // 获取玩家的猜测点

                var direction = guessPoint.GetDirectionTo(hurklePoint);  // 获取猜测点到目标点的方向
                switch(direction)
                {
                    case CardinalDirection.None:  // 如果方向为None，表示猜测点与目标点重合
                        _view.ShowVictory(new VictoryViewModel{CurrentGuessNumber = K});  // 显示胜利信息
                        return;  // 结束游戏
                    default:
                        _view.ShowDirection(new FailedGuessViewModel{Direction = direction});  // 显示猜测方向信息
                        continue;  # 继续下一次循环，跳过当前循环中剩余的代码
                }
            }

            _view.ShowLoss(new LossViewModel{MaxGuesses = guesses, HurkleLocation = hurklePoint } );  # 在视图中显示失败信息，包括最大猜测次数和Hurkle的位置
        }
    }
}
```