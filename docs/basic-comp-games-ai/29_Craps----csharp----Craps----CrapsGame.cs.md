# `basic-computer-games\29_Craps\csharp\Craps\CrapsGame.cs`

```
// 定义名为 Result 的枚举类型，表示掷骰子游戏的结果
namespace Craps
{
    public enum Result
    {
        // 没有结果，通常用于枚举类型中，表示无效值或未初始化的情况
        noResult,
        // 大自然胜利
        naturalWin,
        // 蛇眼输
        snakeEyesLoss,
        // 大自然输
        naturalLoss,
        // 点数输
        pointLoss,
        // 点数胜利
        pointWin,
    };

    // CrapsGame 类，表示掷骰子游戏
    class CrapsGame
    {
        // 只读字段，表示用户界面
        private readonly UserInterface ui;
        // 骰子1对象
        private Dice dice1 = new Dice();
        // 骰子2对象
        private Dice dice2 = new Dice();

        // 构造函数，初始化 CrapsGame 对象
        public CrapsGame(ref UserInterface ui)
        {
            this.ui = ui;
        }

        // Play 方法，进行游戏并返回结果
        public Result Play(out int diceRoll)
        {
            // 掷骰子并计算点数
            diceRoll = dice1.Roll() + dice2.Roll();

            // 判断是否赢得游戏
            if (Win(diceRoll))
            {
                return Result.naturalWin;
            }
            // 判断是否输掉游戏
            else if (Lose(diceRoll))
            {
                // 如果点数为2，则返回蛇眼输，否则返回大自然输
                return (diceRoll == 2) ? Result.snakeEyesLoss : Result.naturalLoss;
            }
            // 进入点数阶段
            else
            {
                var point = diceRoll;
                ui.Point(point);

                // 循环直到游戏结束
                while (true)
                {
                    var newRoll = dice1.Roll() + dice2.Roll();
                    // 如果新的点数等于初始点数，则返回点数胜利
                    if (newRoll == point)
                    {
                        diceRoll = newRoll;
                        return Result.pointWin;
                    }
                    // 如果新的点数为7，则返回点数输
                    else if (newRoll == 7)
                    {
                        diceRoll = newRoll;
                        return Result.pointLoss;
                    }

                    // 显示新的点数
                    ui.NoPoint(newRoll);
                }
            }
        }

        // 判断是否输掉游戏
        private bool Lose(int diceRoll)
        {
            return diceRoll == 2 || diceRoll == 3 || diceRoll == 12;
        }

        // 判断是否赢得游戏
        private bool Win(int diceRoll)
        {
            return diceRoll == 7 || diceRoll == 11;
        }
    }
}
```