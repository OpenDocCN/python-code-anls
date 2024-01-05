# `d:/src/tocomm/basic-computer-games\29_Craps\csharp\Craps\CrapsGame.cs`

```
// 创建一个名为 Craps 的命名空间
namespace Craps
{
    // 创建一个名为 Result 的枚举类型
    public enum Result
    {
        // 枚举类型中的成员，表示没有结果
        noResult,
        // 枚举类型中的成员，表示掷骰子结果为自然胜利
        naturalWin,
        // 枚举类型中的成员，表示掷骰子结果为蛇眼输
        snakeEyesLoss,
        // 枚举类型中的成员，表示掷骰子结果为自然失败
        naturalLoss,
        // 枚举类型中的成员，表示掷骰子结果为点数失败
        pointLoss,
        // 枚举类型中的成员，表示掷骰子结果为点数胜利
        pointWin,
    };

    // 创建一个名为 CrapsGame 的类
    class CrapsGame
    {
        // 创建一个只读的 UserInterface 类型的字段
        private readonly UserInterface ui;
        // 创建一个名为 dice1 的 Dice 类型的字段，并初始化为一个 Dice 对象
        private Dice dice1 = new Dice();
        // 创建一个名为 dice2 的 Dice 类型的字段，并初始化为一个 Dice 对象
        private Dice dice2 = new Dice();
        # 构造函数，接受一个 UserInterface 对象的引用
        public CrapsGame(ref UserInterface ui)
        {
            # 将传入的 UserInterface 对象赋值给类的成员变量
            this.ui = ui;
        }

        # 游戏进行方法，返回游戏结果和骰子点数
        public Result Play(out int diceRoll)
        {
            # 掷骰子，计算点数
            diceRoll = dice1.Roll() + dice2.Roll();

            # 判断是否赢得游戏
            if (Win(diceRoll))
            {
                return Result.naturalWin;
            }
            # 判断是否输掉游戏
            else if (Lose(diceRoll))
            {
                # 如果点数为2，则返回蛇眼输，否则返回自然输
                return (diceRoll == 2) ? Result.snakeEyesLoss : Result.naturalLoss;
            }
            else
            {
                # 将点数赋值给变量point
                var point = diceRoll;
                # 在用户界面上显示点数
                ui.Point(point);

                # 循环直到满足条件退出
                while (true)
                {
                    # 掷两个骰子并将结果相加
                    var newRoll = dice1.Roll() + dice2.Roll();
                    # 如果新的点数等于之前设定的点数
                    if (newRoll == point)
                    {
                        # 更新点数并返回点数胜利的结果
                        diceRoll = newRoll;
                        return Result.pointWin;
                    }
                    # 如果新的点数等于7
                    else if (newRoll == 7)
                    {
                        # 更新点数并返回点数失败的结果
                        diceRoll = newRoll;
                        return Result.pointLoss;
                    }

                    # 在用户界面上显示新的点数
                    ui.NoPoint(newRoll);
                }
            }
        }

        private bool Lose(int diceRoll)
        {
            return diceRoll == 2 || diceRoll == 3 || diceRoll == 12;  # 如果骰子点数为2、3或12，则返回true，表示输了
        }

        private bool Win(int diceRoll)
        {
            return diceRoll == 7 || diceRoll == 11;  # 如果骰子点数为7或11，则返回true，表示赢了
        }
    }
}
```