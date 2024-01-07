# `basic-computer-games\29_Craps\csharp\Craps\CrapsGame.cs`

```

// 声明一个名为 Result 的枚举类型，用于表示掷骰子游戏的结果
public enum Result
{
    // 在这个程序中没有使用，但通常在枚举中包含一个 "none" 值是一个好主意，
    // 这样你可以将枚举的实例设置为 "invalid" 或初始化为 "none of the valid values"
    noResult,
    naturalWin,
    snakeEyesLoss,
    naturalLoss,
    pointLoss,
    pointWin,
};

// 定义 CrapsGame 类
class CrapsGame
{
    // 声明一个私有的 UserInterface 类型的变量 ui
    private readonly UserInterface ui;
    // 创建两个骰子对象
    private Dice dice1 = new Dice();
    private Dice dice2 = new Dice();

    // CrapsGame 类的构造函数，接受一个 UserInterface 类型的引用参数
    public CrapsGame(ref UserInterface ui)
    {
        this.ui = ui;
    }

    // Play 方法，用于进行游戏并返回结果
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
            // 如果点数为2，则返回 snakeEyesLoss，否则返回 naturalLoss
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
                // 如果新的点数等于之前的点数，则赢得游戏
                if (newRoll == point)
                {
                    diceRoll = newRoll;
                    return Result.pointWin;
                }
                // 如果新的点数为7，则输掉游戏
                else if (newRoll == 7)
                {
                    diceRoll = newRoll;
                    return Result.pointLoss;
                }

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

```