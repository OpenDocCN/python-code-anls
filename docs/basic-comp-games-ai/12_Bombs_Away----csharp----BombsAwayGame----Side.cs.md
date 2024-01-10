# `basic-computer-games\12_Bombs_Away\csharp\BombsAwayGame\Side.cs`

```
// 表示游戏中的主角
internal abstract class Side
{
    // 使用给定的用户界面创建实例
    // <param name="ui">要使用的用户界面</param>
    public Side(IUserInterface ui)
    {
        UI = ui;
    }

    // 执行这一方的游戏
    public abstract void Play();

    // 供构造函数使用的用户界面
    protected IUserInterface UI { get; }

    // 用于当前游戏过程的随机数生成器
    private readonly Random _random = new();

    // 获取大于等于零且小于一的随机浮点数
    // <returns>大于等于零且小于一的随机浮点数</returns>
    protected double RandomFrac() => _random.NextDouble();

    // 获取指定范围内的随机整数
    // <param name="minValue">返回的数字的包含下限</param>
    // <param name="maxValue">返回的数字的排除上限</param>
    // <returns>指定范围内的随机整数</returns>
    protected int RandomInteger(int minValue, int maxValue) => _random.Next(minValue: minValue, maxValue: maxValue);

    // 显示指示任务成功的消息
    protected void MissionSucceeded()
    {
        UI.Output("DIRECT HIT!!!! " + RandomInteger(0, 100) + " KILLED.");
        UI.Output("MISSION SUCCESSFUL.");
    }

    // 获取敌方火炮的类型为Guns
    protected EnemyArtillery Guns { get; } = new("GUNS", 0);

    // 获取敌方火炮的类型为Missiles
    protected EnemyArtillery Missiles { get; } = new("MISSILES", 35);

    // ...
}
    /// Gets the Both Guns and Missiles type of enemy artillery.
    /// </summary>
    protected EnemyArtillery Both { get; } = new("BOTH", 35);
    
    /// <summary>
    /// Perform enemy counterattack using the given artillery and hit rate percent.
    /// </summary>
    /// <param name="artillery">Enemy artillery to use.</param>
    /// <param name="hitRatePercent">Hit rate percent for enemy.</param>
    protected void EnemyCounterattack(EnemyArtillery artillery, int hitRatePercent)
    {
        // 如果命中率百分比加上火炮的准确度大于随机生成的0到100之间的整数
        if (hitRatePercent + artillery.Accuracy > RandomInteger(0, 100))
        {
            // 任务失败
            MissionFailed();
        }
        else
        {
            // 输出消息表示成功通过敌方炮火
            UI.Output("YOU MADE IT THROUGH TREMENDOUS FLAK!!");
        }
    }
    
    /// <summary>
    /// Display messages indicating the mission failed.
    /// </summary>
    protected void MissionFailed()
    {
        // 输出任务失败的相关消息
        UI.Output("* * * * BOOM * * * *");
        UI.Output("YOU HAVE BEEN SHOT DOWN.....");
        UI.Output("DEARLY BELOVED, WE ARE GATHERED HERE TODAY TO PAY OUR");
        UI.Output("LAST TRIBUTE...");
    }
# 闭合前面的函数定义
```