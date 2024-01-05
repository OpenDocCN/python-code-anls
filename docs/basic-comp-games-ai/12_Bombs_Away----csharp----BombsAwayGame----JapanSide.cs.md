# `12_Bombs_Away\csharp\BombsAwayGame\JapanSide.cs`

```
// 命名空间声明，指明该类所属的命名空间
namespace BombsAwayGame;

/// <summary>
/// 日本方主角。执行神风特攻任务，其逻辑与 <see cref="MissionSide"/> 不同。
/// </summary>
// 内部类声明，表示该类只能在当前程序集中访问
internal class JapanSide : Side
{
    // 构造函数，接受一个 IUserInterface 对象作为参数
    public JapanSide(IUserInterface ui)
        : base(ui)
    {
    }

    /// <summary>
    /// 执行神风特攻任务。如果是第一次神风特攻任务，成功概率为 65%。如果不是第一次神风特攻任务，则执行敌方反击。
    /// </summary>
    // 重写基类的 Play 方法
    public override void Play()
    {
        // 输出提示信息到用户界面
        UI.Output("YOU'RE FLYING A KAMIKAZE MISSION OVER THE USS LEXINGTON.");
        // 询问玩家是否进行第一次特技飞行任务
        bool isFirstMission = UI.ChooseYesOrNo("YOUR FIRST KAMIKAZE MISSION(Y OR N)?");
        // 如果不是第一次任务
        if (!isFirstMission)
        {
            // 在原始BASIC代码的第207行：hitRatePercent被初始化为0，
            // 但是炮兵的类型R根本没有被初始化。设置R = 1，也就是说EnemyArtillery = Guns，得到的结果是一样的。
            EnemyCounterattack(Guns, hitRatePercent: 0);
        }
        // 如果是第一次任务，并且随机小数大于0.65
        else if (RandomFrac() > 0.65)
        {
            // 任务成功
            MissionSucceeded();
        }
        // 如果是第一次任务，并且随机小数小于等于0.65
        else
        {
            // 任务失败
            MissionFailed();
        }
    }
}
```