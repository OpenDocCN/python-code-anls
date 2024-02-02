# `basic-computer-games\12_Bombs_Away\csharp\BombsAwayGame\JapanSide.cs`

```py
// 定义了日本方的飞机，执行特殊的神风特攻任务
internal class JapanSide : Side
{
    public JapanSide(IUserInterface ui)
        : base(ui)
    {
    }

    // 执行神风特攻任务，如果是第一次特攻，成功率为65%，否则执行敌方反击
    public override void Play()
    {
        UI.Output("YOU'RE FLYING A KAMIKAZE MISSION OVER THE USS LEXINGTON.");

        // 判断是否是第一次特攻
        bool isFirstMission = UI.ChooseYesOrNo("YOUR FIRST KAMIKAZE MISSION(Y OR N)?");
        if (!isFirstMission)
        {
            // 如果不是第一次特攻，执行敌方反击
            // 原始BASIC代码的207行：hitRatePercent初始化为0，但R（火炮类型）根本没有初始化。设置R = 1，即EnemyArtillery = Guns，得到相同的结果。
            EnemyCounterattack(Guns, hitRatePercent: 0);
        }
        else if (RandomFrac() > 0.65)
        {
            // 如果是第一次特攻且随机数大于0.65，任务成功
            MissionSucceeded();
        }
        else
        {
            // 如果是第一次特攻但随机数小于等于0.65，任务失败
            MissionFailed();
        }
    }
}
```