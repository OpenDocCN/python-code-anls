# `basic-computer-games\12_Bombs_Away\csharp\BombsAwayGame\JapanSide.cs`

```

// 定义了一个名为 JapanSide 的内部类，继承自 Side 类
internal class JapanSide : Side
{
    // 构造函数，接受 IUserInterface 对象作为参数
    public JapanSide(IUserInterface ui)
        : base(ui)
    {
    }

    // 重写了父类的 Play 方法
    public override void Play()
    {
        // 输出提示信息
        UI.Output("YOU'RE FLYING A KAMIKAZE MISSION OVER THE USS LEXINGTON.");

        // 判断是否是第一次神风特攻
        bool isFirstMission = UI.ChooseYesOrNo("YOUR FIRST KAMIKAZE MISSION(Y OR N)?");
        if (!isFirstMission)
        {
            // 如果不是第一次特攻，则执行敌方反击
            EnemyCounterattack(Guns, hitRatePercent: 0);
        }
        else if (RandomFrac() > 0.65)
        {
            // 如果是第一次特攻且随机数大于0.65，则任务成功
            MissionSucceeded();
        }
        else
        {
            // 否则任务失败
            MissionFailed();
        }
    }
}

```