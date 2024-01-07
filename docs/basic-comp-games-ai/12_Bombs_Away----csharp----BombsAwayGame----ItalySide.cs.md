# `basic-computer-games\12_Bombs_Away\csharp\BombsAwayGame\ItalySide.cs`

```

// 定义意大利方的角色，可以执行飞行任务到阿尔巴尼亚、希腊和北非
internal class ItalySide : MissionSide
{
    public ItalySide(IUserInterface ui)
        : base(ui)
    {
    }

    // 选择任务的消息
    protected override string ChooseMissionMessage => "YOUR TARGET";

    // 所有任务的列表
    protected override IList<Mission> AllMissions => new Mission[]
    {
        new("ALBANIA", "SHOULD BE EASY -- YOU'RE FLYING A NAZI-MADE PLANE."),
        new("GREECE", "BE CAREFUL!!!"),
        new("NORTH AFRICA", "YOU'RE GOING FOR THE OIL, EH?")
    };
}

```