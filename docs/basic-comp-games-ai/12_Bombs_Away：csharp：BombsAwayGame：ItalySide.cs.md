# `12_Bombs_Away\csharp\BombsAwayGame\ItalySide.cs`

```
/// <summary>
/// Italy protagonist. Can fly missions to Albania, Greece, and North Africa.
/// </summary>
internal class ItalySide : MissionSide
{
    public ItalySide(IUserInterface ui)
        : base(ui)
    {
    }

    // 选择任务时显示的消息
    protected override string ChooseMissionMessage => "YOUR TARGET";

    // 所有任务的列表
    protected override IList<Mission> AllMissions => new Mission[]
    {
        // 创建任务对象，包括任务名称和任务描述
        new Mission("ALBANIA", "SHOULD BE EASY -- YOU'RE FLYING A NAZI-MADE PLANE."),
        new Mission("GREECE", "BE CAREFUL!!!"),
        new Mission("NORTH AFRICA", "YOU'RE GOING FOR THE OIL, EH?")
    };
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```