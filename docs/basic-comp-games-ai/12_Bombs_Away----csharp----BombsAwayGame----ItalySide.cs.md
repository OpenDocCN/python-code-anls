# `basic-computer-games\12_Bombs_Away\csharp\BombsAwayGame\ItalySide.cs`

```py
// 定义意大利方面的类，继承自任务方面的基类
internal class ItalySide : MissionSide
{
    // 构造函数，接受用户界面对象作为参数
    public ItalySide(IUserInterface ui)
        : base(ui)
    {
    }

    // 选择任务的消息
    protected override string ChooseMissionMessage => "YOUR TARGET";

    // 所有任务的列表
    protected override IList<Mission> AllMissions => new Mission[]
    {
        // 定义三个任务对象，包括任务名称和描述
        new("ALBANIA", "SHOULD BE EASY -- YOU'RE FLYING A NAZI-MADE PLANE."),
        new("GREECE", "BE CAREFUL!!!"),
        new("NORTH AFRICA", "YOU'RE GOING FOR THE OIL, EH?")
    };
}
```