# `basic-computer-games\12_Bombs_Away\csharp\BombsAwayGame\GermanySide.cs`

```py
/// <summary>
/// 德国方面的主角。可以执行对俄罗斯、英格兰和法国的任务。
/// </summary>
internal class GermanySide : MissionSide
{
    public GermanySide(IUserInterface ui)
        : base(ui)
    {
    }

    // 选择任务的消息
    protected override string ChooseMissionMessage => "A NAZI, EH?  OH WELL.  ARE YOU GOING FOR";

    // 所有可选任务的列表
    protected override IList<Mission> AllMissions => new Mission[]
    {
        new("RUSSIA", "YOU'RE NEARING STALINGRAD."),
        new("ENGLAND", "NEARING LONDON.  BE CAREFUL, THEY'VE GOT RADAR."),
        new("FRANCE", "NEARING VERSAILLES.  DUCK SOUP.  THEY'RE NEARLY DEFENSELESS.")
    };
}
```