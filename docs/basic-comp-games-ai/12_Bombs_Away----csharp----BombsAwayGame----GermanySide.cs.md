# `basic-computer-games\12_Bombs_Away\csharp\BombsAwayGame\GermanySide.cs`

```

// 定义德国方的任务类，继承自 MissionSide 类
internal class GermanySide : MissionSide
{
    // 构造函数，接受用户界面对象作为参数
    public GermanySide(IUserInterface ui)
        : base(ui)
    {
    }

    // 选择任务的提示信息
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