# `d:/src/tocomm/basic-computer-games\12_Bombs_Away\csharp\BombsAwayGame\AlliesSide.cs`

```
/// <summary>
/// 盟军方主角。可以在解放者、B-29、B-17或兰开斯特飞行任务。
/// </summary>
internal class AlliesSide : MissionSide
{
    public AlliesSide(IUserInterface ui)
        : base(ui)
    {
    }

    // 选择任务消息
    protected override string ChooseMissionMessage => "AIRCRAFT";

    // 所有任务列表
    protected override IList<Mission> AllMissions => new Mission[]
    {
        new Mission("LIBERATOR", "YOU'VE GOT 2 TONS OF BOMBS FLYING FOR PLOESTI."),
        new Mission("B-29", "YOU'RE DUMPING THE A-BOMB ON HIROSHIMA."),
        new Mission("B-17", "YOU'RE CHASING THE BISMARK IN THE NORTH SEA."),
        new Mission("LANCASTER", "YOU'RE BUSTING A GERMAN HEAVY WATER PLANT IN THE RUHR.")
    };
}
抱歉，这段代码看起来不完整，缺少了一些关键信息，无法为其添加注释。
```