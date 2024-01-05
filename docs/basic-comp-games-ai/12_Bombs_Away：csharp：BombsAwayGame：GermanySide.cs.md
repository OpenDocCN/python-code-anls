# `d:/src/tocomm/basic-computer-games\12_Bombs_Away\csharp\BombsAwayGame\GermanySide.cs`

```
/// <summary>
/// Germany protagonist. Can fly missions to Russia, England, and France.
/// </summary>
internal class GermanySide : MissionSide
{
    public GermanySide(IUserInterface ui)
        : base(ui) // 调用父类的构造函数，传入用户界面对象
    {
    }

    protected override string ChooseMissionMessage => "A NAZI, EH?  OH WELL.  ARE YOU GOING FOR"; // 重写父类的属性，返回选择任务的消息

    protected override IList<Mission> AllMissions => new Mission[] // 重写父类的属性，返回所有任务的列表
    {
        new("RUSSIA", "YOU'RE NEARING STALINGRAD."), // 创建新的任务对象，传入任务名称和描述
        new("ENGLAND", "NEARING LONDON.  BE CAREFUL, THEY'VE GOT RADAR."), // 创建新的任务对象，传入任务名称和描述
        new("FRANCE", "NEARING VERSAILLES.  DUCK SOUP.  THEY'RE NEARLY DEFENSELESS.") // 创建新的任务对象，传入任务名称和描述
    };
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```