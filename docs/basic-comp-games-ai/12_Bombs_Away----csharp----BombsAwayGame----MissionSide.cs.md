# `12_Bombs_Away\csharp\BombsAwayGame\MissionSide.cs`

```
// 命名空间声明
namespace BombsAwayGame;

/// <summary>
/// 代表选择标准（非神风）任务的主角。
/// </summary>
// 内部抽象类 MissionSide 继承自 Side 类
internal abstract class MissionSide : Side
{
    /// <summary>
    /// 使用给定的用户界面创建实例。
    /// </summary>
    /// <param name="ui">要使用的用户界面。</param>
    // 构造函数，调用基类的构造函数
    public MissionSide(IUserInterface ui)
        : base(ui)
    {
    }

    /// <summary>
    /// 先前飞行任务的合理上限。
    /// </summary>
    // 声明私有常量 MaxMissionCount，值为 160
    private const int MaxMissionCount = 160;
    /// <summary>
    /// 选择一个任务并尝试完成它。如果尝试失败，执行敌人的反击。
    /// </summary>
    public override void Play()
    {
        // 选择一个任务
        Mission mission = ChooseMission();
        // 在界面上输出任务描述
        UI.Output(mission.Description);

        // 从界面获取任务次数
        int missionCount = MissionCountFromUI();
        // 对任务次数进行评论
        CommentOnMissionCount(missionCount);

        // 尝试完成任务
        AttemptMission(missionCount);
    }

    /// <summary>
    /// 选择一个任务。
    /// </summary>
    /// <returns>选择的任务。</returns>
    private Mission ChooseMission()
    {
        // 获取所有任务列表
        IList<Mission> missions = AllMissions;
        // 将任务名称转换为数组
        string[] missionNames = missions.Select(a => a.Name).ToArray();
        // 从用户界面中选择一个任务的索引
        int index = UI.Choose(ChooseMissionMessage, missionNames);
        // 返回选择的任务
        return missions[index];
    }

    /// <summary>
    /// 选择任务时显示的消息
    /// </summary>
    protected abstract string ChooseMissionMessage { get; }

    /// <summary>
    /// 所有可选择的任务列表
    /// </summary>
    protected abstract IList<Mission> AllMissions { get; }

    /// <summary>
    /// 从用户界面获取任务数量。如果任务数量超过合理的最大值，则再次询问用户界面
    /// </summary>
    /// <returns>Mission count from UI.</returns>
    private int MissionCountFromUI()
    {
        // 定义常量字符串，用于提示用户输入任务数量
        const string HowManyMissions = "HOW MANY MISSIONS HAVE YOU FLOWN?";
        // 将提示信息赋值给输入信息变量
        string inputMessage = HowManyMissions;

        // 定义变量用于判断输入结果是否有效
        bool resultIsValid;
        // 定义变量用于存储输入结果
        int result;
        // 使用循环确保用户输入的任务数量有效
        do
        {
            // 在UI上输出提示信息
            UI.Output(inputMessage);
            // 从UI获取用户输入的整数
            result = UI.InputInteger();
            // 如果输入结果小于0，则提示用户任务数量不能为负数
            if (result < 0)
            {
                UI.Output($"NUMBER OF MISSIONS CAN'T BE NEGATIVE.");
                resultIsValid = false;
            }
            // 如果输入结果大于最大任务数量，则提示用户输入结果无效
            else if (result > MaxMissionCount)
            {
                resultIsValid = false;
                UI.Output($"MISSIONS, NOT MILES...{MaxMissionCount} MISSIONS IS HIGH EVEN FOR OLD-TIMERS.");
```
这行代码输出一个关于任务数量的消息，如果任务数量超过了最大任务数量(MaxMissionCount)，则输出警告消息。

```
                inputMessage = "NOW THEN, " + HowManyMissions;
```
这行代码将一个包含任务数量信息的消息赋值给inputMessage变量。

```
            }
            else
            {
                resultIsValid = true;
            }
        }
        while (!resultIsValid);
```
这段代码是一个do-while循环，当resultIsValid为false时，继续执行循环。当resultIsValid为true时，跳出循环。

```
        return result;
    }
```
返回result变量的值。

```
    /// <summary>
    /// Display a message about the given mission count, if it is unusually high or low.
    /// </summary>
    /// <param name="missionCount">Mission count to comment on.</param>
```
这段代码是一个注释块，用于说明CommentOnMissionCount方法的作用和参数。

```
    private void CommentOnMissionCount(int missionCount)
    {
        if (missionCount >= 100)
```
这是一个私有方法CommentOnMissionCount，接受一个整数参数missionCount。如果missionCount大于等于100，则执行下面的代码。
        {
            // 如果任务次数小于 25，输出“FRESH OUT OF TRAINING, EH?”
            UI.Output("FRESH OUT OF TRAINING, EH?");
        }
        else if (missionCount < 25)
        {
            // 如果任务次数小于 25，输出“FRESH OUT OF TRAINING, EH?”
            UI.Output("FRESH OUT OF TRAINING, EH?");
        }
    }

    /// <summary>
    /// Attempt mission.
    /// </summary>
    /// <param name="missionCount">Number of missions previously flown. Higher mission counts will yield a higher probability of success.</param>
    private void AttemptMission(int missionCount)
    {
        // 如果任务次数小于随机生成的一个数字，调用 MissedTarget() 方法
        if (missionCount < RandomInteger(0, MaxMissionCount))
        {
            MissedTarget();
        }
        else
        {
            MissionSucceeded(); // 调用 MissionSucceeded() 函数，表示任务成功
        }
    }

    /// <summary>
    /// Display message indicating that target was missed. Choose enemy artillery and perform a counterattack.
    /// </summary>
    private void MissedTarget()
    {
        UI.Output("MISSED TARGET BY " + (2 + RandomInteger(0, 30)) + " MILES!"); // 在界面上输出“MISSED TARGET BY X MILES!”的消息，X为2到32之间的随机整数
        UI.Output("NOW YOU'RE REALLY IN FOR IT !!"); // 在界面上输出“NOW YOU'RE REALLY IN FOR IT !!”的消息

        // Choose enemy and counterattack.
        EnemyArtillery enemyArtillery = ChooseEnemyArtillery(); // 选择敌方炮兵

        if (enemyArtillery == Missiles) // 如果敌方炮兵是导弹
        {
            EnemyCounterattack(enemyArtillery, hitRatePercent: 0); // 进行反击，命中率为0
        }
        else
        {
            // 从用户界面获取敌人的命中率百分比
            int hitRatePercent = EnemyHitRatePercentFromUI();
            // 如果命中率低于最小敌人命中率百分比，则输出信息并调用MissionFailed函数
            if (hitRatePercent < MinEnemyHitRatePercent)
            {
                UI.Output("YOU LIE, BUT YOU'LL PAY...");
                MissionFailed();
            }
            // 如果命中率高于最小敌人命中率百分比，则调用EnemyCounterattack函数
            else
            {
                EnemyCounterattack(enemyArtillery, hitRatePercent);
            }
        }
    }

    /// <summary>
    /// 从用户界面选择敌人的炮火
    /// </summary>
    /// <returns>选择的炮火</returns>
    private EnemyArtillery ChooseEnemyArtillery()
    {
        // 创建一个敌方炮兵数组，包括 Guns, Missiles, Both 三种类型
        EnemyArtillery[] artilleries = new EnemyArtillery[] { Guns, Missiles, Both };
        // 从炮兵数组中提取炮兵名称，转换为数组
        string[] artilleryNames = artilleries.Select(a => a.Name).ToArray();
        // 从用户界面中选择敌方炮兵类型
        int index = UI.Choose("DOES THE ENEMY HAVE", artilleryNames);
        // 返回用户选择的敌方炮兵类型
        return artilleries[index];
    }

    /// <summary>
    /// 最小允许的命中率百分比。
    /// </summary>
    private const int MinEnemyHitRatePercent = 10;

    /// <summary>
    /// 最大允许的命中率百分比。
    /// </summary>
    private const int MaxEnemyHitRatePercent = 50;

    /// <summary>
    /// 从用户界面获取敌方命中率百分比。值必须在零和 <see cref="MaxEnemyHitRatePercent"/> 之间。
    /// 如果值小于 <see cref="MinEnemyHitRatePercent"/>，则任务自动失败，因为用户
    /// <summary>
    /// 从用户界面获取敌人命中率百分比
    /// </summary>
    /// <returns>从用户界面获取的敌人命中率百分比</returns>
    private int EnemyHitRatePercentFromUI()
    {
        // 输出敌人炮手的命中率百分比范围
        UI.Output($"WHAT'S THE PERCENT HIT RATE OF ENEMY GUNNERS ({MinEnemyHitRatePercent} TO {MaxEnemyHitRatePercent})");

        bool resultIsValid;
        int result;
        do
        {
            // 从用户输入获取一个整数
            result = UI.InputInteger();
            // 如果输入的数在规定的范围内，则结果有效
            if (0 <= result && result <= MaxEnemyHitRatePercent)
            {
                resultIsValid = true;
            }
            else
            {
                resultIsValid = false;
                UI.Output($"NUMBER MUST BE FROM {MinEnemyHitRatePercent} TO {MaxEnemyHitRatePercent}");  # 输出提示信息，显示最小和最大敌人命中率百分比
            }  # 结束 if 语句
        }  # 结束 do-while 循环
        while (!resultIsValid);  # 当结果无效时继续循环
        return result;  # 返回有效的结果
    }  # 结束方法
}  # 结束类
```