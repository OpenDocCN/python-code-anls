# `basic-computer-games\12_Bombs_Away\csharp\BombsAwayGame\MissionSide.cs`

```
// 表示一个选择标准（非敢死队）任务的主角
internal abstract class MissionSide : Side
{
    // 使用给定的 UI 创建实例
    /// <param name="ui">要使用的 UI。</param>
    public MissionSide(IUserInterface ui)
        : base(ui)
    {
    }

    // 先前飞行任务的合理上限
    private const int MaxMissionCount = 160;

    // 选择一个任务并尝试执行。如果尝试失败，则执行敌方反击。
    public override void Play()
    {
        Mission mission = ChooseMission();
        UI.Output(mission.Description);

        int missionCount = MissionCountFromUI();
        CommentOnMissionCount(missionCount);

        AttemptMission(missionCount);
    }

    // 选择一个任务
    /// <returns>选择的任务。</returns>
    private Mission ChooseMission()
    {
        IList<Mission> missions = AllMissions;
        string[] missionNames = missions.Select(a => a.Name).ToArray();
        int index = UI.Choose(ChooseMissionMessage, missionNames);
        return missions[index];
    }

    // 选择任务时显示的消息
    protected abstract string ChooseMissionMessage { get; }

    // 可供选择的所有任务
    protected abstract IList<Mission> AllMissions { get; }

    // 从 UI 获取任务计数。如果任务计数超过合理的最大值，则再次询问 UI。
    /// <returns>来自 UI 的任务计数。</returns>
    private int MissionCountFromUI()
    {
        // 定义一个常量字符串，表示“你飞了多少次任务？”
        const string HowManyMissions = "HOW MANY MISSIONS HAVE YOU FLOWN?";
        // 初始化一个字符串变量，赋值为上面定义的常量字符串
        string inputMessage = HowManyMissions;
    
        // 定义一个布尔变量，表示结果是否有效
        bool resultIsValid;
        // 定义一个整型变量，表示结果
        int result;
        // 使用 do...while 循环，直到结果有效为止
        do
        {
            // 输出提示信息
            UI.Output(inputMessage);
            // 获取用户输入的整数
            result = UI.InputInteger();
            // 如果结果小于 0
            if (result < 0)
            {
                // 输出“任务数量不能为负数”
                UI.Output($"NUMBER OF MISSIONS CAN'T BE NEGATIVE.");
                // 将结果标记为无效
                resultIsValid = false;
            }
            // 如果结果大于最大任务数量
            else if (result > MaxMissionCount)
            {
                // 将结果标记为无效
                resultIsValid = false;
                // 输出“任务，不是英里...100 个任务对老手来说也太多了”
                UI.Output($"MISSIONS, NOT MILES...{MaxMissionCount} MISSIONS IS HIGH EVEN FOR OLD-TIMERS.");
                // 更新提示信息
                inputMessage = "NOW THEN, " + HowManyMissions;
            }
            // 如果结果有效
            else
            {
                // 将结果标记为有效
                resultIsValid = true;
            }
        }
        while (!resultIsValid);
    
        // 返回结果
        return result;
    }
    
    /// <summary>
    /// 根据给定的任务数量显示一条消息，如果数量异常高或低
    /// </summary>
    /// <param name="missionCount">要评论的任务数量</param>
    private void CommentOnMissionCount(int missionCount)
    {
        // 如果任务数量大于等于 100
        if (missionCount >= 100)
        {
            // 输出“那太冒险了！”
            UI.Output("THAT'S PUSHING THE ODDS!");
        }
        // 如果任务数量小于 25
        else if (missionCount < 25)
        {
            // 输出“刚刚结束训练，嗯？”
            UI.Output("FRESH OUT OF TRAINING, EH?");
        }
    }
    
    /// <summary>
    /// 尝试执行任务
    /// </summary>
    /// <param name="missionCount">先前执行的任务数量。更高的任务数量将产生更高的成功概率。</param>
    private void AttemptMission(int missionCount)
    {
        // 如果任务数量小于一个随机整数（范围为 0 到最大任务数量）
        if (missionCount < RandomInteger(0, MaxMissionCount))
        {
            // 调用 MissedTarget 方法
            MissedTarget();
        }
        // 否则
        else
        {
            // 调用 MissionSucceeded 方法
            MissionSucceeded();
        }
    }
    
    /// <summary>
    /// 显示指示未命中目标的消息。选择敌方炮兵并执行反击。
    /// </summary>
    private void MissedTarget()
    {
        // 输出未命中目标的距离，范围在2到32之间
        UI.Output("MISSED TARGET BY " + (2 + RandomInteger(0, 30)) + " MILES!");
        // 输出提示信息
        UI.Output("NOW YOU'RE REALLY IN FOR IT !!");

        // 选择敌人并进行反击
        EnemyArtillery enemyArtillery = ChooseEnemyArtillery();

        // 如果敌人的炮火是导弹
        if (enemyArtillery == Missiles)
        {
            // 对敌人进行反击，命中率为0
            EnemyCounterattack(enemyArtillery, hitRatePercent: 0);
        }
        else
        {
            // 从UI获取敌人的命中率百分比
            int hitRatePercent = EnemyHitRatePercentFromUI();
            // 如果命中率低于最小允许值
            if (hitRatePercent < MinEnemyHitRatePercent)
            {
                // 输出提示信息
                UI.Output("YOU LIE, BUT YOU'LL PAY...");
                // 任务失败
                MissionFailed();
            }
            else
            {
                // 对敌人进行反击，命中率为获取到的值
                EnemyCounterattack(enemyArtillery, hitRatePercent);
            }
        }
    }

    /// <summary>
    /// 从UI选择敌人的炮火类型
    /// </summary>
    /// <returns>选择的炮火类型</returns>
    private EnemyArtillery ChooseEnemyArtillery()
    {
        // 定义敌人的炮火类型数组
        EnemyArtillery[] artilleries = new EnemyArtillery[] { Guns, Missiles, Both };
        // 获取炮火类型的名称数组
        string[] artilleryNames = artilleries.Select(a => a.Name).ToArray();
        // 从UI中选择炮火类型
        int index = UI.Choose("DOES THE ENEMY HAVE", artilleryNames);
        // 返回选择的炮火类型
        return artilleries[index];
    }

    /// <summary>
    /// 敌人命中率的最小允许值
    /// </summary>
    private const int MinEnemyHitRatePercent = 10;

    /// <summary>
    /// 敌人命中率的最大允许值
    /// </summary>
    private const int MaxEnemyHitRatePercent = 50;

    /// <summary>
    /// 从UI获取敌人的命中率百分比。值必须在0到<see cref="MaxEnemyHitRatePercent"/>之间。
    /// 如果值小于<see cref="MinEnemyHitRatePercent"/>，则任务自动失败，因为假设用户不诚实。
    /// </summary>
    /// <returns>从UI获取的敌人命中率百分比</returns>
    private int EnemyHitRatePercentFromUI()
    {
        # 输出敌方炮手的命中率范围
        UI.Output($"WHAT'S THE PERCENT HIT RATE OF ENEMY GUNNERS ({MinEnemyHitRatePercent} TO {MaxEnemyHitRatePercent})");
    
        # 定义变量，用于标记输入结果是否有效
        bool resultIsValid;
        # 定义变量，用于存储输入结果
        int result;
        # 循环，直到输入结果有效为止
        do
        {
            # 获取用户输入的整数
            result = UI.InputInteger();
            # 如果输入结果在规定范围内，则标记为有效
            if (0 <= result && result <= MaxEnemyHitRatePercent)
            {
                resultIsValid = true;
            }
            # 如果输入结果不在规定范围内，则标记为无效，并提示用户重新输入
            else
            {
                resultIsValid = false;
                UI.Output($"NUMBER MUST BE FROM {MinEnemyHitRatePercent} TO {MaxEnemyHitRatePercent}");
            }
        }
        while (!resultIsValid);
    
        # 返回有效的输入结果
        return result;
    }
# 闭合前面的函数定义
```