# `basic-computer-games\17_Bullfight\csharp\Program.cs`

```

// 命名空间声明，表示该类属于 Game 命名空间
namespace Game
{
    // 程序入口点
    class Program
    {
        static void Main()
        {
            // 调用 Controller 类的 StartGame 方法开始游戏
            Controller.StartGame();

            // 创建中介者对象
            var mediator = new Mediator();
            // 遍历 BullFight 类的 Begin 方法返回的事件集合
            foreach (var evt in BullFight.Begin(mediator))
            {
                // 根据事件类型进行不同的处理
                switch (evt)
                {
                    // 比赛开始事件
                    case Events.MatchStarted matchStarted:
                        // 调用 View 类的 ShowStartingConditions 方法显示比赛开始的条件
                        View.ShowStartingConditions(matchStarted);
                        break;

                    // 公牛冲锋事件
                    case Events.BullCharging bullCharging:
                        // 调用 View 类的 ShowStartOfPass 方法显示冲锋开始
                        View.ShowStartOfPass(bullCharging.PassNumber);
                        // 调用 Controller 类的 GetPlayerIntention 方法获取玩家意图
                        var (action, riskLevel) = Controller.GetPlayerIntention(bullCharging.PassNumber);
                        // 根据玩家意图进行不同的处理
                        switch (action)
                        {
                            case Action.Dodge:
                                // 调用中介者对象的 Dodge 方法进行躲避
                                mediator.Dodge(riskLevel);
                                break;
                            case Action.Kill:
                                // 调用中介者对象的 Kill 方法进行击杀
                                mediator.Kill(riskLevel);
                                break;
                            case Action.Panic:
                                // 调用中介者对象的 Panic 方法进行恐慌
                                mediator.Panic();
                                break;
                        }
                        break;

                    // 玩家被公牛刺伤事件
                    case Events.PlayerGored playerGored:
                        // 调用 View 类的 ShowPlayerGored 方法显示玩家被刺伤情况
                        View.ShowPlayerGored(playerGored.Panicked, playerGored.FirstGoring);
                        break;

                    // 玩家幸存事件
                    case Events.PlayerSurvived:
                        // 调用 View 类的 ShowPlayerSurvives 方法显示玩家幸存
                        View.ShowPlayerSurvives();
                        // 根据玩家是否逃离斗牛场进行不同的处理
                        if (Controller.GetPlayerRunsFromRing())
                            // 调用中介者对象的 RunFromRing 方法逃离斗牛场
                            mediator.RunFromRing();
                        else
                            // 调用中介者对象的 ContinueFighting 方法继续战斗
                            mediator.ContinueFighting();
                        break;

                    // 比赛结束事件
                    case Events.MatchCompleted matchCompleted:
                        // 调用 View 类的 ShowFinalResult 方法显示比赛最终结果
                        View.ShowFinalResult(matchCompleted.Result, matchCompleted.ExtremeBravery, matchCompleted.Reward);
                        break;
                }
            }
        }
    }
}

```