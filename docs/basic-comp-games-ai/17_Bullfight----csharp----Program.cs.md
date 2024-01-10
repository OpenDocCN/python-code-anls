# `basic-computer-games\17_Bullfight\csharp\Program.cs`

```
// 命名空间 Game
namespace Game
{
    // 程序入口
    class Program
    {
        // 主函数
        static void Main()
        {
            // 启动游戏控制器
            Controller.StartGame();

            // 创建中介者对象
            var mediator = new Mediator();
            // 遍历比赛开始事件
            foreach (var evt in BullFight.Begin(mediator))
            {
                // 根据不同的事件类型进行处理
                switch (evt)
                {
                    // 比赛开始事件
                    case Events.MatchStarted matchStarted:
                        // 显示比赛开始的条件
                        View.ShowStartingConditions(matchStarted);
                        break;

                    // 公牛冲锋事件
                    case Events.BullCharging bullCharging:
                        // 显示冲锋开始的信息
                        View.ShowStartOfPass(bullCharging.PassNumber);
                        // 获取玩家的意图和风险级别
                        var (action, riskLevel) = Controller.GetPlayerIntention(bullCharging.PassNumber);
                        // 根据玩家的意图进行处理
                        switch (action)
                        {
                            // 躲避
                            case Action.Dodge:
                                mediator.Dodge(riskLevel);
                                break;
                            // 击杀
                            case Action.Kill:
                                mediator.Kill(riskLevel);
                                break;
                            // 恐慌
                            case Action.Panic:
                                mediator.Panic();
                                break;
                        }
                        break;

                    // 玩家被公牛刺伤事件
                    case Events.PlayerGored playerGored:
                        // 显示玩家被刺伤的情况
                        View.ShowPlayerGored(playerGored.Panicked, playerGored.FirstGoring);
                        break;

                    // 玩家幸存事件
                    case Events.PlayerSurvived:
                        // 显示玩家幸存
                        View.ShowPlayerSurvives();
                        // 根据玩家是否逃离斗牛场进行处理
                        if (Controller.GetPlayerRunsFromRing())
                            mediator.RunFromRing();
                        else
                            mediator.ContinueFighting();
                        break;

                    // 比赛结束事件
                    case Events.MatchCompleted matchCompleted:
                        // 显示比赛最终结果
                        View.ShowFinalResult(matchCompleted.Result, matchCompleted.ExtremeBravery, matchCompleted.Reward);
                        break;
                }
            }
        }
    }
    # 结束函数定义的代码块
# 闭合前面的函数定义
```