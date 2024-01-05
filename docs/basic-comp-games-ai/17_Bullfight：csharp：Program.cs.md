# `17_Bullfight\csharp\Program.cs`

```
// 创建一个 Mediator 实例
var mediator = new Mediator();

// 遍历 BullFight.Begin(mediator) 返回的事件
foreach (var evt in BullFight.Begin(mediator))
{
    // 根据不同的事件类型进行不同的处理
    switch (evt)
    {
        // 如果是比赛开始事件
        case Events.MatchStarted matchStarted:
            // 调用 View 类的 ShowStartingConditions 方法显示比赛开始的条件
            View.ShowStartingConditions(matchStarted);
            break;

        // 如果是公牛冲锋事件
        case Events.BullCharging bullCharging:
            // 调用 View 类的 ShowStartOfPass 方法显示通行开始
            View.ShowStartOfPass(bullCharging.PassNumber);
            // 调用 Controller 类的 GetPlayerIntention 方法获取玩家的意图和风险水平
            var (action, riskLevel) = Controller.GetPlayerIntention(bullCharging.PassNumber);
                        switch (action)  # 开始一个 switch 语句，根据 action 的值进行不同的操作
                        {
                            case Action.Dodge:  # 如果 action 的值是 Dodge
                                mediator.Dodge(riskLevel);  # 调用 mediator 的 Dodge 方法，传入 riskLevel 参数
                                break;  # 结束当前 case
                            case Action.Kill:  # 如果 action 的值是 Kill
                                mediator.Kill(riskLevel);  # 调用 mediator 的 Kill 方法，传入 riskLevel 参数
                                break;  # 结束当前 case
                            case Action.Panic:  # 如果 action 的值是 Panic
                                mediator.Panic();  # 调用 mediator 的 Panic 方法
                                break;  # 结束当前 case
                        }
                        break;  # 结束 switch 语句

                    case Events.PlayerGored playerGored:  # 如果事件是 PlayerGored
                        View.ShowPlayerGored(playerGored.Panicked, playerGored.FirstGoring);  # 调用 View 的 ShowPlayerGored 方法，传入 playerGored.Panicked 和 playerGored.FirstGoring 参数
                        break;  # 结束当前 case

                    case Events.PlayerSurvived:  # 如果事件是 PlayerSurvived
                        View.ShowPlayerSurvives();  # 调用 View 的 ShowPlayerSurvives 方法
                        # 如果玩家从比赛中逃跑
                        if (Controller.GetPlayerRunsFromRing())
                            # 调用中介者的逃跑方法
                            mediator.RunFromRing();
                        # 否则
                        else
                            # 调用中介者的继续战斗方法
                            mediator.ContinueFighting();
                        # 结束当前的事件处理
                        break;

                    # 如果事件是比赛完成事件
                    case Events.MatchCompleted matchCompleted:
                        # 显示最终结果，包括比赛结果、极端勇气和奖励
                        View.ShowFinalResult(matchCompleted.Result, matchCompleted.ExtremeBravery, matchCompleted.Reward);
                        # 结束当前的事件处理
                        break;
                }
            }
        }
    }
}
```