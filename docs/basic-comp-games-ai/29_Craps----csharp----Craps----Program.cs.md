# `29_Craps\csharp\Craps\Program.cs`

```
            // 创建用户界面对象
            var ui = new UserInterface();
            // 创建 Craps 游戏对象，并传入用户界面对象的引用
            var game = new CrapsGame(ref ui);
            // 初始化赢得的奖金为 0
            int winnings = 0;

            // 显示游戏介绍
            ui.Intro();

            // 进入游戏循环
            do
            {
                // 用户下注
                var bet = ui.PlaceBet();
                // 进行游戏，并获取游戏结果和骰子点数
                var result = game.Play(out int diceRoll);
# 根据游戏结果进行不同的操作
switch (result):
    # 如果是自然赢，将赌注加到总赢利中
    case Result.naturalWin:
        winnings += bet;
        break;

    # 如果是自然输、掷骰子输或点数输，将赌注从总赢利中减去
    case Result.naturalLoss:
    case Result.snakeEyesLoss:
    case Result.pointLoss:
        winnings -= bet;
        break;

    # 如果是点数赢，将两倍赌注加到总赢利中
    case Result.pointWin:
        winnings += (2 * bet);
        break;

    # 包含一个默认情况，以便在枚举值发生变化时提醒我们添加新值的处理代码
    default:
                        Debug.Assert(false); // We should never get here. 
                        // 断言，如果程序执行到这里，表示出现了意料之外的情况
                        break;
                }

                ui.ShowResult(result, diceRoll, bet);
                // 调用 UI 对象的 ShowResult 方法，显示游戏结果、骰子点数和赌注

            } while (ui.PlayAgain(winnings));
            // 使用 UI 对象的 PlayAgain 方法判断是否继续游戏，如果赢得了奖金则继续

            ui.GoodBye(winnings);
            // 调用 UI 对象的 GoodBye 方法，显示结束语并显示总奖金
        }
    }
}
```