# `d:/src/tocomm/basic-computer-games\28_Combat\csharp\Program.cs`

```
            // 显示游戏横幅
            View.ShowBanner();
            // 显示游戏说明
            View.ShowInstructions();

            // 创建计算机武装力量对象，包括陆军、海军和空军
            var computerForces = new ArmedForces { Army = 30000, Navy = 20000, AirForce = 22000 };
            // 获取玩家初始武装力量
            var playerForces   = Controller.GetInitialForces(computerForces);

            // 创建初始战役状态对象
            var state = (WarState) new InitialCampaign(computerForces, playerForces);
            // 标记是否是第一回合
            var isFirstTurn = true;

            // 当战争状态的最终结果未确定时，执行循环
            while (!state.FinalOutcome.HasValue)
            {
                // 获取攻击分支
                var branch = Controller.GetAttackBranch(state, isFirstTurn);
                // 获取攻击规模
                var attackSize = Controller.GetAttackSize(state.PlayerForces[branch]);
                var (nextState, message) = state.LaunchAttack(branch, attackSize);
                # 调用状态对象的LaunchAttack方法，传入参数branch和attackSize，返回下一个状态和消息
                View.ShowMessage(message);
                # 调用视图对象的ShowMessage方法，显示消息

                state = nextState;
                # 将状态更新为下一个状态
                isFirstTurn = false;
                # 将isFirstTurn标记更新为false
            }

            View.ShowResult(state);
            # 调用视图对象的ShowResult方法，显示最终结果
        }
    }
}
```