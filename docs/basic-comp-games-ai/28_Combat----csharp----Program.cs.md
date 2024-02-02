# `basic-computer-games\28_Combat\csharp\Program.cs`

```py
namespace Game
{
    class Program
    {
        static void Main()
        {
            // 显示游戏横幅
            View.ShowBanner();
            // 显示游戏说明
            View.ShowInstructions();

            // 创建计算机武装力量对象
            var computerForces = new ArmedForces { Army = 30000, Navy = 20000, AirForce = 22000 };
            // 获取玩家初始武装力量
            var playerForces   = Controller.GetInitialForces(computerForces);

            // 创建初始战役状态
            var state = (WarState) new InitialCampaign(computerForces, playerForces);
            // 标记是否是第一回合
            var isFirstTurn = true;

            // 当战争状态没有最终结果时循环
            while (!state.FinalOutcome.HasValue)
            {
                // 获取攻击分支
                var branch = Controller.GetAttackBranch(state, isFirstTurn);
                // 获取攻击规模
                var attackSize = Controller.GetAttackSize(state.PlayerForces[branch]);

                // 发动攻击，获取下一个状态和消息
                var (nextState, message) = state.LaunchAttack(branch, attackSize);
                // 显示消息
                View.ShowMessage(message);

                // 更新状态和回合标记
                state = nextState;
                isFirstTurn = false;
            }

            // 显示最终结果
            View.ShowResult(state);
        }
    }
}
```