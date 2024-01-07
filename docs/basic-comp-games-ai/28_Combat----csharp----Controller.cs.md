# `basic-computer-games\28_Combat\csharp\Controller.cs`

```

// 命名空间 Game 包含了与用户交互相关的函数
namespace Game
{
    /// <summary>
    /// 包含了与用户交互相关的函数
    /// </summary>
    public class Controller
    {
        /// <summary>
        /// 获取玩家的初始武装力量分配
        /// </summary>
        /// <param name="computerForces">
        /// 计算机的初始武装力量
        /// </param>
        public static ArmedForces GetInitialForces(ArmedForces computerForces)
        {
            var playerForces = default(ArmedForces);

            // BUG: 此循环允许玩家为某些分支分配负值，导致奇怪的结果
            do
            {
                View.ShowDistributeForces();

                View.PromptArmySize(computerForces.Army);
                var army = InputInteger();

                View.PromptNavySize(computerForces.Navy);
                var navy = InputInteger();

                View.PromptAirForceSize(computerForces.AirForce);
                var airForce = InputInteger();

                playerForces = new ArmedForces
                {
                    Army     = army,
                    Navy     = navy,
                    AirForce = airForce
                };
            }
            while (playerForces.TotalTroops > computerForces.TotalTroops);

            return playerForces;
        }

        /// <summary>
        /// 获取用户下一次攻击的军事分支
        /// </summary>
        public static MilitaryBranch GetAttackBranch(WarState state, bool isFirstTurn)
        {
            if (isFirstTurn)
                View.PromptFirstAttackBranch();
            else
                View.PromptNextAttackBranch(state.ComputerForces, state.PlayerForces);

            // 如果用户在原始游戏中输入了无效的分支编号，代码会继续执行到陆军分支。我们将保留这种行为。
            return Console.ReadLine() switch
            {
                "2" => MilitaryBranch.Navy,
                "3" => MilitaryBranch.AirForce,
                _   => MilitaryBranch.Army
            };
        }

        /// <summary>
        /// 为给定的武装力量分支获取有效的攻击规模
        /// </summary>
        /// <param name="troopsAvailable">
        /// 可用的部队数量
        /// </param>
        public static int GetAttackSize(int troopsAvailable)
        {
            var attackSize = 0;

            do
            {
                View.PromptAttackSize();
                attackSize = InputInteger();
            }
            while (attackSize < 0 || attackSize > troopsAvailable);

            return attackSize;
        }

        /// <summary>
        /// 从用户获取整数值
        /// </summary>
        public static int InputInteger()
        {
            var value = default(int);

            while (!Int32.TryParse(Console.ReadLine(), out value))
                View.PromptValidInteger();

            return value;
        }
    }
}

```