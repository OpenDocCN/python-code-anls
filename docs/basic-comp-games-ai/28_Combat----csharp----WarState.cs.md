# `basic-computer-games\28_Combat\csharp\WarState.cs`

```

// 命名空间 Game，表示游戏相关的类
namespace Game
{
    /// <summary>
    /// 表示战争的当前状态
    /// </summary>
    public abstract class WarState
    {
        /// <summary>
        /// 获取计算机的武装力量
        /// </summary>
        public ArmedForces ComputerForces { get; }

        /// <summary>
        /// 获取玩家的武装力量
        /// </summary>
        public ArmedForces PlayerForces { get; }

        /// <summary>
        /// 获取一个标志，指示此状态是否代表玩家的绝对胜利
        /// </summary>
        public virtual bool IsAbsoluteVictory => false;

        /// <summary>
        /// 获取战争的最终结果
        /// </summary>
        /// <remarks>
        /// 如果战争仍在进行中，此属性将为 null
        /// </remarks>
        public virtual WarResult? FinalOutcome => null;

        /// <summary>
        /// 初始化状态类的新实例
        /// </summary>
        /// <param name="computerForces">
        /// 计算机的力量
        /// </param>
        /// <param name="playerForces">
        /// 玩家的力量
        /// </param>
        public WarState(ArmedForces computerForces, ArmedForces playerForces) =>
            (ComputerForces, PlayerForces) = (computerForces, playerForces);

        /// <summary>
        /// 发动攻击
        /// </summary>
        /// <param name="branch">
        /// 用于攻击的军事部门
        /// </param>
        /// <param name="attackSize">
        /// 用于攻击的人数
        /// </param>
        /// <returns>
        /// 由攻击产生的新游戏状态和描述结果的消息
        /// </returns>
        public (WarState nextState, string message) LaunchAttack(MilitaryBranch branch, int attackSize) =>
            branch switch
            {
                MilitaryBranch.Army     => AttackWithArmy(attackSize),
                MilitaryBranch.Navy     => AttackWithNavy(attackSize),
                MilitaryBranch.AirForce => AttackWithAirForce(attackSize),
                _               => throw new ArgumentException("INVALID BRANCH")
            };

        /// <summary>
        /// 使用玩家的陆军进行攻击
        /// </summary>
        /// <param name="attackSize">
        /// 攻击中使用的人数
        /// </param>
        /// <returns>
        /// 新的游戏状态和描述结果的消息
        /// </returns>
        protected abstract (WarState nextState, string message) AttackWithArmy(int attackSize);

        /// <summary>
        /// 使用玩家的海军进行攻击
        /// </summary>
        /// <param name="attackSize">
        /// 攻击中使用的人数
        /// </param>
        /// <returns>
        /// 新的游戏状态和描述结果的消息
        /// </returns>
        protected abstract (WarState nextState, string message) AttackWithNavy(int attackSize);

        /// <summary>
        /// 使用玩家的空军进行攻击
        /// </summary>
        /// <param name="attackSize">
        /// 攻击中使用的人数
        /// </param>
        /// <returns>
        /// 新的游戏状态和描述结果的消息
        /// </returns>
        protected abstract (WarState nextState, string message) AttackWithAirForce(int attackSize);
    }
}

```