# `basic-computer-games\28_Combat\csharp\Ceasefire.cs`

```

// 引入 System 命名空间
using System;

// 命名空间 Game
namespace Game
{
    /// <summary>
    /// 表示达成停火后的游戏状态
    /// </summary>
    public sealed class Ceasefire : WarState
    {
        /// <summary>
        /// 获取一个指示玩家是否取得绝对胜利的标志
        /// </summary>
        public override bool IsAbsoluteVictory { get; }

        /// <summary>
        /// 获取战争的最终结果
        /// </summary>
        public override WarResult? FinalOutcome
        {
            get
            {
                if (IsAbsoluteVictory || PlayerForces.TotalTroops > 3 / 2 * ComputerForces.TotalTroops)
                    return WarResult.PlayerVictory;
                else
                if (PlayerForces.TotalTroops < 2 / 3 * ComputerForces.TotalTroops)
                    return WarResult.ComputerVictory;
                else
                    return WarResult.PeaceTreaty;
            }
        }

        /// <summary>
        /// 初始化 Ceasefire 类的新实例
        /// </summary>
        /// <param name="computerForces">
        /// 计算机的军队
        /// </param>
        /// <param name="playerForces">
        /// 玩家的军队
        /// </param>
        /// <param name="absoluteVictory">
        /// 指示玩家是否取得绝对胜利（击败计算机而不摧毁其军事力量）
        /// </param>
        public Ceasefire(ArmedForces computerForces, ArmedForces playerForces, bool absoluteVictory = false)
            : base(computerForces, playerForces)
        {
            IsAbsoluteVictory = absoluteVictory;
        }

        // 重写 AttackWithArmy 方法
        protected override (WarState nextState, string message) AttackWithArmy(int attackSize) =>
            throw new InvalidOperationException("THE WAR IS OVER");

        // 重写 AttackWithNavy 方法
        protected override (WarState nextState, string message) AttackWithNavy(int attackSize) =>
            throw new InvalidOperationException("THE WAR IS OVER");

        // 重写 AttackWithAirForce 方法
        protected override (WarState nextState, string message) AttackWithAirForce(int attackSize) =>
            throw new InvalidOperationException("THE WAR IS OVER");
    }
}

```