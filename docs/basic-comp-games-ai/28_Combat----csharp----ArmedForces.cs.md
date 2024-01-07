# `basic-computer-games\28_Combat\csharp\ArmedForces.cs`

```

// 引入 System 命名空间
using System;

// 命名空间 Game
namespace Game
{
    /// <summary>
    /// 表示一个国家的武装力量。
    /// </summary>
    public record ArmedForces
    {
        /// <summary>
        /// 获取陆军的男女人数。
        /// </summary>
        public int Army { get; init; }

        /// <summary>
        /// 获取海军的男女人数。
        /// </summary>
        public int Navy { get; init; }

        /// <summary>
        /// 获取空军的男女人数。
        /// </summary>
        public int AirForce { get; init; }

        /// <summary>
        /// 获取武装力量的总人数。
        /// </summary>
        public int TotalTroops => Army + Navy + AirForce;

        /// <summary>
        /// 获取给定军种的男女人数。
        /// </summary>
        public int this[MilitaryBranch branch] =>
            branch switch
            {
                MilitaryBranch.Army     => Army,
                MilitaryBranch.Navy     => Navy,
                MilitaryBranch.AirForce => AirForce,
                _                       => throw new ArgumentException("INVALID BRANCH")
            };
    }
}

```