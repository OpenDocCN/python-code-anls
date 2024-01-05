# `d:/src/tocomm/basic-computer-games\28_Combat\csharp\ArmedForces.cs`

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
        /// <summary>
        /// 获取空军中男性和女性的人数。
        /// </summary>
        public int AirForce { get; init; }  // 定义一个属性，用于获取空军中的人数

        /// <summary>
        /// 获取武装部队中的总人数。
        /// </summary>
        public int TotalTroops => Army + Navy + AirForce;  // 定义一个只读属性，用于获取武装部队的总人数，通过计算陆军、海军和空军的人数之和得到

        /// <summary>
        /// 获取给定军种中男性和女性的人数。
        /// </summary>
        public int this[MilitaryBranch branch] =>  // 定义一个索引器，根据传入的军种返回相应的人数
            branch switch
            {
                MilitaryBranch.Army     => Army,  // 如果是陆军，返回陆军的人数
                MilitaryBranch.Navy     => Navy,  // 如果是海军，返回海军的人数
                MilitaryBranch.AirForce => AirForce,  // 如果是空军，返回空军的人数
                _                       => throw new ArgumentException("INVALID BRANCH")  // 如果是其他军种，抛出参数异常
            };
    }
```

这部分代码是一个缩进错误，应该删除这两行代码。
```