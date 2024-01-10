# `basic-computer-games\28_Combat\csharp\ArmedForces.cs`

```
// 命名空间 Game，表示游戏
namespace Game
{
    /// <summary>
    /// 表示一个国家的武装力量
    /// </summary>
    public record ArmedForces
    {
        /// <summary>
        /// 获取陆军中男性和女性的人数
        /// </summary>
        public int Army { get; init; }

        /// <summary>
        /// 获取海军中男性和女性的人数
        /// </summary>
        public int Navy { get; init; }

        /// <summary>
        /// 获取空军中男性和女性的人数
        /// </summary>
        public int AirForce { get; init; }

        /// <summary>
        /// 获取武装力量中的总部队人数
        /// </summary>
        public int TotalTroops => Army + Navy + AirForce;

        /// <summary>
        /// 获取给定军种中男性和女性的人数
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