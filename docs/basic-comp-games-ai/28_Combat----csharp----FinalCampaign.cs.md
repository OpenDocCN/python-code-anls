# `28_Combat\csharp\FinalCampaign.cs`

```
// 命名空间 Game，表示游戏的命名空间
namespace Game
{
    /// <summary>
    /// 表示战争最终战役期间游戏状态的类
    /// </summary>
    public sealed class FinalCampaign : WarState
    {
        /// <summary>
        /// 初始化 FinalCampaign 类的新实例
        /// </summary>
        /// <param name="computerForces">
        /// 计算机的力量
        /// </param>
        /// <param name="playerForces">
        /// 玩家的力量
        /// </param>
        public FinalCampaign(ArmedForces computerForces, ArmedForces playerForces)
            : base(computerForces, playerForces)
        {
        }
        protected override (WarState nextState, string message) AttackWithArmy(int attackSize)
        {
            // 检查攻击规模是否小于对方军队规模的一半
            if (attackSize < ComputerForces.Army / 2)
            {
                // 如果是，则返回停火状态和更新后的玩家军队规模
                return
                (
                    new Ceasefire(
                        ComputerForces,
                        PlayerForces with
                        {
                            Army = PlayerForces.Army - attackSize
                        }),
                    "I WIPED OUT YOUR ATTACK!"
                );
            }
            else
            {
                // 如果不是，则继续战斗
                return
                (
                    new Ceasefire(
                        ComputerForces with
                        {
                            Army = 0
                        },
                        PlayerForces),
                    "YOU DESTROYED MY ARMY!"
                );
            }
        }
```

注释：这部分代码是一个类的方法，用于处理攻击海军的情况。如果攻击规模小于计算机部队海军规模的一半，返回一个停火对象和一条消息。

```
        protected override (WarState nextState, string message) AttackWithNavy(int attackSize)
        {
            if (attackSize < ComputerForces.Navy / 2)
            {
                return
                (
                    new Ceasefire(
                        ComputerForces,
                        PlayerForces with
```

注释：这部分代码是一个类的方法，用于处理攻击海军的情况。如果攻击规模小于计算机部队海军规模的一半，返回一个停火对象和一条消息。
                        {
                            Army = PlayerForces.Army / 4,  # 计算玩家军队的四分之一
                            Navy = PlayerForces.Navy / 2   # 计算玩家海军的二分之一
                        }),
                    "I SUNK TWO OF YOUR BATTLESHIPS, AND MY AIR FORCE\n" +  # 返回一条消息，描述了玩家的行动
                    "WIPED OUT YOUR UNGAURDED CAPITOL."
                );
            }
            else
            {
                return
                (
                    new Ceasefire(  # 创建一个停火对象
                        ComputerForces with  # 使用计算机军队的属性
                        {
                            AirForce = 2 * ComputerForces.AirForce / 3,  # 计算计算机空军的三分之二
                            Navy     = ComputerForces.Navy / 2   # 计算计算机海军的二分之一
                        },
                        PlayerForces),  # 使用玩家军队的属性
                    "YOUR NAVY SHOT DOWN THREE OF MY XIII PLANES,\n" +  # 返回一条消息，描述了计算机的行动
                    "AND SUNK THREE BATTLESHIPS."
                );
            }
        }
```

注释：
- 这是一个多行注释，用于解释下面的代码逻辑和可能存在的 bug。

```
        protected override (WarState nextState, string message) AttackWithAirForce(int attackSize)
```

注释：
- 这是一个方法的定义，用于攻击敌方空军的逻辑。

```
        // BUG? Usually, larger attacks lead to better outcomes.
        //  It seems odd that the logic is suddenly reversed here,
        //  but this could be intentional.
```

注释：
- 这是一个单行注释，用于标记可能存在的 bug，并对代码逻辑进行了解释。

```
        if (attackSize > ComputerForces.AirForce / 2)
```

注释：
- 这是一个条件语句，判断攻击规模是否大于敌方空军规模的一半。

```
        return
        (
            new Ceasefire(
                ComputerForces,
                PlayerForces with
                {
                    Army     = PlayerForces.Army  / 3,
                    Navy     = PlayerForces.Navy / 3,
```

注释：
- 这是一个返回语句，返回了一个新的停火状态和相关信息。
# 如果玩家的空军力量大于计算机的空军力量的三分之一，则执行以下代码块
if (PlayerForces.AirForce > ComputerForces.AirForce / 3)
{
    # 返回一个停火对象，其中计算机的空军力量和玩家的空军力量被减少了三分之一
    return
    (
        new Ceasefire(
            ComputerForces,
            PlayerForces,
            absoluteVictory: false,
            forcesReduced: new Forces
            {
                Navy = PlayerForces.Navy / 3,
                AirForce = PlayerForces.AirForce / 3
            }),
        "MY NAVY AND AIR FORCE IN A COMBINED ATTACK LEFT\n" +
        "YOUR COUNTRY IN SHAMBLES."
    );
}
# 如果玩家的空军力量不大于计算机的空军力量的三分之一，则执行以下代码块
else
{
    # 返回一个停火对象，其中计算机的空军力量和玩家的空军力量被减少了三分之一，并且绝对胜利标志为真
    return
    (
        new Ceasefire(
            ComputerForces,
            PlayerForces,
            absoluteVictory: true),
        "ONE OF YOUR PLANES CRASHED INTO MY HOUSE. I AM DEAD.\n" +
        "MY COUNTRY FELL APART."
    );
}
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```