# `d:/src/tocomm/basic-computer-games\28_Combat\csharp\Ceasefire.cs`

```
// 声明一个命名空间为 Game
namespace Game
{
    /// <summary>
    /// 表示在达成停火后游戏的状态。
    /// </summary>
    public sealed class Ceasefire : WarState
    {
        /// <summary>
        /// 获取一个指示玩家是否取得了绝对胜利的标志。
        /// </summary>
        public override bool IsAbsoluteVictory { get; }

        /// <summary>
        /// 获取战争的最终结果。
        /// </summary>
        public override WarResult? FinalOutcome
        {
            get
            {
                // 如果是绝对胜利或者玩家部队总数大于电脑部队总数的3/2，则返回玩家胜利
                if (IsAbsoluteVictory || PlayerForces.TotalTroops > 3 / 2 * ComputerForces.TotalTroops)
                    return WarResult.PlayerVictory;
                else
                // 如果玩家部队总数小于电脑部队总数的2/3，则返回电脑胜利
                if (PlayerForces.TotalTroops < 2 / 3 * ComputerForces.TotalTroops)
                    return WarResult.ComputerVictory;
                else
                // 否则返回和平条约
                    return WarResult.PeaceTreaty;
            }
        }

        /// <summary>
        /// Initializes a new instance of the Ceasefire class.
        /// </summary>
        /// <param name="computerForces">
        /// The computer's forces.
        /// </param>
        /// <param name="playerForces">
        /// The player's forces.
        /// </param>
/// <param name="absoluteVictory">
/// 表示玩家是否取得绝对胜利（在不摧毁对方军事力量的情况下击败电脑）。
/// </param>
public Ceasefire(ArmedForces computerForces, ArmedForces playerForces, bool absoluteVictory = false)
    : base(computerForces, playerForces)
{
    IsAbsoluteVictory = absoluteVictory;
}

protected override (WarState nextState, string message) AttackWithArmy(int attackSize) =>
    throw new InvalidOperationException("THE WAR IS OVER");

protected override (WarState nextState, string message) AttackWithNavy(int attackSize) =>
    throw new InvalidOperationException("THE WAR IS OVER");

protected override (WarState nextState, string message) AttackWithAirForce(int attackSize) =>
    throw new InvalidOperationException("THE WAR IS OVER");
}
```

需要注释的代码解释了Ceasefire类的构造函数和三个攻击方法的重写。构造函数接受电脑和玩家的军事力量，以及一个布尔值参数表示是否取得绝对胜利。三个攻击方法都抛出了一个InvalidOperationException异常，表示战争已经结束。
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```