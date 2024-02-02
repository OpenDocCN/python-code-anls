# `basic-computer-games\84_Super_Star_Trek\csharp\Objects\Klingon.cs`

```py
// 引入 Games.Common.Randomness 命名空间
using Games.Common.Randomness;
// 引入 SuperStarTrek.Commands 命名空间
using SuperStarTrek.Commands;
// 引入 SuperStarTrek.Space 命名空间
using SuperStarTrek.Space;

// 在 SuperStarTrek.Objects 命名空间下定义 Klingon 类
internal class Klingon
{
    // 声明私有字段 _random，用于生成随机数
    private readonly IRandom _random;

    // Klingon 类的构造函数，接受 sector 和 random 两个参数
    internal Klingon(Coordinates sector, IRandom random)
    {
        // 初始化 Sector 属性
        Sector = sector;
        // 初始化 _random 字段
        _random = random;
        // 为 Energy 属性生成随机初始值
        Energy = _random.NextFloat(100, 300);
    }

    // 声明 Energy 属性，只能在类内部进行设置
    internal float Energy { get; private set; }

    // 声明 Sector 属性，只能在类内部进行设置
    internal Coordinates Sector { get; private set; }

    // 重写 ToString 方法，返回 "+K+"
    public override string ToString() => "+K+";

    // Klingon 类的 FireOn 方法，接受 enterprise 参数，返回 CommandResult 对象
    internal CommandResult FireOn(Enterprise enterprise)
    {
        // 生成攻击强度
        var attackStrength = _random.NextFloat();
        // 计算到 Enterprise 的距离
        var distanceToEnterprise = Sector.GetDistanceTo(enterprise.SectorCoordinates);
        // 计算命中强度
        var hitStrength = (int)(Energy * (2 + attackStrength) / distanceToEnterprise);
        // 能量减少
        Energy /= 3 + attackStrength;

        // 返回 Enterprise 的受击结果
        return enterprise.TakeHit(Sector, hitStrength);
    }

    // Klingon 类的 TakeHit 方法，接受 hitStrength 参数，返回布尔值
    internal bool TakeHit(int hitStrength)
    {
        // 如果命中强度小于能量的 15%，返回 false
        if (hitStrength < 0.15 * Energy) { return false; }

        // 能量减少
        Energy -= hitStrength;
        return true;
    }

    // Klingon 类的 MoveTo 方法，接受 newSector 参数，用于移动到新的坐标
    internal void MoveTo(Coordinates newSector) => Sector = newSector;
}
```