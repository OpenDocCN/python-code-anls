# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\csharp\Objects\Klingon.cs`

```
using Games.Common.Randomness;  # 导入 Games.Common.Randomness 模块
using SuperStarTrek.Commands;  # 导入 SuperStarTrek.Commands 模块
using SuperStarTrek.Space;  # 导入 SuperStarTrek.Space 模块

namespace SuperStarTrek.Objects;  # 定义 SuperStarTrek.Objects 命名空间

internal class Klingon  # 定义 Klingon 类
{
    private readonly IRandom _random;  # 声明私有只读属性 _random，类型为 IRandom 接口

    internal Klingon(Coordinates sector, IRandom random)  # 定义 Klingon 类的构造函数，接受 Coordinates 类型的 sector 和 IRandom 类型的 random 参数
    {
        Sector = sector;  # 将参数 sector 赋值给 Sector 属性
        _random = random;  # 将参数 random 赋值给 _random 属性
        Energy = _random.NextFloat(100, 300);  # 使用 _random 生成一个介于 100 和 300 之间的随机浮点数，并赋值给 Energy 属性
    }

    internal float Energy { get; private set; }  # 定义只读属性 Energy，类型为 float

    internal Coordinates Sector { get; private set; }  # 定义只读属性 Sector，类型为 Coordinates
    public override string ToString() => "+K+";  // 重写 ToString 方法，返回字符串 "+K+"

    internal CommandResult FireOn(Enterprise enterprise)
    {
        var attackStrength = _random.NextFloat();  // 生成攻击强度的随机数
        var distanceToEnterprise = Sector.GetDistanceTo(enterprise.SectorCoordinates);  // 获取到企业的距离
        var hitStrength = (int)(Energy * (2 + attackStrength) / distanceToEnterprise);  // 计算攻击力
        Energy /= 3 + attackStrength;  // 减少能量

        return enterprise.TakeHit(Sector, hitStrength);  // 对企业进行攻击，并返回攻击结果
    }

    internal bool TakeHit(int hitStrength)
    {
        if (hitStrength < 0.15 * Energy) { return false; }  // 如果攻击力小于能量的 15%，则返回 false

        Energy -= hitStrength;  // 减少能量
        return true;  // 返回 true
    }
# 将当前对象的Sector属性设置为新的坐标
def MoveTo(newSector):
    self.Sector = newSector
```