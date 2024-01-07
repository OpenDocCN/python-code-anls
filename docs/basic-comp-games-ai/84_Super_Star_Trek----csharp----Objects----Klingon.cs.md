# `basic-computer-games\84_Super_Star_Trek\csharp\Objects\Klingon.cs`

```

// 引入所需的命名空间
using Games.Common.Randomness;
using SuperStarTrek.Commands;
using SuperStarTrek.Space;

// 定义 Klingon 类，表示星际迷航游戏中的克林贡飞船
namespace SuperStarTrek.Objects
{
    // Klingon 类
    internal class Klingon
    {
        // 私有字段，用于生成随机数
        private readonly IRandom _random;

        // Klingon 类的构造函数，初始化位置和随机数生成器
        internal Klingon(Coordinates sector, IRandom random)
        {
            Sector = sector;
            _random = random;
            // 初始化能量值
            Energy = _random.NextFloat(100, 300);
        }

        // 克林贡飞船的能量属性
        internal float Energy { get; private set; }

        // 克林贡飞船所在的位置属性
        internal Coordinates Sector { get; private set; }

        // 重写 ToString 方法，返回克林贡飞船的字符串表示形式
        public override string ToString() => "+K+";

        // 克林贡飞船开火的方法，返回命令执行结果
        internal CommandResult FireOn(Enterprise enterprise)
        {
            // 计算攻击强度、与企业飞船的距离等参数
            var attackStrength = _random.NextFloat();
            var distanceToEnterprise = Sector.GetDistanceTo(enterprise.SectorCoordinates);
            var hitStrength = (int)(Energy * (2 + attackStrength) / distanceToEnterprise);
            Energy /= 3 + attackStrength;

            // 调用企业飞船的受击方法，返回命令执行结果
            return enterprise.TakeHit(Sector, hitStrength);
        }

        // 克林贡飞船受到攻击的方法，返回是否受到攻击
        internal bool TakeHit(int hitStrength)
        {
            if (hitStrength < 0.15 * Energy) { return false; }

            Energy -= hitStrength;
            return true;
        }

        // 克林贡飞船移动到指定位置的方法
        internal void MoveTo(Coordinates newSector) => Sector = newSector;
    }
}

```