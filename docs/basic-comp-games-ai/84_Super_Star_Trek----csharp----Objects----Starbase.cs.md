# `basic-computer-games\84_Super_Star_Trek\csharp\Objects\Starbase.cs`

```

// 引入所需的命名空间
using Games.Common.IO;
using Games.Common.Randomness;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;

// 定义星舰基地类
namespace SuperStarTrek.Objects
{
    internal class Starbase
    {
        // 读写接口和修复延迟属性
        private readonly IReadWrite _io;
        private readonly float _repairDelay;

        // 构造函数，初始化星舰基地的位置、修复延迟和读写接口
        internal Starbase(Coordinates sector, IRandom random, IReadWrite io)
        {
            Sector = sector;
            _repairDelay = random.NextFloat(0.5f);
            _io = io;
        }

        // 星舰基地所在的坐标属性
        internal Coordinates Sector { get; }

        // 重写 ToString 方法
        public override string ToString() => ">!<";

        // 尝试修复星舰系统，返回修复时间
        internal bool TryRepair(Enterprise enterprise, out float repairTime)
        {
            // 计算修复时间
            repairTime = enterprise.DamagedSystemCount * 0.1f + _repairDelay;
            if (repairTime >= 1) { repairTime = 0.9f; }

            // 输出修复时间估计
            _io.Write(Strings.RepairEstimate, repairTime);
            // 获取用户输入，决定是否修复
            if (_io.GetYesNo(Strings.RepairPrompt, IReadWriteExtensions.YesNoMode.TrueOnY))
            {
                // 逐个修复星舰系统
                foreach (var system in enterprise.Systems)
                {
                    system.Repair();
                }
                return true;
            }

            // 修复时间置零，返回修复失败
            repairTime = 0;
            return false;
        }

        // 保护星舰企业
        internal void ProtectEnterprise() => _io.WriteLine(Strings.Protected);
    }
}

```