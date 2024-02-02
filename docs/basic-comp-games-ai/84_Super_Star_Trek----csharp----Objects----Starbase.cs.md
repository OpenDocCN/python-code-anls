# `basic-computer-games\84_Super_Star_Trek\csharp\Objects\Starbase.cs`

```py
// 引入所需的命名空间
using Games.Common.IO;
using Games.Common.Randomness;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;

// 定义 Starbase 类
namespace SuperStarTrek.Objects
{
    // 声明 Starbase 类
    internal class Starbase
    {
        // 声明私有字段 _io 和 _repairDelay
        private readonly IReadWrite _io;
        private readonly float _repairDelay;

        // Starbase 类的构造函数，接受 sector、random 和 io 三个参数
        internal Starbase(Coordinates sector, IRandom random, IReadWrite io)
        {
            // 初始化 Sector 字段
            Sector = sector;
            // 生成随机的修复延迟时间
            _repairDelay = random.NextFloat(0.5f);
            // 初始化 _io 字段
            _io = io;
        }

        // 声明 Sector 属性
        internal Coordinates Sector { get; }

        // 重写 ToString 方法
        public override string ToString() => ">!<";

        // 尝试修复 Enterprise 的方法，返回修复时间
        internal bool TryRepair(Enterprise enterprise, out float repairTime)
        {
            // 计算修复时间
            repairTime = enterprise.DamagedSystemCount * 0.1f + _repairDelay;
            // 如果修复时间大于等于 1，则修复时间设为 0.9
            if (repairTime >= 1) { repairTime = 0.9f; }

            // 输出修复时间的估计
            _io.Write(Strings.RepairEstimate, repairTime);
            // 获取用户是否同意修复的输入
            if (_io.GetYesNo(Strings.RepairPrompt, IReadWriteExtensions.YesNoMode.TrueOnY))
            {
                // 遍历 Enterprise 的系统列表，逐个修复
                foreach (var system in enterprise.Systems)
                {
                    system.Repair();
                }
                return true;
            }

            // 如果用户不同意修复，则修复时间设为 0，返回 false
            repairTime = 0;
            return false;
        }

        // 保护 Enterprise 的方法，输出保护信息
        internal void ProtectEnterprise() => _io.WriteLine(Strings.Protected);
    }
}
```