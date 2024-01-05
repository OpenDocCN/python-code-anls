# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\csharp\Objects\Starbase.cs`

```
using Games.Common.IO;  # 导入 Games.Common.IO 模块
using Games.Common.Randomness;  # 导入 Games.Common.Randomness 模块
using SuperStarTrek.Resources;  # 导入 SuperStarTrek.Resources 模块
using SuperStarTrek.Space;  # 导入 SuperStarTrek.Space 模块

namespace SuperStarTrek.Objects;  # 定义 SuperStarTrek.Objects 命名空间

internal class Starbase  # 定义 Starbase 类
{
    private readonly IReadWrite _io;  # 声明私有成员变量 _io，类型为 IReadWrite 接口
    private readonly float _repairDelay;  # 声明私有成员变量 _repairDelay，类型为 float

    internal Starbase(Coordinates sector, IRandom random, IReadWrite io)  # 定义 Starbase 类的构造函数，接受 Coordinates、IRandom 和 IReadWrite 三个参数
    {
        Sector = sector;  # 初始化 Sector 成员变量
        _repairDelay = random.NextFloat(0.5f);  # 初始化 _repairDelay 成员变量，调用 random 对象的 NextFloat 方法
        _io = io;  # 初始化 _io 成员变量
    }

    internal Coordinates Sector { get; }  # 定义 Sector 属性，只读
    public override string ToString() => ">!<";  # 重写 ToString 方法，返回字符串 ">!<"

    internal bool TryRepair(Enterprise enterprise, out float repairTime)  # 尝试修复方法，接受企业对象和修复时间作为输出参数
    {
        repairTime = enterprise.DamagedSystemCount * 0.1f + _repairDelay;  # 计算修复时间，受损系统数量乘以0.1再加上修复延迟时间
        if (repairTime >= 1) { repairTime = 0.9f; }  # 如果修复时间大于等于1，将修复时间设为0.9

        _io.Write(Strings.RepairEstimate, repairTime);  # 使用 _io 对象写入修复时间的估计值
        if (_io.GetYesNo(Strings.RepairPrompt, IReadWriteExtensions.YesNoMode.TrueOnY))  # 从 _io 对象获取用户输入的是或否，根据用户输入进行下一步操作
        {
            foreach (var system in enterprise.Systems)  # 遍历企业对象的系统列表
            {
                system.Repair();  # 对每个系统进行修复
            }
            return true;  # 返回修复成功
        }

        repairTime = 0;  # 修复时间设为0
        return false;  # 返回修复失败
    }  # 结束 ProtectEnterprise 方法

    internal void ProtectEnterprise() => _io.WriteLine(Strings.Protected);  # 定义 ProtectEnterprise 方法，调用 _io 对象的 WriteLine 方法输出字符串 "Protected"
}
```