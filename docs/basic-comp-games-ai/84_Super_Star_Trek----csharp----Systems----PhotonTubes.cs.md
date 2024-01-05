# `84_Super_Star_Trek\csharp\Systems\PhotonTubes.cs`

```
using Games.Common.IO; // 导入 Games.Common.IO 命名空间
using SuperStarTrek.Commands; // 导入 SuperStarTrek.Commands 命名空间
using SuperStarTrek.Objects; // 导入 SuperStarTrek.Objects 命名空间
using SuperStarTrek.Space; // 导入 SuperStarTrek.Space 命名空间

namespace SuperStarTrek.Systems; // 定义 SuperStarTrek.Systems 命名空间

internal class PhotonTubes : Subsystem // 定义名为 PhotonTubes 的内部类，继承自 Subsystem 类
{
    private readonly int _tubeCount; // 声明私有只读整型变量 _tubeCount
    private readonly Enterprise _enterprise; // 声明私有只读 Enterprise 类型变量 _enterprise
    private readonly IReadWrite _io; // 声明私有只读 IReadWrite 类型变量 _io

    internal PhotonTubes(int tubeCount, Enterprise enterprise, IReadWrite io) // 定义名为 PhotonTubes 的构造函数，接受 tubeCount、enterprise 和 io 三个参数
        : base("Photon Tubes", Command.TOR, io) // 调用基类 Subsystem 的构造函数，传入 "Photon Tubes"、Command.TOR 和 io 三个参数
    {
        TorpedoCount = _tubeCount = tubeCount; // 将 tubeCount 的值赋给 TorpedoCount 和 _tubeCount
        _enterprise = enterprise; // 将 enterprise 的值赋给 _enterprise
        _io = io; // 将 io 的值赋给 _io
    }
    internal int TorpedoCount { get; private set; }  // 定义了一个内部的整型属性TorpedoCount，可以被外部访问，但只能在类的内部进行设置

    protected override bool CanExecuteCommand() => HasTorpedoes() && IsOperational("{name} are not operational");  // 重写了父类的CanExecuteCommand方法，判断是否可以执行命令，条件是拥有光子鱼雷并且处于操作状态

    private bool HasTorpedoes()  // 定义了一个私有方法HasTorpedoes，用于判断是否还有光子鱼雷
    {
        if (TorpedoCount > 0) { return true; }  // 如果拥有光子鱼雷数量大于0，则返回true

        _io.WriteLine("All photon torpedoes expended");  // 如果没有光子鱼雷了，则输出信息“所有光子鱼雷已用完”
        return false;  // 返回false
    }

    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)  // 重写了父类的ExecuteCommandCore方法，用于执行命令
    {
        if (!_io.TryReadCourse("Photon torpedo course", "Ensign Chekov", out var course))  // 如果无法读取“光子鱼雷航线”，则返回CommandResult.Ok
        {
            return CommandResult.Ok;
        }
        TorpedoCount -= 1;  # 减少鱼雷数量

        var isHit = false;  # 初始化是否击中的标志为假
        _io.WriteLine("Torpedo track:");  # 输出鱼雷轨迹信息
        foreach (var sector in course.GetSectorsFrom(_enterprise.SectorCoordinates))  # 遍历从企业当前坐标开始的扇区
        {
            _io.WriteLine($"                {sector}");  # 输出每个扇区的信息

            if (quadrant.TorpedoCollisionAt(sector, out var message, out var gameOver))  # 检查是否在当前扇区发生了鱼雷碰撞
            {
                _io.WriteLine(message);  # 输出碰撞信息
                isHit = true;  # 设置击中标志为真
                if (gameOver) { return CommandResult.GameOver; }  # 如果游戏结束，返回游戏结束结果
                break;  # 结束循环
            }
        }

        if (!isHit) { _io.WriteLine("Torpedo missed!"); }  # 如果没有击中，输出鱼雷未命中信息

        return quadrant.KlingonsFireOnEnterprise();  # 返回克林贡人对企业的攻击结果
    }  # 结束 ReplenishTorpedoes 方法

    internal void ReplenishTorpedoes() => TorpedoCount = _tubeCount;  # 创建 ReplenishTorpedoes 方法，用于重新装填鱼雷

```