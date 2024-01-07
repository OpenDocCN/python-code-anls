# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\PhotonTubes.cs`

```

// 引入所需的命名空间
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Objects;
using SuperStarTrek.Space;

// 定义 PhotonTubes 类，继承自 Subsystem 类
internal class PhotonTubes : Subsystem
{
    // 声明私有变量
    private readonly int _tubeCount;
    private readonly Enterprise _enterprise;
    private readonly IReadWrite _io;

    // 构造函数，初始化 PhotonTubes 对象
    internal PhotonTubes(int tubeCount, Enterprise enterprise, IReadWrite io)
        : base("Photon Tubes", Command.TOR, io)
    {
        // 初始化 TorpedoCount 和其他私有变量
        TorpedoCount = _tubeCount = tubeCount;
        _enterprise = enterprise;
        _io = io;
    }

    // 声明 TorpedoCount 属性
    internal int TorpedoCount { get; private set; }

    // 重写父类的 CanExecuteCommand 方法
    protected override bool CanExecuteCommand() => HasTorpedoes() && IsOperational("{name} are not operational");

    // 声明 HasTorpedoes 方法
    private bool HasTorpedoes()
    {
        // 检查是否还有鱼雷
        if (TorpedoCount > 0) { return true; }

        // 输出信息并返回 false
        _io.WriteLine("All photon torpedoes expended");
        return false;
    }

    // 重写父类的 ExecuteCommandCore 方法
    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
    {
        // 尝试读取鱼雷的航向
        if (!_io.TryReadCourse("Photon torpedo course", "Ensign Chekov", out var course))
        {
            return CommandResult.Ok;
        }

        // 减少鱼雷数量
        TorpedoCount -= 1;

        // 初始化变量
        var isHit = false;
        _io.WriteLine("Torpedo track:");

        // 遍历航向上的各个区块
        foreach (var sector in course.GetSectorsFrom(_enterprise.SectorCoordinates))
        {
            _io.WriteLine($"                {sector}");

            // 检查是否击中目标
            if (quadrant.TorpedoCollisionAt(sector, out var message, out var gameOver))
            {
                _io.WriteLine(message);
                isHit = true;
                if (gameOver) { return CommandResult.GameOver; }
                break;
            }
        }

        // 输出结果
        if (!isHit) { _io.WriteLine("Torpedo missed!"); }

        // 返回结果
        return quadrant.KlingonsFireOnEnterprise();
    }

    // 声明 ReplenishTorpedoes 方法，用于补充鱼雷数量
    internal void ReplenishTorpedoes() => TorpedoCount = _tubeCount;
}

```