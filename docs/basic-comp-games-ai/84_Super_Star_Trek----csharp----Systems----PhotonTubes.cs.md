# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\PhotonTubes.cs`

```py
// 引入所需的命名空间
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Objects;
using SuperStarTrek.Space;

// 定义名为PhotonTubes的子系统类，继承自Subsystem类
namespace SuperStarTrek.Systems
{
    // 声明PhotonTubes类为内部类
    internal class PhotonTubes : Subsystem
    {
        // 声明私有只读整型变量_tubeCount
        private readonly int _tubeCount;
        // 声明私有只读Enterprise对象_enterprise
        private readonly Enterprise _enterprise;
        // 声明私有只读IReadWrite对象_io
        private readonly IReadWrite _io;

        // 定义PhotonTubes类的构造函数，接受tubeCount、enterprise和io作为参数
        internal PhotonTubes(int tubeCount, Enterprise enterprise, IReadWrite io)
            // 调用基类Subsystem的构造函数，传入"Photon Tubes"、Command.TOR和io作为参数
            : base("Photon Tubes", Command.TOR, io)
        {
            // 初始化TorpedoCount和_tubeCount为tubeCount的值
            TorpedoCount = _tubeCount = tubeCount;
            // 初始化_enterprise为enterprise的值
            _enterprise = enterprise;
            // 初始化_io为io的值
            _io = io;
        }

        // 声明TorpedoCount属性，可读写
        internal int TorpedoCount { get; private set; }

        // 重写基类的CanExecuteCommand方法，判断是否能执行命令
        protected override bool CanExecuteCommand() => HasTorpedoes() && IsOperational("{name} are not operational");

        // 声明私有方法HasTorpedoes，判断是否还有鱼雷
        private bool HasTorpedoes()
        {
            // 如果TorpedoCount大于0，返回true
            if (TorpedoCount > 0) { return true; }
            // 否则，输出"All photon torpedoes expended"，返回false
            _io.WriteLine("All photon torpedoes expended");
            return false;
        }

        // 重写基类的ExecuteCommandCore方法，执行核心命令
        protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
        {
            // 如果无法读取鱼雷航线，返回CommandResult.Ok
            if (!_io.TryReadCourse("Photon torpedo course", "Ensign Chekov", out var course))
            {
                return CommandResult.Ok;
            }

            // 鱼雷数量减1
            TorpedoCount -= 1;

            // 初始化isHit为false
            var isHit = false;
            // 输出"Torpedo track:"
            _io.WriteLine("Torpedo track:");
            // 遍历航线上的各个区块
            foreach (var sector in course.GetSectorsFrom(_enterprise.SectorCoordinates))
            {
                // 输出区块坐标
                _io.WriteLine($"                {sector}");

                // 如果在该区块有鱼雷碰撞，输出消息，设置isHit为true，如果游戏结束，返回CommandResult.GameOver
                if (quadrant.TorpedoCollisionAt(sector, out var message, out var gameOver))
                {
                    _io.WriteLine(message);
                    isHit = true;
                    if (gameOver) { return CommandResult.GameOver; }
                    break;
                }
            }

            // 如果没有击中，输出"Torpedo missed!"
            if (!isHit) { _io.WriteLine("Torpedo missed!"); }

            // 返回quadrant.KlingonsFireOnEnterprise()的结果
            return quadrant.KlingonsFireOnEnterprise();
        }

        // 声明内部方法ReplenishTorpedoes，用于补充鱼雷数量
        internal void ReplenishTorpedoes() => TorpedoCount = _tubeCount;
    }
}
```