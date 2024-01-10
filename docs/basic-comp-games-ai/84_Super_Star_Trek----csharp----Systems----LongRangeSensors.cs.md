# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\LongRangeSensors.cs`

```
using System.Linq; // 导入 LINQ 扩展方法
using Games.Common.IO; // 导入通用游戏输入输出命名空间
using SuperStarTrek.Commands; // 导入超级星际远征命令
using SuperStarTrek.Space; // 导入超级星际远征空间命名空间

namespace SuperStarTrek.Systems
{
    internal class LongRangeSensors : Subsystem
    {
        private readonly Galaxy _galaxy; // 声明私有的星系对象
        private readonly IReadWrite _io; // 声明私有的读写接口对象

        internal LongRangeSensors(Galaxy galaxy, IReadWrite io) // 构造函数，初始化星系和读写接口
            : base("Long Range Sensors", Command.LRS, io) // 调用基类的构造函数
        {
            _galaxy = galaxy; // 初始化星系对象
            _io = io; // 初始化读写接口对象
        }

        protected override bool CanExecuteCommand() => IsOperational("{name} are inoperable"); // 检查是否可以执行命令

        protected override CommandResult ExecuteCommandCore(Quadrant quadrant) // 执行核心命令
        {
            _io.WriteLine($"Long range scan for quadrant {quadrant.Coordinates}"); // 输出长程扫描的象限坐标
            _io.WriteLine("-------------------"); // 输出分隔线
            foreach (var quadrants in _galaxy.GetNeighborhood(quadrant)) // 遍历获取邻近象限
            {
                _io.WriteLine(": " + string.Join(" : ", quadrants.Select(q => q?.Scan() ?? "***")) + " :"); // 输出扫描结果
                _io.WriteLine("-------------------"); // 输出分隔线
            }

            return CommandResult.Ok; // 返回命令执行结果
        }
    }
}
```