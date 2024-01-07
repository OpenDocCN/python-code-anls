# `basic-computer-games\77_Salvo\csharp\Fleet.cs`

```

using System.Collections.Immutable; // 导入不可变集合的命名空间
using System.Diagnostics.CodeAnalysis; // 导入排除分析的命名空间

namespace Salvo; // 命名空间声明

internal class Fleet // 内部舰队类
{
    private readonly List<Ship> _ships; // 私有的船只列表

    internal Fleet(IReadWrite io) // 内部舰队构造函数，接受 IReadWrite 接口类型的参数
    {
        io.WriteLine(Prompts.Coordinates); // 使用 IReadWrite 接口的 WriteLine 方法输出坐标提示
        _ships = new() // 初始化船只列表
        {
            new Battleship(io), // 添加一个战舰对象到船只列表
            new Cruiser(io), // 添加一个巡洋舰对象到船只列表
            new Destroyer("A", io), // 添加一个以"A"命名的驱逐舰对象到船只列表
            new Destroyer("B", io) // 添加一个以"B"命名的驱逐舰对象到船只列表
        };
    }

    internal Fleet(IRandom random) // 内部舰队构造函数，接受 IRandom 接口类型的参数
    {
        _ships = new(); // 初始化船只列表
        while (true) // 无限循环
        {
            _ships.Add(new Battleship(random)); // 向船只列表添加一个随机生成的战舰对象
            if (TryPositionShip(() => new Cruiser(random)) && // 尝试定位巡洋舰
                TryPositionShip(() => new Destroyer("A", random)) && // 尝试定位以"A"命名的驱逐舰
                TryPositionShip(() => new Destroyer("B", random))) // 尝试定位以"B"命名的驱逐舰
            {
                return; // 如果成功定位所有船只，则结束循环
            } 
            _ships.Clear(); // 清空船只列表
        }

        bool TryPositionShip(Func<Ship> shipFactory) // 尝试定位船只的方法，接受一个返回 Ship 对象的委托
        {
            var shipGenerationAttempts = 0; // 船只生成尝试次数
            while (true) // 无限循环
            {
                var ship = shipFactory.Invoke(); // 通过委托生成船只对象
                shipGenerationAttempts++; // 尝试次数加一
                if (shipGenerationAttempts > 25) { return false; } // 如果尝试次数超过25次，则返回失败
                if (_ships.Min(ship.DistanceTo) >= 3.59) // 如果船只与已有船只的最小距离大于等于3.59
                {
                    _ships.Add(ship); // 向船只列表添加船只
                    return true; // 返回成功
                }
            }
        }
    }

    internal IEnumerable<Ship> Ships => _ships.AsEnumerable(); // 返回船只列表的可枚举接口

    internal void ReceiveShots(IEnumerable<Position> shots, Action<Ship> reportHit) // 接收射击的方法，接受射击位置列表和报告击中的委托
    {
        foreach (var position in shots) // 遍历射击位置列表
        {
            var ship = _ships.FirstOrDefault(s => s.IsHit(position)); // 查找被击中的船只
            if (ship == null) { continue; } // 如果没有被击中的船只，则继续下一次循环
            if (ship.IsDestroyed) { _ships.Remove(ship); } // 如果船只被摧毁，则从船只列表中移除
            reportHit(ship); // 报告击中的船只
        }
    }
}

```