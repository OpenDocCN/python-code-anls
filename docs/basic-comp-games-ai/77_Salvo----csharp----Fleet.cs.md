# `basic-computer-games\77_Salvo\csharp\Fleet.cs`

```
// 使用不可变集合和可空引用类型
using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;

// 命名空间 Salvo
namespace Salvo
{
    // 内部类 Fleet
    internal class Fleet
    {
        // 只读字段 _ships，存储船只列表
        private readonly List<Ship> _ships;

        // Fleet 类的构造函数，接受 IReadWrite 接口类型的参数 io
        internal Fleet(IReadWrite io)
        {
            // 输出坐标提示信息
            io.WriteLine(Prompts.Coordinates);
            // 初始化 _ships 列表，包含四艘船只
            _ships = new()
            {
                new Battleship(io),
                new Cruiser(io),
                new Destroyer("A", io),
                new Destroyer("B", io)
            };
        }

        // Fleet 类的构造函数，接受 IRandom 接口类型的参数 random
        internal Fleet(IRandom random)
        {
            // 初始化 _ships 列表
            _ships = new();
            // 无限循环
            while (true)
            {
                // 添加一艘战舰到 _ships 列表
                _ships.Add(new Battleship(random));
                // 尝试放置巡洋舰、驱逐舰A和驱逐舰B，如果成功则返回
                if (TryPositionShip(() => new Cruiser(random)) &&
                    TryPositionShip(() => new Destroyer("A", random)) &&
                    TryPositionShip(() => new Destroyer("B", random)))
                {
                    return;
                }
                // 清空 _ships 列表
                _ships.Clear();
            }
        }

        // 尝试放置船只的方法
        bool TryPositionShip(Func<Ship> shipFactory)
        {
            // 放置船只的尝试次数
            var shipGenerationAttempts = 0;
            // 无限循环
            while (true)
            {
                // 使用船只工厂创建船只对象
                var ship = shipFactory.Invoke();
                // 增加放置船只的尝试次数
                shipGenerationAttempts++;
                // 如果尝试次数超过25次，则放置失败
                if (shipGenerationAttempts > 25) { return false; }
                // 如果新船只与现有船只的最小距离大于等于3.59，则放置成功
                if (_ships.Min(ship.DistanceTo) >= 3.59)
                {
                    // 将新船只添加到 _ships 列表
                    _ships.Add(ship);
                    return true;
                }
            }
        }

        // 返回 _ships 列表的只读副本
        internal IEnumerable<Ship> Ships => _ships.AsEnumerable();

        // 接收射击信息的方法
        internal void ReceiveShots(IEnumerable<Position> shots, Action<Ship> reportHit)
        {
            // 遍历射击位置
            foreach (var position in shots)
            {
                // 查找被击中的船只
                var ship = _ships.FirstOrDefault(s => s.IsHit(position));
                // 如果没有被击中的船只，则继续下一次循环
                if (ship == null) { continue; }
                // 如果被击中的船只已被摧毁，则从 _ships 列表中移除
                if (ship.IsDestroyed) { _ships.Remove(ship); }
                // 报告船只被击中
                reportHit(ship);
            }
        }
    }
}
```