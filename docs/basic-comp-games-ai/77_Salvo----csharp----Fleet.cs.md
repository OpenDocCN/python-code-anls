# `77_Salvo\csharp\Fleet.cs`

```
# 导入必要的模块
using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;

# 定义命名空间
namespace Salvo;

# 定义内部类 Fleet
internal class Fleet
{
    # 声明私有成员变量 _ships，用于存储船只对象
    private readonly List<Ship> _ships;

    # 定义构造函数，接受一个 IReadWrite 类型的参数 io
    internal Fleet(IReadWrite io)
    {
        # 调用 io 对象的 WriteLine 方法，输出提示信息 "Coordinates"
        io.WriteLine(Prompts.Coordinates);
        # 初始化 _ships 列表，包含四艘船只对象：一艘战舰、一艘巡洋舰、两艘驱逐舰
        _ships = new()
        {
            new Battleship(io),
            new Cruiser(io),
            new Destroyer("A", io),
            new Destroyer("B", io)
        };
    }
```
```python
    # 创建一个内部的 Fleet 类，接受一个 IRandom 接口类型的参数 random
    internal Fleet(IRandom random)
    {
        # 初始化一个空的船队列表
        _ships = new();
        # 无限循环，直到成功生成船队
        while (true)
        {
            # 向船队列表中添加一艘战舰
            _ships.Add(new Battleship(random));
            # 尝试放置巡洋舰和两艘驱逐舰，如果成功则返回
            if (TryPositionShip(() => new Cruiser(random)) &&
                TryPositionShip(() => new Destroyer("A", random)) &&
                TryPositionShip(() => new Destroyer("B", random)))
            {
                return;
            } 
            # 如果放置失败，则清空船队列表，重新尝试生成船队
            _ships.Clear();
        }

        # 尝试放置一艘船的方法，接受一个返回 Ship 类型对象的函数作为参数
        bool TryPositionShip(Func<Ship> shipFactory)
        {
            # 初始化放置船的尝试次数
            var shipGenerationAttempts = 0;
            # 无限循环，直到成功放置船或者放置尝试次数超过限制
            while (true)
            {
                // 调用 shipFactory 创建一个新的船只对象
                var ship = shipFactory.Invoke();
                // 增加船只生成尝试次数
                shipGenerationAttempts++;
                // 如果船只生成尝试次数超过 25 次，则返回 false
                if (shipGenerationAttempts > 25) { return false; }
                // 如果船只与现有船只的最小距离大于等于 3.59，则将船只添加到船只列表中并返回 true
                if (_ships.Min(ship.DistanceTo) >= 3.59)
                {
                    _ships.Add(ship);
                    return true; 
                }
            }
        }
    }

    // 返回船只列表的可枚举集合
    internal IEnumerable<Ship> Ships => _ships.AsEnumerable();

    // 接收射击位置的集合，并根据射击结果调用 reportHit 方法
    internal void ReceiveShots(IEnumerable<Position> shots, Action<Ship> reportHit)
    {
        // 遍历射击位置集合
        foreach (var position in shots)
        {
            // 查找第一个被击中的船只，并调用 reportHit 方法
            var ship = _ships.FirstOrDefault(s => s.IsHit(position));
            if (ship == null) { continue; }  # 如果船为空，跳过当前循环，继续下一次循环
            if (ship.IsDestroyed) { _ships.Remove(ship); }  # 如果船被摧毁，从船列表中移除该船
            reportHit(ship);  # 报告船被击中
        }
    }
}
```