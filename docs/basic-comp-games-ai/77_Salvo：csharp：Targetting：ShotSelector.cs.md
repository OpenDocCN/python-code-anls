# `77_Salvo\csharp\Targetting\ShotSelector.cs`

```
namespace Salvo.Targetting;  # 命名空间声明

internal abstract class ShotSelector  # 声明一个内部抽象类 ShotSelector
{
    private readonly Fleet _source;  # 声明一个私有只读字段 _source，类型为 Fleet
    private readonly Dictionary<Position, int> _previousShots = new();  # 声明一个私有只读字段 _previousShots，类型为 Position 到 int 的字典

    internal ShotSelector(Fleet source)  # 声明一个内部构造函数 ShotSelector，参数为 source
    {
        _source = source;  # 将参数 source 赋值给字段 _source
    }

    internal int NumberOfShots => _source.Ships.Sum(s => s.Shots);  # 声明一个内部属性 NumberOfShots，返回 _source 中所有船只的射击次数之和
    internal bool CanTargetAllRemainingSquares => NumberOfShots >= 100 - _previousShots.Count;  # 声明一个内部属性 CanTargetAllRemainingSquares，返回是否可以瞄准所有剩余的方块

    internal bool WasSelectedPreviously(Position position) => _previousShots.ContainsKey(position);  # 声明一个内部方法 WasSelectedPreviously，返回是否之前已选择过指定位置的方块

    internal bool WasSelectedPreviously(Position position, out int turn)  # 声明一个内部方法 WasSelectedPreviously，返回是否之前已选择过指定位置的方块，并将选择的轮次赋值给参数 turn
        => _previousShots.TryGetValue(position, out turn);  # 尝试从 _previousShots 中获取指定位置的值，并将结果赋值给参数 turn
}
    internal IEnumerable<Position> GetShots(int turnNumber)
    {
        // 从 GetShots() 方法中获取射击位置的集合
        foreach (var shot in GetShots())
        {
            // 将射击位置和回合数添加到 _previousShots 字典中
            _previousShots.Add(shot, turnNumber);
            // 返回当前射击位置
            yield return shot;
        }
    }

    // 定义一个抽象方法，用于获取射击位置的集合
    protected abstract IEnumerable<Position> GetShots();
}
```