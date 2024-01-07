# `basic-computer-games\77_Salvo\csharp\Targetting\ShotSelectionStrategy.cs`

```

// 命名空间 Salvo.Targetting，表示该类属于 Salvo.Targetting 命名空间
internal abstract class ShotSelectionStrategy
{
    // 只读字段 _shotSelector，用于存储 ShotSelector 对象
    private readonly ShotSelector _shotSelector;
    
    // 受保护的构造函数，用于初始化 ShotSelectionStrategy 对象
    protected ShotSelectionStrategy(ShotSelector shotSelector)
    {
        _shotSelector = shotSelector;
    }

    // 抽象方法，用于获取一定数量的射击位置
    internal abstract IEnumerable<Position> GetShots(int numberOfShots);

    // 受保护的方法，用于检查指定位置是否之前已经选择过
    protected bool WasSelectedPreviously(Position position) => _shotSelector.WasSelectedPreviously(position);

    // 受保护的方法，用于检查指定位置是否之前已经选择过，并返回选择的轮次
    protected bool WasSelectedPreviously(Position position, out int turn)
        => _shotSelector.WasSelectedPreviously(position, out turn);
}

```