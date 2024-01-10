# `basic-computer-games\77_Salvo\csharp\Targetting\ShotSelectionStrategy.cs`

```
# 命名空间 Salvo.Targetting 下的抽象类 ShotSelectionStrategy
internal abstract class ShotSelectionStrategy
{
    # 只读字段，用于存储 ShotSelector 对象
    private readonly ShotSelector _shotSelector;
    
    # 构造函数，接受 ShotSelector 对象作为参数
    protected ShotSelectionStrategy(ShotSelector shotSelector)
    {
        _shotSelector = shotSelector;
    }

    # 抽象方法，用于获取一定数量的射击位置
    internal abstract IEnumerable<Position> GetShots(int numberOfShots);

    # 保护方法，用于检查指定位置是否之前已经选择过
    protected bool WasSelectedPreviously(Position position) => _shotSelector.WasSelectedPreviously(position);

    # 保护方法，用于检查指定位置是否之前已经选择过，并返回选择的轮次
    protected bool WasSelectedPreviously(Position position, out int turn)
        => _shotSelector.WasSelectedPreviously(position, out turn);
}
```