# `basic-computer-games\77_Salvo\csharp\Targetting\ShotSelector.cs`

```

// 命名空间 Salvo.Targetting，表示该类在 Salvo.Targetting 命名空间下
internal abstract class ShotSelector
{
    // 私有字段 _source，表示射击选择器的来源舰队
    private readonly Fleet _source;
    // 私有字段 _previousShots，表示之前的射击位置和对应的回合数
    private readonly Dictionary<Position, int> _previousShots = new();

    // 构造函数，初始化射击选择器的来源舰队
    internal ShotSelector(Fleet source)
    {
        _source = source;
    }

    // 属性 NumberOfShots，表示剩余射击次数
    internal int NumberOfShots => _source.Ships.Sum(s => s.Shots);
    // 属性 CanTargetAllRemainingSquares，表示是否可以瞄准所有剩余的方格
    internal bool CanTargetAllRemainingSquares => NumberOfShots >= 100 - _previousShots.Count;

    // 方法 WasSelectedPreviously，判断指定位置是否之前已经选择过
    internal bool WasSelectedPreviously(Position position) => _previousShots.ContainsKey(position);

    // 方法 WasSelectedPreviously，判断指定位置是否之前已经选择过，并返回对应的回合数
    internal bool WasSelectedPreviously(Position position, out int turn)
        => _previousShots.TryGetValue(position, out turn);

    // 方法 GetShots，获取射击位置，并记录对应的回合数
    internal IEnumerable<Position> GetShots(int turnNumber)
    {
        foreach (var shot in GetShots())
        {
            _previousShots.Add(shot, turnNumber);
            yield return shot;
        }
    }

    // 抽象方法 GetShots，获取射击位置的抽象方法，具体实现由子类实现
    protected abstract IEnumerable<Position> GetShots();
}

```