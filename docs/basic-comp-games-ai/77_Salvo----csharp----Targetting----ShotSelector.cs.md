# `basic-computer-games\77_Salvo\csharp\Targetting\ShotSelector.cs`

```py
// 命名空间 Salvo.Targetting 下的内部抽象类 ShotSelector
internal abstract class ShotSelector
{
    // 只读字段 _source，表示发射源舰队
    private readonly Fleet _source;
    // 字典 _previousShots，用于记录之前的射击位置和次数
    private readonly Dictionary<Position, int> _previousShots = new();

    // 构造函数，初始化发射源舰队
    internal ShotSelector(Fleet source)
    {
        _source = source;
    }

    // 属性 NumberOfShots，表示剩余射击次数
    internal int NumberOfShots => _source.Ships.Sum(s => s.Shots);
    // 属性 CanTargetAllRemainingSquares，表示是否可以瞄准所有剩余的方格
    internal bool CanTargetAllRemainingSquares => NumberOfShots >= 100 - _previousShots.Count;

    // 方法 WasSelectedPreviously，判断指定位置是否之前已被选择过
    internal bool WasSelectedPreviously(Position position) => _previousShots.ContainsKey(position);

    // 方法 WasSelectedPreviously，判断指定位置是否之前已被选择过，并返回射击次数
    internal bool WasSelectedPreviously(Position position, out int turn)
        => _previousShots.TryGetValue(position, out turn);

    // 方法 GetShots，根据回合数获取射击位置
    internal IEnumerable<Position> GetShots(int turnNumber)
    {
        // 遍历获取的射击位置，记录到_previousShots中，并返回
        foreach (var shot in GetShots())
        {
            _previousShots.Add(shot, turnNumber);
            yield return shot;
        }
    }

    // 抽象方法 GetShots，用于获取射击位置
    protected abstract IEnumerable<Position> GetShots();
}
```