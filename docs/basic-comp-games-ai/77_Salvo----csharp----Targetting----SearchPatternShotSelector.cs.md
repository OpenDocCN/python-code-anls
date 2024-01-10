# `basic-computer-games\77_Salvo\csharp\Targetting\SearchPatternShotSelector.cs`

```
// 命名空间 Salvo.Targetting 下的内部类 SearchPatternShotSelectionStrategy 继承自 ShotSelectionStrategy
internal class SearchPatternShotSelectionStrategy : ShotSelectionStrategy
{
    // 声明常量 MaxSearchPatternAttempts 并赋值为 100
    private const int MaxSearchPatternAttempts = 100;
    // 声明只读字段 _random，类型为 IRandom
    private readonly IRandom _random;
    // 声明只读字段 _searchPattern，类型为 SearchPattern，并初始化为新的实例
    private readonly SearchPattern _searchPattern = new();
    // 声明只读字段 _shots，类型为 List<Position>，并初始化为空列表
    private readonly List<Position> _shots = new();

    // SearchPatternShotSelectionStrategy 类的构造函数，接受 ShotSelector 和 IRandom 参数
    internal SearchPatternShotSelectionStrategy(ShotSelector shotSelector, IRandom random) 
        : base(shotSelector)
    {
        // 将参数 random 赋值给只读字段 _random
        _random = random;
    }

    // 重写基类的 GetShots 方法，返回一个包含 Position 的 IEnumerable，接受一个 int 类型的参数 numberOfShots
    internal override IEnumerable<Position> GetShots(int numberOfShots)
    {
        // 清空 _shots 列表
        _shots.Clear();
        // 当 _shots 列表长度小于 numberOfShots 时循环
        while(_shots.Count < numberOfShots)
        {
            // 从 _random 中获取下一个船的位置，并赋值给 seed
            var (seed, _) = _random.NextShipPosition();
            // 从 seed 开始搜索
            SearchFrom(numberOfShots, seed);
        }
        // 返回 _shots 列表
        return _shots;
    }

    // 从 candidateShot 开始搜索
    private void SearchFrom(int numberOfShots, Position candidateShot)
    {
        // 初始化 attemptsLeft 为 MaxSearchPatternAttempts
        var attemptsLeft = MaxSearchPatternAttempts;
        // 无限循环
        while (true)
        {
            // 重置 _searchPattern
            _searchPattern.Reset();
            // 如果 attemptsLeft 减到 0，则返回
            if (attemptsLeft-- == 0) { return; }
            // 将 candidateShot 带入范围内
            candidateShot = candidateShot.BringIntoRange(_random);
            // 如果找到有效的射击位置，则返回
            if (FindValidShots(numberOfShots, ref candidateShot)) { return; }
        }
    }

    // 查找有效的射击位置
    private bool FindValidShots(int numberOfShots, ref Position candidateShot)
    {
        // 无限循环
        while (true)
        {
            // 如果 candidateShot 是有效的射击位置
            if (IsValidShot(candidateShot))
            {
                // 将 candidateShot 添加到 _shots 列表中
                _shots.Add(candidateShot);
                // 如果 _shots 列表长度等于 numberOfShots，则返回 true
                if (_shots.Count == numberOfShots) { return true; }
            }
            // 如果无法获取偏移量，则返回 false
            if (!_searchPattern.TryGetOffset(out var offset)) { return false; }
            // 将 candidateShot 加上偏移量
            candidateShot += offset;
        }
    }

    // 判断 candidate 是否为有效的射击位置
    private bool IsValidShot(Position candidate)
        => candidate.IsInRange && !WasSelectedPreviously(candidate) && !_shots.Contains(candidate);
}
```