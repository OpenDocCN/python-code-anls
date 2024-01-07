# `basic-computer-games\77_Salvo\csharp\Targetting\SearchPatternShotSelector.cs`

```

// 命名空间 Salvo.Targetting 下的内部类 SearchPatternShotSelectionStrategy 继承自 ShotSelectionStrategy
internal class SearchPatternShotSelectionStrategy : ShotSelectionStrategy
{
    // 声明常量 MaxSearchPatternAttempts 并赋值为 100
    private const int MaxSearchPatternAttempts = 100;
    // 声明私有变量 _random 和 _searchPattern，并初始化为 IRandom 接口的实例和 SearchPattern 类的实例
    private readonly IRandom _random;
    private readonly SearchPattern _searchPattern = new();
    // 声明私有变量 _shots，并初始化为 Position 类的列表
    private readonly List<Position> _shots = new();

    // 构造函数，接受 ShotSelector 和 IRandom 参数，并调用基类的构造函数
    internal SearchPatternShotSelectionStrategy(ShotSelector shotSelector, IRandom random) 
        : base(shotSelector)
    {
        // 将传入的 random 参数赋值给 _random
        _random = random;
    }

    // 重写基类的 GetShots 方法
    internal override IEnumerable<Position> GetShots(int numberOfShots)
    {
        // 清空 _shots 列表
        _shots.Clear();
        // 循环直到 _shots 列表中包含的元素数量达到 numberOfShots
        while(_shots.Count < numberOfShots)
        {
            // 从 _random 中获取下一个船的位置，并调用 SearchFrom 方法
            var (seed, _) = _random.NextShipPosition();
            SearchFrom(numberOfShots, seed);
        }
        // 返回 _shots 列表
        return _shots;
    }

    // 定义私有方法 SearchFrom，接受 numberOfShots 和 candidateShot 作为参数
    private void SearchFrom(int numberOfShots, Position candidateShot)
    {
        // 初始化 attemptsLeft 变量为 MaxSearchPatternAttempts
        var attemptsLeft = MaxSearchPatternAttempts;
        // 无限循环
        while (true)
        {
            // 重置 _searchPattern
            _searchPattern.Reset();
            // 如果 attemptsLeft 减到 0，则返回
            if (attemptsLeft-- == 0) { return; }
            // 将 candidateShot 限制在范围内
            candidateShot = candidateShot.BringIntoRange(_random);
            // 如果找到有效的射击位置，则返回
            if (FindValidShots(numberOfShots, ref candidateShot)) { return; }
        }
    }

    // 定义私有方法 FindValidShots，接受 numberOfShots 和 candidateShot 的引用作为参数
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
                // 如果 _shots 列表中的元素数量达到 numberOfShots，则返回 true
                if (_shots.Count == numberOfShots) { return true; }
            }
            // 如果无法获取偏移量，则返回 false
            if (!_searchPattern.TryGetOffset(out var offset)) { return false; }
            // 将 candidateShot 加上偏移量
            candidateShot += offset;
        }
    }

    // 定义私有方法 IsValidShot，接受 candidate 作为参数
    private bool IsValidShot(Position candidate)
        // 返回 candidate 是否在范围内，并且之前没有被选择过，并且不在 _shots 列表中
        => candidate.IsInRange && !WasSelectedPreviously(candidate) && !_shots.Contains(candidate);
}

```