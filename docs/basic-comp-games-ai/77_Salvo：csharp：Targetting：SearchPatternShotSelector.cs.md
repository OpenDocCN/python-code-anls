# `d:/src/tocomm/basic-computer-games\77_Salvo\csharp\Targetting\SearchPatternShotSelector.cs`

```
namespace Salvo.Targetting;

internal class SearchPatternShotSelectionStrategy : ShotSelectionStrategy
{
    private const int MaxSearchPatternAttempts = 100; // 定义最大搜索模式尝试次数
    private readonly IRandom _random; // 声明一个私有的随机数生成器接口
    private readonly SearchPattern _searchPattern = new(); // 声明一个搜索模式对象
    private readonly List<Position> _shots = new(); // 声明一个位置列表用于存储射击位置

    internal SearchPatternShotSelectionStrategy(ShotSelector shotSelector, IRandom random) 
        : base(shotSelector) // 调用基类的构造函数
    {
        _random = random; // 初始化随机数生成器
    }

    internal override IEnumerable<Position> GetShots(int numberOfShots) // 重写基类的获取射击位置的方法
    {
        _shots.Clear(); // 清空射击位置列表
        while(_shots.Count < numberOfShots) // 当射击位置列表中的位置数量小于指定的射击数量时执行循环
        {
            // 生成随机种子和船的方向
            var (seed, _) = _random.NextShipPosition();
            // 从生成的随机种子位置开始搜索
            SearchFrom(numberOfShots, seed);
        }
        // 返回搜索到的射击位置
        return _shots;
    }

    // 从候选射击位置开始搜索
    private void SearchFrom(int numberOfShots, Position candidateShot)
    {
        // 设置最大搜索尝试次数
        var attemptsLeft = MaxSearchPatternAttempts;
        // 循环搜索
        while (true)
        {
            // 重置搜索模式
            _searchPattern.Reset();
            // 如果尝试次数用尽，则返回
            if (attemptsLeft-- == 0) { return; }
            // 将候选射击位置调整到合适的范围内
            candidateShot = candidateShot.BringIntoRange(_random);
            // 查找有效的射击位置
            if (FindValidShots(numberOfShots, ref candidateShot)) { return; }
        }
    }

    // 查找有效的射击位置
    private bool FindValidShots(int numberOfShots, ref Position candidateShot)
    {
        while (true)  # 进入一个无限循环
        {
            if (IsValidShot(candidateShot))  # 如果候选射击位置有效
            {
                _shots.Add(candidateShot);  # 将候选射击位置添加到射击列表中
                if (_shots.Count == numberOfShots) { return true; }  # 如果射击列表中的射击数量达到预定数量，则返回true
            }
            if (!_searchPattern.TryGetOffset(out var offset)) { return false; }  # 如果搜索模式无法获取偏移量，则返回false
            candidateShot += offset;  # 将候选射击位置增加偏移量
        }
    }

    private bool IsValidShot(Position candidate)  # 定义一个私有方法，用于判断候选射击位置是否有效
        => candidate.IsInRange && !WasSelectedPreviously(candidate) && !_shots.Contains(candidate);  # 候选射击位置必须在有效范围内，并且之前没有被选择过，并且不在射击列表中
}
```