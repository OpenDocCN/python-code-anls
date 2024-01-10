# `basic-computer-games\77_Salvo\csharp\Targetting\ComputerShotSelector.cs`

```
// 命名空间 Salvo.Targetting 下的内部类 ComputerShotSelector，继承自 ShotSelector
internal class ComputerShotSelector : ShotSelector
{
    // 私有字段，已知命中的射击选择策略
    private readonly KnownHitsShotSelectionStrategy _knownHitsStrategy;
    // 私有字段，搜索模式的射击选择策略
    private readonly SearchPatternShotSelectionStrategy _searchPatternStrategy;
    // 私有字段，输入输出接口
    private readonly IReadWrite _io;
    // 私有字段，是否显示射击
    private readonly bool _showShots;

    // 内部构造函数，接受 Fleet、IRandom 和 IReadWrite 参数
    internal ComputerShotSelector(Fleet source, IRandom random, IReadWrite io) 
        : base(source) // 调用基类的构造函数
    {
        // 初始化已知命中的射击选择策略
        _knownHitsStrategy = new KnownHitsShotSelectionStrategy(this);
        // 初始化搜索模式的射击选择策略
        _searchPatternStrategy = new SearchPatternShotSelectionStrategy(this, random);
        // 初始化输入输出接口
        _io = io;
        // 根据用户输入判断是否显示射击
        _showShots = io.ReadString(Prompts.SeeShots).Equals("yes", StringComparison.InvariantCultureIgnoreCase);
    }

    // 重写基类的方法，获取射击位置
    protected override IEnumerable<Position> GetShots()
    {
        // 获取选择策略的射击位置，并转换为数组
        var shots = GetSelectionStrategy().GetShots(NumberOfShots).ToArray();
        // 如果需要显示射击位置
        if (_showShots)
        {
            // 输出射击位置
            _io.WriteLine(string.Join(Environment.NewLine, shots));
        }
        // 返回射击位置数组
        return shots;
    }

    // 内部方法，记录命中的船只和回合数
    internal void RecordHit(Ship ship, int turn) => _knownHitsStrategy.RecordHit(ship, turn);

    // 私有方法，根据已知命中的射击选择策略和搜索模式的射击选择策略返回合适的选择策略
    private ShotSelectionStrategy GetSelectionStrategy()
        => _knownHitsStrategy.KnowsOfDamagedShips ? _knownHitsStrategy : _searchPatternStrategy;
}
```