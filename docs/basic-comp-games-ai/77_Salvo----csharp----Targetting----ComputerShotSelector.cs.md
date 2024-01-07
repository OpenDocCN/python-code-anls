# `basic-computer-games\77_Salvo\csharp\Targetting\ComputerShotSelector.cs`

```

// 命名空间 Salvo.Targetting，表示该类属于 Salvo.Targetting 命名空间
internal class ComputerShotSelector : ShotSelector
{
    // 声明私有字段 KnownHitsShotSelectionStrategy，SearchPatternShotSelectionStrategy，IReadWrite 和 bool 类型
    private readonly KnownHitsShotSelectionStrategy _knownHitsStrategy;
    private readonly SearchPatternShotSelectionStrategy _searchPatternStrategy;
    private readonly IReadWrite _io;
    private readonly bool _showShots;

    // 构造函数，接受 Fleet、IRandom 和 IReadWrite 参数
    internal ComputerShotSelector(Fleet source, IRandom random, IReadWrite io) 
        : base(source) // 调用基类的构造函数
    {
        // 初始化私有字段
        _knownHitsStrategy = new KnownHitsShotSelectionStrategy(this);
        _searchPatternStrategy = new SearchPatternShotSelectionStrategy(this, random);
        _io = io;
        // 根据用户输入判断是否显示射击信息
        _showShots = io.ReadString(Prompts.SeeShots).Equals("yes", StringComparison.InvariantCultureIgnoreCase);
    }

    // 重写基类的方法，返回一个包含位置的集合
    protected override IEnumerable<Position> GetShots()
    {
        // 调用 GetSelectionStrategy 方法获取射击位置集合，并转换为数组
        var shots = GetSelectionStrategy().GetShots(NumberOfShots).ToArray();
        // 如果需要显示射击信息，则将射击位置集合以换行符连接后输出
        if (_showShots)
        {
            _io.WriteLine(string.Join(Environment.NewLine, shots));
        }
        // 返回射击位置集合
        return shots;
    }

    // 记录击中信息，接受 Ship 和 turn 参数
    internal void RecordHit(Ship ship, int turn) => _knownHitsStrategy.RecordHit(ship, turn);

    // 获取射击策略，根据已知击中船只的情况返回不同的策略
    private ShotSelectionStrategy GetSelectionStrategy()
        => _knownHitsStrategy.KnowsOfDamagedShips ? _knownHitsStrategy : _searchPatternStrategy;
}

```