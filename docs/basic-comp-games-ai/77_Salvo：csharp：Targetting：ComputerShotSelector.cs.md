# `d:/src/tocomm/basic-computer-games\77_Salvo\csharp\Targetting\ComputerShotSelector.cs`

```
namespace Salvo.Targetting;  // 命名空间声明，表示该类所属的命名空间

internal class ComputerShotSelector : ShotSelector  // 声明一个内部类 ComputerShotSelector，继承自 ShotSelector 类
{
    private readonly KnownHitsShotSelectionStrategy _knownHitsStrategy;  // 声明一个私有的 KnownHitsShotSelectionStrategy 类型的成员变量
    private readonly SearchPatternShotSelectionStrategy _searchPatternStrategy;  // 声明一个私有的 SearchPatternShotSelectionStrategy 类型的成员变量
    private readonly IReadWrite _io;  // 声明一个私有的 IReadWrite 类型的成员变量
    private readonly bool _showShots;  // 声明一个私有的布尔类型的成员变量

    internal ComputerShotSelector(Fleet source, IRandom random, IReadWrite io)  // 声明一个内部的构造函数，接受 Fleet、IRandom 和 IReadWrite 类型的参数
        : base(source)  // 调用基类的构造函数，传入 source 参数
    {
        _knownHitsStrategy = new KnownHitsShotSelectionStrategy(this);  // 初始化 _knownHitsStrategy 成员变量，使用 KnownHitsShotSelectionStrategy 类的构造函数
        _searchPatternStrategy = new SearchPatternShotSelectionStrategy(this, random);  // 初始化 _searchPatternStrategy 成员变量，使用 SearchPatternShotSelectionStrategy 类的构造函数
        _io = io;  // 将传入的 io 参数赋值给 _io 成员变量
        _showShots = io.ReadString(Prompts.SeeShots).Equals("yes", StringComparison.InvariantCultureIgnoreCase);  // 将从 io 读取的字符串与 "yes" 进行比较，结果赋值给 _showShots 成员变量
    }

    protected override IEnumerable<Position> GetShots()  // 重写基类的 GetShots 方法
    {
        // 从选择策略中获取射击的位置数组
        var shots = GetSelectionStrategy().GetShots(NumberOfShots).ToArray();
        // 如果需要展示射击位置，则将其以换行符连接后输出
        if (_showShots)
        {
            _io.WriteLine(string.Join(Environment.NewLine, shots));
        }
        // 返回射击位置数组
        return shots;
    }

    // 记录击中的船只和回合数
    internal void RecordHit(Ship ship, int turn) => _knownHitsStrategy.RecordHit(ship, turn);

    // 获取选择策略
    private ShotSelectionStrategy GetSelectionStrategy()
        => _knownHitsStrategy.KnowsOfDamagedShips ? _knownHitsStrategy : _searchPatternStrategy;
```