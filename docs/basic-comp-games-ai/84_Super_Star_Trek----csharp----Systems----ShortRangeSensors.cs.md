# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\ShortRangeSensors.cs`

```
{
    // 定义 ShortRangeSensors 类，继承自 Subsystem 类
    private readonly Enterprise _enterprise; // 保存 Enterprise 对象的引用
    private readonly Galaxy _galaxy; // 保存 Galaxy 对象的引用
    private readonly Game _game; // 保存 Game 对象的引用
    private readonly IReadWrite _io; // 保存 IReadWrite 对象的引用

    // ShortRangeSensors 类的构造函数，接受 Enterprise、Galaxy、Game 和 IReadWrite 对象的引用
    internal ShortRangeSensors(Enterprise enterprise, Galaxy galaxy, Game game, IReadWrite io)
        : base("Short Range Sensors", Command.SRS, io) // 调用父类 Subsystem 的构造函数
    {
        _enterprise = enterprise; // 初始化 _enterprise 成员变量
        _galaxy = galaxy; // 初始化 _galaxy 成员变量
        _game = game; // 初始化 _game 成员变量
        _io = io; // 初始化 _io 成员变量
    }

    // 重写父类 Subsystem 的 ExecuteCommandCore 方法，执行短程传感器的核心命令逻辑
    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
    {
        // 如果 Enterprise 被停靠，则输出相应信息
        if (_enterprise.IsDocked)
        {
            _io.WriteLine(Strings.ShieldsDropped);
        }

        // 如果 Condition 小于 0，则输出相应信息
        if (Condition < 0)
        {
            _io.WriteLine(Strings.ShortRangeSensorsOut);
        }

        // 输出分隔线
        _io.WriteLine("---------------------------------");
        // 获取象限的显示行，并与状态行进行合并，然后输出
        quadrant.GetDisplayLines()
            .Zip(GetStatusLines(), (sectors, status) => $" {sectors}         {status}")
            .ToList()
            .ForEach(l => _io.WriteLine(l));
        // 输出分隔线
        _io.WriteLine("---------------------------------");

        // 返回命令执行结果为 Ok
        return CommandResult.Ok;
    }

    // 获取状态行的方法
    internal IEnumerable<string> GetStatusLines()
}
    {
        // 返回星际日期
        yield return $"Stardate           {_game.Stardate}";
        // 返回企业号的状态
        yield return $"Condition          {_enterprise.Condition}";
        // 返回企业号所在的象限坐标
        yield return $"Quadrant           {_enterprise.QuadrantCoordinates}";
        // 返回企业号所在的扇区坐标
        yield return $"Sector             {_enterprise.SectorCoordinates}";
        // 返回光子鱼雷数量
        yield return $"Photon torpedoes   {_enterprise.PhotonTubes.TorpedoCount}";
        // 返回总能量
        yield return $"Total energy       {Math.Ceiling(_enterprise.TotalEnergy)}";
        // 返回护盾能量
        yield return $"Shields            {(int)_enterprise.ShieldControl.ShieldEnergy}";
        // 返回剩余克林贡人数
        yield return $"Klingons remaining {_galaxy.KlingonCount}";
    }
# 闭合前面的函数定义
```