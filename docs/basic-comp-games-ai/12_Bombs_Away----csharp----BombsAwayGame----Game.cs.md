# `basic-computer-games\12_Bombs_Away\csharp\BombsAwayGame\Game.cs`

```

// 命名空间声明，指定代码所属的命名空间
namespace BombsAwayGame;

/// <summary>
/// 使用提供的 <see cref="IUserInterface"/> 执行 Bombs Away 游戏。
/// </summary>
// 定义 Game 类
public class Game
{
    // 保存 IUserInterface 实例的私有字段
    private readonly IUserInterface _ui;

    /// <summary>
    /// 使用给定的 UI 创建游戏实例。
    /// </summary>
    /// <param name="ui">用于游戏的 UI。</param>
    // Game 类的构造函数
    public Game(IUserInterface ui)
    {
        _ui = ui;
    }

    /// <summary>
    /// 进行游戏。选择一方并执行该方的逻辑。
    /// </summary>
    // Play 方法
    public void Play()
    {
        _ui.Output("YOU ARE A PILOT IN A WORLD WAR II BOMBER.");
        Side side = ChooseSide();
        side.Play();
    }

    // 定义一个记录类型 SideDescriptor
    /// <summary>
    /// 表示一个 <see cref="Side"/>。
    /// </summary>
    /// <param name="Name">方的名称。</param>
    /// <param name="CreateSide">创建该描述符表示的方的实例。</param>
    private record class SideDescriptor(string Name, Func<Side> CreateSide);

    /// <summary>
    /// 选择一方并返回该方的新实例。
    /// </summary>
    /// <returns>选择的方的新实例。</returns>
    // ChooseSide 方法
    private Side ChooseSide()
    {
        SideDescriptor[] sides = AllSideDescriptors;
        string[] sideNames = sides.Select(a => a.Name).ToArray();
        int index = _ui.Choose("WHAT SIDE", sideNames);
        return sides[index].CreateSide();
    }

    /// <summary>
    /// 所有方的描述符。
    /// </summary>
    // AllSideDescriptors 属性
    private SideDescriptor[] AllSideDescriptors => new SideDescriptor[]
    {
        new("ITALY", () => new ItalySide(_ui)),
        new("ALLIES", () => new AlliesSide(_ui)),
        new("JAPAN", () => new JapanSide(_ui)),
        new("GERMANY", () => new GermanySide(_ui)),
    };
}

```