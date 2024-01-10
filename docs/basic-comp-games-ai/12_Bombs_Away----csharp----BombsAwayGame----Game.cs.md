# `basic-computer-games\12_Bombs_Away\csharp\BombsAwayGame\Game.cs`

```
// 命名空间声明，定义了代码所属的命名空间
namespace BombsAwayGame;

/// <summary>
/// 使用提供的 <see cref="IUserInterface"/> 执行 Bombs Away 游戏。
/// </summary>
// 定义 Game 类
public class Game
{
    // 只读字段，存储 IUserInterface 实例
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
        // 输出提示信息
        _ui.Output("YOU ARE A PILOT IN A WORLD WAR II BOMBER.");
        // 选择一方并执行该方的逻辑
        Side side = ChooseSide();
        side.Play();
    }

    /// <summary>
    /// 表示一个 <see cref="Side"/>。
    /// </summary>
    /// <param name="Name">方的名称。</param>
    /// <param name="CreateSide">创建该描述符表示的方的实例。</param>
    // 定义一个记录类型 SideDescriptor
    private record class SideDescriptor(string Name, Func<Side> CreateSide);

    /// <summary>
    /// 选择一方并返回该方的新实例。
    /// </summary>
    /// <returns>选择的方的新实例。</returns>
    // ChooseSide 方法
    private Side ChooseSide()
    {
        // 获取所有方的描述符
        SideDescriptor[] sides = AllSideDescriptors;
        // 获取所有方的名称
        string[] sideNames = sides.Select(a => a.Name).ToArray();
        // 从用户界面中选择一方
        int index = _ui.Choose("WHAT SIDE", sideNames);
        // 返回选择的方的新实例
        return sides[index].CreateSide();
    }

    /// <summary>
    /// 所有方的描述符。
    /// </summary>
    // AllSideDescriptors 属性
    private SideDescriptor[] AllSideDescriptors => new SideDescriptor[]
    {
        // 初始化所有方的描述符
        new("ITALY", () => new ItalySide(_ui)),
        new("ALLIES", () => new AlliesSide(_ui)),
        new("JAPAN", () => new JapanSide(_ui)),
        new("GERMANY", () => new GermanySide(_ui)),
    };
}
```