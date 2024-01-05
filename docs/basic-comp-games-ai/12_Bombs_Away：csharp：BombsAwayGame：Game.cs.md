# `12_Bombs_Away\csharp\BombsAwayGame\Game.cs`

```
/// <summary>
/// Plays the Bombs Away game using a supplied <see cref="IUserInterface"/>.
/// </summary>
public class Game
{
    private readonly IUserInterface _ui; // 声明私有变量 _ui，用于存储 IUserInterface 接口的实例

    /// <summary>
    /// Create game instance using the given UI.
    /// </summary>
    /// <param name="ui">UI to use for game.</param>
    public Game(IUserInterface ui) // 构造函数，接受一个 IUserInterface 接口的实例作为参数
    {
        _ui = ui; // 将传入的 IUserInterface 实例赋值给私有变量 _ui
    }

    /// <summary>
    /// Play game. Choose a side and play the side's logic.
    /// </summary>
    public void Play()
    {
        // 输出提示信息
        _ui.Output("YOU ARE A PILOT IN A WORLD WAR II BOMBER.");
        // 选择阵营
        Side side = ChooseSide();
        // 开始游戏
        side.Play();
    }

    /// <summary>
    /// Represents a <see cref="Side"/>.
    /// </summary>
    /// <param name="Name">Name of side.</param>
    /// <param name="CreateSide">Create instance of side that this descriptor represents.</param>
    // 定义一个记录类型，表示阵营的描述符，包括名称和创建该阵营实例的函数
    private record class SideDescriptor(string Name, Func<Side> CreateSide);

    /// <summary>
    /// Choose side and return a new instance of that side.
    /// </summary>
    /// <returns>New instance of side that was chosen.</returns>
    // 选择阵营并返回该阵营的新实例
    private Side ChooseSide()
    {
        // 获取所有边描述符
        SideDescriptor[] sides = AllSideDescriptors;
        // 获取所有边的名称
        string[] sideNames = sides.Select(a => a.Name).ToArray();
        // 通过用户界面选择边的索引
        int index = _ui.Choose("WHAT SIDE", sideNames);
        // 根据选择的索引创建对应的边对象并返回
        return sides[index].CreateSide();
    }

    /// <summary>
    /// All side descriptors.
    /// </summary>
    // 获取所有边描述符的私有属性
    private SideDescriptor[] AllSideDescriptors => new SideDescriptor[]
    {
        // 创建意大利边描述符
        new("ITALY", () => new ItalySide(_ui)),
        // 创建盟军边描述符
        new("ALLIES", () => new AlliesSide(_ui)),
        // 创建日本边描述符
        new("JAPAN", () => new JapanSide(_ui)),
        // 创建德国边描述符
        new("GERMANY", () => new GermanySide(_ui)),
    };
}
```