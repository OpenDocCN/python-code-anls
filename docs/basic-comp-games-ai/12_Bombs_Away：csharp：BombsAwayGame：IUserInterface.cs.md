# `d:/src/tocomm/basic-computer-games\12_Bombs_Away\csharp\BombsAwayGame\IUserInterface.cs`

```
// 命名空间声明
namespace BombsAwayGame;

/// <summary>
/// 代表为游戏提供数据的接口。
/// </summary>
/// <remarks>
/// 抽象化用户界面允许我们集中其关注点在代码的一个部分，并且在不创建任何改变游戏逻辑的风险的情况下改变用户界面行为。
/// 它还允许我们为测试提供自动化用户界面。
/// </remarks>
// 用户界面接口声明
public interface IUserInterface
{
    /// <summary>
    /// 显示给定的消息。
    /// </summary>
    /// <param name="message">要显示的消息。</param>
    // 输出方法声明
    void Output(string message);

    /// <summary>
    /// 从给定的选项中选择一个项目。
    /// </summary>
    // 选择方法声明
    /// <param name="message">要显示的消息。</param>
    /// <param name="choices">可供选择的选项。</param>
    /// <returns>用户选择的选项在 <paramref name="choices"/> 中的索引。</returns>
    int Choose(string message, IList<string> choices);

    /// <summary>
    /// 允许用户选择是或否。
    /// </summary>
    /// <param name="message">要显示的消息。</param>
    /// <returns>如果用户选择是，则为 true，如果用户选择否，则为 false。</returns>
    bool ChooseYesOrNo(string message);

    /// <summary>
    /// 从用户获取整数。
    /// </summary>
    /// <returns>用户提供的整数。</returns>
    int InputInteger();
}
```