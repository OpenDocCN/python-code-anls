# `basic-computer-games\12_Bombs_Away\csharp\BombsAwayConsole\Program.cs`

```
/// 使用 ConsoleUserInterface 创建并玩游戏
PlayGameWhileUserWantsTo(new ConsoleUserInterface());

void PlayGameWhileUserWantsTo(ConsoleUserInterface ui)
{
    // 当用户想要继续玩游戏时
    do
    {
        // 创建新的游戏并进行游戏
        new Game(ui).Play();
    }
    while (UserWantsToPlayAgain(ui));
}

// 判断用户是否想再玩一次游戏
bool UserWantsToPlayAgain(IUserInterface ui)
{
    // 用户选择是否再玩一次游戏
    bool result = ui.ChooseYesOrNo("ANOTHER MISSION (Y OR N)?");
    // 如果用户选择不再玩游戏
    if (!result)
    {
        // 输出提示信息
        Console.WriteLine("CHICKEN !!!");
    }

    return result;
}
```