# `basic-computer-games\12_Bombs_Away\csharp\BombsAwayConsole\Program.cs`

```

// 使用控制台用户界面创建和玩游戏
PlayGameWhileUserWantsTo(new ConsoleUserInterface());

// 当用户想要玩游戏时进行游戏
void PlayGameWhileUserWantsTo(ConsoleUserInterface ui)
{
    do
    {
        // 创建新的游戏并进行游戏
        new Game(ui).Play();
    }
    // 当用户想要再玩一次时继续进行游戏
    while (UserWantsToPlayAgain(ui));
}

// 检查用户是否想再玩一次游戏
bool UserWantsToPlayAgain(IUserInterface ui)
{
    // 用户选择是否再玩一次游戏
    bool result = ui.ChooseYesOrNo("ANOTHER MISSION (Y OR N)?");
    // 如果用户选择不再玩一次游戏，则输出"CHICKEN !!!"
    if (!result)
    {
        Console.WriteLine("CHICKEN !!!");
    }
    // 返回用户选择的结果
    return result;
}

```