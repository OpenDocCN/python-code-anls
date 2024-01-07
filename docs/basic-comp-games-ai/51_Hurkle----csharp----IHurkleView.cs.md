# `basic-computer-games\51_Hurkle\csharp\IHurkleView.cs`

```

# 定义了一个名为hurkle的命名空间
namespace hurkle
{
    # 定义了一个名为IHurkleView的接口
    internal interface IHurkleView
    {
        # 定义了一个方法，用于获取猜测的游戏点
        GamePoint GetGuess(GuessViewModel guessViewModel);
        # 定义了一个方法，用于展示获胜的视图模型
        void ShowVictory(VictoryViewModel victoryViewModel);
        # 定义了一个方法，用于展示猜测错误的方向
        void ShowDirection(FailedGuessViewModel failedGuessViewModel);
        # 定义了一个方法，用于展示失败的视图模型
        void ShowLoss(LossViewModel lossViewModel);
    }
}

```