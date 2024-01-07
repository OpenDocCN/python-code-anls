# `basic-computer-games\31_Depth_Charge\csharp\View.cs`

```

// 命名空间 DepthCharge 包含了用于向用户显示信息的方法
namespace DepthCharge
{
    /// <summary>
    /// 包含了用于向用户显示信息的方法
    /// </summary>
    static class View
    {
        // 显示游戏横幅
        public static void ShowBanner()
        {
            Console.WriteLine("                             DEPTH CHARGE");
            Console.WriteLine("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
        }

        // 显示游戏说明
        public static void ShowInstructions(int maximumGuesses)
        {
            Console.WriteLine("YOU ARE THE CAPTAIN OF THE DESTROYER USS COMPUTER");
            Console.WriteLine("AN ENEMY SUB HAS BEEN CAUSING YOU TROUBLE.  YOUR");
            Console.WriteLine($"MISSION IS TO DESTROY IT.  YOU HAVE {maximumGuesses} SHOTS.");
            Console.WriteLine("SPECIFY DEPTH CHARGE EXPLOSION POINT WITH A");
            Console.WriteLine("TRIO OF NUMBERS -- THE FIRST TWO ARE THE");
            Console.WriteLine("SURFACE COORDINATES; THE THIRD IS THE DEPTH.");
            Console.WriteLine();
        }

        // 显示游戏开始信息
        public static void ShowStartGame()
        {
            Console.WriteLine("GOOD LUCK !");
            Console.WriteLine();
        }

        // 显示猜测的位置
        public static void ShowGuessPlacement((int x, int y, int depth) actual, (int x, int y, int depth) guess)
        {
            // 根据猜测的位置和实际位置显示结果
            // ...
            Console.WriteLine();
        }

        // 显示游戏结果
        public static void ShowGameResult((int x, int y, int depth) submarineLocation, (int x, int y, int depth) finalGuess, int trailNumber)
        {
            // 根据潜艇位置、最终猜测和尝试次数显示游戏结果
            // ...
        }

        // 显示告别信息
        public static void ShowFarewell()
        {
            Console.WriteLine ("OK.  HOPE YOU ENJOYED YOURSELF.");
        }

        // 显示无效数字提示
        public static void ShowInvalidNumber()
        {
            Console.WriteLine("PLEASE ENTER A NUMBER");
        }

        // 显示无效维度提示
        public static void ShowInvalidDimension()
        {
            Console.WriteLine("PLEASE ENTER A VALID DIMENSION");
        }

        // 显示坐标过少提示
        public static void ShowTooFewCoordinates()
        {
            Console.WriteLine("TOO FEW COORDINATES");
        }

        // 显示坐标过多提示
        public static void ShowTooManyCoordinates()
        {
            Console.WriteLine("TOO MANY COORDINATES");
        }

        // 显示无效的是或否提示
        public static void ShowInvalidYesOrNo()
        {
            Console.WriteLine("PLEASE ENTER Y OR N");
        }

        // 提示输入搜索区域的维度
        public static void PromptDimension()
        {
            Console.Write("DIMENSION OF SEARCH AREA? ");
        }

        // 提示输入猜测
        public static void PromptGuess(int trailNumber)
        {
            Console.WriteLine();
            Console.Write($"TRIAL #{trailNumber}? ");
        }

        // 提示是否再玩一局
        public static void PromptPlayAgain()
        {
            Console.WriteLine();
            Console.Write("ANOTHER GAME (Y OR N)? ");
        }
    }
}

```