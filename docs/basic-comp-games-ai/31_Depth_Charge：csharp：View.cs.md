# `31_Depth_Charge\csharp\View.cs`

```
// 使用System命名空间
using System;

namespace DepthCharge
{
    /// <summary>
    /// 包含用于向用户显示信息的方法。
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
            // 打印游戏开始的提示信息
            Console.WriteLine("YOU ARE THE CAPTAIN OF THE DESTROYER USS COMPUTER");
            Console.WriteLine("AN ENEMY SUB HAS BEEN CAUSING YOU TROUBLE.  YOUR");
            Console.WriteLine($"MISSION IS TO DESTROY IT.  YOU HAVE {maximumGuesses} SHOTS.");
            Console.WriteLine("SPECIFY DEPTH CHARGE EXPLOSION POINT WITH A");
            Console.WriteLine("TRIO OF NUMBERS -- THE FIRST TWO ARE THE");
            Console.WriteLine("SURFACE COORDINATES; THE THIRD IS THE DEPTH.");
            Console.WriteLine();
        }

        public static void ShowStartGame()
        {
            // 打印游戏开始的祝福信息
            Console.WriteLine("GOOD LUCK !");
            Console.WriteLine();
        }

        public static void ShowGuessPlacement((int x, int y, int depth) actual, (int x, int y, int depth) guess)
        {
            // 打印猜测的位置信息
            Console.Write("SONAR REPORTS SHOT WAS ");
            if (guess.y > actual.y)
                Console.Write("NORTH");
```
```csharp
            // 如果猜测的 y 坐标大于实际的 y 坐标，则打印 "NORTH"
            # 如果猜测的 y 坐标小于实际的 y 坐标，则输出 "SOUTH"
            if (guess.y < actual.y)
                Console.Write("SOUTH");
            # 如果猜测的 x 坐标大于实际的 x 坐标，则输出 "EAST"
            if (guess.x > actual.x)
                Console.Write("EAST");
            # 如果猜测的 x 坐标小于实际的 x 坐标，则输出 "WEST"
            if (guess.x < actual.x)
                Console.Write("WEST");
            # 如果猜测的 y 坐标不等于实际的 y 坐标 或者 猜测的 x 坐标不等于实际的 x 坐标，则输出 " AND"
            if (guess.y != actual.y || guess.x != actual.y)
                Console.Write(" AND");
            # 如果猜测的深度大于实际的深度，则输出 " TOO LOW."
            if (guess.depth > actual.depth)
                Console.Write (" TOO LOW.");
            # 如果猜测的深度小于实际的深度，则输出 " TOO HIGH."
            if (guess.depth < actual.depth)
                Console.Write(" TOO HIGH.");
            # 如果猜测的深度等于实际的深度，则输出 " DEPTH OK."
            if (guess.depth == actual.depth)
                Console.Write(" DEPTH OK.");

            # 输出换行符
            Console.WriteLine();
        }

        # 显示游戏结果
        public static void ShowGameResult((int x, int y, int depth) submarineLocation, (int x, int y, int depth) finalGuess, int trailNumber)
        {
            Console.WriteLine();  // 输出空行

            if (submarineLocation == finalGuess)  // 如果潜艇位置等于最终猜测位置
            {
                Console.WriteLine($"B O O M ! ! YOU FOUND IT IN {trailNumber} TRIES!");  // 输出找到潜艇的消息和尝试次数
            }
            else
            {
                Console.WriteLine("YOU HAVE BEEN TORPEDOED!  ABANDON SHIP!");  // 输出被击中的消息
                Console.WriteLine($"THE SUBMARINE WAS AT {submarineLocation.x}, {submarineLocation.y}, {submarineLocation.depth}");  // 输出潜艇的真实位置
            }
        }

        public static void ShowFarewell()  // 显示道别消息
        {
            Console.WriteLine ("OK.  HOPE YOU ENJOYED YOURSELF.");  // 输出道别消息
        }

        public static void ShowInvalidNumber()  // 显示无效数字消息
        {
            Console.WriteLine("PLEASE ENTER A NUMBER");
```
这行代码用于在控制台输出提示信息"PLEASE ENTER A NUMBER"。

```python
        public static void ShowInvalidDimension()
```
这行代码定义了一个名为ShowInvalidDimension的函数，用于在控制台输出提示信息"PLEASE ENTER A VALID DIMENSION"。

```python
        public static void ShowTooFewCoordinates()
```
这行代码定义了一个名为ShowTooFewCoordinates的函数，用于在控制台输出提示信息"TOO FEW COORDINATES"。

```python
        public static void ShowTooManyCoordinates()
```
这行代码定义了一个名为ShowTooManyCoordinates的函数，用于在控制台输出提示信息"TOO MANY COORDINATES"。

```python
        public static void ShowInvalidYesOrNo()
```
这行代码定义了一个名为ShowInvalidYesOrNo的函数，用于在控制台输出提示信息"INVALID YES OR NO"。
# 输出提示信息，要求用户输入 Y 或 N
Console.WriteLine("PLEASE ENTER Y OR N");

# 输出提示信息，要求用户输入搜索区域的维度
public static void PromptDimension()
{
    Console.Write("DIMENSION OF SEARCH AREA? ");
}

# 输出提示信息，要求用户输入猜测的次数
public static void PromptGuess(int trailNumber)
{
    Console.WriteLine();
    Console.Write($"TRIAL #{trailNumber}? ");
}

# 输出提示信息，询问用户是否再玩一局游戏
public static void PromptPlayAgain()
{
    Console.WriteLine();
    Console.Write("ANOTHER GAME (Y OR N)? ");
}
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，确保不会造成内存泄漏。
```