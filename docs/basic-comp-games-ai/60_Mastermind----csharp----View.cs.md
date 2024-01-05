# `60_Mastermind\csharp\View.cs`

```
# 引入所需的模块
import zipfile
from io import BytesIO

# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
        # 显示总的可能性数量
        public static void ShowTotalPossibilities(int possibilities)
        {
            Console.WriteLine($"TOTAL POSSIBILITIES = {possibilities}");
            Console.WriteLine();
        }

        # 显示颜色表
        public static void ShowColorTable(int numberOfColors)
        {
            Console.WriteLine();
            Console.WriteLine("COLOR     LETTER")
            Console.WriteLine("=====     ======")

            # 遍历颜色列表，显示颜色的长名和短名
            foreach (var color in Colors.List.Take(numberOfColors))
                Console.WriteLine($"{color.LongName,-13}{color.ShortName}")

            Console.WriteLine()
        }

        # 显示回合开始
        public static void ShowStartOfRound(int roundNumber)
            Console.WriteLine();
            // 打印当前轮数
            Console.WriteLine($"ROUND NUMBER {roundNumber} ----");
            Console.WriteLine();
            Console.WriteLine("GUESS MY COMBINATION.");
            Console.WriteLine();
        }

        public static void ShowBoard(IEnumerable<TurnResult> history)
        {
            Console.WriteLine();
            Console.WriteLine("BOARD");
            Console.WriteLine("MOVE     GUESS          BLACK     WHITE");

            var moveNumber = 0;
            // 遍历历史记录，打印每一步的猜测结果
            foreach (var result in history)
                Console.WriteLine($"{++moveNumber,-9}{result.Guess,-16}{result.Blacks,-10}{result.Whites}");

            Console.WriteLine();
        }
        # 显示退出游戏信息，打印出玩家未能猜中的组合
        public static void ShowQuitGame(Code code)
        {
            Console.WriteLine($"QUITTER!  MY COMBINATION WAS: {code}");
            Console.WriteLine("GOOD BYE");
        }

        # 显示游戏结果，打印出玩家猜中的黑色和白色数量
        public static void ShowResults(int blacks, int whites)
        {
            Console.WriteLine($"YOU HAVE  {blacks}  BLACKS AND  {whites}  WHITES.");
        }

        # 显示玩家猜中组合的次数
        public static void ShowHumanGuessedCode(int guessNumber)
        {
            Console.WriteLine($"YOU GUESSED IT IN  {guessNumber}  MOVES!");
        }

        # 显示玩家未能猜中组合的信息，打印出正确的组合
        public static void ShowHumanFailedToGuessCode(Code code)
        {
            // 注意：原始代码没有打印出组合，但这似乎是一个错误。
            Console.WriteLine("YOU RAN OUT OF MOVES!  THAT'S ALL YOU GET!");  # 打印消息，提示玩家已经用完了所有的移动次数
            Console.WriteLine($"THE ACTUAL COMBINATION WAS: {code}");  # 打印消息，显示实际的组合代码
        }

        public static void ShowScores(int humanScore, int computerScore, bool isFinal)
        {
            if (isFinal)
            {
                Console.WriteLine("GAME OVER");  # 如果是最终得分，打印游戏结束的消息
                Console.WriteLine("FINAL SCORE:");  # 打印最终得分的消息
            }
            else
                Console.WriteLine("SCORE:");  # 如果不是最终得分，打印得分的消息

            Console.WriteLine($"     COMPUTER  {computerScore}");  # 打印计算机的得分
            Console.WriteLine($"     HUMAN     {humanScore}");  # 打印玩家的得分
            Console.WriteLine();
        }

        public static void ShowComputerStartTurn()
        {
            Console.WriteLine("NOW I GUESS.  THINK OF A COMBINATION.");
        }
```
这段代码是一个方法，用于在控制台输出一条消息，提示用户现在轮到计算机猜测密码了。

```
        public static void ShowInconsistentInformation()
        {
            Console.WriteLine("YOU HAVE GIVEN ME INCONSISTENT INFORMATION.");
            Console.WriteLine("TRY AGAIN, AND THIS TIME PLEASE BE MORE CAREFUL.");
        }
```
这段代码是一个方法，用于在控制台输出两条消息，提示用户输入的信息不一致，要求用户重新输入并更加仔细。

```
        public static void ShowComputerGuessedCode(int guessNumber)
        {
            Console.WriteLine($"I GOT IT IN  {guessNumber}  MOVES!");
        }
```
这段代码是一个方法，用于在控制台输出一条消息，提示计算机猜测密码成功，并显示猜测所用的次数。

```
        public static void ShowComputerFailedToGuessCode()
        {
            Console.WriteLine("I USED UP ALL MY MOVES!");
            Console.WriteLine("I GUESS MY CPU IS JUST HAVING AN OFF DAY.");
        }
```
这段代码是一个方法，用于在控制台输出两条消息，提示计算机猜测密码失败，并显示计算机已经用尽所有的猜测次数，以及计算机可能出现了问题。
# 提示用户输入颜色的数量
public static void PromptNumberOfColors()
{
    Console.Write("NUMBER OF COLORS? ");
}

# 提示用户输入位置的数量
public static void PromptNumberOfPositions()
{
    Console.Write("NUMBER OF POSITIONS? ");
}

# 提示用户输入回合的数量
public static void PromptNumberOfRounds()
{
    Console.Write("NUMBER OF ROUNDS? ");
}

# 提示用户输入猜测的颜色和位置
public static void PromptGuess(int moveNumber)
{
    Console.Write($"MOVE #  {moveNumber}  GUESS ? ");
}
# 提示用户准备好后按回车键
public static void PromptReady()
{
    Console.Write("HIT RETURN WHEN READY ? ");
}

# 提示用户猜测的代码，并要求输入黑白猜测结果
public static void PromptBlacksWhites(Code code)
{
    Console.Write($"MY GUESS IS: {code}");
    Console.Write("  BLACKS, WHITES ? ");
}

# 提示用户输入两个值，用逗号分隔
public static void PromptTwoValues()
{
    Console.WriteLine("PLEASE ENTER TWO VALUES, SEPARATED BY A COMMA");
}

# 提示用户输入一个有效的整数值
public static void PromptValidInteger()
{
    Console.WriteLine("PLEASE ENTER AN INTEGER VALUE");
}
        }  # 结束 NotifyTooManyColors 方法的定义

        public static void NotifyBadNumberOfPositions()
        {
            Console.WriteLine("BAD NUMBER OF POSITIONS");  # 打印错误消息：位置数量错误
        }

        public static void NotifyInvalidColor(char colorKey)
        {
            Console.WriteLine($"'{colorKey}' IS UNRECOGNIZED.");  # 打印错误消息：颜色不被识别
        }

        public static void NotifyTooManyColors(int maxColors)
        {
            Console.WriteLine($"NO MORE THAN {maxColors}, PLEASE!");  # 打印错误消息：请不要超过最大颜色数量
        }
    }
}
```