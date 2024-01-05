# `d:/src/tocomm/basic-computer-games\60_Mastermind\csharp\Program.cs`

```
# 导入所需的模块
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
            // 初始化计算机得分
            var computerScore = 0;

            // 循环游戏回合
            for (var round = 1; round <= rounds; ++round)
            {
                // 显示回合开始信息
                View.ShowStartOfRound(round);

                // 如果玩家不能进行回合，则结束游戏
                if (!HumanTakesTurn())
                    return;

                // 当计算机不能进行回合时，显示信息提示
                while (!ComputerTakesTurn())
                    View.ShowInconsistentInformation();
            }

            // 显示最终得分
            View.ShowScores(humanScore, computerScore, isFinal: true);

            /// <summary>
            /// 从用户获取游戏开始参数
            /// </summary>
            (CodeFactory codeFactory, int rounds) StartGame()
                View.ShowBanner(); // 显示游戏横幅

                var colors    = Controller.GetNumberOfColors(); // 获取颜色数量
                var positions = Controller.GetNumberOfPositions(); // 获取位置数量
                var rounds    = Controller.GetNumberOfRounds(); // 获取回合数

                var codeFactory = new CodeFactory(positions, colors); // 创建一个代码工厂对象，传入位置和颜色数量

                View.ShowTotalPossibilities(codeFactory.Possibilities); // 显示总的可能性数量
                View.ShowColorTable(codeFactory.Colors); // 显示颜色表

                return (codeFactory, rounds); // 返回代码工厂对象和回合数
            }

            /// <summary>
            /// Executes the human's turn. // 执行玩家的回合
            /// </summary>
            /// <returns>
            /// True if thue human completed his or her turn and false if
            /// he or she quit the game. // 如果玩家完成了回合则返回true，如果退出游戏则返回false
            /// <summary>
            /// 人类玩家进行回合
            /// </summary>
            /// <returns>回合是否成功</returns>
            bool HumanTakesTurn()
            {
                // 存储人类猜测的历史记录（用于下面的显示面板命令）
                var history     = new List<TurnResult>();
                // 创建一个代码实例
                var code        = codeFactory.Create(random);
                // 猜测次数
                var guessNumber = default(int);

                for (guessNumber = 1; guessNumber <= MaximumGuesses; ++guessNumber)
                {
                    var guess = default(Code);

                    // 当猜测为空时，继续循环
                    while (guess is null)
                    {
                        // 获取人类输入的命令
                        switch (Controller.GetCommand(guessNumber, codeFactory.Positions, codeFactory.Colors))
                        {
                            // 如果是猜测命令，将输入的代码赋给guess
                            case (Command.MakeGuess, Code input):
                                guess = input;
                                break;
                            case (Command.ShowBoard, _):  # 当命令为显示游戏板时
                                View.ShowBoard(history);  # 调用视图的显示游戏板方法，传入历史记录参数
                                break;  # 结束当前的 case 分支
                            case (Command.Quit, _):  # 当命令为退出游戏时
                                View.ShowQuitGame(code);  # 调用视图的显示退出游戏方法，传入代码参数
                                return false;  # 返回 false，表示游戏结束
                        }
                    }

                    var (blacks, whites) = code.Compare(guess);  # 使用代码对象的比较方法，比较猜测和代码，得到黑白猜中数量
                    if (blacks == codeFactory.Positions)  # 如果猜中的黑色数量等于代码的位置数量
                        break;  # 退出循环

                    View.ShowResults(blacks, whites);  # 调用视图的显示结果方法，传入黑白猜中数量

                    history.Add(new TurnResult(guess, blacks, whites));  # 将当前猜测和猜中数量添加到历史记录中

                }

                if (guessNumber <= MaximumGuesses)  # 如果猜测次数小于等于最大猜测次数
                    View.ShowHumanGuessedCode(guessNumber);  # 调用视图的显示玩家猜测代码方法，传入猜测次数参数
                else
                    View.ShowHumanFailedToGuessCode(code);  # 如果猜测失败，显示人类猜测失败的消息

                humanScore += guessNumber;  # 将猜测的数字加到人类得分上

                View.ShowScores(humanScore, computerScore, isFinal: false);  # 显示分数，isFinal参数为false
                return true;  # 返回true表示计算机完成了其回合
            }

            /// <summary>
            /// Executes the computers turn.
            /// </summary>
            /// <returns>
            /// True if the computer completes its turn successfully and false
            /// if it does not (due to human error).
            /// </returns>
            bool ComputerTakesTurn()  # 定义计算机进行回合的函数
            {
                var isCandidate = new bool[codeFactory.Possibilities];  # 创建一个布尔数组，用于存储可能的猜测
                var guessNumber = default(int);  # 初始化一个整数变量用于存储猜测的数字
                // 将isCandidate数组中的所有元素设置为true
                Array.Fill(isCandidate, true);

                // 显示计算机开始回合的界面
                View.ShowComputerStartTurn();
                // 控制器等待直到准备就绪
                Controller.WaitUntilReady();

                // 从1开始循环猜测，直到达到最大猜测次数为止
                for (guessNumber = 1; guessNumber <= MaximumGuesses; ++guessNumber)
                {
                    // 从一个随机的代码开始，循环遍历代码，直到找到仍然是候选解的代码为止。
                    // 如果没有剩余的候选解，则意味着用户在一个或多个响应中出错。
                    var codeNumber = EnumerableExtensions.Cycle(random.Next(codeFactory.Possibilities), codeFactory.Possibilities)
                        .FirstOrDefault(i => isCandidate[i], -1);

                    // 如果codeNumber小于0，则返回false
                    if (codeNumber < 0)
                        return false;

                    // 根据codeNumber创建一个猜测
                    var guess = codeFactory.Create(codeNumber);
                    // 调用 Controller 的 GetBlacksWhites 方法，获取猜测的黑白棋子数量
                    var (blacks, whites) = Controller.GetBlacksWhites(guess);
                    // 如果猜测的黑白棋子数量等于答案的位置数量，跳出循环
                    if (blacks == codeFactory.Positions)
                        break;

                    // 标记不再是潜在解决方案的代码。我们知道当前猜测与解决方案相比产生了上述数量的黑白棋子，因此任何产生不同数量的黑白棋子的代码都不可能是答案。
                    foreach (var (candidate, index) in codeFactory.EnumerateCodes().Select((candidate, index) => (candidate, index)))
                    {
                        // 如果是潜在解决方案
                        if (isCandidate[index])
                        {
                            // 比较猜测和候选代码的黑白棋子数量
                            var (candidateBlacks, candidateWhites) = guess.Compare(candidate);
                            // 如果黑白棋子数量不等于猜测的数量，将该候选代码标记为不再是潜在解决方案
                            if (blacks != candidateBlacks || whites != candidateWhites)
                                isCandidate[index] = false;
                        }
                    }
                }
                if (guessNumber <= MaximumGuesses)  # 如果猜测次数小于等于最大猜测次数
                    View.ShowComputerGuessedCode(guessNumber);  # 调用视图方法展示计算机猜测的代码和次数
                else  # 否则
                    View.ShowComputerFailedToGuessCode();  # 调用视图方法展示计算机未能猜测到代码

                computerScore += guessNumber;  # 计算机得分增加猜测次数
                View.ShowScores(humanScore, computerScore, isFinal: false);  # 调用视图方法展示得分，不是最终得分

                return true;  # 返回true表示计算机猜测完成
            }
        }
    }
}
```