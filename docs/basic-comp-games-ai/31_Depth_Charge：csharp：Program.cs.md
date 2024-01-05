# `31_Depth_Charge\csharp\Program.cs`

```
# 引入System命名空间
using System;

namespace DepthCharge
{
    class Program
    {
        static void Main(string[] args)
        {
            # 创建一个Random对象
            var random = new Random();

            # 调用View类的ShowBanner方法显示游戏横幅
            View.ShowBanner();

            # 调用Controller类的InputDimension方法获取游戏维度
            var dimension = Controller.InputDimension();
            # 调用CalculateMaximumGuesses方法计算最大猜测次数
            var maximumGuesses = CalculateMaximumGuesses();

            # 调用View类的ShowInstructions方法显示游戏说明
            View.ShowInstructions(maximumGuesses);

            # 显示游戏开始提示
            do
            {
                View.ShowStartGame();
                # 定义变量 submarineCoordinates，并调用 PlaceSubmarine() 函数来获取潜艇的坐标
                var submarineCoordinates = PlaceSubmarine();
                # 定义变量 trailNumber，并赋值为 1
                var trailNumber = 1;
                # 定义变量 guess，并赋值为 (0, 0, 0)
                var guess = (0, 0, 0);

                # 使用 do-while 循环来进行猜测游戏，直到猜中潜艇的位置或者达到最大猜测次数
                do
                {
                    # 调用 Controller.InputCoordinates() 函数来获取玩家的猜测坐标，并赋值给 guess
                    guess = Controller.InputCoordinates(trailNumber);
                    # 如果猜测的坐标不等于潜艇的坐标，则调用 View.ShowGuessPlacement() 函数显示猜测的位置
                    if (guess != submarineCoordinates)
                        View.ShowGuessPlacement(submarineCoordinates, guess);
                }
                while (guess != submarineCoordinates && trailNumber++ < maximumGuesses);  # 当猜测的坐标不等于潜艇的坐标并且猜测次数小于最大猜测次数时继续循环

                # 调用 View.ShowGameResult() 函数显示游戏结果
                View.ShowGameResult(submarineCoordinates, guess, trailNumber);
            }
            while (Controller.InputPlayAgain());  # 当玩家选择再玩一次时继续循环

            # 调用 View.ShowFarewell() 函数显示道别信息

            # 定义 CalculateMaximumGuesses() 函数来计算最大猜测次数
            int CalculateMaximumGuesses() =>
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制数据，并封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```