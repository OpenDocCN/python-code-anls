# `39_Golf\csharp\Program.cs`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制数据，并封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
// Windows compile:
// 在 Windows 平台上编译程序
// 使用 dotnet 命令发布，并指定目标平台为 win-x64，发布模式为 Release，同时设置单文件发布为 true

// INDEX
// ----------------- methods
// constructor
// NewHole
// TeeUp
// Stroke
// PlotBall
// InterpretResults
// ReportCurrentScore
// FindBall
// IsOnFairway
// IsOnGreen
// IsInHazard
// IsInRough
// IsOutOfBounds
// ScoreCardNewHole
// 上面是程序中的方法列表，包括构造函数和各种操作方法
// ScoreCardRecordStroke - 记录一杆高尔夫球的击球数
// ScoreCardGetPreviousStroke - 获取上一杆高尔夫球的击球数
// ScoreCardGetTotal - 获取总的高尔夫球击球数
// Ask - 提出询问
// Wait - 等待
// ReviewBag - 检查袋子
// Quit - 退出
// GameOver - 游戏结束
// ----------------- DATA
// Clubs - 球杆信息
// CourseInfo - 球场信息
// ----------------- classes
// HoleInfo - 洞信息
// CircleGameObj - 圆形游戏对象
// RectGameObj - 矩形游戏对象
// HoleGeometry - 洞的几何信息
// Plot - 绘图
// ----------------- helper methods
// GetDistance - 获取距离
// IsInRectangle - 是否在矩形内
// ToRadians
// 将角度转换为弧度

// ToDegrees360
// 将弧度转换为360度制的角度

// Odds
// 赔率

// 尽管这是一个基于文本的游戏，但代码使用简单的几何学来模拟球场。
// 球道是宽40码的矩形，周围有5码的粗糙区域。
// 球洞周围是半径为10码的圆形的果岭。
// 球洞始终在点(0,0)处。

// 使用基本的三角法，我们可以根据击球的距离和偏离角度（勾钩/切球）来绘制球的位置。

// 击球距离基于不同球杆类型的真实世界平均值。
// 大量的随机化，"业务规则"和运气影响游戏玩法。
// 概率在代码中有注释说明。

// 注意：'courseInfo'、'clubs'和'scoreCard'数组每个都包括一个空对象，以便从1开始索引。
// 像所有优秀的程序员一样，我们从零开始计数，但在这种情况下，当第一洞在索引一处时，更自然的是从一开始计数。
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制数据，并封装成字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建一个 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名列表，读取每个文件的数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
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
//  The cup is always at point: 0,0.
//  We use atan2 to compute the angle between the cup and the ball.
//  Setting the cup's vector to 0,-1 on a 360 circle is equivalent to:
//  0 deg = 12 o'clock;  90 deg = 3 o'clock;  180 deg = 6 o'clock;  270 = 9 o'clock
//  The reverse angle between the cup and the ball is a difference of PI (using radians).
//
//  Given the angle and stroke distance (hypotenuse), we use cosine to compute
//  the opposite and adjacent sides of the triangle, which, is the ball's new position.
//
//           0
//           |
//    270 - cup - 90
//           |
//          180
//
//
//          cup
//           |
//           |
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制数据，并封装成字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
//  www.bonthron.com
//  for my father, Raymond Bonthron, an avid golfer
//
//  Inspired by the 1978 "Golf" from "Basic Computer Games"
//  by Steve North, who modified an existing golf game by an unknown author
//
//
```
这部分注释是关于程序的作者和灵感来源的说明。

```python
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading;
```
这些语句是引入所需的命名空间，以便在程序中使用相应的类和方法。

```python
namespace Golf
{
    using Ball = Golf.CircleGameObj;
    using Hazard = Golf.CircleGameObj;
```
这是定义了一个名为Golf的命名空间，并且给Ball和Hazard类起了别名，使得在程序中可以使用别名来代替完整的类名。

```python
    // --------------------------------------------------------------------------- Program
```
这是一个注释，用于标识程序的开始。
    # 创建一个名为 Program 的类
    class Program
    {
        # 创建一个名为 Main 的静态方法，接受一个参数 args
        static void Main(string[] args)
        {
            # 创建一个 Golf 类的实例对象 g
            Golf g = new Golf();
        }
    }


    // --------------------------------------------------------------------------- Golf
    # 创建一个名为 Golf 的公共类
    public class Golf
    {
        # 声明一个名为 BALL 的成员变量
        Ball BALL;
        # 声明一个名为 HOLE_NUM 的成员变量并初始化为 0
        int HOLE_NUM = 0;
        # 声明一个名为 STROKE_NUM 的成员变量并初始化为 0
        int STROKE_NUM = 0;
        # 声明一个名为 Handicap 的成员变量并初始化为 0
        int Handicap = 0;
        # 声明一个名为 PlayerDifficulty 的成员变量并初始化为 0
        int PlayerDifficulty = 0;
        # 声明一个名为 holeGeometry 的成员变量
        HoleGeometry holeGeometry;

        // all fairways are 40 yards wide, extend 5 yards beyond the cup, and
        # 注释说明 fairways 的宽度为 40 码，超出杯口 5 码
        // 定义球道的宽度为40码
        const int FairwayWidth = 40;
        // 定义球道延伸长度为5码
        const int FairwayExtension = 5;
        // 定义粗糙地的宽度为5码
        const int RoughAmt = 5;

        // ScoreCard 记录每一杆击球后球的位置
        // 每个球洞都有一个新的列表
        // 包括一个空列表，使得索引1等于球洞1
        List<List<Ball>> ScoreCard = new List<List<Ball>> { new List<Ball>() };

        // 定义一个用于输出的函数
        static void w(string s) { Console.WriteLine(s); } // WRITE
        // 创建一个随机数生成器
        Random RANDOM = new Random();


        // --------------------------------------------------------------- 构造函数
        public Golf()
        {
            Console.Clear();
            w(" ");
            # 打印文本内容
            w("          8\"\"\"\"8 8\"\"\"88 8     8\"\"\"\" ");
            w("          8    \" 8    8 8     8     ");
            w("          8e     8    8 8e    8eeee ");
            w("          88  ee 8    8 88    88    ");
            w("          88   8 8    8 88    88    ");
            w("          88eee8 8eeee8 88eee 88    ");
            w(" ");
            w("Welcome to the Creative Computing Country Club,");
            w("an eighteen hole championship layout located a short");
            w("distance from scenic downtown Lambertville, New Jersey.");
            w("The game will be explained as you play.");
            w("Enjoy your game! See you at the 19th hole...");
            w(" ");
            w("Type QUIT at any time to leave the game.");
            w("Type BAG at any time to review the clubs in your bag.");
            w(" ");

            # 等待用户输入
            Wait((z) =>
            {
                w(" ");
                # 输出字符串 "YOUR BAG" 到控制台
                w("              YOUR BAG");
                # 调用 ReviewBag() 函数
                ReviewBag();
                # 输出字符串 "Type BAG at any time to review the clubs in your bag." 到控制台
                w("Type BAG at any time to review the clubs in your bag.");
                # 输出空字符串到控制台
                w(" ");

                # 等待用户输入
                Wait((zz) =>
                {
                    # 输出空字符串到控制台
                    w(" ");

                    # 提示用户输入 PGA 等级，范围为 0 到 30
                    Ask("PGA handicaps range from 0 to 30.\nWhat is your handicap?", 0, 30, (i) =>
                    {
                        # 将用户输入的 PGA 等级赋值给 Handicap 变量
                        Handicap = i;
                        # 输出空字符串到控制台
                        w(" ");

                        # 提示用户选择常见的高尔夫困难，选项为 1 到 5
                        Ask("Common difficulties at golf include:\n1=Hook, 2=Slice, 3=Poor Distance, 4=Trap Shots, 5=Putting\nWhich one is your worst?", 1, 5, (j) =>
                        {
                            # 将用户选择的困难赋值给 PlayerDifficulty 变量
                            PlayerDifficulty = j;
                            # 清空控制台
                            Console.Clear();
                            # 调用 NewHole() 函数
                            NewHole();
                        });
                    });
                });
            });
        }
```
这部分代码是一个函数的结束标志，表示函数NewHole的结束。

```
        // --------------------------------------------------------------- NewHole
```
这是一个注释，用于标识下面的函数是一个新的洞（高尔夫球场的洞）。

```
        void NewHole()
```
这是一个函数的声明，表示定义了一个名为NewHole的函数，该函数没有返回值（void）。

```
        {
```
这是函数体的开始标志。

```
            HOLE_NUM++;
            STROKE_NUM = 0;
```
这两行代码分别将变量HOLE_NUM和STROKE_NUM的值分别加1和设为0。

```
            HoleInfo info = CourseInfo[HOLE_NUM];
```
这行代码声明了一个名为info的变量，类型为HoleInfo，用于存储CourseInfo中索引为HOLE_NUM的元素。

```
            int yards = info.Yards;  // from tee to cup
            int par = info.Par;
```
这两行代码分别将info中的Yards和Par属性的值分别赋给变量yards和par。

```
            var cup = new CircleGameObj(0, 0, 0, GameObjType.CUP);
            var green = new CircleGameObj(0, 0, 10, GameObjType.GREEN);
```
这两行代码分别创建了名为cup和green的CircleGameObj对象，用于表示球洞的杯和果岭。

```
            var fairway = new RectGameObj(0 - (FairwayWidth / 2),
```
这行代码创建了名为fairway的RectGameObj对象，用于表示球道。
            // 创建一个矩形游戏对象，代表果岭
            var green = new RectGameObj(cup.X - green.Radius,
                                        cup.Y - green.Radius,
                                        2 * green.Radius,
                                        2 * green.Radius,
                                        GameObjType.GREEN);

            // 创建一个矩形游戏对象，代表球道
            var fairway = new RectGameObj(green.X - FairwayExtension,
                                          0 - (green.Radius + FairwayExtension),
                                          FairwayWidth,
                                          yards + (green.Radius + FairwayExtension) + 1,
                                          GameObjType.FAIRWAY);

            // 创建一个矩形游戏对象，代表粗糙地形
            var rough = new RectGameObj(fairway.X - RoughAmt,
                                        fairway.Y - RoughAmt,
                                        fairway.Width + (2 * RoughAmt),
                                        fairway.Length + (2 * RoughAmt),
                                        GameObjType.ROUGH);

            // 创建一个球对象，代表球
            BALL = new Ball(0, yards, 0, GameObjType.BALL);

            // 开始新的一轮比赛
            ScoreCardStartNewHole();

            // 创建一个HoleGeometry对象，代表球场的几何形状
            holeGeometry = new HoleGeometry(cup, green, fairway, rough, info.Hazard);

            // 输出当前球洞的编号
            w("                |> " + HOLE_NUM);
            // 输出空行
            w("                |        ");
            // 输出空行
            w("                |        ");
            w("          ^^^^^^^^^^^^^^^");  # 打印分隔线

            Console.WriteLine("Hole #{0}. You are at the tee. Distance {1} yards, par {2}.", HOLE_NUM, info.Yards, info.Par);  # 打印当前球洞信息
            w(info.Description);  # 打印球洞描述

            TeeUp();  # 调用 TeeUp() 函数，准备击球
        }


        // --------------------------------------------------------------- TeeUp
        // on the green? automatically select putter
        // otherwise Ask club and swing strength

        void TeeUp()
        {
            if (IsOnGreen(BALL) && !IsInHazard(BALL, GameObjType.SAND))  # 如果球在果岭上且不在沙坑中
            {
                var putt = 10;  # 设置推杆距离为 10 码
                w("[PUTTER: average 10 yards]");  # 打印推杆信息
                var msg = Odds(20) ? "Keep your head down.\n" : "";  # 根据概率生成提示信息
                // 询问用户选择推杆力量，并根据用户输入的值执行相应的操作
                Ask(msg + "Choose your putt potency. (1-10)", 1, 10, (strength) =>
                {
                    // 根据用户选择的推杆，获取相应的俱乐部信息
                    var putter = Clubs[putt];
                    // 根据用户选择的推杆和力量，执行推杆动作
                    Stroke(Convert.ToDouble((double)putter.Item2 * ((double)strength / 10.0)), putt);
                });
            }
            else
            {
                // 询问用户选择球杆，并根据用户输入的值执行相应的操作
                Ask("What club do you choose? (1-10)", 1, 10, (c) =>
                {
                    // 根据用户选择的球杆，获取相应的俱乐部信息
                    var club = Clubs[c];

                    // 打印俱乐部的平均码数
                    w(" ");
                    Console.WriteLine("[{0}: average {1} yards]", club.Item1.ToUpper(), club.Item2);

                    // 询问用户选择挥杆力量，并根据用户输入的值执行相应的操作
                    Ask("Now gauge your distance by a percentage of a full swing. (1-10)", 1, 10, (strength) =>
                    {
                        // 根据用户选择的球杆和力量，执行挥杆动作
                        Stroke(Convert.ToDouble((double)club.Item2 * ((double)strength / 10.0)), c);
                    });
// -------------------------------------------------------- bitwise Flags
// 使用二进制表示不同的标志位，每个变量代表一个特定的标志位
int dub         = 0b00000000000001;  // 双杆
int hook        = 0b00000000000010;  // 钩球
int slice       = 0b00000000000100;  // 切球
int passedCup   = 0b00000000001000;  // 球洞已过
int inCup       = 0b00000000010000;  // 球在洞内
int onFairway   = 0b00000000100000;  // 在球道上
int onGreen     = 0b00000001000000;  // 在果岭上
int inRough     = 0b00000010000000;  // 在草丛中
int inSand      = 0b00000100000000;  // 在沙坑中
int inTrees     = 0b00001000000000;  // 在树丛中
int inWater     = 0b00010000000000;  // 在水中
int outOfBounds = 0b00100000000000;  // 球出界
int luck        = 0b01000000000000;  // 幸运
int ace         = 0b10000000000000;  // 神奇一杆
        // --------------------------------------------------------------- Stroke
        // Stroke function to handle each golf stroke
        void Stroke(double clubAmt, int clubIndex)
        {
            // Increment the stroke number
            STROKE_NUM++;

            // Initialize flags variable with binary value 000000000000
            var flags = 0b000000000000;

            // Display "Fore!" message if it's the first stroke and the club amount is greater than 210 yards with a 30% chance
            if ((STROKE_NUM == 1) && (clubAmt > 210) && Odds(30)) { w("\"...Fore !\""); };

            // Set the dub flag if there's a 5% chance of dubbing it
            if (Odds(5)) { flags |= dub; }; 

            // Check if the ball is in the rough or sand, and the club used is not a wedge
            if ((IsInRough(BALL) || IsInHazard(BALL, GameObjType.SAND)) &&
                !(clubIndex == 8 || clubIndex == 9))
            {
                // Set the dub flag with a 40% chance if the ball is in rough or sand
                if (Odds(40)) { flags |= dub; };
            };

            // trap difficulty
            // 如果球在沙坑中并且玩家难度为4，则执行以下操作
            if (IsInHazard(BALL, GameObjType.SAND) && PlayerDifficulty == 4)
            {
                // 20%的概率设置dub标志位
                if (Odds(20)) { flags |= dub; };
            }

            // hook/slice
            // 有10%的概率进行hook或slice
            // 如果是已知的玩家难度，则增加概率到30%
            // 如果是推杆并且推杆是玩家难度，则增加概率到30%
            bool randHookSlice = (PlayerDifficulty == 1 ||
                                  PlayerDifficulty == 2 ||
                                  (PlayerDifficulty == 5 && IsOnGreen(BALL))) ? Odds(30) : Odds(10);

            if (randHookSlice)
            {
                if (PlayerDifficulty == 1)
                {
                    // 如果玩家难度为1，80%的概率添加hook标志，否则添加slice标志
                    if (Odds(80)) { flags |= hook; } else { flags |= slice; };
                }
                else if (PlayerDifficulty == 2)
                {
                    // 如果玩家难度为2，80%的概率添加slice标志，否则添加hook标志
                    if (Odds(80)) { flags |= slice; } else { flags |= hook; };
                }
                else
                {
                    // 如果玩家难度不是1或2，50%的概率添加hook标志，否则添加slice标志
                    if (Odds(50)) { flags |= hook; } else { flags |= slice; };
                };
            };

            // 初学者的幸运！
            // 如果障碍大于15，有10%的概率避免所有错误
            if ((Handicap > 15) && (Odds(10))) { flags |= luck; };

            // 王牌
            // 在3杆洞上有10%的概率出现王牌
            if (CourseInfo[HOLE_NUM].Par == 3 && Odds(10) && STROKE_NUM == 1) { flags |= ace; };
            // 声明一个双精度浮点数变量 distance
            double distance;
            // 生成一个1到100之间的随机数
            int rnd = RANDOM.Next(1, 101);

            // 如果球员的差点小于15
            if (Handicap < 15)
            {
                // 如果随机数小于等于25
                if (rnd <= 25)
                {
                    // 计算距离，球的飞行距离等于球杆标准距离减去球杆标准距离乘以差点的百分比
                    distance = clubAmt - (clubAmt * ((double)Handicap / 100.0));
                }
                // 如果随机数大于25且小于等于75
                else if (rnd > 25 && rnd <= 75)
                {
                    distance = clubAmt;  // 如果球员难度为3，距离等于球杆距离
                }
                else
                {
                    distance = clubAmt + (clubAmt * 0.10);  // 否则，距离等于球杆距离加上10%的额外距离
                };
            }
            else
            {
                if (rnd <= 75)  // 如果随机数小于等于75
                {
                    distance = clubAmt - (clubAmt * ((double)Handicap / 100.0));  // 距离等于球杆距离减去球员的残障百分比
                }
                else
                {
                    distance = clubAmt;  // 否则，距离等于球杆距离
                };
            };

            if (PlayerDifficulty == 3)  // 如果球员难度为3
            {
                if (Odds(80)) { distance = (distance * 0.80); };  // 如果80%的概率成立，距离减少20%
            };

            if ((flags & luck) == luck) { distance = clubAmt; }  // 如果运气标志位被设置，距离设为clubAmt

            // angle
            // 对于所有击球，存在4度的可能“漂移”
            // 钩或切增加5-10度的角度，钩使用负度数
            int angle = RANDOM.Next(0, 5);  // 生成0到4之间的随机数作为角度
            if ((flags & slice) == slice) { angle = RANDOM.Next(5, 11); };  // 如果切标志位被设置，角度设为5到10之间的随机数
            if ((flags & hook) == hook) { angle = 0 - RANDOM.Next(5, 11); };  // 如果钩标志位被设置，角度设为-5到-10之间的随机数
            if ((flags & luck) == luck) { angle = 0; };  // 如果运气标志位被设置，角度设为0

            var plot = PlotBall(BALL, distance, Convert.ToDouble(angle));  // 计算新的位置
            if ((flags & luck) == luck) { if(plot.Y > 0){ plot.Y = 2; }; };  // 如果运气标志位被设置，且Y坐标大于0，将Y坐标设为2

            flags = FindBall(new Ball(plot.X, plot.Y, plot.Offline, GameObjType.BALL), flags);  // 查找球的位置

            InterpretResults(plot, flags);  // 解释结果
        // --------------------------------------------------------------- plotBall
        // 根据球的位置、击球距离和偏移角度计算球的新位置
        Plot PlotBall(Ball ball, double strokeDistance, double degreesOff)
        {
            // 创建一个指向杯子的向量
            var cupVector = new Point(0, -1);
            // 计算球相对于杯子的角度
            double radFromCup = Math.Atan2((double)ball.Y, (double)ball.X) - Math.Atan2((double)cupVector.Y, (double)cupVector.X);
            // 计算球相对于击球方向的角度
            double radFromBall = radFromCup - Math.PI;

            // 计算相邻边和对边
            var hypotenuse = strokeDistance;
            var adjacent = Math.Cos(radFromBall + ToRadians(degreesOff)) * hypotenuse;
            var opposite = Math.Sqrt(Math.Pow(hypotenuse, 2) - Math.Pow(adjacent, 2));

            // 计算新的球的位置
            Point newPos;
            if (ToDegrees360(radFromBall + ToRadians(degreesOff)) > 180)
            {
                newPos = new Point(Convert.ToInt32(ball.X - opposite),
                                   Convert.ToInt32(ball.Y - adjacent));
            }
            else
            {
                // 如果条件不满足，计算新的位置
                newPos = new Point(Convert.ToInt32(ball.X + opposite),
                                   Convert.ToInt32(ball.Y - adjacent));
            }

            // 返回新的位置和相反边的长度作为结果
            return new Plot(newPos.X, newPos.Y, Convert.ToInt32(opposite));
        }


        // --------------------------------------------------------------- InterpretResults
        // 解释结果并进行相应操作
        void InterpretResults(Plot plot, int flags)
        {
            // 计算球到洞的距离
            int cupDistance = Convert.ToInt32(GetDistance(new Point(plot.X, plot.Y),
                                                          new Point(holeGeometry.Cup.X, holeGeometry.Cup.Y)));
            // 计算球到目标位置的距离
            int travelDistance = Convert.ToInt32(GetDistance(new Point(plot.X, plot.Y),
                                                             new Point(BALL.X, BALL.Y)));

            // 输出空行
            w(" ");
        }
            # 如果标志位中包含ace，则输出“Hole in One! You aced it.”，记录一杆，并报告当前分数，然后返回
            if ((flags & ace) == ace)
            {
                w("Hole in One! You aced it.");
                ScoreCardRecordStroke(new Ball(0, 0, 0, GameObjType.BALL));
                ReportCurrentScore();
                return;
            };

            # 如果标志位中包含inTrees，则输出“Your ball is lost in the trees. Take a penalty stroke.”，记录一杆，重新放球，然后返回
            if ((flags & inTrees) == inTrees)
            {
                w("Your ball is lost in the trees. Take a penalty stroke.");
                ScoreCardRecordStroke(BALL);
                TeeUp();
                return;
            };

            # 如果标志位中包含inWater，则根据50%的几率输出“Your ball has gone to a watery grave.”或“Your ball is lost in the water.”，然后输出“Take a penalty stroke.”
            if ((flags & inWater) == inWater)
            {
                var msg = Odds(50) ? "Your ball has gone to a watery grave." : "Your ball is lost in the water.";
                w(msg + " Take a penalty stroke.");
                ScoreCardRecordStroke(BALL);  # 记录球的击球数
                TeeUp();  # 准备下一杆击球
                return;  # 结束当前函数，返回结果
            };

            if ((flags & outOfBounds) == outOfBounds)  # 如果球出界
            {
                w("Out of bounds. Take a penalty stroke.");  # 输出信息：出界，需要罚杆
                ScoreCardRecordStroke(BALL);  # 记录球的击球数
                TeeUp();  # 准备下一杆击球
                return;  # 结束当前函数，返回结果
            };

            if ((flags & dub) == dub)  # 如果球被挖起
            {
                w("You dubbed it.");  # 输出信息：你挖起了球
                ScoreCardRecordStroke(BALL);  # 记录球的击球数
                TeeUp();  # 准备下一杆击球
                return;  # 结束当前函数，返回结果
            };
            # 如果球在杯中
            if ((flags & inCup) == inCup):
                # 有50%的几率输出"You holed it."，另外50%的几率输出"It's in!"
                var msg = Odds(50) ? "You holed it." : "It's in!";
                # 输出消息
                w(msg);
                # 记录击球信息
                ScoreCardRecordStroke(new Ball(plot.X, plot.Y, 0, GameObjType.BALL));
                # 报告当前得分
                ReportCurrentScore();
                # 返回
                return;

            # 如果球切出界并且不在果岭上
            if (((flags & slice) == slice) &&
                !((flags & onGreen) == onGreen)):
                # 如果球切出界，输出"You sliced badly: {1} yards offline."，否则输出"You sliced: {1} yards offline."
                var bad = ((flags & outOfBounds) == outOfBounds) ? " badly" : "";
                Console.WriteLine("You sliced{0}: {1} yards offline.", bad, plot.Offline);

            # 如果球钩出界并且不在果岭上
            if (((flags & hook) == hook) &&
                !((flags & onGreen) == onGreen)):
                var bad = ((flags & outOfBounds) == outOfBounds) ? " badly" : "";  // 检查球是否越界，如果是则设置 bad 变量为 " badly"
                Console.WriteLine("You hooked{0}: {1} yards offline.", bad, plot.Offline);  // 打印球的偏离距离

            };

            if (STROKE_NUM > 1)  // 如果击球次数大于1
            {
                var prevBall = ScoreCardGetPreviousStroke();  // 获取上一次击球的球的位置
                var d1 = GetDistance(new Point(prevBall.X, prevBall.Y),  // 计算当前球到洞的距离
                                     new Point(holeGeometry.Cup.X, holeGeometry.Cup.Y));
                var d2 = cupDistance;  // 获取当前球到洞的距离
                if (d2 > d1) { w("Too much club."); };  // 如果当前球到洞的距离大于上一次击球到洞的距离，则打印 "Too much club."

            };

            if ((flags & inRough) == inRough) { w("You're in the rough."); };  // 如果球在草地上，则打印 "You're in the rough."

            if ((flags & inSand) == inSand) { w("You're in a sand trap."); };  // 如果球在沙坑中，则打印 "You're in a sand trap."

            if ((flags & onGreen) == onGreen)  // 如果球在果岭上
            {
                var pd = (cupDistance < 4) ? ((cupDistance * 3) + " feet") : (cupDistance + " yards");  // 根据球到洞的距离计算距离信息
                Console.WriteLine("You're on the green. It's {0} from the pin.", pd);
            };
```
这段代码是一个条件语句，如果满足条件则打印出当前位置距离目标的距离。

```
            if (((flags & onFairway) == onFairway) ||
                ((flags & inRough) == inRough))
            {
                Console.WriteLine("Shot went {0} yards. It's {1} yards from the cup.", travelDistance, cupDistance);
            };
```
这段代码也是一个条件语句，如果满足条件则打印出击球距离和当前位置距离目标的距离。

```
            ScoreCardRecordStroke(new Ball(plot.X, plot.Y, 0, GameObjType.BALL));
```
这段代码调用了一个函数，记录了一次击球的信息。

```
            BALL = new Ball(plot.X, plot.Y, 0, GameObjType.BALL);
```
这段代码创建了一个新的球对象。

```
            TeeUp();
```
这段代码调用了一个函数，准备下一次击球。

```
        // --------------------------------------------------------------- ReportCurrentScore
        void ReportCurrentScore()
        {
```
这段代码是一个函数的声明，用于报告当前的比分。
            # 获取当前球洞的标准杆数
            var par = CourseInfo[HOLE_NUM].Par;
            # 判断当前球洞的成绩是否为标准杆加一，如果是则输出信息
            if (ScoreCard[HOLE_NUM].Count == par + 1) { w("A bogey. One above par."); };
            # 判断当前球洞的成绩是否为标准杆，如果是则输出信息
            if (ScoreCard[HOLE_NUM].Count == par) { w("Par. Nice."); };
            # 判断当前球洞的成绩是否为标准杆减一，如果是则输出信息
            if (ScoreCard[HOLE_NUM].Count == (par - 1)) { w("A birdie! One below par."); };
            # 判断当前球洞的成绩是否为标准杆减二，如果是则输出信息
            if (ScoreCard[HOLE_NUM].Count == (par - 2)) { w("An Eagle! Two below par."); };
            # 判断当前球洞的成绩是否为标准杆减三，如果是则输出信息
            if (ScoreCard[HOLE_NUM].Count == (par - 3)) { w("Double Eagle! Unbelievable."); };

            # 计算所有球洞的标准杆总和
            int totalPar = 0;
            for (var i = 1; i <= HOLE_NUM; i++) { totalPar += CourseInfo[i].Par; };

            # 输出总结信息，包括球洞数、总标准杆数和总成绩
            w(" ");
            w("-----------------------------------------------------");
            Console.WriteLine(" Total par for {0} hole{1} is: {2}. Your total is: {3}.",
                              HOLE_NUM,
                              ((HOLE_NUM > 1) ? "s" : ""), //plural
                              totalPar,
                              ScoreCardGetTotal());
            w("-----------------------------------------------------");
            w(" ");
            if (HOLE_NUM == 18)  # 如果当前球洞号为18
            {
                GameOver();  # 游戏结束
            }
            else
            {
                Thread.Sleep(2000);  # 线程休眠2秒
                NewHole();  # 进入下一个球洞
            };
        }


        // --------------------------------------------------------------- FindBall
        int FindBall(Ball ball, int flags)  # 定义一个函数FindBall，接受球和标志位作为参数
        {
            if (IsOnFairway(ball) && !IsOnGreen(ball)) { flags |= onFairway; }  # 如果球在球道上且不在果岭上，将标志位onFairway加入到flags中
            if (IsOnGreen(ball)) { flags |= onGreen; }  # 如果球在果岭上，将标志位onGreen加入到flags中
            if (IsInRough(ball)) { flags |= inRough; }  # 如果球在粗糙地面上，将标志位inRough加入到flags中
            if (IsOutOfBounds(ball)) { flags |= outOfBounds; }  # 如果球出界，将标志位outOfBounds加入到flags中
            if (IsInHazard(ball, GameObjType.WATER)) { flags |= inWater; }  # 如果球在水障碍物中，将标志位inWater加入到flags中
            // 如果球在树木区域内，将标志位设置为在树木区域内
            if (IsInHazard(ball, GameObjType.TREES)) { flags |= inTrees; }
            // 如果球在沙坑区域内，将标志位设置为在沙坑区域内
            if (IsInHazard(ball, GameObjType.SAND))  { flags |= inSand;  }

            // 如果球的 Y 坐标小于 0，将标志位设置为通过杯口
            if (ball.Y < 0) { flags |= passedCup; }

            // 计算球到洞口的距离，如果小于 2，将标志位设置为在洞内
            var d = GetDistance(new Point(ball.X, ball.Y),
                                new Point(holeGeometry.Cup.X, holeGeometry.Cup.Y));
            if (d < 2) { flags |= inCup; };

            // 返回标志位
            return flags;
        }

        // --------------------------------------------------------------- IsOnFairway
        // 判断球是否在球道上
        bool IsOnFairway(Ball ball)
        {
            return IsInRectangle(ball, holeGeometry.Fairway);
        }
        // --------------------------------------------------------------- IsOnGreen
        // 检查球是否在果岭上
        bool IsOnGreen(Ball ball)
        {
            // 计算球到洞的距离
            var d = GetDistance(new Point(ball.X, ball.Y),
                                new Point(holeGeometry.Cup.X, holeGeometry.Cup.Y));
            // 判断球是否在果岭上
            return d < holeGeometry.Green.Radius;
        }


        // --------------------------------------------------------------- IsInHazard
        // 检查球是否在危险区内
        bool IsInHazard(Ball ball, GameObjType hazard)
        {
            // 初始化结果为false
            bool result = false;
            // 遍历危险区数组
            Array.ForEach(holeGeometry.Hazards, (Hazard h) =>
            {
                // 计算球到危险区的距离
                var d = GetDistance(new Point(ball.X, ball.Y), new Point(h.X, h.Y));
                // 如果球在危险区内且类型匹配，则将结果设为true
                if ((d < h.Radius) && h.Type == hazard) { result = true; };
            });
            // 返回结果
            return result;
        // --------------------------------------------------------------- IsInRough
        // 检查球是否在粗糙区域内，返回布尔值
        bool IsInRough(Ball ball)
        {
            return IsInRectangle(ball, holeGeometry.Rough) &&
                (IsInRectangle(ball, holeGeometry.Fairway) == false);
        }


        // --------------------------------------------------------------- IsOutOfBounds
        // 检查球是否出界，返回布尔值
        bool IsOutOfBounds(Ball ball)
        {
            return (IsOnFairway(ball) == false) && (IsInRough(ball) == false);
        }


        // --------------------------------------------------------------- ScoreCardNewHole
        // 开始新的一轮比赛，无返回值
        void ScoreCardStartNewHole()
        {
            ScoreCard.Add(new List<Ball>());  # 在ScoreCard列表中添加一个新的空列表，用于存储球的信息
        }


        // --------------------------------------------------------------- ScoreCardRecordStroke
        void ScoreCardRecordStroke(Ball ball)
        {
            var clone = new Ball(ball.X, ball.Y, 0, GameObjType.BALL);  # 创建一个新的球对象，与传入的球对象具有相同的位置信息
            ScoreCard[HOLE_NUM].Add(clone);  # 将新创建的球对象添加到ScoreCard列表中的指定位置
        }


        // ------------------------------------------------------------ ScoreCardGetPreviousStroke
        Ball ScoreCardGetPreviousStroke()
        {
            return ScoreCard[HOLE_NUM][ScoreCard[HOLE_NUM].Count - 1];  # 返回ScoreCard列表中指定位置的球对象，该位置为当前位置减去1
        }
        // --------------------------------------------------------------- ScoreCardGetTotal
        // 计算得分卡中所有分数的总和
        int ScoreCardGetTotal()
        {
            int total = 0;
            // 遍历得分卡中的每个分数，累加到总和中
            ScoreCard.ForEach((h) => { total += h.Count; });
            return total;
        }


        // --------------------------------------------------------------- Ask
        // 从控制台输入总是一个传递给回调函数的整数
        // 或者输入"quit"来结束游戏

        void Ask(string question, int min, int max, Action<int> callback)
        {
            w(question); // 输出问题到控制台
            string i = Console.ReadLine().Trim().ToLower(); // 从控制台读取输入并转换为小写
            if (i == "quit") { Quit(); return; }; // 如果输入为"quit"，则结束游戏
            if (i == "bag") { ReviewBag(); }; // 如果输入为"bag"，则查看背包
            # 声明一个整数变量n
            int n;
            # 尝试将字符串i转换为整数，成功则返回True并将结果赋值给n
            bool success = Int32.TryParse(i, out n);

            # 如果转换成功
            if (success)
            {
                # 如果n在指定范围内
                if (n >= min && n <= max)
                {
                    # 调用回调函数，并传入n作为参数
                    callback(n);
                }
                # 如果n不在指定范围内
                else
                {
                    # 重新询问问题，直到得到符合范围的输入
                    Ask(question, min, max, callback);
                }
            }
            # 如果转换失败
            else
            {
                # 重新询问问题，直到得到符合要求的输入
                Ask(question, min, max, callback);
            };
        }
        // --------------------------------------------------------------- Wait
        // 等待用户按下任意键，然后执行回调函数
        void Wait(Action<int> callback)
        {
            // 输出提示信息
            w("Press any key to continue.");

            // 循环等待用户按下任意键
            ConsoleKeyInfo keyinfo;
            do { keyinfo = Console.ReadKey(true); }
            while (keyinfo.KeyChar < 0);
            // 清空控制台
            Console.Clear();
            // 执行回调函数
            callback(0);
        }


        // --------------------------------------------------------------- ReviewBag
        // 输出球袋中球杆的信息
        void ReviewBag()
        {
            // 输出空行和表头
            w(" ");
            w("  #     Club      Average Yardage");
            w("-----------------------------------");
            w("  1    Driver           250");  # 输出字符串，表示编号1的球杆是Driver，价格为250
            w("  2    3 Wood           225");  # 输出字符串，表示编号2的球杆是3 Wood，价格为225
            w("  3    5 Wood           200");  # 输出字符串，表示编号3的球杆是5 Wood，价格为200
            w("  4    Hybrid           190");  # 输出字符串，表示编号4的球杆是Hybrid，价格为190
            w("  5    4 Iron           170");  # 输出字符串，表示编号5的球杆是4 Iron，价格为170
            w("  6    7 Iron           150");  # 输出字符串，表示编号6的球杆是7 Iron，价格为150
            w("  7    9 Iron           125");  # 输出字符串，表示编号7的球杆是9 Iron，价格为125
            w("  8    Pitching wedge   110");  # 输出字符串，表示编号8的球杆是Pitching wedge，价格为110
            w("  9    Sand wedge        75");   # 输出字符串，表示编号9的球杆是Sand wedge，价格为75
            w(" 10    Putter            10");   # 输出字符串，表示编号10的球杆是Putter，价格为10
            w(" ");  # 输出空字符串
        }


        // --------------------------------------------------------------- Quit
        void Quit()
        {
            w("");  # 输出空字符串
            w("Looks like rain. Goodbye!");  # 输出字符串，表示天气看起来像是要下雨，然后道别
            w("");  # 输出空字符串
            Wait((z) => { });
            return;
```
这段代码是一个匿名函数，它使用了一个名为Wait的函数，该函数接受一个回调函数作为参数。在这个例子中，回调函数是一个匿名函数，它不执行任何操作，只是等待。然后返回。

```
        // --------------------------------------------------------------- GameOver
        void GameOver()
        {
            var net = ScoreCardGetTotal() - Handicap;
            w("Good game!");
            w("Your net score is: " + net);
            w("Let's visit the pro shop...");
            w(" ");
            Wait((z) => { });
            return;
        }
```
这段代码定义了一个名为GameOver的函数。在函数内部，它计算了玩家的净得分（net），然后打印一些消息，最后调用了Wait函数，等待一段时间。最后返回。

```
        // YOUR BAG
        // ======================================================== Clubs
```
这段代码是一段注释，它描述了接下来的代码部分的内容。在这个例子中，它描述了一个名为"Clubs"的部分。
        (string, int)[] Clubs = new (string, int)[] {
            ("",0),  // 创建一个包含字符串和整数的元组数组，初始化为空字符串和0

                // name, average yardage
                ("Driver", 250),  // 添加元组，表示球杆名称和平均码数
                ("3 Wood", 225),
                ("5 Wood", 200),
                ("Hybrid", 190),
                ("4 Iron", 170),
                ("7 Iron", 150),
                ("9 Iron", 125),
                ("Pitching wedge", 110),
                ("Sand wedge", 75),
                ("Putter", 10)
                };


        // THE COURSE
        // ======================================================== CourseInfo
```
```python
        // THE COURSE
        // ======================================================== CourseInfo
```

在这段代码中，我们定义了一个包含不同高尔夫球杆名称和平均码数的元组数组。接下来的注释是关于高尔夫球场的信息，但是由于这是C#代码，而不是Python代码，因此这段注释可能不适用于Python。
        HoleInfo[] CourseInfo = new HoleInfo[]{
            new HoleInfo(0, 0, 0, new Hazard[]{}, ""), // include a blank so index 1 == hole 1
            // 创建一个包含 HoleInfo 对象的数组，用于存储高尔夫球场每个洞的信息

            // -------------------------------------------------------- front 9
            // hole, yards, par, hazards, (description)

            new HoleInfo(1, 361, 4,
                         new Hazard[]{
                             new Hazard( 20, 100, 10, GameObjType.TREES),
                             new Hazard(-20,  80, 10, GameObjType.TREES),
                             new Hazard(-20, 100, 10, GameObjType.TREES)
                         },
                         "There are a couple of trees on the left and right."),
            // 创建一个新的 HoleInfo 对象，表示第一个洞的信息，包括洞号、码数、标准杆数、障碍物和描述

            new HoleInfo(2, 389, 4,
                         new Hazard[]{
                             new Hazard(0, 160, 20, GameObjType.WATER)
                         },
                         "There is a large water hazard across the fairway about 150 yards."),
            // 创建一个新的 HoleInfo 对象，表示第二个洞的信息，包括洞号、码数、标准杆数、障碍物和描述
            // 创建一个新的HoleInfo对象，包含第3个洞的信息
            new HoleInfo(3, 206, 3,
                         // 创建一个包含3个Hazard对象的数组，表示该洞的危险区域
                         new Hazard[]{
                             new Hazard( 20,  20,  5, GameObjType.WATER),  // 水障碍物
                             new Hazard(-20, 160, 10, GameObjType.WATER),  // 水障碍物
                             new Hazard( 10,  12,  5, GameObjType.SAND)    // 沙坑障碍物
                         },
                         "There is some sand and water near the green."),  // 描述该洞的信息

            // 创建一个新的HoleInfo对象，包含第4个洞的信息
            new HoleInfo(4, 500, 5,
                         // 创建一个包含1个Hazard对象的数组，表示该洞的危险区域
                         new Hazard[]{
                             new Hazard(-14, 12, 12, GameObjType.SAND)  // 沙坑障碍物
                         },
                         "There's a bunker to the left of the green."),  // 描述该洞的信息

            // 创建一个新的HoleInfo对象，包含第5个洞的信息
            new HoleInfo(5, 408, 4,
                         // 创建一个包含3个Hazard对象的数组，表示该洞的危险区域
                         new Hazard[]{
                             new Hazard(20, 120, 20, GameObjType.TREES),  // 树木障碍物
                             new Hazard(20, 160, 20, GameObjType.TREES),  // 树木障碍物
                             new Hazard(10,  20,  5, GameObjType.SAND)     // 沙坑障碍物
# 创建一个新的HoleInfo对象，包含编号为6的球洞信息
new HoleInfo(6, 359, 4,
                         # 创建一个包含两个Hazard对象的数组
                         new Hazard[]{
                             # 创建一个位于坐标(14, 0)的沙坑
                             new Hazard( 14, 0, 4, GameObjType.SAND),
                             # 创建一个位于坐标(-14, 0)的沙坑
                             new Hazard(-14, 0, 4, GameObjType.SAND)
                         },
                         ""),

# 创建一个新的HoleInfo对象，包含编号为7的球洞信息
new HoleInfo(7, 424, 5,
                         # 创建一个包含三个Hazard对象的数组
                         new Hazard[]{
                             # 创建一个位于坐标(20, 200)的沙坑
                             new Hazard(20, 200, 10, GameObjType.SAND),
                             # 创建一个位于坐标(10, 180)的沙坑
                             new Hazard(10, 180, 10, GameObjType.SAND),
                             # 创建一个位于坐标(20, 160)的沙坑
                             new Hazard(20, 160, 10, GameObjType.SAND)
                         },
                         "There are several sand traps along your right."),

# 创建一个新的HoleInfo对象，包含编号为8的球洞信息
new HoleInfo(8, 388, 4,
                         # 创建一个空的Hazard数组
                         new Hazard[]{
            // 创建新的HoleInfo对象，表示第9洞，长度196码，标准杆3杆
            new HoleInfo(9, 196, 3,
                         // 包含两个Hazard对象的数组，分别表示树木和沙坑的位置和大小
                         new Hazard[]{
                             new Hazard(-30, 180, 20, GameObjType.TREES),  // 第一个Hazard对象表示树木
                             new Hazard( 14,  -8,  5, GameObjType.SAND)    // 第二个Hazard对象表示沙坑
                         },
                         ""),
            // 创建新的HoleInfo对象，表示第10洞，长度400码，标准杆4杆
            new HoleInfo(10, 400, 4,
                         // 包含两个Hazard对象的数组，分别表示两个沙坑的位置和大小
                         new Hazard[]{
                             new Hazard(-14, -8, 5, GameObjType.SAND),  // 第一个Hazard对象表示沙坑
                             new Hazard( 14, -8, 5, GameObjType.SAND)   // 第二个Hazard对象表示沙坑
                         },
                         "")
            new HoleInfo(11, 560, 5,  # 创建一个新的HoleInfo对象，参数分别为球洞编号、码数、标准杆数
                         new Hazard[]{  # 创建一个新的Hazard数组
                             new Hazard(-20, 400, 10, GameObjType.TREES),  # 在数组中添加一个新的Hazard对象，参数分别为位置坐标和障碍物类型
                             new Hazard(-10, 380, 10, GameObjType.TREES),
                             new Hazard(-20, 260, 10, GameObjType.TREES),
                             new Hazard(-20, 200, 10, GameObjType.TREES),
                             new Hazard(-10, 180, 10, GameObjType.TREES),
                             new Hazard(-20, 160, 10, GameObjType.TREES)
                         },
                         "Lots of trees along the left of the fairway."),  # 添加一个描述字符串

            new HoleInfo(12, 132, 3,  # 创建一个新的HoleInfo对象，参数分别为球洞编号、码数、标准杆数
                         new Hazard[]{  # 创建一个新的Hazard数组
                             new Hazard(-10, 120, 10, GameObjType.WATER),  # 在数组中添加一个新的Hazard对象，参数分别为位置坐标和障碍物类型
                             new Hazard( -5, 100, 10, GameObjType.SAND)
                         },
                         "There is water and sand directly in front of you. A good drive should clear both."),  # 添加一个描述字符串

            new HoleInfo(13, 357, 4,  # 创建一个新的HoleInfo对象，参数分别为球洞编号、码数、标准杆数
# 创建一个新的HoleInfo对象，包括位置、大小、障碍物数组和描述信息
new HoleInfo(14, 294, 4,
             # 创建一个包含障碍物的数组
             new Hazard[]{
                 # 创建一个新的Hazard对象，包括位置、大小和类型
                 new Hazard(0, 20, 10, GameObjType.SAND)
             },
             # 描述信息为空字符串
             ""),

# 创建一个新的HoleInfo对象，包括位置、大小、障碍物数组和描述信息
new HoleInfo(15, 475, 5,
             # 创建一个包含障碍物的数组
             new Hazard[]{
                 # 创建一个新的Hazard对象，包括位置、大小和类型
                 new Hazard(-20, 20, 10, GameObjType.WATER),
                 new Hazard( 10, 20, 10, GameObjType.SAND)
             },
             # 描述信息为"Some sand and water near the green."
             "Some sand and water near the green."),
# 创建一个新的HoleInfo对象，包括编号、距离、折数、障碍物数组和描述
new HoleInfo(16, 375, 4,
             new Hazard[]{
                 new Hazard(-14, -8, 5, GameObjType.SAND)
             },
             ""),

# 创建一个新的HoleInfo对象，包括编号、距离、折数、障碍物数组和描述
new HoleInfo(17, 180, 3,
             new Hazard[]{
                 new Hazard( 20, 100, 10, GameObjType.TREES),
                 new Hazard(-20,  80, 10, GameObjType.TREES)
             },
             ""),

# 创建一个新的HoleInfo对象，包括编号、距离、折数、障碍物数组和描述
new HoleInfo(18, 550, 5,
             new Hazard[]{
                 new Hazard(20, 30, 15, GameObjType.WATER)
             },
             "There is a water hazard near the green.")
        // -------------------------------------------------------- HoleInfo
        // 定义了一个名为 HoleInfo 的类，用于存储高尔夫球场上每个洞的信息
        class HoleInfo
        {
            // 每个洞的编号
            public int Hole { get; }
            // 每个洞的码数
            public int Yards { get; }
            // 每个洞的标准杆数
            public int Par { get; }
            // 每个洞可能的危险区域
            public Hazard[] Hazard { get; }
            // 每个洞的描述
            public string Description { get; }

            // 构造函数，用于初始化 HoleInfo 对象的属性
            public HoleInfo(int hole, int yards, int par, Hazard[] hazard, string description)
            {
                Hole = hole;
                Yards = yards;
                Par = par;
                Hazard = hazard;
                Description = description;
            }
        }
# 定义枚举类型，包括球、杯、绿地、球道、草地、树木、水、沙
public enum GameObjType { BALL, CUP, GREEN, FAIRWAY, ROUGH, TREES, WATER, SAND }

# 定义圆形游戏对象类
public class CircleGameObj
{
    # 游戏对象类型
    public GameObjType Type { get; }
    # X 坐标
    public int X { get; }
    # Y 坐标
    public int Y { get; }
    # 半径
    public int Radius { get; }

    # 构造函数，初始化圆形游戏对象的位置、半径和类型
    public CircleGameObj(int x, int y, int r, GameObjType type)
    {
        Type = type;
        X = x;
        Y = y;
        Radius = r;
    }
}
        // -------------------------------------------------------- RectGameObj
        // 定义一个名为 RectGameObj 的公共类
        public class RectGameObj
        {
            // 定义 Type 属性，表示游戏对象的类型
            public GameObjType Type { get; }
            // 定义 X 属性，表示游戏对象的横坐标
            public int X { get; }
            // 定义 Y 属性，表示游戏对象的纵坐标
            public int Y { get; }
            // 定义 Width 属性，表示游戏对象的宽度
            public int Width { get; }
            // 定义 Length 属性，表示游戏对象的长度
            public int Length { get; }

            // 定义一个构造函数，用于初始化游戏对象的属性
            public RectGameObj(int x, int y, int w, int l, GameObjType type)
            {
                // 将传入的参数赋值给对应的属性
                Type = type;
                X = x;
                Y = y;
                Width = w;
                Length = l;
            }
        // -------------------------------------------------------- HoleGeometry
        // 定义 HoleGeometry 类，表示高尔夫球场的洞的几何信息
        public class HoleGeometry
        {
            // 表示洞的杯子位置的圆形游戏对象
            public CircleGameObj Cup { get; }
            // 表示洞的果岭位置的圆形游戏对象
            public CircleGameObj Green { get; }
            // 表示洞的球道位置的矩形游戏对象
            public RectGameObj Fairway { get; }
            // 表示洞的草地位置的矩形游戏对象
            public RectGameObj Rough { get; }
            // 表示洞的危险区域的数组
            public Hazard[] Hazards { get; }

            // 构造函数，初始化洞的几何信息
            public HoleGeometry(CircleGameObj cup, CircleGameObj green, RectGameObj fairway, RectGameObj rough, Hazard[] haz)
            {
                // 初始化洞的杯子位置
                Cup = cup;
                // 初始化洞的果岭位置
                Green = green;
                // 初始化洞的球道位置
                Fairway = fairway;
                // 初始化洞的草地位置
                Rough = rough;
                // 初始化洞的危险区域
                Hazards = haz;
            }
        // -------------------------------------------------------- Plot
        // 定义名为 Plot 的类
        public class Plot
        {
            // 定义 X 属性，只有 get 方法
            public int X { get; }
            // 定义 Y 属性，包含 get 和 set 方法
            public int Y { get; set; }
            // 定义 Offline 属性，只有 get 方法
            public int Offline { get; }

            // 定义 Plot 类的构造函数，接受 x、y、offline 三个参数
            public Plot(int x, int y, int offline)
            {
                // 初始化 X 属性
                X = x;
                // 初始化 Y 属性
                Y = y;
                // 初始化 Offline 属性
                Offline = offline;
            }
        }


        // -------------------------------------------------------- GetDistance
        // 定义名为 GetDistance 的函数
        // 计算两点之间的距离
        double GetDistance(Point pt1, Point pt2)
        {
            return Math.Sqrt(Math.Pow((pt2.X - pt1.X), 2) + Math.Pow((pt2.Y - pt1.Y), 2));
        }


        // -------------------------------------------------------- IsInRectangle
        // 检查点是否在矩形内部
        bool IsInRectangle(CircleGameObj pt, RectGameObj rect)
        {
            return ((pt.X > rect.X) &&
                    (pt.X < rect.X + rect.Width) &&
                    (pt.Y > rect.Y) &&
                    (pt.Y < rect.Y + rect.Length));
        }


        // -------------------------------------------------------- ToRadians
        // 将角度转换为弧度
        double ToRadians(double angle) { return angle * (Math.PI / 180.0); }
        // -------------------------------------------------------- ToDegrees360
        // 将弧度转换为360度
        double ToDegrees360(double angle)
        {
            double deg = angle * (180.0 / Math.PI);  // 将角度转换为弧度
            if (deg < 0.0) { deg += 360.0; }  // 如果角度小于0，则加上360度
            return deg;  // 返回转换后的角度
        }


        // -------------------------------------------------------- Odds
        // 返回一个整数小于等于给定参数的概率
        // 介于1-100之间
        Random RND = new Random();  // 创建一个随机数生成器

        bool Odds(int x)
        {
            return RND.Next(1, 101) <= x;  // 返回一个1-100之间的随机数是否小于等于给定参数
        }
    }
```

这部分代码是一个缩进错误，应该删除。
```