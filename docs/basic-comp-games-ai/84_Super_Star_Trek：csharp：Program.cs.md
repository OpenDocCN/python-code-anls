# `84_Super_Star_Trek\csharp\Program.cs`

```
# SUPER STARTREK - MAY 16,1978 - REQUIRES 24K MEMORY
# 1978年5月16日发布的 SUPER STARTREK 游戏，需要24K内存
#
# ****         **** STAR TREK ****        ****
# ****  SIMULATION OF A MISSION OF THE STARSHIP ENTERPRISE,
# ****  AS SEEN ON THE STAR TREK TV SHOW.
# ****  ORIGIONAL PROGRAM BY MIKE MAYFIELD, MODIFIED VERSION
# ****  PUBLISHED IN DEC'S "101 BASIC GAMES", BY DAVE AHL.
# ****  MODIFICATIONS TO THE LATTER (PLUS DEBUGGING) BY BOB
# ****  LEEDOM - APRIL & DECEMBER 1974,
# ****  WITH A LITTLE HELP FROM HIS FRIENDS . . .
# ****  COMMENTS, EPITHETS, AND SUGGESTIONS SOLICITED --
# ****  SEND TO:  R. C. LEEDOM
# ****            WESTINGHOUSE DEFENSE & ELECTRONICS SYSTEMS CNTR.
# ****            BOX 746, M.S. 338
# ****            BALTIMORE, MD  21203
# ****
# ****  CONVERTED TO MICROSOFT 8 K BASIC 3/16/78 BY JOHN GORDERS
# ****  LINE NUMBERS FROM VERSION STREK7 OF 1/12/75 PRESERVED AS
# ****  MUCH AS POSSIBLE WHILE USING MULTIPLE STATEMENTS PER LINE
# ****  SOME LINES ARE LONGER THAN 72 CHARACTERS; THIS WAS DONE
// 导入 Games.Common.IO 命名空间
using Games.Common.IO;
// 导入 Games.Common.Randomness 命名空间
using Games.Common.Randomness;
// 导入 SuperStarTrek 命名空间
using SuperStarTrek;

// 创建 ConsoleIO 实例
var io = new ConsoleIO();
// 创建 RandomNumberGenerator 实例
var random = new RandomNumberGenerator();

// 创建 Game 实例，传入 ConsoleIO 实例和 RandomNumberGenerator 实例
var game = new Game(io, random);

// 执行游戏介绍
game.DoIntroduction();

// 执行游戏并在游戏需要重新开始时继续执行
do
{
    game.Play();
} while (game.Replay());
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```