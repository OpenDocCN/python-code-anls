# `basic-computer-games\71_Poker\java\Poker.java`

```

# 导入需要的类
import java.util.Random;
import java.util.Scanner;

# 导入静态成员 out
import static java.lang.System.out;

# 以下是对该程序的一些说明和来源信息
/**
 * 将 CREATIVE COMPUTING Poker 游戏从 Commodore 64 Basic 移植到普通的 Java
 *
 * 原始来源是从杂志扫描得到的：https://www.atariarchives.org/basicgames/showpage.php?page=129
 *
 * 我基于这里的 OCR'ed 源代码进行移植：https://github.com/coding-horror/basic-computer-games/blob/main/71_Poker/poker.bas
 *
 * 为什么？因为我记得当我还是一个小开发者的时候，我曾经在我的 C64 上输入过这个程序，并且玩得很开心！
 *
 * 目标：保持算法和用户体验基本不变；改进控制流程一点（在 Java 中不使用 goto！）并且重命名一些东西以便更容易理解。
 *
 * 结果：可能会有一些 bug，请告诉我。
 */

```