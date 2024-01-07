# `basic-computer-games\60_Mastermind\java\Mastermind.java`

```

// 导入所需的 Java 类
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.Scanner;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * 这是一个基于 Java 的 BASIC Mastermind 游戏的移植版本。
 *
 * 与原始 BASIC 版本的不同之处：
 *    使用了一种将解决方案 ID 转换为颜色代码字符串的数字基数转换方法。原始版本使用了一种效率低下的加法和进位技术，其中每次进位都会增加下一个位置的颜色 ID。
 *
 *    实现了对秘密代码中位置数量的上限检查，以防止内存耗尽。由于计算机用于推断玩家秘密代码的算法，它会搜索整个可能的解决方案范围。这个范围可能非常大，因为它是（颜色数量）^（位置数量）。原始版本如果这个数字太大，它会愉快地尝试在系统上分配所有内存。如果它成功地在一个大的解决方案集上分配了内存，那么它也会通过前面提到的技术花费太长时间来计算代码字符串。
 *
 *    在开始时会额外给出一条消息，提醒玩家关于 BOARD 和 QUIT 命令。
 */

```