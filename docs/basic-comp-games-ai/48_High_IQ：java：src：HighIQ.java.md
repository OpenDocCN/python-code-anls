# `48_High_IQ\java\src\HighIQ.java`

```
# 导入所需的模块
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

# 游戏的主类，名为 HighIQ
public class HighIQ {

    # 游戏棋盘，使用一个映射来表示位置号和对应的值
    private final Map<Integer, Boolean> board;

    # 输出流
    private final PrintStream out;
    //Input scanner to use
    private final Scanner scanner; // 声明一个私有的Scanner对象

    // 构造函数，初始化out为System.out，初始化scanner为传入的参数，初始化board为一个HashMap
    public HighIQ(Scanner scanner) {
        out = System.out;
        this.scanner = scanner;
        board = new HashMap<>();

        //Set of all locations to put initial pegs on
        // 初始化一个整型数组，包含初始放置钉子的位置
        int[] locations = new int[]{
                13, 14, 15, 22, 23, 24, 29, 30, 31, 32, 33, 34, 35, 38, 39, 40, 42, 43, 44, 47, 48, 49, 50, 51, 52, 53, 58, 59, 60, 67, 68, 69
        };

        // 遍历locations数组，将每个位置作为key，true作为value放入board中
        for (int i : locations) {
            board.put(i, true);
        }

        // 将位置41作为key，false作为value放入board中
        board.put(41, false);
    }
    }

    /**
     * Plays the actual game, from start to finish.
     */
    public void play() {
        // 打印游戏板
        printBoard();
        // 当游戏未结束时，执行以下操作
        do {
            // 执行移动操作，直到合法移动为止
            while (!move()) {
                out.println("ILLEGAL MOVE, TRY AGAIN...");
            }
        } while (!isGameFinished());

        // 计算剩余的棋子数量
        int pegCount = 0;
        for (Integer key : board.keySet()) {
            if (board.getOrDefault(key, false)) {
                pegCount++;
            }
        }
        out.println("YOU HAD " + pegCount + " PEGS REMAINING");  // 打印剩余的棋子数量

        if (pegCount == 1) {  // 如果剩余的棋子数量为1
            out.println("BRAVO!  YOU MADE A PERFECT SCORE!");  // 打印祝贺消息
            out.println("SAVE THIS PAPER AS A RECORD OF YOUR ACCOMPLISHMENT!");  // 提示用户保存记录
        }
    }

    /**
     * Makes an individual move
     * @return True if the move was valid, false if the user made an error and the move is invalid
     */
    public boolean move() {
        out.println("MOVE WHICH PIECE");  // 提示用户移动哪个棋子
        int from = scanner.nextInt();  // 从用户输入中获取要移动的棋子位置

        // 使用 getOrDefault 方法，如果位置无效则返回 false
        if (!board.getOrDefault(from, false)) {  // 如果要移动的位置无效
            return false;  // 返回移动无效
        }
        out.println("TO WHERE");  // 输出提示信息到控制台，要求输入目标位置
        int to = scanner.nextInt();  // 从用户输入中获取目标位置的整数值

        if (board.getOrDefault(to, true)) {  // 如果目标位置在棋盘中不存在或者为空
            return false;  // 返回 false
        }

        // 如果起始位置和目标位置相同，则不做任何操作
        if (from == to) {
            return true;  // 返回 true
        }

        // 使用差值来检查相对位置是否有效
        int difference = Math.abs(to - from);  // 计算起始位置和目标位置的差值的绝对值
        if (difference != 2 && difference != 18) {  // 如果差值不等于 2 且不等于 18
            return false;  // 返回 false
        }

        // 检查起始位置和目标位置之间是否有一个棋子
        // 检查是否有跳棋的中间位置为空，如果为空则返回 false
        if (!board.getOrDefault((to + from) / 2, false)) {
            return false;
        }

        // 实际移动棋子
        board.put(from,false); // 将起始位置的棋子移除
        board.put(to,true); // 在目标位置放置棋子
        board.put((from + to) / 2, false); // 将跳棋的中间位置的棋子移除

        return true; // 返回移动成功
    }

    /**
     * 检查游戏是否结束
     * @return 如果没有更多的移动则返回 True，否则返回 False
     */
    public boolean isGameFinished() {
        for (Integer key : board.keySet()) {
            if (board.get(key)) {
                // 间距要么是 1，要么是 9
                // 从每个点向右和向下查看，检查移动的两个方向
                for (int space : new int[]{1, 9}) {
                    // 检查相邻位置是否有木板
                    Boolean nextToPeg = board.getOrDefault(key + space, false);
                    // 检查是否有可移动的空间
                    Boolean hasMovableSpace = !board.getOrDefault(key - space, true) || !board.getOrDefault(key + space * 2, true);
                    // 如果有相邻木板且有可移动的空间，则返回 false
                    if (nextToPeg && hasMovableSpace) {
                        return false;
                    }
                }
            }
        }
        // 如果没有找到相邻木板和可移动的空间，则返回 true
        return true;
    }

    public void printBoard() {
        // 打印游戏板
        for (int i = 0; i < 7; i++) {
            for (int j = 11; j < 18; j++) {
                out.print(getChar(j + 9 * i));  // 获取并打印指定位置的字符
            }
            out.println();  // 换行
        }
    }  # 结束 getChar 方法的定义

    private char getChar(int position) {  # 定义一个返回字符类型的私有方法 getChar，参数为 position
        Boolean value = board.get(position);  # 从 board 中获取指定位置的值，并赋给变量 value
        if (value == null) {  # 如果 value 为 null
            return ' ';  # 返回空格字符
        } else if (value) {  # 如果 value 为 true
            return '!';  # 返回感叹号字符
        } else {  # 否则
            return 'O';  # 返回大写字母 O 字符
        }
    }
}  # 结束类的定义
```