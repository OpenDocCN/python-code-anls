# `basic-computer-games\48_High_IQ\java\src\HighIQ.java`

```

// 导入所需的类
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

/**
 * Game of HighIQ
 * <p>
 * Based on the Basic Game of HighIQ Here:
 * https://github.com/coding-horror/basic-computer-games/blob/main/48_High_IQ/highiq.bas
 *
 * No additional functionality has been added
 */
public class HighIQ {

    // 游戏板，作为位置数字到它们的值的映射
    private final Map<Integer, Boolean> board;

    // 输出流
    private final PrintStream out;

    // 要使用的输入扫描器
    private final Scanner scanner;


    public HighIQ(Scanner scanner) {
        out = System.out;
        this.scanner = scanner;
        board = new HashMap<>();

        // 设置所有位置放置初始钉子
        int[] locations = new int[]{
                13, 14, 15, 22, 23, 24, 29, 30, 31, 32, 33, 34, 35, 38, 39, 40, 42, 43, 44, 47, 48, 49, 50, 51, 52, 53, 58, 59, 60, 67, 68, 69
        };

        for (int i : locations) {
            board.put(i, true);
        }

        board.put(41, false);
    }

    /**
     * 执行实际游戏，从开始到结束。
     */
    public void play() {
        do {
            printBoard();
            while (!move()) {
                out.println("ILLEGAL MOVE, TRY AGAIN...");
            }
        } while (!isGameFinished());

        int pegCount = 0;
        for (Integer key : board.keySet()) {
            if (board.getOrDefault(key, false)) {
                pegCount++;
            }
        }

        out.println("YOU HAD " + pegCount + " PEGS REMAINING");

        if (pegCount == 1) {
            out.println("BRAVO!  YOU MADE A PERFECT SCORE!");
            out.println("SAVE THIS PAPER AS A RECORD OF YOUR ACCOMPLISHMENT!");
        }
    }

    /**
     * 进行单个移动
     * @return 如果移动有效则返回true，如果用户出错并且移动无效则返回false
     */
    public boolean move() {
        out.println("MOVE WHICH PIECE");
        int from = scanner.nextInt();

        // 使用getOrDefault，如果位置无效则将语句设为false
        if (!board.getOrDefault(from, false)) {
            return false;
        }

        out.println("TO WHERE");
        int to = scanner.nextInt();

        if (board.getOrDefault(to, true)) {
            return false;
        }

        // 如果它们相同则不执行任何操作
        if (from == to) {
            return true;
        }

        // 使用差值检查相对位置是否有效
        int difference = Math.abs(to - from);
        if (difference != 2 && difference != 18) {
            return false;
        }

        // 检查from和to之间是否有一个钉子
        if (!board.getOrDefault((to + from) / 2, false)) {
            return false;
        }

        // 实际移动
        board.put(from,false);
        board.put(to,true);
        board.put((from + to) / 2, false);

        return true;
    }

    /**
     * 检查游戏是否结束
     * @return 如果没有更多的移动则返回true，否则返回false
     */
    public boolean isGameFinished() {
        for (Integer key : board.keySet()) {
            if (board.get(key)) {
                // 间距为1或9
                // 从每个点向右和向下查看，检查移动的两个方向
                for (int space : new int[]{1, 9}) {
                    Boolean nextToPeg = board.getOrDefault(key + space, false);
                    Boolean hasMovableSpace = !board.getOrDefault(key - space, true) || !board.getOrDefault(key + space * 2, true);
                    if (nextToPeg && hasMovableSpace) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    // 打印游戏板
    public void printBoard() {
        for (int i = 0; i < 7; i++) {
            for (int j = 11; j < 18; j++) {
                out.print(getChar(j + 9 * i));
            }
            out.println();
        }
    }

    // 获取位置的字符表示
    private char getChar(int position) {
        Boolean value = board.get(position);
        if (value == null) {
            return ' ';
        } else if (value) {
            return '!';
        } else {
            return 'O';
        }
    }
}

```