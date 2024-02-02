# `basic-computer-games\48_High_IQ\java\src\HighIQ.java`

```py
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

    //Game board, as a map of position numbers to their values
    private final Map<Integer, Boolean> board;

    //Output stream
    private final PrintStream out;

    //Input scanner to use
    private final Scanner scanner;


    public HighIQ(Scanner scanner) {
        out = System.out;
        this.scanner = scanner;
        board = new HashMap<>();

        //Set of all locations to put initial pegs on
        int[] locations = new int[]{
                13, 14, 15, 22, 23, 24, 29, 30, 31, 32, 33, 34, 35, 38, 39, 40, 42, 43, 44, 47, 48, 49, 50, 51, 52, 53, 58, 59, 60, 67, 68, 69
        };

        // Initialize the game board with pegs at the specified locations
        for (int i : locations) {
            board.put(i, true);
        }

        // Set one position on the board to false
        board.put(41, false);
    }

    /**
     * Plays the actual game, from start to finish.
     */
    public void play() {
        // Play the game until it is finished
        do {
            // Print the current state of the game board
            printBoard();
            // Keep prompting the user for a move until a valid move is made
            while (!move()) {
                out.println("ILLEGAL MOVE, TRY AGAIN...");
            }
        } while (!isGameFinished());

        // Count the remaining pegs on the board
        int pegCount = 0;
        for (Integer key : board.keySet()) {
            if (board.getOrDefault(key, false)) {
                pegCount++;
            }
        }

        // Print the number of remaining pegs
        out.println("YOU HAD " + pegCount + " PEGS REMAINING");

        // Print a message based on the number of remaining pegs
        if (pegCount == 1) {
            out.println("BRAVO!  YOU MADE A PERFECT SCORE!");
            out.println("SAVE THIS PAPER AS A RECORD OF YOUR ACCOMPLISHMENT!");
        }
    }

    /**
     * Makes an individual move
     * @return True if the move was valid, false if the user made an error and the move is invalid
     */
    public boolean move() {
        // 输出提示信息，要求输入移动的棋子位置
        out.println("MOVE WHICH PIECE");
        // 读取输入的起始位置
        int from = scanner.nextInt();

        // 使用 getOrDefault 方法，如果起始位置无效则返回 false
        if (!board.getOrDefault(from, false)) {
            return false;
        }

        // 输出提示信息，要求输入目标位置
        out.println("TO WHERE");
        // 读取输入的目标位置
        int to = scanner.nextInt();

        // 如果目标位置已经有棋子，则返回 false
        if (board.getOrDefault(to, true)) {
            return false;
        }

        // 如果起始位置和目标位置相同，则不执行移动，直接返回 true
        if (from == to) {
            return true;
        }

        // 计算起始位置和目标位置的差值，用于检查相对位置是否有效
        int difference = Math.abs(to - from);
        // 如果差值不是 2 或 18，则返回 false
        if (difference != 2 && difference != 18) {
            return false;
        }

        // 检查起始位置和目标位置之间是否有棋子，如果没有则返回 false
        if (!board.getOrDefault((to + from) / 2, false)) {
            return false;
        }

        // 实际执行移动操作，更新棋盘状态
        board.put(from,false);
        board.put(to,true);
        board.put((from + to) / 2, false);

        return true;
    }

    /**
     * 检查游戏是否结束
     * @return 如果没有更多的移动则返回 True，否则返回 False
     */
    public boolean isGameFinished() {
        // 遍历棋盘上的每个位置
        for (Integer key : board.keySet()) {
            // 如果当前位置有棋子
            if (board.get(key)) {
                // 从当前位置向右和向下分别检查相邻位置，检查两个方向的可移动性
                for (int space : new int[]{1, 9}) {
                    // 检查相邻位置是否有棋子
                    Boolean nextToPeg = board.getOrDefault(key + space, false);
                    // 检查当前位置的两个方向是否有可移动的空位
                    Boolean hasMovableSpace = !board.getOrDefault(key - space, true) || !board.getOrDefault(key + space * 2, true);
                    // 如果存在相邻位置有棋子且有可移动的空位，则游戏未结束，返回 false
                    if (nextToPeg && hasMovableSpace) {
                        return false;
                    }
                }
            }
        }
        // 如果所有位置都经过检查且没有找到可移动的棋子，则游戏结束，返回 true
        return true;
    }
    # 打印游戏板的方法
    public void printBoard() {
        # 遍历行
        for (int i = 0; i < 7; i++) {
            # 遍历列
            for (int j = 11; j < 18; j++) {
                # 打印指定位置的字符
                out.print(getChar(j + 9 * i));
            }
            # 换行
            out.println();
        }
    }

    # 获取指定位置的字符
    private char getChar(int position) {
        # 获取指定位置的布尔值
        Boolean value = board.get(position);
        # 如果值为空，返回空格
        if (value == null) {
            return ' ';
        } 
        # 如果值为真，返回感叹号
        else if (value) {
            return '!';
        } 
        # 如果值为假，返回大写字母O
        else {
            return 'O';
        }
    }
# 闭合前面的函数定义
```