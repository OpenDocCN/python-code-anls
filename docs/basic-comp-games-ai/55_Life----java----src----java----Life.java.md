# `basic-computer-games\55_Life\java\src\java\Life.java`

```py
import java.util.ArrayList;  // 导入 ArrayList 类
import java.util.List;  // 导入 List 接口
import java.util.Scanner;  // 导入 Scanner 类

/**
 * 生命游戏类。<br>
 * <br>
 * 模仿 BASIC 版本的行为，但是 Java 代码与原始版本没有太多共同之处。
 * <br>
 * 行为上的差异:
 * <ul>
 *     <li>输入支持 "." 字符，但是它是可选的。</li>
 *     <li>关于 "DONE" 字符串的输入不区分大小写。</li>
 * </ul>
 */
public class Life {

    private static final byte DEAD  = 0;  // 常量，表示死细胞
    private static final byte ALIVE = 1;  // 常量，表示活细胞
    private static final String NEWLINE = "\n";  // 常量，表示换行符

    private final Scanner consoleReader = new Scanner(System.in);  // 创建一个用于从控制台读取输入的 Scanner 对象

    private final byte[][] matrix = new byte[21][67];  // 创建一个二维数组，表示细胞矩阵
    private int generation = 0;  // 表示当前的代数
    private int population = 0;  // 表示当前的总人口数
    boolean stopAfterGen = false;  // 表示是否在特定代数后停止
    boolean invalid = false;  // 表示输入是否无效

    /**
     * 构造函数。
     *
     * @param args 命令行参数
     */
    public Life(String[] args) {
        parse(args);  // 调用 parse 方法解析命令行参数
    }

    private void parse(String[] args) {
        for (String arg : args) {
            if ("-s".equals(arg)) {  // 如果命令行参数包含 "-s"
                stopAfterGen = true;  // 设置 stopAfterGen 为 true
                break;  // 跳出循环
            }
        }
    }

    /**
     * 启动游戏。
     */
    public void start() {
        printGameHeader();  // 调用打印游戏标题的方法
        readPattern();  // 调用读取模式的方法
        while (true) {  // 无限循环
            printGeneration();  // 调用打印代数的方法
            advanceToNextGeneration();  // 调用前进到下一代的方法
            if (stopAfterGen) {  // 如果设置了 stopAfterGen
                System.out.print("PRESS ENTER TO CONTINUE");  // 打印提示信息
                consoleReader.nextLine();  // 从控制台读取输入
            }
        }
    }
}
    private void advanceToNextGeneration() {
        // 存储所有细胞转换的列表，即死细胞变为活细胞，或活细胞死亡
        List<Transition> transitions = new ArrayList<>();
        // 还有优化的空间：不必遍历矩阵中的所有细胞，可以只考虑包含模式的部分，就像 BASIC 版本一样
        for (int y = 0; y < matrix.length; y++) {
            for (int x = 0; x < matrix[y].length; x++) {
                int neighbours = countNeighbours(y, x);
                if (matrix[y][x] == ALIVE) {
                    if (neighbours < 2 || neighbours > 3) {
                        transitions.add(new Transition(y, x, DEAD));
                        population--;
                    }
                } else { // 细胞已死
                    if (neighbours == 3) {
                        if (x < 2 || x > 67 || y < 2 || y > 21) {
                            invalid = true;
                        }
                        transitions.add(new Transition(y, x, ALIVE));
                        population++;
                    }
                }
            }
        }
        // 将所有转换应用到矩阵中
        transitions.forEach(t -> matrix[t.y()][t.x()] = t.newState());
        generation++;
    }

    private int countNeighbours(int y, int x) {
        int neighbours = 0;
        for (int row = Math.max(y - 1, 0); row <= Math.min(y + 1, matrix.length - 1); row++) {
            for (int col = Math.max(x - 1, 0); col <= Math.min(x + 1, matrix[row].length - 1); col++) {
                if (row == y && col == x) {
                    continue;
                }
                if (matrix[row][col] == ALIVE) {
                    neighbours++;
                }
            }
        }
        return neighbours;
    }
    // 读取用户输入的图案模式
    private void readPattern() {
        // 提示用户输入图案
        System.out.println("ENTER YOUR PATTERN:");
        // 创建存储用户输入的列表
        List<String> lines = new ArrayList<>();
        String line;
        // 初始化最大行长度为0
        int maxLineLength = 0;
        // 初始化读取状态为true
        boolean reading = true;
        // 循环读取用户输入
        while (reading) {
            // 提示用户输入
            System.out.print("? ");
            // 读取用户输入的行
            line = consoleReader.nextLine();
            // 如果用户输入为"done"，则结束读取
            if (line.equalsIgnoreCase("done")) {
                reading = false;
            } else {
                // 将用户输入的行添加到列表中，并替换其中的'.'为空格
                lines.add(line.replace('.', ' '));
                // 更新最大行长度
                maxLineLength = Math.max(maxLineLength, line.length());
            }
        }
        // 填充矩阵
        fillMatrix(lines, maxLineLength);
    }

    // 填充矩阵
    private void fillMatrix(List<String> lines, int maxLineLength) {
        // 计算x和y的最小值
        float xMin = 33 - maxLineLength / 2f;
        float yMin = 11 - lines.size() / 2f;
        // 遍历每一行
        for (int y = 0; y < lines.size(); y++) {
            String line = lines.get(y);
            // 遍历每一列
            for (int x = 1; x <= line.length(); x++) {
                // 如果当前位置为'*'，则将矩阵中对应位置设置为ALIVE，并增加population计数
                if (line.charAt(x-1) == '*') {
                    matrix[floor(yMin + y)][floor(xMin + x)] = ALIVE;
                    population++;
                }
            }
        }
    }

    // 向下取整
    private int floor(float f) {
        return (int) Math.floor(f);
    }

    // 打印游戏标题
    private void printGameHeader() {
        // 打印游戏标题和作者信息
        printIndented(34, "LIFE");
        printIndented(15, "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        // 打印空行
        System.out.println(NEWLINE.repeat(3));
    }

    // 打印带缩进的字符串
    private void printIndented(int spaces, String str) {
        System.out.println(" ".repeat(spaces) + str);
    }

    // 打印当前代的矩阵
    private void printGeneration() {
        // 打印当前代的标题
        printGenerationHeader();
        // 遍历矩阵的每个位置，打印'*'或空格
        for (int y = 0; y < matrix.length; y++) {
            for (int x = 0; x < matrix[y].length; x++) {
                System.out.print(matrix[y][x] == 1 ? "*" : " ");
            }
            System.out.println();
        }
    }
    // 打印生成的标题信息，包括代数、种群数量和是否有效的标识
    private void printGenerationHeader() {
        // 如果标记为无效，则显示"INVALID!"，否则显示空字符串
        String invalidText = invalid ? "INVALID!" : "";
        // 格式化输出生成的标题信息
        System.out.printf("GENERATION: %-13d POPULATION: %d %s\n", generation, population, invalidText);
    }

    /**
     * 启动程序的主方法
     *
     * @param args 命令行参数:
     *             <pre>-s: 在每一代后停止 (按回车键继续)</pre>
     * @throws Exception 如果发生错误
     */
    public static void main(String[] args) throws Exception {
        // 创建 Life 对象并启动程序
        new Life(args).start();
    }
// 结束 Transition 记录的定义
}

/**
 * 代表矩阵中单个单元格的状态变化。
 *
 * @param y 单元格的 y 坐标（行）
 * @param x 单元格的 x 坐标（列）
 * @param newState 单元格的新状态（DEAD 或 ALIVE）
 */
record Transition(int y, int x, byte newState) { }
```