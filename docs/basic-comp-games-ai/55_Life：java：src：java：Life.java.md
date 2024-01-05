# `d:/src/tocomm/basic-computer-games\55_Life\java\src\java\Life.java`

```
import java.util.ArrayList;  // 导入 ArrayList 类
import java.util.List;  // 导入 List 接口
import java.util.Scanner;  // 导入 Scanner 类

/**
 * The Game of Life class.<br>
 * <br>
 * Mimics the behaviour of the BASIC version, however the Java code does not have much in common with the original.
 * <br>
 * Differences in behaviour:
 * <ul>
 *     <li>Input supports the "." character, but it's optional.</li>
 *     <li>Input regarding the "DONE" string is case insensitive.</li>
 * </ul>
 */
public class Life {

    private static final byte DEAD  = 0;  // 定义常量 DEAD，表示细胞死亡
    private static final byte ALIVE = 1;  // 定义常量 ALIVE，表示细胞存活
    private static final String NEWLINE = "\n";  // 定义常量 NEWLINE，表示换行符
    private final Scanner consoleReader = new Scanner(System.in);  // 创建一个用于从控制台读取输入的 Scanner 对象

    private final byte[][] matrix = new byte[21][67];  // 创建一个二维数组，用于表示生命游戏的细胞矩阵
    private int generation = 0;  // 初始化代数为 0
    private int population = 0;  // 初始化种群数量为 0
    boolean stopAfterGen = false;  // 初始化是否在特定代数后停止的标志为 false
    boolean invalid = false;  // 初始化参数是否无效的标志为 false

    /**
     * Constructor.
     *
     * @param args the command line arguments
     */
    public Life(String[] args) {  // 构造函数，接受命令行参数
        parse(args);  // 调用 parse 方法解析参数
    }

    private void parse(String[] args) {  // 解析参数的方法
        for (String arg : args) {  // 遍历参数数组
            if ("-s".equals(arg)) {  # 如果参数是"-s"
                stopAfterGen = true;  # 将stopAfterGen设置为true
                break;  # 跳出循环
            }
        }
    }

    /**
     * Starts the game.
     */
    public void start() {  # 开始游戏的方法
        printGameHeader();  # 打印游戏标题
        readPattern();  # 读取游戏模式
        while (true) {  # 进入无限循环
            printGeneration();  # 打印当前的生成状态
            advanceToNextGeneration();  # 进入下一代
            if (stopAfterGen) {  # 如果stopAfterGen为true
                System.out.print("PRESS ENTER TO CONTINUE");  # 打印提示信息
                consoleReader.nextLine();  # 读取用户输入
            }
        }
    }

    private void advanceToNextGeneration() {
        // store all cell transitions in a list, i.e. if a dead cell becomes alive, or a living cell dies
        List<Transition> transitions = new ArrayList<>(); // 创建一个存储细胞转换的列表，即死细胞变为活细胞，或活细胞死亡
        // there's still room for optimization: instead of iterating over all cells in the matrix,
        // we could consider only the section containing the pattern(s), as in the BASIC version
        for (int y = 0; y < matrix.length; y++) { // 遍历矩阵中的所有行
            for (int x = 0; x < matrix[y].length; x++) { // 遍历当前行中的所有列
                int neighbours = countNeighbours(y, x); // 计算当前细胞周围的活细胞数量
                if (matrix[y][x] == ALIVE) { // 如果当前细胞是活细胞
                    if (neighbours < 2 || neighbours > 3) { // 如果活细胞周围的活细胞数量小于2或大于3
                        transitions.add(new Transition(y, x, DEAD)); // 将当前细胞的转换添加到列表中，状态变为死细胞
                        population--; // 总体细胞数量减一
                    }
                } else { // 如果当前细胞是死细胞
                    if (neighbours == 3) { // 如果死细胞周围的活细胞数量为3
                        if (x < 2 || x > 67 || y < 2 || y > 21) { // 如果细胞位置在指定范围之外
                            invalid = true; // 将invalid标记为true
                        }
                        transitions.add(new Transition(y, x, ALIVE));  // 将新的细胞状态添加到转换集合中
                        population++;  // 更新细胞总数
                    }
                }
            }
        }
        // apply all transitions to the matrix
        transitions.forEach(t -> matrix[t.y()][t.x()] = t.newState());  // 将所有转换应用到矩阵中
        generation++;  // 更新代数
    }

    private int countNeighbours(int y, int x) {
        int neighbours = 0;  // 初始化邻居数量
        for (int row = Math.max(y - 1, 0); row <= Math.min(y + 1, matrix.length - 1); row++) {  // 遍历行
            for (int col = Math.max(x - 1, 0); col <= Math.min(x + 1, matrix[row].length - 1); col++) {  // 遍历列
                if (row == y && col == x) {  // 如果是当前细胞位置，则跳过
                    continue;
                }
                if (matrix[row][col] == ALIVE) {  // 如果邻居细胞是活着的
                    neighbours++;
                }
            }
        }
        return neighbours;
    }
```
这段代码是一个函数的结束标志，表示函数的定义结束。

```
    private void readPattern() {
        System.out.println("ENTER YOUR PATTERN:");
        List<String> lines = new ArrayList<>();
        String line;
        int maxLineLength = 0;
        boolean reading = true;
        while (reading) {
            System.out.print("? ");
            line = consoleReader.nextLine();
            if (line.equalsIgnoreCase("done")) {
                reading = false;
            } else {
                // optional support for the '.' that is needed in the BASIC version
```
这段代码是一个函数的开始标志，表示函数的定义开始。
                lines.add(line.replace('.', ' '));  # 将每行中的句号替换为空格，并添加到列表中
                maxLineLength = Math.max(maxLineLength, line.length());  # 计算每行中最长的长度
            }
        }
        fillMatrix(lines, maxLineLength);  # 调用fillMatrix函数，填充矩阵
    }

    private void fillMatrix(List<String> lines, int maxLineLength) {
        float xMin = 33 - maxLineLength / 2f;  # 计算x轴的最小值
        float yMin = 11 - lines.size() / 2f;  # 计算y轴的最小值
        for (int y = 0; y < lines.size(); y++) {  # 遍历每一行
            String line = lines.get(y);  # 获取当前行的内容
            for (int x = 1; x <= line.length(); x++) {  # 遍历当前行的每个字符
                if (line.charAt(x-1) == '*') {  # 如果当前字符是'*'
                    matrix[floor(yMin + y)][floor(xMin + x)] = ALIVE;  # 在矩阵中对应位置标记为ALIVE
                    population++;  # 计算ALIVE的数量
                }
            }
        }
    }
    # 将浮点数向下取整并转换为整数
    private int floor(float f) {
        return (int) Math.floor(f);
    }

    # 打印游戏标题
    private void printGameHeader() {
        # 打印游戏标题和创意计算的地点
        printIndented(34, "LIFE");
        printIndented(15, "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        # 打印三个换行符
        System.out.println(NEWLINE.repeat(3));
    }

    # 打印带有缩进的字符串
    private void printIndented(int spaces, String str) {
        System.out.println(" ".repeat(spaces) + str);
    }

    # 打印当前生命代的标题
    private void printGeneration() {
        printGenerationHeader();
        # 遍历矩阵并打印相应的符号
        for (int y = 0; y < matrix.length; y++) {
            for (int x = 0; x < matrix[y].length; x++) {
                System.out.print(matrix[y][x] == 1 ? "*" : " ");
    }
    System.out.println();
}
```
这段代码是一个循环的结束标志，打印一个空行。

```
private void printGenerationHeader() {
    String invalidText = invalid ? "INVALID!" : "";
    System.out.printf("GENERATION: %-13d POPULATION: %d %s\n", generation, population, invalidText);
}
```
这是一个私有方法，用于打印每一代的头部信息，包括代数、种群数量和是否有效的标志。

```
/**
 * Main method that starts the program.
 *
 * @param args the command line arguments:
 *             <pre>-s: Stop after each generation (press enter to continue)</pre>
 * @throws Exception if something goes wrong.
 */
public static void main(String[] args) throws Exception {
    new Life(args).start();
}
```
这是程序的主方法，用于启动程序。它接受命令行参数，并创建一个新的Life对象，然后调用start方法开始程序的执行。
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 创建一个字节流对象，用于存储 ZIP 文件的二进制数据
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    zip.close()  # 关闭 ZIP 对象
    return fdict  # 返回结果字典
```

```java
/**
 * Represents a state change for a single cell within the matrix.
 *
 * @param y the y coordinate (row) of the cell
 * @param x the x coordinate (column) of the cell
 * @param newState the new state of the cell (either DEAD or ALIVE)
 */
record Transition(int y, int x, byte newState) { }  // 定义一个 Transition 记录类型，表示矩阵中单个单元格的状态变化
```