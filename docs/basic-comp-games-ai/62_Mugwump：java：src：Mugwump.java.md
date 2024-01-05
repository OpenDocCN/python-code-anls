# `d:/src/tocomm/basic-computer-games\62_Mugwump\java\src\Mugwump.java`

```
import java.util.Arrays;  // 导入 Arrays 类，用于操作数组
import java.util.Scanner;  // 导入 Scanner 类，用于接收用户输入

/**
 * Game of Mugwump
 * <p>
 * Based on the Basic game of Mugwump here
 * https://github.com/coding-horror/basic-computer-games/blob/main/62%20Mugwump/mugwump.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */

public class Mugwump {

    public static final int NUMBER_OF_MUGWUMPS = 4;  // 定义常量，表示 Mugwump 的数量

    public static final int MAX_TURNS = 10;  // 定义常量，表示最大回合数

    public static final int FOUND = -1;  // 定义常量，表示找到 Mugwump 的标志
    // 用于键盘输入
    private final Scanner kbScanner; // 声明一个用于键盘输入的Scanner对象

    private enum GAME_STATE {
        INIT, // 初始化游戏状态
        GAME_START, // 游戏开始状态
        PLAY_TURN // 游戏进行状态
    }

    // 当前游戏状态
    private GAME_STATE gameState; // 声明一个用于存储游戏状态的枚举变量

    int[][] mugwumpLocations; // 声明一个二维数组用于存储mugwump的位置

    int turn; // 声明一个整数变量用于存储回合数

    public Mugwump() {
        kbScanner = new Scanner(System.in); // 初始化键盘输入的Scanner对象
        gameState = GAME_STATE.INIT; // 将游戏状态初始化为INIT
    }

    # 主游戏循环
    def play():

        # 执行游戏循环
        do:
            # 根据游戏状态执行不同的操作
            switch (gameState):

                # 游戏初始化状态
                case INIT:
                    # 执行游戏介绍
                    intro()
                    # 将游戏状态设置为游戏开始
                    gameState = GAME_STATE.GAME_START

                    # 跳出当前循环
                    break

                # 游戏开始状态
                case GAME_START:

                    # 将回合数初始化为0
                    turn = 0
                    // 初始化所有数组元素为0
                    mugwumpLocations = new int[NUMBER_OF_MUGWUMPS][2];

                    // 放置4个mugwumps
                    for (int i = 0; i < NUMBER_OF_MUGWUMPS; i++) {
                        for (int j = 0; j < 2; j++) {
                            mugwumpLocations[i][j] = (int) (Math.random() * 10);
                        }
                    }
                    gameState = GAME_STATE.PLAY_TURN;
                    break;

                case PLAY_TURN:
                    turn++;
                    // 获取玩家猜测的位置
                    String locations = displayTextAndGetInput("TURN NO." + turn + " -- WHAT IS YOUR GUESS? ");
                    int distanceRightGuess = getDelimitedValue(locations, 0);
                    int distanceUpGuess = getDelimitedValue(locations, 1);

                    int numberFound = 0;
                    // 遍历mugwump位置数组
                    for (int i = 0; i < NUMBER_OF_MUGWUMPS; i++) {
                        if (mugwumpLocations[i][0] == FOUND) {  # 如果找到了Mugwump
                            numberFound++;  # 找到的Mugwump数量加一
                        }

                        int right = mugwumpLocations[i][0];  # 获取Mugwump的水平位置
                        int up = mugwumpLocations[i][1];  # 获取Mugwump的垂直位置

                        if (right == distanceRightGuess && up == distanceUpGuess) {  # 如果猜测的位置和实际位置相同
                            if (right != FOUND) {  # 如果Mugwump还没有被找到
                                System.out.println("YOU HAVE FOUND MUGWUMP " + (i + 1));  # 输出找到Mugwump的信息
                                mugwumpLocations[i][0] = FOUND;  # 将Mugwump标记为已找到
                            }
                            numberFound++;  # 找到的Mugwump数量加一
                        } else {
                            // Not found so show distance  # 如果没有找到Mugwump，则显示距离
                            if (mugwumpLocations[i][0] != FOUND) {  # 如果Mugwump还没有被找到
                                double distance = Math.sqrt((Math.pow(right - distanceRightGuess, 2.0d))  # 计算实际位置和猜测位置的距离
                                        + (Math.pow(up - distanceUpGuess, 2.0d)));
# 输出距离MUGWUMP的距离
System.out.println("YOU ARE " + (int) ((distance * 10) / 10) + " UNITS FROM MUGWUMP");

# 如果找到所有的MUGWUMP
if (numberFound == NUMBER_OF_MUGWUMPS) {
    # 输出找到所有MUGWUMP所用的轮数，并将游戏状态设置为开始状态
    System.out.println("YOU GOT THEM ALL IN " + turn + " TURNS!");
    gameState = GAME_STATE.GAME_START;
} 
# 如果轮数超过最大轮数
else if (turn >= MAX_TURNS) {
    # 输出达到最大轮数的提示，并显示MUGWUMP的位置
    System.out.println("SORRY, THAT'S " + MAX_TURNS + " TRIES.  HERE IS WHERE THEY'RE HIDING");
    for (int i = 0; i < NUMBER_OF_MUGWUMPS; i++) {
        if (mugwumpLocations[i][0] != FOUND) {
            System.out.println("MUGWUMP " + (i + 1) + " IS AT ("
                    + mugwumpLocations[i][0] + "," + mugwumpLocations[i][1] + ")");
        }
    }
    # 将游戏状态设置为开始状态
    gameState = GAME_STATE.GAME_START;
}

# 游戏结束？
                    if (gameState != GAME_STATE.PLAY_TURN) {
                        // 如果游戏状态不是PLAY_TURN，则打印以下消息
                        System.out.println("THAT WAS FUN! LET'S PLAY AGAIN.......");
                        System.out.println("FOUR MORE MUGWUMPS ARE NOW IN HIDING.");
                    }
            }
            // 无限循环 - 基于原始的基本版本
        } while (true);
    }

    private void intro() {
        // 打印游戏介绍信息
        System.out.println(addSpaces(33) + "MUGWUMP");
        System.out.println(addSpaces(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("THE OBJECT OF THIS GAME IS TO FIND FOUR MUGWUMPS");
        System.out.println("HIDDEN ON A 10 BY 10 GRID.  HOMEBASE IS POSITION 0,0.");
        System.out.println("ANY GUESS YOU MAKE MUST BE TWO NUMBERS WITH EACH");
        System.out.println("NUMBER BETWEEN 0 AND 9, INCLUSIVE.  FIRST NUMBER");
        System.out.println("IS DISTANCE TO RIGHT OF HOMEBASE AND SECOND NUMBER");
        System.out.println("IS DISTANCE ABOVE HOMEBASE.");
        System.out.println();
        System.out.println("YOU GET 10 TRIES.  AFTER EACH TRY, I WILL TELL");
        System.out.println("YOU HOW FAR YOU ARE FROM EACH MUGWUMP.");
    }
```
这段代码用于在屏幕上打印两条消息。

```
    /**
     * Accepts a string delimited by comma's and returns the pos'th delimited
     * value (starting at count 0).
     *
     * @param text - text with values separated by comma's
     * @param pos  - which position to return a value for
     * @return the int representation of the value
     */
    private int getDelimitedValue(String text, int pos) {
        String[] tokens = text.split(",");
        return Integer.parseInt(tokens[pos]);
    }
```
这段代码是一个私有方法，用于接受一个由逗号分隔的字符串，并返回指定位置的值。

```
    /*
     * Print a message on the screen, then accept input from Keyboard.
     *
```
这段代码是一个注释，说明了下面的代码将在屏幕上打印消息，然后从键盘接受输入。
    /**
     * 显示文本并获取玩家输入的信息
     *
     * @param text 要在屏幕上显示的消息
     * @return 玩家输入的内容
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.nextLine();
    }

    /**
     * 返回包含 x 个空格的字符串
     *
     * @param spaces 所需的空格数
     * @return 包含指定数量空格的字符串
     */
    private String addSpaces(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }
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
```

需要注释的代码：

```python
    public static void main(String[] args) {
        # 创建一个Mugwump对象
        Mugwump mugwump = new Mugwump();
        # 调用Mugwump对象的play方法
        mugwump.play();
    }
}
```