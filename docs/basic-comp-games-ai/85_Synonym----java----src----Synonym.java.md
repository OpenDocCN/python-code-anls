# `basic-computer-games\85_Synonym\java\src\Synonym.java`

```py
import java.util.ArrayList;  // 导入 ArrayList 类
import java.util.Arrays;  // 导入 Arrays 类
import java.util.Scanner;  // 导入 Scanner 类

/**
 * Game of Synonym
 * <p>
 * Based on the Basic game of Synonym here
 * https://github.com/coding-horror/basic-computer-games/blob/main/85%20Synonym/synonym.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Synonym {

    public static final String[] RANDOM_ANSWERS = {"RIGHT", "CORRECT", "FINE", "GOOD!", "CHECK"};  // 定义常量数组 RANDOM_ANSWERS

    // Used for keyboard input
    private final Scanner kbScanner;  // 创建 Scanner 对象用于键盘输入

    // List of words and synonyms
    private final ArrayList<SynonymList> synonyms;  // 创建 SynonymList 对象的 ArrayList

    private enum GAME_STATE {  // 创建枚举类型 GAME_STATE
        INIT,  // 初始化状态
        PLAY,  // 游戏进行状态
        GAME_OVER  // 游戏结束状态
    }

    // Current game state
    private GAME_STATE gameState;  // 当前游戏状态

    private int currentQuestion;  // 当前问题编号

    public Synonym() {  // 构造函数

        kbScanner = new Scanner(System.in);  // 初始化 Scanner 对象
        synonyms = new ArrayList<>();  // 初始化 ArrayList

        gameState = GAME_STATE.INIT;  // 初始化游戏状态为 INIT
    }

    /**
     * Main game loop
     */
    }

    private void intro() {  // 游戏介绍方法
        System.out.println(simulateTabs(33) + "SYNONYM");  // 打印游戏标题
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印创意计算机的信息
        System.out.println();  // 打印空行
        System.out.println("A SYNONYM OF A WORD MEANS ANOTHER WORD IN THE ENGLISH");  // 打印游戏规则
        System.out.println("LANGUAGE WHICH HAS THE SAME OR VERY NEARLY THE SAME");
        System.out.println(" MEANING.");
        System.out.println("I CHOOSE A WORD -- YOU TYPE A SYNONYM.");
        System.out.println("IF YOU CAN'T THINK OF A SYNONYM, TYPE THE WORD 'HELP'");
        System.out.println("AND I WILL TELL YOU A SYNONYM.");
        System.out.println();
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     * Converts input to uppercase.
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    # 显示文本并获取用户输入，将输入转换为大写后返回
    private String displayTextAndGetInput(String text) {
        # 打印文本
        System.out.print(text);
        # 获取用户输入并转换为大写
        return kbScanner.next().toUpperCase();
    }

    /**
     * 模拟旧的基本tab(xx)命令，该命令通过xx个空格缩进文本。
     *
     * @param spaces 需要的空格数
     * @return 具有指定空格数的字符串
     */
    private String simulateTabs(int spaces) {
        # 创建一个包含指定空格数的字符数组
        char[] spacesTemp = new char[spaces];
        # 用空格填充字符数组
        Arrays.fill(spacesTemp, ' ');
        # 将字符数组转换为字符串并返回
        return new String(spacesTemp);
    }
# 闭合前面的函数定义
```