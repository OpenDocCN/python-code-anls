# `basic-computer-games\82_Stars\java\src\Stars.java`

```
// 导入必要的类
import java.util.Arrays;
import java.util.Scanner;

/**
 * 星星游戏
 *
 * 基于这里的基本星星游戏
 * https://github.com/coding-horror/basic-computer-games/blob/main/82%20Stars/stars.bas
 *
 * 注意：本意是在Java中创建一个1970年代Basic游戏的版本，没有引入新功能-没有添加额外的文本，错误检查等。
 */
public class Stars {

    // 定义常量
    public static final int HIGH_NUMBER_RANGE = 100;
    public static final int MAX_GUESSES = 7;

    // 枚举游戏状态
    private enum GAME_STATE {
        STARTING,
        INSTRUCTIONS,
        START_GAME,
        GUESSING,
        WON,
        LOST,
        GAME_OVER
    }

    // 用于键盘输入
    private final Scanner kbScanner;

    // 当前游戏状态
    private GAME_STATE gameState;

    // 玩家猜测次数
    private int playerTotalGuesses;

    // 玩家当前猜测
    private int playerCurrentGuess;

    // 计算机的随机数
    private int computersNumber;

    // 构造函数
    public Stars() {
        // 初始化游戏状态
        gameState = GAME_STATE.STARTING;
        // 初始化键盘扫描器
        kbScanner = new Scanner(System.in);
    }

    /**
     * 主游戏循环
     */
    }

    /**
     * 根据玩家的猜测与计算机的数字的接近程度显示一系列星星-星星越多，越接近数字。
     */
    private void showStars() {
        // 计算玩家猜测与计算机数字的差值
        int d = Math.abs(playerCurrentGuess - computersNumber);
        int starsToShow;
        // 根据差值确定要显示的星星数量
        if(d >=64) {
            starsToShow = 1;
        } else if(d >=32) {
            starsToShow = 2;
        } else if (d >= 16) {
            starsToShow = 3;
        } else if (d >=8) {
            starsToShow = 4;
        } else if( d>= 4) {
            starsToShow = 5;
        } else if(d>= 2) {
            starsToShow = 6;
        } else {
            starsToShow = 7;
        }
        // 打印相应数量的星星
        System.out.println(stars(starsToShow));
    }
    /**
     * 显示一定数量的星号（*）
     * @param number 需要的星号数量
     * @return 用所需星号数量编码的字符串
     */
    private String stars(int number) {
        // 创建一个包含指定数量星号的字符数组
        char[] stars = new char[number];
        // 用星号填充字符数组
        Arrays.fill(stars, '*');
        // 将字符数组转换为字符串并返回
        return new String(stars);
    }

    /**
     * 在每次新游戏开始前初始化变量
     *
     */
    private void init() {
        // 初始化玩家总猜测次数为1
        playerTotalGuesses = 1;
        // 生成计算机的随机数
        computersNumber = randomNumber();
    }

    public void instructions() {
        System.out.println("I AM THINKING OF A WHOLE NUMBER FROM 1 TO " + HIGH_NUMBER_RANGE);
        System.out.println("TRY TO GUESS MY NUMBER.  AFTER YOU GUESS, I");
        System.out.println("WILL TYPE ONE OR MORE STARS (*).  THE MORE");
        System.out.println("STARS I TYPE, THE CLOSER YOU ARE TO MY NUMBER.");
        System.out.println("ONE STAR (*) MEANS FAR AWAY, SEVEN STARS (*******)");
        System.out.println("MEANS REALLY CLOSE!  YOU GET " + MAX_GUESSES + " GUESSES.");
    }

    public void intro() {
        System.out.println("STARS");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
    }

    /**
     * 从键盘获取玩家的猜测
     *
     * @return 玩家的猜测作为整数
     */
    private int playerGuess() {
        return Integer.parseInt((displayTextAndGetInput("YOUR GUESS? ")));
    }

    /**
     * 检查玩家是否输入了Y或YES作为答案
     *
     * @param text  从键盘获取的玩家字符串
     * @return 如果输入了Y或YES，则返回true，否则返回false
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }
    /**
     * 检查字符串是否等于一系列可变数量的值之一
     * 用于检查例如 Y 或 YES 等情况
     * 比较不区分大小写。
     *
     * @param text 源字符串
     * @param values 要与源字符串进行比较的一系列值
     * @return 如果在传递的可变数量的字符串中找到了匹配，则返回 true
     */
    private boolean stringIsAnyValue(String text, String... values) {

        // 循环遍历可变数量的值并逐个进行测试
        for(String val:values) {
            if(text.equalsIgnoreCase(val)) {
                return true;
            }
        }

        // 没有匹配项
        return false;
    }

    /*
     * 在屏幕上显示消息，然后从键盘接受输入。
     *
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * 生成随机数
     *
     * @return 随机数
     */
    private int randomNumber() {
        return (int) (Math.random()
                * (HIGH_NUMBER_RANGE) + 1);
    }
# 闭合前面的函数定义
```