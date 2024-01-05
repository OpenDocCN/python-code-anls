# `85_Synonym\java\src\Synonym.java`

```
import java.util.ArrayList;  # 导入 ArrayList 类
import java.util.Arrays;  # 导入 Arrays 类
import java.util.Scanner;  # 导入 Scanner 类

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

    public static final String[] RANDOM_ANSWERS = {"RIGHT", "CORRECT", "FINE", "GOOD!", "CHECK"};  # 创建包含随机答案的字符串数组

    // Used for keyboard input
    private final Scanner kbScanner;  # 创建私有的 Scanner 对象用于键盘输入
    // List of words and synonyms
    // 词语和其同义词的列表
    private final ArrayList<SynonymList> synonyms;

    // Enum representing different game states
    // 代表不同游戏状态的枚举
    private enum GAME_STATE {
        INIT,       // 初始化
        PLAY,       // 游戏进行中
        GAME_OVER   // 游戏结束
    }

    // Current game state
    // 当前游戏状态
    private GAME_STATE gameState;

    // Index of the current question
    // 当前问题的索引
    private int currentQuestion;

    // Constructor for the Synonym class
    // Synonym 类的构造函数
    public Synonym() {

        // Scanner for user input
        // 用户输入的扫描器
        kbScanner = new Scanner(System.in);
        
        // Initialize the list of synonyms
        // 初始化同义词列表
        synonyms = new ArrayList<>();

        // Set the initial game state to INIT
        // 将初始游戏状态设置为 INIT
        gameState = GAME_STATE.INIT;
    }

    /**
     * Main game loop
     */
    public void play() {

        do {
            switch (gameState) {

                case INIT:
                    intro(); // 调用intro()函数，显示游戏介绍
                    currentQuestion = 0; // 将currentQuestion设置为0

                    // Load data
                    synonyms.add(new SynonymList("FIRST", new String[]{"START", "BEGINNING", "ONSET", "INITIAL"})); // 向synonyms列表中添加一个SynonymList对象，包含"FIRST"和其同义词数组
                    synonyms.add(new SynonymList("SIMILAR", new String[]{"SAME", "LIKE", "RESEMBLING"})); // 向synonyms列表中添加一个SynonymList对象，包含"SIMILAR"和其同义词数组
                    synonyms.add(new SynonymList("MODEL", new String[]{"PATTERN", "PROTOTYPE", "STANDARD", "CRITERION"})); // 向synonyms列表中添加一个SynonymList对象，包含"MODEL"和其同义词数组
                    synonyms.add(new SynonymList("SMALL", new String[]{"INSIGNIFICANT", "LITTLE", "TINY", "MINUTE"})); // 向synonyms列表中添加一个SynonymList对象，包含"SMALL"和其同义词数组
                    synonyms.add(new SynonymList("STOP", new String[]{"HALT", "STAY", "ARREST", "CHECK", "STANDSTILL"})); // 向synonyms列表中添加一个SynonymList对象，包含"STOP"和其同义词数组
                    // 添加同义词列表到同义词集合中
                    synonyms.add(new SynonymList("HOUSE", new String[]{"DWELLING", "RESIDENCE", "DOMICILE", "LODGING", "HABITATION"}));
                    synonyms.add(new SynonymList("PIT", new String[]{"HOLE", "HOLLOW", "WELL", "GULF", "CHASM", "ABYSS"}));
                    synonyms.add(new SynonymList("PUSH", new String[]{"SHOVE", "THRUST", "PROD", "POKE", "BUTT", "PRESS"}));
                    synonyms.add(new SynonymList("RED", new String[]{"ROUGE", "SCARLET", "CRIMSON", "FLAME", "RUBY"}));
                    synonyms.add(new SynonymList("PAIN", new String[]{"SUFFERING", "HURT", "MISERY", "DISTRESS", "ACHE", "DISCOMFORT"}));

                    // 设置游戏状态为PLAY
                    gameState = GAME_STATE.PLAY;
                    break;

                case PLAY:

                    // 获取要询问的单词和其同义词列表
                    SynonymList synonym = synonyms.get(currentQuestion);
                    // 获取玩家输入的答案
                    String getAnswer = displayTextAndGetInput("     WHAT IS A SYNONYM OF " + synonym.getWord() + " ? ");

                    // 如果输入为"HELP"，则随机给出当前单词的一个同义词
                    if (getAnswer.equals("HELP")) {
                        int randomSynonym = (int) (Math.random() * synonym.size());
                        System.out.println("**** A SYNONYM OF " + synonym.getWord() + " IS " + synonym.getSynonyms()[randomSynonym] + ".");
                    } else {
// 检查输入的单词是否在同义词列表中
if (synonym.exists(getAnswer)) {
    // 如果是，给出一个随机的“正确”回答
    System.out.println(RANDOM_ANSWERS[(int) (Math.random() * RANDOM_ANSWERS.length)]);
    currentQuestion++;
    // 我们是否已经到达文件中的最后一个单词/同义词？
    if (currentQuestion == synonyms.size()) {
        // 是的，结束游戏
        System.out.println("SYNONYM DRILL COMPLETED.");
        gameState = GAME_STATE.GAME_OVER;
    }
} else {
    // 单词在同义词列表中不存在
    System.out.println("TRY AGAIN.");
}
    // 打印介绍信息
    private void intro() {
        System.out.println(simulateTabs(33) + "SYNONYM");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("A SYNONYM OF A WORD MEANS ANOTHER WORD IN THE ENGLISH");
        System.out.println("LANGUAGE WHICH HAS THE SAME OR VERY NEARLY THE SAME");
        System.out.println(" MEANING.");
        System.out.println("I CHOOSE A WORD -- YOU TYPE A SYNONYM.");
        System.out.println("IF YOU CAN'T THINK OF A SYNONYM, TYPE THE WORD 'HELP'");
        System.out.println("AND I WILL TELL YOU A SYNONYM.");
        System.out.println();
    }

    /*
     * 在屏幕上打印一条消息，然后从键盘接受输入。
     * 将输入转换为大写。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    // 显示文本并获取用户输入，将输入转换为大写
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next().toUpperCase();
    }

    /**
     * 模拟旧的基本tab(xx)命令，该命令通过xx个空格缩进文本。
     *
     * @param spaces 需要的空格数
     * @return 包含指定数量空格的字符串
     */
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');  // 用空格填充字符数组
        return new String(spacesTemp);  // 将字符数组转换为字符串
    }
}
```