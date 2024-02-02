# `basic-computer-games\44_Hangman\java\Hangman.java`

```py
import java.util.Arrays;  // 导入 Arrays 类
import java.util.LinkedHashSet;  // 导入 LinkedHashSet 类
import java.util.List;  // 导入 List 类
import java.util.Scanner;  // 导入 Scanner 类
import java.util.Set;  // 导入 Set 类
import java.util.stream.Collectors;  // 导入 Collectors 类

/**
 * HANGMAN
 *
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 */

public class Hangman {

    //50 word list
    private final static List<String> words = List.of(  // 创建包含50个单词的不可变列表
            "GUM", "SIN", "FOR", "CRY", "LUG", "BYE", "FLY",
            "UGLY", "EACH", "FROM", "WORK", "TALK", "WITH", "SELF",
            "PIZZA", "THING", "FEIGN", "FIEND", "ELBOW", "FAULT", "DIRTY",
            "BUDGET", "SPIRIT", "QUAINT", "MAIDEN", "ESCORT", "PICKAX",
            "EXAMPLE", "TENSION", "QUININE", "KIDNEY", "REPLICA", "SLEEPER",
            "TRIANGLE", "KANGAROO", "MAHOGANY", "SERGEANT", "SEQUENCE",
            "MOUSTACHE", "DANGEROUS", "SCIENTIST", "DIFFERENT", "QUIESCENT",
            "MAGISTRATE", "ERRONEOUSLY", "LOUDSPEAKER", "PHYTOTOXIC",
            "MATRIMONIAL", "PARASYMPATHOMIMETIC", "THIGMOTROPISM");
    public static void main(String[] args) {
        // 创建一个用于从控制台读取输入的 Scanner 对象
        Scanner scan = new Scanner(System.in);
    
        // 调用打印游戏介绍的方法
        printIntro();
    
        // 创建一个长度为50的整型数组，用于记录单词是否已经被使用过
        int[] usedWords = new int[50];
        // 初始化游戏回合数为1
        int roundNumber = 1;
        // 获取单词总数
        int totalWords = words.size();
        // 初始化是否继续游戏的标志为false
        boolean continueGame = false;
    
        // 开始游戏循环
        do {
            // 如果回合数大于总单词数，输出提示信息并结束游戏
            if (roundNumber > totalWords) {
                System.out.println("\nYOU DID ALL THE WORDS!!");
                break;
            }
    
            int randomWordIndex;
            // 生成一个随机的单词索引，直到找到一个未使用过的单词
            do {
                randomWordIndex = ((int) (totalWords * Math.random())) + 1;
            } while (usedWords[randomWordIndex] == 1);
            usedWords[randomWordIndex] = 1;
    
            // 调用 playRound 方法进行游戏回合，判断玩家是否猜对单词
            boolean youWon = playRound(scan, words.get(randomWordIndex - 1));
            // 根据玩家是否猜对单词输出不同的提示信息
            if (!youWon) {
                System.out.print("\nYOU MISSED THAT ONE.  DO YOU WANT ANOTHER WORD? ");
            } else {
                System.out.print("\nWANT ANOTHER WORD? ");
            }
            // 读取玩家是否想要继续游戏的输入
            final String anotherWordChoice = scan.next();
    
            // 根据玩家的输入判断是否继续游戏
            if (anotherWordChoice.toUpperCase().equals("YES") || anotherWordChoice.toUpperCase().equals("Y")) {
                continueGame = true;
            }
            // 回合数加一
            roundNumber++;
        } while (continueGame);
    
        // 输出结束游戏的提示信息
        System.out.println("\nIT'S BEEN FUN!  BYE FOR NOW.");
    }
    
    // 打印已猜出的字母
    private static void printDiscoveredLetters(char[] D$) {
        System.out.println(new String(D$));
        System.out.println("\n");
    }
    
    // 打印已使用的字母
    private static void printLettersUsed(Set<Character> lettersUsed) {
        System.out.println("\nHERE ARE THE LETTERS YOU USED:");
        System.out.println(lettersUsed.stream()
                .map(Object::toString).collect(Collectors.joining(",")));
        System.out.println("\n");
    }
    
    // 打印游戏介绍
    private static void printIntro() {
        System.out.println("                                HANGMAN");
        System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println("\n\n\n");
    }
# 闭合前面的函数定义
```