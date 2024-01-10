# `basic-computer-games\78_Sine_Wave\java\src\SineWave.java`

```
/**
 * Sine Wave
 *
 * Based on the Sine Wave program here
 * https://github.com/coding-horror/basic-computer-games/blob/main/78%20Sine%20Wave/sinewave.bas
 *
 * Note:  The idea was to create a version of the 1970's Basic program in Java, without introducing
 *        new features - no additional text, error checking, etc has been added.
 */
public class SineWave {

    public static void main(String[] args) {
        // 打印标题
        System.out.println("""
           SINE WAVE
           CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY
           """);
        // 初始化变量isCreative为true
        var isCreative = true;
        // 循环40次，每次增加0.25
        for(var t = 0d; t<40; t += .25) {
            // 计算缩进量
            var indentations = 26 + (int) (25 * Math.sin(t));
            // 打印空格，实现缩进
            System.out.print(" ".repeat(indentations));
            // 根据isCreative的值选择输出不同的单词
            var word = isCreative ? "CREATIVE" : "COMPUTING";
            System.out.println(word);
            // 切换isCreative的值
            isCreative = !isCreative ;
        }
    }
}
```