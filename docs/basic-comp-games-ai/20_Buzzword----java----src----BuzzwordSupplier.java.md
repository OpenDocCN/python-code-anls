# `basic-computer-games\20_Buzzword\java\src\BuzzwordSupplier.java`

```
import java.util.Random;
import java.util.function.Supplier;

/**
 * 一个字符串供应商，提供无尽的随机流行语。
 */
public class BuzzwordSupplier implements Supplier<String> {

    private static final String[] SET_1 = {
            "ABILITY","BASAL","BEHAVIORAL","CHILD-CENTERED",
            "DIFFERENTIATED","DISCOVERY","FLEXIBLE","HETEROGENEOUS",
            "HOMOGENEOUS","MANIPULATIVE","MODULAR","TAVISTOCK",
            "INDIVIDUALIZED" };

    private static final String[] SET_2 = {
            "LEARNING","EVALUATIVE","OBJECTIVE",
            "COGNITIVE","ENRICHMENT","SCHEDULING","HUMANISTIC",
            "INTEGRATED","NON-GRADED","TRAINING","VERTICAL AGE",
            "MOTIVATIONAL","CREATIVE" };

    private static final String[] SET_3 = {
            "GROUPING","MODIFICATION", "ACCOUNTABILITY","PROCESS",
            "CORE CURRICULUM","ALGORITHM", "PERFORMANCE",
            "REINFORCEMENT","OPEN CLASSROOM","RESOURCE", "STRUCTURE",
            "FACILITY","ENVIRONMENT" };

    private final Random random = new Random();

    /**
     * 通过连接三个词组中的随机单词来创建一个流行语。
     */
    @Override
    public String get() {
        return SET_1[random.nextInt(SET_1.length)] + ' ' +
                SET_2[random.nextInt(SET_2.length)] + ' ' +
                SET_3[random.nextInt(SET_3.length)];
    }
}
```