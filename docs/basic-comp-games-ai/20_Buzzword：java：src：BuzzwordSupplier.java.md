# `d:/src/tocomm/basic-computer-games\20_Buzzword\java\src\BuzzwordSupplier.java`

```
import java.util.Random;  // 导入 Random 类，用于生成随机数
import java.util.function.Supplier;  // 导入 Supplier 接口，用于提供一个泛型类型的值

/**
 * A string supplier that provides an endless stream of random buzzwords.
 */
public class BuzzwordSupplier implements Supplier<String> {  // 创建一个 BuzzwordSupplier 类，实现 Supplier 接口并指定泛型类型为 String

	private static final String[] SET_1 = {  // 创建一个名为 SET_1 的静态常量数组，包含一组字符串
			"ABILITY","BASAL","BEHAVIORAL","CHILD-CENTERED",
			"DIFFERENTIATED","DISCOVERY","FLEXIBLE","HETEROGENEOUS",
			"HOMOGENEOUS","MANIPULATIVE","MODULAR","TAVISTOCK",
			"INDIVIDUALIZED" };

	private static final String[] SET_2 = {  // 创建一个名为 SET_2 的静态常量数组，包含一组字符串
			"LEARNING","EVALUATIVE","OBJECTIVE",
			"COGNITIVE","ENRICHMENT","SCHEDULING","HUMANISTIC",
			"INTEGRATED","NON-GRADED","TRAINING","VERTICAL AGE",
			"MOTIVATIONAL","CREATIVE" };
	private static final String[] SET_3 = {
			"GROUPING","MODIFICATION", "ACCOUNTABILITY","PROCESS",
			"CORE CURRICULUM","ALGORITHM", "PERFORMANCE",
			"REINFORCEMENT","OPEN CLASSROOM","RESOURCE", "STRUCTURE",
			"FACILITY","ENVIRONMENT" };
    // 创建一个包含常用术语的字符串数组

	private final Random random = new Random();
    // 创建一个随机数生成器对象

	/**
	 * Create a buzzword by concatenating a random word from each of the
	 * three word sets.
	 */
	@Override
	public String get() {
        // 从每个字符串数组中随机选择一个单词，并将它们连接起来，形成一个术语
		return SET_1[random.nextInt(SET_1.length)] + ' ' +
				SET_2[random.nextInt(SET_2.length)] + ' ' +
				SET_3[random.nextInt(SET_3.length)];
	}
}
```