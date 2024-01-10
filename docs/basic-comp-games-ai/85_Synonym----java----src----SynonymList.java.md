# `basic-computer-games\85_Synonym\java\src\SynonymList.java`

```
// 导入 ArrayList 和 Arrays 类
import java.util.ArrayList;
import java.util.Arrays;

/**
 * 存储一个单词和该单词的同义词列表
 */
public class SynonymList {

    // 单词
    private final String word;

    // 同义词列表
    private final ArrayList<String> synonyms;

    // 构造函数，接受单词和同义词数组作为参数
    public SynonymList(String word, String[] synonyms) {
        this.word = word;
        // 将同义词数组转换为 ArrayList
        this.synonyms = new ArrayList<>(Arrays.asList(synonyms));
    }

    /**
     * 检查传递给该方法的单词是否存在于同义词列表中
     * 注意：不区分大小写
     *
     * @param word 要搜索的单词
     * @return 如果找到则返回 true，否则返回 false
     */
    public boolean exists(String word) {
        // 使用流式操作检查是否有任何同义词与给定单词不区分大小写匹配
        return synonyms.stream().anyMatch(str -> str.equalsIgnoreCase(word));
    }

    // 获取单词
    public String getWord() {
        return word;
    }

    // 获取同义词列表的大小
    public int size() {
        return synonyms.size();
    }

    /**
     * 以字符串数组格式返回该单词的所有同义词
     *
     * @return 同义词数组
     */
    public String[] getSynonyms() {
        // toArray 方法的参数确定了结果数组的类型
        return synonyms.toArray(new String[0]);
    }
}
```