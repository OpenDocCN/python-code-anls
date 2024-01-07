# `basic-computer-games\85_Synonym\java\src\SynonymList.java`

```

// 导入必要的类
import java.util.ArrayList;
import java.util.Arrays;

/**
 * 存储一个单词及其同义词列表
 */
public class SynonymList {

    private final String word; // 单词

    private final ArrayList<String> synonyms; // 同义词列表

    // 构造函数，接受单词和同义词数组作为参数
    public SynonymList(String word, String[] synonyms) {
        this.word = word; // 初始化单词
        this.synonyms = new ArrayList<>(Arrays.asList(synonyms)); // 初始化同义词列表
    }

    /**
     * 检查传入的单词是否存在于同义词列表中
     * 注意：不区分大小写
     *
     * @param word 要搜索的单词
     * @return 如果找到则返回true，否则返回false
     */
    public boolean exists(String word) {
        return synonyms.stream().anyMatch(str -> str.equalsIgnoreCase(word)); // 使用流进行检查
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
        // toArray方法的参数确定了结果数组的类型
        return synonyms.toArray(new String[0]);
    }
}

```