# `d:/src/tocomm/basic-computer-games\85_Synonym\java\src\SynonymList.java`

```
import java.util.ArrayList;  // 导入 ArrayList 类
import java.util.Arrays;  // 导入 Arrays 类

/**
 * 存储一个单词和该单词的同义词列表
 */
public class SynonymList {

    private final String word;  // 单词

    private final ArrayList<String> synonyms;  // 同义词列表

    public SynonymList(String word, String[] synonyms) {  // 构造函数，传入单词和同义词数组
        this.word = word;  // 初始化单词
        this.synonyms = new ArrayList<>(Arrays.asList(synonyms));  // 初始化同义词列表
    }

    /**
     * 检查传递给此方法的单词是否存在于同义词列表中
     * 注意：不区分大小写
    *
    * @param word 要搜索的单词
    * @return 如果找到则返回true，否则返回false
    */
   public boolean exists(String word) {
       return synonyms.stream().anyMatch(str -> str.equalsIgnoreCase(word));
   }

   /**
    * 返回单词
    * @return 单词
    */
   public String getWord() {
       return word;
   }

   /**
    * 返回同义词列表的大小
    * @return 同义词列表的大小
    */
   public int size() {
       return synonyms.size();
   }

   /**
    * 以字符串数组格式返回该单词的所有同义词
    * @return 同义词数组
    */
    */
    public String[] getSynonyms() {
        // Parameter to toArray method determines type of the resultant array
        // 使用 toArray 方法的参数确定结果数组的类型
        return synonyms.toArray(new String[0]);
        // 返回同义词数组
    }
}
```