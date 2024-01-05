# `20_Buzzword\javascript\buzzword.js`

```
// 创建一个新的 Promise 对象，用于处理输入操作
// 创建一个 INPUT 元素，用于接收用户输入
// 在页面上打印提示符 "? "
// 设置 INPUT 元素的类型为文本输入
// 设置输入框的长度为50
input_element.setAttribute("length", "50");
// 将输入框添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
// 让输入框获得焦点
input_element.focus();
// 初始化输入字符串为 undefined
input_str = undefined;
// 添加键盘按下事件监听器，当按下回车键时执行相应操作
input_element.addEventListener("keydown", function (event) {
    // 如果按下的是回车键
    if (event.keyCode == 13) {
        // 将输入框中的值赋给 input_str
        input_str = input_element.value;
        // 从 id 为 "output" 的元素中移除输入框
        document.getElementById("output").removeChild(input_element);
        // 打印输入的字符串
        print(input_str);
        // 打印换行符
        print("\n");
        // 返回输入的字符串
        resolve(input_str);
    }
});
// 结束键盘按下事件监听器的添加
// 结束函数定义
});

// 定义一个名为 tab 的函数，参数为 space
function tab(space)
{
    // 初始化一个空字符串 str
    var str = "";
    // 当 space 大于 0 时执行循环
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串
}

var a = ["",  // 创建一个包含空字符串的数组
         "ABILITY","BASAL","BEHAVIORAL","CHILD-CENTERED",  // 向数组中添加字符串元素
         "DIFFERENTIATED","DISCOVERY","FLEXIBLE","HETEROGENEOUS",
         "HOMOGENEOUS","MANIPULATIVE","MODULAR","TAVISTOCK",
         "INDIVIDUALIZED","LEARNING","EVALUATIVE","OBJECTIVE",
         "COGNITIVE","ENRICHMENT","SCHEDULING","HUMANISTIC",
         "INTEGRATED","NON-GRADED","TRAINING","VERTICAL AGE",
         "MOTIVATIONAL","CREATIVE","GROUPING","MODIFICATION",
         "ACCOUNTABILITY","PROCESS","CORE CURRICULUM","ALGORITHM",
         "PERFORMANCE","REINFORCEMENT","OPEN CLASSROOM","RESOURCE",
         "STRUCTURE","FACILITY","ENVIRONMENT",
         ];  // 创建包含多个字符串的数组

// Main program
async function main()  // 定义一个异步函数
{
    # 打印标题
    print(tab(26) + "BUZZWORD GENERATOR\n");
    # 打印作者信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    # 打印空行
    print("\n");
    print("\n");
    print("\n");
    # 打印程序介绍
    print("THIS PROGRAM PRINTS HIGHLY ACCEPTABLE PHRASES IN\n");
    print("'EDUCATOR-SPEAK' THAT YOU CAN WORK INTO REPORTS\n");
    print("AND SPEECHES.  WHENEVER A QUESTION MARK IS PRINTED,\n");
    print("TYPE A 'Y' FOR ANOTHER PHRASE OR 'N' TO QUIT.\n");
    print("\n");
    print("\n");
    # 打印提示
    print("HERE'S THE FIRST PHRASE:\n");
    # 循环生成短语
    do {
        # 打印随机短语
        print(a[Math.floor(Math.random() * 13 + 1)] + " ");
        print(a[Math.floor(Math.random() * 13 + 14)] + " ");
        print(a[Math.floor(Math.random() * 13 + 27)] + "\n");
        print("\n");
        # 等待用户输入
        y = await input();
    } while (y == "Y") ;
    # 打印结束语
    print("COME BACK WHEN YOU NEED HELP WITH ANOTHER REPORT!\n");
}

# 调用主函数
main();
```