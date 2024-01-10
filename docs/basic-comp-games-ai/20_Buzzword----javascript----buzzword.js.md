# `basic-computer-games\20_Buzzword\javascript\buzzword.js`

```
// 定义一个打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       // 设置输入框类型为文本，长度为50
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       // 初始化输入字符串
                       input_str = undefined;
                       // 监听输入框的按键事件
                       input_element.addEventListener("keydown", function (event) {
                                                      // 如果按下的是回车键
                                                      if (event.keyCode == 13) {
                                                      // 获取输入框的值
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      // 打印换行符
                                                      print("\n");
                                                      // 解析 Promise 对象
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}
# 创建一个包含空字符串和一系列教育术语的数组
var a = ["",
         "ABILITY","BASAL","BEHAVIORAL","CHILD-CENTERED",
         "DIFFERENTIATED","DISCOVERY","FLEXIBLE","HETEROGENEOUS",
         "HOMOGENEOUS","MANIPULATIVE","MODULAR","TAVISTOCK",
         "INDIVIDUALIZED","LEARNING","EVALUATIVE","OBJECTIVE",
         "COGNITIVE","ENRICHMENT","SCHEDULING","HUMANISTIC",
         "INTEGRATED","NON-GRADED","TRAINING","VERTICAL AGE",
         "MOTIVATIONAL","CREATIVE","GROUPING","MODIFICATION",
         "ACCOUNTABILITY","PROCESS","CORE CURRICULUM","ALGORITHM",
         "PERFORMANCE","REINFORCEMENT","OPEN CLASSROOM","RESOURCE",
         "STRUCTURE","FACILITY","ENVIRONMENT",
         ];

# 主程序
async function main()
{
    # 打印标题
    print(tab(26) + "BUZZWORD GENERATOR\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS PROGRAM PRINTS HIGHLY ACCEPTABLE PHRASES IN\n");
    print("'EDUCATOR-SPEAK' THAT YOU CAN WORK INTO REPORTS\n");
    print("AND SPEECHES.  WHENEVER A QUESTION MARK IS PRINTED,\n");
    print("TYPE A 'Y' FOR ANOTHER PHRASE OR 'N' TO QUIT.\n");
    print("\n");
    print("\n");
    print("HERE'S THE FIRST PHRASE:\n");
    # 循环生成教育术语短语
    do {
        # 打印随机选择的教育术语短语
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

# 调用主程序
main();
```