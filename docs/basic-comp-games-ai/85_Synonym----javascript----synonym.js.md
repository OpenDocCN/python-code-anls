# `basic-computer-games\85_Synonym\javascript\synonym.js`

```py
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
                       // 设置输入框类型和长度
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
                                                      // 获取输入的字符串
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

// 定义一个生成指定数量空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义一个数组，包含一些字符串元素
var ra = [, "RIGHT", "CORRECT", "FINE", "GOOD!", "CHECK"];
// 定义一个空数组
var la = [];
// 定义一个空数组
var tried = [];
// 创建一个包含同义词列表的二维数组，每个子数组包含一个数字和若干同义词
var synonym = [[5,"FIRST","START","BEGINNING","ONSET","INITIAL"],
               [5,"SIMILAR","ALIKE","SAME","LIKE","RESEMBLING"],
               [5,"MODEL","PATTERN","PROTOTYPE","STANDARD","CRITERION"],
               [5,"SMALL","INSIGNIFICANT","LITTLE","TINY","MINUTE"],
               [6,"STOP","HALT","STAY","ARREST","CHECK","STANDSTILL"],
               [6,"HOUSE","DWELLING","RESIDENCE","DOMICILE","LODGING","HABITATION"],
               [7,"PIT","HOLE","HOLLOW","WELL","GULF","CHASM","ABYSS"],
               [7,"PUSH","SHOVE","THRUST","PROD","POKE","BUTT","PRESS"],
               [6,"RED","ROUGE","SCARLET","CRIMSON","FLAME","RUBY"],
               [7,"PAIN","SUFFERING","HURT","MISERY","DISTRESS","ACHE","DISCOMFORT"]
               ];

// 主程序
async function main()
{
    // 打印标题
    print(tab(33) + "SYNONYM\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 初始化一个数组，用于记录用户是否尝试过每个同义词
    for (c = 0; c <= synonym.length; c++)
        tried[c] = false;
    // 打印关于同义词的说明
    print("A SYNONYM OF A WORD MEANS ANOTHER WORD IN THE ENGLISH\n");
    print("LANGUAGE WHICH HAS THE SAME OR VERY NEARLY THE SAME");
    print(" MEANING.\n");
    print("I CHOOSE A WORD -- YOU TYPE A SYNONYM.\n");
    print("IF YOU CAN'T THINK OF A SYNONYM, TYPE THE WORD 'HELP'\n");
    print("AND I WILL TELL YOU A SYNONYM.\n");
    print("\n");
    c = 0;
}
    # 当 c 小于同义词列表的长度时执行循环
    while (c < synonym.length) {
        c++;
        # 生成一个不重复的随机数 n1，用于选择同义词列表中的单词
        do {
            n1 = Math.floor(Math.random() * synonym.length + 1);
        } while (tried[n1]) ;
        # 标记已经尝试过的单词
        tried[n1] = true;
        # 获取选定单词的同义词列表的长度
        n2 = synonym[n1][0];    // Length of synonym list
        # 初始化一个数组，用于记录未显示的单词
        // This array keeps a list of words not shown
        for (j = 1; j <= n2; j++)
            la[j] = j;
        la[0] = n2;
        # 初始化 g 为 1，始终显示第一个单词
        g = 1;  // Always show first word
        print("\n");
        # 将第一个单词替换为最后一个单词
        la[g] = la[la[0]];  // Replace first word with last word
        # 减小未显示单词列表的大小
        la[0] = n2 - 1; // Reduce size of list by one.
        print("\n");
        # 进入循环，进行同义词测试
        while (1) {
            # 打印提示信息，询问用户同义词
            print("     WHAT IS A SYNONYM OF " + synonym[n1][g]);
            # 等待用户输入
            str = await input();
            # 如果用户输入 HELP，则随机选择一个未显示的同义词并显示
            if (str == "HELP") {
                g1 = Math.floor(Math.random() * la[0] + 1);
                print("**** A SYNONYM OF " + synonym[n1][g] + " IS " + synonym[n1][la[g1]] + ".\n");
                print("\n");
                la[g1] = la[la[0]];
                la[0]--;
                continue;
            }
            # 检查用户输入的同义词是否正确
            for (k = 1; k <= n2; k++) {
                if (g == k)
                    continue;
                if (str == synonym[n1][k])
                    break;
            }
            # 如果用户输入的同义词不正确，则提示用户重试
            if (k > n2) {
                print("     TRY AGAIN.\n");
            } else {
                # 如果用户输入的同义词正确，则随机选择一个同义词并显示
                print(synonym[n1][Math.floor(Math.random() * 5 + 1)] + "\n");
                break;
            }
        }
    }
    # 同义词测试完成
    print("\n");
    print("SYNONYM DRILL COMPLETED.\n");
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```