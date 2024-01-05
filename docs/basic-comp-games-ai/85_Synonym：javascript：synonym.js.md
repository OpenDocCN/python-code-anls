# `d:/src/tocomm/basic-computer-games\85_Synonym\javascript\synonym.js`

```
# 定义函数print，用于向页面输出内容
def print(str):
    document.getElementById("output").appendChild(document.createTextNode(str))

# 定义函数input，用于获取用户输入
def input():
    # 声明变量
    var input_element
    var input_str

    # 返回一个Promise对象，用于异步处理用户输入
    return new Promise(function (resolve) {
        # 创建一个input元素
        input_element = document.createElement("INPUT")
        # 输出提示符
        print("? ")
        # 设置input元素的类型为文本
        input_element.setAttribute("type", "text")
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时执行相应操作
input_element.addEventListener("keydown", function (event) {
    # 如果按下的键是回车键
    if (event.keyCode == 13) {
        # 将输入元素的值赋给输入字符串
        input_str = input_element.value;
        # 从 id 为 "output" 的元素中移除输入元素
        document.getElementById("output").removeChild(input_element);
        # 打印输入字符串
        print(input_str);
        # 打印换行符
        print("\n");
        # 解析并返回输入字符串
        resolve(input_str);
    }
});
# 结束键盘按下事件监听器的添加
});
}

# 定义一个函数，用于生成指定数量的空格
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环添加空格到 str 中
    while (space-- > 0)
var la = [];
// 创建一个空数组 la

var tried = [];
// 创建一个空数组 tried

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
// 创建一个二维数组 synonym，包含一些同义词列表，每个列表的第一个元素表示列表长度，后面的元素为同义词

// Main program
async function main()
{
    // 打印标题
    print(tab(33) + "SYNONYM\n");
    // 打印创意计算的信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印空行
    print("\n");
    print("\n");
    print("\n");
    // 初始化 tried 数组，用于记录单词是否已经尝试过
    for (c = 0; c <= synonym.length; c++)
        tried[c] = false;
    // 打印提示信息
    print("A SYNONYM OF A WORD MEANS ANOTHER WORD IN THE ENGLISH\n");
    print("LANGUAGE WHICH HAS THE SAME OR VERY NEARLY THE SAME");
    print(" MEANING.\n");
    print("I CHOOSE A WORD -- YOU TYPE A SYNONYM.\n");
    print("IF YOU CAN'T THINK OF A SYNONYM, TYPE THE WORD 'HELP'\n");
    print("AND I WILL TELL YOU A SYNONYM.\n");
    print("\n");
    // 初始化 c 变量
    c = 0;
    // 循环处理 synonym 数组
    while (c < synonym.length) {
        c++;
        // 使用 do-while 循环来随机选择一个未尝试过的同义词
        do {
            n1 = Math.floor(Math.random() * synonym.length + 1);
        } while (tried[n1]) ;
        tried[n1] = true;
        n2 = synonym[n1][0];    // 同义词列表的长度
        // 该数组用于保存未显示的单词列表
        for (j = 1; j <= n2; j++)
            la[j] = j;
        la[0] = n2;
        g = 1;  // 总是显示第一个单词
        print("\n");
        la[g] = la[la[0]];  // 用最后一个单词替换第一个单词
        la[0] = n2 - 1; // 减小列表的大小
        print("\n");
        // 使用 while 循环来进行交互，询问用户同义词
        while (1) {
            print("     WHAT IS A SYNONYM OF " + synonym[n1][g]);
            str = await input();
            if (str == "HELP") {
                g1 = Math.floor(Math.random() * la[0] + 1);
                print("**** A SYNONYM OF " + synonym[n1][g] + " IS " + synonym[n1][la[g1]] + ".\n");
                print("\n");  # 打印空行
                la[g1] = la[la[0]];  # 将数组 la 中索引为 g1 的元素赋值为数组 la 中索引为 la[0] 的元素
                la[0]--;  # 数组 la 中索引为 0 的元素减一
                continue;  # 继续执行下一次循环
            }
            for (k = 1; k <= n2; k++) {  # 循环，k 从 1 到 n2
                if (g == k)  # 如果 g 等于 k
                    continue;  # 继续执行下一次循环
                if (str == synonym[n1][k])  # 如果 str 等于 synonym[n1][k]
                    break;  # 跳出循环
            }
            if (k > n2) {  # 如果 k 大于 n2
                print("     TRY AGAIN.\n");  # 打印提示信息
            } else {
                print(synonym[n1][Math.floor(Math.random() * 5 + 1)] + "\n");  # 打印 synonym[n1] 中随机索引的元素
                break;  # 跳出循环
            }
        }
    }
    print("\n");  # 打印空行
    print("SYNONYM DRILL COMPLETED.\n");  # 打印提示信息，表示同义词练习已完成

}

main();  # 调用名为main的函数
```