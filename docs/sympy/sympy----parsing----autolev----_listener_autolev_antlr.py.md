# `D:\src\scipysrc\sympy\sympy\parsing\autolev\_listener_autolev_antlr.py`

```
import collections  # 导入collections模块，用于操作Python内置数据结构的扩展集合
import warnings  # 导入warnings模块，用于处理警告信息

from sympy.external import import_module  # 从sympy.external中导入import_module函数，用于动态导入模块

# 使用import_module函数导入AutolevParser类，如果模块不存在则返回None
autolevparser = import_module('sympy.parsing.autolev._antlr.autolevparser',
                              import_kwargs={'fromlist': ['AutolevParser']})

# 使用import_module函数导入AutolevLexer类，如果模块不存在则返回None
autolevlexer = import_module('sympy.parsing.autolev._antlr.autolevlexer',
                             import_kwargs={'fromlist': ['AutolevLexer']})

# 使用import_module函数导入AutolevListener类，如果模块不存在则返回None
autolevlistener = import_module('sympy.parsing.autolev._antlr.autolevlistener',
                                import_kwargs={'fromlist': ['AutolevListener']})

# 从autolevparser模块中获取AutolevParser类，如果模块中没有则为None
AutolevParser = getattr(autolevparser, 'AutolevParser', None)

# 从autolevlexer模块中获取AutolevLexer类，如果模块中没有则为None
AutolevLexer = getattr(autolevlexer, 'AutolevLexer', None)

# 从autolevlistener模块中获取AutolevListener类，如果模块中没有则为None
AutolevListener = getattr(autolevlistener, 'AutolevListener', None)


def strfunc(z):
    # 根据z的值返回相应的字符串
    if z == 0:
        return ""
    elif z == 1:
        return "_d"
    else:
        return "_" + "d" * z

def declare_phy_entities(self, ctx, phy_type, i, j=None):
    # 根据phy_type声明物理实体，调用相应的声明函数
    if phy_type in ("frame", "newtonian"):
        declare_frames(self, ctx, i, j)
    elif phy_type == "particle":
        declare_particles(self, ctx, i, j)
    elif phy_type == "point":
        declare_points(self, ctx, i, j)
    elif phy_type == "bodies":
        declare_bodies(self, ctx, i, j)

def declare_frames(self, ctx, i, j=None):
    # 声明框架类型的物理实体
    if "{" in ctx.getText():
        # 如果文本中包含"{"，根据i和j的值构建name1
        if j:
            name1 = ctx.ID().getText().lower() + str(i) + str(j)
        else:
            name1 = ctx.ID().getText().lower() + str(i)
    else:
        name1 = ctx.ID().getText().lower()

    # 构建frame的名称
    name2 = "frame_" + name1

    # 如果ctx.parentCtx.varType()的值为"newtonian"，则更新self.newtonian的值
    if self.getValue(ctx.parentCtx.varType()) == "newtonian":
        self.newtonian = name2

    # 更新symbol_table2，将name1映射到name2
    self.symbol_table2.update({name1: name2})

    # 更新symbol_table，分别将name1 + "1>", name1 + "2>", name1 + "3>"映射到相应的坐标
    self.symbol_table.update({name1 + "1>": name2 + ".x"})
    self.symbol_table.update({name1 + "2>": name2 + ".y"})
    self.symbol_table.update({name1 + "3>": name2 + ".z"})

    # 更新type2，将name1映射到"frame"
    self.type2.update({name1: "frame"})

    # 输出frame的声明信息
    self.write(name2 + " = " + "_me.ReferenceFrame('" + name1 + "')\n")

def declare_points(self, ctx, i, j=None):
    # 声明点类型的物理实体
    if "{" in ctx.getText():
        # 如果文本中包含"{"，根据i和j的值构建name1
        if j:
            name1 = ctx.ID().getText().lower() + str(i) + str(j)
        else:
            name1 = ctx.ID().getText().lower() + str(i)
    else:
        name1 = ctx.ID().getText().lower()

    # 构建point的名称
    name2 = "point_" + name1

    # 更新symbol_table2，将name1映射到name2
    self.symbol_table2.update({name1: name2})

    # 更新type2，将name1映射到"point"
    self.type2.update({name1: "point"})

    # 输出point的声明信息
    self.write(name2 + " = " + "_me.Point('" + name1 + "')\n")

def declare_particles(self, ctx, i, j=None):
    # 声明粒子类型的物理实体
    if "{" in ctx.getText():
        # 如果文本中包含"{"，根据i和j的值构建name1
        if j:
            name1 = ctx.ID().getText().lower() + str(i) + str(j)
        else:
            name1 = ctx.ID().getText().lower() + str(i)
    else:
        name1 = ctx.ID().getText().lower()

    # 构建particle的名称
    name2 = "particle_" + name1

    # 更新symbol_table2，将name1映射到name2
    self.symbol_table2.update({name1: name2})

    # 更新type2，将name1映射到"particle"
    self.type2.update({name1: "particle"})

    # 更新bodies，将name1映射到name2
    self.bodies.update({name1: name2})

    # 输出particle的声明信息
    self.write(name2 + " = " + "_me.Particle('" + name1 + "', " + "_me.Point('" +
                name1 + "_pt" + "'), " + "_sm.Symbol('m'))\n")
# 定义一个方法来声明物体（bodies），更新符号表和类型表
def declare_bodies(self, ctx, i, j=None):
    # 如果文本中包含 `{`，表示存在复杂的标识符
    if "{" in ctx.getText():
        # 如果 j 存在，则使用形如 "id_ij" 的名称
        if j:
            name1 = ctx.ID().getText().lower() + str(i) + str(j)
        else:
            # 否则使用形如 "id_i" 的名称
            name1 = ctx.ID().getText().lower() + str(i)
    else:
        # 否则直接使用标识符的小写形式作为名称
        name1 = ctx.ID().getText().lower()

    # 构建 body 的名称，形如 "body_id_i"
    name2 = "body_" + name1
    # 更新 bodies 字典，将 name1 映射到 name2
    self.bodies.update({name1: name2})
    # 构建质心名称，形如 "body_id_i_cm"
    masscenter = name2 + "_cm"
    # 构建参考框架名称，形如 "body_id_i_f"
    refFrame = name2 + "_f"

    # 更新符号表2，将 name1 映射到 name2
    self.symbol_table2.update({name1: name2})
    # 更新符号表2，将 name1o 映射到 masscenter
    self.symbol_table2.update({name1 + "o": masscenter})
    # 更新符号表，将 name11> 映射到 refFrame.x
    self.symbol_table.update({name1 + "1>": refFrame+".x"})
    # 更新符号表，将 name12> 映射到 refFrame.y
    self.symbol_table.update({name1 + "2>": refFrame+".y"})
    # 更新符号表，将 name13> 映射到 refFrame.z
    self.symbol_table.update({name1 + "3>": refFrame+".z"})

    # 更新类型表2，将 name1 映射到 "bodies"
    self.type2.update({name1: "bodies"})
    # 更新类型表2，将 name1o 映射到 "point"
    self.type2.update({name1+"o": "point"})

    # 写入质心的初始化代码
    self.write(masscenter + " = " + "_me.Point('" + name1 + "_cm" + "')\n")
    # 如果有牛顿模式，则设置质心的速度为 0
    if self.newtonian:
        self.write(masscenter + ".set_vel(" + self.newtonian + ", " + "0)\n")
    # 写入参考框架的初始化代码
    self.write(refFrame + " = " + "_me.ReferenceFrame('" + name1 + "_f" + "')\n")
    
    # 设置一个虚拟的质量和惯性矩阵
    # 这些值将在代码的后续部分通过设置器重置
    self.write(name2 + " = " + "_me.RigidBody('" + name1 + "', " + masscenter + ", " +
                refFrame + ", " + "_sm.symbols('m'), (_me.outer(" + refFrame +
                ".x," + refFrame + ".x)," + masscenter + "))\n")

# 定义一个方法来计算惯性函数
def inertia_func(self, v1, v2, l, frame):
    # 如果 v1 对应的类型为 "particle"
    if self.type2[v1] == "particle":
        # 将惯性函数追加到列表 l 中
        l.append("_me.inertia_of_point_mass(" + self.bodies[v1] + ".mass, " + self.bodies[v1] +
                 ".point.pos_from(" + self.symbol_table2[v2] + "), " + frame + ")")
    # 如果类型2中对应于v1的值为"bodies"
    elif self.type2[v1] == "bodies":
        # 惯性是相对质心定义的。
        # 如果惯性点与v1+"o"相同
        if self.inertia_point[v1] == v1 + "o":
            # 询问点也是质心
            if v2 == self.inertia_point[v1]:
                # 将符号表2中对应v1的符号加入惯性数组的第一个分量
                l.append(self.symbol_table2[v1] + ".inertia[0]")

            # 询问点不是质心
            else:
                # 将bodies字典中v1对应的物体的惯性数组的第一个分量加上以质心为参考点的点质量惯性函数
                l.append(self.bodies[v1] + ".inertia[0]" + " + " +
                         "_me.inertia_of_point_mass(" + self.bodies[v1] +
                         ".mass, " + self.bodies[v1] + ".masscenter" +
                         ".pos_from(" + self.symbol_table2[v2] +
                         "), " + frame + ")")

        # 惯性是相对于其他点定义的
        else:
            # 询问点是定义的点
            if v2 == self.inertia_point[v1]:
                # 将符号表2中对应v1的符号加入惯性数组的第一个分量
                l.append(self.symbol_table2[v1] + ".inertia[0]")
            # 询问点是质心
            elif v2 == v1 + "o":
                # 将bodies字典中v1对应的物体的惯性数组的第一个分量减去以惯性点为参考点的点质量惯性函数
                l.append(self.bodies[v1] + ".inertia[0]" + " - " +
                         "_me.inertia_of_point_mass(" + self.bodies[v1] +
                         ".mass, " + self.bodies[v1] + ".masscenter" +
                         ".pos_from(" + self.symbol_table2[self.inertia_point[v1]] +
                         "), " + frame + ")")
            # 询问点是其他点
            else:
                # 将bodies字典中v1对应的物体的惯性数组的第一个分量减去以惯性点为参考点的点质量惯性函数，再加上以v2为参考点的点质量惯性函数
                l.append(self.bodies[v1] + ".inertia[0]" + " - " +
                         "_me.inertia_of_point_mass(" + self.bodies[v1] +
                         ".mass, " + self.bodies[v1] + ".masscenter" +
                         ".pos_from(" + self.symbol_table2[self.inertia_point[v1]] +
                         "), " + frame + ")" + " + " +
                         "_me.inertia_of_point_mass(" + self.bodies[v1] +
                         ".mass, " + self.bodies[v1] + ".masscenter" +
                         ".pos_from(" + self.symbol_table2[v2] +
                         "), " + frame + ")")
# 处理常量声明的方法，例如：Constants F = 3, g = 9.81
def processConstants(self, ctx):
    # 获取常量名，并转换为小写
    name = ctx.ID().getText().lower()
    
    # 检查声明中是否包含等号，如果有，则表示声明有赋值
    if "=" in ctx.getText():
        # 更新符号表，将常量名映射到自身
        self.symbol_table.update({name: name})
        # 写入生成的代码行，将常量名和值转换为字符串表示，并添加到生成的代码中
        self.write(self.symbol_table[name] + " = " + "_sm.S(" + self.getValue(ctx.getChild(2)) + ")\n")
        # 更新常量类型为"constants"
        self.type.update({name: "constants"})
        return

    # 处理不带赋值的常量声明，例如：Constants A, B
    else:
        if "{" not in ctx.getText():
            # 将常量名添加到符号表，并设置类型为"constants"
            self.symbol_table[name] = name
            self.type[name] = "constants"

    # 处理带有符号的常量声明，例如：Constants C+, D-
    if ctx.getChildCount() == 2:
        # 设置常量的符号，"+" 表示非负，"-" 表示非正
        if ctx.getChild(1).getText() == "+":
            self.sign[name] = "+"
        elif ctx.getChild(1).getText() == "-":
            self.sign[name] = "-"
    else:
        if "{" not in ctx.getText():
            # 如果没有指定符号，默认为"o"
            self.sign[name] = "o"

    # 处理带有花括号的常量声明，例如：Constants K{4}, a{1:2, 1:2}, b{1:2}
    if "{" in ctx.getText():
        if ":" in ctx.getText():
            # 解析花括号内的数字范围
            num1 = int(ctx.INT(0).getText())
            num2 = int(ctx.INT(1).getText()) + 1
        else:
            num1 = 1
            num2 = int(ctx.INT(0).getText()) + 1

        if ":" in ctx.getText():
            if "," in ctx.getText():
                # 解析第二个花括号内的数字范围
                num3 = int(ctx.INT(2).getText())
                num4 = int(ctx.INT(3).getText()) + 1
                # 循环遍历所有可能的组合，并添加到符号表中
                for i in range(num1, num2):
                    for j in range(num3, num4):
                        key = name + str(i) + str(j)
                        self.symbol_table[key] = key
                        self.type[key] = "constants"
                        self.var_list.append(key)
                        self.sign[key] = "o"
            else:
                # 解析第一个花括号内的数字范围
                for i in range(num1, num2):
                    key = name + str(i)
                    self.symbol_table[key] = key
                    self.type[key] = "constants"
                    self.var_list.append(key)
                    self.sign[key] = "o"

        elif "," in ctx.getText():
            # 处理不带冒号的花括号内的声明
            for i in range(1, int(ctx.INT(0).getText()) + 1):
                for j in range(1, int(ctx.INT(1).getText()) + 1):
                    key = name + str(i) + str(j)
                    self.symbol_table[key] = key
                    self.type[key] = "constants"
                    self.var_list.append(key)
                    self.sign[key] = "o"

        else:
            # 处理仅有一个数字的花括号内的声明
            for i in range(num1, num2):
                key = name + str(i)
                self.symbol_table[key] = key
                self.type[key] = "constants"
                self.var_list.append(key)
                self.sign[key] = "o"
    # 检查当前文本内容中是否不包含左花括号 '{'，如果不包含则执行下面的语句
    if "{" not in ctx.getText():
        # 将当前的变量名 name 添加到 self.var_list 列表末尾
        self.var_list.append(name)
# 将一组变量中标记为 'o' 的变量过滤出来，存入列表 l1
l1 = list(filter(lambda x: self.sign[x] == "o", self.var_list))

# 将一组变量中标记为 '+' 的变量过滤出来，存入列表 l2
l2 = list(filter(lambda x: self.sign[x] == "+", self.var_list))

# 将一组变量中标记为 '-' 的变量过滤出来，存入列表 l3
l3 = list(filter(lambda x: self.sign[x] == "-", self.var_list))

# 尝试获取设置中的复数属性，若不存在则设置 real 为空字符串，否则根据设置值确定 real 属性为 True 或空字符串
try:
    if self.settings["complex"] == "on":
        real = ", real=True"
    elif self.settings["complex"] == "off":
        real = ""
except Exception:
    real = ", real=True"

# 如果列表 l1 不为空，生成 l1 中变量的声明语句，写入输出
if l1:
    a = ", ".join(l1) + " = " + "_sm.symbols(" + "'" +\
        " ".join(l1) + "'" + real + ")\n"
    self.write(a)

# 如果列表 l2 不为空，生成 l2 中变量的声明语句，写入输出
if l2:
    a = ", ".join(l2) + " = " + "_sm.symbols(" + "'" +\
        " ".join(l2) + "'" + real + ", nonnegative=True)\n"
    self.write(a)

# 如果列表 l3 不为空，生成 l3 中变量的声明语句，写入输出
if l3:
    a = ", ".join(l3) + " = " + "_sm.symbols(" + "'" + \
        " ".join(l3) + "'" + real + ", nonpositive=True)\n"
    self.write(a)

# 清空变量列表
self.var_list = []

def processVariables(self, ctx):
    # 获取当前上下文中的变量名并转换为小写
    name = ctx.ID().getText().lower()

    # 如果当前上下文文本中包含 '='，表示该变量有赋值操作，生成相应的赋值语句写入输出
    if "=" in ctx.getText():
        text = name + "'"*(ctx.getChildCount()-3)
        self.write(text + " = " + self.getValue(ctx.expr()) + "\n")
        return

    # 处理形如 'Variables qA, qB' 的变量声明
    if ctx.getChildCount() == 1:
        # 在符号表中为变量添加条目
        self.symbol_table[name] = name

        # 如果父上下文的第一个子上下文的值为特定类型之一，更新变量类型字典
        if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
            self.type.update({name: self.getValue(ctx.parentCtx.getChild(0))})

        # 将变量名添加到变量列表中，并标记其符号为 0
        self.var_list.append(name)
        self.sign[name] = 0

    # 处理形如 'Variables x', 'y'' 的变量声明
    elif "'" in ctx.getText() and "{" not in ctx.getText():
        # 计算变量名中 ' 符号的数量，更新最大符号数目
        if ctx.getText().count("'") > self.maxDegree:
            self.maxDegree = ctx.getText().count("'")

        # 对每个子上下文的变量名添加符号标记，并更新符号表和类型字典
        for i in range(ctx.getChildCount()):
            self.sign[name + strfunc(i)] = i
            self.symbol_table[name + "'"*i] = name + strfunc(i)
            if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
                self.type.update({name + "'"*i: self.getValue(ctx.parentCtx.getChild(0))})
            self.var_list.append(name + strfunc(i))

def writeVariables(self, ctx):
    # 输出符号表和变量表，暂时注释掉
    #print(self.sign)
    #print(self.symbol_table)
    # 如果 self.var_list 不为空，则执行以下代码块
    if self.var_list:
        # 遍历从 0 到 self.maxDegree+1 的整数序列
        for i in range(self.maxDegree+1):
            # 如果 i 等于 0，则初始化 j 和 t 为空字符串
            if i == 0:
                j = ""
                t = ""
            else:
                # 否则将 i 转换为字符串，并设置 t 为逗号加空格
                j = str(i)
                t = ", "
            
            # 初始化空列表 l
            l = []
            # 对于 self.var_list 中满足 self.sign[x] == i 的元素 x 进行筛选
            for k in list(filter(lambda x: self.sign[x] == i, self.var_list)):
                # 如果 i 等于 0，则直接将 k 添加到列表 l
                if i == 0:
                    l.append(k)
                # 如果 i 等于 1，则将 k 的最后一个字符之前的部分添加到列表 l
                if i == 1:
                    l.append(k[:-1])
                # 如果 i 大于 1，则将 k 的最后两个字符之前的部分添加到列表 l
                if i > 1:
                    l.append(k[:-2])
            
            # 将满足条件的变量名连接成字符串，形如 "var1, var2 = _me.dynamicsymbols('var1, var2', 1), var3 = _me.dynamicsymbols('var3', 2)"
            a = ", ".join(list(filter(lambda x: self.sign[x] == i, self.var_list))) + " = " +\
                "_me.dynamicsymbols(" + "'" + " ".join(l) + "'" + t + j + ")\n"
            
            # 初始化空列表 l
            l = []
            # 将生成的字符串 a 写入到某处（可能是文件或输出流）
            self.write(a)
        
        # 将 self.maxDegree 重置为 0
        self.maxDegree = 0
    
    # 将 self.var_list 置为空列表
    self.var_list = []
# 处理虚数类型变量声明，将变量名添加到符号表、类型表和变量列表中
def processImaginary(self, ctx):
    # 获取上下文中的标识符文本，并转换为小写格式
    name = ctx.ID().getText().lower()
    # 在符号表中以变量名为键存储变量名本身
    self.symbol_table[name] = name
    # 在类型表中以变量名为键存储变量类型为 "imaginary"
    self.type[name] = "imaginary"
    # 将变量名添加到变量列表中
    self.var_list.append(name)

# 生成虚数类型变量声明的代码，并写入目标输出
def writeImaginary(self, ctx):
    # 构建用于声明变量的字符串a，形如 "var1, var2 = _sm.symbols('var1 var2')\n"
    a = ", ".join(self.var_list) + " = " + "_sm.symbols(" + "'" + \
        " ".join(self.var_list) + "')\n"
    # 构建用于设定变量为虚数的字符串b，形如 "var1, var2 = _sm.I\n"
    b = ", ".join(self.var_list) + " = " + "_sm.I\n"
    # 将生成的代码写入目标输出
    self.write(a)
    self.write(b)
    # 清空变量列表，以便处理下一组虚数类型变量声明
    self.var_list = []
```