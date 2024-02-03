# `.\PaddleOCR\ppstructure\table\table_metric\table_metric.py`

```py
# 导入所需的库和模块
from rapidfuzz.distance import Levenshtein
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from collections import deque
from .parallel import parallel_process
from tqdm import tqdm

# 定义一个自定义的树结构类 TableTree，继承自 Tree 类
class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        # 初始化树节点的标签、列跨度、行跨度、内容和子节点
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    # 定义一个方法用于以括号表示法展示树结构
    def bracket(self):
        """Show tree using brackets notation"""
        # 如果节点标签为 'td'，则展示标签、列跨度、行跨度和内容
        if self.tag == 'td':
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        # 遍历子节点，递归调用 bracket 方法
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)

# 定义一个自定义的配置类 CustomConfig，继承自 Config 类
class CustomConfig(Config):
    # 定义一个方法用于比较树的属性
    def rename(self, node1, node2):
        """Compares attributes of trees"""
        # 如果节点标签、列跨度或行跨度不相等，则返回 1
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        # 如果节点标签为 'td'，且内容不为空，则计算内容的 Levenshtein 距离
        if node1.tag == 'td':
            if node1.content or node2.content:
                return Levenshtein.normalized_distance(node1.content, node2.content)
        return 0.

# 定义一个自定义的配置类 CustomConfig_del_short，继承自 Config 类
    # 定义一个方法用于重命名节点，比较两个节点的属性
    def rename(self, node1, node2):
        """Compares attributes of trees"""
        # 如果节点的标签、列数或行数不相等，则返回1
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        # 如果节点的标签为 'td'
        if node1.tag == 'td':
            # 如果节点1或节点2有内容
            if node1.content or node2.content:
                # 备注内容
                #print('before')
                #print(node1.content, node2.content)
                #print('after')
                # 将节点1和节点2的内容存储在变量中
                node1_content = node1.content
                node2_content = node2.content
                # 如果节点1的内容长度小于3，则用 ['####'] 替换
                if len(node1_content) < 3:
                    node1_content = ['####']
                # 如果节点2的内容长度小于3，则用 ['####'] 替换
                if len(node2_content) < 3:
                    node2_content = ['####']   
                # 返回节点1和节点2内容的Levenshtein标准化距离
                return Levenshtein.normalized_distance(node1_content, node2_content)
        # 如果节点标签不为 'td'，则返回0
        return 0.
class CustomConfig_del_block(Config):
    # 自定义配置类，继承自 Config 类
    def rename(self, node1, node2):
        # 重命名方法，比较两个树的属性
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            # 如果节点标签不同或者 colspan、rowspan 属性不同，则返回 1
            return 1.
        if node1.tag == 'td':
            # 如果节点标签为 'td'
            if node1.content or node2.content:
                # 如果节点内容不为空
                node1_content = node1.content
                node2_content = node2.content
                while ' '  in node1_content:
                    # 循环删除节点内容中的空格
                    print(node1_content.index(' '))
                    node1_content.pop(node1_content.index(' '))
                while ' ' in node2_content:
                    # 循环删除节点内容中的空格
                    print(node2_content.index(' '))
                    node2_content.pop(node2_content.index(' '))
                return Levenshtein.normalized_distance(node1_content, node2_content)
                # 返回节点内容的 Levenshtein 标准化距离
        return 0.
    
class TEDS(object):
    # TEDS 类
    ''' Tree Edit Distance basead Similarity
    '''
    # 基于树编辑距离的相似度

    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        # 初始化方法
        assert isinstance(n_jobs, int) and (
            n_jobs >= 1), 'n_jobs must be an integer greather than 1'
        # 断言 n_jobs 是整数且大于等于 1
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        # 标记方法，对表格单元格进行标记
        ''' Tokenizes table cells
        '''
        # 标记表格单元格
        self.__tokens__.append('<%s>' % node.tag)
        # 添加节点标签到标记列表中
        if node.text is not None:
            self.__tokens__ += list(node.text)
        # 如果节点文本不为空，则将文本添加到标记列表中
        for n in node.getchildren():
            self.tokenize(n)
        # 递归调用标记方法
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        # 如果节点标签不为 'unk'，则添加结束标记到标记列表中
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)
        # 如果节点标签不为 'td' 且尾部文本不为空，则将尾部文本添加到标记列表中
    # 将 HTML 树转换为 apted 所需的格式
    def load_html_tree(self, node, parent=None):
        ''' Converts HTML tree to the format required by apted
        '''
        # 声明全局变量 __tokens__
        global __tokens__
        # 如果节点的标签是 'td'
        if node.tag == 'td':
            # 如果只需要结构，创建空列表 cell
            if self.structure_only:
                cell = []
            else:
                # 否则，初始化 self.__tokens__ 为空列表，对节点进行标记化
                self.__tokens__ = []
                self.tokenize(node)
                # 从第二个到倒数第二个元素复制到 cell 中
                cell = self.__tokens__[1:-1].copy()
            # 创建新的 TableTree 节点
            new_node = TableTree(node.tag,
                                 int(node.attrib.get('colspan', '1')),
                                 int(node.attrib.get('rowspan', '1')),
                                 cell, *deque())
        else:
            # 如果节点的标签不是 'td'，创建新的 TableTree 节点
            new_node = TableTree(node.tag, None, None, None, *deque())
        # 如果存在父节点，将新节点添加到父节点的 children 列表中
        if parent is not None:
            parent.children.append(new_node)
        # 如果节点的标签不是 'td'，递归处理节点的子节点
        if node.tag != 'td':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        # 如果不存在父节点，返回新节点
        if parent is None:
            return new_node
    # 定义一个方法，用于计算给定样本的预测值和真实值之间的TEDS分数
    def evaluate(self, pred, true):
        ''' Computes TEDS score between the prediction and the ground truth of a
            given sample
        '''
        # 如果预测值或真实值为空，则返回0.0
        if (not pred) or (not true):
            return 0.0
        # 创建一个HTML解析器对象，用于解析HTML内容，去除注释，指定编码为utf-8
        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        # 使用指定的解析器解析预测值和真实值的HTML内容
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        # 如果预测值和真实值都包含body/table元素
        if pred.xpath('body/table') and true.xpath('body/table'):
            # 获取预测值和真实值中的第一个body/table元素
            pred = pred.xpath('body/table')[0]
            true = true.xpath('body/table')[0]
            # 如果存在需要忽略的节点，则从预测值和真实值中去除这些节点
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            # 计算预测值和真实值中节点的数量
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            # 取节点数量的最大值作为节点数
            n_nodes = max(n_nodes_pred, n_nodes_true)
            # 加载预测值和真实值的HTML树
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            # 计算预测值和真实值之间的编辑距离
            distance = APTED(tree_pred, tree_true,
                             CustomConfig()).compute_edit_distance()
            # 返回TEDS分数，即1减去编辑距离除以节点数的比例
            return 1.0 - (float(distance) / n_nodes)
        else:
            # 如果预测值和真实值中没有body/table元素，则返回0.0
            return 0.0
    # 批量计算一组样本的预测和真实值之间的TEDS分数
    def batch_evaluate(self, pred_json, true_json):
        ''' Computes TEDS score between the prediction and the ground truth of
            a batch of samples
            @params pred_json: {'FILENAME': 'HTML CODE', ...}
            @params true_json: {'FILENAME': {'html': 'HTML CODE'}, ...}
            @output: {'FILENAME': 'TEDS SCORE', ...}
        '''
        # 获取真实值字典中的所有样本文件名
        samples = true_json.keys()
        # 如果只有一个进程
        if self.n_jobs == 1:
            # 逐个计算每个样本的TEDS分数
            scores = [self.evaluate(pred_json.get(
                filename, ''), true_json[filename]['html']) for filename in tqdm(samples)]
        else:
            # 构建输入参数列表
            inputs = [{'pred': pred_json.get(
                filename, ''), 'true': true_json[filename]['html']} for filename in samples]
            # 并行处理输入参数列表，计算TEDS分数
            scores = parallel_process(
                inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1)
        # 将文件名和对应的TEDS分数组成字典返回
        scores = dict(zip(samples, scores))
        return scores

    # 批量计算一组样本的预测和真实值之间的TEDS分数
    def batch_evaluate_html(self, pred_htmls, true_htmls):
        ''' Computes TEDS score between the prediction and the ground truth of
            a batch of samples
        '''
        # 如果只有一个进程
        if self.n_jobs == 1:
            # 逐个计算每个样本的TEDS分数
            scores = [self.evaluate(pred_html, true_html) for (
                pred_html, true_html) in zip(pred_htmls, true_htmls)]
        else:
            # 构建输入参数列表
            inputs = [{"pred": pred_html, "true": true_html} for(
                pred_html, true_html) in zip(pred_htmls, true_htmls)]
            # 并行处理输入参数列表，计算TEDS分数
            scores = parallel_process(
                inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1)
        return scores
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == '__main__':
    # 导入 json 模块
    import json
    # 导入 pprint 模块
    import pprint
    # 打开并读取名为 'sample_pred.json' 的文件，将其解析为 JSON 格式
    with open('sample_pred.json') as fp:
        pred_json = json.load(fp)
    # 打开并读取名为 'sample_gt.json' 的文件，将其解析为 JSON 格式
    with open('sample_gt.json') as fp:
        true_json = json.load(fp)
    # 创建 TEDS 对象，指定并行处理的任务数为 4
    teds = TEDS(n_jobs=4)
    # 对预测结果和真实结果进行批量评估，返回评估分数
    scores = teds.batch_evaluate(pred_json, true_json)
    # 创建 PrettyPrinter 对象
    pp = pprint.PrettyPrinter()
    # 使用 PrettyPrinter 对象打印评估分数
    pp.pprint(scores)
```