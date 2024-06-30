# `D:\src\scipysrc\scikit-learn\examples\applications\wikipedia_principal_eigenvector.py`

```
# %%
# Download data, if not already on disk
# -------------------------------------
# DBpedia redirects and page links URLs
redirects_url = "http://downloads.dbpedia.org/3.5.1/en/redirects_en.nt.bz2"
page_links_url = "http://downloads.dbpedia.org/3.5.1/en/page_links_en.nt.bz2"

# Extracting filenames from URLs
redirects_filename = redirects_url.rsplit("/", 1)[1]
page_links_filename = page_links_url.rsplit("/", 1)[1]

# List of resources to download
resources = [
    (redirects_url, redirects_filename),
    (page_links_url, page_links_filename),
]

# Loop through resources, download if file does not exist
for url, filename in resources:
    if not os.path.exists(filename):
        print("Downloading data from '%s', please wait..." % url)
        opener = urlopen(url)
        with open(filename, "wb") as f:
            f.write(opener.read())
        print()
    # 使用 BZ2File 打开指定文件，并按行枚举其内容
    for l, line in enumerate(BZ2File(redirects_filename)):
        # 将每行内容按空格分割
        split = line.split()
        # 如果分割后的列表长度不等于4，表示该行格式错误，输出警告信息并继续下一行处理
        if len(split) != 4:
            print("ignoring malformed line: " + line)
            continue
        # 将第一个和第三个分割出的名称进行短名处理，并将结果存入 redirects 字典中
        redirects[short_name(split[0])] = short_name(split[2])
        # 每处理1000000行，输出当前时间戳和行数信息
        if l % 1000000 == 0:
            print("[%s] line: %08d" % (datetime.now().isoformat(), l))

    # 计算重定向关系的传递闭包
    print("Computing the transitive closure of the redirect relation")
    # 遍历 redirects 字典中的键（即源名称）
    for l, source in enumerate(redirects.keys()):
        transitive_target = None
        target = redirects[source]
        seen = {source}
        # 循环直到找不到更多的目标或者出现循环引用
        while True:
            transitive_target = target
            target = redirects.get(target)
            # 如果找不到目标或者目标已经在已见过的集合中，停止循环
            if target is None or target in seen:
                break
            seen.add(target)
        # 将源名称对应的最终目标（传递闭包结果）存入 redirects 字典中
        redirects[source] = transitive_target
        # 每处理1000000行，输出当前时间戳和行数信息
        if l % 1000000 == 0:
            print("[%s] line: %08d" % (datetime.now().isoformat(), l))

    # 返回计算后的 redirects 字典，包含了传递闭包后的重定向关系
    return redirects
# %%
# 计算邻接矩阵
# ------------------------------
def get_adjacency_matrix(redirects_filename, page_links_filename, limit=None):
    """从维基百科数据文件中提取邻接图的稀疏矩阵表示
    
    首先解析重定向信息。

    返回 X，一个 scipy 稀疏邻接矩阵，redirects 是从文章名到文章名的字典，
    index_map 是从文章名到文章索引（整数）的字典。
    """

    print("正在计算重定向映射")
    redirects = get_redirects(redirects_filename)

    print("正在计算整数索引映射")
    index_map = dict()
    links = list()
    for l, line in enumerate(BZ2File(page_links_filename)):
        split = line.split()
        if len(split) != 4:
            print("忽略格式错误的行：" + line)
            continue
        i = index(redirects, index_map, short_name(split[0]))
        j = index(redirects, index_map, short_name(split[2]))
        links.append((i, j))
        if l % 1000000 == 0:
            print("[%s] 行: %08d" % (datetime.now().isoformat(), l))

        if limit is not None and l >= limit - 1:
            break

    print("正在计算邻接矩阵")
    X = sparse.lil_matrix((len(index_map), len(index_map)), dtype=np.float32)
    for i, j in links:
        X[i, j] = 1.0
    del links
    print("转换为 CSR 格式")
    X = X.tocsr()
    print("CSR 转换完成")
    return X, redirects, index_map


# 在处理完500万个链接后停止，以便在内存中处理
X, redirects, index_map = get_adjacency_matrix(
    redirects_filename, page_links_filename, limit=5000000
)
names = {i: name for name, i in index_map.items()}


# %%
# 使用随机化奇异值分解计算主奇异向量
# --------------------------------------------------------
print("正在使用 randomized_svd 计算主奇异向量")
t0 = time()
U, s, V = randomized_svd(X, 5, n_iter=3)
print("计算完成，耗时 %0.3fs" % (time() - t0))

# 打印维基百科中与主奇异向量相关的最强组件的页面名称，这应该类似于最大特征向量
print("根据主奇异向量的顶部维基百科页面")
pprint([names[i] for i in np.abs(U.T[0]).argsort()[-10:]])
pprint([names[i] for i in np.abs(V[0]).argsort()[-10:]])


# %%
# 计算中心性得分
# ---------------------------
def centrality_scores(X, alpha=0.85, max_iter=100, tol=1e-10):
    """使用幂迭代计算主特征向量
    
    此方法也称为 Google PageRank，实现基于 NetworkX 项目的实现
    （同样是 BSD 许可证），版权由以下人员所有：
    
      Aric Hagberg <hagberg@lanl.gov>
      Dan Schult <dschult@colgate.edu>
      Pieter Swart <swart@lanl.gov>
    """
    n = X.shape[0]
    X = X.copy()
    incoming_counts = np.asarray(X.sum(axis=1)).ravel()

    print("正在对图进行归一化")
    # 对于每个非零元素的行索引进行迭代
    for i in incoming_counts.nonzero()[0]:
        # 将 X 中每行对应的数据乘以 1.0 除以 incoming_counts[i]
        X.data[X.indptr[i] : X.indptr[i + 1]] *= 1.0 / incoming_counts[i]
    
    # 创建一个数组 dangle，其中包含与 X 的每行和为零的行对应的元素
    dangle = np.asarray(np.where(np.isclose(X.sum(axis=1), 0), 1.0 / n, 0)).ravel()

    # 初始化得分向量 scores，所有元素初始值为 1/n
    scores = np.full(n, 1.0 / n, dtype=np.float32)
    
    # 进行最大迭代次数 max_iter 次的幂迭代
    for i in range(max_iter):
        # 打印当前迭代次数
        print("power iteration #%d" % i)
        
        # 保存前一次迭代的得分
        prev_scores = scores
        
        # 根据 PageRank 的公式更新得分向量 scores
        scores = (
            alpha * (scores * X + np.dot(dangle, prev_scores))
            + (1 - alpha) * prev_scores.sum() / n
        )
        
        # 检查收敛性：计算归一化 l_inf 范数
        scores_max = np.abs(scores).max()
        if scores_max == 0.0:
            scores_max = 1.0
        
        # 计算当前迭代的误差 err
        err = np.abs(scores - prev_scores).max() / scores_max
        print("error: %0.6f" % err)
        
        # 如果误差小于设定的容差 n * tol，则返回当前的 scores
        if err < n * tol:
            return scores
    
    # 如果达到最大迭代次数仍未收敛，则返回当前的 scores
    return scores
# 打印提示信息，说明正在使用幂迭代方法计算主特征向量分数
print("Computing principal eigenvector score using a power iteration method")

# 记录开始计时
t0 = time()

# 使用指定的数据集 X 计算中心性分数，最大迭代次数为 100
scores = centrality_scores(X, max_iter=100)

# 打印计算完成后的信息，显示计算耗时
print("done in %0.3fs" % (time() - t0))

# 使用排序后的绝对值最大的 10 个分数的索引，获取它们对应的名称，并输出
pprint([names[i] for i in np.abs(scores).argsort()[-10:]])
```