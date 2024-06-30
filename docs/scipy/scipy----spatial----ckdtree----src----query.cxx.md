# `D:\src\scipysrc\scipy\scipy\spatial\ckdtree\src\query.cxx`

```
/*
 * Priority queue
 * ==============
 */

// 定义一个联合体用于堆的内容，可以是整数或指针
union heapcontents {
    ckdtree_intp_t intdata; // 整数数据
    void *ptrdata; // 指针数据
};

// 定义堆中的项，包括优先级和内容
struct heapitem {
    double priority; // 优先级
    heapcontents contents; // 内容
};

// 定义堆结构
struct heap {

    std::vector<heapitem> _heap; // 堆的容器，存储堆项
    ckdtree_intp_t n; // 当前堆中项的数量
    ckdtree_intp_t space; // 堆的空间大小

    // 堆的构造函数，初始化为给定大小
    heap(ckdtree_intp_t initial_size) : _heap(initial_size) {
        space = initial_size;
        n = 0;
    }

    // 将一个项推入堆中
    inline void push(heapitem &item) {
        ckdtree_intp_t i;
        heapitem t;
        n++;

        // 如果堆已满，则扩展堆的大小
        if (n > space) _heap.resize(2 * space + 1);
        space = _heap.size();

        i = n - 1;
        _heap[i] = item;

        // 调整堆，维持堆的性质
        while ((i > 0) && (_heap[i].priority < _heap[(i - 1) / 2].priority)) {
            t = _heap[(i - 1) / 2];
            _heap[(i - 1) / 2] = _heap[i];
            _heap[i] = t;
            i = (i - 1) / 2;
        }
    }

    // 查看堆顶的项，不移除
    inline heapitem peek() { return _heap[0]; }

    // 移除堆顶的项
    inline void remove() {
        heapitem t;
        ckdtree_intp_t i, j, k, l, nn;
        _heap[0] = _heap[n - 1];
        n--;

        // 不需要在堆为空时释放空间，整个堆在查询结束时都会释放
        nn = n;
        i = 0;
        j = 1;
        k = 2;

        // 调整堆，维持堆的性质
        while (((j < nn) && (_heap[i].priority > _heap[j].priority)) ||
               ((k < nn) && (_heap[i].priority > _heap[k].priority))) {
            if ((k < nn) && (_heap[j].priority > _heap[k].priority))
                l = k;
            else
                l = j;
            t = _heap[l];
            _heap[l] = _heap[i];
            _heap[i] = t;
            i = l;
            j = 2 * i + 1;
            k = 2 * i + 2;
        }
    }

    // 弹出堆顶的项
    inline heapitem pop() {
        heapitem it = _heap[0];
        remove();
        return it;
    }
};

/*
 * nodeinfo
 * ========
 */

// 节点信息结构体
struct nodeinfo {
    const ckdtreenode *node; // 指向 KD 树节点的指针
    ckdtree_intp_t m; // 维度数
    double min_distance; // 完整的最小距离
    double buf[1]; // 旧式结构体的内存技巧

    // 访问 'packed' 属性的访问器
    inline double *const side_distances() {
        /* 每个方向到查询点的最小距离；随着查询的进行更新 */
        return buf;
    }
    inline double *const maxes() {
        return buf + m; // 返回最大值数组的指针
    }
    inline double *const mins() {
        return buf + 2 * m; // 返回最小值数组的指针
    }

    // 初始化盒子的方法，从另一个节点信息复制
    inline void init_box(const struct nodeinfo *from) {
        std::memcpy(buf, from->buf, sizeof(double) * (3 * m)); // 复制内存
        min_distance = from->min_distance; // 复制最小距离
    }
};
    // 初始化函数，从给定的节点信息中复制数据到当前对象
    inline void init_plain(const struct nodeinfo * from) {
        /* 跳过复制最小值和最大值，因为在这种情况下我们只需要 side_distance 数组。 */
        // 使用 memcpy 复制指定长度的 double 类型数据到当前对象的 buf 数组中
        std::memcpy(buf, from->buf, sizeof(double) * m);
        // 设置最小距离为给定节点信息中的最小距离
        min_distance = from->min_distance;
    }

    // 更新侧向距离函数，根据条件更新 min_distance 和 side_distances 数组中的值
    inline void update_side_distance(const int d, const double new_side_distance, const double p) {
        // 如果 p 是无穷大，则将 min_distance 设为 min_distance 和 new_side_distance 中的较大值
        if (CKDTREE_UNLIKELY(std::isinf(p))) {
            min_distance = ckdtree_fmax(min_distance, new_side_distance);
        } else {
            // 否则，更新 min_distance 为 min_distance 加上 new_side_distance 减去当前 side_distances 数组中索引为 d 的值
            min_distance += new_side_distance - side_distances()[d];
        }
        // 将 side_distances 数组中索引为 d 的值设置为 new_side_distance
        side_distances()[d] = new_side_distance;
    }
};

/*
 * Memory pool for nodeinfo structs
 * ================================
 */

struct nodeinfo_pool {

    std::vector<char*> pool;  // 存储内存池中分配的内存块指针的向量

    ckdtree_intp_t alloc_size;  // 每个nodeinfo结构体的分配大小
    ckdtree_intp_t arena_size;  // 内存池的总大小
    ckdtree_intp_t m;           // 维度数
    char *arena;                 // 指向内存池起始位置的指针
    char *arena_ptr;             // 指向当前可分配位置的指针

    nodeinfo_pool(ckdtree_intp_t m) {
        alloc_size = sizeof(nodeinfo) + (3 * m -1)*sizeof(double);  // 计算每个nodeinfo结构体的实际大小
        alloc_size = 64*(alloc_size/64)+64;  // 对齐到64字节的倍数
        arena_size = 4096*((64*alloc_size)/4096)+4096;  // 计算内存池总大小并对齐到4096字节的倍数
        arena = new char[arena_size];  // 分配内存池空间
        arena_ptr = arena;  // 初始化可分配位置指针
        pool.push_back(arena);  // 将起始位置加入内存池
        this->m = m;  // 初始化维度数
    }

    ~nodeinfo_pool() {
        for (ckdtree_intp_t i = pool.size()-1; i >= 0; --i)
            delete [] pool[i];  // 释放内存池中所有内存块
    }

    inline nodeinfo *allocate() {
        nodeinfo *ni1;
        ckdtree_intp_t m1 = (ckdtree_intp_t)arena_ptr;
        ckdtree_intp_t m0 = (ckdtree_intp_t)arena;
        if ((arena_size-(ckdtree_intp_t)(m1-m0))<alloc_size) {
            arena = new char[arena_size];  // 如果内存不足则重新分配内存池
            arena_ptr = arena;
            pool.push_back(arena);  // 将新的内存块加入内存池
        }
        ni1 = (nodeinfo*)arena_ptr;  // 当前可分配位置即为新的nodeinfo结构体的位置
        ni1->m = m;  // 初始化新nodeinfo结构体的维度数
        arena_ptr += alloc_size;  // 更新可分配位置指针
        return ni1;  // 返回新分配的nodeinfo结构体指针
    }
};

/* k-nearest neighbor search for a single point x */
template <typename MinMaxDist>
static void
query_single_point(const ckdtree *self,
                   double   *result_distances,
                   ckdtree_intp_t      *result_indices,
                   const double  *x,
                   const ckdtree_intp_t     *k,
                   const ckdtree_intp_t     nk,
                   const ckdtree_intp_t     kmax,
                   const double  eps,
                   const double  p,
                   double  distance_upper_bound)
{
    static double inf = strtod("INF", NULL);

    /* memory pool to allocate and automatically reclaim nodeinfo structs */
    nodeinfo_pool nipool(self->m);  // 创建一个内存池用于分配和自动回收nodeinfo结构体

    /*
     * priority queue for chasing nodes
     * entries are:
     *  - minimum distance between the cell and the target
     *  - distances between the nearest side of the cell and the target
     *    the head node of the cell
     */
    heap q(12);  // 创建一个优先队列q，用于追踪节点，初始化容量为12

    /*
     *  priority queue for chasing nodes
     *  entries are:
     *   - minimum distance between the cell and the target
     *   - distances between the nearest side of the cell and the target
     *     the head node of the cell
     */
    heap neighbors(kmax);  // 创建一个优先队列neighbors，用于保存最近邻节点，初始化容量为kmax

    ckdtree_intp_t      i;
    const ckdtree_intp_t m = self->m;
    nodeinfo      *ni1;
    nodeinfo      *ni2;
    double   d;
    double   epsfac;
    heapitem      it, it2, neighbor;
    const ckdtreenode   *node;
    const ckdtreenode   *inode;

    /* set up first nodeifo */
    ni1 = nipool.allocate();  // 分配一个新的nodeinfo结构体，用于存储第一个节点信息
    ni1->node = self->ctree;  // 设置第一个节点为树的根节点

    /* initialize first node, update min_distance */
    ni1->min_distance = 0;  // 初始化第一个节点的最小距离为0
    // 对于每一个维度，将 ni1 结构中的最小值和最大值设置为 self 结构中对应的原始最小值和最大值
    for (i=0; i<m; ++i) {
        ni1->mins()[i] = self->raw_mins[i];
        ni1->maxes()[i] = self->raw_maxes[i];

        // 计算当前维度的侧向距离
        double side_distance;
        // 如果存在原始盒子大小数据，则使用 BoxDist1D 计算侧向距离
        if(self->raw_boxsize_data != NULL) {
            side_distance = BoxDist1D::side_distance_from_min_max(
                self, x[i], self->raw_mins[i], self->raw_maxes[i], i);
        } else {
            // 否则使用 PlainDist1D 计算侧向距离
            side_distance = PlainDist1D::side_distance_from_min_max(
                self, x[i], self->raw_mins[i], self->raw_maxes[i], i);
        }
        // 根据距离的 p 次方进行距离转换
        side_distance = MinMaxDist::distance_p(side_distance, p);

        // 将计算得到的侧向距离存储到 ni1 结构中
        ni1->side_distances()[i] = 0;
        ni1->update_side_distance(i, side_distance, p);
    }

    /* fiddle approximation factor */
    // 调整近似因子 epsfac
    if (CKDTREE_LIKELY(p == 2.0)) {
        double tmp = 1. + eps;
        epsfac = 1. / (tmp*tmp);
    }
    else if (eps == 0.)
        epsfac = 1.;
    else if (std::isinf(p))
        epsfac = 1. / (1. + eps);
    else
        epsfac = 1. / std::pow((1. + eps), p);

    /* internally we represent all distances as distance**p */
    // 内部所有距离表示为距离的 p 次方
    if (CKDTREE_LIKELY(p == 2.0)) {
        double tmp = distance_upper_bound;
        distance_upper_bound = tmp*tmp;
    }
    else if ((!std::isinf(p)) && (!std::isinf(distance_upper_bound)))
        distance_upper_bound = std::pow(distance_upper_bound,p);

    /* heapsort */
    // 使用堆排序对邻居进行排序
    std::vector<heapitem> sorted_neighbors(kmax);
    ckdtree_intp_t nnb = neighbors.n;
    for(i = neighbors.n - 1; i >=0; --i) {
        sorted_neighbors[i] = neighbors.pop();
    }

    /* fill output arrays with sorted neighbors */
    // 将排序好的邻居填充到输出数组中
    for (i = 0; i < nk; ++i) {
        // 如果 k[i]-1 超出了邻居的数量，则将结果索引设为 self 结构中的 n，距离设为无穷大
        if(CKDTREE_UNLIKELY(k[i] - 1 >= nnb)) {
            result_indices[i] = self->n;
            result_distances[i] = inf;
        } else {
            // 否则取出排序后的第 k[i]-1 个邻居
            neighbor = sorted_neighbors[k[i] - 1];
            // 将邻居的索引存入结果索引数组
            result_indices[i] = neighbor.contents.intdata;
            // 根据距离的 p 次方计算距离
            if (CKDTREE_LIKELY(p == 2.0))
                result_distances[i] = std::sqrt(-neighbor.priority);
            else if ((p == 1.) || (std::isinf(p)))
                result_distances[i] = -neighbor.priority;
            else
                result_distances[i] = std::pow((-neighbor.priority),(1./p));
        }
    }
/* 查询 n 个点的 k 个最近邻 */

int
query_knn(const ckdtree      *self,                         // 函数声明，self 是指向 ckdtree 结构的指针，表示 k-d 树对象
          double        *dd,                               // 存储距离的数组指针
          ckdtree_intp_t           *ii,                     // 存储索引的数组指针
          const double  *xx,                                // 输入点集的数组指针
          const ckdtree_intp_t     n,                       // 点集中点的数量
          const ckdtree_intp_t*     k,                      // 查询的最近邻个数
          const ckdtree_intp_t     nk,                      // k 的数量（通常与 n 相同）
          const ckdtree_intp_t     kmax,                    // 最大允许的 k 值
          const double  eps,                                // 精度参数
          const double  p,                                  // Minkowski 距离的 p 参数
          const double  distance_upper_bound)               // 距离上限参数
{
#define HANDLE(cond, kls) \                                 // 条件宏定义，根据条件调用特定的距离计算函数模板
    if(cond) { \
        query_single_point<kls>(self, dd_row, ii_row, xx_row, k, nk, kmax, eps, p, distance_upper_bound); \
    } else

    ckdtree_intp_t m = self->m;                             // 获取 k-d 树的维度 m
    ckdtree_intp_t i;                                       // 循环索引变量

    if(CKDTREE_LIKELY(!self->raw_boxsize_data)) {            // 如果 k-d 树不包含原始框尺寸数据
        for (i=0; i<n; ++i) {                               // 遍历每个输入点
            double *dd_row = dd + (i*nk);                   // 当前点的距离存储位置
            ckdtree_intp_t *ii_row = ii + (i*nk);           // 当前点的索引存储位置
            const double *xx_row = xx + (i*m);              // 当前点的坐标数据起始位置
            HANDLE(CKDTREE_LIKELY(p == 2), MinkowskiDistP2) // 根据 p 值选择合适的距离计算模板
            HANDLE(p == 1, MinkowskiDistP1)
            HANDLE(std::isinf(p), MinkowskiDistPinf)
            HANDLE(1, MinkowskiDistPp)
            {}                                              // 空代码块，结束 HANDLE 宏展开
        }
    } else {                                                // 如果 k-d 树包含原始框尺寸数据
        std::vector<double> row(m);                         // 创建 m 维的临时数组
        double * xx_row = &row[0];                          // 指向临时数组的指针
        int j;                                              // 循环索引变量
        for (i=0; i<n; ++i) {                               // 遍历每个输入点
            double *dd_row = dd + (i*nk);                   // 当前点的距离存储位置
            ckdtree_intp_t *ii_row = ii + (i*nk);           // 当前点的索引存储位置
            const double *old_xx_row = xx + (i*m);          // 原始输入点的坐标数据起始位置
            for(j=0; j<m; ++j) {                            // 遍历每个坐标维度
                xx_row[j] = BoxDist1D::wrap_position(old_xx_row[j], self->raw_boxsize_data[j]);  // 调整坐标以适应框尺寸
            }
            HANDLE(CKDTREE_LIKELY(p == 2), BoxMinkowskiDistP2)  // 根据 p 值选择合适的带框距离计算模板
            HANDLE(p == 1, BoxMinkowskiDistP1)
            HANDLE(std::isinf(p), BoxMinkowskiDistPinf)
            HANDLE(1, BoxMinkowskiDistPp) {}
        }

    }
    return 0;                                               // 函数执行成功返回 0
}
```