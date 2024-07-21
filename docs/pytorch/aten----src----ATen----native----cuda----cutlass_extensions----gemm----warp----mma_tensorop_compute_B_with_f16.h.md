# `.\pytorch\aten\src\ATen\native\cuda\cutlass_extensions\gemm\warp\mma_tensorop_compute_B_with_f16.h`

```
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Data type of A elements
    typename ElementA_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA_,
    /// Data type of B elements
    typename ElementB_,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB_,
    /// Element type of C matrix
    typename ElementC_,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: gemm::GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: gemm::GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: gemm::GemmShape)
    typename InstructionShape_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM (concept: arch::OpClassSimt or arch::OpClassTensorOp)
    typename Operator_ = arch::OpMultiplyAdd,
    /// Implicit gemm mode
    typename ImplicitGemmMode_ = ImplicitGemmMode::GEMM_NT,
    /// Indicates type of multiplication
    typename TensorOp_ = void
>
struct Mma;



    /// Data type of A elements
    typename ElementA_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA_,
    /// Data type of B elements
    typename ElementB_,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB_,
    /// Element type of C matrix
    typename ElementC_,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC_,



    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: gemm::GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: gemm::GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: gemm::GemmShape)
    typename InstructionShape_,



    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM (concept: arch::OpClassSimt or arch::OpClassTensorOp)
    typename Operator_ = arch::OpMultiplyAdd,
    /// Implicit gemm mode
    typename ImplicitGemmMode_ = ImplicitGemmMode::GEMM_NT,
    /// Indicates type of multiplication
    typename TensorOp_ = void
>



struct Mma;



/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template<
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,



    /// Data type of A elements
    typename ElementA_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA_,
    /// Data type of B elements
    typename ElementB_,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB_,
    /// Element type of C matrix
    typename ElementA_,
    /// Element type of A matrix
    typename LayoutA_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename ElementB_,
    /// Element type of B matrix
    typename LayoutB_,
    /// Layout of B matrix (concept: MatrixLayout)
    typename ElementC_,
    /// Element type of C matrix
    typename LayoutC_,
    /// Layout of C matrix (concept: MatrixLayout)
    typename Policy_,
    /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
    typename SharedMemoryInstructionShape_,
    /// Instruction shape to override shared memory iterators with
    int PartitionsK_ = 1,
    /// Number of partitions along K dimension
    bool AccumulatorsInRowMajor = false,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    typename Enable = bool>
    /// Used for partial specialization
/// 定义一个名为 MmaTensorOpComputeBWithF16 的模板类
class MmaTensorOpComputeBWithF16 {
public:
    /// 声明 warp 级矩阵操作的形状 (概念: GemmShape)
    using Shape = Shape_;

    /// 定义乘数 A 的数据类型
    using ElementA = ElementA_;

    /// 定义乘数 A 的布局
    using LayoutA = LayoutA_;

    /// 定义乘数 B 的数据类型
    using ElementB = ElementB_;

    /// 定义乘数 B 的布局
    using LayoutB = LayoutB_;

    /// 定义累加器矩阵 C 的数据类型
    using ElementC = ElementC_;

    /// 定义累加器矩阵 C 的布局
    using LayoutC = LayoutC_;

    /// 定义 warp 单位中线程的形状 (概念: MmaLanePolicySimt)
    using Policy = Policy_;

    /// 底层矩阵乘法运算符 (概念: arch::Mma)
    using ArchMmaOperator = typename Policy::Operator;

    /// 指示数学运算符
    using MathOperator = typename ArchMmaOperator::Operator;

    /// 来自底层指令的架构标签
    using ArchTag = typename ArchMmaOperator::ArchTag;

    /// 静态断言，确保只支持特定的半精度乘法运算
    static_assert((platform::is_same<typename ArchMmaOperator::ElementA, half_t>::value
                   && platform::is_same<typename ArchMmaOperator::ElementB, half_t>::value)
                      || (platform::is_same<typename ArchMmaOperator::ElementA, bfloat16_t>::value
                          && platform::is_same<typename ArchMmaOperator::ElementB, bfloat16_t>::value
                          && ArchTag::kMinComputeCapability >= 80),
                  "MmaTensorOpCvtBToA only supports underlying HMMA");

    /// 静态断言，确保只支持特定的浮点数类型和架构
    static_assert(platform::is_same<ElementA, half_t>::value
                      || (platform::is_same<ElementA, bfloat16_t>::value && ArchTag::kMinComputeCapability >= 80),
                  "MmaTensorOpCvtBToA only supports Fp16 A or Bf16 A on Ampere+");

    /// 指示矩阵操作的类别
    using OperatorClass = arch::OpClassTensorOp;

    /// 指示底层指令的形状
    using InstructionShape = typename ArchMmaOperator::Shape;

    /// 指定用于共享内存迭代器覆盖的指令形状
    using SharedMemoryInstructionShape = SharedMemoryInstructionShape_;

    /// 静态断言，确保计算指令的 M 维度与加载一致
    static_assert(SharedMemoryInstructionShape::kM == InstructionShape::kM,
                  "M dimension of compute instruction must match load");

    /// 静态断言，确保计算指令的 N 维度与加载一致
    static_assert(SharedMemoryInstructionShape::kN == InstructionShape::kN,
                  "N dimension of compute instruction must match load");

    /// 计算 K 维度分区的扩展因子
    static constexpr int kExpansionFactor = SharedMemoryInstructionShape::kK / InstructionShape::kK;

    /// 静态断言，确保 Shape 的 K 维度是 SharedMemoryInstructionShape 的 K 维度的倍数
    static_assert(!(Shape::kK % SharedMemoryInstructionShape::kK), "");

    /// 定义在 A 操作数上的复杂变换
    static ComplexTransform const kTransformA = ComplexTransform::kNone;

    /// 定义在 B 操作数上的复杂变换
    static ComplexTransform const kTransformB = ComplexTransform::kNone;

    /// 参与 warp 级矩阵乘积的线程数目
    static int const kThreadCount = 32;

    /// K 维度上的分区数目
    static int const kPartitionsK = PartitionsK_;
    /// Iterates over the A operand in memory
    using IteratorA = MmaTensorOpMultiplicandTileIterator<MatrixShape<Shape::kM, Shape::kK>,
                                                          Operand::kA,
                                                          ElementA,
                                                          LayoutA,
                                                          MatrixShape<InstructionShape::kM, InstructionShape::kK>,
                                                          Policy::OpDelta::kRow,
                                                          kThreadCount,
                                                          kPartitionsK>;

    /// Storage for A tile
    using FragmentA = typename IteratorA::Fragment;

    /// Storage for transformed A tile
    using TransformedFragmentA = Array<typename ArchMmaOperator::ElementA, FragmentA::kElements>;

    /// Iterates over the B operand in memory
    using IteratorB =
        MmaTensorOpMultiplicandTileIterator<MatrixShape<Shape::kK, Shape::kN>,
                                            Operand::kB,
                                            ElementB,
                                            LayoutB,
                                            MatrixShape<SharedMemoryInstructionShape::kK, InstructionShape::kN>,
                                            Policy::OpDelta::kRow,
                                            kThreadCount,
                                            kPartitionsK>;

    /// Storage for B tile
    using FragmentB = typename IteratorB::Fragment;

    /// Storage for transformed B tile
    using TransformedFragmentB = Array<typename ArchMmaOperator::ElementB, FragmentB::kElements>;

    /// Iterates over the C operand in memory
    using IteratorC = MmaTensorOpAccumulatorTileIterator<MatrixShape<Shape::kM, Shape::kN>,
                                                         ElementC,
                                                         LayoutC,
                                                         typename ArchMmaOperator::Shape,
                                                         typename Policy::OpDelta>;

    /// Storage for C tile
    using FragmentC = typename IteratorC::Fragment;

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<(Shape::kM + ArchMmaOperator::Shape::kM - 1) / ArchMmaOperator::Shape::kM,
                                      (Shape::kN + ArchMmaOperator::Shape::kN - 1) / ArchMmaOperator::Shape::kN>;

public:
    /// Underlying matrix multiply operator (concept: arch::Mma)
    ArchMmaOperator mma;

public:
    //
    // Methods
    //

    /// Ctor
    CUTLASS_DEVICE
    MmaTensorOpComputeBWithF16() {}

    /// Performs a warp-level matrix multiply-accumulate operation
    CUTLASS_DEVICE



    // 构造函数，无参数
    MmaTensorOpComputeBWithF16() {}

    // 执行 warp 级别的矩阵乘累加操作
    CUTLASS_DEVICE
    // 仿函数的调用运算符重载，用于执行矩阵乘法操作。
    // 参数说明：
    //   D: 输出的结果矩阵片段C
    //   A: 转换后的矩阵片段A
    //   B: 转换后的矩阵片段B
    //   C: 输入的矩阵片段C
    //   warp_tileB_k_offset: warp内部块B的偏移量

    using MmaOperandA = typename ArchMmaOperator::FragmentA;  // 定义MMA操作数A类型
    using MmaOperandB = typename ArchMmaOperator::FragmentB;  // 定义MMA操作数B类型
    using MmaOperandC = typename ArchMmaOperator::FragmentC;  // 定义MMA操作数C类型

    static_assert(
        TransformedFragmentB::kElements == MmaOperandB::kElements * kExpansionFactor * MmaIterations::kColumn,
        "Each thread should have a pack of mma registers for each column iteration AND for the expanded K dim of B");
        // 断言：确保转换后的片段B的元素数量等于MMA操作数B的元素数量乘以扩展因子kExpansionFactor和MMA迭代的列数，
        // 表示每个线程应该为每个列迭代和B的扩展K维度拥有一个MMA寄存器包

    D = C;  // 将输入矩阵片段C的内容复制给输出矩阵片段D

    MmaOperandA const* ptr_A = reinterpret_cast<MmaOperandA const*>(&A);  // 将转换后的片段A解释为MMA操作数A的指针
    MmaOperandB const* ptr_B = reinterpret_cast<MmaOperandB const*>(&B);  // 将转换后的片段B解释为MMA操作数B的指针
    MmaOperandC*       ptr_D = reinterpret_cast<MmaOperandC*>(&D);        // 将输出矩阵片段D解释为MMA操作数C的指针
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
        // 在CUDA架构低于8.0的设备上执行以下代码段
        // 使用蛇形访问顺序以最大化对Rb的重用
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < MmaIterations::kColumn; ++n) {

            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < MmaIterations::kRow; ++m) {

                // 根据n的奇偶性确定m_serpentine的计算方式，以实现蛇形访问
                int m_serpentine = ((n % 2) ? (MmaIterations::kRow - 1 - m) : m);

                // 计算偏移量，用于从matrix B中读取数据
                int n_offsetB = warp_tileB_k_offset + kExpansionFactor * n;
                if (AccumulatorsInRowMajor) {  // 若matrix B已重新排序
                    // 执行MMA操作，将结果写入ptr_D中的适当位置
                    mma(ptr_D[n + m_serpentine * MmaIterations::kColumn],
                        ptr_A[m_serpentine],
                        ptr_B[n_offsetB],
                        ptr_D[n + m_serpentine * MmaIterations::kColumn]);
                }
                else {
                    // 执行MMA操作，将结果写入ptr_D中的适当位置（针对列主序的累加器）
                    mma(ptr_D[m_serpentine + n * MmaIterations::kRow],
                        ptr_A[m_serpentine],
                        ptr_B[n_offsetB],
                        ptr_D[m_serpentine + n * MmaIterations::kRow]);
                }
            }
        }
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        // 在CUDA架构不低于8.0的设备上执行以下代码段
        // 使用蛇形访问顺序以最大化对Ra的重用
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < MmaIterations::kRow; ++m) {

            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < MmaIterations::kColumn; ++n) {

                // 根据m的奇偶性确定n_serpentine的计算方式，以实现蛇形访问
                int n_serpentine = ((m % 2) ? (MmaIterations::kColumn - 1 - n) : n);

                // 计算偏移量，用于从matrix B中读取数据
                int n_serpentine_offsetB = warp_tileB_k_offset + kExpansionFactor * n_serpentine;
                if (AccumulatorsInRowMajor) {  // 若matrix B已重新排序
                    // 执行MMA操作，将结果写入ptr_D中的适当位置
                    mma(ptr_D[n_serpentine + m * MmaIterations::kColumn],
                        ptr_A[m],
                        ptr_B[n_serpentine_offsetB],
                        ptr_D[n_serpentine + m * MmaIterations::kColumn]);
                }
                else {
                    // 执行MMA操作，将结果写入ptr_D中的适当位置（针对列主序的累加器）
                    mma(ptr_D[m + n_serpentine * MmaIterations::kRow],
                        ptr_A[m],
                        ptr_B[n_serpentine_offsetB],
                        ptr_D[m + n_serpentine * MmaIterations::kRow]);
                }
            }
        }
#else
        // 如果未定义适用的CUDA架构版本，则断言失败
        assert(0);
#endif
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace warp
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
```