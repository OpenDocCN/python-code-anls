# `.\pandas-ta\tests\test_ext_indicator_momentum.py`

```
# 从config模块中导入sample_data变量
from .config import sample_data
# 从context模块中导入pandas_ta模块
from .context import pandas_ta
# 从unittest模块中导入skip和TestCase类
from unittest import skip, TestCase
# 从pandas模块中导入DataFrame类
from pandas import DataFrame

# 定义测试类TestMomentumExtension，继承自TestCase类
class TestMomentumExtension(TestCase):
    # 类方法setUpClass，用于设置测试类的初始状态
    @classmethod
    def setUpClass(cls):
        # 初始化数据，使用sample_data
        cls.data = sample_data

    # 类方法tearDownClass，用于清理测试类的状态
    @classmethod
    def tearDownClass(cls):
        # 删除数据
        del cls.data

    # 实例方法setUp，用于设置每个测试方法的初始状态
    def setUp(self): pass
    # 实例方法tearDown，用于清理每个测试方法的状态
    def tearDown(self): pass

    # 测试AO扩展函数
    def test_ao_ext(self):
        # 在数据上计算AO指标，并将结果追加到数据中
        self.data.ta.ao(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"AO_5_34"
        self.assertEqual(self.data.columns[-1], "AO_5_34")

    # 测试APO扩展函数
    def test_apo_ext(self):
        # 在数据上计算APO指标，并将结果追加到数据中
        self.data.ta.apo(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"APO_12_26"
        self.assertEqual(self.data.columns[-1], "APO_12_26")

    # 测试BIAS扩展函数
    def test_bias_ext(self):
        # 在数据上计算BIAS指标，并将结果追加到数据中
        self.data.ta.bias(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"BIAS_SMA_26"
        self.assertEqual(self.data.columns[-1], "BIAS_SMA_26")

    # 测试BOP扩展函数
    def test_bop_ext(self):
        # 在数据上计算BOP指标，并将结果追加到数据中
        self.data.ta.bop(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"BOP"
        self.assertEqual(self.data.columns[-1], "BOP")

    # 测试BRAR扩展函数
    def test_brar_ext(self):
        # 在数据上计算BRAR指标，并将结果追加到数据中
        self.data.ta.brar(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言倒数第二列和最后一列的列名分别为"AR_26"和"BR_26"
        self.assertEqual(list(self.data.columns[-2:]), ["AR_26", "BR_26"])

    # 测试CCI扩展函数
    def test_cci_ext(self):
        # 在数据上计算CCI指标，并将结果追加到数据中
        self.data.ta.cci(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"CCI_14_0.015"
        self.assertEqual(self.data.columns[-1], "CCI_14_0.015")

    # 测试CFO扩展函数
    def test_cfo_ext(self):
        # 在数据上计算CFO指标，并将结果追加到数据中
        self.data.ta.cfo(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"CFO_9"
        self.assertEqual(self.data.columns[-1], "CFO_9")

    # 测试CG扩展函数
    def test_cg_ext(self):
        # 在数据上计算CG指标，并将结果追加到数据中
        self.data.ta.cg(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"CG_10"
        self.assertEqual(self.data.columns[-1], "CG_10")

    # 测试CMO扩展函数
    def test_cmo_ext(self):
        # 在数据上计算CMO指标，并将结果追加到数据中
        self.data.ta.cmo(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"CMO_14"
        self.assertEqual(self.data.columns[-1], "CMO_14")

    # 测试Coppock指标扩展函数
    def test_coppock_ext(self):
        # 在数据上计算Coppock指标，并将结果追加到数据中
        self.data.ta.coppock(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"COPC_11_14_10"
        self.assertEqual(self.data.columns[-1], "COPC_11_14_10")

    # 测试CTI扩展函数
    def test_cti_ext(self):
        # 在数据上计算CTI指标，并将结果追加到数据中
        self.data.ta.cti(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"CTI_12"
        self.assertEqual(self.data.columns[-1], "CTI_12")

    # 测试ER扩展函数
    def test_er_ext(self):
        # 在数据上计算ER指标，并将结果追加到数据中
        self.data.ta.er(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"ER_10"
        self.assertEqual(self.data.columns[-1], "ER_10")

    # 测试ERI扩展函数
    def test_eri_ext(self):
        # 在数据上计算ERI指标，并将结果追加到数据中
        self
    # 测试计算惯性指标，并将结果追加到数据框中
    def test_inertia_ext(self):
        # 调用惯性指标计算函数，将结果追加到数据框中
        self.data.ta.inertia(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"INERTIA_20_14"
        self.assertEqual(self.data.columns[-1], "INERTIA_20_14")

    # 测试计算经过优化的惯性指标，并将结果追加到数据框中
    def test_inertia_refined_ext(self):
        # 调用经过优化的惯性指标计算函数，将结果追加到数据框中
        self.data.ta.inertia(refined=True, append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"INERTIAr_20_14"
        self.assertEqual(self.data.columns[-1], "INERTIAr_20_14")

    # 测试计算经过划分的惯性指标，并将结果追加到数据框中
    def test_inertia_thirds_ext(self):
        # 调用经过划分的惯性指标计算函数，将结果追加到数据框中
        self.data.ta.inertia(thirds=True, append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"INERTIAt_20_14"
        self.assertEqual(self.data.columns[-1], "INERTIAt_20_14")

    # 测试计算KDJ指标，并将结果追加到数据框中
    def test_kdj_ext(self):
        # 调用KDJ指标计算函数，将结果追加到数据框中
        self.data.ta.kdj(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后三列的列名分别为"K_9_3", "D_9_3", "J_9_3"
        self.assertEqual(list(self.data.columns[-3:]), ["K_9_3", "D_9_3", "J_9_3"])

    # 测试计算KST指标，并将结果追加到数据框中
    def test_kst_ext(self):
        # 调用KST指标计算函数，将结果追加到数据框中
        self.data.ta.kst(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后两列的列名分别为"KST_10_15_20_30_10_10_10_15", "KSTs_9"
        self.assertEqual(list(self.data.columns[-2:]), ["KST_10_15_20_30_10_10_10_15", "KSTs_9"])

    # 测试计算MACD指标，并将结果追加到数据框中
    def test_macd_ext(self):
        # 调用MACD指标计算函数，将结果追加到数据框中
        self.data.ta.macd(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后三列的列名分别为"MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"
        self.assertEqual(list(self.data.columns[-3:]), ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"])

    # 测试计算动量指标，并将结果追加到数据框中
    def test_mom_ext(self):
        # 调用动量指标计算函数，将结果追加到数据框中
        self.data.ta.mom(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"MOM_10"
        self.assertEqual(self.data.columns[-1], "MOM_10")

    # 测试计算价格振荡指标，并将结果追加到数据框中
    def test_pgo_ext(self):
        # 调用价格振荡指标计算函数，将结果追加到数据框中
        self.data.ta.pgo(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"PGO_14"
        self.assertEqual(self.data.columns[-1], "PGO_14")

    # 测试计算价格百分比振荡指标，并将结果追加到数据框中
    def test_ppo_ext(self):
        # 调用价格百分比振荡指标计算函数，将结果追加到数据框中
        self.data.ta.ppo(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后三列的列名分别为"PPO_12_26_9", "PPOh_12_26_9", "PPOs_12_26_9"
        self.assertEqual(list(self.data.columns[-3:]), ["PPO_12_26_9", "PPOh_12_26_9", "PPOs_12_26_9"])

    # 测试计算平滑移动线指标，并将结果追加到数据框中
    def test_psl_ext(self):
        # 调用平滑移动线指标计算函数，将结果追加到数据框中
        self.data.ta.psl(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"PSL_12"
        self.assertEqual(self.data.columns[-1], "PSL_12")

    # 测试计算成交量价格振荡指标，并将结果追加到数据框中
    def test_pvo_ext(self):
        # 调用成交量价格振荡指标计算函数，将结果追加到数据框中
        self.data.ta.pvo(append=True)
        # 断言数据类型为
    # 测试 RSX 指标是否正确计算并追加到数据中
    def test_rsx_ext(self):
        # 计算 RSX 指标并追加到数据中
        self.data.ta.rsx(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为 "RSX_14"
        self.assertEqual(self.data.columns[-1], "RSX_14")

    # 测试 RVGI 指标是否正确计算并追加到数据中
    def test_rvgi_ext(self):
        # 计算 RVGI 指标并追加到数据中
        self.data.ta.rvgi(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后两列的列名为 "RVGI_14_4" 和 "RVGIs_14_4"
        self.assertEqual(list(self.data.columns[-2:]), ["RVGI_14_4", "RVGIs_14_4"])

    # 测试斜率指标是否正确计算并追加到数据中
    def test_slope_ext(self):
        # 计算斜率指标并追加到数据中
        self.data.ta.slope(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为 "SLOPE_1"
        self.assertEqual(self.data.columns[-1], "SLOPE_1")

        # 计算角度形式的斜率指标并追加到数据中
        self.data.ta.slope(append=True, as_angle=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为 "ANGLEr_1"
        self.assertEqual(self.data.columns[-1], "ANGLEr_1")

        # 计算以角度形式表示的斜率指标并追加到数据中
        self.data.ta.slope(append=True, as_angle=True, to_degrees=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为 "ANGLEd_1"
        self.assertEqual(self.data.columns[-1], "ANGLEd_1")

    # 测试 SMI 指标是否正确计算并追加到数据中
    def test_smi_ext(self):
        # 计算 SMI 指标并追加到数据中
        self.data.ta.smi(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后三列的列名为 "SMI_5_20_5", "SMIs_5_20_5", "SMIo_5_20_5"
        self.assertEqual(list(self.data.columns[-3:]), ["SMI_5_20_5", "SMIs_5_20_5", "SMIo_5_20_5"])

        # 计算带有自定义标量的 SMI 指标并追加到数据中
        self.data.ta.smi(scalar=10, append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后三列的列名为 "SMI_5_20_5_10.0", "SMIs_5_20_5_10.0", "SMIo_5_20_5_10.0"
        self.assertEqual(list(self.data.columns[-3:]), ["SMI_5_20_5_10.0", "SMIs_5_20_5_10.0", "SMIo_5_20_5_10.0"])

    # 测试挤牌指标是否正确计算并追加到数据中
    def test_squeeze_ext(self):
        # 计算挤牌指标并追加到数据中
        self.data.ta.squeeze(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后四列的列名为 "SQZ_20_2.0_20_1.5", "SQZ_ON", "SQZ_OFF", "SQZ_NO"
        self.assertEqual(list(self.data.columns[-4:]), ["SQZ_20_2.0_20_1.5", "SQZ_ON", "SQZ_OFF", "SQZ_NO"])

        # 计算不带有挤牌 true range 的挤牌指标并追加到数据中
        self.data.ta.squeeze(tr=False, append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后四列的列名为 "SQZ_ON", "SQZ_OFF", "SQZ_NO", "SQZhlr_20_2.0_20_1.5"
        self.assertEqual(list(self.data.columns[-4:]),
            ["SQZ_ON", "SQZ_OFF", "SQZ_NO", "SQZhlr_20_2.0_20_1.5"]
        )

    # 测试高级挤牌指标是否正确计算并追加到数据中
    def test_squeeze_pro_ext(self):
        # 计算高级挤牌指标并追加到数据中
        self.data.ta.squeeze_pro(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后四列的列名为 "SQZPRO_ON_NORMAL", "SQZPRO_ON_NARROW", "SQZPRO_OFF", "SQZPRO_NO"
        self.assertEqual(list(self.data.columns[-4:]), ["SQZPRO_ON_NORMAL", "SQZPRO_ON_NARROW", "SQZPRO_OFF", "SQZPRO_NO"])

        # 计算不带有挤牌 true range 的高级挤牌指标并追加到数据中
        self.data.ta.squeeze_pro(tr=False, append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后四列的列名为 "SQZPRO_ON_NARROW", "SQZPRO_OFF", "SQZPRO_NO", "SQZPROhlr_20_2.0_20_2_1.5_1"
        self.assertEqual(
            list(self.data.columns[-4:]),
            ["SQZPRO_ON_NARROW", "SQZPRO_OFF", "SQZPRO_NO", "SQZPROhlr_20_2.0_20_2_1.5_1"]
        )

    # 测试 ST
    # 测试 Stochastic RSI 扩展功能
    def test_stochrsi_ext(self):
        # 计算 Stochastic RSI，并将结果追加到数据框中
        self.data.ta.stochrsi(append=True)
        # 断言数据对象是 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后两列的列名为指定值
        self.assertEqual(list(self.data.columns[-2:]), ["STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"])

    # 跳过该测试
    @skip
    def test_td_seq_ext(self):
        """TS Sequential DataFrame: Working but SLOW implementation"""
        # 计算 TD Sequential，并将结果追加到数据框中
        self.data.ta.td_seq(show_all=False, append=True)
        # 断言数据对象是 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后两列的列名为指定值
        self.assertEqual(list(self.data.columns[-2:]), ["TD_SEQ_UP", "TD_SEQ_DN"])

        # 计算 TD Sequential，并将结果追加到数据框中
        self.data.ta.td_seq(show_all=True, append=True)
        # 断言数据对象是 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后两列的列名为指定值
        self.assertEqual(list(self.data.columns[-2:]), ["TD_SEQ_UPa", "TD_SEQ_DNa"])

    # 测试 TRIX 扩展功能
    def test_trix_ext(self):
        # 计算 TRIX，并将结果追加到数据框中
        self.data.ta.trix(append=True)
        # 断言数据对象是 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后两列的列名为指定值
        self.assertEqual(list(self.data.columns[-2:]), ["TRIX_30_9", "TRIXs_30_9"])

    # 测试 TSI 扩展功能
    def test_tsi_ext(self):
        # 计算 TSI，并将结果追加到数据框中
        self.data.ta.tsi(append=True)
        # 断言数据对象是 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后两列的列名为指定值
        self.assertEqual(list(self.data.columns[-2:]), ["TSI_13_25_13", "TSIs_13_25_13"])

    # 测试 Ultimate Oscillator 扩展功能
    def test_uo_ext(self):
        # 计算 Ultimate Oscillator，并将结果追加到数据框中
        self.data.ta.uo(append=True)
        # 断言数据对象是 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为指定值
        self.assertEqual(self.data.columns[-1], "UO_7_14_28")

    # 测试 Williams %R 扩展功能
    def test_willr_ext(self):
        # 计算 Williams %R，并将结果追加到数据框中
        self.data.ta.willr(append=True)
        # 断言数据对象是 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为指定值
        self.assertEqual(self.data.columns[-1], "WILLR_14")
```