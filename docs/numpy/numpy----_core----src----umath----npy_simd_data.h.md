# `.\numpy\numpy\_core\src\umath\npy_simd_data.h`

```
#ifndef __NPY_SIMD_DATA_H_
#define __NPY_SIMD_DATA_H_

#if defined NPY_HAVE_AVX512F
#if !(defined(__clang__) && (__clang_major__ < 10 || (__clang_major__ == 10 && __clang_minor__ < 1)))
/*
 * Constants used in vector implementation of float64 exp(x)
 */
#define NPY_RINT_CVT_MAGIC 0x1.8p52           // Magic number for rounding in floating-point operations
#define NPY_INV_LN2_MUL_32 0x1.71547652b82fep+5 // Precomputed constant for logarithm base 2
#define NPY_TANG_NEG_L1 -0x1.62e42fefp-6       // Constant coefficient for tangent polynomial approximation
#define NPY_TANG_NEG_L2 -0x1.473de6af278edp-39 // Constant coefficient for tangent polynomial approximation
#define NPY_TANG_A1 0x1p-1                      // Coefficient for tangent polynomial approximation
#define NPY_TANG_A2 0x1.5555555548f7cp-3        // Coefficient for tangent polynomial approximation
#define NPY_TANG_A3 0x1.5555555545d4ep-5        // Coefficient for tangent polynomial approximation
#define NPY_TANG_A4 0x1.11115b7aa905ep-7        // Coefficient for tangent polynomial approximation
#define NPY_TANG_A5 0x1.6c1728d739765p-10       // Coefficient for tangent polynomial approximation

#if !defined NPY_HAVE_AVX512_SKX || !defined NPY_CAN_LINK_SVML
/* Lookup table for 2^(j/32) */
static npy_uint64 EXP_Table_top[32] = {
    0x3FF0000000000000,    // 2^0
    0x3FF059B0D3158540,    // 2^(1/32)
    0x3FF0B5586CF98900,    // 2^(2/32)
    0x3FF11301D0125B40,    // 2^(3/32)
    0x3FF172B83C7D5140,    // 2^(4/32)
    0x3FF1D4873168B980,    // 2^(5/32)
    0x3FF2387A6E756200,    // 2^(6/32)
    0x3FF29E9DF51FDEC0,    // 2^(7/32)
    0x3FF306FE0A31B700,    // 2^(8/32)
    0x3FF371A7373AA9C0,    // 2^(9/32)
    0x3FF3DEA64C123400,    // 2^(10/32)
    0x3FF44E0860618900,    // 2^(11/32)
    0x3FF4BFDAD5362A00,    // 2^(12/32)
    0x3FF5342B569D4F80,    // 2^(13/32)
    0x3FF5AB07DD485400,    // 2^(14/32)
    0x3FF6247EB03A5580,    // 2^(15/32)
    0x3FF6A09E667F3BC0,    // 2^(16/32)
    0x3FF71F75E8EC5F40,    // 2^(17/32)
    0x3FF7A11473EB0180,    // 2^(18/32)
    0x3FF82589994CCE00,    // 2^(19/32)
    0x3FF8ACE5422AA0C0,    // 2^(20/32)
    0x3FF93737B0CDC5C0,    // 2^(21/32)
    0x3FF9C49182A3F080,    // 2^(22/32)
    0x3FFA5503B23E2540,    // 2^(23/32)
    0x3FFAE89F995AD380,    // 2^(24/32)
    0x3FFB7F76F2FB5E40,    // 2^(25/32)
    0x3FFC199BDD855280,    // 2^(26/32)
    0x3FFCB720DCEF9040,    // 2^(27/32)
    0x3FFD5818DCFBA480,    // 2^(28/32)
    0x3FFDFC97337B9B40,    // 2^(29/32)
    0x3FFEA4AFA2A490C0,    // 2^(30/32)
    0x3FFF50765B6E4540,    // 2^(31/32)
};

static npy_uint64 EXP_Table_tail[32] = {
    0x0000000000000000,    // 2^(-32)
    0x3D0A1D73E2A475B4,    // 2^(-31/32)
    0x3CEEC5317256E308,    // 2^(-30/32)
    0x3CF0A4EBBF1AED93,    // 2^(-29/32)
    0x3D0D6E6FBE462876,    // 2^(-28/32)
    0x3D053C02DC0144C8,    // 2^(-27/32)
    0x3D0C3360FD6D8E0B,    // 2^(-26/32)
    0x3D009612E8AFAD12,    // 2^(-25/32)
    0x3CF52DE8D5A46306,    // 2^(-24/32)
    0x3CE54E28AA05E8A9,    // 2^(-23/32)
    0x3D011ADA0911F09F,    // 2^(-22/32)
    0x3D068189B7A04EF8,    // 2^(-21/32)
    0x3D038EA1CBD7F621,    // 2^(-20/32)
    0x3CBDF0A83C49D86A,    // 2^(-19/32)
    0x3D04AC64980A8C8F,    // 2^(-18/32)
    0x3CD2C7C3E81BF4B7,    // 2^(-17/32)
    0x3CE921165F626CDD,    // 2^(-16/32)
    0x3D09EE91B8797785,    // 2^(-15/32)
    0x3CDB5F54408FDB37,    // 2^(-14/32)
    0x3CF28ACF88AFAB35,    // 2^(-13/32)
    0x3CFB5BA7C55A192D,    // 2^(-12/32)
    0x3D027A280E1F92A0,    // 2^(-11/32)
    0x3CF01C7C46B071F3,    // 2^(-10/32)
    0x3CFC8B424491CAF8,    // 2^(-9/32)
    0x3D06AF439A68BB99,    // 2^(-8/32)
    0x3CDBAA9EC206AD4F,    // 2^(-7/32)
    0x3CFC2220CB12A092,    // 2^(-6/32)
    0x3D048A81E5E8F4A5,    // 2^(-5/32)
    0x3CDC976816BAD9B8,    // 2^(-4/32)
    0x3CFEB968CAC39ED3,    // 2^(-3/32)
    0x3CF9858F73A18F5E,    // 2^(-2/32)
    0x3C99D3E12DD8A18B,    // 2^(-1/32)
};
#endif // #if !defined NPY_HAVE_AVX512_SKX || !defined NPY_CAN_LINK_SVML
#endif
#endif

/*
 * Constants used in vector implementation of exp(x)
 */
#define NPY_RINT_CVT_MAGICf 0x1.800000p+23f         // Magic number for rounding in floating-point operations
#define NPY_CODY_WAITE_LOGE_2_HIGHf -6.93145752e-1f // High part of Cody
#define NPY_COEFF_Q0_EXPf 1.000000000000000000000e+00f
#define NPY_COEFF_Q1_EXPf -2.742335390411667452936e-01f
#define NPY_COEFF_Q2_EXPf 2.159509375685829852307e-02f

/*
 * Constants used in vector implementation of log(x)
 */
#define NPY_COEFF_P0_LOGf 0.000000000000000000000e+00f
#define NPY_COEFF_P1_LOGf 9.999999999999998702752e-01f
#define NPY_COEFF_P2_LOGf 2.112677543073053063722e+00f
#define NPY_COEFF_P3_LOGf 1.480000633576506585156e+00f
#define NPY_COEFF_P4_LOGf 3.808837741388407920751e-01f
#define NPY_COEFF_P5_LOGf 2.589979117907922693523e-02f
#define NPY_COEFF_Q0_LOGf 1.000000000000000000000e+00f
#define NPY_COEFF_Q1_LOGf 2.612677543073109236779e+00f
#define NPY_COEFF_Q2_LOGf 2.453006071784736363091e+00f
#define NPY_COEFF_Q3_LOGf 9.864942958519418960339e-01f
#define NPY_COEFF_Q4_LOGf 1.546476374983906719538e-01f
#define NPY_COEFF_Q5_LOGf 5.875095403124574342950e-03f

/*
 * Lookup table of log(c_k)
 * Reference form: Tang, Ping-Tak Peter. "Table-driven implementation of the
 *     logarithm function in IEEE floating-point arithmetic." ACM Transactions
 *     on Mathematical Software (TOMS) 16.4 (1990): 378-400.
 */
#if defined NPY_HAVE_AVX512F
#if !(defined(__clang__) && (__clang_major__ < 10 || (__clang_major__ == 10 && __clang_minor__ < 1)))
#if !defined NPY_HAVE_AVX512_SKX || !defined NPY_CAN_LINK_SVML
static npy_uint64 LOG_TABLE_TOP[64] = {
    // 64-bit integer array defining the upper 64 bits of each entry in the logarithm lookup table
    0x0000000000000000,  // First entry: 0x0000000000000000
    0x3F8FC0A8B1000000,  // Second entry: 0x3F8FC0A8B1000000
    0x3F9F829B0E780000,  // Third entry: 0x3F9F829B0E780000
    0x3FA77458F6340000,  // Fourth entry: 0x3FA77458F6340000
    0x3FAF0A30C0100000,  // Fifth entry: 0x3FAF0A30C0100000
    0x3FB341D7961C0000,  // Sixth entry: 0x3FB341D7961C0000
    0x3FB6F0D28AE60000,  // Seventh entry: 0x3FB6F0D28AE60000
    0x3FBA926D3A4A0000,  // Eighth entry: 0x3FBA926D3A4A0000
    0x3FBE27076E2A0000,  // Ninth entry: 0x3FBE27076E2A0000
    0x3FC0D77E7CD10000,  // Tenth entry: 0x3FC0D77E7CD10000
    0x3FC29552F8200000,  // Eleventh entry: 0x3FC29552F8200000
    0x3FC44D2B6CCB0000,  // Twelfth entry: 0x3FC44D2B6CCB0000
    0x3FC5FF3070A80000,  // Thirteenth entry: 0x3FC5FF3070A80000
    0x3FC7AB8902110000,  // Fourteenth entry: 0x3FC7AB8902110000
    0x3FC9525A9CF40000,  // Fifteenth entry: 0x3FC9525A9CF40000
    0x3FCAF3C94E810000,  // Sixteenth entry: 0x3FCAF3C94E810000
    0x3FCC8FF7C79B0000,  // Seventeenth entry: 0x3FCC8FF7C79B0000
    0x3FCE27076E2B0000,  // Eighteenth entry: 0x3FCE27076E2B0000
    0x3FCFB9186D5E0000,  // Nineteenth entry: 0x3FCFB9186D5E0000
    0x3FD0A324E2738000,  // Twentieth entry: 0x3FD0A324E2738000
    0x3FD1675CABAB8000,  // Twenty-first entry: 0x3FD1675CABAB8000
    0x3FD22941FBCF8000,  // Twenty-second entry: 0x3FD22941FBCF8000
    0x3FD2E8E2BAE10000,  // Twenty-third entry: 0x3FD2E8E2BAE10000
    0x3FD3A64C55698000,  // Twenty-fourth entry: 0x3FD3A64C55698000
    0x3FD4618BC21C8000,  // Twenty-fifth entry: 0x3FD4618BC21C8000
    0x3FD51AAD872E0000,  // Twenty-sixth entry: 0x3FD51AAD872E0000
    0x3FD5D1BDBF580000,  // Twenty-seventh entry: 0x3FD5D1BDBF580000
    0x3FD686C81E9B0000,  // Twenty-eighth entry: 0x3FD686C81E9B0000
    0x3FD739D7F6BC0000,  // Twenty-ninth entry: 0x3FD739D7F6BC0000
    0x3FD7EAF83B828000,  // Thirtieth entry: 0x3FD7EAF83B828000
    0x3FD89A3386C18000,  // Thirty-first entry: 0x3FD89A3386C18000
    0x3FD947941C210000,  // Thirty-second entry: 0x3FD947941C210000
    0x3FD9F323ECBF8000,  // Thirty-third entry: 0x3FD9F323ECBF8000
    0x3FDA9CEC9A9A0000,  // Thirty-fourth entry: 0x3FDA9CEC9A9A0000
    0x3FDB44F77BCC8000,  // Thirty-fifth entry: 0x3FDB44F77BCC8000
    0x3FDBEB4D9DA70000,  // Thirty-sixth entry: 0x3FDBEB4D9DA70000
    0x3FDC8FF7C79A8000,  // Thirty-seventh entry: 0x3FDC8FF7C79A8000
    0x3FDD32FE7E010000,  // Thirty-eighth entry: 0x3FDD32FE7E010000
    0x3FDDD46A04C20000,  // Thirty-ninth entry: 0x3FDDD46A04C20000
    0x3FDE744261D68000,  // Fortieth entry: 0x3FDE744261D68000
    0x3FDF128F5FAF0000,  // Forty-first entry: 0x3FDF128F5FAF0000
    0x3FDFAF588F790000,  // Forty-second entry: 0x3FDFAF588F790000
    0x3FE02552A5A5C000,  // Forty-third entry: 0x3FE02552A5A5C000
    0x3FE0723E5C1CC000,  // Forty-fourth entry: 0x3FE0723E5C1CC000
    0x3FE0BE72E425400
    # 定义一个包含多个64位整数的列表
    numbers = [
        0xBD5FE0E183092C59,
        0x3D2980267C7E09E4,
        0xBD62303B9CB0D5E1,
        0x3D662A6617CC9717,
        0xBD4717B6B33E44F8,
        0xBD62968C836CC8C2,
        0x3D6AAC6CA17A4554,
        0x3D6E5CBD3D50FFFC,
        0xBD6C69A65A23A170,
        0xBD35B967F4471DFC,
        0x3D6F4799F4F6543E,
        0xBD6B0B0DE3077D7E,
        0xBD537B720E4A694B,
        0x3D65AD1D904C1D4E,
        0xBD600349CC67F9B2,
        0xBD697794F689F843,
        0xBD3A342C2AF0003C,
        0x3D5F1546AAA3361C,
        0x3D50E35F73F7A018,
        0x3D630701CE63EAB9,
        0xBD3A6976F5EB0963,
        0x3D5D309C2CC91A85,
        0xBD6D0B1C68651946,
        0xBD609EC17A426426,
        0xBD3F4BD8DB0A7CC1,
        0x3D4394A11B1C1EE4,
        0x3D54AEC442BE1015,
        0xBD67FCB18ED9D603,
        0x3D67E1B259D2F3DA,
        0xBD6ED2A52C73BF78,
        0x3D56FABA4CDD147D,
        0x3D584BF2B68D766F,
        0x3D40931A909FEA5E,
        0x3D4EC5197DDB55D3,
        0x3D5B7BF7861D37AC,
        0x3D5A21AC25DB1EF3,
        0xBD542A9E21373414,
        0xBD6DAFA08CECADB1,
        0x3D3E1F8DF68DBCF3,
        0x3D3BB2CD720EC44C,
        0xBD49C24CA098362B,
        0x3D60FEC69C695D7F,
        0x3D6F404E57963891,
        0xBD657D49676844CC,
        0x3D592DFBC7D93617,
        0x3D65E9A98F33A396,
        0x3D52DD98B97BAEF0,
        0x3D1A07BD8B34BE7C,
        0xBD17AFA4392F1BA7,
        0xBD5DCA290F818480,
        0x3D5D1772F5386374,
        0x3D60BE1FB590A1F5,
        0xBD6E2CE9146D271A,
        0xBD65E6563BBD9FC9,
        0x3D66FAA404263D0B,
        0xBD5AA33736867A17,
        0x3D6EC27D0B7B37B3,
        0xBD244FDD840B8591,
        0x3D6BB09CB0985646,
        0x3D46ABB9DF22BC57,
        0xBD58CD7DC73BD194,
        0x3D6F2CFB29AAA5F0,
        0x3D66757006095FD2,
    ]
#define NPY_TANG_LOG_A1 0x1.55555555554e6p-4
// 定义常量 NPY_TANG_LOG_A1，表示浮点数 0x1.55555555554e6 乘以 2 的 -4 次方

#define NPY_TANG_LOG_A2 0x1.9999999bac6d4p-7
// 定义常量 NPY_TANG_LOG_A2，表示浮点数 0x1.9999999bac6d4 乘以 2 的 -7 次方

#define NPY_TANG_LOG_A3 0x1.2492307f1519fp-9
// 定义常量 NPY_TANG_LOG_A3，表示浮点数 0x1.2492307f1519f 乘以 2 的 -9 次方

#define NPY_TANG_LOG_A4 0x1.c8034c85dfffp-12
// 定义常量 NPY_TANG_LOG_A4，表示浮点数 0x1.c8034c85dfff 乘以 2 的 -12 次方

#define NPY_TANG_LOG_LN2HI 0x1.62e42fefa4p-1
// 定义常量 NPY_TANG_LOG_LN2HI，表示浮点数 0x1.62e42fefa4 乘以 2 的 -1 次方

#define NPY_TANG_LOG_LN2LO -0x1.8432a1b0e2634p-43
// 定义常量 NPY_TANG_LOG_LN2LO，表示浮点数 -0x1.8432a1b0e2634 乘以 2 的 -43 次方
#endif
#endif

#endif
```