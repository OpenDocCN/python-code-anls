# Infer-Out Model 2 Evaluation (merged.O3.func_map.infer-host)

- Timestamp: 20251119-171533
- Source JSONL: merged.O3.func_map.infer.jsonl
- Target: host
- Total cases: 359
- Replacement success: 359 (100.00%)
- Compilable: 114 (31.75%)
- Executable: 106 (29.53%)

## Benchmark Breakdown
| Benchmark | Cases | Replacement% | Build% | Exec% |
| --- | --- | --- | --- | --- |
| ackermann | 2 | 100.00% | 50.00% | 50.00% |
| aes | 11 | 100.00% | 27.27% | 27.27% |
| anagram | 13 | 100.00% | 38.46% | 38.46% |
| audio-codec | 3 | 100.00% | 33.33% | 33.33% |
| avl-tree | 15 | 100.00% | 13.33% | 13.33% |
| banner | 1 | 100.00% | 0.00% | 0.00% |
| bit-kernels | 3 | 100.00% | 66.67% | 66.67% |
| blake2b | 3 | 100.00% | 0.00% | 0.00% |
| bloom-filter | 4 | 100.00% | 25.00% | 25.00% |
| boyer-moore-search | 3 | 100.00% | 0.00% | 0.00% |
| bubble-sort | 3 | 100.00% | 100.00% | 100.00% |
| c-interp | 10 | 100.00% | 40.00% | 40.00% |
| ccmac | 1 | 100.00% | 0.00% | 0.00% |
| checkers | 13 | 100.00% | 61.54% | 61.54% |
| cipher | 3 | 100.00% | 33.33% | 0.00% |
| congrad | 1 | 100.00% | 0.00% | 0.00% |
| connect4-minimax | 11 | 100.00% | 45.45% | 45.45% |
| convex-hull | 4 | 100.00% | 50.00% | 50.00% |
| dhrystone | 5 | 100.00% | 40.00% | 40.00% |
| distinctness | 2 | 100.00% | 0.00% | 0.00% |
| fft-int | 4 | 100.00% | 0.00% | 0.00% |
| flood-fill | 2 | 100.00% | 50.00% | 50.00% |
| frac-calc | 9 | 100.00% | 22.22% | 22.22% |
| fuzzy-match | 3 | 100.00% | 33.33% | 33.33% |
| fy-shuffle | 3 | 100.00% | 33.33% | 33.33% |
| gcd-list | 2 | 100.00% | 0.00% | 0.00% |
| grad-descent | 4 | 100.00% | 0.00% | 0.00% |
| graph-tests | 19 | 100.00% | 5.26% | 5.26% |
| hanoi | 1 | 100.00% | 0.00% | 0.00% |
| heapsort | 2 | 100.00% | 0.00% | 0.00% |
| heat-calc | 1 | 100.00% | 0.00% | 0.00% |
| huff-encode | 12 | 100.00% | 83.33% | 83.33% |
| idct-alg | 3 | 100.00% | 66.67% | 33.33% |
| indirect-test | 2 | 100.00% | 50.00% | 50.00% |
| k-means | 5 | 100.00% | 0.00% | 0.00% |
| kadane | 2 | 100.00% | 50.00% | 50.00% |
| kepler | 7 | 100.00% | 14.29% | 14.29% |
| knapsack | 3 | 100.00% | 33.33% | 33.33% |
| knights-tour | 3 | 100.00% | 33.33% | 33.33% |
| life | 14 | 100.00% | 21.43% | 14.29% |
| longdiv | 7 | 100.00% | 71.43% | 71.43% |
| lu-decomp | 3 | 100.00% | 33.33% | 33.33% |
| lz-compress | 2 | 100.00% | 100.00% | 100.00% |
| mandelbrot | 1 | 100.00% | 0.00% | 0.00% |
| max-subseq | 2 | 100.00% | 0.00% | 0.00% |
| mersenne | 4 | 100.00% | 0.00% | 0.00% |
| minspan | 8 | 100.00% | 25.00% | 25.00% |
| monte-carlo | 1 | 100.00% | 0.00% | 0.00% |
| murmur-hash | 2 | 100.00% | 0.00% | 0.00% |
| n-queens | 3 | 100.00% | 66.67% | 66.67% |
| natlog | 1 | 100.00% | 0.00% | 0.00% |
| nbody-sim | 1 | 100.00% | 0.00% | 0.00% |
| nr-solver | 1 | 100.00% | 100.00% | 100.00% |
| packet-filter | 4 | 100.00% | 0.00% | 0.00% |
| parrondo | 2 | 100.00% | 50.00% | 50.00% |
| pascal | 3 | 100.00% | 66.67% | 66.67% |
| pi-calc | 1 | 100.00% | 0.00% | 0.00% |
| primal-test | 3 | 100.00% | 66.67% | 66.67% |
| priority-queue | 5 | 100.00% | 40.00% | 40.00% |
| qsort-demo | 7 | 100.00% | 28.57% | 28.57% |
| qsort-test | 5 | 100.00% | 80.00% | 80.00% |
| quaternions | 4 | 100.00% | 0.00% | 0.00% |
| rabinkarp-search | 2 | 100.00% | 0.00% | 0.00% |
| rand-test | 3 | 100.00% | 0.00% | 0.00% |
| ransac | 2 | 100.00% | 50.00% | 0.00% |
| regex-parser | 8 | 100.00% | 25.00% | 25.00% |
| rho-factor | 1 | 100.00% | 100.00% | 100.00% |
| rle-compress | 2 | 100.00% | 0.00% | 0.00% |
| rsa-cipher | 4 | 100.00% | 0.00% | 0.00% |
| sat-solver | 5 | 100.00% | 60.00% | 40.00% |
| shortest-path | 3 | 100.00% | 33.33% | 33.33% |
| sieve | 1 | 100.00% | 0.00% | 0.00% |
| simple-grep | 1 | 100.00% | 0.00% | 0.00% |
| spelt2num | 1 | 100.00% | 0.00% | 0.00% |
| spirograph | 2 | 100.00% | 50.00% | 0.00% |
| sudoku-solver | 4 | 100.00% | 75.00% | 75.00% |
| tetris-sim | 12 | 100.00% | 58.33% | 50.00% |
| tiny-NN | 4 | 100.00% | 25.00% | 25.00% |
| topo-sort | 7 | 100.00% | 0.00% | 0.00% |
| totient | 2 | 100.00% | 50.00% | 50.00% |
| transcend | 1 | 100.00% | 0.00% | 0.00% |
| uniquify | 1 | 100.00% | 0.00% | 0.00% |
| vectors-3d | 8 | 100.00% | 12.50% | 0.00% |
| verlet | 1 | 100.00% | 0.00% | 0.00% |
| weekday | 2 | 100.00% | 0.00% | 0.00% |

## Compilation Failures
- ackermann/ackermann.c::main@0x1100
- aes/aes.c::add_round_key@0x1810
- aes/aes.c::aes_decrypt@0x2760
- aes/aes.c::aes_encrypt@0x2200
- aes/aes.c::inv_shift_rows@0x1af0
- aes/aes.c::key_expansion@0x1ff0
- aes/aes.c::main@0x1100
- aes/aes.c::mix_columns@0x1bd0
- aes/aes.c::shift_rows@0x1a30
- anagram/anagram.c::BuildMask@0x1620
- anagram/anagram.c::BuildWord@0x1940
- anagram/anagram.c::DumpCandidates@0x1c10
- anagram/anagram.c::DumpWords@0x1ca0
- anagram/anagram.c::FindAnagram@0x1d00
- anagram/anagram.c::ReadDict@0x14c0
- anagram/anagram.c::SortCandidates@0x1f10
- anagram/anagram.c::main@0x1120
- audio-codec/audio-codec.c::decode@0x1590
- audio-codec/audio-codec.c::main@0x1100
- avl-tree/avlcore.c::CheckTreeNodeRotation@0x1c50
- avl-tree/element.c::Compare@0x1af0
- avl-tree/avlcore.c::DeleteByElement@0x2e50
- avl-tree/avlcore.c::DeleteByElementRecursive@0x2bf0
- avl-tree/avlcore.c::DeleteLeftMost@0x2720
- avl-tree/avlcore.c::DoubleLeftRotation@0x1c20
- avl-tree/avlcore.c::DoubleRightRotation@0x1bf0
- avl-tree/avlcore.c::FindByElement@0x1b20
- avl-tree/avlcore.c::Insert@0x1f40
- avl-tree/avlcore.c::InsertNode@0x1e10
- avl-tree/avlcore.c::MakeEmpty@0x2090
- avl-tree/avl-tree.c::breadth@0x1780
- avl-tree/avl-tree.c::main@0x1120
- banner/banner.c::main@0x1120
- bit-kernels/bit-kernels.c::main@0x1120
- blake2b/blake2b.c::F@0x12e0
- blake2b/blake2b.c::blake2b@0x17b0
- blake2b/blake2b.c::test@0x1b50
- bloom-filter/bloom-filter.c::bad_search@0x1450
- bloom-filter/tinybloom.c::bfilter_intersect@0x1570
- bloom-filter/bloom-filter.c::main@0x1120
- boyer-moore-search/boyer-moore-search.c::badCharHeuristic@0x15d0
- boyer-moore-search/boyer-moore-search.c::main@0x1140
- boyer-moore-search/boyer-moore-search.c::search@0x1630
- c-interp/c-interp.c::enum_declaration@0x34f0
- c-interp/c-interp.c::eval@0x3ea0
- c-interp/c-interp.c::function_body@0x37f0
- c-interp/c-interp.c::function_declaration@0x3a10
- c-interp/c-interp.c::main@0x1120
- c-interp/c-interp.c::next@0x15a0
- ccmac/ccmac.c::main@0x1120
- checkers/functions.c::fill_print_initial@0x18e0
- checkers/functions.c::free_tree@0x6210
- checkers/functions.c::generate_node_children@0x35d0
- checkers/functions.c::link_new_node@0x34c0
- checkers/checkers.c::main@0x1130
- cipher/cipher.c::encipher@0x12f0
- cipher/cipher.c::main@0x1100
- congrad/congrad.c::main@0x1100
- connect4-minimax/connect4-minimax.c::board_full@0x1500
- connect4-minimax/connect4-minimax.c::evaluate_window@0x2380
- connect4-minimax/connect4-minimax.c::init_board@0x1230
- connect4-minimax/connect4-minimax.c::main@0x1100
- connect4-minimax/connect4-minimax.c::minimax@0x3c30
- connect4-minimax/connect4-minimax.c::play_game@0x4260
- convex-hull/convex-hull.c::main@0x1100
- convex-hull/convex-hull.c::sortPoints@0x1740
- dhrystone/dhrystone.c::PFunc_1@0x1980
- dhrystone/dhrystone.c::PProc_8@0x1910
- dhrystone/dhrystone.c::main@0x1100
- distinctness/distinctness.c::isDistinct@0x12a0
- distinctness/distinctness.c::main@0x1100
- fft-int/fft-int.c::db_from_ampl@0x1c50
- fft-int/fft-int.c::fix_fft@0x1370
- fft-int/fft-int.c::fix_loud@0x1a90
- fft-int/fft-int.c::window@0x1650
- flood-fill/flood-fill.c::main@0x1100
- frac-calc/frac-calc.c::avaliatokens@0x1730
- frac-calc/frac-calc.c::copyr@0x1550
- frac-calc/frac-calc.c::divtokens@0x1980
- frac-calc/frac-calc.c::help@0x14a0
- frac-calc/frac-calc.c::main@0x1120
- frac-calc/frac-calc.c::misto@0x1610
- frac-calc/frac-calc.c::simplifica@0x28f0
- fuzzy-match/fuzzy-match.c::fuzzy_match_recurse@0x23e0
- fuzzy-match/fuzzy-match.c::main@0x2100
- fy-shuffle/fy-shuffle.c::fy_shuffle@0x1440
- fy-shuffle/fy-shuffle.c::main@0x1100
- gcd-list/gcd-list.c::gcd@0x1310
- gcd-list/gcd-list.c::main@0x1120
- grad-descent/grad-descent.c::derivateWRTBias@0x12e0
- grad-descent/grad-descent.c::derivateWRTWeight@0x1270
- grad-descent/grad-descent.c::gradientDescent@0x1350
- grad-descent/grad-descent.c::main@0x1100
- graph-tests/graph-tests.c::DFS_test@0x2340
- graph-tests/graph-tests.c::addEdge@0x1610
- graph-tests/graph-tests.c::addVertex@0x1f80
- graph-tests/graph-tests.c::bfs@0x1830
- graph-tests/graph-tests.c::bfs_test@0x1a70
- graph-tests/graph-tests.c::bubbleSort@0x1db0
- graph-tests/graph-tests.c::createGraph@0x1550
- graph-tests/graph-tests.c::createNode@0x1530
- graph-tests/graph-tests.c::createQueue@0x1680
- graph-tests/graph-tests.c::depthFirstSearch@0x2110
- graph-tests/graph-tests.c::dequeue@0x1720
- graph-tests/graph-tests.c::enqueue@0x16d0
- graph-tests/graph-tests.c::insertAtTheBegin@0x1d70
- graph-tests/graph-tests.c::link_list@0x1e20
- graph-tests/graph-tests.c::main@0x1180
- graph-tests/graph-tests.c::printQueue@0x17b0
- graph-tests/graph-tests.c::swap@0x1da0
- graph-tests/graph-tests.c::towers@0x2490
- hanoi/hanoi.c::main@0x1100
- heapsort/heapsort.c::HSORT@0x12f0
- heapsort/heapsort.c::main@0x11a0
- heat-calc/heat-calc.c::main@0x1100
- huff-encode/huff-encode.c::buildHuffmanTree@0x18b0
- huff-encode/huff-encode.c::main@0x1120
- idct-alg/idct-alg.c::main@0x1100
- indirect-test/indirect-test.c::main@0x1100
- k-means/k-means.c::calculateCentroid@0x1390
- k-means/k-means.c::calculateNearst@0x1310
- k-means/k-means.c::kMeans@0x1400
- k-means/k-means.c::main@0x1120
- k-means/k-means.c::printEPS@0x16c0
- kadane/kadane.c::main@0x1100
- kepler/kepler.c::J@0x1b80
- kepler/kepler.c::bin_fact@0x1ad0
- kepler/kepler.c::binary@0x16a0
- kepler/kepler.c::e_series@0x1740
- kepler/kepler.c::j_series@0x1920
- kepler/kepler.c::main@0x1100
- knapsack/knapsack.c::main@0x1100
- knapsack/knapsack.c::max@0x1310
- knights-tour/knights-tour.c::solveKT@0x1830
- knights-tour/knights-tour.c::solveKTUtil@0x1980
- life/life.c::getDown@0x1960
- life/life.c::getDownLeft@0x19f0
- life/life.c::getDownRight@0x1a20
- life/life.c::getLeft@0x18d0
- life/life.c::getNumNeigbors@0x16d0
- life/life.c::getRight@0x1900
- life/life.c::getUp@0x1930
- life/life.c::getUpLeft@0x1990
- life/life.c::getUpRight@0x19c0
- life/life.c::main@0x1100
- life/life.c::process@0x1430
- longdiv/longdiv.c::main@0x1120
- longdiv/longdiv.c::sub@0x1a80
- lu-decomp/lu-decomp.c::main@0x1100
- lu-decomp/lu-decomp.c::print_matrix@0x1320
- mandelbrot/mandelbrot.c::main@0x1100
- max-subseq/max-subseq.c::lcsAlgo@0x1290
- max-subseq/max-subseq.c::main@0x1120
- mersenne/mersenne.c::genrand@0x1380
- mersenne/mersenne.c::lsgenrand@0x1320
- mersenne/mersenne.c::main@0x1100
- mersenne/mersenne.c::sgenrand@0x12d0
- minspan/minspan.c::displayGraph@0x1db0
- minspan/minspan.c::displayGraph1@0x1ee0
- minspan/minspan.c::displayPath@0x2020
- minspan/minspan.c::displayTree@0x20c0
- minspan/minspan.c::main@0x1100
- minspan/minspan.c::minSpanTree@0x1400
- monte-carlo/monte-carlo.c::main@0x1100
- murmur-hash/murmur-hash.c::main@0x1100
- murmur-hash/murmur-hash.c::murmurhash@0x1290
- n-queens/n-queens.c::main@0x1120
- natlog/natlog.c::main@0x1100
- nbody-sim/nbody-sim.c::main@0x1100
- packet-filter/packet-filter.c::check_packet_filter@0x1520
- packet-filter/packet-filter.c::generate_packet@0x13d0
- packet-filter/packet-filter.c::main@0x1100
- packet-filter/packet-filter.c::print_packet@0x1580
- parrondo/parrondo.c::main@0x1100
- pascal/pascal.c::main@0x1100
- pi-calc/pi-calc.c::main@0x1100
- primal-test/primal-test.c::main@0x1100
- priority-queue/priority-queue.c::main@0x1120
- priority-queue/priority-queue.c::newNode@0x13a0
- priority-queue/priority-queue.c::push@0x1420
- qsort-demo/qsort-demo.c::main@0x1120
- qsort-demo/qsort-demo.c::print_struct_array@0x15b0
- qsort-demo/qsort-demo.c::sort_cstrings_example@0x1480
- qsort-demo/qsort-demo.c::sort_integers_example@0x1310
- qsort-demo/qsort-demo.c::sort_structs_example@0x1630
- qsort-test/qsort-test.c::main@0x1120
- quaternions/quaternions.c::euler_from_quat@0x1550
- quaternions/quaternions.c::main@0x1100
- quaternions/quaternions.c::quat_from_euler@0x13e0
- quaternions/quaternions.c::quaternion_multiply@0x1670
- rabinkarp-search/rabinkarp-search.c::main@0x1120
- rabinkarp-search/rabinkarp-search.c::search@0x15a0
- rand-test/rand-test.c::bad_rand@0x1240
- rand-test/rand-test.c::main@0x1100
- rand-test/rand-test.c::run_tests@0x1280
- ransac/ransac.c::main@0x1100
- regex-parser/regex-parser.c::main@0x2100
- regex-parser/regex-parser.c::matchcharclass@0x2420
- regex-parser/regex-parser.c::matchone@0x25c0
- regex-parser/regex-parser.c::matchpattern@0x26d0
- regex-parser/regex-parser.c::re_compile@0x2ac0
- regex-parser/regex-parser.c::re_print@0x2e30
- rle-compress/rle-compress.c::main@0x1120
- rle-compress/rle-compress.c::run_length_encode@0x1330
- rsa-cipher/rsa-cipher.c::main@0x1100
- rsa-cipher/rsa-cipher.c::mod_inverse@0x15a0
- rsa-cipher/rsa-cipher.c::mod_pow@0x14b0
- rsa-cipher/rsa-cipher.c::print_hex_int128@0x16c0
- sat-solver/sat-solver.c::main@0x1100
- sat-solver/sat-solver.c::printFormula@0x1680
- shortest-path/shortest-path.c::floydWarshall@0x1330
- shortest-path/shortest-path.c::main@0x1100
- sieve/sieve.c::main@0x1100
- simple-grep/simple-grep.c::main@0x1120
- spelt2num/spelt2num.c::main@0x1100
- spirograph/spirograph.c::spirograph@0x1230
- sudoku-solver/sudoku-solver.c::main@0x1100
- tetris-sim/tetris-sim.c::aggregate_height@0x1b20
- tetris-sim/tetris-sim.c::best_move@0x21d0
- tetris-sim/tetris-sim.c::count_holes@0x1b70
- tetris-sim/tetris-sim.c::evaluate_board@0x1ca0
- tetris-sim/tetris-sim.c::main@0x1100
- tiny-NN/tiny-NN.c::main@0x1120
- tiny-NN/tiny-NN.c::sampleSine@0x12d0
- tiny-NN/tiny-NN.c::train@0x13e0
- topo-sort/topo-sort.c::addEdge@0x13f0
- topo-sort/topo-sort.c::createGraph@0x1380
- topo-sort/topo-sort.c::createListNode@0x1360
- topo-sort/topo-sort.c::createStackNode@0x1340
- topo-sort/topo-sort.c::main@0x1120
- topo-sort/topo-sort.c::topologicalSort@0x18b0
- topo-sort/topo-sort.c::topologicalSortUtil@0x1440
- totient/totient.c::main@0x1100
- transcend/transcend.c::main@0x1120
- uniquify/uniquify.c::main@0x1120
- vectors-3d/vectors-3d.c::get_cross_matrix@0x1850
- vectors-3d/vectors-3d.c::main@0x1100
- vectors-3d/vectors-3d.c::print_vector@0x1730
- vectors-3d/vectors-3d.c::unit_vec@0x17a0
- vectors-3d/vectors-3d.c::vector_add@0x1650
- vectors-3d/vectors-3d.c::vector_prod@0x16b0
- vectors-3d/vectors-3d.c::vector_sub@0x1620
- verlet/verlet.c::main@0x1100
- weekday/weekday.c::dayOfWeek@0x1290
- weekday/weekday.c::main@0x1100

## Execution Failures
- cipher/cipher.c::decipher@0x1360
- idct-alg/idct-alg.c::idct_2d@0x12f0
- life/life.c::init@0x12c0
- ransac/ransac.c::ransac_line_fitting@0x1410
- sat-solver/sat-solver.c::solveSAT@0x13a0
- spirograph/spirograph.c::test@0x1390
- tetris-sim/tetris-sim.c::clear_lines@0x19a0
- vectors-3d/vectors-3d.c::get_angle@0x18c0