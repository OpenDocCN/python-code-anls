# Infer-Out Model 2 Evaluation (merged.O2.func_map.infer-host)

- Timestamp: 20251119-170633
- Source JSONL: merged.O2.func_map.infer.jsonl
- Target: host
- Total cases: 368
- Replacement success: 368 (100.00%)
- Compilable: 139 (37.77%)
- Executable: 126 (34.24%)

## Benchmark Breakdown
| Benchmark | Cases | Replacement% | Build% | Exec% |
| --- | --- | --- | --- | --- |
| ackermann | 2 | 100.00% | 50.00% | 50.00% |
| aes | 10 | 100.00% | 20.00% | 20.00% |
| anagram | 13 | 100.00% | 46.15% | 46.15% |
| audio-codec | 3 | 100.00% | 33.33% | 33.33% |
| avl-tree | 15 | 100.00% | 20.00% | 20.00% |
| banner | 1 | 100.00% | 0.00% | 0.00% |
| bit-kernels | 3 | 100.00% | 66.67% | 66.67% |
| blake2b | 4 | 100.00% | 0.00% | 0.00% |
| bloom-filter | 4 | 100.00% | 50.00% | 50.00% |
| boyer-moore-search | 3 | 100.00% | 0.00% | 0.00% |
| bubble-sort | 3 | 100.00% | 100.00% | 100.00% |
| c-interp | 10 | 100.00% | 50.00% | 50.00% |
| ccmac | 1 | 100.00% | 0.00% | 0.00% |
| checkers | 16 | 100.00% | 68.75% | 62.50% |
| cipher | 3 | 100.00% | 66.67% | 0.00% |
| congrad | 1 | 100.00% | 0.00% | 0.00% |
| connect4-minimax | 13 | 100.00% | 61.54% | 53.85% |
| convex-hull | 4 | 100.00% | 75.00% | 75.00% |
| dhrystone | 5 | 100.00% | 20.00% | 20.00% |
| distinctness | 2 | 100.00% | 0.00% | 0.00% |
| fft-int | 4 | 100.00% | 50.00% | 50.00% |
| flood-fill | 2 | 100.00% | 50.00% | 50.00% |
| frac-calc | 10 | 100.00% | 50.00% | 50.00% |
| fuzzy-match | 3 | 100.00% | 33.33% | 33.33% |
| fy-shuffle | 3 | 100.00% | 33.33% | 33.33% |
| gcd-list | 2 | 100.00% | 50.00% | 0.00% |
| grad-descent | 4 | 100.00% | 25.00% | 25.00% |
| graph-tests | 20 | 100.00% | 10.00% | 10.00% |
| hanoi | 1 | 100.00% | 0.00% | 0.00% |
| heapsort | 2 | 100.00% | 50.00% | 50.00% |
| heat-calc | 1 | 100.00% | 0.00% | 0.00% |
| huff-encode | 13 | 100.00% | 92.31% | 92.31% |
| idct-alg | 3 | 100.00% | 66.67% | 33.33% |
| indirect-test | 2 | 100.00% | 50.00% | 50.00% |
| k-means | 6 | 100.00% | 33.33% | 33.33% |
| kadane | 2 | 100.00% | 50.00% | 50.00% |
| kepler | 7 | 100.00% | 14.29% | 14.29% |
| knapsack | 3 | 100.00% | 33.33% | 33.33% |
| knights-tour | 3 | 100.00% | 33.33% | 33.33% |
| life | 14 | 100.00% | 21.43% | 14.29% |
| longdiv | 6 | 100.00% | 50.00% | 50.00% |
| lu-decomp | 3 | 100.00% | 33.33% | 33.33% |
| lz-compress | 2 | 100.00% | 100.00% | 100.00% |
| mandelbrot | 1 | 100.00% | 0.00% | 0.00% |
| matmult | 1 | 100.00% | 0.00% | 0.00% |
| max-subseq | 2 | 100.00% | 0.00% | 0.00% |
| mersenne | 3 | 100.00% | 0.00% | 0.00% |
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
| primal-test | 3 | 100.00% | 33.33% | 33.33% |
| priority-queue | 5 | 100.00% | 80.00% | 80.00% |
| qsort-demo | 7 | 100.00% | 28.57% | 28.57% |
| qsort-test | 5 | 100.00% | 80.00% | 80.00% |
| quaternions | 4 | 100.00% | 0.00% | 0.00% |
| rabinkarp-search | 2 | 100.00% | 0.00% | 0.00% |
| rand-test | 3 | 100.00% | 0.00% | 0.00% |
| ransac | 2 | 100.00% | 50.00% | 0.00% |
| regex-parser | 7 | 100.00% | 28.57% | 14.29% |
| rho-factor | 3 | 100.00% | 66.67% | 66.67% |
| rle-compress | 2 | 100.00% | 0.00% | 0.00% |
| rsa-cipher | 4 | 100.00% | 0.00% | 0.00% |
| sat-solver | 5 | 100.00% | 60.00% | 60.00% |
| shortest-path | 3 | 100.00% | 66.67% | 66.67% |
| sieve | 1 | 100.00% | 0.00% | 0.00% |
| simple-grep | 1 | 100.00% | 0.00% | 0.00% |
| spelt2num | 1 | 100.00% | 0.00% | 0.00% |
| spirograph | 2 | 100.00% | 50.00% | 0.00% |
| sudoku-solver | 4 | 100.00% | 50.00% | 50.00% |
| tetris-sim | 12 | 100.00% | 75.00% | 58.33% |
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
- aes/aes.c::aes_decrypt@0x18c0
- aes/aes.c::aes_encrypt@0x1780
- aes/aes.c::inv_mix_columns@0x1640
- aes/aes.c::inv_shift_rows@0x14f0
- aes/aes.c::key_expansion@0x16d0
- aes/aes.c::main@0x1100
- aes/aes.c::mix_columns@0x1580
- aes/aes.c::shift_rows@0x1480
- anagram/anagram.c::BuildMask@0x14c0
- anagram/anagram.c::BuildWord@0x17d0
- anagram/anagram.c::DumpCandidates@0x19a0
- anagram/anagram.c::DumpWords@0x1a30
- anagram/anagram.c::FindAnagram@0x1a90
- anagram/anagram.c::ReadDict@0x1360
- anagram/anagram.c::main@0x1120
- audio-codec/audio-codec.c::decode@0x1440
- audio-codec/audio-codec.c::main@0x1100
- avl-tree/avlcore.c::CheckTreeNodeRotation@0x1c30
- avl-tree/element.c::Compare@0x1ad0
- avl-tree/avlcore.c::DeleteByElement@0x2860
- avl-tree/avlcore.c::DeleteByElementRecursive@0x26d0
- avl-tree/avlcore.c::DeleteLeftMost@0x2610
- avl-tree/avlcore.c::DoubleLeftRotation@0x1c00
- avl-tree/avlcore.c::DoubleRightRotation@0x1bd0
- avl-tree/avlcore.c::FindByElement@0x1b00
- avl-tree/avlcore.c::Insert@0x1f30
- avl-tree/avlcore.c::MakeEmpty@0x1f80
- avl-tree/avl-tree.c::breadth@0x1760
- avl-tree/avl-tree.c::main@0x1120
- banner/banner.c::main@0x1120
- bit-kernels/bit-kernels.c::main@0x1120
- blake2b/blake2b.c::F@0x12a0
- blake2b/blake2b.c::G@0x1230
- blake2b/blake2b.c::blake2b@0x1620
- blake2b/blake2b.c::test@0x19d0
- bloom-filter/bloom-filter.c::bad_search@0x1430
- bloom-filter/bloom-filter.c::main@0x1120
- boyer-moore-search/boyer-moore-search.c::badCharHeuristic@0x15d0
- boyer-moore-search/boyer-moore-search.c::main@0x1140
- boyer-moore-search/boyer-moore-search.c::search@0x1630
- c-interp/c-interp.c::eval@0x3e90
- c-interp/c-interp.c::function_body@0x37f0
- c-interp/c-interp.c::function_declaration@0x3a10
- c-interp/c-interp.c::main@0x1120
- c-interp/c-interp.c::next@0x1580
- ccmac/ccmac.c::main@0x1120
- checkers/functions.c::fill_print_initial@0x1630
- checkers/functions.c::free_tree@0x2460
- checkers/functions.c::generate_node_children@0x21c0
- checkers/functions.c::link_new_node@0x20e0
- checkers/checkers.c::main@0x1150
- cipher/cipher.c::main@0x1100
- congrad/congrad.c::main@0x1100
- connect4-minimax/connect4-minimax.c::init_board@0x1230
- connect4-minimax/connect4-minimax.c::main@0x1100
- connect4-minimax/connect4-minimax.c::minimax@0x1840
- connect4-minimax/connect4-minimax.c::play_game@0x1c90
- connect4-minimax/connect4-minimax.c::score_position@0x1620
- convex-hull/convex-hull.c::main@0x1100
- dhrystone/dhrystone.c::PFunc_1@0x1970
- dhrystone/dhrystone.c::PFunc_2@0x1990
- dhrystone/dhrystone.c::PProc_8@0x1900
- dhrystone/dhrystone.c::main@0x1100
- distinctness/distinctness.c::isDistinct@0x12a0
- distinctness/distinctness.c::main@0x1100
- fft-int/fft-int.c::db_from_ampl@0x1670
- fft-int/fft-int.c::fix_fft@0x1320
- flood-fill/flood-fill.c::main@0x1100
- frac-calc/frac-calc.c::avaliatokens@0x15f0
- frac-calc/frac-calc.c::copyr@0x1460
- frac-calc/frac-calc.c::divtokens@0x1840
- frac-calc/frac-calc.c::help@0x13b0
- frac-calc/frac-calc.c::main@0x1120
- fuzzy-match/fuzzy-match.c::fuzzy_match_recurse@0x2360
- fuzzy-match/fuzzy-match.c::main@0x2100
- fy-shuffle/fy-shuffle.c::fy_shuffle@0x1440
- fy-shuffle/fy-shuffle.c::main@0x1100
- gcd-list/gcd-list.c::main@0x1120
- grad-descent/grad-descent.c::derivateWRTBias@0x12d0
- grad-descent/grad-descent.c::derivateWRTWeight@0x1270
- grad-descent/grad-descent.c::main@0x1100
- graph-tests/graph-tests.c::DFS_test@0x1c20
- graph-tests/graph-tests.c::addEdge@0x1320
- graph-tests/graph-tests.c::addVertex@0x1a50
- graph-tests/graph-tests.c::bfs@0x1540
- graph-tests/graph-tests.c::bfs_test@0x1720
- graph-tests/graph-tests.c::bubbleSort@0x1880
- graph-tests/graph-tests.c::createGraph@0x1260
- graph-tests/graph-tests.c::createNode@0x1240
- graph-tests/graph-tests.c::createQueue@0x1390
- graph-tests/graph-tests.c::depthFirstSearch@0x1b20
- graph-tests/graph-tests.c::dequeue@0x1430
- graph-tests/graph-tests.c::enqueue@0x13e0
- graph-tests/graph-tests.c::getAdjUnvisitedVertex@0x1ac0
- graph-tests/graph-tests.c::insertAtTheBegin@0x1840
- graph-tests/graph-tests.c::link_list@0x18e0
- graph-tests/graph-tests.c::main@0x1120
- graph-tests/graph-tests.c::printQueue@0x14c0
- graph-tests/graph-tests.c::swap@0x1870
- hanoi/hanoi.c::main@0x1100
- heapsort/heapsort.c::main@0x1100
- heat-calc/heat-calc.c::main@0x1100
- huff-encode/huff-encode.c::main@0x1120
- idct-alg/idct-alg.c::main@0x1100
- indirect-test/indirect-test.c::main@0x1100
- k-means/k-means.c::calculateNearst@0x1310
- k-means/k-means.c::kMeans@0x1420
- k-means/k-means.c::main@0x1120
- k-means/k-means.c::printEPS@0x16b0
- kadane/kadane.c::main@0x1100
- kepler/kepler.c::J@0x1920
- kepler/kepler.c::bin_fact@0x1740
- kepler/kepler.c::binary@0x16a0
- kepler/kepler.c::e_series@0x17e0
- kepler/kepler.c::j_series@0x1a20
- kepler/kepler.c::main@0x1100
- knapsack/knapsack.c::main@0x1100
- knapsack/knapsack.c::max@0x1310
- knights-tour/knights-tour.c::solveKT@0x1390
- knights-tour/knights-tour.c::solveKTUtil@0x14f0
- life/life.c::getDown@0x16e0
- life/life.c::getDownLeft@0x1770
- life/life.c::getDownRight@0x17a0
- life/life.c::getLeft@0x1650
- life/life.c::getNumNeigbors@0x1390
- life/life.c::getRight@0x1680
- life/life.c::getUp@0x16b0
- life/life.c::getUpLeft@0x1710
- life/life.c::getUpRight@0x1740
- life/life.c::main@0x1100
- life/life.c::process@0x1550
- longdiv/longdiv.c::main@0x1120
- longdiv/longdiv.c::sbc@0x1a20
- longdiv/longdiv.c::sub@0x19c0
- lu-decomp/lu-decomp.c::main@0x1100
- lu-decomp/lu-decomp.c::print_matrix@0x13a0
- mandelbrot/mandelbrot.c::main@0x1100
- matmult/matmult.c::main@0x1100
- max-subseq/max-subseq.c::lcsAlgo@0x1290
- max-subseq/max-subseq.c::main@0x1120
- mersenne/mersenne.c::genrand@0x1310
- mersenne/mersenne.c::main@0x1100
- mersenne/mersenne.c::sgenrand@0x1290
- minspan/minspan.c::displayGraph@0x14f0
- minspan/minspan.c::displayGraph1@0x15f0
- minspan/minspan.c::displayPath@0x1700
- minspan/minspan.c::displayTree@0x17a0
- minspan/minspan.c::main@0x1100
- minspan/minspan.c::minSpanTree@0x12f0
- monte-carlo/monte-carlo.c::main@0x1100
- murmur-hash/murmur-hash.c::main@0x1100
- murmur-hash/murmur-hash.c::murmurhash@0x1290
- n-queens/n-queens.c::main@0x1120
- natlog/natlog.c::main@0x1100
- nbody-sim/nbody-sim.c::main@0x1100
- packet-filter/packet-filter.c::check_packet_filter@0x1430
- packet-filter/packet-filter.c::generate_packet@0x12d0
- packet-filter/packet-filter.c::main@0x1100
- packet-filter/packet-filter.c::print_packet@0x1490
- parrondo/parrondo.c::main@0x1100
- pascal/pascal.c::main@0x1100
- pi-calc/pi-calc.c::main@0x1100
- primal-test/primal-test.c::main@0x1100
- primal-test/primal-test.c::miller_rabin_int@0x1510
- priority-queue/priority-queue.c::main@0x1120
- qsort-demo/qsort-demo.c::main@0x1120
- qsort-demo/qsort-demo.c::print_struct_array@0x15c0
- qsort-demo/qsort-demo.c::sort_cstrings_example@0x14a0
- qsort-demo/qsort-demo.c::sort_integers_example@0x1310
- qsort-demo/qsort-demo.c::sort_structs_example@0x1640
- qsort-test/qsort-test.c::main@0x1120
- quaternions/quaternions.c::euler_from_quat@0x1580
- quaternions/quaternions.c::main@0x1100
- quaternions/quaternions.c::quat_from_euler@0x13f0
- quaternions/quaternions.c::quaternion_multiply@0x16b0
- rabinkarp-search/rabinkarp-search.c::main@0x1120
- rabinkarp-search/rabinkarp-search.c::search@0x13a0
- rand-test/rand-test.c::bad_rand@0x1240
- rand-test/rand-test.c::main@0x1100
- rand-test/rand-test.c::run_tests@0x1280
- ransac/ransac.c::main@0x1100
- regex-parser/regex-parser.c::main@0x2100
- regex-parser/regex-parser.c::matchcharclass@0x23b0
- regex-parser/regex-parser.c::matchone@0x2560
- regex-parser/regex-parser.c::re_compile@0x2930
- regex-parser/regex-parser.c::re_print@0x2bf0
- rho-factor/rho-factor.c::main@0x1120
- rle-compress/rle-compress.c::main@0x1120
- rle-compress/rle-compress.c::run_length_encode@0x1330
- rsa-cipher/rsa-cipher.c::main@0x1100
- rsa-cipher/rsa-cipher.c::mod_inverse@0x1670
- rsa-cipher/rsa-cipher.c::mod_pow@0x1580
- rsa-cipher/rsa-cipher.c::print_hex_int128@0x1790
- sat-solver/sat-solver.c::main@0x1100
- sat-solver/sat-solver.c::printFormula@0x1390
- shortest-path/shortest-path.c::main@0x1100
- sieve/sieve.c::main@0x1100
- simple-grep/simple-grep.c::main@0x1120
- spelt2num/spelt2num.c::main@0x1100
- spirograph/spirograph.c::spirograph@0x1230
- sudoku-solver/sudoku-solver.c::isSafe@0x1250
- sudoku-solver/sudoku-solver.c::main@0x1100
- tetris-sim/tetris-sim.c::best_move@0x1860
- tetris-sim/tetris-sim.c::evaluate_board@0x1640
- tetris-sim/tetris-sim.c::main@0x1120
- tiny-NN/tiny-NN.c::main@0x1120
- tiny-NN/tiny-NN.c::sampleSine@0x12d0
- tiny-NN/tiny-NN.c::train@0x13e0
- topo-sort/topo-sort.c::addEdge@0x1370
- topo-sort/topo-sort.c::createGraph@0x1300
- topo-sort/topo-sort.c::createListNode@0x12e0
- topo-sort/topo-sort.c::createStackNode@0x12c0
- topo-sort/topo-sort.c::main@0x1120
- topo-sort/topo-sort.c::topologicalSort@0x1450
- topo-sort/topo-sort.c::topologicalSortUtil@0x13c0
- totient/totient.c::main@0x1100
- transcend/transcend.c::main@0x1120
- uniquify/uniquify.c::main@0x1120
- vectors-3d/vectors-3d.c::get_cross_matrix@0x1760
- vectors-3d/vectors-3d.c::main@0x1100
- vectors-3d/vectors-3d.c::print_vector@0x1620
- vectors-3d/vectors-3d.c::unit_vec@0x1690
- vectors-3d/vectors-3d.c::vector_add@0x1550
- vectors-3d/vectors-3d.c::vector_prod@0x15c0
- vectors-3d/vectors-3d.c::vector_sub@0x1510
- verlet/verlet.c::main@0x1100
- weekday/weekday.c::dayOfWeek@0x1350
- weekday/weekday.c::main@0x1100

## Execution Failures
- checkers/functions.c::all_possible_moves@0x1a60
- cipher/cipher.c::decipher@0x1360
- cipher/cipher.c::encipher@0x12f0
- connect4-minimax/connect4-minimax.c::terminal_score@0x1800
- gcd-list/gcd-list.c::gcd@0x1310
- idct-alg/idct-alg.c::idct_2d@0x12f0
- life/life.c::init@0x1220
- ransac/ransac.c::ransac_line_fitting@0x1410
- regex-parser/regex-parser.c::matchpattern@0x2670
- spirograph/spirograph.c::test@0x1390
- tetris-sim/tetris-sim.c::clear_lines@0x1480
- tetris-sim/tetris-sim.c::simulate_board@0x17c0
- vectors-3d/vectors-3d.c::get_angle@0x17d0