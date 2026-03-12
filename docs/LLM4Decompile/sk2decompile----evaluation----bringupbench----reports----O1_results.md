# Infer-Out Model 2 Evaluation (merged.O1.func_map.infer-host)

- Timestamp: 20251119-171212
- Source JSONL: merged.O1.func_map.infer.jsonl
- Target: host
- Total cases: 379
- Replacement success: 379 (100.00%)
- Compilable: 155 (40.90%)
- Executable: 148 (39.05%)

## Benchmark Breakdown
| Benchmark | Cases | Replacement% | Build% | Exec% |
| --- | --- | --- | --- | --- |
| ackermann | 2 | 100.00% | 50.00% | 50.00% |
| aes | 9 | 100.00% | 33.33% | 33.33% |
| anagram | 13 | 100.00% | 53.85% | 53.85% |
| audio-codec | 3 | 100.00% | 0.00% | 0.00% |
| avl-tree | 17 | 100.00% | 29.41% | 29.41% |
| banner | 1 | 100.00% | 0.00% | 0.00% |
| bit-kernels | 5 | 100.00% | 80.00% | 80.00% |
| blake2b | 5 | 100.00% | 20.00% | 20.00% |
| bloom-filter | 4 | 100.00% | 50.00% | 50.00% |
| boyer-moore-search | 3 | 100.00% | 0.00% | 0.00% |
| bubble-sort | 3 | 100.00% | 100.00% | 100.00% |
| c-interp | 10 | 100.00% | 60.00% | 60.00% |
| ccmac | 1 | 100.00% | 0.00% | 0.00% |
| checkers | 16 | 100.00% | 81.25% | 81.25% |
| cipher | 3 | 100.00% | 33.33% | 0.00% |
| congrad | 2 | 100.00% | 0.00% | 0.00% |
| connect4-minimax | 13 | 100.00% | 61.54% | 61.54% |
| convex-hull | 4 | 100.00% | 75.00% | 75.00% |
| dhrystone | 5 | 100.00% | 40.00% | 40.00% |
| distinctness | 2 | 100.00% | 0.00% | 0.00% |
| fft-int | 4 | 100.00% | 75.00% | 75.00% |
| flood-fill | 2 | 100.00% | 50.00% | 50.00% |
| frac-calc | 10 | 100.00% | 40.00% | 40.00% |
| fuzzy-match | 3 | 100.00% | 33.33% | 33.33% |
| fy-shuffle | 3 | 100.00% | 33.33% | 33.33% |
| gcd-list | 2 | 100.00% | 0.00% | 0.00% |
| grad-descent | 4 | 100.00% | 0.00% | 0.00% |
| graph-tests | 19 | 100.00% | 21.05% | 21.05% |
| hanoi | 2 | 100.00% | 50.00% | 50.00% |
| heapsort | 2 | 100.00% | 50.00% | 50.00% |
| heat-calc | 1 | 100.00% | 0.00% | 0.00% |
| huff-encode | 13 | 100.00% | 92.31% | 92.31% |
| idct-alg | 3 | 100.00% | 66.67% | 33.33% |
| indirect-test | 2 | 100.00% | 50.00% | 50.00% |
| k-means | 6 | 100.00% | 50.00% | 50.00% |
| kadane | 2 | 100.00% | 50.00% | 50.00% |
| kepler | 7 | 100.00% | 14.29% | 14.29% |
| knapsack | 3 | 100.00% | 33.33% | 33.33% |
| knights-tour | 3 | 100.00% | 66.67% | 66.67% |
| life | 14 | 100.00% | 21.43% | 14.29% |
| longdiv | 7 | 100.00% | 71.43% | 71.43% |
| lu-decomp | 3 | 100.00% | 33.33% | 33.33% |
| lz-compress | 2 | 100.00% | 100.00% | 100.00% |
| mandelbrot | 1 | 100.00% | 0.00% | 0.00% |
| matmult | 1 | 100.00% | 0.00% | 0.00% |
| max-subseq | 2 | 100.00% | 0.00% | 0.00% |
| mersenne | 3 | 100.00% | 0.00% | 0.00% |
| minspan | 8 | 100.00% | 37.50% | 25.00% |
| monte-carlo | 1 | 100.00% | 0.00% | 0.00% |
| murmur-hash | 2 | 100.00% | 0.00% | 0.00% |
| n-queens | 3 | 100.00% | 66.67% | 66.67% |
| natlog | 1 | 100.00% | 0.00% | 0.00% |
| nbody-sim | 1 | 100.00% | 0.00% | 0.00% |
| nr-solver | 1 | 100.00% | 100.00% | 100.00% |
| packet-filter | 4 | 100.00% | 25.00% | 25.00% |
| parrondo | 2 | 100.00% | 0.00% | 0.00% |
| pascal | 3 | 100.00% | 33.33% | 33.33% |
| pi-calc | 1 | 100.00% | 0.00% | 0.00% |
| primal-test | 3 | 100.00% | 33.33% | 33.33% |
| priority-queue | 5 | 100.00% | 80.00% | 80.00% |
| qsort-demo | 7 | 100.00% | 28.57% | 28.57% |
| qsort-test | 5 | 100.00% | 80.00% | 80.00% |
| quaternions | 4 | 100.00% | 0.00% | 0.00% |
| rabinkarp-search | 2 | 100.00% | 0.00% | 0.00% |
| rand-test | 3 | 100.00% | 0.00% | 0.00% |
| ransac | 2 | 100.00% | 0.00% | 0.00% |
| regex-parser | 8 | 100.00% | 25.00% | 12.50% |
| rho-factor | 4 | 100.00% | 75.00% | 75.00% |
| rle-compress | 2 | 100.00% | 0.00% | 0.00% |
| rsa-cipher | 4 | 100.00% | 0.00% | 0.00% |
| sat-solver | 5 | 100.00% | 60.00% | 60.00% |
| shortest-path | 3 | 100.00% | 66.67% | 66.67% |
| sieve | 1 | 100.00% | 0.00% | 0.00% |
| simple-grep | 1 | 100.00% | 0.00% | 0.00% |
| spelt2num | 1 | 100.00% | 0.00% | 0.00% |
| spirograph | 2 | 100.00% | 50.00% | 50.00% |
| sudoku-solver | 4 | 100.00% | 50.00% | 50.00% |
| tetris-sim | 12 | 100.00% | 75.00% | 66.67% |
| tiny-NN | 5 | 100.00% | 40.00% | 40.00% |
| topo-sort | 7 | 100.00% | 0.00% | 0.00% |
| totient | 4 | 100.00% | 50.00% | 50.00% |
| transcend | 1 | 100.00% | 0.00% | 0.00% |
| uniquify | 1 | 100.00% | 0.00% | 0.00% |
| vectors-3d | 8 | 100.00% | 12.50% | 0.00% |
| verlet | 1 | 100.00% | 0.00% | 0.00% |
| weekday | 2 | 100.00% | 0.00% | 0.00% |

## Compilation Failures
- ackermann/ackermann.c::main@0x131c
- aes/aes.c::aes_decrypt@0x161b
- aes/aes.c::aes_encrypt@0x1560
- aes/aes.c::inv_shift_rows@0x12cd
- aes/aes.c::key_expansion@0x14c3
- aes/aes.c::main@0x16d1
- aes/aes.c::shift_rows@0x1248
- anagram/anagram.c::BuildMask@0x1372
- anagram/anagram.c::BuildWord@0x15cd
- anagram/anagram.c::DumpWords@0x17e8
- anagram/anagram.c::FindAnagram@0x1839
- anagram/anagram.c::ReadDict@0x1233
- anagram/anagram.c::main@0x1a93
- audio-codec/audio-codec.c::decode@0x1271
- audio-codec/audio-codec.c::encode@0x11e9
- audio-codec/audio-codec.c::main@0x12d7
- avl-tree/avlcore.c::CheckTreeNodeRotation@0x186a
- avl-tree/element.c::Compare@0x1764
- avl-tree/avlcore.c::DeleteByElement@0x1d2b
- avl-tree/avlcore.c::DeleteByElementRecursive@0x1b8b
- avl-tree/avlcore.c::DoubleLeftRotation@0x1845
- avl-tree/avlcore.c::DoubleRightRotation@0x1821
- avl-tree/avlcore.c::FindByElement@0x1790
- avl-tree/avlcore.c::Height@0x1d6e
- avl-tree/avlcore.c::Insert@0x1a73
- avl-tree/avlcore.c::InsertNode@0x199b
- avl-tree/avl-tree.c::main@0x1380
- avl-tree/avl-tree.c::printTree@0x11e9
- banner/banner.c::main@0x11e9
- bit-kernels/bit-kernels.c::main@0x12e8
- blake2b/blake2b.c::F@0x1258
- blake2b/blake2b.c::G@0x11e9
- blake2b/blake2b.c::blake2b@0x1616
- blake2b/blake2b.c::test@0x1982
- bloom-filter/bloom-filter.c::bad_search@0x11e9
- bloom-filter/bloom-filter.c::main@0x1217
- boyer-moore-search/boyer-moore-search.c::badCharHeuristic@0x11e9
- boyer-moore-search/boyer-moore-search.c::main@0x1329
- boyer-moore-search/boyer-moore-search.c::search@0x1223
- c-interp/c-interp.c::eval@0x35d3
- c-interp/c-interp.c::function_body@0x310b
- c-interp/c-interp.c::main@0x3c45
- c-interp/c-interp.c::next@0x11e9
- ccmac/ccmac.c::main@0x11e9
- checkers/functions.c::fill_print_initial@0x15dd
- checkers/functions.c::link_new_node@0x204d
- checkers/checkers.c::main@0x11e9
- cipher/cipher.c::encipher@0x11e9
- cipher/cipher.c::main@0x12b3
- congrad/congrad.c::cg_spmv@0x11e9
- congrad/congrad.c::main@0x125a
- connect4-minimax/connect4-minimax.c::init_board@0x11e9
- connect4-minimax/connect4-minimax.c::main@0x1c5d
- connect4-minimax/connect4-minimax.c::minimax@0x17ed
- connect4-minimax/connect4-minimax.c::play_game@0x1b13
- connect4-minimax/connect4-minimax.c::score_position@0x158e
- convex-hull/convex-hull.c::main@0x130d
- dhrystone/dhrystone.c::PFunc_1@0x12ab
- dhrystone/dhrystone.c::PFunc_2@0x12c8
- dhrystone/dhrystone.c::main@0x1311
- distinctness/distinctness.c::isDistinct@0x11e9
- distinctness/distinctness.c::main@0x1342
- fft-int/fft-int.c::db_from_ampl@0x1513
- flood-fill/flood-fill.c::main@0x130f
- frac-calc/frac-calc.c::avaliatokens@0x1421
- frac-calc/frac-calc.c::calcula@0x172a
- frac-calc/frac-calc.c::copyr@0x12b5
- frac-calc/frac-calc.c::divtokens@0x1636
- frac-calc/frac-calc.c::help@0x11e9
- frac-calc/frac-calc.c::main@0x18c1
- fuzzy-match/fuzzy-match.c::fuzzy_match_recurse@0x21e9
- fuzzy-match/fuzzy-match.c::main@0x2391
- fy-shuffle/fy-shuffle.c::fy_shuffle@0x11e9
- fy-shuffle/fy-shuffle.c::main@0x12de
- gcd-list/gcd-list.c::gcd@0x11e9
- gcd-list/gcd-list.c::main@0x121c
- grad-descent/grad-descent.c::derivateWRTBias@0x1247
- grad-descent/grad-descent.c::derivateWRTWeight@0x11e9
- grad-descent/grad-descent.c::gradientDescent@0x129d
- grad-descent/grad-descent.c::main@0x1312
- graph-tests/graph-tests.c::addEdge@0x127b
- graph-tests/graph-tests.c::addVertex@0x1743
- graph-tests/graph-tests.c::bfs@0x144f
- graph-tests/graph-tests.c::bfs_test@0x150f
- graph-tests/graph-tests.c::bubbleSort@0x15e7
- graph-tests/graph-tests.c::createGraph@0x1206
- graph-tests/graph-tests.c::createNode@0x11e9
- graph-tests/graph-tests.c::createQueue@0x12cd
- graph-tests/graph-tests.c::dequeue@0x1357
- graph-tests/graph-tests.c::enqueue@0x130a
- graph-tests/graph-tests.c::insertAtTheBegin@0x15ae
- graph-tests/graph-tests.c::link_list@0x163c
- graph-tests/graph-tests.c::main@0x1a0e
- graph-tests/graph-tests.c::printQueue@0x13cc
- graph-tests/graph-tests.c::swap@0x15da
- hanoi/hanoi.c::main@0x1261
- heapsort/heapsort.c::main@0x13d4
- heat-calc/heat-calc.c::main@0x11e9
- huff-encode/huff-encode.c::main@0x15ef
- idct-alg/idct-alg.c::main@0x140e
- indirect-test/indirect-test.c::main@0x1257
- k-means/k-means.c::calculateNearst@0x11e9
- k-means/k-means.c::main@0x1922
- k-means/k-means.c::printEPS@0x1546
- kadane/kadane.c::main@0x123b
- kepler/kepler.c::J@0x18c0
- kepler/kepler.c::bin_fact@0x1718
- kepler/kepler.c::binary@0x121d
- kepler/kepler.c::e_series@0x17a2
- kepler/kepler.c::j_series@0x19bb
- kepler/kepler.c::main@0x131f
- knapsack/knapsack.c::main@0x128b
- knapsack/knapsack.c::max@0x11e9
- knights-tour/knights-tour.c::solveKT@0x1341
- life/life.c::getDown@0x1406
- life/life.c::getDownLeft@0x1487
- life/life.c::getDownRight@0x14b4
- life/life.c::getLeft@0x1390
- life/life.c::getNumNeigbors@0x14e2
- life/life.c::getRight@0x13b7
- life/life.c::getUp@0x13df
- life/life.c::getUpLeft@0x142e
- life/life.c::getUpRight@0x145a
- life/life.c::main@0x1664
- life/life.c::process@0x15a3
- longdiv/longdiv.c::main@0x1691
- longdiv/longdiv.c::sub@0x11e9
- lu-decomp/lu-decomp.c::main@0x13ad
- lu-decomp/lu-decomp.c::print_matrix@0x11e9
- mandelbrot/mandelbrot.c::main@0x120d
- matmult/matmult.c::main@0x11e9
- max-subseq/max-subseq.c::lcsAlgo@0x11e9
- max-subseq/max-subseq.c::main@0x14c4
- mersenne/mersenne.c::genrand@0x125b
- mersenne/mersenne.c::main@0x1398
- mersenne/mersenne.c::sgenrand@0x11e9
- minspan/minspan.c::displayGraph@0x13f5
- minspan/minspan.c::displayGraph1@0x14f3
- minspan/minspan.c::displayPath@0x15fa
- minspan/minspan.c::main@0x175b
- minspan/minspan.c::minSpanTree@0x1231
- monte-carlo/monte-carlo.c::main@0x11e9
- murmur-hash/murmur-hash.c::main@0x12a3
- murmur-hash/murmur-hash.c::murmurhash@0x11e9
- n-queens/n-queens.c::main@0x12b1
- natlog/natlog.c::main@0x11e9
- nbody-sim/nbody-sim.c::main@0x11e9
- packet-filter/packet-filter.c::check_packet_filter@0x133d
- packet-filter/packet-filter.c::generate_packet@0x11e9
- packet-filter/packet-filter.c::main@0x145c
- parrondo/parrondo.c::main@0x127d
- parrondo/parrondo.c::play_c@0x1238
- pascal/pascal.c::main@0x12d1
- pascal/pascal.c::print_centered@0x122b
- pi-calc/pi-calc.c::main@0x11e9
- primal-test/primal-test.c::main@0x13ea
- primal-test/primal-test.c::miller_rabin_int@0x1243
- priority-queue/priority-queue.c::main@0x130a
- qsort-demo/qsort-demo.c::main@0x163f
- qsort-demo/qsort-demo.c::print_struct_array@0x1470
- qsort-demo/qsort-demo.c::sort_cstrings_example@0x13b3
- qsort-demo/qsort-demo.c::sort_integers_example@0x1292
- qsort-demo/qsort-demo.c::sort_structs_example@0x14d2
- qsort-test/qsort-test.c::main@0x133f
- quaternions/quaternions.c::euler_from_quat@0x136c
- quaternions/quaternions.c::main@0x15bf
- quaternions/quaternions.c::quat_from_euler@0x11e9
- quaternions/quaternions.c::quaternion_multiply@0x1487
- rabinkarp-search/rabinkarp-search.c::main@0x1366
- rabinkarp-search/rabinkarp-search.c::search@0x11e9
- rand-test/rand-test.c::bad_rand@0x11e9
- rand-test/rand-test.c::main@0x1514
- rand-test/rand-test.c::run_tests@0x1220
- ransac/ransac.c::main@0x13cf
- ransac/ransac.c::ransac_line_fitting@0x1238
- regex-parser/regex-parser.c::main@0x2b4b
- regex-parser/regex-parser.c::matchalphanum@0x21fc
- regex-parser/regex-parser.c::matchcharclass@0x222a
- regex-parser/regex-parser.c::matchone@0x23e1
- regex-parser/regex-parser.c::re_compile@0x270b
- regex-parser/regex-parser.c::re_print@0x2964
- rho-factor/rho-factor.c::main@0x3ef0
- rle-compress/rle-compress.c::main@0x1318
- rle-compress/rle-compress.c::run_length_encode@0x11e9
- rsa-cipher/rsa-cipher.c::main@0x1527
- rsa-cipher/rsa-cipher.c::mod_inverse@0x12f3
- rsa-cipher/rsa-cipher.c::mod_pow@0x11e9
- rsa-cipher/rsa-cipher.c::print_hex_int128@0x1444
- sat-solver/sat-solver.c::main@0x141e
- sat-solver/sat-solver.c::printFormula@0x12ff
- shortest-path/shortest-path.c::main@0x1333
- sieve/sieve.c::main@0x11e9
- simple-grep/simple-grep.c::main@0x11e9
- spelt2num/spelt2num.c::main@0x11e9
- spirograph/spirograph.c::spirograph@0x11e9
- sudoku-solver/sudoku-solver.c::isSafe@0x11e9
- sudoku-solver/sudoku-solver.c::main@0x13e5
- tetris-sim/tetris-sim.c::best_move@0x157c
- tetris-sim/tetris-sim.c::evaluate_board@0x144b
- tetris-sim/tetris-sim.c::main@0x180d
- tiny-NN/tiny-NN.c::main@0x16a4
- tiny-NN/tiny-NN.c::sampleSine@0x1251
- tiny-NN/tiny-NN.c::train@0x133c
- topo-sort/topo-sort.c::addEdge@0x127d
- topo-sort/topo-sort.c::createGraph@0x1223
- topo-sort/topo-sort.c::createListNode@0x1206
- topo-sort/topo-sort.c::createStackNode@0x11e9
- topo-sort/topo-sort.c::main@0x1424
- topo-sort/topo-sort.c::topologicalSort@0x132c
- topo-sort/topo-sort.c::topologicalSortUtil@0x12b7
- totient/totient.c::main@0x12bf
- totient/totient.c::my_gcd@0x11e9
- transcend/transcend.c::main@0x11e9
- uniquify/uniquify.c::main@0x1201
- vectors-3d/vectors-3d.c::get_cross_matrix@0x13c2
- vectors-3d/vectors-3d.c::main@0x14cb
- vectors-3d/vectors-3d.c::print_vector@0x12dc
- vectors-3d/vectors-3d.c::unit_vec@0x1331
- vectors-3d/vectors-3d.c::vector_add@0x121f
- vectors-3d/vectors-3d.c::vector_prod@0x127e
- vectors-3d/vectors-3d.c::vector_sub@0x11e9
- verlet/verlet.c::main@0x11e9
- weekday/weekday.c::dayOfWeek@0x11e9
- weekday/weekday.c::main@0x12ea

## Execution Failures
- cipher/cipher.c::decipher@0x1251
- idct-alg/idct-alg.c::idct_2d@0x1216
- life/life.c::init@0x11e9
- minspan/minspan.c::displayTree@0x16b7
- regex-parser/regex-parser.c::matchpattern@0x2491
- tetris-sim/tetris-sim.c::clear_lines@0x12b6
- vectors-3d/vectors-3d.c::get_angle@0x1429