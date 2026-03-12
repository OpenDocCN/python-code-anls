# Infer-Out Model 2 Evaluation (merged.O0.func_map.infer-host)

- Timestamp: 20251119-171008
- Source JSONL: merged.O0.func_map.infer.jsonl
- Target: host
- Total cases: 382
- Replacement success: 382 (100.00%)
- Compilable: 192 (50.26%)
- Executable: 189 (49.48%)

## Benchmark Breakdown
| Benchmark | Cases | Replacement% | Build% | Exec% |
| --- | --- | --- | --- | --- |
| ackermann | 2 | 100.00% | 50.00% | 50.00% |
| aes | 9 | 100.00% | 33.33% | 33.33% |
| anagram | 12 | 100.00% | 58.33% | 58.33% |
| audio-codec | 4 | 100.00% | 50.00% | 50.00% |
| avl-tree | 14 | 100.00% | 35.71% | 35.71% |
| banner | 1 | 100.00% | 0.00% | 0.00% |
| bit-kernels | 5 | 100.00% | 100.00% | 100.00% |
| blake2b | 6 | 100.00% | 16.67% | 16.67% |
| bloom-filter | 3 | 100.00% | 33.33% | 33.33% |
| boyer-moore-search | 3 | 100.00% | 0.00% | 0.00% |
| bubble-sort | 2 | 100.00% | 100.00% | 100.00% |
| c-interp | 10 | 100.00% | 70.00% | 70.00% |
| ccmac | 2 | 100.00% | 50.00% | 50.00% |
| checkers | 15 | 100.00% | 80.00% | 80.00% |
| cipher | 3 | 100.00% | 33.33% | 33.33% |
| congrad | 6 | 100.00% | 66.67% | 66.67% |
| connect4-minimax | 13 | 100.00% | 61.54% | 61.54% |
| convex-hull | 4 | 100.00% | 75.00% | 75.00% |
| dhrystone | 5 | 100.00% | 60.00% | 60.00% |
| distinctness | 2 | 100.00% | 0.00% | 0.00% |
| fft-int | 4 | 100.00% | 50.00% | 50.00% |
| flood-fill | 2 | 100.00% | 50.00% | 50.00% |
| frac-calc | 10 | 100.00% | 60.00% | 60.00% |
| fuzzy-match | 4 | 100.00% | 25.00% | 25.00% |
| fy-shuffle | 4 | 100.00% | 50.00% | 50.00% |
| gcd-list | 2 | 100.00% | 0.00% | 0.00% |
| grad-descent | 4 | 100.00% | 75.00% | 75.00% |
| graph-tests | 19 | 100.00% | 21.05% | 21.05% |
| hanoi | 2 | 100.00% | 50.00% | 50.00% |
| heapsort | 2 | 100.00% | 50.00% | 50.00% |
| heat-calc | 1 | 100.00% | 0.00% | 0.00% |
| huff-encode | 12 | 100.00% | 91.67% | 91.67% |
| idct-alg | 4 | 100.00% | 50.00% | 50.00% |
| indirect-test | 2 | 100.00% | 50.00% | 50.00% |
| k-means | 6 | 100.00% | 100.00% | 100.00% |
| kadane | 2 | 100.00% | 50.00% | 50.00% |
| kepler | 7 | 100.00% | 28.57% | 28.57% |
| knapsack | 3 | 100.00% | 33.33% | 33.33% |
| knights-tour | 3 | 100.00% | 66.67% | 66.67% |
| life | 14 | 100.00% | 78.57% | 71.43% |
| longdiv | 7 | 100.00% | 71.43% | 71.43% |
| lu-decomp | 3 | 100.00% | 33.33% | 33.33% |
| lz-compress | 2 | 100.00% | 100.00% | 100.00% |
| mandelbrot | 1 | 100.00% | 0.00% | 0.00% |
| matmult | 1 | 100.00% | 0.00% | 0.00% |
| max-subseq | 2 | 100.00% | 0.00% | 0.00% |
| mersenne | 3 | 100.00% | 0.00% | 0.00% |
| minspan | 8 | 100.00% | 62.50% | 62.50% |
| monte-carlo | 1 | 100.00% | 0.00% | 0.00% |
| murmur-hash | 2 | 100.00% | 0.00% | 0.00% |
| n-queens | 3 | 100.00% | 66.67% | 66.67% |
| natlog | 1 | 100.00% | 0.00% | 0.00% |
| nbody-sim | 1 | 100.00% | 0.00% | 0.00% |
| nr-solver | 1 | 100.00% | 100.00% | 100.00% |
| packet-filter | 3 | 100.00% | 33.33% | 33.33% |
| parrondo | 3 | 100.00% | 33.33% | 33.33% |
| pascal | 3 | 100.00% | 100.00% | 100.00% |
| pi-calc | 1 | 100.00% | 0.00% | 0.00% |
| primal-test | 3 | 100.00% | 0.00% | 0.00% |
| priority-queue | 5 | 100.00% | 80.00% | 80.00% |
| qsort-demo | 5 | 100.00% | 0.00% | 0.00% |
| qsort-test | 3 | 100.00% | 66.67% | 66.67% |
| quaternions | 4 | 100.00% | 0.00% | 0.00% |
| rabinkarp-search | 2 | 100.00% | 50.00% | 50.00% |
| rand-test | 2 | 100.00% | 0.00% | 0.00% |
| ransac | 2 | 100.00% | 50.00% | 50.00% |
| regex-parser | 11 | 100.00% | 72.73% | 63.64% |
| rho-factor | 4 | 100.00% | 75.00% | 75.00% |
| rle-compress | 2 | 100.00% | 50.00% | 50.00% |
| rsa-cipher | 4 | 100.00% | 0.00% | 0.00% |
| sat-solver | 5 | 100.00% | 60.00% | 60.00% |
| shortest-path | 3 | 100.00% | 66.67% | 66.67% |
| sieve | 2 | 100.00% | 50.00% | 50.00% |
| simple-grep | 1 | 100.00% | 0.00% | 0.00% |
| spelt2num | 1 | 100.00% | 0.00% | 0.00% |
| spirograph | 2 | 100.00% | 50.00% | 50.00% |
| sudoku-solver | 4 | 100.00% | 75.00% | 75.00% |
| tetris-sim | 12 | 100.00% | 75.00% | 75.00% |
| tiny-NN | 2 | 100.00% | 50.00% | 50.00% |
| topo-sort | 7 | 100.00% | 0.00% | 0.00% |
| totient | 4 | 100.00% | 75.00% | 75.00% |
| transcend | 3 | 100.00% | 66.67% | 66.67% |
| uniquify | 1 | 100.00% | 0.00% | 0.00% |
| vectors-3d | 8 | 100.00% | 12.50% | 12.50% |
| verlet | 4 | 100.00% | 25.00% | 0.00% |
| weekday | 2 | 100.00% | 0.00% | 0.00% |

## Compilation Failures
- ackermann/ackermann.c::main@0x13b9
- aes/aes.c::aes_decrypt@0x1a65
- aes/aes.c::aes_encrypt@0x1943
- aes/aes.c::inv_shift_rows@0x1396
- aes/aes.c::key_expansion@0x179a
- aes/aes.c::main@0x1b87
- aes/aes.c::shift_rows@0x12e5
- anagram/anagram.c::BuildMask@0x13e7
- anagram/anagram.c::BuildWord@0x17e5
- anagram/anagram.c::FindAnagram@0x1ba6
- anagram/anagram.c::ReadDict@0x121f
- anagram/anagram.c::main@0x1f71
- audio-codec/audio-codec.c::decode@0x12f5
- audio-codec/audio-codec.c::main@0x14b3
- avl-tree/avlcore.c::DeleteByElement@0x240f
- avl-tree/avlcore.c::DeleteByElementRecursive@0x21af
- avl-tree/avlcore.c::DeleteLeftMost@0x2086
- avl-tree/avlcore.c::FindByElement@0x1a46
- avl-tree/avlcore.c::Height@0x2475
- avl-tree/avlcore.c::Insert@0x1fc4
- avl-tree/avlcore.c::SingleLeftRotation@0x1b3a
- avl-tree/avl-tree.c::main@0x1399
- avl-tree/avl-tree.c::printTree@0x11e9
- banner/banner.c::main@0x11e9
- blake2b/blake2b.c::BLAKE2B@0x1a9b
- blake2b/blake2b.c::F@0x1502
- blake2b/blake2b.c::G@0x1258
- blake2b/blake2b.c::blake2b@0x1cd3
- blake2b/blake2b.c::test@0x2071
- bloom-filter/bloom-filter.c::bad_search@0x11e9
- bloom-filter/bloom-filter.c::main@0x123d
- boyer-moore-search/boyer-moore-search.c::badCharHeuristic@0x11e9
- boyer-moore-search/boyer-moore-search.c::main@0x146d
- boyer-moore-search/boyer-moore-search.c::search@0x126d
- c-interp/c-interp.c::eval@0x457c
- c-interp/c-interp.c::main@0x4e03
- c-interp/c-interp.c::next@0x11e9
- ccmac/ccmac.c::main@0x127e
- checkers/functions.c::fill_print_initial@0x1793
- checkers/functions.c::generate_node_children@0x29ff
- checkers/checkers.c::main@0x11e9
- cipher/cipher.c::encipher@0x11e9
- cipher/cipher.c::main@0x13cd
- congrad/congrad.c::cg_solve@0x1643
- congrad/congrad.c::main@0x199b
- connect4-minimax/connect4-minimax.c::init_board@0x11e9
- connect4-minimax/connect4-minimax.c::main@0x2299
- connect4-minimax/connect4-minimax.c::minimax@0x1d07
- connect4-minimax/connect4-minimax.c::play_game@0x20d1
- connect4-minimax/connect4-minimax.c::score_position@0x1a02
- convex-hull/convex-hull.c::main@0x13e7
- dhrystone/dhrystone.c::Proc_1@0x199f
- dhrystone/dhrystone.c::main@0x11e9
- distinctness/distinctness.c::isDistinct@0x11e9
- distinctness/distinctness.c::main@0x15d8
- fft-int/fft-int.c::db_from_ampl@0x1807
- fft-int/fft-int.c::fix_fft@0x11e9
- flood-fill/flood-fill.c::main@0x144d
- frac-calc/frac-calc.c::copyr@0x14d4
- frac-calc/frac-calc.c::divtokens@0x15b8
- frac-calc/frac-calc.c::help@0x13d9
- frac-calc/frac-calc.c::main@0x11e9
- fuzzy-match/fuzzy-match.c::compute_score@0x2379
- fuzzy-match/fuzzy-match.c::fuzzy_match_recurse@0x2283
- fuzzy-match/fuzzy-match.c::main@0x24b3
- fy-shuffle/fy-shuffle.c::main@0x1378
- fy-shuffle/fy-shuffle.c::rand_int@0x11e9
- gcd-list/gcd-list.c::gcd@0x11e9
- gcd-list/gcd-list.c::main@0x125e
- grad-descent/grad-descent.c::main@0x1413
- graph-tests/graph-tests.c::addEdge@0x12c9
- graph-tests/graph-tests.c::addVertex@0x19f6
- graph-tests/graph-tests.c::bfs@0x15ce
- graph-tests/graph-tests.c::bfs_test@0x16e9
- graph-tests/graph-tests.c::bubbleSort@0x1829
- graph-tests/graph-tests.c::createGraph@0x1221
- graph-tests/graph-tests.c::createNode@0x11e9
- graph-tests/graph-tests.c::createQueue@0x1372
- graph-tests/graph-tests.c::dequeue@0x145d
- graph-tests/graph-tests.c::enqueue@0x13d7
- graph-tests/graph-tests.c::insertAtTheBegin@0x17b1
- graph-tests/graph-tests.c::link_list@0x18b8
- graph-tests/graph-tests.c::main@0x1d6c
- graph-tests/graph-tests.c::printQueue@0x151b
- graph-tests/graph-tests.c::swap@0x17f8
- hanoi/hanoi.c::main@0x12d4
- heapsort/heapsort.c::main@0x155f
- heat-calc/heat-calc.c::main@0x11e9
- huff-encode/huff-encode.c::main@0x192d
- idct-alg/idct-alg.c::C@0x11e9
- idct-alg/idct-alg.c::main@0x1472
- indirect-test/indirect-test.c::main@0x12c9
- kadane/kadane.c::main@0x1276
- kepler/kepler.c::bin_fact@0x1b3e
- kepler/kepler.c::binary@0x12c6
- kepler/kepler.c::e_series@0x1389
- kepler/kepler.c::j_series@0x1501
- kepler/kepler.c::main@0x1608
- knapsack/knapsack.c::main@0x138e
- knapsack/knapsack.c::max@0x11e9
- knights-tour/knights-tour.c::solveKT@0x12d6
- life/life.c::getNumNeigbors@0x156f
- life/life.c::main@0x11e9
- life/life.c::process@0x1426
- longdiv/longdiv.c::main@0x18fd
- longdiv/longdiv.c::sub@0x11e9
- lu-decomp/lu-decomp.c::main@0x1520
- lu-decomp/lu-decomp.c::print_matrix@0x11e9
- mandelbrot/mandelbrot.c::main@0x1220
- matmult/matmult.c::main@0x11e9
- max-subseq/max-subseq.c::lcsAlgo@0x11e9
- max-subseq/max-subseq.c::main@0x171a
- mersenne/mersenne.c::genrand@0x12ee
- mersenne/mersenne.c::main@0x153a
- mersenne/mersenne.c::sgenrand@0x11e9
- minspan/minspan.c::displayPath@0x1af2
- minspan/minspan.c::main@0x1d8f
- minspan/minspan.c::minSpanTree@0x1297
- monte-carlo/monte-carlo.c::main@0x11e9
- murmur-hash/murmur-hash.c::main@0x13a9
- murmur-hash/murmur-hash.c::murmurhash@0x11e9
- n-queens/n-queens.c::main@0x12ec
- natlog/natlog.c::main@0x11e9
- nbody-sim/nbody-sim.c::main@0x11e9
- packet-filter/packet-filter.c::generate_packet@0x11e9
- packet-filter/packet-filter.c::main@0x14c3
- parrondo/parrondo.c::cointoss@0x11e9
- parrondo/parrondo.c::main@0x12cb
- pi-calc/pi-calc.c::main@0x11e9
- primal-test/primal-test.c::main@0x1459
- primal-test/primal-test.c::miller_rabin_int@0x12fd
- primal-test/primal-test.c::powm@0x11e9
- priority-queue/priority-queue.c::main@0x13ee
- qsort-demo/qsort-demo.c::main@0x17bf
- qsort-demo/qsort-demo.c::print_struct_array@0x155e
- qsort-demo/qsort-demo.c::sort_cstrings_example@0x1401
- qsort-demo/qsort-demo.c::sort_integers_example@0x1280
- qsort-demo/qsort-demo.c::sort_structs_example@0x1603
- qsort-test/qsort-test.c::main@0x1415
- quaternions/quaternions.c::euler_from_quat@0x1447
- quaternions/quaternions.c::quat_from_euler@0x11e9
- quaternions/quaternions.c::quaternion_multiply@0x1655
- quaternions/quaternions.c::test@0x18b2
- rabinkarp-search/rabinkarp-search.c::main@0x1341
- rand-test/rand-test.c::main@0x1913
- rand-test/rand-test.c::run_tests@0x1258
- ransac/ransac.c::main@0x1466
- regex-parser/regex-parser.c::main@0x32b9
- regex-parser/regex-parser.c::re_compile@0x22e1
- regex-parser/regex-parser.c::re_print@0x278f
- rho-factor/rho-factor.c::main@0x5c7d
- rle-compress/rle-compress.c::run_length_encode@0x11e9
- rsa-cipher/rsa-cipher.c::main@0x1634
- rsa-cipher/rsa-cipher.c::mod_inverse@0x1363
- rsa-cipher/rsa-cipher.c::mod_pow@0x11e9
- rsa-cipher/rsa-cipher.c::print_hex_int128@0x14ef
- sat-solver/sat-solver.c::main@0x1518
- sat-solver/sat-solver.c::printFormula@0x1391
- shortest-path/shortest-path.c::main@0x1469
- sieve/sieve.c::main@0x1300
- simple-grep/simple-grep.c::main@0x11e9
- spelt2num/spelt2num.c::main@0x11e9
- spirograph/spirograph.c::spirograph@0x11e9
- sudoku-solver/sudoku-solver.c::main@0x1532
- tetris-sim/tetris-sim.c::best_move@0x1810
- tetris-sim/tetris-sim.c::evaluate_board@0x1686
- tetris-sim/tetris-sim.c::main@0x1ba5
- tiny-NN/tiny-NN.c::train@0x1485
- topo-sort/topo-sort.c::addEdge@0x12cf
- topo-sort/topo-sort.c::createGraph@0x1259
- topo-sort/topo-sort.c::createListNode@0x1221
- topo-sort/topo-sort.c::createStackNode@0x11e9
- topo-sort/topo-sort.c::main@0x153d
- topo-sort/topo-sort.c::topologicalSort@0x13fd
- topo-sort/topo-sort.c::topologicalSortUtil@0x1332
- totient/totient.c::my_gcd@0x11e9
- transcend/transcend.c::init_inputs_f64@0x1235
- uniquify/uniquify.c::main@0x1228
- vectors-3d/vectors-3d.c::get_cross_matrix@0x1601
- vectors-3d/vectors-3d.c::print_vector@0x144f
- vectors-3d/vectors-3d.c::test@0x17fb
- vectors-3d/vectors-3d.c::unit_vec@0x1510
- vectors-3d/vectors-3d.c::vector_add@0x126d
- vectors-3d/vectors-3d.c::vector_prod@0x1373
- vectors-3d/vectors-3d.c::vector_sub@0x11e9
- verlet/verlet.c::main@0x170b
- verlet/verlet.c::vb_init@0x1271
- verlet/verlet.c::vb_step_avg@0x13aa
- weekday/weekday.c::dayOfWeek@0x11e9
- weekday/weekday.c::main@0x130d

## Execution Failures
- life/life.c::init@0x1237
- regex-parser/regex-parser.c::matchpattern@0x313f
- verlet/verlet.c::vb_checksum@0x160b