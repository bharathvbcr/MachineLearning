# Recomputed architecture evidence

Snapshot: 2026-09-04T21:48:01.047462+00:00; Git HEAD: `9ccc0530532ed04af07141a7a52eee498790271b`.

Local snapshot only. Final CE is used; lower is better. Intervals describe seed variation.

3463 result artifacts inventoried; 1252 metrics files parsed.

| Suite | Arm | Finished n | Final CE | 95% seed interval | Parameters | Mean elapsed s |
|---|---|---:|---:|---|---:|---:|
| crossover50m_loop32 | attention | 5 | 4.218434 | [4.19123, 4.24564] | [123699612] | 1491.4 |
| crossover50m_loop32 | attn3 | 5 | 4.410038 | [4.38483, 4.43524] | [59900583] | 674.7 |
| crossover50m_loop32 | attn6 | 5 | 4.295998 | [4.26957, 4.32243] | [81166926] | 831.0 |
| crossover50m_loop32 | looped_attn3x4 | 5 | 4.274210 | [4.24853, 4.29989] | [59900583] | 1414.9 |
| crossover50m_loop32 | looped_attn6x2 | 5 | 4.232161 | [4.20440, 4.25993] | [81166926] | 1440.7 |
| crossover50m_moe32b | attention | 5 | 4.218250 | [4.19065, 4.24585] | [123699612] | 981.3 |
| crossover50m_moe32b | moe_e1k1 | 5 | 4.337491 | [4.31084, 4.36415] | [123708828] | 1090.1 |
| crossover50m_moe32b | moe_e4k1 | 5 | 4.347251 | [4.31745, 4.37705] | [293605788] | 1337.3 |
| crossover50m_moe32b | moe_e8k1 | 5 | 4.354235 | [4.32603, 4.38244] | [520135068] | 1635.9 |
| crossover50m_moe32c | attention | 5 | 4.219769 | [4.19252, 4.24702] | [123699612] | 980.3 |
| crossover50m_moe32c | moe_e1k1 | 5 | 4.218481 | [4.18981, 4.24715] | [123708828] | 1089.3 |
| crossover50m_ratio32 | hybrid_mingru10_attn2 | 5 | 4.230594 | [4.20510, 4.25608] | [153097262] | 1570.9 |
| crossover50m_ratioplace32 | hybrid_mingru10_attn2 | 5 | 4.231331 | [4.20100, 4.26166] | [153097262] | 1575.9 |
| crossover50m_ratioplace32 | hybrid_mingru11_attn1 | 5 | 4.252951 | [4.22567, 4.28024] | [156037027] | 1841.1 |
| crossover50m_ratioplace32 | hybrid_mingru8_attn4 | 5 | 4.201570 | [4.17319, 4.22995] | [147217732] | 1742.4 |
| crossover50m_ratioplace32 | hybrid_mingru_bookend | 5 | 4.259620 | [4.23058, 4.28866] | [153097262] | 1812.3 |
| crossover50m_ratioplace32 | hybrid_mingru_periodic | 5 | 4.193875 | [4.16718, 4.22057] | [150157497] | 1777.8 |
| crossover50m_swa2k | attention | 5 | 4.199478 | [4.16549, 4.23347] | [123699612] | 994.4 |
| crossover50m_swa2k | gdn | 5 | 4.477195 | [4.43947, 4.51492] | [123819552] | 8338.6 |
| crossover50m_swa2k | swa_w256 | 5 | 4.220131 | [4.19196, 4.24830] | [123699612] | 1616.9 |
| crossover50m_swa2k | swa_w512 | 5 | 4.197756 | [4.16363, 4.23188] | [123699612] | 1487.4 |
| crossover50m_swa2k | swa_w64_nosink | 5 | 4.335075 | [4.31058, 4.35957] | [123699612] | 1599.8 |
| crossover50m_swa2k | swa_w64 | 5 | 4.346312 | [4.32213, 4.37049] | [123699612] | 1315.6 |
| crossover50m_swa32 | attention | 5 | 4.217875 | [4.19083, 4.24492] | [123699612] | 985.8 |
| crossover50m_swa32 | gdn | 5 | 4.439986 | [4.41241, 4.46757] | [123819552] | 2581.1 |
| crossover50m_swa32 | hybrid_mingru10_swa2 | 5 | 4.267694 | [4.23620, 4.29918] | [153097262] | 1201.6 |
| crossover50m_swa32 | swa_w128 | 5 | 4.254328 | [4.22653, 4.28213] | [123699612] | 1215.0 |
| crossover50m_swa32 | swa_w256 | 5 | 4.222840 | [4.19584, 4.24984] | [123699612] | 1051.4 |
| crossover50m_swa32 | swa_w64_nosink | 5 | 4.308216 | [4.27959, 4.33684] | [123699612] | 1203.4 |
| crossover50m_swa32 | swa_w64 | 5 | 4.316473 | [4.28617, 4.34678] | [123699612] | 1187.6 |
| crossover_ladder50m | w1152_attention_lr80 | 5 | 4.447686 | [4.42128, 4.47409] | [249332580] | 1480.4 |
| crossover_ladder50m | w1152_mingru_lr40 | 5 | 4.599363 | [4.56579, 4.63294] | [328708248] | 1786.3 |
| crossover_ladder50m | w384_attention_lr80 | 5 | 4.542512 | [4.51094, 4.57408] | [40589268] | 566.1 |
| crossover_ladder50m | w384_mingru_lr40 | 5 | 4.689588 | [4.66413, 4.71505] | [49407384] | 731.7 |
| crossover_ladder50m | w768_attention_lr80 | 5 | 4.470184 | [4.44511, 4.49526] | [123699612] | 985.1 |
| crossover_ladder50m | w768_mingru_lr40 | 5 | 4.628687 | [4.60209, 4.65529] | [158976792] | 1249.4 |
| crossover_ladder_probe | w1152_attention_lr05 | 1 | 5.322640 | not estimated | [249332580] | 295.7 |
| crossover_ladder_probe | w1152_attention_lr10 | 1 | 5.282390 | not estimated | [249332580] | 294.6 |
| crossover_ladder_probe | w1152_attention_lr160 | 1 | 5.187717 | not estimated | [249332580] | 296.9 |
| crossover_ladder_probe | w1152_attention_lr20 | 1 | 5.226696 | not estimated | [249332580] | 293.7 |
| crossover_ladder_probe | w1152_attention_lr320 | 1 | 5.279570 | not estimated | [249332580] | 296.7 |
| crossover_ladder_probe | w1152_attention_lr40 | 1 | 5.192415 | not estimated | [249332580] | 307.8 |
| crossover_ladder_probe | w1152_attention_lr80 | 1 | 5.154124 | not estimated | [249332580] | 307.7 |
| crossover_ladder_probe | w1152_mingru_lr05 | 1 | 5.200211 | not estimated | [328708248] | 371.2 |
| crossover_ladder_probe | w1152_mingru_lr10 | 1 | 5.066927 | not estimated | [328708248] | 371.4 |
| crossover_ladder_probe | w1152_mingru_lr20 | 1 | 5.000186 | not estimated | [328708248] | 293.3 |
| crossover_ladder_probe | w1152_mingru_lr40 | 1 | 4.994708 | not estimated | [328708248] | 384.1 |
| crossover_ladder_probe | w1152_mingru_lr80 | 1 | 5.047858 | not estimated | [328708248] | 384.9 |
| crossover_ladder_probe | w384_attention_lr05 | 1 | 5.821424 | not estimated | [40589268] | 115.8 |
| crossover_ladder_probe | w384_attention_lr10 | 1 | 5.580648 | not estimated | [40589268] | 115.8 |
| crossover_ladder_probe | w384_attention_lr160 | 1 | 5.278161 | not estimated | [40589268] | 115.6 |
| crossover_ladder_probe | w384_attention_lr20 | 1 | 5.397338 | not estimated | [40589268] | 114.5 |
| crossover_ladder_probe | w384_attention_lr320 | 1 | 5.286253 | not estimated | [40589268] | 116.0 |
| crossover_ladder_probe | w384_attention_lr40 | 1 | 5.292028 | not estimated | [40589268] | 115.8 |
| crossover_ladder_probe | w384_attention_lr80 | 1 | 5.228669 | not estimated | [40589268] | 115.8 |
| crossover_ladder_probe | w384_mingru_lr05 | 1 | 5.755710 | not estimated | [49407384] | 147.4 |
| crossover_ladder_probe | w384_mingru_lr10 | 1 | 5.484487 | not estimated | [49407384] | 147.1 |
| crossover_ladder_probe | w384_mingru_lr20 | 1 | 5.269626 | not estimated | [49407384] | 146.0 |
| crossover_ladder_probe | w384_mingru_lr40 | 1 | 5.165134 | not estimated | [49407384] | 150.5 |
| crossover_ladder_probe | w384_mingru_lr80 | 1 | 5.177213 | not estimated | [49407384] | 150.4 |
| crossover_ladder_probe | w768_attention_lr05 | 1 | 5.494775 | not estimated | [123699612] | 197.2 |
| crossover_ladder_probe | w768_attention_lr10 | 1 | 5.360766 | not estimated | [123699612] | 196.9 |
| crossover_ladder_probe | w768_attention_lr160 | 1 | 5.199858 | not estimated | [123699612] | 197.3 |
| crossover_ladder_probe | w768_attention_lr20 | 1 | 5.270066 | not estimated | [123699612] | 196.9 |
| crossover_ladder_probe | w768_attention_lr320 | 1 | 5.261701 | not estimated | [123699612] | 197.5 |
| crossover_ladder_probe | w768_attention_lr40 | 1 | 5.210232 | not estimated | [123699612] | 203.2 |
| crossover_ladder_probe | w768_attention_lr80 | 1 | 5.166343 | not estimated | [123699612] | 203.0 |
| crossover_ladder_probe | w768_mingru_lr05 | 1 | 5.377460 | not estimated | [158976792] | 249.4 |
| crossover_ladder_probe | w768_mingru_lr10 | 1 | 5.184515 | not estimated | [158976792] | 248.5 |
| crossover_ladder_probe | w768_mingru_lr20 | 1 | 5.076468 | not estimated | [158976792] | 246.3 |
| crossover_ladder_probe | w768_mingru_lr40 | 1 | 5.049375 | not estimated | [158976792] | 258.3 |
| crossover_ladder_probe | w768_mingru_lr80 | 1 | 5.091859 | not estimated | [158976792] | 258.2 |
| crossover_wcloop32 | attention | 5 | 4.651038 | [4.64225, 4.65982] | [123699612] | 672.3 |
| crossover_wcloop32 | attn3 | 5 | 4.411078 | [4.38582, 4.43634] | [59900583] | 671.3 |
| crossover_wcloop32 | attn6 | 5 | 4.430246 | [4.41436, 4.44613] | [81166926] | 582.7 |
| crossover_wcloop32 | looped_attn3x4 | 5 | 4.682466 | [4.65306, 4.71188] | [59900583] | 674.4 |
| crossover_wcloop32 | looped_attn6x2 | 5 | 4.637415 | [4.61153, 4.66330] | [81166926] | 672.8 |

## Recall cells

A hybrid is identified by its full layer layout; grouping only by the base mixer would combine different architectures.

| Suite | Family / pairs / steps / batch / context | Solved | Median recall |
|---|---|---:|---:|
| mqar_e16_board | ('attention', 64, 3000, 64, 255) | 11/15 | 0.999971 |
| mqar_e16_board | ('gdn', 64, 3000, 64, 255) | 0/15 | 0.003896 |
| mqar_e16_board | ('mingru', 64, 3000, 64, 255) | 0/15 | 0.003896 |
| mqar_e16_board | ('swa_w64', 64, 3000, 64, 255) | 0/15 | 0.060518 |
| mqar_e16_board | ('swa_w64_nosink', 64, 3000, 64, 255) | 0/15 | 0.029404 |
| mqar_e16_control | ('attention', 16, 3000, 256, 63) | 3/15 | 0.244473 |
| mqar_e16_control | ('swa_w64', 16, 3000, 256, 63) | 3/15 | 0.245020 |
| mqar_e16_seq511 | ('attention', 128, 3000, 128, 511) | 8/10 | 0.999990 |
| mqar_e16_seq511 | ('gdn', 128, 3000, 128, 511) | 0/10 | 0.001991 |
| mqar_e16_seq511 | ('mingru', 128, 3000, 128, 511) | 0/10 | 0.001982 |
| mqar_e16_seq511 | ('swa_w64', 128, 3000, 128, 511) | 0/10 | 0.013997 |
| mqar_e16_seq511 | ('swa_w64_nosink', 128, 3000, 128, 511) | 0/10 | 0.003176 |
| mqar_e16_window511 | ('swa_w128', 128, 3000, 128, 511) | 0/10 | 0.005748 |
| mqar_e16_window511 | ('swa_w256', 128, 3000, 128, 511) | 6/10 | 0.952000 |
| mqar_e16_window511 | ('swa_w512', 128, 3000, 128, 511) | 8/10 | 0.999982 |
| mqar_e8 | ('attention', 4, 3000, 256, None) | 15/15 | 1.000000 |
| mqar_e8 | ('attention', 4, 9000, 256, None) | 15/15 | 1.000000 |
| mqar_e8 | ('attention', 6, 3000, 256, None) | 9/15 | 0.998177 |
| mqar_e8 | ('attention', 6, 9000, 256, None) | 15/15 | 0.999974 |
| mqar_e8 | ('attention', 8, 3000, 256, None) | 2/15 | 0.380801 |
| mqar_e8 | ('attention', 8, 9000, 256, None) | 12/15 | 1.000000 |
| mqar_e8 | ('attention', None, 6000, None, None) | 0/1 | 0.542500 |
| mqar_e8 | ('gdn', 4, 3000, 256, None) | 7/15 | 0.719766 |
| mqar_e8 | ('gdn', 4, 9000, 256, None) | 13/15 | 0.998984 |
| mqar_e8 | ('gdn', 6, 3000, 256, None) | 1/15 | 0.446927 |
| mqar_e8 | ('gdn', 8, 3000, 256, None) | 2/15 | 0.373496 |
| mqar_e8 | ('gdn', 8, 9000, 256, None) | 6/15 | 0.422539 |
| mqar_e8 | ('gdn*3,attention,gdn*3,attention,gdn*3,attention', 4, 3000, 256, None) | 9/15 | 0.998555 |
| mqar_e8 | ('gdn*3,attention,gdn*3,attention,gdn*3,attention', 4, 9000, 256, None) | 15/15 | 0.999844 |
| mqar_e8 | ('gdn*3,attention,gdn*3,attention,gdn*3,attention', 6, 3000, 256, None) | 7/15 | 0.453047 |
| mqar_e8 | ('gdn*3,attention,gdn*3,attention,gdn*3,attention', 8, 3000, 256, None) | 6/15 | 0.382637 |
| mqar_e8 | ('gdn*3,attention,gdn*3,attention,gdn*3,attention', 8, 9000, 256, None) | 13/15 | 1.000000 |
| mqar_e8 | ('mingru', 4, 3000, 256, None) | 1/15 | 0.559805 |
| mqar_e8 | ('mingru', 4, 9000, 256, None) | 1/15 | 0.562461 |
| mqar_e8 | ('mingru', 6, 3000, 256, None) | 0/15 | 0.433828 |
| mqar_e8 | ('mingru', 8, 3000, 256, None) | 0/15 | 0.361309 |
| mqar_e8 | ('mingru', 8, 9000, 256, None) | 0/15 | 0.375762 |
| mqar_e8 | ('mingru*10,attention*2', 4, 3000, 256, None) | 5/15 | 0.561797 |
| mqar_e8 | ('mingru*10,attention*2', 4, 9000, 256, None) | 12/15 | 1.000000 |
| mqar_e8 | ('mingru*10,attention*2', 6, 3000, 256, None) | 8/15 | 0.999583 |
| mqar_e8 | ('mingru*10,attention*2', 6, 9000, 256, None) | 12/15 | 1.000000 |
| mqar_e8 | ('mingru*10,attention*2', 8, 3000, 256, None) | 12/15 | 1.000000 |
| mqar_e8 | ('mingru*10,attention*2', 8, 9000, 256, None) | 13/15 | 1.000000 |
