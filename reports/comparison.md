# v1 vs v2 on development (40 answers, G=376)

| metric | v1 | v2 | delta |
|---|---|---|---|
| joint F1 (main) | 0.380 | 0.299 | -0.081 |
| joint precision | 0.433 | 0.372 | -0.062 |
| joint recall = coverage C/G | 0.338 | 0.250 | -0.088 |
| extraction F1 | 0.604 | 0.550 | -0.054 |
| extraction precision | 0.689 | 0.684 | -0.006 |
| extraction recall | 0.537 | 0.460 | -0.077 |
| macro F1 | 0.383 | 0.301 | -0.082 |
| supported F1 | 0.450 | 0.413 | -0.037 |
| contradicted F1 | 0.354 | 0.268 | -0.086 |
| unknown F1 | 0.344 | 0.221 | -0.123 |

| count | v1 | v2 |
|---|---|---|
| P | 293 | 253 |
| M | 202 | 173 |
| C | 127 | 94 |
| unmatched_predictions | 91 | 80 |
| answer_errors | 18 | 20 |

## Per-answer joint F1 changes

| answer | v1 | v2 | delta | status v1/v2 |
|---|---|---|---|---|
| A03 | 0.44 | 0.42 | -0.02 | ok/ok |
| A04 | 0.38 | 0.00 | -0.38 | ok/error |
| A05 | 0.94 | 0.82 | -0.12 | ok/ok |
| A06 | 0.00 | 0.37 | +0.37 | error/ok |
| A07 | 0.36 | 0.00 | -0.36 | ok/error |
| A08 | 0.64 | 0.50 | -0.14 | ok/ok |
| A09 | 0.16 | 0.00 | -0.16 | ok/error |
| A12 | 0.00 | 0.33 | +0.33 | error/ok |
| A13 | 0.00 | 0.30 | +0.30 | error/ok |
| A14 | 0.38 | 0.00 | -0.38 | ok/error |
| A15 | 0.50 | 0.00 | -0.50 | ok/error |
| A16 | 0.36 | 0.44 | +0.08 | ok/ok |
| A17 | 0.69 | 0.57 | -0.12 | ok/ok |
| A18 | 0.00 | 0.36 | +0.36 | error/ok |
| A19 | 0.73 | 0.00 | -0.73 | ok/error |
| A21 | 0.57 | 0.38 | -0.19 | ok/ok |
| A23 | 0.00 | 0.35 | +0.35 | error/ok |
| A24 | 0.73 | 0.00 | -0.73 | ok/error |
| A26 | 0.00 | 0.30 | +0.30 | error/ok |
| A27 | 0.00 | 0.35 | +0.35 | error/ok |
| A28 | 0.38 | 0.48 | +0.10 | ok/ok |
| A29 | 0.35 | 0.33 | -0.01 | ok/ok |
| A31 | 0.44 | 0.37 | -0.07 | ok/ok |
| A32 | 0.00 | 0.76 | +0.76 | error/ok |
| A33 | 0.54 | 0.00 | -0.54 | ok/error |
| A35 | 0.33 | 0.00 | -0.33 | ok/error |
| A36 | 0.00 | 0.27 | +0.27 | error/ok |
| A37 | 0.67 | 0.00 | -0.67 | ok/error |
| A38 | 0.58 | 0.00 | -0.58 | ok/error |
| A40 | 0.45 | 0.42 | -0.04 | ok/ok |

Answers improved: 11; regressed: 19; unchanged: 10.

## Claim-level: fixed 43, regressed 76

Regressed claims (correct in v1, not in v2):

- A03:1564:1742: wrong_label
- A04:1984:2165: missed
- A04:2363:2540: missed
- A04:324:493: missed
- A04:696:899: missed
- A04:900:1097: missed
- A05:1830:1980: wrong_label
- A07:1534:1719: missed
- A07:1721:1919: missed
- A07:1920:2143: missed
- A07:987:1153: missed
- A08:335:525: missed
- A08:687:889: wrong_label
- A09:1964:2148: missed
- A09:324:491: missed
- A14:1324:1471: missed
- A14:1472:1662: missed
- A14:558:756: missed
- A14:944:1127: missed
- A15:1075:1236: missed
- A15:1433:1627: missed
- A15:1628:1825: missed
- A15:1827:2026: missed
- A15:2027:2174: missed
- A15:700:889: missed
- A17:1055:1247: wrong_label
- A17:1444:1607: wrong_label
- A17:1608:1785: wrong_label
- A17:335:506: missed
- A19:1068:1268: missed
- A19:1465:1654: missed
- A19:1655:1832: missed
- A19:2009:2201: missed
- A19:2202:2372: missed
- A19:371:555: missed
- A19:731:902: missed
- A19:903:1067: missed
- A21:1284:1478: wrong_label
- A21:1479:1676: wrong_label
- A24:1092:1287: missed
- A24:1484:1675: missed
- A24:1676:1854: missed
- A24:1856:2025: missed
- A24:2026:2214: missed
- A24:2215:2381: missed
- A24:685:890: missed
- A24:891:1091: missed
- A29:1550:1701: wrong_label
- A29:1883:2066: wrong_label
- A31:1435:1615: wrong_label
- A31:655:854: wrong_label
- A33:1575:1780: missed
- A33:1952:2111: missed
- A33:2112:2307: missed
- A33:2309:2509: missed
- A33:499:656: missed
- A33:658:842: missed
- A33:843:1027: missed
- A35:1071:1245: missed
- A35:1607:1804: missed
- A35:2130:2321: missed
- A35:689:883: missed
- A37:1097:1275: missed
- A37:1472:1658: missed
- A37:1659:1853: missed
- A37:2010:2217: missed
- A37:2218:2407: missed
- A37:371:556: missed
- A37:918:1096: missed
- A38:1208:1388: missed
- A38:1389:1592: missed
- A38:1593:1792: missed
- A38:1978:2164: missed
- A38:487:662: missed
- A38:664:843: missed
- A38:844:1011: missed

Transitions (v1 -> v2):

- correct -> correct: 51
- correct -> missed: 64
- correct -> wrong_label: 12
- missed -> correct: 39
- missed -> missed: 95
- missed -> wrong_label: 40
- wrong_label -> correct: 4
- wrong_label -> missed: 44
- wrong_label -> wrong_label: 27
