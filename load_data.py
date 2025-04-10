import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io
from sklearn.preprocessing import MinMaxScaler
import random
import time
import h5py

seed = int(time.time() * 1000) % 2**32
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class BDGP(Dataset):
    def __init__(self, path, num_user, Dirichlet_alpha, w):
        X1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)         # (2500, 1750)
        X2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)         # (2500, 79)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()             # (2500, 1)

        self.x1 = X1
        self.x2 = X2
        self.y = labels

        self.num_user = num_user
        self.Dirichlet_alpha = Dirichlet_alpha
        self.w = w
        self.user_data = self.split_data()

    def __getitem__(self, idx):
        item_1 = torch.from_numpy(self.x1[idx])
        item_2 = torch.from_numpy(self.x2[idx])
        item_label = torch.from_numpy(self.y[idx])

        return [item_1,item_2], item_label, torch.from_numpy(np.array(idx)).long()

    def __len__(self):
        return self.x1.shape[0]

    def split_data(self):  # partial noniid, partial iid
        N = int(len(self.y))
        n_classes = int(max(self.y) + 1)
        iiduser_num = int(self.num_user * self.w)
        noniiduser_num = self.num_user - iiduser_num
        iidsample_num = int(N * self.w)
        iid_lst = np.random.choice(N, size=iidsample_num, replace=False)
        mask = np.isin(np.arange(N), iid_lst)[:, np.newaxis]

        dict_users = {i: np.array([]) for i in range(self.num_user)}
        min_require_size = 10

        # iid
        min_size = 0
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(iiduser_num)]
            for label in range(n_classes):
                label_idx = np.where((self.y == label) & mask)[0]
                np.random.shuffle(label_idx)

                label_distribution = np.random.dirichlet([999999] * iiduser_num)  # (n_classes, self.num_user)
                label_distribution = np.array([p * (len(idx_j) < (iidsample_num / iiduser_num)) for p, idx_j in zip(label_distribution, idx_batch)])
                label_distribution = label_distribution / label_distribution.sum()
                label_distribution = (np.cumsum(label_distribution) * len(label_idx)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(label_idx, label_distribution))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(iiduser_num):
                np.random.shuffle(idx_batch[j])
                dict_users[j] = idx_batch[j]

        # noniid
        min_size = 0
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(noniiduser_num)]
            for label in range(n_classes):
                label_idx = np.where((self.y == label) & ~mask)[0]
                np.random.shuffle(label_idx)

                label_distribution = np.random.dirichlet([self.Dirichlet_alpha] * noniiduser_num)
                label_distribution = np.array([p * (len(idx_j) < ((N-iidsample_num) / noniiduser_num)) for p, idx_j in zip(label_distribution, idx_batch)])
                label_distribution = label_distribution / label_distribution.sum()
                label_distribution = (np.cumsum(label_distribution) * len(label_idx)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(label_idx, label_distribution))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(noniiduser_num):
                np.random.shuffle(idx_batch[j])
                dict_users[j+iiduser_num] = idx_batch[j]

        dict_users = {0: [15, 1608, 532, 1375, 292, 1392, 2124, 2104, 133, 696, 2172, 701, 1510, 318, 2095, 1691, 224, 1003, 53, 1285, 1097, 1134, 1949, 1060, 1217, 1137, 453, 381, 1823, 2405, 286, 943, 1707, 1671, 713, 2335, 700, 205, 260, 388, 1729, 1515, 409, 2155, 1900, 2066, 1359, 2164, 976, 2217, 1020, 422, 2357, 1491, 1647, 1930, 1463, 213, 2227, 2119, 1499, 351, 1116, 469, 1035, 926, 178, 1976, 735, 2371, 1096, 618, 1171, 1298, 1099, 856, 1686, 1269, 1986, 649, 1659, 2370, 1961, 1890, 2109, 78, 2150, 1832, 2438, 536, 1252, 1237, 1450, 2308, 2493, 1646, 2180, 761, 1833, 1387, 882, 993, 818, 74, 61, 1505, 1698, 1965, 534, 925, 1932, 2484, 674, 1627, 114, 1071, 2264, 613, 1398, 101, 452, 310, 1587, 969, 857, 2105, 1172, 1436, 425, 1353, 1775, 2026, 132, 2229, 1607, 1914, 680, 2192, 418, 1946, 898, 1321, 2047, 2169, 1194, 175, 2319, 257, 584, 601, 458, 658, 733, 1789, 2330, 1037, 667, 1271, 2196, 1981, 634, 2443, 2097, 2410, 1720, 520, 2378, 1652, 1956, 293, 93, 1964, 754, 1362, 805, 1813, 475, 1358, 2138, 70, 595, 1667, 244, 1576, 305, 2260, 136, 1379, 350, 1511, 2223, 608, 2037, 1382, 1886, 670, 1323, 1440, 893, 2391, 2081, 49, 1710, 1064, 144, 359, 1971, 170, 657, 2299, 1372, 935, 47, 426, 1027, 2101, 2445, 312, 2417, 2411, 349, 1879, 2152, 1910, 321, 1462, 2139, 1110, 1118, 1098, 2351, 253, 2486, 1049, 837, 220, 1717, 2179, 1062, 1810, 1804, 1812, 924, 2244, 2379, 1639, 1584, 1370], 1: [706, 431, 1687, 560, 468, 357, 279, 231, 119, 1336, 1342, 446, 755, 767, 692, 1921, 632, 1982, 216, 609, 284, 2272, 1151, 1355, 1498, 1881, 378, 1681, 1365, 2111, 1803, 1407, 1021, 2353, 210, 1255, 2480, 373, 914, 380, 909, 2149, 1808, 2456, 903, 1760, 2043, 1006, 278, 390, 1073, 1388, 392, 1042, 960, 2036, 2466, 721, 1247, 2317, 652, 1561, 717, 1457, 1795, 1947, 1837, 2365, 400, 651, 112, 2288, 2329, 2322, 1747, 1297, 743, 1085, 1393, 148, 1929, 2136, 1653, 1272, 2235, 1396, 1851, 1229, 1074, 2476, 1010, 2086, 1995, 1276, 287, 2208, 1602, 1496, 1571, 1858, 1022, 2153, 2313, 707, 230, 1679, 2248, 1918, 1913, 1397, 384, 2446, 1018, 2121, 1897, 372, 745, 685, 161, 130, 522, 1993, 507, 1787, 2291, 489, 1746, 1562, 563, 556, 2254, 457, 1724, 1292, 627, 1367, 1014, 2112, 1485, 1427, 795, 1354, 1429, 1361, 356, 2412, 2294, 325, 2428, 1186, 2062, 218, 591, 1426, 415, 97, 2469, 323, 594, 2204, 1624, 1357, 1291, 1709, 1494, 313, 2485, 1683, 1901, 1893, 1836, 770, 1968, 52, 603, 1087, 888, 819, 1101, 481, 822, 1088, 1520, 896, 1959, 796, 2372, 7, 283, 83, 1211, 2110, 2226, 2199, 1636, 1103, 2441, 2394, 600, 1001, 1174, 1433, 85, 124, 1419, 142, 2432, 68, 642, 1517, 214, 2269, 340, 660, 2452, 2332, 1193, 2321, 663, 2347, 812, 2267, 209, 2477, 1754, 241, 2054, 1924, 2211, 989, 539, 687, 2135, 1549, 165, 288, 2374, 1133, 2400, 2461, 251, 5, 327, 583, 2212, 1386, 1123, 2073, 677], 2: [691, 143, 1514, 258, 648, 744, 1973, 2129, 1082, 1256, 319, 605, 1519, 803, 1310, 897, 398, 248, 401, 1592, 2475, 235, 348, 1868, 1938, 156, 899, 1772, 1712, 2316, 2362, 2146, 421, 1127, 140, 1430, 1360, 1814, 300, 1222, 1448, 791, 1262, 1412, 2079, 1575, 1559, 1950, 410, 1633, 1741, 75, 1296, 1809, 424, 1609, 2420, 2033, 1453, 656, 2219, 92, 1849, 336, 2238, 2050, 970, 1210, 82, 1208, 2345, 887, 1793, 499, 972, 764, 1785, 370, 1776, 2465, 1770, 2407, 1148, 1591, 1817, 2483, 1075, 1469, 299, 1771, 326, 1188, 753, 799, 2216, 1711, 1066, 1935, 1650, 1167, 2458, 871, 1287, 890, 107, 776, 688, 1731, 1104, 1200, 1522, 99, 2115, 2247, 2473, 1016, 282, 1493, 777, 492, 981, 404, 710, 317, 904, 537, 2044, 147, 1702, 2029, 1978, 711, 1648, 109, 529, 1597, 2016, 597, 766, 664, 152, 472, 526, 13, 1552, 2070, 2290, 948, 1626, 1925, 2327, 1391, 987, 775, 817, 1313, 2189, 968, 1769, 1763, 354, 40, 874, 202, 1799, 1990, 1319, 332, 2132, 1927, 2174, 1077, 512, 544, 2213, 828, 41, 462, 1759, 2368, 1215, 1280, 1801, 1190, 1599, 510, 42, 190, 297, 728, 1955, 1301, 330, 1246, 307, 1011, 90, 306, 840, 1268, 920, 1542, 581, 1058, 1908, 2008, 1598, 249, 1050, 2214, 1732, 703, 683, 2004, 1774, 201, 207, 912, 2363, 1504, 1703, 179, 1079, 1144, 934, 596, 1749, 853, 675, 574, 781, 944, 153, 2059, 2241, 1416, 1314, 1962, 518, 2256, 1040, 834, 2419, 1674, 639, 502, 1798, 982, 2178, 2406, 1158, 247, 786], 3: [1377, 1829, 1820, 223, 1872, 444, 1967, 511, 2209, 564, 524, 465, 1816, 2117, 1034, 2078, 1033, 2107, 1684, 324, 1431, 1516, 56, 1445, 50, 203, 2251, 341, 2168, 184, 1212, 1341, 227, 149, 2, 842, 246, 1413, 1091, 1178, 501, 1090, 1585, 2297, 1767, 2048, 1512, 720, 995, 1216, 1043, 1797, 957, 1625, 334, 1998, 221, 1822, 980, 1111, 1792, 168, 1941, 876, 1112, 855, 669, 1790, 1363, 1563, 2280, 1366, 173, 435, 1704, 2228, 103, 352, 1558, 1180, 598, 937, 1786, 2022, 55, 1482, 1338, 236, 1309, 135, 1157, 2203, 131, 990, 504, 242, 1565, 1409, 2137, 1470, 2181, 922, 1737, 1017, 868, 830, 1991, 276, 1957, 1185, 772, 280, 163, 1307, 98, 86, 1131, 104, 548, 2457, 785, 2447, 689, 1481, 999, 546, 1108, 814, 1443, 1288, 1551, 1325, 1614, 705, 2123, 2042, 1882, 1161, 2071, 1628, 237, 1898, 54, 945, 1642, 1826, 94, 1884, 2429, 157, 141, 371, 906, 1102, 1092, 162, 1315, 1142, 931, 947, 1350, 530, 25, 2005, 1114, 2206, 4, 1281, 403, 867, 245, 1383, 1213, 673, 127, 1460, 2089, 1324, 918, 181, 1863, 1320, 2087, 1473, 1421, 21, 2312, 402, 2270, 256, 1214, 763, 1282, 637, 27, 2421, 1744, 439, 1756, 185, 2398, 1302, 1458, 1169, 182, 1274, 610, 742, 2401, 2038, 959, 2373, 2360, 2186, 1348, 1500, 527, 704, 946, 900, 197, 192, 1610, 2380, 2495, 1061, 2262, 1780, 751, 488, 509, 1725, 1963, 259, 538, 940, 1661, 498, 1889, 2210, 35, 304, 2333, 1086, 1227, 1506, 933, 1513, 1293, 1199, 1670, 1745, 820, 28, 682, 2497, 369, 1364, 1782, 226, 1009, 102, 1722, 2258, 975, 269, 2283, 965, 1303, 450, 2237, 612, 275, 1574, 471, 1238, 1434, 1069, 1286, 2013, 24, 96, 121, 2158, 771, 391, 1682, 850, 1317, 1241, 2133, 1788, 1613, 155, 1569, 1346, 1933, 2464, 1529, 998, 1564, 2198, 1121, 1906, 183, 1796, 2274, 1635, 1461, 2120, 1435, 848, 358, 802, 1339, 1502, 2085, 614, 1345, 1107, 1308, 2392, 1945, 37, 807, 411, 2233, 1844, 437, 1580, 1207, 322, 1761, 2303, 731, 1805, 1239, 690, 1454, 1507], 4: [2193, 1538, 895, 716, 1168, 1781, 549, 1410, 1654, 314, 1875, 1442, 2068, 1449, 1985, 2170, 1369, 1177, 2245, 1381, 2479, 1586, 1573, 180, 1343, 339, 1406, 752, 2271, 1059, 1533, 1155, 171, 1106, 1560, 1977, 665, 1263, 1471, 875, 2148, 838, 1371, 1743, 773, 2454, 984, 1206, 1047, 46, 727, 467, 1225, 2091, 1159, 1306, 1777, 2386, 1539, 629, 586, 1334, 1726, 723, 1952, 808, 902, 1773, 978, 740, 1936, 115, 1705, 347, 311, 442, 2184, 420, 111, 335, 51, 1044, 1657, 1672, 921, 1468, 1439, 1994, 849, 2468, 430, 1025, 1250, 1524, 2492, 654, 979, 1395, 788, 1842, 456, 2130, 1865, 1331, 905, 122, 1664, 1905, 408, 942, 1880, 676, 1176, 916, 545, 827, 2315, 1052, 2141, 1980, 353, 1873, 116, 243, 1140, 1265, 1553, 145, 1057, 2006, 973, 936, 39, 861, 1557, 1417, 554, 1888, 963, 2282, 1715, 22, 174, 31, 798, 364, 1915, 606, 2349, 891, 1668, 1415, 296, 2134, 631, 108, 1221, 342, 1432, 579, 26, 816, 659, 580, 478, 1290, 543, 622, 907, 2032, 232, 653, 1160, 880, 375, 1960, 1028, 277, 9, 2358, 1368, 1606, 2108, 1951, 1762, 1424, 57, 1224, 460, 571, 2467, 2250, 2284, 2307, 158, 2202, 2342, 113, 1128, 17, 1024, 1757, 2414, 780, 2144, 1983, 2338, 765, 738, 2075, 2220, 2224, 923, 138, 1078, 2051, 1827, 63, 443, 1204, 702, 1583, 2449, 542, 919, 1456, 843, 483, 2408, 137, 2205, 2301, 1904, 126, 412, 932, 1351, 1630, 1730, 1165, 2002, 678, 2496, 1048, 2278, 1145, 2197, 69, 747, 274, 562, 414, 835, 187, 2065, 2331, 1570, 1312, 533, 1638, 2128, 552, 1699, 14, 555, 2423, 1854, 204, 1948, 517, 821, 1231, 2252, 1768, 0, 1465, 1907, 1779, 289, 1825, 1031, 134, 1690, 1926, 1534, 2444, 695, 2261, 387, 762, 2494, 860, 2014, 950, 1917, 1677, 2143, 2448, 32, 996, 45, 1845, 864, 303, 1716, 84, 2320, 2225, 513, 1958, 2289, 1197, 2246, 589, 2279, 1444, 636, 2356, 2118, 1919, 699, 1695, 1697, 212, 726, 521, 1472, 1547, 1187, 2045, 1676, 2287, 1045, 1582], 5: [1007, 186, 966, 2167, 264, 271, 2314, 16, 588, 681, 1437, 2096, 1546, 952, 2439, 454, 1311, 558, 2286, 337, 1999, 908, 1000, 1404, 1209, 177, 344, 1264, 810, 1693, 1405, 2142, 1794, 1438, 1992, 1939, 2334, 1294, 1567, 1401, 1540, 570, 1590, 1605, 604, 623, 826, 2399, 1885, 873, 1411, 1588, 281, 432, 967, 2498, 1532, 1390, 1899, 759, 1895, 1490, 1566, 1727, 722, 1080, 1550, 1572, 2201, 396, 2187, 1242, 1089, 1195, 429, 1594, 2034, 697, 1975, 2292, 576, 1937, 2259, 617, 1824, 2156, 1251, 164, 2092, 1072, 2041, 2057, 1029, 1378, 1032, 2009, 1568, 793, 2140, 2409, 1046, 2450, 2188, 628, 929, 67, 252, 2127, 748, 1067, 1902, 2236, 1811, 151, 1254, 1696, 2277, 2425, 228, 2296, 1304, 736, 1877, 91, 646, 2434, 1700, 1486, 1015, 988, 294, 495, 405, 2324, 593, 1053, 448, 2011, 302, 2055, 2182, 198, 2375, 1940, 2298, 1556, 1219, 1253, 2369, 865, 1130, 616, 550, 641, 718, 1673, 493, 1081, 1723, 2100, 2470, 1518, 750, 2424, 80, 1867, 1666, 2102, 859, 2154, 2350, 385, 626, 1943, 1555, 1987, 2306, 778, 886, 2341, 746, 2166, 2103, 800, 961, 12, 1329, 1109, 2039, 379, 1192, 1132, 1326, 1615, 668, 285, 1896, 1335, 519, 1601, 189, 1755, 714, 1617, 955, 1818, 18, 515, 2381, 2145, 466, 565, 436, 2430, 1467, 30, 2106, 1352, 1487, 129, 789, 1175, 1521, 2403, 1464, 268, 2240, 647, 1783, 395, 2422, 10, 1126, 2415, 1008, 1189, 1750, 1408, 1894, 169, 1742, 2177, 2125, 1328, 1012, 2383, 1719, 1641, 602, 29, 451, 1595, 1013, 1497, 366, 377, 1523, 1183, 719, 1267, 2090, 33, 2234, 797, 643, 1989, 2069, 1531, 2015, 760, 1316, 939, 1970, 938, 1530, 1283, 615, 2185, 2442, 2035, 1656, 2427, 645, 433, 573, 709, 500, 732, 100, 1916, 389, 376, 59, 329, 193, 1840, 644, 2361, 233, 62, 222, 845, 839, 2253, 1495, 712, 2376, 506, 419, 1764, 428, 866, 1076, 2207, 2352, 892, 2131, 1751, 1119, 927, 2490, 196, 2304, 1418, 368, 1245, 1284, 1441], 6: [572, 1616, 2072, 1612, 1536, 1030, 1279, 2339, 1736, 1618, 1923, 2390, 679, 60, 854, 48, 737, 877, 1850, 2431, 1589, 176, 200, 1202, 1479, 2433, 1476, 20, 464, 1266, 166, 1508, 2413, 1373, 715, 240, 1643, 234, 1230, 569, 434, 829, 2191, 1147, 2323, 64, 194, 2326, 523, 485, 1603, 81, 941, 308, 2396, 491, 540, 1455, 2336, 343, 1205, 423, 298, 1620, 917, 1041, 362, 459, 1577, 1866, 159, 2215, 1428, 2012, 1136, 2049, 2099, 2046, 1446, 1928, 445, 2018, 2393, 1541, 1384, 1640, 1356, 2255, 1526, 225, 2159, 1105, 427, 2076, 333, 1452, 1023, 2305, 2242, 806, 1876, 1234, 2003, 1039, 34, 1056, 1385, 508, 991, 870, 2010, 1483, 360, 1259, 1115, 505, 219, 1289, 2397, 2231, 578, 2265, 1503, 758, 2232, 954, 449, 2435, 2126, 2366, 638, 1484, 2462, 983, 1807, 1125, 1376, 2116, 120, 599, 2389, 1864, 1911, 2163, 2451, 2001, 1596, 1828, 1477, 1665, 2151, 1019, 44, 913, 1051, 724, 1240, 1002, 2162, 473, 949, 1139, 2348, 1765, 1394, 206, 1891, 930, 790, 1862, 547, 1425, 729, 910, 1545, 1852, 1054, 316, 1150, 1678, 1488, 1869, 1619, 2157, 1887, 2460, 1474, 393, 2190, 894, 1248, 1149, 1273, 1036, 382, 150, 1389, 1846, 1632, 1766, 1124, 1228, 1912, 1972, 2367, 1680, 851, 2311, 881, 1644, 2453, 1806, 1374, 1318, 756, 2455, 1651, 1578, 739, 2474, 567, 2489, 1243, 1966, 774, 1660, 1543, 1152, 831, 516, 568, 2243, 1347, 1094, 1182, 355, 328, 1261, 1713, 1191], 7: [1856, 480, 217, 383, 76, 315, 1735, 476, 1847, 2377, 1344, 438, 1953, 1878, 88, 1623, 528, 878, 974, 1299, 1063, 1400, 525, 65, 89, 2310, 635, 2239, 1226, 1100, 811, 290, 783, 2487, 823, 1349, 1738, 1752, 2478, 441, 1669, 417, 2063, 640, 66, 1838, 2195, 497, 2387, 2488, 1203, 2094, 2416, 494, 1544, 273, 1475, 1117, 1819, 2355, 566, 77, 650, 1170, 1942, 741, 749, 154, 2083, 479, 1093, 1154, 1909, 2114, 768, 1748, 482, 1420, 686, 607, 585, 2067, 1675, 1065, 2293, 1874, 1509, 38, 1728, 1277, 1871, 167, 36, 1839, 95, 1138, 1857, 211, 118, 985, 1903, 266, 986, 1163, 79, 1528, 1855, 1068, 901, 2031, 1997, 23, 2077, 1414, 661, 734, 2023, 1934, 2093, 824, 2053, 1645, 1954, 406, 577, 250, 461, 592, 836, 994, 879, 2354, 2030, 1791, 825, 397, 1129, 2344, 1278, 693, 1853, 2027, 1701, 684, 971, 2194, 2459, 2302, 1270, 1337, 1232, 2058, 730, 1721, 447, 541, 195, 1835, 1758, 2318, 1861, 2175, 1525, 2402, 784, 2309, 1480, 1859, 1184, 208, 624, 1718, 809, 1631, 2176, 872, 1459, 463, 345, 1083, 309, 2122, 1658, 484, 440, 869, 2230, 1173, 1611, 2404, 1333, 841, 2384, 953, 239, 1095, 1831, 8, 1466, 1143, 363, 951, 1604, 915, 2183, 1422, 261, 2052, 1637, 106, 110, 1662, 1848, 883, 587, 2257, 365, 1122, 1579, 958, 1257, 725, 862, 416, 6, 2268, 1233, 2064, 2276, 1005, 2173, 1694, 272, 1153, 215, 346, 575, 804, 846, 2160, 87, 1537, 1340, 407, 1038, 2171, 2165, 1821, 2025, 956, 964, 792, 1778, 1156, 619, 769, 1198, 2222, 2000, 254, 997, 295, 1223, 590, 2074, 367, 1166, 1489, 1969, 1714, 928, 301, 11, 2040, 1860, 2019, 2499, 2263, 815, 1260, 413, 1733, 1330, 1201, 672, 1423, 1478, 620, 1922, 172, 455, 1815, 394, 1581, 1447, 490, 2295, 487, 2098, 361, 1055, 1649, 71, 1141, 801, 2056, 1004, 2491, 1753, 1196, 1244, 1979, 2437, 123, 1634, 2472, 1451, 557, 470, 1179, 229, 1249, 1800, 2343, 1, 331, 1883, 399, 2395, 1548, 474, 199, 374, 2418, 1984, 844, 2218, 2436, 794, 477, 2007, 1295, 1593, 2346, 1275, 2200, 2088, 1739, 1685], 8: [338, 1892, 72, 1802, 1834, 2161, 1332, 2328, 621, 863, 885, 787, 2340, 1235, 2471, 2060, 1706, 128, 1841, 263, 1527, 1629, 1220, 2285, 2385, 1622, 1535, 1740, 535, 2325, 813, 1689, 146, 2388, 911, 2017, 265, 19, 1708, 1070, 270, 1974, 2061, 1399, 611, 2020, 1688, 633, 992, 561, 291, 655, 1258, 267, 962, 2382, 779, 884, 559, 1830, 2113, 105, 782, 2482, 698, 43, 117, 2364, 262, 1403, 1380, 832, 833, 1402, 551, 1322, 1784, 662, 1120, 1843, 1920, 2147, 1146, 757, 2463, 1870, 2084, 2440, 2266, 1663, 320, 1300, 1655, 2281, 1692, 2300, 1600, 553, 1327, 2481, 503, 1554, 2080, 666, 2221, 847, 2359, 1734, 858, 1501, 977, 1135, 889, 486, 2249, 58, 625, 238, 125, 2273], 9: [1621, 2028, 630, 191, 496, 531, 694, 1162, 582, 2426, 73, 852, 1931, 1026, 3, 139, 1181, 1218, 1996, 514, 2021, 255, 2337, 1988, 1236, 1305, 1164, 2275, 1944, 708, 1084, 160, 671, 188, 1113, 386, 2024, 2082, 1492]}

        return dict_users


class Cora(Dataset):
    def __init__(self, path, num_user, Dirichlet_alpha, w):
        data = scipy.io.loadmat(path + "Cora.mat")
        self.x1 = data['coracites'].astype(np.float32)
        self.x2 = data['coracontent'].astype(np.float32)
        self.x3 = data['corainbound'].astype(np.float32)
        self.x4 = data['coraoutbound'].astype(np.float32).transpose()
        self.y = np.copy(data['y']).astype(np.int32).reshape(2708,1)

        self.num_user = num_user
        self.view = 4
        self.Dirichlet_alpha = Dirichlet_alpha
        self.w = w
        self.user_data = self.split_data()


    def __len__(self):
        return self.x1.shape[0]

    def split_data(self):
        N = int(len(self.y))
        n_classes = int(max(self.y))

        iiduser_num = int(self.num_user * self.w)
        noniiduser_num = self.num_user - iiduser_num
        iidsample_num = int(N * self.w)
        iid_lst = np.random.choice(N, size=iidsample_num, replace=False)
        mask = np.isin(np.arange(N), iid_lst)[:, np.newaxis]


        dict_users = {i: np.array([]) for i in range(self.num_user)}
        min_require_size = 30

        # iid
        min_size = 0
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(iiduser_num)]
            for label in range(1, n_classes+1):
                label_idx = np.where((self.y == label) & mask)[0]
                np.random.shuffle(label_idx)

                label_distribution = np.random.dirichlet([999999] * iiduser_num)  # (n_classes, self.num_user)
                label_distribution = np.array([p * (len(idx_j) < (iidsample_num / iiduser_num)) for p, idx_j in
                                               zip(label_distribution, idx_batch)])
                label_distribution = label_distribution / label_distribution.sum()
                label_distribution = (np.cumsum(label_distribution) * len(label_idx)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                             zip(idx_batch, np.split(label_idx, label_distribution))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(iiduser_num):
                np.random.shuffle(idx_batch[j])
                dict_users[j] = idx_batch[j]

        # noniid
        min_size = 0
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(noniiduser_num)]
            for label in range(1, n_classes+1):
                label_idx = np.where((self.y == label) & ~mask)[0]
                np.random.shuffle(label_idx)

                label_distribution = np.random.dirichlet([self.Dirichlet_alpha] * noniiduser_num)
                label_distribution = np.array([p * (len(idx_j) < ((N - iidsample_num) / noniiduser_num)) for p, idx_j in
                                               zip(label_distribution, idx_batch)])
                label_distribution = label_distribution / label_distribution.sum()
                label_distribution = (np.cumsum(label_distribution) * len(label_idx)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                             zip(idx_batch, np.split(label_idx, label_distribution))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(noniiduser_num):
                np.random.shuffle(idx_batch[j])
                dict_users[j + iiduser_num] = idx_batch[j]

        return dict_users


class Wiki(Dataset):
    def __init__(self, path, num_user, Dirichlet_alpha, w):

        data = scipy.io.loadmat(path + "Wiki_fea.mat")
        X = data['X']
        self.x1 = X[0,0].astype(np.float32)   # (2866, 128)
        self.x2 = X[1,0].astype(np.float32)
        self.y = np.copy(data['Y']).astype(np.int32).reshape(2866,1)
        self.num_user = num_user
        self.view = 2
        self.Dirichlet_alpha = Dirichlet_alpha
        self.w = w
        self.user_data = self.split_data()


    def __len__(self):
        return self.x1.shape[0]

    def split_data(self):
        N = int(len(self.y))
        n_classes = int(max(self.y))

        iiduser_num = int(self.num_user * self.w)
        noniiduser_num = self.num_user - iiduser_num
        iidsample_num = int(N * self.w)
        iid_lst = np.random.choice(N, size=iidsample_num, replace=False)
        mask = np.isin(np.arange(N), iid_lst)[:, np.newaxis]


        dict_users = {i: np.array([]) for i in range(self.num_user)}
        min_require_size = 30

        # iid
        min_size = 0
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(iiduser_num)]
            for label in range(1, n_classes+1):
                label_idx = np.where((self.y == label) & mask)[0]
                np.random.shuffle(label_idx)

                label_distribution = np.random.dirichlet([999999] * iiduser_num)
                label_distribution = np.array([p * (len(idx_j) < (iidsample_num / iiduser_num)) for p, idx_j in
                                               zip(label_distribution, idx_batch)])
                label_distribution = label_distribution / label_distribution.sum()
                label_distribution = (np.cumsum(label_distribution) * len(label_idx)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                             zip(idx_batch, np.split(label_idx, label_distribution))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(iiduser_num):
                np.random.shuffle(idx_batch[j])
                dict_users[j] = idx_batch[j]

        # noniid
        min_size = 0
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(noniiduser_num)]
            for label in range(1, n_classes+1):
                label_idx = np.where((self.y == label) & ~mask)[0]
                np.random.shuffle(label_idx)

                label_distribution = np.random.dirichlet([self.Dirichlet_alpha] * noniiduser_num)
                label_distribution = np.array([p * (len(idx_j) < ((N - iidsample_num) / noniiduser_num)) for p, idx_j in
                                               zip(label_distribution, idx_batch)])
                label_distribution = label_distribution / label_distribution.sum()
                label_distribution = (np.cumsum(label_distribution) * len(label_idx)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                             zip(idx_batch, np.split(label_idx, label_distribution))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(noniiduser_num):
                np.random.shuffle(idx_batch[j])
                dict_users[j + iiduser_num] = idx_batch[j]

        return dict_users



class Caltech_5(Dataset):
    def __init__(self, path,num_user, Dirichlet_alpha, w):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()

        self.x1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.x2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.x3 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.x4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.x5 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.y = data['Y'].astype(np.int32).transpose()

        self.num_user = num_user
        self.view = 5
        self.Dirichlet_alpha = Dirichlet_alpha
        self.w = w
        self.user_data = self.split_data()

    def __len__(self):
        return 1400

    def split_data(self):  # noniid
        N = int(len(self.y))
        n_classes = int(max(self.y) + 1)

        iiduser_num = int(self.num_user * self.w)
        noniiduser_num = self.num_user - iiduser_num
        iidsample_num = int(N * self.w)
        iid_lst = np.random.choice(N, size=iidsample_num, replace=False)
        mask = np.isin(np.arange(N), iid_lst)[:, np.newaxis]


        dict_users = {i: np.array([]) for i in range(self.num_user)}
        min_require_size = 10

        # iid
        min_size = 0
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(iiduser_num)]
            for label in range(n_classes):
                label_idx = np.where((self.y == label) & mask)[0]
                np.random.shuffle(label_idx)

                label_distribution = np.random.dirichlet(
                    [999999] * iiduser_num)  # (n_classes, self.num_user)

                label_distribution = np.array([p * (len(idx_j) < (iidsample_num / iiduser_num)) for p, idx_j in
                                               zip(label_distribution, idx_batch)])
                label_distribution = label_distribution / label_distribution.sum()
                label_distribution = (np.cumsum(label_distribution) * len(label_idx)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                             zip(idx_batch, np.split(label_idx, label_distribution))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(iiduser_num):
                np.random.shuffle(idx_batch[j])
                dict_users[j] = idx_batch[j]


        # noniid
        min_size = 0
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(noniiduser_num)]
            for label in range(n_classes):
                label_idx = np.where((self.y == label) & ~mask)[0]
                np.random.shuffle(label_idx)
                label_distribution = np.random.dirichlet([self.Dirichlet_alpha] * noniiduser_num)
                label_distribution = np.array([p * (len(idx_j) < ((N - iidsample_num) / noniiduser_num)) for p, idx_j in
                                               zip(label_distribution, idx_batch)])
                label_distribution = label_distribution / label_distribution.sum()
                label_distribution = (np.cumsum(label_distribution) * len(label_idx)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                             zip(idx_batch, np.split(label_idx, label_distribution))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(noniiduser_num):
                np.random.shuffle(idx_batch[j])
                dict_users[j + iiduser_num] = idx_batch[j]


        return dict_users

class STL10(Dataset):
    def __init__(self, path, num_user, Dirichlet_alpha, w):
        with h5py.File(path, 'r') as data:
            self.x1 = np.array(data[data['X'][0][0]]).astype(np.float32).transpose()
            self.x2 = np.array(data[data['X'][1][0]]).astype(np.float32).transpose()
            self.x3 = np.array(data[data['X'][2][0]]).astype(np.float32).transpose()

            self.y = np.array(data['Y']).astype(np.int32).reshape(13000, 1) - 1
        scaler = MinMaxScaler()
        self.x1 = scaler.fit_transform(self.x1)
        self.x2 = scaler.fit_transform(self.x2)
        self.x3 = scaler.fit_transform(self.x3)
        self.view = 3

        self.num_user = num_user
        self.Dirichlet_alpha = Dirichlet_alpha
        self.w = w
        self.user_data = self.split_data()

    def __len__(self):
        return 13000

    def split_data(self):  # partial noniid, partial iid
        N = int(len(self.y))
        n_classes = int(max(self.y))+1

        iiduser_num = int(self.num_user * self.w)
        noniiduser_num = self.num_user - iiduser_num
        iidsample_num = int(N * self.w)
        iid_lst = np.random.choice(N, size=iidsample_num, replace=False)
        mask = np.isin(np.arange(N), iid_lst)[:, np.newaxis]

        dict_users = {i: np.array([]) for i in range(self.num_user)}
        min_require_size = 30

        # iid
        min_size = 0
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(iiduser_num)]
            for label in range(n_classes):
                label_idx = np.where((self.y == label) & mask)[0]
                np.random.shuffle(label_idx)

                label_distribution = np.random.dirichlet([999999] * iiduser_num)
                label_distribution = np.array([p * (len(idx_j) < (iidsample_num / iiduser_num)) for p, idx_j in
                                               zip(label_distribution, idx_batch)])
                label_distribution = label_distribution / label_distribution.sum()
                label_distribution = (np.cumsum(label_distribution) * len(label_idx)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                             zip(idx_batch, np.split(label_idx, label_distribution))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(iiduser_num):
                np.random.shuffle(idx_batch[j])
                dict_users[j] = idx_batch[j]

        # noniid
        min_size = 0
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(noniiduser_num)]
            for label in range(n_classes):
                label_idx = np.where((self.y == label) & ~mask)[0]
                np.random.shuffle(label_idx)

                label_distribution = np.random.dirichlet([self.Dirichlet_alpha] * noniiduser_num)
                label_distribution = np.array([p * (len(idx_j) < ((N - iidsample_num) / noniiduser_num)) for p, idx_j in
                                               zip(label_distribution, idx_batch)])
                label_distribution = label_distribution / label_distribution.sum()
                label_distribution = (np.cumsum(label_distribution) * len(label_idx)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                             zip(idx_batch, np.split(label_idx, label_distribution))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(noniiduser_num):
                np.random.shuffle(idx_batch[j])
                dict_users[j + iiduser_num] = idx_batch[j]

        return dict_users

class CCV(Dataset):
    def __init__(self, path,  num_user, Dirichlet_alpha, w):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.x1 = scaler.fit_transform(self.data1).astype(np.float32)
        self.x2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.x3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.y = np.load(path+'label.npy').astype(np.int32).reshape(6773, 1)

        self.view = 3
        self.num_user = num_user
        self.Dirichlet_alpha = Dirichlet_alpha
        self.w = w
        self.user_data = self.split_data()

    def __len__(self):
        return 6773

    def split_data(self):
        N = int(len(self.y))
        n_classes = int(max(self.y))+1

        iiduser_num = int(self.num_user * self.w)
        noniiduser_num = self.num_user - iiduser_num
        iidsample_num = int(N * self.w)
        iid_lst = np.random.choice(N, size=iidsample_num, replace=False)
        mask = np.isin(np.arange(N), iid_lst)[:, np.newaxis]

        dict_users = {i: np.array([]) for i in range(self.num_user)}
        min_require_size = 30

        # iid
        min_size = 0
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(iiduser_num)]
            for label in range(n_classes):
                label_idx = np.where((self.y == label) & mask)[0]
                np.random.shuffle(label_idx)
                label_distribution = np.random.dirichlet(
                    [999999] * iiduser_num)  # (n_classes, self.num_user))
                label_distribution = np.array([p * (len(idx_j) < (iidsample_num / iiduser_num)) for p, idx_j in
                                               zip(label_distribution, idx_batch)])
                label_distribution = label_distribution / label_distribution.sum()
                label_distribution = (np.cumsum(label_distribution) * len(label_idx)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                             zip(idx_batch, np.split(label_idx, label_distribution))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(iiduser_num):
                np.random.shuffle(idx_batch[j])
                dict_users[j] = idx_batch[j]

        # noniid
        min_size = 0
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(noniiduser_num)]
            for label in range(n_classes):
                label_idx = np.where((self.y == label) & ~mask)[0]
                np.random.shuffle(label_idx)
                label_distribution = np.random.dirichlet([self.Dirichlet_alpha] * noniiduser_num)
                label_distribution = np.array([p * (len(idx_j) < ((N - iidsample_num) / noniiduser_num)) for p, idx_j in
                                               zip(label_distribution, idx_batch)])
                label_distribution = label_distribution / label_distribution.sum()
                label_distribution = (np.cumsum(label_distribution) * len(label_idx)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                             zip(idx_batch, np.split(label_idx, label_distribution))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(noniiduser_num):
                np.random.shuffle(idx_batch[j])
                dict_users[j + iiduser_num] = idx_batch[j]

        return dict_users





class DatasetSplit(Dataset):
  """An abstract Dataset class wrapped around Pytorch Dataset class."""

  def __init__(self, dataset_x, dataset_y, idxs, dim):
    self.dataset_x = dataset_x[idxs]
    self.dataset_y = dataset_y[idxs]
    self.idxs = [int(i) for i in idxs]
    self.dim = dim

  def __len__(self):
    return len(self.idxs)

  def __getitem__(self, item):
    image, label = self.dataset_x[item], self.dataset_y[item]
    image = image.reshape(self.dim)
    return torch.tensor(image), torch.tensor(label)


def load_data(dataset, num_user, Dirichlet_alpha, w):
    if dataset == 'BDGP':
        dataset = BDGP('./dataset/', num_user, Dirichlet_alpha, w)
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5


    elif dataset == "Cora":
        dataset = Cora('./dataset/', num_user, Dirichlet_alpha, w)
        dims = [2708, 1433, 2706, 2706]
        view = 4
        data_size = 2708
        class_num = 7

    elif dataset == "Wiki":
        dataset = Wiki('./dataset/', num_user, Dirichlet_alpha, w)
        dims = [128, 10]
        view = 2
        data_size = 2866
        class_num = 10

    elif dataset == "Caltech-5V":
        dataset = Caltech_5('dataset/Caltech-5V.mat', num_user, Dirichlet_alpha, w)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7

    elif dataset == "STL":
        dataset = STL10('dataset/stl10_fea.mat', num_user, Dirichlet_alpha, w)
        dims = [1024, 512, 2048]
        view = 3
        data_size = 13000
        class_num = 10

    elif dataset == "CCV":
        dataset = CCV('dataset/', num_user, Dirichlet_alpha, w)
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    else:
        raise NotImplementedError

    return dataset, dims, view, data_size, class_num