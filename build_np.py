import numpy as np
import os
import tifffile
import cv2

_MEAN_LIST = [
  403.9970442	,	399.3350785	,	391.4158155	,	383.5183992	,	376.1250729	,
  369.868478	,	365.3758411	,	363.3138552	,	364.2372746	,	366.0175499	,
  368.2019423	,	370.640761	,	373.364225	,	376.3314482	,	379.46682	,
  382.7787281	,	386.0693423	,	389.5330984	,	392.7384255	,	396.4825254	,
  400.8495805	,	406.0679206	,	412.1993745	,	419.4003077	,	428.3905154	,
  439.7776489	,	454.0011284	,	471.16563	,	490.854098	,	512.4456165	,
  535.2371301	,	558.5278612	,	581.5453206	,	604.1030753	,	625.1151172	,
  644.0042579	,	660.0362959	,	672.8390722	,	682.5975482	,	689.519247	,
  694.1334702	,	696.4386375	,	696.7048455	,	695.7582212	,	694.4333766	,
  693.4694162	,	693.3301343	,	693.9277654	,	695.1954662	,	697.0092552	,
  698.96597	,	700.8202691	,	702.7159582	,	704.9048301	,	707.2597617	,
  709.282556	,	711.0838134	,	713.1199997	,	715.0148905	,	716.676995	,
  717.6893232	,	717.621364	,	716.445471	,	714.2756982	,	711.3582217	,
  707.8357252	,	704.2276132	,	702.4383739	,	705.0290051	,	714.8557509	,
  734.7512489	,	766.995361	,	812.3448344	,	870.1665367	,	940.6643203	,
  1023.38692	,	1118.324137	,	1224.602922	,	1339.875438	,	1460.630323	,
  1583.767001	,	1707.168678	,	1830.606339	,	1949.635724	,	2051.78434	,
  2140.071198	,	2214.798062	,	2276.644035	,	2326.639123	,	2366.148864	,
  2398.092228	,	2425.007065	,	2448.817098	,	2471.112376	,	2493.401133	,
  2549.65154	,	2541.093776	,	2560.352107	,	2568.884553	,	2576.711484	,
  2584.591089	,	2592.459716	,	2599.28125	,	2605.746353	,	2612.244971	,
  2619.648884	,	2627.297983	,	2635.681017	,	2645.227505	,	2655.692061	,
  2666.592974	,	2676.220251	,	2683.540816	,	2689.676589	,	2694.069626	,
  2713.713336	,	2726.161704	,	2728.408069	,	2727.325474	,	2696.727797	,
  2687.949153	,	2675.383758	,	2658.022666	,	2696.293021	,	2672.464839	,
  2648.866765	,	2600.891042	,	2578.984203	,	2560.646188	,	2547.754811	,
  2540.035257	,	2536.129914	,	2533.711009	,	2532.27558	,	2532.860575	,
  2533.927945	,	2535.354189	,	2537.327642	,	2540.528554	,	2545.567213	,
  2551.424695	,	2556.395378	,	2562.780745	,	2569.538421	,	2576.535695	,
  2582.887166	,	2587.29295	,	2589.910844	,	2592.014918	,	2594.012795	
]

_STD_LIST = [
  316.2201778	,	312.71877	,	310.1482062	,	307.8590699	,	306.1111278	,
  305.2263211	,	305.4858257	,	307.0320001	,	309.9858041	,	313.9114911	,
  318.4856284	,	323.5302343	,	328.9137908	,	334.3836789	,	339.6600736	,
  344.7864838	,	349.7888833	,	354.8701747	,	360.084343	,	365.4120809	,
  370.8669248	,	376.3878439	,	381.9182299	,	387.2327506	,	392.4564662	,
  397.5759155	,	402.7435651	,	407.9509828	,	413.2772352	,	418.9840081	,
  425.2556881	,	432.2093166	,	439.7895365	,	447.9133491	,	456.4244887	,
  465.1713957	,	473.8960793	,	482.48064	,	491.0372406	,	499.5179851	,
  507.9891041	,	516.2488042	,	524.2780841	,	532.2939789	,	540.2944239	,
  548.2885331	,	556.1051447	,	563.5275653	,	570.6182323	,	577.5105164	,
  584.1985622	,	590.6908013	,	597.1059023	,	603.5548434	,	609.9741778	,
  616.3216235	,	622.767677	,	629.517239	,	636.3245934	,	643.1826549	,
  649.9753951	,	656.6318886	,	663.2623193	,	669.8967577	,	676.4956312	,
  682.9078079	,	689.1258801	,	695.3936516	,	701.6555993	,	707.4502423	,
  712.3283047	,	716.2017271	,	719.5603014	,	723.4713214	,	728.8909473	,
  737.37718	,	750.6982919	,	770.7706814	,	799.6553537	,	838.8547615	,
  889.5043926	,	951.2451148	,	1023.169321	,	1099.936859	,	1170.978531	,
  1236.934081	,	1293.903838	,	1340.691482	,	1377.531787	,	1404.685108	,
  1424.574355	,	1439.210316	,	1450.360379	,	1459.685241	,	1468.512646	,
  1949.679572	,	1487.883534	,	1493.55081	,	1492.566784	,	1491.643178	,
  1489.729824	,	1487.409358	,	1484.385908	,	1480.923864	,	1477.464611	,
  1474.647914	,	1472.102064	,	1470.53027	,	1470.327759	,	1471.126672	,
  1472.4569	,	1473.66216	,	1474.058952	,	1473.887938	,	1473.033914	,
  1740.584351	,	1873.210244	,	1871.34027	,	1867.512014	,	1453.920923	,
  1441.749406	,	1425.737631	,	1403.985164	,	2221.512903	,	2200.542553	,
  2177.966211	,	1812.165791	,	1781.651611	,	1752.599307	,	1727.675737	,
  1708.390801	,	1695.285165	,	1686.886252	,	1682.7294	,	1682.668614	,
  1685.777404	,	1691.638358	,	1699.50688	,	1709.30336	,	1720.291284	,
  1730.245961	,	1737.180942	,	1742.673309	,	1745.825506	,	1747.224694	,
  1747.147568	,	1745.551399	,	1743.864749	,	1742.891057	,	1743.230295	
]

ignore_label = 255

label_mapping = {-1: ignore_label, 0: 0,
                              5: 1, 10: 2,
                              15: 3, 20: 4,
                              25: 5, 30: 6,
                              35: 7, 40: 8,
                              45: 9, 50: 10,
                              55: 11, 80: 12,
                              90: 0, 95:0, 120:0}



data_path = "dataset"

MEAN_LIST = np.asarray(_MEAN_LIST)
STD_LIST = np.asarray(_STD_LIST)
MEAN_LIST = np.expand_dims(MEAN_LIST, axis=(1,2))
STD_LIST = np.expand_dims(STD_LIST, axis=(1,2))


for split in ["train", "test"]:
    print("=" * 50)
    print("Start to build numpy format data for", split)
    print("=" * 50)
    txt_file = f"{split}.txt"

    lines = open(os.path.join(data_path, txt_file), 'rt').readlines()
    filenames = []

    images = np.empty((0, 150,128,128), dtype=np.float32)
    labels = np.empty((0, 128,128), dtype=np.int64)

    for line in lines:
        filename = line.strip()
        img = tifffile.imread(os.path.join(data_path, "image", filename + ".tif"))
        img = np.asarray(img, dtype=np.float32)
        img = (img - MEAN_LIST) / STD_LIST
        img = np.expand_dims(img, axis=0)
        images = np.append(images, img, axis=0)
        
        target = cv2.imread(os.path.join(data_path, "label", filename + "_gt.tif"), cv2.IMREAD_GRAYSCALE)
        label = np.zeros(np.shape(target))
        for k, v in label_mapping.items():
            label[target == k] = v
        label = np.asarray(label, dtype=np.int64)
        print(filename)
        label = np.expand_dims(label, axis=0)
        labels = np.append(labels, label, axis=0)

        # print(images.shape)

    np.save(os.path.join(data_path, f'{split}_image.npy'), images)
    np.save(os.path.join(data_path, f'{split}_label.npy'), labels)

    print("=" * 50)
    print("Finish to build numpy format data for", split)
    print("=" * 50)