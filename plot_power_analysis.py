import numpy as np
import matplotlib.pyplot as plt

# Data
human_corr = [0.32287320282227366, 0.42474398869008084, 0.4470524865544557, 0.4912216421661969,
              0.5130588753643014, 0.5375884544763164, 0.5609434694819505, 0.5778145601116464]
human_stdev = [0.2410261091807557, 0.22913161705698046, 0.21246802554817398, 0.2103430339098439,
               0.19880155217647275, 0.19758463946881646, 0.18806661784688627, 0.18373515624847167]

null_corr = [0.10790180891789856, 0.1908512449716566, 0.16260753364102393, 0.19142352398883766,
             0.20626915819144864, 0.24559848937569162, 0.24948005141216936, 0.25209586036145976]
null_stdev = [0.1297419943061997, 0.1696543487094471, 0.15213209707064185, 0.17875360131773407,
              0.17009567214314936, 0.19445878729582114, 0.1571407354183409, 0.18483782066809282]

# X-axis: Subjects from 3 to 10.
x = np.arange(3, 11)

# Compute error bars: standard deviation divided by sqrt(186)
n = 186
err_human = np.array(human_stdev) / np.sqrt(n)
err_null = np.array(null_stdev) / np.sqrt(n)

plt.figure(figsize=(12, 8))
plt.errorbar(x, human_corr, yerr=err_human, fmt='-o', markersize=10, linewidth=3, label='Human Correlation')
plt.errorbar(x, null_corr, yerr=err_null, fmt='-s', markersize=10, linewidth=3, label='Null Correlation')
plt.xlabel("Number of Subjects Per Image", fontsize=22, fontweight='bold')
plt.ylabel("Alignment (Spearman)", fontsize=22, fontweight='bold')
plt.title("ClickMe 2.0 Power Analysis", fontsize=28, fontweight='bold')
plt.xticks(x, fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.show()
