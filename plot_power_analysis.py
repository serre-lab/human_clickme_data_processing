import matplotlib.pyplot as plt
import numpy as np

# Data
num_subjects = np.arange(3, 19)  # Subjects from 3 to 18

num_images = 670
num_iterations = 100

avg_corr = np.array([0.3446527387373417, 0.42376321394987987, 0.46814070280541004,
                     0.5138981114929807, 0.5355055354217811, 0.569223531173753,
                     0.5882340141671173, 0.6106291343675455, 0.6243911735671871,
                     0.638232125042545, 0.6527289709621268, 0.6645768941992727,
                     0.6729530791509648, 0.6842618863083281, 0.6921077683813978,
                     0.7015506292531952])
std_corr = np.array([0.22229090870059834, 0.2066206333009592, 0.18665901165477053,
                     0.19020722771563622, 0.18112638602926623, 0.17707219871660118,
                     0.17016635573052932, 0.1662384228649881, 0.15818469594729245,
                     0.15897384429620418, 0.14528364483336836, 0.15379228319309524,
                     0.14226172003974355, 0.14104288329225684, 0.13698920293955896,
                     0.13227616633564498])

avg_null = np.array([0.05891959007048204, 0.13483131471250848, 0.11301162958454036,
                     0.15022625977733303, 0.1542826425914332, 0.17126830279071947,
                     0.17857080023135813, 0.19445101153566735, 0.1893968841876873,
                     0.24765132063008963, 0.22518285591865414, 0.2497178586057861,
                     0.23211821540852173, 0.22746495391235882, 0.23070554397378834,
                     0.2673953309652816])
std_null = np.array([0.1313799411854781, 0.18997030400745837, 0.18753222117487872,
                     0.20966552067544522, 0.21207781772478276, 0.23640374709999024,
                     0.2353965593665213, 0.251617041451966, 0.24624245271362286,
                     0.2501991264914405, 0.2524312943488681, 0.24649442120960496,
                     0.26284819924702385, 0.2722618604237423, 0.28323149505127815,
                     0.27335179012182464])

# Compute standard errors using the number of iterations
stderr_corr = std_corr / np.sqrt(num_iterations)
stderr_null = std_null / np.sqrt(num_iterations)

# Plotting
plt.figure(figsize=(12, 8))

# Plot lines with error bars
plt.errorbar(num_subjects, avg_corr, yerr=stderr_corr, label="Human vs. Human", fmt='-o', capsize=5, linewidth=3)
plt.errorbar(num_subjects, avg_null, yerr=stderr_null, label="Null Correlation", fmt='--o', capsize=5, linewidth=3)

# Horizontal line for ClickMe 1.0
plt.axhline(y=0.58, color='green', linestyle=':', linewidth=3, label='ClickMe 1.0')

# Labels and styling
plt.title("ClickMe 2.0 Consistency", fontsize=28)
plt.xlabel("Number of Subjects", fontsize=24)
plt.ylabel("Correlation (Spearman's R)", fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig("ClickMe_2.0_Consistency.png", dpi=300)
plt.close()