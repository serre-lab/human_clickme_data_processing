import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure directory exists
output_dir = "jay_work_in_progress/plots"
os.makedirs(output_dir, exist_ok=True)

# Data
x_values = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
y_values = np.array([0.2715987318424568, 0.3369444030615471, 0.416845319520975, 0.45309790316470655, 
                     0.49500952098055834, 0.5114085460417982, 0.5437883999919165, 0.5639440789615802, 
                     0.5806985988373958, 0.5874104296718563])
ci_lower = np.array([0.265947877364877, 0.3323308368405406, 0.4120066014810233, 0.44784035198519273, 
                     0.4885216727938215, 0.5028957322977581, 0.5320194421129983, 0.5477692380564296, 
                     0.5543491804426244, 0.5466770979048965])
ci_upper = np.array([0.2772495863200366, 0.3415579692825536, 0.42168403756092665, 0.45835545434422037, 
                     0.5014973691672951, 0.5199213597858383, 0.5555573578708347, 0.5801189198667307, 
                     0.6070480172321673, 0.6281437614388161])
num_images = np.array([9184, 9722, 8180, 5966, 3714, 2056, 990, 415, 186, 61])

null_values = np.array([0.06755357269450703, 0.10138498291367282, 0.14434855648009876, 0.1731407014442737, 
                        0.2028476987095444, 0.2200828400749844, 0.24057714646956446, 0.2485353260965924, 
                        0.2654561422127516, 0.2545488068316878])


clickme_1_human_corr = 0.58  # Value for ClickMe 1.0 human correlation

# Compute error bars (upper - mean)
y_err = np.array([y_values - ci_lower, ci_upper - y_values])

# Plot setup
fig, ax = plt.subplots(figsize=(10, 6))

# Plot line connecting core points (blue with CI)
ax.plot(x_values, y_values, marker='o', linestyle='-', color='royalblue', markersize=8, linewidth=2, label="Mean Human Correlation")
ax.errorbar(x_values, y_values, yerr=y_err, fmt='o', color='royalblue', capsize=5, elinewidth=2, markeredgewidth=2)

# Plot null correlation line (red, no CI)
ax.plot(x_values, null_values, marker='s', linestyle='--', color='red', markersize=8, linewidth=2, label="Null Correlation")

# Plot ClickMe 1.0 Human Correlation (green dotted line)
ax.axhline(y=clickme_1_human_corr, color='green', linestyle='dotted', linewidth=2, label="ClickMe 1.0 Human Correlation")

# Annotate each point with number of images
for i, txt in enumerate(num_images):
    ax.annotate(f"{txt}", (x_values[i], y_values[i]), textcoords="offset points", xytext=(0,10), 
                ha='center', fontsize=14, color='black', fontweight='bold')

# Formatting
ax.set_xlabel("Number of Maps per Image", fontsize=24, fontweight='bold', labelpad=15)
ax.set_ylabel("Correlation (Spearman)", fontsize=24, fontweight='bold', labelpad=15)
ax.set_title("ClickMe 2.0 Power Analysis", fontsize=30, fontweight='bold', pad=20)
ax.set_ylim(bottom=0)  # Set lower bound of y-axis to 0

# Grid and aesthetics
ax.grid(True, linestyle='--', alpha=0.6)
ax.tick_params(axis='both', which='major', labelsize=18)

# Add legend in bottom right with large font size
ax.legend(fontsize=20, loc="lower right")

# Save plot
png_path = os.path.join(output_dir, "clickme_power_analysis.png")
plt.tight_layout()
plt.savefig(png_path, dpi=300)
plt.show()

print(f"Saved PNG to {png_path}")
