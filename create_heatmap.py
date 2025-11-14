#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set matplotlib parameters as requested
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 14})

# Read the CSV file
csv_file = '/data/zshao/ArcticInference/7b_instruct_sim.csv'
df = pd.read_csv(csv_file, index_col=0)

# Convert to numeric, handling any potential string values
df = df.apply(pd.to_numeric, errors='coerce')

# Create a mask for the lower triangle (including diagonal)
mask = np.tril(np.ones_like(df, dtype=bool))

# Create the heatmap
plt.figure(figsize=(5, 5))

# Create heatmap with specified min and max values
heatmap = sns.heatmap(df, 
                     vmin=1.4, 
                     vmax=4.5,
                     cmap='viridis',
                     cbar_kws={'label': 'Similarity Score'},
                     xticklabels=False,
                     yticklabels=False,
                     mask=mask)

# Put title at the bottom
plt.figtext(0.5, 0.02, 'Pairwise Similarity', ha='center', fontsize=16)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
output_file = '/data/zshao/ArcticInference/7b_instruct_sim_heatmap.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Heatmap saved to: {output_file}")

# Also save as PDF
output_pdf = '/data/zshao/ArcticInference/7b_instruct_sim_heatmap.pdf'
plt.savefig(output_pdf, bbox_inches='tight')
print(f"Heatmap also saved as PDF to: {output_pdf}")

plt.show()
