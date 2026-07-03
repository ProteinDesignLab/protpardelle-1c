import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns


metric = "ca_scaffold_scrmsd"
metric = "plddt"

# df_esmfold_fp = "/scratch/users/tianyulu/protpardelle-1c-sampling-results/old-07/07_rfdiffusion/cc58-epoch416-sampling_sidechain_conditional-ss1.2-schurn200-ccstart0.0-dx0.0-dy0.0-dz0.0-rewindNone/1prw/esm_metrics.csv"
df_esmfold_fp = "/scratch/users/tianyulu/protpardelle-1c-sampling-results/old-08-rfdiffusion-allatom/cc91-epoch383-sampling_sidechain_conditional_allatom_s1-ss1.2-schurn200-ccstart0.0-dx0.0-dy0.0-dz0.0-rewindNone/1prw/esm_metrics.csv"

# df_likelihood_fp = "/scratch/users/tianyulu/protpardelle-1c-sampling-results/likelihood_cc58_416_07/1prw/likelihood_result.csv"
# df_likelihood_fp = "/scratch/users/tianyulu/protpardelle-1c-sampling-results/likelihood_cc58_416_08/1prw/likelihood_result.csv"
# df_likelihood_fp = "/scratch/users/tianyulu/protpardelle-1c-sampling-results/likelihood_cc89_415/1prw/likelihood_result.csv"
df_likelihood_fp = "/scratch/users/tianyulu/protpardelle-1c-sampling-results/likelihood_cc91_383/1prw/likelihood_result.csv"

df_esmfold = pd.read_csv(df_esmfold_fp)
df_esmfold["pdb_stem"] = df_esmfold["motif_name"].astype(str) + "_" + df_esmfold["structure_index"].astype(str)

df_likelihood = pd.read_csv(df_likelihood_fp)

df = df_esmfold.merge(df_likelihood, on='pdb_stem', how='left')
df.rename(columns={"npa": "nats/atom"}, inplace=True)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

# Create the plot with marginal histograms
fig = plt.figure(figsize=(10, 8))

# Create a grid layout: 2x2 with different sizes
gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3], 
                      hspace=0.05, wspace=0.05)

# Main scatter plot (bottom left)
ax_main = fig.add_subplot(gs[1, 0])

# Create scatter plot
sns.scatterplot(data=df, x='nats/atom', y=metric, ax=ax_main, 
                alpha=0.6, s=8, color='steelblue')

# Add regression line
sns.regplot(data=df, x='nats/atom', y=metric, ax=ax_main, 
            scatter=False, color='green', line_kws={'linewidth': 2})

# Calculate and display correlation coefficient
correlation, _ = pearsonr(df['nats/atom'], df[metric])
ax_main.text(0.05, 0.95, f'pearson r = {correlation:.4f}', 
             transform=ax_main.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='white', alpha=0.8))

# Set labels
ax_main.set_xlabel('nats/atom', fontsize=12)
ax_main.set_ylabel(metric, fontsize=12)

# Top histogram (x-axis distribution)
ax_hist_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
ax_hist_top.hist(df['nats/atom'], bins=50, alpha=0.7, color='steelblue', 
                 edgecolor='black', linewidth=0.5)
ax_hist_top.set_ylabel('')
ax_hist_top.set_xlabel('')
ax_hist_top.tick_params(bottom=False, labelbottom=False)
ax_hist_top.tick_params(left=True, labelleft=False)

# Right histogram (y-axis distribution)
ax_hist_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
ax_hist_right.hist(df[metric], bins=50, orientation='horizontal', 
                   alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
ax_hist_right.set_ylabel('')
ax_hist_right.set_xlabel('')
ax_hist_right.tick_params(left=False, labelleft=False)
ax_hist_right.tick_params(bottom=True, labelbottom=False)

# Make the plot look cleaner
ax_main.grid(True, alpha=0.3)
ax_hist_top.grid(True, alpha=0.3)
ax_hist_right.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('likelihood.pdf')

