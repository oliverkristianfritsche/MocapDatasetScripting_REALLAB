import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare a light Arial font style for consistent use
plt.rcParams.update({'font.family': 'Arial', 'font.weight': 'light'})

data = {
    'ID': ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12', 'P13'],
    'Gender': ['Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Female'],
    'Age': [22, 22, 22, 20, 31, 30, 24, 28, 30, 28, 19, 20, 21],
    'Height': [170, 175, 175, 173, 165, 160, 175, 175, 168, 173, 157, 156, 160],
    'Mass': [59, 74, 57, 77, 63, 54.5, 62, 90, 75, 76, 54, 50, 49]
}

df = pd.DataFrame(data)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

fs = 22

# Age distribution plot
sns.kdeplot(df['Age'], fill=True, ax=axes[0], color="slateblue")
# axes[0].set_title("Age Distribution", fontsize=20, fontweight='normal')
# axes[0].set_xlabel("Age (years)", fontsize=18, fontweight='normal')
# axes[0].set_ylabel("Density", fontsize=18, fontweight='normal')
axes[0].tick_params(axis='both', labelsize=fs, width=0.8, color='gray')
axes[0].set_title("(a)", fontsize=fs-2, fontweight='bold')

axes[0].text(0.95, 0.8, f'Mean: {df["Age"].mean():.2f}\nStd Dev: {df["Age"].std():.2f}', 
             transform=axes[0].transAxes, fontsize=fs, ha='right')

print(f'Mean: {df["Age"].mean():.2f}\nMedian: {df["Age"].median():.2f}\nStd Dev: {df["Age"].std():.2f} for age')

# Height distribution plot
sns.kdeplot(df['Height'], fill=True, ax=axes[1], color="skyblue")
# axes[1].set_title("Height Distribution", fontsize=20, fontweight='normal')
# axes[1].set_xlabel("Height (cm)", fontsize=18, fontweight='normal')
# axes[1].set_ylabel("Density", fontsize=18, fontweight='normal')
axes[1].tick_params(axis='both', labelsize=fs, width=0.8, color='gray')
axes[1].set_title("(b)", fontsize=fs, fontweight='bold')
axes[1].text(0.05, 0.8, f'Mean: {df["Height"].mean():.2f}\nStd Dev: {df["Height"].std():.2f}', 
             transform=axes[1].transAxes, fontsize=fs-2, ha='left')
# axes[1].legend(loc='upper right', fontsize=12, frameon=False)
print(f'Mean: {df["Height"].mean():.2f}\nMedian: {df["Height"].median():.2f}\nStd Dev: {df["Height"].std():.2f} for height')

# Mass distribution plot
sns.kdeplot(df['Mass'], fill=True, ax=axes[2], color="mediumseagreen")
# axes[2].set_title("Mass Distribution", fontsize=20, fontweight='normal')
# axes[2].set_xlabel("Mass (kg)", fontsize=18, fontweight='normal')
# axes[2].set_ylabel("Density", fontsize=18, fontweight='normal')
axes[2].tick_params(axis='both', labelsize=fs, width=0.8, color='gray')
axes[2].set_title("(c)", fontsize=fs, fontweight='bold')
axes[2].text(0.95, 0.8, f'Mean: {df["Mass"].mean():.2f}\nStd Dev: {df["Mass"].std():.2f}', 
             transform=axes[2].transAxes, fontsize=fs-2, ha='right')
print(f'Mean: {df["Mass"].mean():.2f}\nMedian: {df["Mass"].median():.2f}\nStd Dev: {df["Mass"].std():.2f} for mass')

#disable y ticks

for ax in axes:
    # ax.set_yticklabels([])
    # ax.set_yticks([])

    #remove right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #remove y label
    ax.set_ylabel(None)
    ax.set_xlabel(None)

plt.tight_layout()
plt.show()
