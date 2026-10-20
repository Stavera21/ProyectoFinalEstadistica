import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('BaseDedatos.xlsx')
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10,6))
sns.boxplot(x=df['ppm'], color='#7CE670')
plt.xticks([0, 100, 200, 300, 400, 500, 600, 700,800, 900])
plt.title("Caja de bigotes ppm", fontsize = 14, fontweight ='bold')
plt.ylabel("ppm", fontsize = 10)
plt.show()
plt.savefig('boxplot_excel.png', dpi=300, bbox_inches='tight')
