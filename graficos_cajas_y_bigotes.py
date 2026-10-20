import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

nombre_marca = " Marca 1"
numero_marca = 1
limite_inferior = 250
limite_superior = 450
df = pd.read_excel('BaseDeDatos.xlsx')
filtro_marca_1 = df[df['Marca'] == numero_marca]

sns.set_theme(style='whitegrid')
plt.figure(figsize=(10, 6))
plt.xlim(limite_inferior, limite_superior)
sns.boxplot(x=filtro_marca_1['ppm'], color="#1D8C48")

plt.title(f'Caja de bigotes ppm - {nombre_marca}', fontsize=14, fontweight='bold')
plt.xlabel('ppm', fontsize=10)

plt.savefig('boxplot_marca_1.png', dpi=300, bbox_inches='tight')

plt.show()