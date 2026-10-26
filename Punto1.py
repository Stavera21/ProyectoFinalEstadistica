import math
import numpy as np
import pandas as pd
from scipy import stats  # Importante para los intervalos de confianza

print('===========================================================================')

print('\n---------- DATOS BASE ----------')
df = pd.read_excel('BaseDeDatos.xlsx')
filtro_marca_1 = df[df['Marca'] == 1]
filtro_marca_2 = df[df['Marca'] == 2]
filtro_marca_3 = df[df['Marca'] == 3]
filtro_marca_4 = df[df['Marca'] == 4]
filtro_marca_5 = df[df['Marca'] == 5]
print(df['ppm'].describe())

mediana = df['ppm'].median()
print(f'Mediana: {mediana}')

print('\n---------- ENUNCIADO a) ----------')
print('Análisis para el diagrama de cajas (ppm) y menor emisión de contaminantes')

q1 = df['ppm'].quantile(0.25)  # Primer cuartil (25%)
q2 = df['ppm'].quantile(0.50)  # Segundo cuartil (50%)
q3 = df['ppm'].quantile(0.75)  # Tercer cuartil (75%)
print(f'Q1 (25%): {q1}')
print(f'Q2 (50%): {q2}')
print(f'Q3 (75%): {q3}')

q1_ppm_marca_1 = filtro_marca_1['ppm'].quantile(0.25)
q2_ppm_marca_1 = filtro_marca_1['ppm'].quantile(0.50)
q3_ppm_marca_1 = filtro_marca_1['ppm'].quantile(0.75)

q1_ppm_marca_2 = filtro_marca_2['ppm'].quantile(0.25)
q2_ppm_marca_2 = filtro_marca_2['ppm'].quantile(0.50)
q3_ppm_marca_2 = filtro_marca_2['ppm'].quantile(0.75)

q1_ppm_marca_3 = filtro_marca_3['ppm'].quantile(0.25)
q2_ppm_marca_3 = filtro_marca_3['ppm'].quantile(0.50)
q3_ppm_marca_3 = filtro_marca_3['ppm'].quantile(0.75)

q1_ppm_marca_4 = filtro_marca_4['ppm'].quantile(0.25)
q2_ppm_marca_4 = filtro_marca_4['ppm'].quantile(0.50)
q3_ppm_marca_4 = filtro_marca_4['ppm'].quantile(0.75)

q1_ppm_marca_5 = filtro_marca_5['ppm'].quantile(0.25)
q2_ppm_marca_5 = filtro_marca_5['ppm'].quantile(0.50)
q3_ppm_marca_5 = filtro_marca_5['ppm'].quantile(0.75)

print(f'Los cuartiles de ppm para la Marca 1 son: Q1 (25%): {q1_ppm_marca_1}, Q2 (50%): {q2_ppm_marca_1}, Q3 (75%): {q3_ppm_marca_1}')
print(f'Los cuartiles de ppm para la Marca 2 son: Q1 (25%): {q1_ppm_marca_2}, Q2 (50%): {q2_ppm_marca_2}, Q3 (75%): {q3_ppm_marca_2}')
print(f'Los cuartiles de ppm para la Marca 3 son: Q1 (25%): {q1_ppm_marca_3}, Q2 (50%): {q2_ppm_marca_3}, Q3 (75%): {q3_ppm_marca_3}')
print(f'Los cuartiles de ppm para la Marca 4 son: Q1 (25%): {q1_ppm_marca_4}, Q2 (50%): {q2_ppm_marca_4}, Q3 (75%): {q3_ppm_marca_4}')
print(f'Los cuartiles de ppm para la Marca 5 son: Q1 (25%): {q1_ppm_marca_5}, Q2 (50%): {q2_ppm_marca_5}, Q3 (75%): {q3_ppm_marca_5}')

promedio_ppm_marca_1 = filtro_marca_1['ppm'].mean()
promedio_ppm_marca_2 = filtro_marca_2['ppm'].mean()
promedio_ppm_marca_3 = filtro_marca_3['ppm'].mean()
promedio_ppm_marca_4 = filtro_marca_4['ppm'].mean()
promedio_ppm_marca_5 = filtro_marca_5['ppm'].mean()

print(f'El promedio de ppm para la Marca 1 es: {promedio_ppm_marca_1}')
print(f'El promedio de ppm para la Marca 2 es: {promedio_ppm_marca_2}')
print(f'El promedio de ppm para la Marca 3 es: {promedio_ppm_marca_3}')
print(f'El promedio de ppm para la Marca 4 es: {promedio_ppm_marca_4}')
print(f'El promedio de ppm para la Marca 5 es: {promedio_ppm_marca_5}')
print('\n')
print('===========================================================================')

print('\n---------- ENUNCIADOS b) y c) ----------')
print('b) Marca que mejor se ajusta a la potencia (2,7 ± 0,22)')
print('c) Definición de la marca con mejor desempeño general')

inf, sup = 2.7 - 0.22, 2.7 + 0.22  # Límites de potencia

resumen = (
    df.groupby('Marca')['Potencia']
    .agg(
        Total_Hornos='count',
        Cumplen=lambda x: ((x >= inf) & (x <= sup)).sum(),
        Media_Potencia='mean',
        Desv_Potencia='std',
    )
    .reset_index()
)

# Calcular el porcentaje de cumplimiento
resumen['Porcentaje_Cumplimiento'] = (
    resumen['Cumplen'] / resumen['Total_Hornos']
) * 100

print(resumen.round(2).to_string(index=False))
print('\n')
print('===========================================================================')

print('\n---------- ENUNCIADO d) ----------')
print('Para la mejor marca (Marca 5): Intervalos de confianza para media, varianza y proporción')

m5 = df[df['Marca'] == 5]
x = m5['Potencia']
n = len(x)
alpha = 0.05

#Media (t de Student)
t = stats.t.ppf(1 - alpha/2, n - 1)
ee = x.std() / math.sqrt(n)
print(f'IC media: ({x.mean() - t*ee:.4f}, {x.mean() + t*ee:.4f})')

#Varianza (chi-cuadrado)
li = (n-1) * x.var() / stats.chi2.ppf(1 - alpha/2, n - 1)
ls = (n-1) * x.var() / stats.chi2.ppf(alpha/2, n - 1)
print(f'IC varianza: ({li:.5f}, {ls:.5f})')

print('\n')
print('===========================================================================')

print('\n---------- ENUNCIADO e) ----------')
print('Diferencia de variabilidad entre marcas 3 y 4 (Prueba F)')

varianza_potencia_m3 = filtro_marca_3['Potencia'].var()
desviacion_potencia_m3 = math.sqrt(varianza_potencia_m3)
varianza_potencia_m4 = filtro_marca_4['Potencia'].var()
desviacion_potencia_m4 = math.sqrt(varianza_potencia_m4)
print(f"Varianza de la potencia 3: {varianza_potencia_m3}")
print(f"Desviacion estandar de la potencia 3: {desviacion_potencia_m3}")
print(f"Varianza de la potencia 4: {varianza_potencia_m4}")
print(f"Desviacion estandar de la potencia 4: {desviacion_potencia_m4}")
print("Ho: σ_3^2 = σ_4^2")
print("Ha: σ_3^2 ≠ σ_4^2")


n3 = filtro_marca_3['Potencia'].count()
n4 = filtro_marca_4['Potencia'].count()
alpha = 0.05

F = varianza_potencia_m3 / varianza_potencia_m4
f_inf = stats.f.ppf(alpha/2, n3 - 1, n4 - 1)
f_sup = stats.f.ppf(1 - alpha/2, n3 - 1, n4 - 1)
p_valor = 2 * min(stats.f.cdf(F, n3 - 1, n4 - 1),
                1 - stats.f.cdf(F, n3 - 1, n4 - 1)) # p = 2 · mín{ P(F ≤ F_calc) , P(F ≥ F_calc) }

print(f'F = {F:.4f}')
print(f'Región de no rechazo: ({f_inf:.4f}, {f_sup:.4f})')
print(f'Valor p = {p_valor:.4e}')
print('Se rechaza H0' if (F < f_inf or F > f_sup) else 'No se rechaza H0')

# IC para el cociente de varianzas
print(f'IC cociente: ({F/f_sup:.4f}, {F/f_inf:.4f})')

print('\n')
print('===========================================================================')

print('\n---------- ENUNCIADO f) ----------')
print('Prueba de afirmación del fabricante (80 de cada 100 hornos cumplen < 400 ppm)')

#Proporción de cumplimiento de emisión (<= 400 ppm)
k = (m5['ppm'] <= 400).sum()
print(f"Hornos que cumplen: {k}")
ic = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method='exact')
print(f'p̂ = {k/n:.4f}, IC exacto: ({ic.low:.4f}, {ic.high:.4f})')
print('\n')
print('===========================================================================')
print('\n')
