import pandas as pd
import math

df = pd.read_excel('BaseDeDatos.xlsx')

print(df['ppm'].describe())

mediana = df['ppm'].median()
print(f"Mediana: {mediana}")

q1 = df['ppm'].quantile(0.25)   # Primer cuartil (25%)
q2 = df['ppm'].quantile(0.50)  # Segundo cuartil (25%)
q3 = df['ppm'].quantile(0.75)  # Tercer cuartil (75%)
print(f"Q1 (25%): {q1}")
print(f"Q2 (50%): {q2}")
print(f"Q3 (75%): {q3}")
print("\n")
print("===========================================================================")
print("\n")
filtro_marca_1 = df[df['Marca'] == 1]
filtro_marca_2 = df[df['Marca'] == 2]
filtro_marca_3 = df[df['Marca'] == 3]
filtro_marca_4 = df[df['Marca'] == 4]
filtro_marca_5 = df[df['Marca'] == 5]

promedio_ppm_marca_1 = filtro_marca_1['ppm'].mean()
promedio_ppm_marca_2 = filtro_marca_2['ppm'].mean()
promedio_ppm_marca_3 = filtro_marca_3['ppm'].mean()
promedio_ppm_marca_4 = filtro_marca_4['ppm'].mean()
promedio_ppm_marca_5 = filtro_marca_5['ppm'].mean()

print(f"El promedio de ppm para la Marca 1 es: {promedio_ppm_marca_1}")
print(f"El promedio de ppm para la Marca 2 es: {promedio_ppm_marca_2}")
print(f"El promedio de ppm para la Marca 3 es: {promedio_ppm_marca_3}")
print(f"El promedio de ppm para la Marca 4 es: {promedio_ppm_marca_4}")
print(f"El promedio de ppm para la Marca 5 es: {promedio_ppm_marca_5}")
print("\n")
print("===========================================================================")
print("\n")

inf, sup = 2.7 - 0.22, 2.7 + 0.22 #PROMEDIO DE POTENCIAS

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
print("\n")
print("===========================================================================")
print("\n")
media_potencia_m5 = filtro_marca_5['Potencia'].mean()
varianza_potencia_m5 = filtro_marca_5['Potencia'].var()
desviacion_potencia_m5 = math.sqrt(varianza_potencia_m5)

#Calcular la proporción de hornos que cumplen la norma de emisión (ppm <= 400)
cumplen_emision_m5 = (filtro_marca_5['ppm'] <= 400).mean() * 100
total_marca_5 = len(filtro_marca_5)
cantidad_cumplen = (filtro_marca_5['ppm'] <= 400).sum()

print('--- RESULTADOS PARA LA MARCA 5 (Punto d) ---')
print(f'1. Media (promedio) de la potencia: {media_potencia_m5:.4f} kW')
print(f'2. Varianza de la potencia: {varianza_potencia_m5:.4f}')
print(f'3. Desviacion de la potencia: {desviacion_potencia_m5:.4f}')

print(
    f'4. Proporción que cumple emisión (<= 400 ppm): {cumplen_emision_m5:.2f}% '
    f'({cantidad_cumplen} de {total_marca_5} hornos)'
)
print("\n")
print("===========================================================================")
print("\n")
varianza_potencia_m3 = filtro_marca_3['Potencia'].var()
varianza_potencia_m4 = filtro_marca_4['Potencia'].var()
print(f"Varianza de la potencia 3: {varianza_potencia_m3}")
print(f"Varianza de la potencia 4: {varianza_potencia_m4}")