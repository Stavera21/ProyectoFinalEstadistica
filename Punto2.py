import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats  # Importante para las distribuciones y las pruebas de hipótesis

print('===========================================================================')

print('\n---------- DATOS BASE ----------')
np.random.seed(42)  # Semilla: con ella todos los integrantes obtenemos los mismos datos aleatorios
tamano_muestra = 60
alpha = 0.05

# Parámetros del enunciado - Escenario 1: Normal (media y desviación estándar)
media_teorica_equipo_A_escenario_1 = 175
desviacion_teorica_equipo_A_escenario_1 = 30
media_teorica_equipo_B_escenario_1 = 170
desviacion_teorica_equipo_B_escenario_1 = 50

# Parámetros del enunciado - Escenario 2: Gamma (alfa y beta)
alfa_teorico_equipo_A_escenario_2 = 1.2
beta_teorico_equipo_A_escenario_2 = 150
alfa_teorico_equipo_B_escenario_2 = 0.95
beta_teorico_equipo_B_escenario_2 = 190

# Distribuciones teóricas (en scipy el parámetro scale de la Gamma es beta)
distribucion_teorica_equipo_A_escenario_1 = stats.norm(loc=media_teorica_equipo_A_escenario_1, scale=desviacion_teorica_equipo_A_escenario_1)
distribucion_teorica_equipo_B_escenario_1 = stats.norm(loc=media_teorica_equipo_B_escenario_1, scale=desviacion_teorica_equipo_B_escenario_1)
distribucion_teorica_equipo_A_escenario_2 = stats.gamma(a=alfa_teorico_equipo_A_escenario_2, scale=beta_teorico_equipo_A_escenario_2)
distribucion_teorica_equipo_B_escenario_2 = stats.gamma(a=alfa_teorico_equipo_B_escenario_2, scale=beta_teorico_equipo_B_escenario_2)

# Varianzas teóricas: Normal -> sigma^2 ; Gamma -> alfa * beta^2
varianza_teorica_equipo_A_escenario_1 = desviacion_teorica_equipo_A_escenario_1 ** 2
varianza_teorica_equipo_B_escenario_1 = desviacion_teorica_equipo_B_escenario_1 ** 2
varianza_teorica_equipo_A_escenario_2 = alfa_teorico_equipo_A_escenario_2 * beta_teorico_equipo_A_escenario_2 ** 2
varianza_teorica_equipo_B_escenario_2 = alfa_teorico_equipo_B_escenario_2 * beta_teorico_equipo_B_escenario_2 ** 2

# Generación de los 60 tiempos de procesamiento (ms) de cada grupo
tiempos_equipo_A_escenario_1 = distribucion_teorica_equipo_A_escenario_1.rvs(size=tamano_muestra)
tiempos_equipo_B_escenario_1 = distribucion_teorica_equipo_B_escenario_1.rvs(size=tamano_muestra)
tiempos_equipo_A_escenario_2 = distribucion_teorica_equipo_A_escenario_2.rvs(size=tamano_muestra)
tiempos_equipo_B_escenario_2 = distribucion_teorica_equipo_B_escenario_2.rvs(size=tamano_muestra)

datos_escenario_1 = pd.DataFrame({'Equipo_A': tiempos_equipo_A_escenario_1, 'Equipo_B': tiempos_equipo_B_escenario_1})
datos_escenario_2 = pd.DataFrame({'Equipo_A': tiempos_equipo_A_escenario_2, 'Equipo_B': tiempos_equipo_B_escenario_2})

print(f'Se generaron {tamano_muestra} datos por equipo y por escenario (semilla 42)')
print('\n')
print('===========================================================================')

print('\n---------- ENUNCIADO a) ----------')
print('Medidas descriptivas, pruebas de bondad de ajuste e histogramas')

print('\nResumen del Escenario 1 (Normal):')
print(datos_escenario_1.describe().round(2))
print('\nResumen del Escenario 2 (Gamma):')
print(datos_escenario_2.describe().round(2))

# Medias y varianzas muestrales (se usan más adelante en los literales b) y c))
media_muestral_equipo_A_escenario_1 = np.mean(tiempos_equipo_A_escenario_1)
media_muestral_equipo_B_escenario_1 = np.mean(tiempos_equipo_B_escenario_1)
media_muestral_equipo_A_escenario_2 = np.mean(tiempos_equipo_A_escenario_2)
media_muestral_equipo_B_escenario_2 = np.mean(tiempos_equipo_B_escenario_2)

varianza_muestral_equipo_A_escenario_1 = np.var(tiempos_equipo_A_escenario_1, ddof=1)
varianza_muestral_equipo_B_escenario_1 = np.var(tiempos_equipo_B_escenario_1, ddof=1)
varianza_muestral_equipo_A_escenario_2 = np.var(tiempos_equipo_A_escenario_2, ddof=1)
varianza_muestral_equipo_B_escenario_2 = np.var(tiempos_equipo_B_escenario_2, ddof=1)

desviacion_muestral_equipo_A_escenario_1 = math.sqrt(varianza_muestral_equipo_A_escenario_1)
desviacion_muestral_equipo_B_escenario_1 = math.sqrt(varianza_muestral_equipo_B_escenario_1)
desviacion_muestral_equipo_A_escenario_2 = math.sqrt(varianza_muestral_equipo_A_escenario_2)
desviacion_muestral_equipo_B_escenario_2 = math.sqrt(varianza_muestral_equipo_B_escenario_2)

# Tabla con las medidas descriptivas de los cuatro grupos
nombres_grupos = ['Esc1 - Equipo A (Normal)', 'Esc1 - Equipo B (Normal)', 'Esc2 - Equipo A (Gamma)', 'Esc2 - Equipo B (Gamma)']
tabla_descriptiva = pd.DataFrame({
    'Media': [media_muestral_equipo_A_escenario_1, media_muestral_equipo_B_escenario_1,
              media_muestral_equipo_A_escenario_2, media_muestral_equipo_B_escenario_2],
    'Mediana': [np.median(tiempos_equipo_A_escenario_1), np.median(tiempos_equipo_B_escenario_1),
                np.median(tiempos_equipo_A_escenario_2), np.median(tiempos_equipo_B_escenario_2)],
    'Desv_Estandar': [desviacion_muestral_equipo_A_escenario_1, desviacion_muestral_equipo_B_escenario_1,
                      desviacion_muestral_equipo_A_escenario_2, desviacion_muestral_equipo_B_escenario_2],
    'Varianza': [varianza_muestral_equipo_A_escenario_1, varianza_muestral_equipo_B_escenario_1,
                 varianza_muestral_equipo_A_escenario_2, varianza_muestral_equipo_B_escenario_2],
    'CV_porcentaje': [desviacion_muestral_equipo_A_escenario_1 / media_muestral_equipo_A_escenario_1 * 100,
                      desviacion_muestral_equipo_B_escenario_1 / media_muestral_equipo_B_escenario_1 * 100,
                      desviacion_muestral_equipo_A_escenario_2 / media_muestral_equipo_A_escenario_2 * 100,
                      desviacion_muestral_equipo_B_escenario_2 / media_muestral_equipo_B_escenario_2 * 100],
    'Asimetria': [stats.skew(tiempos_equipo_A_escenario_1), stats.skew(tiempos_equipo_B_escenario_1),
                  stats.skew(tiempos_equipo_A_escenario_2), stats.skew(tiempos_equipo_B_escenario_2)],
    'Curtosis': [stats.kurtosis(tiempos_equipo_A_escenario_1), stats.kurtosis(tiempos_equipo_B_escenario_1),
                 stats.kurtosis(tiempos_equipo_A_escenario_2), stats.kurtosis(tiempos_equipo_B_escenario_2)],
}, index=nombres_grupos)
print('\nMedidas descriptivas de los cuatro grupos:')
print(tabla_descriptiva.round(2).to_string())

print('\nBondad de ajuste')
print('Ho: los datos provienen de la distribución indicada')
print('Ha: los datos NO provienen de la distribución indicada')
print(f'Se rechaza Ho si el valor p < {alpha}')

# Kolmogorov-Smirnov contra la distribución teórica del enunciado
estadistico_ks_equipo_A_escenario_1, valor_p_ks_equipo_A_escenario_1 = stats.kstest(tiempos_equipo_A_escenario_1, distribucion_teorica_equipo_A_escenario_1.cdf)
estadistico_ks_equipo_B_escenario_1, valor_p_ks_equipo_B_escenario_1 = stats.kstest(tiempos_equipo_B_escenario_1, distribucion_teorica_equipo_B_escenario_1.cdf)
estadistico_ks_equipo_A_escenario_2, valor_p_ks_equipo_A_escenario_2 = stats.kstest(tiempos_equipo_A_escenario_2, distribucion_teorica_equipo_A_escenario_2.cdf)
estadistico_ks_equipo_B_escenario_2, valor_p_ks_equipo_B_escenario_2 = stats.kstest(tiempos_equipo_B_escenario_2, distribucion_teorica_equipo_B_escenario_2.cdf)

print('\nKolmogorov-Smirnov (comparación con la distribución teórica):')
print(f'Esc1 - Equipo A (Normal): D = {estadistico_ks_equipo_A_escenario_1:.4f}, valor p = {valor_p_ks_equipo_A_escenario_1:.4f} ->', 'Se rechaza Ho' if valor_p_ks_equipo_A_escenario_1 < alpha else 'No se rechaza Ho')
print(f'Esc1 - Equipo B (Normal): D = {estadistico_ks_equipo_B_escenario_1:.4f}, valor p = {valor_p_ks_equipo_B_escenario_1:.4f} ->', 'Se rechaza Ho' if valor_p_ks_equipo_B_escenario_1 < alpha else 'No se rechaza Ho')
print(f'Esc2 - Equipo A (Gamma):  D = {estadistico_ks_equipo_A_escenario_2:.4f}, valor p = {valor_p_ks_equipo_A_escenario_2:.4f} ->', 'Se rechaza Ho' if valor_p_ks_equipo_A_escenario_2 < alpha else 'No se rechaza Ho')
print(f'Esc2 - Equipo B (Gamma):  D = {estadistico_ks_equipo_B_escenario_2:.4f}, valor p = {valor_p_ks_equipo_B_escenario_2:.4f} ->', 'Se rechaza Ho' if valor_p_ks_equipo_B_escenario_2 < alpha else 'No se rechaza Ho')

# Shapiro-Wilk: solo evalúa si los datos son normales
estadistico_shapiro_equipo_A_escenario_1, valor_p_shapiro_equipo_A_escenario_1 = stats.shapiro(tiempos_equipo_A_escenario_1)
estadistico_shapiro_equipo_B_escenario_1, valor_p_shapiro_equipo_B_escenario_1 = stats.shapiro(tiempos_equipo_B_escenario_1)
estadistico_shapiro_equipo_A_escenario_2, valor_p_shapiro_equipo_A_escenario_2 = stats.shapiro(tiempos_equipo_A_escenario_2)
estadistico_shapiro_equipo_B_escenario_2, valor_p_shapiro_equipo_B_escenario_2 = stats.shapiro(tiempos_equipo_B_escenario_2)

print('\nShapiro-Wilk (Ho: los datos son normales):')
print(f'Esc1 - Equipo A: W = {estadistico_shapiro_equipo_A_escenario_1:.4f}, valor p = {valor_p_shapiro_equipo_A_escenario_1:.4f} ->', 'Se rechaza Ho' if valor_p_shapiro_equipo_A_escenario_1 < alpha else 'No se rechaza Ho')
print(f'Esc1 - Equipo B: W = {estadistico_shapiro_equipo_B_escenario_1:.4f}, valor p = {valor_p_shapiro_equipo_B_escenario_1:.4f} ->', 'Se rechaza Ho' if valor_p_shapiro_equipo_B_escenario_1 < alpha else 'No se rechaza Ho')
print(f'Esc2 - Equipo A: W = {estadistico_shapiro_equipo_A_escenario_2:.4f}, valor p = {valor_p_shapiro_equipo_A_escenario_2:.4g} ->', 'Se rechaza Ho' if valor_p_shapiro_equipo_A_escenario_2 < alpha else 'No se rechaza Ho')
print(f'Esc2 - Equipo B: W = {estadistico_shapiro_equipo_B_escenario_2:.4f}, valor p = {valor_p_shapiro_equipo_B_escenario_2:.4g} ->', 'Se rechaza Ho' if valor_p_shapiro_equipo_B_escenario_2 < alpha else 'No se rechaza Ho')

# Histogramas con la densidad teórica encima
figura, ejes = plt.subplots(2, 2, figsize=(12, 8))
figura.suptitle('Histogramas del tiempo de procesamiento (ms) con la densidad teórica')

valores_x_equipo_A_escenario_1 = np.linspace(tiempos_equipo_A_escenario_1.min() * 0.9, tiempos_equipo_A_escenario_1.max() * 1.05, 300)
valores_x_equipo_B_escenario_1 = np.linspace(tiempos_equipo_B_escenario_1.min() * 0.9, tiempos_equipo_B_escenario_1.max() * 1.05, 300)
valores_x_equipo_A_escenario_2 = np.linspace(tiempos_equipo_A_escenario_2.min() * 0.9, tiempos_equipo_A_escenario_2.max() * 1.05, 300)
valores_x_equipo_B_escenario_2 = np.linspace(tiempos_equipo_B_escenario_2.min() * 0.9, tiempos_equipo_B_escenario_2.max() * 1.05, 300)

ejes[0, 0].hist(tiempos_equipo_A_escenario_1, bins=15, density=True, color='skyblue', edgecolor='black')
ejes[0, 0].plot(valores_x_equipo_A_escenario_1, distribucion_teorica_equipo_A_escenario_1.pdf(valores_x_equipo_A_escenario_1), 'r-', linewidth=2)
ejes[0, 0].set_title(f'Escenario 1 - Equipo A: Normal (media {media_teorica_equipo_A_escenario_1}, desv. {desviacion_teorica_equipo_A_escenario_1})', fontsize=11)

ejes[0, 1].hist(tiempos_equipo_B_escenario_1, bins=15, density=True, color='salmon', edgecolor='black')
ejes[0, 1].plot(valores_x_equipo_B_escenario_1, distribucion_teorica_equipo_B_escenario_1.pdf(valores_x_equipo_B_escenario_1), 'r-', linewidth=2)
ejes[0, 1].set_title(f'Escenario 1 - Equipo B: Normal (media {media_teorica_equipo_B_escenario_1}, desv. {desviacion_teorica_equipo_B_escenario_1})', fontsize=11)

ejes[1, 0].hist(tiempos_equipo_A_escenario_2, bins=15, density=True, color='lightgreen', edgecolor='black')
ejes[1, 0].plot(valores_x_equipo_A_escenario_2, distribucion_teorica_equipo_A_escenario_2.pdf(valores_x_equipo_A_escenario_2), 'r-', linewidth=2)
ejes[1, 0].set_title(f'Escenario 2 - Equipo A: Gamma (α = {alfa_teorico_equipo_A_escenario_2}, β = {beta_teorico_equipo_A_escenario_2})', fontsize=11)

ejes[1, 1].hist(tiempos_equipo_B_escenario_2, bins=15, density=True, color='gold', edgecolor='black')
ejes[1, 1].plot(valores_x_equipo_B_escenario_2, distribucion_teorica_equipo_B_escenario_2.pdf(valores_x_equipo_B_escenario_2), 'r-', linewidth=2)
ejes[1, 1].set_title(f'Escenario 2 - Equipo B: Gamma (α = {alfa_teorico_equipo_B_escenario_2}, β = {beta_teorico_equipo_B_escenario_2})', fontsize=11)

for eje in ejes.flat:
    eje.set_xlabel('Tiempo (ms)')
    eje.set_ylabel('Densidad')
    eje.containers[0].set_label('Datos generados (n = 60)')
    eje.lines[0].set_label('Densidad teórica')
    eje.set_ylim(top=eje.get_ylim()[1] * 1.3)  # Espacio arriba para que la leyenda no tape las barras
    eje.legend(loc='upper right')

plt.tight_layout()
plt.savefig('histogramas_punto_2.png', dpi=150)
plt.show()

print('\n')
print('===========================================================================')

print('\n---------- ENUNCIADO b) ----------')
print('Comparación de medias: t de Student vs Normal (Z)')
print('Ho: μ_A = μ_B')
print('Ha: μ_A ≠ μ_B')
print(f'Prueba bilateral con alpha = {alpha}')

z_critico = stats.norm.ppf(1 - alpha / 2)
grados_libertad_agrupada = 2 * tamano_muestra - 2

# ---------------------------- ESCENARIO 1 ----------------------------
print('\n--- Escenario 1 (Normal) ---')
diferencia_medias_escenario_1 = media_muestral_equipo_A_escenario_1 - media_muestral_equipo_B_escenario_1
print(f'Media A = {media_muestral_equipo_A_escenario_1:.2f} | Media B = {media_muestral_equipo_B_escenario_1:.2f} | Diferencia = {diferencia_medias_escenario_1:.2f}')

# Prueba F para saber si las varianzas son iguales (justifica el uso de Welch)
estadistico_F_escenario_1 = varianza_muestral_equipo_A_escenario_1 / varianza_muestral_equipo_B_escenario_1
valor_p_F_escenario_1 = 2 * min(stats.f.cdf(estadistico_F_escenario_1, tamano_muestra - 1, tamano_muestra - 1),
                                1 - stats.f.cdf(estadistico_F_escenario_1, tamano_muestra - 1, tamano_muestra - 1))
print(f'Prueba F de varianzas: F = {estadistico_F_escenario_1:.4f}, valor p = {valor_p_F_escenario_1:.4g} ->', 'varianzas distintas' if valor_p_F_escenario_1 < alpha else 'varianzas iguales')

# t de Welch (varianzas distintas)
error_estandar_muestral_escenario_1 = math.sqrt(varianza_muestral_equipo_A_escenario_1 / tamano_muestra + varianza_muestral_equipo_B_escenario_1 / tamano_muestra)
estadistico_t_welch_escenario_1 = diferencia_medias_escenario_1 / error_estandar_muestral_escenario_1
grados_libertad_welch_escenario_1 = (varianza_muestral_equipo_A_escenario_1 / tamano_muestra + varianza_muestral_equipo_B_escenario_1 / tamano_muestra) ** 2 / (
    (varianza_muestral_equipo_A_escenario_1 / tamano_muestra) ** 2 / (tamano_muestra - 1) + (varianza_muestral_equipo_B_escenario_1 / tamano_muestra) ** 2 / (tamano_muestra - 1))
t_critico_welch_escenario_1 = stats.t.ppf(1 - alpha / 2, grados_libertad_welch_escenario_1)
valor_p_t_welch_escenario_1 = 2 * stats.t.sf(abs(estadistico_t_welch_escenario_1), grados_libertad_welch_escenario_1)
limite_inferior_t_welch_escenario_1 = diferencia_medias_escenario_1 - t_critico_welch_escenario_1 * error_estandar_muestral_escenario_1
limite_superior_t_welch_escenario_1 = diferencia_medias_escenario_1 + t_critico_welch_escenario_1 * error_estandar_muestral_escenario_1

# t con varianzas iguales (agrupada)
varianza_agrupada_escenario_1 = ((tamano_muestra - 1) * varianza_muestral_equipo_A_escenario_1 + (tamano_muestra - 1) * varianza_muestral_equipo_B_escenario_1) / grados_libertad_agrupada
error_estandar_agrupado_escenario_1 = math.sqrt(varianza_agrupada_escenario_1 * (1 / tamano_muestra + 1 / tamano_muestra))
estadistico_t_agrupada_escenario_1 = diferencia_medias_escenario_1 / error_estandar_agrupado_escenario_1
t_critico_agrupada = stats.t.ppf(1 - alpha / 2, grados_libertad_agrupada)
valor_p_t_agrupada_escenario_1 = 2 * stats.t.sf(abs(estadistico_t_agrupada_escenario_1), grados_libertad_agrupada)
limite_inferior_t_agrupada_escenario_1 = diferencia_medias_escenario_1 - t_critico_agrupada * error_estandar_agrupado_escenario_1
limite_superior_t_agrupada_escenario_1 = diferencia_medias_escenario_1 + t_critico_agrupada * error_estandar_agrupado_escenario_1

# Z con las varianzas de la muestra
estadistico_z_muestral_escenario_1 = diferencia_medias_escenario_1 / error_estandar_muestral_escenario_1
valor_p_z_muestral_escenario_1 = 2 * stats.norm.sf(abs(estadistico_z_muestral_escenario_1))
limite_inferior_z_muestral_escenario_1 = diferencia_medias_escenario_1 - z_critico * error_estandar_muestral_escenario_1
limite_superior_z_muestral_escenario_1 = diferencia_medias_escenario_1 + z_critico * error_estandar_muestral_escenario_1

# Z con las varianzas teóricas (conocidas por el enunciado)
error_estandar_teorico_escenario_1 = math.sqrt(varianza_teorica_equipo_A_escenario_1 / tamano_muestra + varianza_teorica_equipo_B_escenario_1 / tamano_muestra)
estadistico_z_teorico_escenario_1 = diferencia_medias_escenario_1 / error_estandar_teorico_escenario_1
valor_p_z_teorico_escenario_1 = 2 * stats.norm.sf(abs(estadistico_z_teorico_escenario_1))
limite_inferior_z_teorico_escenario_1 = diferencia_medias_escenario_1 - z_critico * error_estandar_teorico_escenario_1
limite_superior_z_teorico_escenario_1 = diferencia_medias_escenario_1 + z_critico * error_estandar_teorico_escenario_1

print(f't de Welch:          t = {estadistico_t_welch_escenario_1:.4f}, crítico = ±{t_critico_welch_escenario_1:.3f} (gl = {grados_libertad_welch_escenario_1:.1f}), valor p = {valor_p_t_welch_escenario_1:.4f}, IC 95%: ({limite_inferior_t_welch_escenario_1:.2f}, {limite_superior_t_welch_escenario_1:.2f}) ->', 'Se rechaza Ho' if valor_p_t_welch_escenario_1 < alpha else 'No se rechaza Ho')
print(f't varianzas iguales: t = {estadistico_t_agrupada_escenario_1:.4f}, crítico = ±{t_critico_agrupada:.3f} (gl = {grados_libertad_agrupada}), valor p = {valor_p_t_agrupada_escenario_1:.4f}, IC 95%: ({limite_inferior_t_agrupada_escenario_1:.2f}, {limite_superior_t_agrupada_escenario_1:.2f}) ->', 'Se rechaza Ho' if valor_p_t_agrupada_escenario_1 < alpha else 'No se rechaza Ho')
print(f'Z muestral:          Z = {estadistico_z_muestral_escenario_1:.4f}, crítico = ±{z_critico:.3f}, valor p = {valor_p_z_muestral_escenario_1:.4f}, IC 95%: ({limite_inferior_z_muestral_escenario_1:.2f}, {limite_superior_z_muestral_escenario_1:.2f}) ->', 'Se rechaza Ho' if valor_p_z_muestral_escenario_1 < alpha else 'No se rechaza Ho')
print(f'Z teórico:           Z = {estadistico_z_teorico_escenario_1:.4f}, crítico = ±{z_critico:.3f}, valor p = {valor_p_z_teorico_escenario_1:.4f}, IC 95%: ({limite_inferior_z_teorico_escenario_1:.2f}, {limite_superior_z_teorico_escenario_1:.2f}) ->', 'Se rechaza Ho' if valor_p_z_teorico_escenario_1 < alpha else 'No se rechaza Ho')

# ---------------------------- ESCENARIO 2 ----------------------------
print('\n--- Escenario 2 (Gamma) ---')
diferencia_medias_escenario_2 = media_muestral_equipo_A_escenario_2 - media_muestral_equipo_B_escenario_2
print(f'Media A = {media_muestral_equipo_A_escenario_2:.2f} | Media B = {media_muestral_equipo_B_escenario_2:.2f} | Diferencia = {diferencia_medias_escenario_2:.2f}')

# Prueba F para saber si las varianzas son iguales
estadistico_F_escenario_2 = varianza_muestral_equipo_A_escenario_2 / varianza_muestral_equipo_B_escenario_2
valor_p_F_escenario_2 = 2 * min(stats.f.cdf(estadistico_F_escenario_2, tamano_muestra - 1, tamano_muestra - 1),
                                1 - stats.f.cdf(estadistico_F_escenario_2, tamano_muestra - 1, tamano_muestra - 1))
print(f'Prueba F de varianzas: F = {estadistico_F_escenario_2:.4f}, valor p = {valor_p_F_escenario_2:.4g} ->', 'varianzas distintas' if valor_p_F_escenario_2 < alpha else 'varianzas iguales')

# t de Welch (varianzas distintas)
error_estandar_muestral_escenario_2 = math.sqrt(varianza_muestral_equipo_A_escenario_2 / tamano_muestra + varianza_muestral_equipo_B_escenario_2 / tamano_muestra)
estadistico_t_welch_escenario_2 = diferencia_medias_escenario_2 / error_estandar_muestral_escenario_2
grados_libertad_welch_escenario_2 = (varianza_muestral_equipo_A_escenario_2 / tamano_muestra + varianza_muestral_equipo_B_escenario_2 / tamano_muestra) ** 2 / (
    (varianza_muestral_equipo_A_escenario_2 / tamano_muestra) ** 2 / (tamano_muestra - 1) + (varianza_muestral_equipo_B_escenario_2 / tamano_muestra) ** 2 / (tamano_muestra - 1))
t_critico_welch_escenario_2 = stats.t.ppf(1 - alpha / 2, grados_libertad_welch_escenario_2)
valor_p_t_welch_escenario_2 = 2 * stats.t.sf(abs(estadistico_t_welch_escenario_2), grados_libertad_welch_escenario_2)
limite_inferior_t_welch_escenario_2 = diferencia_medias_escenario_2 - t_critico_welch_escenario_2 * error_estandar_muestral_escenario_2
limite_superior_t_welch_escenario_2 = diferencia_medias_escenario_2 + t_critico_welch_escenario_2 * error_estandar_muestral_escenario_2

# t con varianzas iguales (agrupada)
varianza_agrupada_escenario_2 = ((tamano_muestra - 1) * varianza_muestral_equipo_A_escenario_2 + (tamano_muestra - 1) * varianza_muestral_equipo_B_escenario_2) / grados_libertad_agrupada
error_estandar_agrupado_escenario_2 = math.sqrt(varianza_agrupada_escenario_2 * (1 / tamano_muestra + 1 / tamano_muestra))
estadistico_t_agrupada_escenario_2 = diferencia_medias_escenario_2 / error_estandar_agrupado_escenario_2
valor_p_t_agrupada_escenario_2 = 2 * stats.t.sf(abs(estadistico_t_agrupada_escenario_2), grados_libertad_agrupada)
limite_inferior_t_agrupada_escenario_2 = diferencia_medias_escenario_2 - t_critico_agrupada * error_estandar_agrupado_escenario_2
limite_superior_t_agrupada_escenario_2 = diferencia_medias_escenario_2 + t_critico_agrupada * error_estandar_agrupado_escenario_2

# Z con las varianzas de la muestra
estadistico_z_muestral_escenario_2 = diferencia_medias_escenario_2 / error_estandar_muestral_escenario_2
valor_p_z_muestral_escenario_2 = 2 * stats.norm.sf(abs(estadistico_z_muestral_escenario_2))
limite_inferior_z_muestral_escenario_2 = diferencia_medias_escenario_2 - z_critico * error_estandar_muestral_escenario_2
limite_superior_z_muestral_escenario_2 = diferencia_medias_escenario_2 + z_critico * error_estandar_muestral_escenario_2

# Z con las varianzas teóricas (alfa * beta^2)
error_estandar_teorico_escenario_2 = math.sqrt(varianza_teorica_equipo_A_escenario_2 / tamano_muestra + varianza_teorica_equipo_B_escenario_2 / tamano_muestra)
estadistico_z_teorico_escenario_2 = diferencia_medias_escenario_2 / error_estandar_teorico_escenario_2
valor_p_z_teorico_escenario_2 = 2 * stats.norm.sf(abs(estadistico_z_teorico_escenario_2))
limite_inferior_z_teorico_escenario_2 = diferencia_medias_escenario_2 - z_critico * error_estandar_teorico_escenario_2
limite_superior_z_teorico_escenario_2 = diferencia_medias_escenario_2 + z_critico * error_estandar_teorico_escenario_2

print(f't de Welch:          t = {estadistico_t_welch_escenario_2:.4f}, crítico = ±{t_critico_welch_escenario_2:.3f} (gl = {grados_libertad_welch_escenario_2:.1f}), valor p = {valor_p_t_welch_escenario_2:.4f}, IC 95%: ({limite_inferior_t_welch_escenario_2:.2f}, {limite_superior_t_welch_escenario_2:.2f}) ->', 'Se rechaza Ho' if valor_p_t_welch_escenario_2 < alpha else 'No se rechaza Ho')
print(f't varianzas iguales: t = {estadistico_t_agrupada_escenario_2:.4f}, crítico = ±{t_critico_agrupada:.3f} (gl = {grados_libertad_agrupada}), valor p = {valor_p_t_agrupada_escenario_2:.4f}, IC 95%: ({limite_inferior_t_agrupada_escenario_2:.2f}, {limite_superior_t_agrupada_escenario_2:.2f}) ->', 'Se rechaza Ho' if valor_p_t_agrupada_escenario_2 < alpha else 'No se rechaza Ho')
print(f'Z muestral:          Z = {estadistico_z_muestral_escenario_2:.4f}, crítico = ±{z_critico:.3f}, valor p = {valor_p_z_muestral_escenario_2:.4f}, IC 95%: ({limite_inferior_z_muestral_escenario_2:.2f}, {limite_superior_z_muestral_escenario_2:.2f}) ->', 'Se rechaza Ho' if valor_p_z_muestral_escenario_2 < alpha else 'No se rechaza Ho')
print(f'Z teórico:           Z = {estadistico_z_teorico_escenario_2:.4f}, crítico = ±{z_critico:.3f}, valor p = {valor_p_z_teorico_escenario_2:.4f}, IC 95%: ({limite_inferior_z_teorico_escenario_2:.2f}, {limite_superior_z_teorico_escenario_2:.2f}) ->', 'Se rechaza Ho' if valor_p_z_teorico_escenario_2 < alpha else 'No se rechaza Ho')

print('\n')
print('===========================================================================')

print('\n---------- ENUNCIADO c) ----------')
print('Prueba no paramétrica de Mann-Whitney (Wilcoxon rango-suma)')
print('Ho: las dos distribuciones son iguales (P(A > B) = 0.5)')
print('Ha: las dos distribuciones son distintas')

numero_total_datos = 2 * tamano_muestra
media_U_bajo_Ho = tamano_muestra * tamano_muestra / 2
desviacion_U_bajo_Ho = math.sqrt(tamano_muestra * tamano_muestra * (numero_total_datos + 1) / 12)

# ---------------------------- ESCENARIO 1 ----------------------------
print('\n--- Escenario 1 (Normal) ---')
estadistico_U_escenario_1, valor_p_mann_whitney_escenario_1 = stats.mannwhitneyu(tiempos_equipo_A_escenario_1, tiempos_equipo_B_escenario_1, alternative='two-sided')

# Comprobación manual con rangos: se juntan los 120 datos y se ordenan
rangos_conjuntos_escenario_1 = stats.rankdata(np.concatenate([tiempos_equipo_A_escenario_1, tiempos_equipo_B_escenario_1]))
suma_rangos_equipo_A_escenario_1 = rangos_conjuntos_escenario_1[:tamano_muestra].sum()
estadistico_U_manual_escenario_1 = suma_rangos_equipo_A_escenario_1 - tamano_muestra * (tamano_muestra + 1) / 2
estadistico_z_mann_whitney_escenario_1 = (estadistico_U_escenario_1 - media_U_bajo_Ho) / desviacion_U_bajo_Ho
probabilidad_A_mayor_que_B_escenario_1 = estadistico_U_escenario_1 / (tamano_muestra * tamano_muestra)

print(f'Suma de rangos del equipo A = {suma_rangos_equipo_A_escenario_1:.1f} (esperada bajo Ho: {tamano_muestra * (numero_total_datos + 1) / 2:.1f})')
print(f'U = {estadistico_U_escenario_1:.1f} (calculado a mano: {estadistico_U_manual_escenario_1:.1f})')
print(f'Z aproximada = {estadistico_z_mann_whitney_escenario_1:.4f}')
print(f'Valor p = {valor_p_mann_whitney_escenario_1:.4f} ->', 'Se rechaza Ho' if valor_p_mann_whitney_escenario_1 < alpha else 'No se rechaza Ho')
print(f'P(A > B) estimada = {probabilidad_A_mayor_que_B_escenario_1:.4f}')

# ---------------------------- ESCENARIO 2 ----------------------------
print('\n--- Escenario 2 (Gamma) ---')
estadistico_U_escenario_2, valor_p_mann_whitney_escenario_2 = stats.mannwhitneyu(tiempos_equipo_A_escenario_2, tiempos_equipo_B_escenario_2, alternative='two-sided')

rangos_conjuntos_escenario_2 = stats.rankdata(np.concatenate([tiempos_equipo_A_escenario_2, tiempos_equipo_B_escenario_2]))
suma_rangos_equipo_A_escenario_2 = rangos_conjuntos_escenario_2[:tamano_muestra].sum()
estadistico_U_manual_escenario_2 = suma_rangos_equipo_A_escenario_2 - tamano_muestra * (tamano_muestra + 1) / 2
estadistico_z_mann_whitney_escenario_2 = (estadistico_U_escenario_2 - media_U_bajo_Ho) / desviacion_U_bajo_Ho
probabilidad_A_mayor_que_B_escenario_2 = estadistico_U_escenario_2 / (tamano_muestra * tamano_muestra)

print(f'Suma de rangos del equipo A = {suma_rangos_equipo_A_escenario_2:.1f} (esperada bajo Ho: {tamano_muestra * (numero_total_datos + 1) / 2:.1f})')
print(f'U = {estadistico_U_escenario_2:.1f} (calculado a mano: {estadistico_U_manual_escenario_2:.1f})')
print(f'Z aproximada = {estadistico_z_mann_whitney_escenario_2:.4f}')
print(f'Valor p = {valor_p_mann_whitney_escenario_2:.4f} ->', 'Se rechaza Ho' if valor_p_mann_whitney_escenario_2 < alpha else 'No se rechaza Ho')
print(f'P(A > B) estimada = {probabilidad_A_mayor_que_B_escenario_2:.4f}')

# ---------------------- COMPARACIÓN CON EL LITERAL b) ----------------------
print('\n--- Comparación de todas las pruebas (valores p) ---')
comparacion_pruebas = pd.DataFrame({
    't de Welch': [valor_p_t_welch_escenario_1, valor_p_t_welch_escenario_2],
    'Z muestral': [valor_p_z_muestral_escenario_1, valor_p_z_muestral_escenario_2],
    'Z teórico': [valor_p_z_teorico_escenario_1, valor_p_z_teorico_escenario_2],
    'Mann-Whitney': [valor_p_mann_whitney_escenario_1, valor_p_mann_whitney_escenario_2],
}, index=['Escenario 1 (Normal)', 'Escenario 2 (Gamma)'])
comparacion_pruebas['Decisión (todas)'] = np.where((comparacion_pruebas[['t de Welch', 'Z muestral', 'Z teórico', 'Mann-Whitney']] < alpha).any(axis=1), 'Alguna rechaza Ho', 'Ninguna rechaza Ho')
print(comparacion_pruebas.round(4).to_string())

print('\nMedias teóricas (las que definen el enunciado):')
print(f'Escenario 1: A = {media_teorica_equipo_A_escenario_1} | B = {media_teorica_equipo_B_escenario_1}')
print(f'Escenario 2: A = alfa*beta = {alfa_teorico_equipo_A_escenario_2 * beta_teorico_equipo_A_escenario_2:.1f} | B = alfa*beta = {alfa_teorico_equipo_B_escenario_2 * beta_teorico_equipo_B_escenario_2:.1f}')
print(f'Medianas teóricas Escenario 2: A = {distribucion_teorica_equipo_A_escenario_2.median():.1f} | B = {distribucion_teorica_equipo_B_escenario_2.median():.1f}')
print('\n')
print('===========================================================================')
