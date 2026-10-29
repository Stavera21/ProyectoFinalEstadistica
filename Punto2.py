import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats  

print('===========================================================================')

print('\n---------- DATOS BASE ----------')
np.random.seed(42)  
tamano_muestra = 60
alpha = 0.05

# Escenario 1: Normal (media y desviación estándar)
media_teorica_equipo_A_escenario_1 = 175
desviacion_teorica_equipo_A_escenario_1 = 30
media_teorica_equipo_B_escenario_1 = 170
desviacion_teorica_equipo_B_escenario_1 = 50

# Escenario 2: Gamma (alfa y beta)
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

# Medias y varianzas muestrales 
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
    eje.set_ylim(top=eje.get_ylim()[1] * 1.3)  
    eje.legend(loc='upper right')

plt.tight_layout()
plt.savefig('histogramas_punto_2.png', dpi=150)
plt.show()

print('\n')
print('===========================================================================')

print('\n---------- ENUNCIADO b) ----------')
print('Comparación de medias: t de Student y distribución Normal (Z)')
print('H0: μ_A = μ_B')
print('H1: μ_A ≠ μ_B')
print(f'Pruebas bilaterales con alpha = {alpha}')
print('Nota: para Z se usan las varianzas teóricas conocidas por los parámetros del enunciado.')

z_critico = stats.norm.ppf(1 - alpha / 2)
grados_libertad_agrupada = 2 * tamano_muestra - 2

resultados_b = {}

for escenario, datos, media_A, media_B, var_A, var_B in [
    (1, datos_escenario_1, media_muestral_equipo_A_escenario_1,
     media_muestral_equipo_B_escenario_1, varianza_muestral_equipo_A_escenario_1,
     varianza_muestral_equipo_B_escenario_1),
    (2, datos_escenario_2, media_muestral_equipo_A_escenario_2,
     media_muestral_equipo_B_escenario_2, varianza_muestral_equipo_A_escenario_2,
     varianza_muestral_equipo_B_escenario_2),
]:
    print(f'\n--- Escenario {escenario} ---')
    diferencia = media_A - media_B
    print(f'Media A = {media_A:.2f} | Media B = {media_B:.2f} | Diferencia A-B = {diferencia:.2f}')

    # t de Student para dos muestras independientes, suponiendo varianzas iguales.
    varianza_agrupada = ((tamano_muestra - 1) * var_A + (tamano_muestra - 1) * var_B) / grados_libertad_agrupada
    error_t = math.sqrt(varianza_agrupada * (1 / tamano_muestra + 1 / tamano_muestra))
    t_obs = diferencia / error_t
    t_critico = stats.t.ppf(1 - alpha / 2, grados_libertad_agrupada)
    p_t = 2 * stats.t.sf(abs(t_obs), grados_libertad_agrupada)
    li_t, ls_t = diferencia - t_critico * error_t, diferencia + t_critico * error_t

    # Z: se conocen las varianzas teóricas a partir de los parámetros dados.
    if escenario == 1:
        var_teorica_A = varianza_teorica_equipo_A_escenario_1
        var_teorica_B = varianza_teorica_equipo_B_escenario_1
    else:
        var_teorica_A = varianza_teorica_equipo_A_escenario_2
        var_teorica_B = varianza_teorica_equipo_B_escenario_2
    error_z = math.sqrt(var_teorica_A / tamano_muestra + var_teorica_B / tamano_muestra)
    z_obs = diferencia / error_z
    p_z = 2 * stats.norm.sf(abs(z_obs))
    li_z, ls_z = diferencia - z_critico * error_z, diferencia + z_critico * error_z

    decision_t = 'Se rechaza H0' if p_t < alpha else 'No se rechaza H0'
    decision_z = 'Se rechaza H0' if p_z < alpha else 'No se rechaza H0'
    print(f't de Student: t = {t_obs:.4f}, gl = {grados_libertad_agrupada}, t crítico = ±{t_critico:.3f}, p = {p_t:.4f}, IC 95% = ({li_t:.2f}, {ls_t:.2f}) -> {decision_t}')
    print(f'Normal (Z):   Z = {z_obs:.4f}, Z crítico = ±{z_critico:.3f}, p = {p_z:.4f}, IC 95% = ({li_z:.2f}, {ls_z:.2f}) -> {decision_z}')
    resultados_b[escenario] = {'t_Student': p_t, 'Z_teorico': p_z}

print('\n---------- ENUNCIADO c) ----------')
print('Prueba no paramétrica de Mann-Whitney (Wilcoxon rango-suma)')
print('H0: las distribuciones de ambos equipos son iguales.')
print('H1: las distribuciones de ambos equipos son diferentes.')

# ---------------------------- ESCENARIO 1 ----------------------------
print('\n--- Escenario 1 (Normal) ---')
estadistico_U_escenario_1, valor_p_mann_whitney_escenario_1 = stats.mannwhitneyu(tiempos_equipo_A_escenario_1, tiempos_equipo_B_escenario_1, alternative='two-sided')

print(f'Valor p = {valor_p_mann_whitney_escenario_1:.4f} ->', 'Se rechaza Ho' if valor_p_mann_whitney_escenario_1 < alpha else 'No se rechaza Ho')

# ---------------------------- ESCENARIO 2 ----------------------------
print('\n--- Escenario 2 (Gamma) ---')
estadistico_U_escenario_2, valor_p_mann_whitney_escenario_2 = stats.mannwhitneyu(tiempos_equipo_A_escenario_2, tiempos_equipo_B_escenario_2, alternative='two-sided')

print(f'Valor p = {valor_p_mann_whitney_escenario_2:.4f} ->', 'Se rechaza Ho' if valor_p_mann_whitney_escenario_2 < alpha else 'No se rechaza Ho')

# ---------------------- COMPARACIÓN CON EL LITERAL b) ----------------------
print('\n--- Comparación de todas las pruebas (valores p) ---')
comparacion_pruebas = pd.DataFrame({
    't de Student': [resultados_b[1]['t_Student'], resultados_b[2]['t_Student']],
    'Z teórico': [resultados_b[1]['Z_teorico'], resultados_b[2]['Z_teorico']],
    'Mann-Whitney': [valor_p_mann_whitney_escenario_1, valor_p_mann_whitney_escenario_2],
}, index=['Escenario 1 (Normal)', 'Escenario 2 (Gamma)'])
comparacion_pruebas['Decisión (todas)'] = np.where((comparacion_pruebas[['t de Student', 'Z teórico', 'Mann-Whitney']] < alpha).any(axis=1), 'Alguna rechaza H0', 'Ninguna rechaza H0')
print(comparacion_pruebas.round(4).to_string())

print('\n')
print('===========================================================================')
