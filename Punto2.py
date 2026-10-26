import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
 
# =============================================================================
# CONFIGURACIÓN Y GENERACIÓN DE DATOS
# =============================================================================
np.random.seed(42)          # Reproducibilidad
n = 60
alpha = 0.05                # Nivel de significancia
 
# Parámetros teóricos
mu_A1, sd_A1 = 175, 30
mu_B1, sd_B1 = 170, 50
a_A2, b_A2 = 1.2, 150       # Gamma: alfa, beta  (scipy: a=alfa, scale=beta)
a_B2, b_B2 = 0.95, 190
 
A1 = stats.norm.rvs(loc=mu_A1, scale=sd_A1, size=n)
B1 = stats.norm.rvs(loc=mu_B1, scale=sd_B1, size=n)
A2 = stats.gamma.rvs(a=a_A2, scale=b_A2, size=n)
B2 = stats.gamma.rvs(a=a_B2, scale=b_B2, size=n)
 
grupos = {
    'E1 - Equipo A (Normal)': A1,
    'E1 - Equipo B (Normal)': B1,
    'E2 - Equipo A (Gamma)':  A2,
    'E2 - Equipo B (Gamma)':  B2,
}
 
# =============================================================================
# a) MEDIDAS DESCRIPTIVAS + BONDAD DE AJUSTE + HISTOGRAMAS
# =============================================================================
print('\n' + '=' * 75)
print('a) Medidas descriptivas y pruebas de bondad de ajuste')
print('=' * 75)
 
filas = []
for nombre, x in grupos.items():
    filas.append({
        'Grupo': nombre,
        'n': len(x),
        'Media': np.mean(x),
        'Mediana': np.median(x),
        'Desv.Est': np.std(x, ddof=1),
        'Varianza': np.var(x, ddof=1),
        'CV(%)': np.std(x, ddof=1) / np.mean(x) * 100,
        'Min': np.min(x),
        'Max': np.max(x),
        'Asimetría': stats.skew(x),
        'Curtosis': stats.kurtosis(x),   # exceso de curtosis (normal = 0)
    })
tabla_desc = pd.DataFrame(filas).set_index('Grupo')
print(tabla_desc.round(2).to_string())
 
# Distribución teórica de cada grupo (para K-S con parámetros conocidos)
teoricas = {
    'E1 - Equipo A (Normal)': ('norm',  (mu_A1, sd_A1)),
    'E1 - Equipo B (Normal)': ('norm',  (mu_B1, sd_B1)),
    'E2 - Equipo A (Gamma)':  ('gamma', (a_A2, 0, b_A2)),
    'E2 - Equipo B (Gamma)':  ('gamma', (a_B2, 0, b_B2)),
}
 
print('\nH0: los datos provienen de la distribución indicada | Ha: no provienen')
print(f'Regla: se rechaza H0 si p-valor < {alpha}\n')
filas = []
for nombre, x in grupos.items():
    dist, params = teoricas[nombre]
    ks_stat, ks_p = stats.kstest(x, getattr(stats, dist)(*params).cdf)
    sw_stat, sw_p = stats.shapiro(x)
    filas.append({
        'Grupo': nombre,
        'KS vs teórica (p)': ks_p,
        'Decisión KS': 'No rechaza H0' if ks_p >= alpha else 'Rechaza H0',
        'Shapiro normalidad (p)': sw_p,
        'Decisión Shapiro': 'Normal' if sw_p >= alpha else 'No normal',
    })
print(pd.DataFrame(filas).set_index('Grupo').round(4).to_string())
 
# Histogramas con la densidad teórica superpuesta
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Histogramas de tiempo de procesamiento (ms) con densidad teórica')
colores = ['skyblue', 'salmon', 'lightgreen', 'gold']
for ax, (nombre, x), c in zip(axes.ravel(), grupos.items(), colores):
    dist, params = teoricas[nombre]
    ax.hist(x, bins=15, density=True, color=c, edgecolor='black', alpha=0.8)
    xs = np.linspace(max(x.min() * 0.9, 0.01), x.max() * 1.05, 300)
    ax.plot(xs, getattr(stats, dist).pdf(xs, *params), 'r-', lw=2, label='Densidad teórica')
    ax.set_title(nombre)
    ax.set_xlabel('Tiempo (ms)')
    ax.set_ylabel('Densidad')
    ax.legend()
plt.tight_layout()
plt.savefig('histogramas.png', dpi=150)
plt.show()
 
# =============================================================================
# b) COMPARACIÓN DE MEDIAS: T-STUDENT vs NORMAL (Z)
# =============================================================================
print('\n' + '=' * 75)
print('b) Comparación de medias: t de Student vs Normal (Z)')
print('=' * 75)
print('H0: mu_A = mu_B   |   Ha: mu_A != mu_B   (bilateral, alpha = 0.05)\n')
 
 
def prueba_t(x, y, equal_var):
    t, p = stats.ttest_ind(x, y, equal_var=equal_var)
    n1, n2 = len(x), len(y)
    v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
    if equal_var:
        gl = n1 + n2 - 2
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / gl
        ee = math.sqrt(sp2 * (1 / n1 + 1 / n2))
    else:
        ee = math.sqrt(v1 / n1 + v2 / n2)
        gl = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    dif = np.mean(x) - np.mean(y)
    tc = stats.t.ppf(1 - alpha / 2, gl)
    return t, p, gl, tc, (dif - tc * ee, dif + tc * ee)
 
 
def prueba_z(x, y, var_x=None, var_y=None):
    """Z con varianzas poblacionales conocidas (si se dan) o con las muestrales."""
    n1, n2 = len(x), len(y)
    v1 = var_x if var_x is not None else np.var(x, ddof=1)
    v2 = var_y if var_y is not None else np.var(y, ddof=1)
    ee = math.sqrt(v1 / n1 + v2 / n2)
    dif = np.mean(x) - np.mean(y)
    z = dif / ee
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    zc = stats.norm.ppf(1 - alpha / 2)
    return z, p, zc, (dif - zc * ee, dif + zc * ee)
 
 
# Varianzas teóricas: Normal -> sigma^2 ; Gamma -> alfa * beta^2
var_A1, var_B1 = sd_A1 ** 2, sd_B1 ** 2
var_A2, var_B2 = a_A2 * b_A2 ** 2, a_B2 * b_B2 ** 2
 
escenarios = {
    'Escenario 1 (Normal)': (A1, B1, var_A1, var_B1),
    'Escenario 2 (Gamma)':  (A2, B2, var_A2, var_B2),
}
 
resultados_b = {}
for nombre, (x, y, vx, vy) in escenarios.items():
    print(f'--- {nombre} ---')
    print(f'Media A = {np.mean(x):.2f} | Media B = {np.mean(y):.2f} | diferencia = {np.mean(x) - np.mean(y):.2f}')
    # Prueba F de igualdad de varianzas (para justificar t con o sin varianzas iguales)
    F = np.var(x, ddof=1) / np.var(y, ddof=1)
    pF = 2 * min(stats.f.cdf(F, len(x) - 1, len(y) - 1), 1 - stats.f.cdf(F, len(x) - 1, len(y) - 1))
    print(f'Prueba F (varianzas iguales?): F = {F:.4f}, p = {pF:.4f} -> '
          f'{"varianzas distintas" if pF < alpha else "varianzas iguales"}')
 
    t_w, p_w, gl_w, tc_w, ic_w = prueba_t(x, y, equal_var=False)
    t_p, p_p, gl_p, tc_p, ic_p = prueba_t(x, y, equal_var=True)
    z_s, p_zs, zc, ic_zs = prueba_z(x, y)
    z_k, p_zk, _, ic_zk = prueba_z(x, y, vx, vy)
 
    tabla = pd.DataFrame([
        ['t Welch (var. distintas)',     t_w, f'±{tc_w:.3f} (gl={gl_w:.1f})', p_w,  ic_w],
        ['t pooled (var. iguales)',      t_p, f'±{tc_p:.3f} (gl={gl_p})',     p_p,  ic_p],
        ['Z (varianzas muestrales)',     z_s, f'±{zc:.3f}',                   p_zs, ic_zs],
        ['Z (varianzas teóricas conoc.)', z_k, f'±{zc:.3f}',                  p_zk, ic_zk],
    ], columns=['Prueba', 'Estadístico', 'Valor crítico', 'p-valor', 'IC 95% (mu_A - mu_B)'])
    tabla['Decisión'] = np.where(tabla['p-valor'] < alpha, 'Rechaza H0', 'No rechaza H0')
    tabla['IC 95% (mu_A - mu_B)'] = tabla['IC 95% (mu_A - mu_B)'].apply(lambda ic: f'({ic[0]:.2f}, {ic[1]:.2f})')
    print(tabla.round(4).to_string(index=False))
    print()
    resultados_b[nombre] = {'t': p_w, 'z': p_zs}
 
# =============================================================================
# c) MANN-WHITNEY (WILCOXON RANGO-SUMA)
# =============================================================================
print('=' * 75)
print('c) Prueba no paramétrica de Mann-Whitney (Wilcoxon rango-suma)')
print('=' * 75)
print('H0: P(X_A > X_B) = 0.5 (misma distribución) | Ha: las distribuciones difieren (bilateral)\n')
 
filas = []
for nombre, (x, y, _, _) in escenarios.items():
    U, p_mw = stats.mannwhitneyu(x, y, alternative='two-sided')
    # Tamaño de efecto: probabilidad de superioridad P(A > B) = U / (n1*n2)
    prob_sup = U / (len(x) * len(y))
    filas.append({
        'Escenario': nombre,
        'U': U,
        'p-valor MW': p_mw,
        'p-valor t Welch': resultados_b[nombre]['t'],
        'p-valor Z': resultados_b[nombre]['z'],
        'P(A>B) estimada': prob_sup,
        'Decisión MW': 'Rechaza H0' if p_mw < alpha else 'No rechaza H0',
        'Decisión t': 'Rechaza H0' if resultados_b[nombre]['t'] < alpha else 'No rechaza H0',
    })
print(pd.DataFrame(filas).set_index('Escenario').round(4).T.to_string())
 
print('\nMedias teóricas -> E1: A = 175, B = 170 | '
      f'E2: A = alfa*beta = {a_A2 * b_A2:.1f}, B = alfa*beta = {a_B2 * b_B2:.1f}')
print('Medianas teóricas -> E2: '
      f'A = {stats.gamma.median(a_A2, scale=b_A2):.1f}, B = {stats.gamma.median(a_B2, scale=b_B2):.1f}')
 