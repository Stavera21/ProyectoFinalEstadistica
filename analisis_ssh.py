"""
Literal c) Análisis estadístico
Intentos de conexión SSH (puerto 22) - dataset CIC-IDS2017

Periodo normal : lunes 3/07/2017 (solo tráfico benigno)
Periodo ataque : martes 4/07/2017, flujos con etiqueta SSH-Patator

Uso:  python analisis_ssh.py
Requiere: pandas, numpy, scipy, matplotlib
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------------------------------------------------
# CONFIGURACIÓN: ajusta las rutas a donde descomprimiste los CSV
# ----------------------------------------------------------------------
CARPETA = Path("datos")
ARCHIVO_NORMAL = CARPETA / "Monday-WorkingHours.pcap_ISCX.csv"
ARCHIVO_ATAQUE = CARPETA / "Tuesday-WorkingHours.pcap_ISCX.csv"
SALIDA = Path("salida")
PUERTO_SSH = 22
ETIQUETA_ATAQUE = "SSH-Patator"
ALFA = 0.05          # nivel de significancia de las pruebas
ALFA_ALERTA = 0.001  # prob. de falsa alarma permitida para el umbral

SALIDA.mkdir(exist_ok=True)
informe = []


def mostrar(texto=""):
    """Imprime y guarda el texto para el archivo resultados.txt"""
    print(texto)
    informe.append(str(texto))


# ----------------------------------------------------------------------
# 1. CARGA Y LIMPIEZA
# ----------------------------------------------------------------------
COLUMNAS = {"Timestamp", "Destination Port", "Label", "Flow Duration"}


def cargar(ruta):
    df = pd.read_csv(
        ruta,
        encoding="latin-1",
        low_memory=False,
        usecols=lambda c: c.strip() in COLUMNAS,
    )
    df.columns = df.columns.str.strip()  # los CSV traen espacios en los nombres
    df["Destination Port"] = pd.to_numeric(df["Destination Port"], errors="coerce")
    df["Flow Duration"] = pd.to_numeric(df["Flow Duration"], errors="coerce")
    df["Timestamp"] = parsear_tiempo(df["Timestamp"])
    df = df.dropna(subset=["Timestamp", "Destination Port"])
    return df


def parsear_tiempo(serie):
    """Convierte el texto a fecha. Prueba día/mes y mes/día y se queda con
    la que da julio de 2017. Corrige la falta de AM/PM (capturas desde ~8:00)."""
    for dayfirst in (True, False):
        t = pd.to_datetime(serie, dayfirst=dayfirst, errors="coerce", format="mixed")
        if t.notna().any() and (t.dt.month.dropna() == 7).all():
            break
    # Los CSV no traen AM/PM: horas 1-7 corresponden a la tarde (13-19 h)
    t = t.where(t.dt.hour >= 8, t + pd.Timedelta(hours=12))
    return t


mostrar("Cargando datos (puede tardar un poco)...")
lunes = cargar(ARCHIVO_NORMAL)
martes = cargar(ARCHIVO_ATAQUE)

mostrar(f"Lunes : {len(lunes):,} flujos | {lunes['Timestamp'].min()} -> {lunes['Timestamp'].max()}")
mostrar(f"Martes: {len(martes):,} flujos | {martes['Timestamp'].min()} -> {martes['Timestamp'].max()}")

# Periodo normal: flujos al puerto 22 del lunes
ssh_normal = lunes[lunes["Destination Port"] == PUERTO_SSH].copy()
# Periodo de ataque: flujos al puerto 22 etiquetados como SSH-Patator
ssh_ataque = martes[
    (martes["Destination Port"] == PUERTO_SSH) & (martes["Label"].str.strip() == ETIQUETA_ATAQUE)
].copy()

mostrar(f"\nFlujos SSH normales (lunes)      : {len(ssh_normal):,}")
mostrar(f"Flujos SSH de ataque (SSH-Patator): {len(ssh_ataque):,}")
if len(ssh_normal) == 0 or len(ssh_ataque) == 0:
    raise SystemExit("No hay datos SSH en alguno de los periodos. Revisa rutas y etiquetas.")

# ----------------------------------------------------------------------
# 2. VARIABLE DISCRETA: intentos por minuto
# ----------------------------------------------------------------------


def conteo_por_minuto(df_ssh, inicio, fin):
    """Cuenta intentos por minuto, incluyendo los minutos con 0 intentos."""
    minutos = pd.date_range(inicio.floor("min"), fin.floor("min"), freq="min")
    conteo = df_ssh["Timestamp"].dt.floor("min").value_counts()
    return conteo.reindex(minutos, fill_value=0).astype(int)


# Normal: todo el lunes (primer a último flujo, de cualquier puerto)
c_normal = conteo_por_minuto(ssh_normal, lunes["Timestamp"].min(), lunes["Timestamp"].max())
# Ataque: desde el primer hasta el último flujo SSH-Patator
c_ataque = conteo_por_minuto(ssh_ataque, ssh_ataque["Timestamp"].min(), ssh_ataque["Timestamp"].max())

tabla = pd.concat(
    [
        pd.DataFrame({"minuto": c_normal.index, "intentos": c_normal.values, "periodo": "normal"}),
        pd.DataFrame({"minuto": c_ataque.index, "intentos": c_ataque.values, "periodo": "ataque"}),
    ]
)
tabla.to_csv(SALIDA / "conteos_por_minuto.csv", index=False)

# ----------------------------------------------------------------------
# 3. ESTADÍSTICA DESCRIPTIVA
# ----------------------------------------------------------------------


def descriptivas(c, nombre):
    media, var = c.mean(), c.var(ddof=1)
    return {
        "periodo": nombre,
        "n (minutos)": len(c),
        "media": media,
        "mediana": c.median(),
        "moda": c.mode().iloc[0],
        "varianza": var,
        "desv. estándar": c.std(ddof=1),
        "mínimo": c.min(),
        "máximo": c.max(),
        "varianza/media": var / media if media > 0 else np.nan,
    }


desc = pd.DataFrame([descriptivas(c_normal, "normal"), descriptivas(c_ataque, "ataque")]).set_index("periodo")
mostrar("\n=== Medidas descriptivas (intentos por minuto) ===")
mostrar(desc.round(3).T.to_string())
mostrar("Nota: en una Poisson, varianza/media debería ser cercana a 1.")
desc.round(4).to_csv(SALIDA / "descriptivas.csv")

# ----------------------------------------------------------------------
# 4. BONDAD DE AJUSTE: ¿los conteos son Poisson?
# ----------------------------------------------------------------------


def chi2_poisson(c):
    """Chi-cuadrado de bondad de ajuste a Poisson(lambda = media muestral).
    Agrupa clases hasta que la frecuencia esperada sea >= 5."""
    n, lam, kmax = len(c), c.mean(), int(c.max())
    obs = np.bincount(c.values, minlength=kmax + 1).astype(float)
    esp = n * stats.poisson.pmf(np.arange(kmax + 1), lam)
    esp[-1] += n * stats.poisson.sf(kmax, lam)  # cola derecha en la última clase
    og, eg, ao, ae = [], [], 0.0, 0.0
    for o, e in zip(obs, esp):
        ao, ae = ao + o, ae + e
        if ae >= 5:
            og.append(ao), eg.append(ae)
            ao = ae = 0.0
    if ae > 0:  # sobrante: se une a la última clase
        if og:
            og[-1] += ao
            eg[-1] += ae
        else:
            og.append(ao), eg.append(ae)
    og, eg = np.array(og), np.array(eg)
    gl = len(og) - 1 - 1  # k clases - 1 - 1 parámetro estimado
    if gl < 1:
        return lam, np.nan, gl, np.nan
    chi2 = ((og - eg) ** 2 / eg).sum()
    return lam, chi2, gl, stats.chi2.sf(chi2, gl)


def indice_dispersion(c):
    """H0: Poisson. Estadístico (n-1)*S^2/media ~ chi2(n-1). Bilateral."""
    n, media, var = len(c), c.mean(), c.var(ddof=1)
    est = (n - 1) * var / media
    p = 2 * min(stats.chi2.cdf(est, n - 1), stats.chi2.sf(est, n - 1))
    return est, min(p, 1.0)


mostrar("\n=== Bondad de ajuste a Poisson (H0: los conteos siguen una Poisson) ===")
for nombre, c in (("normal", c_normal), ("ataque", c_ataque)):
    lam, chi2, gl, p = chi2_poisson(c)
    est_d, p_d = indice_dispersion(c)
    mostrar(f"[{nombre}] lambda = {lam:.3f}")
    mostrar(f"   Chi-cuadrado: X2 = {chi2:.3f}, gl = {gl}, valor p = {p:.4g} -> "
            + ("se rechaza H0" if p < ALFA else "no se rechaza H0"))
    mostrar(f"   Índice de dispersión: estadístico = {est_d:.1f}, valor p = {p_d:.4g} -> "
            + ("se rechaza H0" if p_d < ALFA else "no se rechaza H0"))

# ----------------------------------------------------------------------
# 5. PRUEBA DE HIPÓTESIS: ¿la media de intentos por minuto es mayor en ataque?
# ----------------------------------------------------------------------
mostrar("\n=== Intervalos de confianza del 95% para la media ===")
for nombre, c in (("normal", c_normal), ("ataque", c_ataque)):
    ic = stats.t.interval(0.95, len(c) - 1, loc=c.mean(), scale=stats.sem(c))
    mostrar(f"[{nombre}] media = {c.mean():.3f}, IC 95% = ({ic[0]:.3f}, {ic[1]:.3f})")

mostrar("\n=== Comparación de medias (H0: mu_normal = mu_ataque ; H1: mu_ataque > mu_normal) ===")
t, p_t = stats.ttest_ind(c_ataque, c_normal, equal_var=False, alternative="greater")
mostrar(f"t de Welch      : t = {t:.3f}, valor p = {p_t:.4g} -> "
        + ("se rechaza H0" if p_t < ALFA else "no se rechaza H0"))
u, p_u = stats.mannwhitneyu(c_ataque, c_normal, alternative="greater")
mostrar(f"Mann-Whitney U  : U = {u:.1f}, valor p = {p_u:.4g} -> "
        + ("se rechaza H0" if p_u < ALFA else "no se rechaza H0")
        + "  (no exige normalidad; sirve de respaldo)")

# ----------------------------------------------------------------------
# 6. UMBRAL DE ALERTA con el modelo Poisson del periodo normal
# ----------------------------------------------------------------------
lam_n = c_normal.mean()
umbral = int(stats.poisson.isf(ALFA_ALERTA, lam_n))  # alerta si intentos > umbral
falsas = (c_normal > umbral).mean()
detect = (c_ataque > umbral).mean()
mostrar("\n=== Umbral de alerta ===")
mostrar(f"Con lambda normal = {lam_n:.3f} y P(falsa alarma) <= {ALFA_ALERTA}:")
mostrar(f"   Se genera alerta si hay MÁS de {umbral} intentos en un minuto.")
mostrar(f"   Falsas alarmas observadas en el periodo normal: {falsas:.2%} de los minutos")
mostrar(f"   Minutos de ataque detectados                  : {detect:.2%}")

# ----------------------------------------------------------------------
# 7. VARIABLE CONTINUA
# ----------------------------------------------------------------------
tiene_segundos = bool((ssh_normal["Timestamp"].dt.second != 0).any())
mostrar("\n=== Variable continua ===")
if tiene_segundos:
    mostrar("Tiempo entre intentos (s) - H0: sigue una exponencial")
    for nombre, d in (("normal", ssh_normal), ("ataque", ssh_ataque)):
        dt = d["Timestamp"].sort_values().diff().dt.total_seconds().dropna()
        dt = dt[dt > 0]
        ks = stats.kstest(dt, "expon", args=(0, dt.mean()))
        mostrar(f"[{nombre}] n = {len(dt)}, media = {dt.mean():.3f} s, KS = {ks.statistic:.3f}, "
                f"valor p = {ks.pvalue:.4g}")
    mostrar("(El valor p del KS es aproximado porque la media se estima con los datos.)")
else:
    mostrar("Las marcas de tiempo vienen redondeadas al minuto: no se puede calcular el tiempo")
    mostrar("entre intentos. Se usa la duración del flujo (segundos) como variable continua.")
    dur_n = ssh_normal["Flow Duration"].dropna() / 1e6
    dur_a = ssh_ataque["Flow Duration"].dropna() / 1e6
    for nombre, d in (("normal", dur_n), ("ataque", dur_a)):
        mostrar(f"[{nombre}] n = {len(d)}, media = {d.mean():.3f} s, mediana = {d.median():.3f} s, "
                f"desv. = {d.std():.3f} s")
    u2, p2 = stats.mannwhitneyu(dur_a, dur_n, alternative="two-sided")
    mostrar(f"Mann-Whitney (duraciones distintas entre periodos): U = {u2:.1f}, valor p = {p2:.4g}")

# ----------------------------------------------------------------------
# 8. GRÁFICOS
# ----------------------------------------------------------------------


def hist_poisson(c, nombre, archivo):
    lam = c.mean()
    ks = np.arange(0, int(c.max()) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(c, bins=np.arange(-0.5, c.max() + 1.5, max(1, int(c.max() // 40) or 1)),
            density=True, alpha=0.6, label="Observado")
    ax.plot(ks, stats.poisson.pmf(ks, lam), "o-", ms=3, color="crimson",
            label=f"Poisson (λ = {lam:.2f})")
    ax.set_title(f"Intentos SSH por minuto - periodo {nombre}")
    ax.set_xlabel("Intentos por minuto")
    ax.set_ylabel("Frecuencia relativa")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDA / archivo, dpi=150)
    plt.close(fig)


hist_poisson(c_normal, "normal", "hist_normal.png")
hist_poisson(c_ataque, "de ataque", "hist_ataque.png")

fig, ax = plt.subplots(figsize=(5, 4))
ax.boxplot([c_normal, c_ataque], tick_labels=["Normal", "Ataque"])
ax.set_ylabel("Intentos por minuto")
ax.set_title("Comparación de periodos")
fig.tight_layout()
fig.savefig(SALIDA / "boxplot.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(range(len(c_ataque)), c_ataque.values, color="crimson")
ax.axhline(umbral, ls="--", color="k", label=f"Umbral = {umbral}")
ax.set_xlabel("Minuto desde el inicio del ataque")
ax.set_ylabel("Intentos")
ax.set_title("Intentos por minuto durante el ataque y umbral de alerta")
ax.legend()
fig.tight_layout()
fig.savefig(SALIDA / "ataque_vs_umbral.png", dpi=150)
plt.close(fig)

(SALIDA / "resultados.txt").write_text("\n".join(informe), encoding="utf-8")
mostrar(f"\nListo. Resultados y gráficos guardados en: {SALIDA.resolve()}")