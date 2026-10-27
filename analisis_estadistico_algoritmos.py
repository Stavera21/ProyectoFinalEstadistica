"""
Proyecto Estadística I: comparación de algoritmos de ordenamiento
Literal C: análisis estadístico reproducible.

Antes de ejecutar:
1. Descargue Online Retail desde:
   https://archive.ics.uci.edu/dataset/352/online+retail
2. Coloque el archivo xlsx en la misma carpeta que este script.
3. Instale dependencias:
   python -m pip install pandas numpy matplotlib scipy openpyxl
4. Ejecute:
   python analisis_estadistico_algoritmos.py

IMPORTANTE:
- El conjunto público aporta los valores de entrada.
- Los tiempos son mediciones nuevas obtenidas al ejecutar este script.
- No cambie los datos para forzar una distribución.
- Los tamaños/repeticiones se pueden ajustar después de una prueba piloto.
"""

from pathlib import Path
import time
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

SEED = 20261027
SIZES = [100, 250, 500, 1000]
SAMPLES_PER_SIZE = 20
TECHNICAL_REPEATS = 3
ALPHA = 0.05
DATA_FILE_CANDIDATES = [
    "Online Retail.xlsx",
    "online_retail.xlsx",
    "OnlineRetail.xlsx",
]
OUT = Path("resultados_algoritmos")
OUT.mkdir(exist_ok=True)

rng = random.Random(SEED)


# ---------- Algoritmos: solo se cronometra el ordenamiento ----------

def bubble_sort(values):
    a = values.copy()
    n = len(a)
    for end in range(n - 1, 0, -1):
        swapped = False
        for i in range(end):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                swapped = True
        if not swapped:
            break
    return a


def insertion_sort(values):
    a = values.copy()
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


def merge_sort(values):
    # Se crea una nueva lista; el coste de la copia inicial se mantiene
    # igual para las tres funciones, pero se excluye del cronómetro.
    if len(values) <= 1:
        return values.copy()
    mid = len(values) // 2
    left = merge_sort(values[:mid])
    right = merge_sort(values[mid:])
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


ALGORITHMS = {
    "Bubble Sort": bubble_sort,
    "Insertion Sort": insertion_sort,
    "Merge Sort": merge_sort,
}


def locate_data_file():
    for name in DATA_FILE_CANDIDATES:
        p = Path(name)
        if p.exists():
            return p
    xlsx_files = list(Path(".").glob("*.xlsx"))
    if len(xlsx_files) == 1:
        return xlsx_files[0]
    raise FileNotFoundError(
        "No encuentro el archivo Online Retail .xlsx. Descárguelo de UCI "
        "y colóquelo junto a este script con el nombre 'Online Retail.xlsx'."
    )


def load_values():
    data_path = locate_data_file()
    df = pd.read_excel(data_path)
    if "Quantity" not in df.columns:
        raise ValueError(
            f"No encuentro la columna Quantity. Columnas detectadas: {list(df.columns)}"
        )
    # No se cambian ni se eliminan los valores de Quantity.
    values = pd.to_numeric(df["Quantity"], errors="raise").dropna().tolist()
    if len(values) < max(SIZES):
        raise ValueError("El archivo tiene menos datos que el tamaño máximo configurado.")
    print(f"Fuente de entrada: {data_path.resolve()}")
    print(f"Filas leídas: {len(df)}; valores válidos de Quantity: {len(values)}")
    print("Nota: estos valores son entradas; los tiempos se medirán localmente.")
    return values


def timed_call(fn, input_values):
    # Se hace la copia antes de iniciar el cronómetro para medir el algoritmo
    # y no el coste de copiar la entrada.
    input_copy = input_values.copy()
    start = time.perf_counter()
    result = fn(input_copy)
    elapsed = time.perf_counter() - start
    if result != sorted(input_values):
        raise RuntimeError(f"{fn.__name__} produjo una salida incorrecta.")
    return elapsed


def collect_measurements(values):
    rows = []
    for n in SIZES:
        print(f"\nProcesando tamaño n={n} ...")
        for sample_num in range(1, SAMPLES_PER_SIZE + 1):
            sample = rng.sample(values, n)
            # La misma muestra se usa para todos los algoritmos.
            # Se alterna el orden de ejecución para reducir sesgo sistemático
            # por calentamiento/carga del equipo.
            algo_names = list(ALGORITHMS.keys())
            rng.shuffle(algo_names)
            for algo_name in algo_names:
                fn = ALGORITHMS[algo_name]
                timings = []
                for rep in range(1, TECHNICAL_REPEATS + 1):
                    elapsed = timed_call(fn, sample)
                    timings.append(elapsed)
                    rows.append({
                        "sample_id": f"n{n}_s{sample_num:02d}",
                        "input_size": n,
                        "algorithm": algo_name,
                        "repeat_id": rep,
                        "runtime_seconds": elapsed,
                        "runtime_milliseconds": elapsed * 1000,
                        "correctness_check": True,
                    })
                print(
                    f"  muestra {sample_num:02d}, {algo_name}: "
                    f"mediana={np.median(timings)*1000:.4f} ms"
                )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mediciones_individuales.csv", index=False)
    # Resumen por muestra/algoritmo: mediana de las repeticiones técnicas.
    per_sample = (
        df.groupby(["sample_id", "input_size", "algorithm"], as_index=False)
          .agg(runtime_seconds=("runtime_seconds", "median"),
               runtime_milliseconds=("runtime_milliseconds", "median"),
               technical_repeats=("repeat_id", "count"))
    )
    per_sample.to_csv(OUT / "medianas_por_muestra.csv", index=False)
    return df, per_sample


def descriptive_analysis(per_sample):
    summary = (
        per_sample.groupby(["input_size", "algorithm"])["runtime_milliseconds"]
        .agg(
            n="count",
            mean_ms="mean",
            median_ms="median",
            std_ms="std",
            min_ms="min",
            max_ms="max",
        )
        .reset_index()
    )
    summary["variance_ms2"] = (
        per_sample.groupby(["input_size", "algorithm"])["runtime_milliseconds"]
        .var().to_numpy()
    )
    summary.to_csv(OUT / "resumen_descriptivo.csv", index=False)
    print("\n" + "=" * 88)
    print("ESTADÍSTICA DESCRIPTIVA: ¿CUÁNTO TARDAN LOS ALGORITMOS?".center(88))
    print("=" * 88)
    print("Variables: input_size = cantidad de elementos; runtime_milliseconds = tiempo de ejecución.")
    print("mean_ms es el promedio de tiempos; median_ms es el valor central; std_ms mide dispersión.")
    print("min_ms y max_ms muestran los extremos observados; variance_ms2 mide variabilidad al cuadrado.")
    print("Se interpretan los tiempos de cada algoritmo por separado para cada tamaño de entrada.")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return summary


def plot_results(per_sample, summary):
    print("\n" + "=" * 88)
    print("GRÁFICOS DESCRIPTIVOS".center(88))
    print("=" * 88)
    print("Gráfico 1: compara tiempo medio y tamaño de entrada. Gráfico 2: muestra mediana, dispersión y posibles valores extremos.")
    print("Gráfico 3: muestra la distribución empírica de los tiempos para n máximo. Los gráficos describen, no prueban hipótesis por sí solos.")
    # 1. Tiempo promedio vs tamaño
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for algo in ALGORITHMS:
        part = summary[summary["algorithm"] == algo]
        ax.plot(part["input_size"], part["mean_ms"], marker="o", label=algo)
    ax.set_xlabel("Tamaño de entrada (n)")
    ax.set_ylabel("Tiempo medio (ms)")
    ax.set_title("Tiempo de ejecución frente al tamaño de entrada")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "01_tiempo_vs_tamano.png", dpi=160)
    plt.close(fig)

    # 2. Boxplots por algoritmo para cada tamaño
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, n in zip(axes.flat, SIZES):
        data = [
            per_sample.loc[
                (per_sample["input_size"] == n) &
                (per_sample["algorithm"] == algo),
                "runtime_milliseconds"
            ].to_numpy()
            for algo in ALGORITHMS
        ]
        ax.boxplot(data, tick_labels=list(ALGORITHMS.keys()))
        ax.set_title(f"n = {n}")
        ax.set_ylabel("Tiempo (ms)")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Distribución empírica de tiempos por algoritmo y tamaño")
    fig.tight_layout()
    fig.savefig(OUT / "02_diagramas_caja.png", dpi=160)
    plt.close(fig)

    # 3. Histogramas por algoritmo en el tamaño máximo
    n = max(SIZES)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, algo in zip(axes, ALGORITHMS):
        vals = per_sample.loc[
            (per_sample["input_size"] == n) &
            (per_sample["algorithm"] == algo),
            "runtime_milliseconds"
        ].to_numpy()
        ax.hist(vals, bins="auto", edgecolor="black")
        ax.set_title(algo)
        ax.set_xlabel("Tiempo (ms)")
        ax.set_ylabel("Frecuencia")
    fig.suptitle(f"Histogramas de tiempos para n = {n}")
    fig.tight_layout()
    fig.savefig(OUT / "03_histogramas_tiempos.png", dpi=160)
    plt.close(fig)


def paired_comparisons(per_sample):
    """
    Comparaciones emparejadas: los algoritmos comparten sample_id.
    La prueba t pareada se aplica a diferencias por muestra.
    Shapiro-Wilk se informa como diagnóstico de normalidad de las diferencias,
    no como prueba de que los datos 'deban' ser normales.
    """
    print("\n" + "=" * 88)
    print("PRUEBAS DE HIPÓTESIS PAREADAS E INTERVALOS DE CONFIANZA".center(88))
    print("=" * 88)
    print("Pregunta: ¿la diferencia media de tiempo entre dos algoritmos es distinta de cero?")
    print("H0: media(D)=0. H1: media(D)≠0, donde D = tiempo_A - tiempo_B en la misma muestra.")
    print("La comparación es pareada porque ambos algoritmos ordenan exactamente la misma muestra.")
    print(f"Nivel de significación alpha={ALPHA:.2f}. Un valor p menor que alpha aporta evidencia de diferencia media.")
    print("El IC del 95% estima un rango plausible para la diferencia media; si incluye 0, no indica diferencia al nivel bilateral del 5% bajo el mismo modelo.")
    rows = []
    for n in SIZES:
        part = per_sample[per_sample["input_size"] == n]
        wide = part.pivot(index="sample_id", columns="algorithm",
                          values="runtime_milliseconds")
        pairs = [
            ("Bubble Sort", "Insertion Sort"),
            ("Bubble Sort", "Merge Sort"),
            ("Insertion Sort", "Merge Sort"),
        ]
        for a, b in pairs:
            differences = (wide[a] - wide[b]).dropna().to_numpy()
            if len(differences) < 3:
                continue
            shapiro_p = np.nan
            if len(differences) >= 3:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    shapiro_p = stats.shapiro(differences).pvalue
            t_res = stats.ttest_rel(wide[a], wide[b], nan_policy="omit")
            mean_diff = float(np.mean(differences))
            sd_diff = float(np.std(differences, ddof=1))
            se = sd_diff / np.sqrt(len(differences))
            tcrit = stats.t.ppf(0.975, df=len(differences)-1)
            ci_low = mean_diff - tcrit * se
            ci_high = mean_diff + tcrit * se
            rows.append({
                "input_size": n,
                "comparison": f"{a} - {b}",
                "n_pairs": len(differences),
                "mean_difference_ms": mean_diff,
                "ci95_low_ms": ci_low,
                "ci95_high_ms": ci_high,
                "paired_t_statistic": float(t_res.statistic),
                "p_value_paired_t": float(t_res.pvalue),
                "shapiro_p_differences": float(shapiro_p),
                "interpretation_note": (
                    "p < 0.05 indica evidencia de diferencia media bajo los supuestos; "
                    "no prueba causalidad ni garantiza relevancia práctica."
                ),
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "comparaciones_pareadas.csv", index=False)
    print("\nRESULTADOS: cada fila compara un par de algoritmos para un tamaño de entrada.")
    print("mean_difference_ms > 0 significa que A tardó más en promedio que B; < 0 significa que A tardó menos.")
    print("shapiro_p_differences diagnostica normalidad de las diferencias; no prueba igualdad de medias.")
    if not result.empty:
        print(result.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\nBONDAD DE AJUSTE: ANDERSON-DARLING PARA LAS DIFERENCIAS")
    print("Se contrasta si las diferencias D son compatibles con una distribución normal.")
    print("H0: las diferencias siguen una distribución normal; H1: no siguen una distribución normal.")
    print("La prueba de Anderson-Darling entrega valores críticos, no un valor p directo en esta implementación.")
    for n in SIZES:
        part = per_sample[per_sample["input_size"] == n]
        wide = part.pivot(index="sample_id", columns="algorithm", values="runtime_milliseconds")
        for a, b in [("Bubble Sort", "Insertion Sort"), ("Bubble Sort", "Merge Sort"), ("Insertion Sort", "Merge Sort")]:
            differences = (wide[a] - wide[b]).dropna().to_numpy()
            if len(differences) < 5:
                print(f"n={n}, {a} - {b}: no hay suficientes diferencias para este diagnóstico.")
                continue
            ad = stats.anderson(differences, dist="norm")
            idx = int(np.argmin(np.abs(ad.significance_level - 5.0)))
            critical = float(ad.critical_values[idx])
            if ad.statistic > critical:
                conclusion = "Se rechaza la normalidad al nivel aproximado del 5%; interpretar con cautela la prueba t pareada."
            else:
                conclusion = "No se rechaza la normalidad al nivel aproximado del 5%; esto no demuestra normalidad perfecta."
            print(f"n={n}, {a} - {b}: estadístico AD={ad.statistic:.6g}; valor crítico 5%={critical:.6g}. {conclusion}")
    return result


def distribution_diagnostics(per_sample):
    """
    Diagnóstico de normalidad de los tiempos para el tamaño mayor.
    Shapiro-Wilk es un diagnóstico complementario, no una orden para forzar
    los datos a normalidad. En muestras pequeñas tiene potencia limitada.
    """
    n = max(SIZES)
    rows = []
    for algo in ALGORITHMS:
        vals = per_sample.loc[
            (per_sample["input_size"] == n) &
            (per_sample["algorithm"] == algo),
            "runtime_milliseconds"
        ].dropna().to_numpy()
        if 3 <= len(vals) <= 5000:
            res = stats.shapiro(vals)
            rows.append({
                "input_size": n,
                "algorithm": algo,
                "n_observations": len(vals),
                "shapiro_statistic": float(res.statistic),
                "shapiro_p_value": float(res.pvalue),
                "note": (
                    "Diagnóstico de compatibilidad con normalidad; no es necesario "
                    "que los tiempos originales sean normales para toda estadística."
                ),
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "diagnostico_distribucion.csv", index=False)
    print("\n" + "=" * 88)
    print("DIAGNÓSTICO DE NORMALIDAD DE LOS TIEMPOS INDIVIDUALES".center(88))
    print("=" * 88)
    print("Se revisan los tiempos individuales de cada algoritmo para el tamaño máximo.")
    print("H0: los tiempos son compatibles con normalidad; H1: no son compatibles con normalidad.")
    print("Si p < 0.05 se rechaza H0; si p >= 0.05, no hay evidencia suficiente para rechazarla.")
    print("No rechazar H0 no demuestra normalidad, y los tiempos individuales no sustituyen revisar las diferencias para la prueba pareada.")
    if not result.empty:
        print(result.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    return result


def main():
    print("=" * 88)
    print("PROYECTO DE ESTADÍSTICA I — LITERAL C".center(88))
    print("Comparación de tiempos de Bubble Sort, Insertion Sort y Merge Sort".center(88))
    print("=" * 88)
    print("La columna Quantity de Online Retail aporta los valores que se ordenan.")
    print("Las variables de respuesta son los tiempos medidos localmente, en milisegundos.")
    print("No se transforman ni eliminan valores para forzar una distribución estadística.")
    print("Las secciones del análisis se separan mediante print; todo el procedimiento está en este archivo Python.")
    print("\nDISEÑO EXPERIMENTAL")
    print(f"Tamaños: {SIZES}; muestras independientes por tamaño: {SAMPLES_PER_SIZE}; repeticiones técnicas por algoritmo y muestra: {TECHNICAL_REPEATS}.")
    print("En cada muestra se ejecutan los tres algoritmos sobre los mismos valores, lo que permite comparaciones pareadas.")
    print("Se usa la mediana de las repeticiones técnicas como tiempo representativo por muestra, reduciendo el efecto de una medición aislada.")
    values = load_values()
    raw, per_sample = collect_measurements(values)
    summary = descriptive_analysis(per_sample)
    plot_results(per_sample, summary)
    paired = paired_comparisons(per_sample)
    diagnostics = distribution_diagnostics(per_sample)

    print("\n" + "=" * 88)
    print("RESUMEN DE DECISIONES Y LIMITACIONES".center(88))
    print("=" * 88)
    print("1. Descriptivos: permiten comparar centro y variabilidad de los tiempos, pero no prueban diferencias por sí solos.")
    print("2. Shapiro-Wilk: diagnostica normalidad de los tiempos para n máximo; no determina por sí solo qué prueba comparativa usar.")
    print("3. Anderson-Darling: evalúa la compatibilidad de las diferencias emparejadas con normalidad, como diagnóstico relacionado con el supuesto de la prueba t pareada.")
    print("4. Prueba t pareada: contrasta si la diferencia media de tiempos es cero. Si la normalidad de las diferencias se cuestiona, la conclusión debe expresarse con cautela.")
    print("5. Intervalos de confianza: muestran la incertidumbre de la diferencia media; se interpretan junto al valor p y al contexto práctico.")
    print("6. No se incluye regresión lineal ni log-log porque el objetivo principal es comparar tiempos; no se añade un modelo que no sea necesario para responder la pregunta del proyecto.")
    print("7. No se añade una prueba F automáticamente: compara varianzas bajo supuestos específicos y no responde a la misma pregunta que comparar tiempos medios. Su inclusión debe justificarse por el objetivo del trabajo y lo visto en clase.")
    print("8. Se realizan múltiples comparaciones (tres pares por cuatro tamaños); los valores p deben interpretarse considerando que hay varias pruebas.")
    print("\nARCHIVOS GENERADOS")
    print("Todos los archivos de resultados se guardan dentro de:", OUT.resolve())
    print("- mediciones_individuales.csv: cada medición técnica")
    print("- medianas_por_muestra.csv: una mediana por algoritmo y muestra")
    print("- resumen_descriptivo.csv: medias, medianas, dispersión y extremos")
    print("- comparaciones_pareadas.csv: prueba t, intervalos y Shapiro-Wilk sobre diferencias")
    print("- diagnostico_distribucion.csv: Shapiro-Wilk sobre tiempos individuales")
    print("- tres gráficos PNG descriptivos")
    print("\nFin del análisis.")


if __name__ == "__main__":
    main()
