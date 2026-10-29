"""
Literal C: análisis estadístico reproducible.

IMPORTANTE:
- El conjunto público aporta los valores de entrada.
- Los tiempos son mediciones nuevas obtenidas al ejecutar este script.
- No cambiar los datos para forzar una distribución.
- La bondad de ajuste (KS) se realiza en Excel con @RISK usando datos_para_RISK.xlsx.
- Los interruptores CALCULAR_IC y CALCULAR_F permiten quitar esas secciones si no se vieron en clase.
"""

from pathlib import Path
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

SEED = 20261027
SIZES = [100, 250, 500, 1000]
SAMPLES_PER_SIZE = 100      # muestras independientes por tamaño (antes 20)
TECHNICAL_REPEATS = 3
ALPHA = 0.05
CALCULAR_IC = True   # intervalos de confianza de la media (t de Student)
CALCULAR_F = True    # prueba F de igualdad de varianzas entre pares de algoritmos
DATA_FILE_CANDIDATES = [
    "Online Retail.xlsx",
    "online_retail.xlsx",
    "OnlineRetail.xlsx",
]
OUT = Path("resultados_algoritmos")
OUT.mkdir(exist_ok=True)

rng = random.Random(SEED)

SHORT = {"Bubble Sort": "Bubble", "Insertion Sort": "Insertion", "Merge Sort": "Merge"}


def save_excel(df, filename, sheet_name="datos"):
    """Guarda un DataFrame como .xlsx dentro de la carpeta de resultados."""
    df.to_excel(OUT / filename, index=False, sheet_name=sheet_name)


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
                        "sample_id": f"n{n}_s{sample_num:03d}",
                        "input_size": n,
                        "algorithm": algo_name,
                        "repeat_id": rep,
                        "runtime_seconds": elapsed,
                        "runtime_milliseconds": elapsed * 1000,
                        "correctness_check": True,
                    })
            if sample_num % 10 == 0:
                print(f"  {sample_num}/{SAMPLES_PER_SIZE} muestras listas")
    df = pd.DataFrame(rows)
    save_excel(df, "mediciones_individuales.xlsx", "mediciones")
    # Resumen por muestra/algoritmo: mediana de las repeticiones técnicas.
    per_sample = (
        df.groupby(["sample_id", "input_size", "algorithm"], as_index=False)
          .agg(runtime_seconds=("runtime_seconds", "median"),
               runtime_milliseconds=("runtime_milliseconds", "median"),
               technical_repeats=("repeat_id", "count"))
    )
    save_excel(per_sample, "medianas_por_muestra.xlsx", "medianas")
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
    summary["cv"] = summary["std_ms"] / summary["mean_ms"]  # dispersión relativa
    save_excel(summary, "resumen_descriptivo.xlsx", "resumen")
    print("\n" + "=" * 88)
    print("ESTADÍSTICA DESCRIPTIVA: ¿CUÁNTO TARDAN LOS ALGORITMOS?".center(88))
    print("=" * 88)
    print("Variables: input_size = cantidad de elementos; runtime_milliseconds = tiempo de ejecución.")
    print("mean_ms es el promedio de tiempos; median_ms es el valor central; std_ms mide dispersión.")
    print("min_ms y max_ms muestran los extremos observados; variance_ms2 mide variabilidad al cuadrado.")
    print("cv = std_ms / mean_ms es la dispersión relativa: permite comparar variabilidad entre algoritmos con tiempos muy distintos.")
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


PAIRS = [
    ("Bubble Sort", "Insertion Sort"),
    ("Bubble Sort", "Merge Sort"),
    ("Insertion Sort", "Merge Sort"),
]


def confidence_intervals(per_sample):
    """Intervalo de confianza del 95% para el tiempo medio de cada algoritmo y tamaño."""
    print("\n" + "=" * 88)
    print("INTERVALOS DE CONFIANZA PARA EL TIEMPO MEDIO".center(88))
    print("=" * 88)
    print("Para cada algoritmo y tamaño: media ± t(0.975, n-1) · s/√n, con n = número de muestras.")
    print("Interpretación: con 95% de confianza, el tiempo medio verdadero está dentro del intervalo.")
    print("Si dos intervalos no se traslapan, los tiempos medios son claramente distintos.")
    rows = []
    for n in SIZES:
        for algo in ALGORITHMS:
            x = per_sample.loc[(per_sample["input_size"] == n) &
                               (per_sample["algorithm"] == algo),
                               "runtime_milliseconds"].to_numpy()
            k = len(x)
            mean, sd = float(x.mean()), float(x.std(ddof=1))
            tcrit = float(stats.t.ppf(0.975, df=k - 1))
            half = tcrit * sd / np.sqrt(k)
            rows.append({
                "input_size": n, "algorithm": algo, "n": k, "mean_ms": mean,
                "std_ms": sd, "t_critical": tcrit,
                "ci95_low_ms": mean - half, "ci95_high_ms": mean + half,
            })
    result = pd.DataFrame(rows)
    save_excel(result, "intervalos_confianza.xlsx", "IC_95")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    return result


def variance_tests(per_sample):
    """Prueba F bilateral de igualdad de varianzas mediante valores críticos."""
    print("\\n" + "=" * 88)
    print("PRUEBA F DE IGUALDAD DE VARIANZAS".center(88))
    print("=" * 88)
    print("H0: varianza_A = varianza_B.  H1: varianza_A ≠ varianza_B (bilateral).")
    print(f"Estadístico: F = s²_A / s²_B; gl_A = n_A - 1; gl_B = n_B - 1; alpha={ALPHA:.2f}.")
    print("Regla de decisión: se rechaza H0 si F < F_inferior o F > F_superior.")
    print("Si F queda entre ambos límites críticos, no se rechaza H0.")
    print("Los límites críticos se calculan automáticamente para una prueba bilateral.")
    rows = []
    for n in SIZES:
        part = per_sample[per_sample["input_size"] == n]
        wide = part.pivot(index="sample_id", columns="algorithm", values="runtime_milliseconds")
        for a, b in PAIRS:
            xa, xb = wide[a].dropna().to_numpy(), wide[b].dropna().to_numpy()
            va, vb = float(xa.var(ddof=1)), float(xb.var(ddof=1))
            F = va / vb
            df1, df2 = len(xa) - 1, len(xb) - 1

            # Valores críticos para una prueba bilateral con alpha = 0.05.
            # Se reparte alpha/2 en cada cola de la distribución F.
            f_lo = float(stats.f.ppf(ALPHA / 2, df1, df2))
            f_hi = float(stats.f.ppf(1 - ALPHA / 2, df1, df2))

            rows.append({
                "input_size": n,
                "comparison": f"{a} / {b}",
                "var_A_ms2": va,
                "var_B_ms2": vb,
                "F": F,
                "gl_A": df1,
                "gl_B": df2,
                "F_inferior": f_lo,
                "F_superior": f_hi,
                "decision": "Se rechaza H0" if (F < f_lo or F > f_hi) else "No se rechaza H0",
            })
    result = pd.DataFrame(rows)
    save_excel(result, "prueba_F_varianzas.xlsx", "prueba_F")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    return result


def build_risk_workbook(per_sample):
    """
    Crea datos_para_RISK.xlsx: una columna por grupo (algoritmo x tamaño),
    lista para seleccionarla en @RISK (Ajustar distribuciones a datos).
    """
    times = {}
    for n in SIZES:
        for algo in ALGORITHMS:
            vals = (per_sample[(per_sample["input_size"] == n) &
                               (per_sample["algorithm"] == algo)]
                    .sort_values("sample_id")["runtime_milliseconds"].to_numpy())
            times[f"{SHORT[algo]}_n{n}"] = vals
    pd.DataFrame(times).to_excel(OUT / "datos_para_RISK.xlsx", sheet_name="tiempos", index=False)
    print("\nSe creó datos_para_RISK.xlsx: hoja 'tiempos' con 12 columnas (algoritmo y tamaño).")
    print("En @RISK se ajusta una columna a la vez; cada columna tiene", SAMPLES_PER_SIZE, "datos independientes.")


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
    print("En cada muestra se ejecutan los tres algoritmos sobre los mismos valores, para comparar los algoritmos en igualdad de condiciones.")
    print("Se usa la mediana de las repeticiones técnicas como tiempo representativo por muestra, reduciendo el efecto de una medición aislada.")
    values = load_values()
    raw, per_sample = collect_measurements(values)
    summary = descriptive_analysis(per_sample)
    plot_results(per_sample, summary)
    if CALCULAR_IC:
        confidence_intervals(per_sample)
    if CALCULAR_F:
        variance_tests(per_sample)
    build_risk_workbook(per_sample)

    print("\n" + "=" * 88)
    print("RESUMEN DE DECISIONES Y LIMITACIONES".center(88))
    print("=" * 88)
    print("1. Descriptivos: permiten comparar centro y variabilidad de los tiempos, pero no prueban diferencias por sí solos.")
    print("2. Bondad de ajuste: se realiza en Excel con @RISK (prueba KS) sobre datos_para_RISK.xlsx, un grupo por columna.")
    print("3. Intervalos de confianza: muestran la incertidumbre del tiempo medio de cada algoritmo; intervalos que no se traslapan indican tiempos medios distintos.")
    print("4. Prueba F: compara varianzas de dos algoritmos mediante F y los valores críticos; requiere los supuestos de la prueba.")
    print("5. No se incluye regresión lineal ni log-log porque el objetivo principal es comparar tiempos; no se añade un modelo que no sea necesario para responder la pregunta del proyecto.")
    print("6. No se usa la prueba t pareada porque no se ha visto en el curso.")
    print("7. Se realizan múltiples comparaciones (tres pares por cuatro tamaños); las conclusiones deben considerar que hay varias pruebas.")
    print("\nARCHIVOS GENERADOS")
    print("Todos los archivos de resultados se guardan dentro de:", OUT.resolve())
    print("- mediciones_individuales.xlsx: cada medición técnica")
    print("- medianas_por_muestra.xlsx: una mediana por algoritmo y muestra")
    print("- resumen_descriptivo.xlsx: medias, medianas, dispersión y extremos")
    print("- intervalos_confianza.xlsx: IC 95% del tiempo medio (si CALCULAR_IC = True)")
    print("- prueba_F_varianzas.xlsx: prueba F entre pares de algoritmos (si CALCULAR_F = True)")
    print("- datos_para_RISK.xlsx: tiempos, una columna por grupo, para ajustar en @RISK")
    print("- tres gráficos PNG descriptivos")
    print("\nFin del análisis.")


if __name__ == "__main__":
    main()