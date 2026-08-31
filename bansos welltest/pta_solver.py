"""
Standalone Gaver-Stehfest pressure transient solver -- versi interaktif.

Backend ini diekstrak dari pressure_model.py (aplikasi Streamlit RadialPTA),
disederhanakan supaya bisa langsung dijalankan lewat terminal tanpa Streamlit.

Alur pakainya:
  1. Double-click run.cmd -> terminal terbuka.
  2. Untuk setiap parameter, terminal menampilkan nilai rekomendasi (default).
     Tinggal tekan Enter untuk pakai nilai itu, atau ketik angka sendiri.
  3. Setelah semua parameter terisi, solver Gaver-Stehfest jalan dan
     menghasilkan 3 gambar terpisah (Cartesian, Semi-log, Diagnostic log-log),
     masing-masing disimpan sebagai PNG dan ditampilkan lewat matplotlib
     (jendelanya punya tombol save sendiri kalau mau disimpan ulang/di-crop).
"""

import os
from math import factorial

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kve


# ---------------------------------------------------------------------------
# Nilai rekomendasi (default) -- sama seperti aplikasi RadialPTA
# ---------------------------------------------------------------------------
DEFAULTS = {
    "k": 75.0,        # permeabilitas (mD)
    "phi": 0.18,      # porositas (fraksi)
    "ct": 1.5e-5,     # total compressibility (psi^-1)
    "mu": 1.20,       # viskositas (cP)
    "bo": 1.15,       # formation volume factor (rb/STB)
    "h": 60.0,        # ketebalan net (ft)
    "rw": 0.328,      # radius sumur (ft)
    "q": 450.0,       # laju alir (STB/day)
    "pi": 3500.0,     # tekanan awal (psi)
    "t_min": 0.001,   # waktu minimum (jam)
    "t_max": 240.0,   # waktu maksimum (jam)
    "n_terms": 12,    # jumlah suku Gaver-Stehfest (genap, 6-18)
}

LABELS = {
    "k": "Permeabilitas k (mD)",
    "phi": "Porositas phi (fraksi, 0-1)",
    "ct": "Kompresibilitas total ct (psi^-1)",
    "mu": "Viskositas mu (cP)",
    "bo": "Formation volume factor Bo (rb/STB)",
    "h": "Ketebalan net h (ft)",
    "rw": "Radius sumur rw (ft)",
    "q": "Laju alir q (STB/day)",
    "pi": "Tekanan awal pi (psi)",
    "t_min": "Waktu minimum (jam)",
    "t_max": "Waktu maksimum (jam)",
    "n_terms": "Jumlah suku Gaver-Stehfest N (genap, 6-18)",
}


# ---------------------------------------------------------------------------
# Backend Gaver-Stehfest (murni matematis, tidak bergantung pada UI apa pun)
# ---------------------------------------------------------------------------
def stehfest_weights(n_terms: int = 12) -> np.ndarray:
    """Bobot Gaver-Stehfest Vj untuk n_terms genap."""
    if n_terms < 6 or n_terms > 18 or n_terms % 2:
        raise ValueError("n_terms harus genap, antara 6 dan 18")

    half = n_terms // 2
    weights = np.zeros(n_terms, dtype=float)
    for k in range(1, n_terms + 1):
        total = 0.0
        for j in range((k + 1) // 2, min(k, half) + 1):
            numerator = (j ** half) * factorial(2 * j)
            denominator = (
                factorial(half - j)
                * factorial(j)
                * factorial(j - 1)
                * factorial(k - j)
                * factorial(2 * j - k)
            )
            total += numerator / denominator
        weights[k - 1] = ((-1) ** (k + half)) * total
    return weights


def laplace_wellbore_pressure(u) -> np.ndarray:
    """Solusi pressure di wellbore pada ruang Laplace: K0(sqrt(u)) / [u^1.5 K1(sqrt(u))]."""
    u_array = np.asarray(u, dtype=float)
    root_u = np.sqrt(u_array)
    bessel_ratio = kve(0, root_u) / kve(1, root_u)
    return bessel_ratio / np.power(u_array, 1.5)


def invert_pressure_gaver_stehfest(dimensionless_time, n_terms: int = 12) -> np.ndarray:
    """Inversi Laplace -> pD(tD) memakai algoritma Gaver-Stehfest."""
    td = np.atleast_1d(np.asarray(dimensionless_time, dtype=float))
    weights = stehfest_weights(n_terms)
    indices = np.arange(1, n_terms + 1, dtype=float)
    u = np.log(2.0) * indices[None, :] / td[:, None]
    pbar = laplace_wellbore_pressure(u)
    pd = (np.log(2.0) / td) * np.sum(weights[None, :] * pbar, axis=1)
    return pd


def dimensionless_time(time_hours: np.ndarray, p: dict) -> np.ndarray:
    """Konversi waktu lapangan (jam) -> waktu dimensionless tD."""
    return (
        0.0002637 * p["k"] * np.asarray(time_hours, dtype=float)
        / (p["phi"] * p["mu"] * p["ct"] * p["rw"] ** 2)
    )


def simulate_drawdown(time_hours: np.ndarray, p: dict) -> dict:
    """Simulasi drawdown constant-rate + logarithmic derivative-nya."""
    td = dimensionless_time(time_hours, p)
    pd = invert_pressure_gaver_stehfest(td, n_terms=p["n_terms"])

    pressure_scale = 141.2 * p["q"] * p["bo"] * p["mu"] / (p["k"] * p["h"])
    delta_pressure = pressure_scale * pd
    flowing_pressure = p["pi"] - delta_pressure
    derivative = np.gradient(delta_pressure, np.log(time_hours), edge_order=2)

    return {
        "time_hours": time_hours,
        "dimensionless_time": td,
        "dimensionless_pressure": pd,
        "pressure_drop_psi": delta_pressure,
        "flowing_pressure_psi": flowing_pressure,
        "pressure_derivative_psi": derivative,
    }


# ---------------------------------------------------------------------------
# Input interaktif dari terminal, dengan nilai rekomendasi siap pakai
# ---------------------------------------------------------------------------
def ask_value(key: str) -> float:
    default = DEFAULTS[key]
    label = LABELS[key]
    while True:
        raw = input(f"  {label} [default {default:g}]: ").strip()
        if raw == "":
            return default
        try:
            value = float(raw)
        except ValueError:
            print("    -> input tidak valid, harus berupa angka. Coba lagi.")
            continue
        return value


def get_inputs() -> dict:
    print("=" * 70)
    print(" GAVER-STEHFEST PTA SOLVER - input parameter")
    print(" Tekan ENTER saja untuk pakai nilai default (rekomendasi).")
    print("=" * 70)

    p = {}
    for key in ["k", "phi", "ct", "mu", "bo", "h", "rw", "q", "pi", "t_min", "t_max", "n_terms"]:
        p[key] = ask_value(key)

    # jaga-jaga: n_terms harus genap & dalam rentang valid, t_max > t_min
    n_terms = int(round(p["n_terms"]))
    if n_terms % 2:
        n_terms += 1
    n_terms = min(max(n_terms, 6), 18)
    p["n_terms"] = n_terms

    if p["t_max"] <= p["t_min"]:
        print("  -> t_max harus lebih besar dari t_min, pakai default keduanya.")
        p["t_min"] = DEFAULTS["t_min"]
        p["t_max"] = DEFAULTS["t_max"]

    print("-" * 70)
    print(" Parameter yang dipakai:")
    for key in p:
        print(f"   {LABELS[key]:<40s} = {p[key]:g}")
    print("-" * 70)
    return p


# ---------------------------------------------------------------------------
# Plot: 3 gambar terpisah (Cartesian, Semi-log, Diagnostic log-log)
# ---------------------------------------------------------------------------
def main() -> None:
    p = get_inputs()

    time_hours = np.geomspace(p["t_min"], p["t_max"], 160)
    result = simulate_drawdown(time_hours, p)

    pwf = result["flowing_pressure_psi"]
    dp = result["pressure_drop_psi"]
    deriv = result["pressure_derivative_psi"]
    deriv_plot = np.where(deriv > 0, deriv, np.nan)

    subtitle = (
        f"N={p['n_terms']}  k={p['k']:g} mD  q={p['q']:g} STB/d  pi={p['pi']:g} psi"
    )
    out_dir = os.path.dirname(os.path.abspath(__file__))
    saved_paths = []

    # 1. Cartesian
    fig1, ax1 = plt.subplots(figsize=(7, 5.5))
    ax1.plot(time_hours, pwf, color="#176bce", linewidth=2)
    ax1.axhline(p["pi"], color="gray", linestyle=":", linewidth=1)
    ax1.set_title(f"1. Cartesian: pwf vs t\n{subtitle}", fontsize=10)
    ax1.set_xlabel("t (jam)")
    ax1.set_ylabel("pwf (psi)")
    ax1.grid(alpha=0.3)
    fig1.tight_layout()

    # 2. Semi-log
    fig2, ax2 = plt.subplots(figsize=(7, 5.5))
    ax2.plot(time_hours, pwf, color="#176bce", linewidth=2)
    ax2.axhline(p["pi"], color="gray", linestyle=":", linewidth=1)
    ax2.set_xscale("log")
    ax2.set_title(f"2. Semi-log: pwf vs log(t)\n{subtitle}", fontsize=10)
    ax2.set_xlabel("t (jam)")
    ax2.set_ylabel("pwf (psi)")
    ax2.grid(alpha=0.3, which="both")
    fig2.tight_layout()

    # 3. Diagnostic log-log (Delta-p & derivative)
    fig3, ax3 = plt.subplots(figsize=(7, 5.5))
    ax3.plot(time_hours, dp, color="#176bce", linewidth=2, label="Delta-p")
    ax3.plot(time_hours, deriv_plot, color="#d24646", linewidth=2, linestyle="--", label="dDelta-p/dln(t)")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_title(f"3. Diagnostic log-log: Delta-p & derivative vs t\n{subtitle}", fontsize=10)
    ax3.set_xlabel("t (jam)")
    ax3.set_ylabel("psi")
    ax3.legend()
    ax3.grid(alpha=0.3, which="both")
    fig3.tight_layout()

    figures = [
        (fig1, "1_cartesian.png"),
        (fig2, "2_semilog.png"),
        (fig3, "3_diagnostic.png"),
    ]
    for fig, filename in figures:
        out_path = os.path.join(out_dir, filename)
        fig.savefig(out_path, dpi=150)
        saved_paths.append(out_path)

    print("\nSelesai. 3 gambar tersimpan otomatis di folder ini:")
    for out_path in saved_paths:
        print(f"  - {out_path}")
    print("\nJendela plot akan terbuka. Tiap jendela punya tombol save (ikon disket)")
    print("kalau mau menyimpan ulang / ganti format / crop area tertentu.")

    plt.show()


if __name__ == "__main__":
    main()
