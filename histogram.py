import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# 1. DATA HARGA PERTAMAX RON 92
# September 2023 - Agustus 2026
# ============================================================

harga = np.array([
    13300, 14000, 13400, 13350,       # Sep-Des 2023

    12950, 12950, 12950, 12950,
    12950, 12950, 12950, 13700,
    12950, 12100, 12100, 12100,       # Jan-Des 2024

    12500, 12900, 12900, 12500,
    12400, 12100, 12500, 12200,
    12200, 12200, 12200, 12750,       # Jan-Des 2025

    12350, 11800, 12300, 12300,
    12300, 16250, 16250, 15950        # Jan-Agu 2026
])


# ============================================================
# 2. STATISTIK DESKRIPTIF
# ============================================================

N = len(harga)

mean = np.mean(harga)
median = np.median(harga)
std = np.std(harga, ddof=1)
minimum = np.min(harga)
maximum = np.max(harga)
skewness = stats.skew(harga)

print("=" * 60)
print("STATISTIK DESKRIPTIF")
print("=" * 60)

print(f"Jumlah observasi : {N}")
print(f"Mean             : Rp{mean:,.2f}/L")
print(f"Median           : Rp{median:,.2f}/L")
print(f"Std. Deviasi     : Rp{std:,.2f}/L")
print(f"Minimum          : Rp{minimum:,.0f}/L")
print(f"Maksimum         : Rp{maximum:,.0f}/L")
print(f"Skewness         : {skewness:.3f}")


# ============================================================
# 3. UJI NORMALITAS DATA ASLI
# ============================================================

stat_normal, p_normal = stats.shapiro(harga)

print("\n" + "=" * 60)
print("UJI NORMALITAS DATA ASLI - SHAPIRO WILK")
print("=" * 60)

print(f"Statistik W : {stat_normal:.4f}")
print(f"P-value     : {p_normal:.6f}")

if p_normal < 0.05:
    print("Hasil       : Data tidak berdistribusi Normal.")
else:
    print("Hasil       : Data tidak menunjukkan penyimpangan signifikan dari Normal.")


# ============================================================
# 4. TRANSFORMASI LOGARITMIK
#    DASAR UNTUK MENGUJI DISTRIBUSI LOG-NORMAL
# ============================================================

log_harga = np.log(harga)

stat_log, p_log = stats.shapiro(log_harga)

print("\n" + "=" * 60)
print("UJI NORMALITAS DATA LOGARITMIK")
print("=" * 60)

print("Transformasi : ln(Harga)")
print(f"Statistik W  : {stat_log:.4f}")
print(f"P-value      : {p_log:.6f}")

if p_log >= 0.05:
    print("Hasil        : ln(Harga) dapat dianggap berdistribusi Normal.")
    print("Interpretasi : Data harga mendukung asumsi Log-Normal.")
else:
    print("Hasil        : ln(Harga) masih menunjukkan penyimpangan dari Normal.")


# ============================================================
# 5. FIT BEBERAPA DISTRIBUSI
# ============================================================

# Normal
normal_params = stats.norm.fit(harga)

# Log-Normal
lognormal_params = stats.lognorm.fit(harga, floc=0)

# Gamma
gamma_params = stats.gamma.fit(harga, floc=0)

# Weibull
weibull_params = stats.weibull_min.fit(harga, floc=0)


# ============================================================
# 6. FUNGSI MENGHITUNG LOG-LIKELIHOOD, AIC, DAN BIC
# ============================================================

def hitung_model(distribution, data, params):

    log_likelihood = np.sum(
        distribution.logpdf(data, *params)
    )

    k = len(params)

    AIC = 2 * k - 2 * log_likelihood

    BIC = k * np.log(len(data)) - 2 * log_likelihood

    return log_likelihood, AIC, BIC


# Hitung masing-masing model
ll_normal, aic_normal, bic_normal = hitung_model(
    stats.norm,
    harga,
    normal_params
)

ll_lognormal, aic_lognormal, bic_lognormal = hitung_model(
    stats.lognorm,
    harga,
    lognormal_params
)

ll_gamma, aic_gamma, bic_gamma = hitung_model(
    stats.gamma,
    harga,
    gamma_params
)

ll_weibull, aic_weibull, bic_weibull = hitung_model(
    stats.weibull_min,
    harga,
    weibull_params
)


# ============================================================
# 7. TABEL PERBANDINGAN DISTRIBUSI
# ============================================================

hasil = {
    "Normal": {
        "AIC": aic_normal,
        "BIC": bic_normal
    },

    "Log-Normal": {
        "AIC": aic_lognormal,
        "BIC": bic_lognormal
    },

    "Gamma": {
        "AIC": aic_gamma,
        "BIC": bic_gamma
    },

    "Weibull": {
        "AIC": aic_weibull,
        "BIC": bic_weibull
    }
}


print("\n" + "=" * 60)
print("PERBANDINGAN DISTRIBUSI")
print("=" * 60)

print(f"{'Distribusi':<15} {'AIC':>12} {'BIC':>12}")
print("-" * 40)

for nama, nilai in hasil.items():
    print(
        f"{nama:<15} "
        f"{nilai['AIC']:>12.2f} "
        f"{nilai['BIC']:>12.2f}"
    )


# ============================================================
# 8. MENENTUKAN DISTRIBUSI TERBAIK
# ============================================================

distribusi_aic = min(
    hasil,
    key=lambda x: hasil[x]["AIC"]
)

distribusi_bic = min(
    hasil,
    key=lambda x: hasil[x]["BIC"]
)

print("\n" + "=" * 60)
print("DISTRIBUSI TERBAIK")
print("=" * 60)

print(f"Berdasarkan AIC : {distribusi_aic}")
print(f"Berdasarkan BIC : {distribusi_bic}")


# ============================================================
# 9. KURVA LOG-NORMAL
# ============================================================

x = np.linspace(
    minimum - 500,
    maximum + 500,
    1000
)

kurva_lognormal = stats.lognorm.pdf(
    x,
    *lognormal_params
)


# ============================================================
# 10. HISTOGRAM + KURVA LOG-NORMAL
# ============================================================

plt.figure(figsize=(12, 7))

plt.hist(
    harga,
    bins=9,
    density=True,
    color="#FFD43B",       # kuning cerah
    edgecolor="#333333",
    linewidth=1.2,
    alpha=0.85,
    label="Harga Pertamax (Empiris)"
)

plt.plot(
    x,
    kurva_lognormal,
    color="#8FAF8A",       # hijau sage
    linewidth=3,
    label="PDF Log-Normal"
)


# ============================================================
# 11. JUDUL
# ============================================================

plt.title(
    "Histogram Distribusi Harga Pertamax (RON 92)\n"
    "September 2023 – Agustus 2026",
    fontsize=17,
    fontweight="bold"
)

plt.xlabel(
    "Harga Pertamax (Rp/L)",
    fontsize=12
)

plt.ylabel(
    "Kepadatan Probabilitas",
    fontsize=12
)


# ============================================================
# 12. KOTAK INFORMASI
# ============================================================

info = (
    f"N = {N} bulan\n"
    f"Mean = Rp{mean:,.0f}/L\n"
    f"Median = Rp{median:,.0f}/L\n"
    f"SD = Rp{std:,.0f}/L\n"
    f"Skewness = {skewness:.2f}\n"
    f"AIC Log-Normal = {aic_lognormal:.2f}"
)

plt.text(
    0.97,
    0.95,
    info,
    transform=plt.gca().transAxes,
    ha="right",
    va="top",
    fontsize=10.5,
    bbox=dict(
        boxstyle="round,pad=0.6",
        facecolor="#F1F5EC",
        edgecolor="#8FAF8A",
        linewidth=1.5
    )
)


# ============================================================
# 13. TAMPILAN
# ============================================================

plt.legend(
    loc="upper left"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    "distribusi_pertamax_lognormal.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 14. KESIMPULAN OTOMATIS
# ============================================================

print("\n" + "=" * 60)
print("KESIMPULAN")
print("=" * 60)

if distribusi_aic == "Log-Normal" and distribusi_bic == "Log-Normal":

    print(
        "Berdasarkan AIC dan BIC, Log-Normal merupakan "
        "distribusi terbaik di antara model yang diuji."
    )

elif distribusi_aic == "Log-Normal":

    print(
        "Berdasarkan AIC, Log-Normal merupakan "
        "distribusi terbaik di antara model yang diuji."
    )

else:

    print(
        "Log-Normal bukan model terbaik berdasarkan AIC. "
        "Gunakan distribusi dengan AIC terendah."
    )