import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Data harga Pertamax (Rp/L)
harga = np.array([
    13300, 14000, 13400, 13350,
    12950, 12950, 12950, 12950, 12950, 12950, 12950,
    13700, 12950,
    12100, 12100, 12100,
    12500, 12900, 12900, 12500, 12400, 12100,
    12500, 12200, 12200, 12200, 12750,
    12350, 11800, 12300, 12300, 12300,
    16250, 16250, 15950
])

# Parameter distribusi Log-Normal
ln_harga = np.log(harga)
mu = np.mean(ln_harga)
sigma = np.std(ln_harga, ddof=1)

# Membuat nilai X untuk kurva
x = np.linspace(harga.min(), harga.max(), 500)

# Menghitung CDF Log-Normal
cdf = lognorm.cdf(
    x,
    s=sigma,
    scale=np.exp(mu)
)

# Membuat grafik
plt.figure(figsize=(9, 5))

plt.plot(
    x,
    cdf,
    linewidth=2,
    label="CDF Log-Normal"
)

# Titik contoh
nilai = [13000, 14000, 15000]

for v in nilai:
    p = lognorm.cdf(v, s=sigma, scale=np.exp(mu))
    plt.scatter(v, p)
    plt.annotate(
        f"{p:.1%}",
        (v, p),
        xytext=(5, 8),
        textcoords="offset points"
    )

# Judul dan label
plt.title("CDF Harga Pertamax (September 2023–Agustus 2026)")
plt.xlabel("Harga Pertamax (Rp/L)")
plt.ylabel("Probabilitas Kumulatif F(x)")

plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

# Simpan sebagai PNG
plt.savefig("cdf_pertamax.png", dpi=300)

plt.show()