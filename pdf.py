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

# Mengubah data ke bentuk log
ln_harga = np.log(harga)

# Parameter Distribusi Log-Normal
mu = np.mean(ln_harga)
sigma = np.std(ln_harga, ddof=1)

# Membuat rentang harga untuk grafik
x = np.linspace(harga.min(), harga.max(), 500)

# Menghitung PDF Log-Normal
pdf = lognorm.pdf(
    x,
    s=sigma,
    scale=np.exp(mu)
)

# Membuat grafik
plt.figure(figsize=(9, 5))

plt.plot(
    x,
    pdf,
    linewidth=2,
    label="PDF Log-Normal"
)

# Judul dan label
plt.title("PDF Harga Pertamax (September 2023–Agustus 2026)")
plt.xlabel("Harga Pertamax (Rp/L)")
plt.ylabel("Kepadatan Probabilitas")

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

# Menyimpan grafik sebagai PNG
plt.savefig("pdf_pertamax.png", dpi=300)

plt.show()