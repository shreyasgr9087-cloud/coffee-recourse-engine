import numpy as np
import pandas as pd

np.random.seed(42)
num_samples = 15000

# 1. Feature Generation
grind_size = np.random.uniform(300, 1200, num_samples)
water_temp = np.random.uniform(80, 100, num_samples)
brew_time = np.random.uniform(30, 300, num_samples)
water_ratio = np.random.uniform(14, 18, num_samples)
roast_level = np.random.choice(['Light', 'Medium', 'Dark'], num_samples)

# 2. The Fickian Diffusion Physics Engine
T_kelvin = water_temp + 273.15
r_microns = grind_size / 2.0 

D_0_map = {'Light': 1.2e5, 'Medium': 1.5e5, 'Dark': 1.8e5}
D_0 = np.array([D_0_map[r] for r in roast_level])
D = D_0 * np.exp(-2500.0 / T_kelvin)
Fo = (D * brew_time) / (r_microns ** 2)

# CLAUDE'S FIX: 200-term series for mathematically perfect boundary conditions
N_TERMS = 200
pi_sq = np.pi ** 2
n = np.arange(1, N_TERMS + 1).reshape(-1, 1)          
Fo_row = Fo.reshape(1, -1)                              
series_matrix = (1.0 / n ** 2) * np.exp(-(n ** 2) * pi_sq * Fo_row)
series_sum = series_matrix.sum(axis=0)                  

Y_eq = 24.0 + (water_ratio - 14.0) * 0.75 
base_extraction = Y_eq * (1.0 - (6.0 / pi_sq) * series_sum)

# 3. Add Noise & Categorize
noise = np.random.normal(0, 1.2, num_samples)
ey = base_extraction + noise

def categorize_ey(val):
    if val < 18.0: return "Sour / Under-extracted"
    elif val > 22.0: return "Bitter / Over-extracted"
    else: return "Balanced"

labels = [categorize_ey(val) for val in ey]

df = pd.DataFrame({
    "roast_level": roast_level,
    "grind_size_microns": np.round(grind_size, 1),
    "water_temp_c": np.round(water_temp, 1),
    "brew_time_seconds": np.round(brew_time, 1),
    "water_ratio": np.round(water_ratio, 1),
    "extraction_yield": np.round(ey, 2),
    "taste_label": labels
})

df.to_csv("coffee_dataset.csv", index=False)
print(f"✅ 200-Term Fickian Diffusion dataset created with {len(df)} samples.")