# Import Libraries
!pip install -U scipy matplotlib pandas --quiet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load Results
csv_path = "/content/event_video_results.csv"
df_results = pd.read_csv(csv_path)


numeric_cols = df_results.select_dtypes(include=np.number).columns
df_mean = df_results[numeric_cols].mean()


shuffle_levels = np.array([0, 3, 5, 8, 16, 20, 25, 100], dtype=float)

iou_cols = [
    "orig_iou",
    "shuffle3_iou",
    "shuffle5_iou",
    "shuffle8_iou",
    "shuffle16_iou",
    "shuffle20_iou",
    "shuffle25_iou",
    "shuffle100_iou"
]

# Extract IoU 
mean_iou = np.array([df_mean[col] for col in iou_cols])


def double_exponential(x, a, b, c, d):
    return a * np.exp(-b * x) + c * np.exp(-d * x)


initial_guess = [0.7, 0.2, 0.3, 0.01]

params, _ = curve_fit(
    double_exponential,
    shuffle_levels,
    mean_iou,
    p0=initial_guess,
    bounds=(0, 5)
)

a, b, c, d = params
print("Fitted parameters:")
print(f"a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}")

# Generate Curve
x_smooth = np.linspace(0, 100, 500)
y_fit = double_exponential(x_smooth, *params)


plt.figure(figsize=(9,6))

# Points
plt.scatter(
    shuffle_levels,
    mean_iou,
    color="purple",
    s=60,
    zorder=3,
    label="Measured Mean IoU"
)

# Curve
plt.plot(
    x_smooth,
    y_fit,
    linestyle="--",
    color="black",
    linewidth=2,
    label="Double-Exponential Fit"
)

# White-out
if "white_iou" in df_mean:
    plt.axhline(
        y=df_mean["white_iou"],
        color="red",
        linestyle=":",
        linewidth=2,
        label="White-out IoU"
    )

plt.xlabel("Shuffling Level (%)")
plt.ylabel("Mean Face IoU")
plt.title("Face IoU vs Shuffling Level with Double-Exponential Fit")
plt.legend()
plt.grid(True)
plt.show()


y_pred = double_exponential(shuffle_levels, *params)
ss_res = np.sum((mean_iou - y_pred) ** 2)
ss_tot = np.sum((mean_iou - np.mean(mean_iou)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"R² of fit: {r2:.4f}")
