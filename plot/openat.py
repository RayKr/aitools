import numpy as np
from matplotlib import pyplot as plt
import os

from scipy.interpolate import make_interp_spline

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def plot_oppt(mtx, name):
    lambda_value = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    f = plt.figure()
    f.set_tight_layout(True)
    plt.grid(True, color="gray", axis="both", ls="--", lw=1)

    nmtx = np.mat(mtx).T

    acc = nmtx[0].tolist()[0]
    aks = nmtx[1].tolist()[0]
    aus = nmtx[2].tolist()[0]
    f1_macro = nmtx[3].tolist()[0]
    f1_weighted = nmtx[4].tolist()[0]

    plt.plot(
        acc,
        label="Acc",
        linewidth=4,
        linestyle="solid",
        color="blue",
        marker="o",
        mfc="white",
    )
    plt.plot(
        aks,
        label="AKS",
        linewidth=4,
        linestyle="--",
        color="green",
        marker="s",
        mfc="white",
    )
    plt.plot(
        aus,
        label="AUS",
        linewidth=4,
        linestyle="--",
        color="red",
        marker="s",
        mfc="white",
    )
    plt.plot(
        f1_macro,
        label="F1-macro",
        linewidth=4,
        linestyle="-",
        color="purple",
        marker="*",
        mfc="white",
    )
    plt.plot(
        f1_weighted,
        label="F1-weighted",
        linewidth=4,
        linestyle="-",
        color="gray",
        marker="*",
        mfc="white",
    )

    legend_font = {
        "family": "Arial",  # 字体
        "style": "normal",
        "size": 14,  # 字号
        "weight": "bold",  # 是否加粗，不加粗
    }

    plt.legend(prop=legend_font)

    plt.xlabel("Threshold", fontsize=18, fontweight="bold")
    plt.ylabel("Accuracy", fontsize=18, fontweight="bold")

    plt.xticks(
        ticks=range(len(lambda_value)),
        labels=lambda_value,
        fontsize=14,
        fontweight="bold",
    )
    plt.yticks(fontsize=14, fontweight="bold")
    plt.ylim(0, 1.05)

    plt.savefig(f"images/{name}.pdf", bbox_inches="tight")


mtx_vanilla_softmax = [
    [0.541, 0.902, 0.000, 0.589, 0.412],
    [0.541, 0.902, 0.000, 0.589, 0.412],
    [0.541, 0.902, 0.000, 0.589, 0.412],
    [0.541, 0.902, 0.000, 0.589, 0.412],
    [0.544, 0.901, 0.009, 0.593, 0.420],
    [0.555, 0.900, 0.038, 0.605, 0.445],
    [0.579, 0.892, 0.108, 0.631, 0.498],
    [0.602, 0.883, 0.180, 0.654, 0.545],
    [0.628, 0.872, 0.263, 0.679, 0.592],
    [0.664, 0.854, 0.378, 0.709, 0.646],
]
plot_oppt(mtx_vanilla_softmax, "vanilla_softmax")

mtx_vanilla_openmax = [
    [0.537, 0.886, 0.013, 0.588, 0.419],
    [0.537, 0.886, 0.013, 0.588, 0.419],
    [0.537, 0.886, 0.013, 0.588, 0.419],
    [0.538, 0.886, 0.015, 0.589, 0.421],
    [0.544, 0.886, 0.033, 0.596, 0.436],
    [0.558, 0.884, 0.070, 0.610, 0.465],
    [0.579, 0.876, 0.133, 0.632, 0.510],
    [0.602, 0.869, 0.203, 0.655, 0.553],
    [0.628, 0.858, 0.282, 0.678, 0.596],
    [0.664, 0.843, 0.395, 0.708, 0.648],
]
plot_oppt(mtx_vanilla_openmax, "vanilla_openmax")

mtx_advtrain_softmax = [
    [0.475, 0.791, 0.000, 0.521, 0.364],
    [0.475, 0.791, 0.000, 0.521, 0.364],
    [0.475, 0.791, 0.000, 0.521, 0.364],
    [0.478, 0.785, 0.017, 0.528, 0.379],
    [0.497, 0.744, 0.126, 0.556, 0.447],
    [0.517, 0.672, 0.285, 0.573, 0.505],
    [0.533, 0.592, 0.446, 0.572, 0.536],
    [0.535, 0.511, 0.570, 0.550, 0.537],
    [0.538, 0.444, 0.680, 0.525, 0.531],
    [0.533, 0.371, 0.776, 0.484, 0.511],
]
plot_oppt(mtx_advtrain_softmax, "advtrain_softmax")

mtx_advtrain_openmax = [
    [0.474, 0.787, 0.004, 0.520, 0.366],
    [0.474, 0.787, 0.004, 0.520, 0.366],
    [0.474, 0.787, 0.004, 0.520, 0.367],
    [0.483, 0.771, 0.052, 0.538, 0.404],
    [0.503, 0.718, 0.182, 0.563, 0.471],
    [0.523, 0.644, 0.340, 0.573, 0.518],
    [0.533, 0.562, 0.490, 0.565, 0.537],
    [0.535, 0.488, 0.606, 0.541, 0.535],
    [0.539, 0.428, 0.705, 0.517, 0.529],
    [0.532, 0.358, 0.794, 0.475, 0.507],
]
plot_oppt(mtx_advtrain_openmax, "advtrain_openmax")
