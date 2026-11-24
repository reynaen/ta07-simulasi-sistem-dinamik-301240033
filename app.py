import io, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from scipy.integrate import solve_ivp

app = Flask(__name__)

df = pd.read_csv("idn_delta_wave.csv")
t, y = df["Time"].values, df["Population"].values
P0, T = y[0], t[-1]
R_FIT, K_FIT = 0.1206, 2.3231  # hasil kalibrasi curve_fit
H_OPTIONS = [0.1, 0.5, 1, 2, 5, 10, 15]


def f(t, P, r, K):
    return r * P * (1 - P / K)


def euler(h, r, K, t1=T):
    n = int(round(t1 / h)); ts = h * np.arange(n + 1); P = np.zeros(n + 1); P[0] = P0
    for i in range(n):
        P[i + 1] = P[i] + h * f(ts[i], P[i], r, K)
    return ts, P


def rk4(h, r, K, t1=T):
    n = int(round(t1 / h)); ts = h * np.arange(n + 1); P = np.zeros(n + 1); P[0] = P0
    for i in range(n):
        k1 = f(ts[i], P[i], r, K)
        k2 = f(ts[i] + h / 2, P[i] + h * k1 / 2, r, K)
        k3 = f(ts[i] + h / 2, P[i] + h * k2 / 2, r, K)
        k4 = f(ts[i] + h, P[i] + h * k3, r, K)
        P[i + 1] = P[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return ts, P


def num(name, default, lo, hi):
    try:
        v = float(request.values.get(name, default))
    except ValueError:
        v = default
    return min(max(v, lo), hi)


@app.route("/", methods=["GET", "POST"])
def index():
    r = num("r", R_FIT, 0.01, 0.30)
    K = num("K", K_FIT, 1.0, 4.0)
    h = num("h", 1, 0.1, 15)
    methods = request.values.getlist("methods") or ["Euler"]
    whatif = request.values.get("whatif") == "on"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, y, "o", ms=3, label="Data riil", color="tab:blue")
    rows = []

    def add(name, ts, Ps, **kw):
        ax.plot(ts, Ps, label=name, **kw)
        yh = np.interp(t, ts, Ps)
        rmse = float(np.sqrt(np.mean((y - yh) ** 2)))
        r2 = float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
        rows.append({"metode": name, "rmse": f"{rmse:.4f}", "r2": f"{r2:.4f}"})

    if "Euler" in methods:
        ts, Ps = euler(h, r, K); add(f"Euler h={h:g}", ts, Ps, color="tab:red")
    if "RK4" in methods:
        ts, Ps = rk4(h, r, K); add(f"RK4 h={h:g}", ts, Ps, color="tab:green", ls="--")
    if "solve_ivp" in methods:
        s = solve_ivp(f, (0, T), [P0], args=(r, K), t_eval=t)
        add("solve_ivp", s.t, s.y[0], color="black", ls=":")
    if whatif:
        ts, Ps = rk4(1, r * 0.5, K, 240)
        ax.plot(ts, Ps, label="What-If: r turun 50%", color="tab:orange")
    ax.set_xlabel("Hari ke-t sejak 1 Juni 2021"); ax.set_ylabel("Kasus kumulatif (juta)")
    ax.grid(True); ax.legend()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
    img = base64.b64encode(buf.getvalue()).decode()

    return render_template("index.html", img=img, rows=rows, r=r, K=K, h=h, methods=methods,
                           whatif=whatif, h_options=H_OPTIONS, r_fit=R_FIT, k_fit=K_FIT)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
