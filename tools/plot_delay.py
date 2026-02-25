import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

session = "emg_subjectS01_session01"
ch = 0  # channel index

raw = np.load(f"data/matthew/{session}_raw.npz", allow_pickle=True)
filt = np.load(f"data/matthew/{session}_filtered.npz", allow_pickle=True)

x_raw = raw["X"][:, ch]
x_filt = filt["emg"][:, ch]
fs = float(filt["fs"])
t_ms = np.arange(len(x_raw)) / fs * 1000

# Estimate lag via cross-correlation
xr = x_raw - x_raw.mean()
xf = x_filt - x_filt.mean()
corr = np.correlate(xf, xr, mode="full")
lags = np.arange(-len(x_raw) + 1, len(x_raw))
lag_samples = lags[np.argmax(corr)]
lag_ms = lag_samples / fs * 1000

fig = make_subplots(rows=1, cols=1)
fig.add_trace(
    go.Scatter(x=t_ms, y=x_raw, name="Raw", line=dict(color="steelblue"))
)
fig.add_trace(
    go.Scatter(x=t_ms, y=x_filt, name="Filtered", line=dict(color="darkorange"))
)

# Start zoomed-in view; you can pan/zoom interactively
fig.update_xaxes(range=[0, 300], title="Time (ms)")
fig.update_yaxes(title="Amplitude")
fig.update_layout(
    height=400,
    width=1000,
    title=f"{session} ch{ch} raw vs filtered (lag≈{lag_ms:.3f} ms)",
)

fig.show()  # interactive; requires an environment that can render Plotly
# Optionally save to HTML:
# fig.write_html("plots/session01_ch0_interactive.html", include_plotlyjs="cdn")
