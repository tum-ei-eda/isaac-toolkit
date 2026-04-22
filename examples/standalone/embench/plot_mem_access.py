import sys
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- Load trace ---
assert len(sys.argv) == 7
df = pd.read_pickle(sys.argv[1])
out_file = sys.argv[2]
# --- Binning parameters (tune if needed) ---
TIME_BINS = int(sys.argv[3])
ADDR_BINS = int(sys.argv[4])

# start_idx = int(sys.argv[4], 0)
# print("start_idx", start_idx)
start_pc = int(sys.argv[5], 0)
print("start_pc", start_pc)
idx_count = int(sys.argv[6])
df = df[df["idx"] != 0]
start = (df.pc == start_pc).idxmax()
print("start", start)
df = df.loc[start:]
start_idx = df.idx.iloc[0]
print("start_idx", start_idx)
end_idx = start_idx + idx_count
print("end_idx", end_idx)
end = (df.idx >= end_idx).idxmax()
print("end", end)
df = df.loc[:end]
end_idx_ = df.idx.iloc[-1]
print("end_idx_", end_idx_)

# df = df[df["pc"] == 268436260]
print("df", df)
# def calc_mode(row):
#     # print("row", row)
#     if not pd.isna(row.num_writes):
#         return "w"
#     if not pd.isna(row.num_reads):
#         return "r"
#     return "?"

# df["mode"] = df.apply(calc_mode, axis=1)

# --- Split ---
reads = df[df["mode"] == "r"]
writes = df[df["mode"] == "w"]


def compute_heatmap(data):
    heatmap, xedges, yedges = np.histogram2d(data["idx"], data["addr"], bins=[TIME_BINS, ADDR_BINS])
    # Convert edges → centers

    xcenters = (xedges[:-1] + xedges[1:]) / 2

    ycenters = (yedges[:-1] + yedges[1:]) / 2

    return np.log1p(heatmap.T), xcenters, ycenters
    # return np.log1p(heatmap), xcenters, ycenters
    # return heatmap, xcenters, ycenters
    # return np.log1p(heatmap.T), xedges, yedges


# --- Compute ---
z_r, x_r, y_r = compute_heatmap(reads)
print("A", z_r, x_r, y_r)
z_w, x_w, y_w = compute_heatmap(writes)
print("B", z_w, x_w, y_w)

# --- Create subplots ---
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Reads", "Writes"),
    # shared_yaxes=True
)

# --- Add traces ---
fig.add_trace(
    go.Heatmap(
        z=z_r,
        x=x_r,
        y=y_r,
        colorscale="Viridis",
        colorbar=dict(title="Density"),
        zmin=0,
        zmax=z_r.max(),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Heatmap(
        z=z_w,
        x=x_w,
        y=y_w,
        colorscale="Viridis",
        colorbar=dict(title="Density"),
        # showscale=False,
        zmin=0,
        zmax=z_r.max(),
    ),
    row=1,
    col=2,
)

# --- Layout ---
fig.update_layout(
    title="Memory Access Heatmaps (Reads vs Writes)", xaxis_title="Time", yaxis_title="Address", height=600
)

# --- Save to HTML ---
fig.write_html(out_file)

print("Saved to {out_file}")
