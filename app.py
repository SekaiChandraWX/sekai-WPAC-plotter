import io
import os
import re
import bz2
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ───────────────────────── Streamlit page ─────────────────────────
st.set_page_config(page_title="Himawari-8/9 B13 Brightness Temp", layout="wide")
st.title("Himawari-8/9 (CEReS) • Band 13 Brightness Temperature")

# ───────────────────────── Constants ─────────────────────────
FTP_HOST = "ftp://hmwr829gr.cr.chiba-u.ac.jp"
FD_VER   = "V20190123"
FAMILY   = "TIR"   # CEReS family
BAND     = "01"    # TIR01 ≈ AHI Band 13
MISSING_RAW = 65535

# Himawari CEReS fixed-grid (FD V20190123)
FD_DOMAIN = dict(w=85.0, e=205.0, s=-60.0, n=60.0)

# West Pacific basin (your earlier box)
WPAC = dict(w=94.9, e=183.5, s=-14.6, n=56.1)

# Grid shape/spacing
GRID_NX, GRID_NY, DDEG = 6000, 6000, 0.02
LON_WEST, LAT_NORTH = 85.0, 60.0

# Repo-local HS13.txt (put HS13.txt next to app.py)
LUT_PATH = Path(__file__).parent / "HS13.txt"

# ───────────────────────── Colormap ─────────────────────────
def rbtop3():
    newcmp = mcolors.LinearSegmentedColormap.from_list("", [
        (0/140, "#000000"),
        (60/140, "#fffdfd"),
        (60/140, "#05fcfe"),
        (70/140, "#010071"),
        (80/140, "#00fe24"),
        (90/140, "#fbff2d"),
        (100/140,"#fd1917"),
        (110/140,"#000300"),
        (120/140,"#e1e4e5"),
        (120/140,"#eb6fc0"),
        (130/140,"#9b1f94"),
        (140/140,"#330f2f")
    ])
    return newcmp.reversed(), 40, -100  # (cmap, vmax, vmin) in °C

# ───────────────────────── Helpers ─────────────────────────
def build_url(year, month, day, hour):
    yyyymm = f"{year:04d}{month:02d}"
    ymdhm  = f"{year:04d}{month:02d}{day:02d}{hour:02d}00"
    fname  = f"{ymdhm}.tir.{BAND}.fld.geoss.bz2"
    path   = f"/gridded/FD/{FD_VER}/{yyyymm}/TIR/{fname}"
    return f"{FTP_HOST}{path}", fname

def decompress_bz2(bz2_path: Path) -> Path:
    raw_path = bz2_path.with_suffix("")  # drop .bz2
    if raw_path.exists():
        return raw_path
    with bz2.open(bz2_path, "rb") as f_in, open(raw_path, "wb") as f_out:
        f_out.write(f_in.read())
    return raw_path

def read_raw_to_dataset(raw_path: Path, valid_dt: datetime) -> xr.Dataset:
    lons = LON_WEST + DDEG * np.arange(GRID_NX, dtype=np.float64)  # W→E
    lats = LAT_NORTH - DDEG * np.arange(GRID_NY, dtype=np.float64) # N→S
    arr = np.fromfile(raw_path, dtype=">u2")
    if arr.size != GRID_NX * GRID_NY:
        raise RuntimeError(f"Size mismatch: expected {GRID_NX*GRID_NY}, got {arr.size}")
    arr = arr.reshape((GRID_NY, GRID_NX))
    data = np.where(arr == MISSING_RAW, np.nan, arr).astype("float32")
    ds = xr.Dataset(
        {f"tir_{BAND}_counts": (("lat","lon"), data)},
        coords={"lat": lats, "lon": lons, "time": np.datetime64(valid_dt)},
    )
    return ds

def read_hs13_lut_to_kelvin(path: Path, desired_len: int = 4096) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"HS13.txt not found at {path.resolve()}")
    dn_vals = {}
    pat = re.compile(r"DN\s*=\s*(\d+)\s+([+-]?\d+(?:\.\d+)?)")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                dn_vals[int(m.group(1))] = float(m.group(2))
    if not dn_vals:
        raise ValueError("No DN→K pairs parsed from HS13.txt")

    N = max(desired_len, max(dn_vals)+1)
    lut = np.full(N, np.nan, dtype=np.float32)
    for dn, v in dn_vals.items():
        if 0 <= dn < N:
            lut[dn] = v
    idx = np.arange(N, dtype=np.float32)
    good = ~np.isnan(lut)
    lut = np.interp(idx, idx[good], lut[good]).astype(np.float32)  # fills gaps & edges
    return lut[:desired_len]

def counts_to_celsius(ds: xr.Dataset, lut_K: np.ndarray) -> xr.Dataset:
    counts = ds[f"tir_{BAND}_counts"].values
    c = np.clip(counts.astype(np.int64), 0, len(lut_K)-1)
    tbb_K = lut_K[c].astype(np.float32)
    tbb_K = np.where(np.isnan(counts), np.nan, tbb_K)
    tbb_C = np.clip(tbb_K, 150.0, 350.0) - 273.15
    out = ds.copy()
    out["tbb_K"] = (("lat","lon"), tbb_K); out["tbb_K"].attrs["units"] = "K"
    out["tbb_C"] = (("lat","lon"), tbb_C); out["tbb_C"].attrs["units"] = "degC"
    return out

def mod360(x): return (np.asarray(x) % 360.0 + 360.0) % 360.0

def inside_box(lon, lat, box):
    lon_u = mod360(lon); w = mod360(box["w"]); e = mod360(box["e"])
    if w <= e: lon_ok = (lon_u >= w) & (lon_u <= e)
    else:      lon_ok = (lon_u >= w) | (lon_u <= e)  # box crosses the IDL
    lat_ok = (lat >= box["s"]) & (lat <= box["n"])
    return bool(lon_ok and lat_ok)

def build_projection_and_extent(lon_w, lon_e, lat_s, lat_n):
    # center on shortest arc between lon_w/lon_e (IDL-safe)
    w = mod360(lon_w); e = mod360(lon_e); d = (e - w) % 360.0
    if d <= 180.0:
        center = (w + d/2.0) % 360.0
        w_u, e_u = w, w + d
    else:
        d2 = 360.0 - d
        center = (e + d2/2.0) % 360.0
        w_u, e_u = w, w + 360.0 - d2
    def to_center(lon): return ((lon - center + 180.0) % 360.0) - 180.0
    proj = ccrs.PlateCarree(central_longitude=float(center))
    extent = [to_center(w_u), to_center(e_u), min(lat_s, lat_n), max(lat_s, lat_n)]
    return proj, ccrs.PlateCarree(central_longitude=float(center)), extent, float(center)

# ───────────────────────── UI ─────────────────────────
# time inputs (no FTP validation; just constrain plausible range)
now = datetime.utcnow()
c1, c2, c3, c4 = st.columns(4)
with c1:
    year = st.number_input("Year (UTC)", min_value=2015, max_value=now.year, value=min(now.year, 2023), step=1)
with c2:
    month = st.number_input("Month", min_value=1, max_value=12, value=10, step=1)
with c3:
    day = st.number_input("Day", min_value=1, max_value=31, value=11, step=1)
with c4:
    hour = st.number_input("Hour", min_value=0, max_value=23, value=18, step=1)

st.subheader("Region")
b1, b2 = st.columns(2)
with b1:
    full_wpac = st.checkbox("Full WPAC basin", value=True)
with b2:
    custom = st.checkbox("Custom 20×20° zoom (center must be inside WPAC & Himawari FD)", value=False)

lon_center = None; lat_center = None
if custom:
    full_wpac = False
    c5, c6 = st.columns(2)
    with c5:
        lon_center = st.number_input("Center Longitude (°E, can exceed 180)", value=140.0, step=0.1, format="%.3f")
    with c6:
        lat_center = st.number_input("Center Latitude (°N)", value=20.0, step=0.1, format="%.3f")

generate = st.button("Generate", type="primary")

# ───────────────────────── Run ─────────────────────────
if generate:
    # domain gating (no remote checks)
    if full_wpac:
        lon_w, lon_e, lat_s, lat_n = WPAC["w"], WPAC["e"], WPAC["s"], WPAC["n"]
    else:
        if lon_center is None or lat_center is None:
            st.error("Enter center coordinates.")
            st.stop()
        if not inside_box(lon_center, lat_center, FD_DOMAIN):
            st.error("Center is outside the Himawari FD domain (85–205E, 60S–60N).")
            st.stop()
        if not inside_box(lon_center, lat_center, WPAC):
            st.error("Center is outside the WPAC basin. Choose a point within WPAC.")
            st.stop()
        lon_w, lon_e = lon_center - 10.0, lon_center + 10.0
        lat_s, lat_n = max(FD_DOMAIN["s"], lat_center - 10.0), min(FD_DOMAIN["n"], lat_center + 10.0)

    # Build URL and local cache path; no existence probing until download
    url, fname = build_url(int(year), int(month), int(day), int(hour))
    cache_dir = Path(".cache_him8"); cache_dir.mkdir(exist_ok=True)
    bz2_path = cache_dir / fname

    try:
        if not bz2_path.exists():
            with st.spinner("Downloading CEReS file..."):
                urllib.request.urlretrieve(url, bz2_path)
        raw_path = decompress_bz2(bz2_path)
    except Exception as e:
        st.error(f"Download/decompress failed: {e}")
        st.stop()

    # Read counts
    try:
        ds_counts = read_raw_to_dataset(raw_path, datetime(int(year), int(month), int(day), int(hour)))
    except Exception as e:
        st.error(f"Read failed: {e}")
        st.stop()

    # Load repo-bundled LUT and convert to °C
    try:
        lut_K = read_hs13_lut_to_kelvin(LUT_PATH, desired_len=4096)
        ds = counts_to_celsius(ds_counts, lut_K)
    except Exception as e:
        st.error(f"LUT/convert failed: {e}  (Expected HS13.txt next to app.py)")
        st.stop()

    # Projection / extent
    proj, extent_crs, extent, center_deg = build_projection_and_extent(lon_w, lon_e, lat_s, lat_n)
    data_crs = ccrs.PlateCarree(central_longitude=center_deg)

    # Build lon/lat arrays and ensure increasing axes
    lons = ds["lon"].values
    lats = ds["lat"].values
    lon_pm180 = ((lons + 180.0) % 360.0) - 180.0
    lon_plot = ((lon_pm180 - center_deg + 180.0) % 360.0) - 180.0
    data = ds["tbb_C"].values
    if lon_plot[0] > lon_plot[-1]:
        lon_plot = lon_plot[::-1]; data = data[:, ::-1]
    if lats[0] > lats[-1]:
        lats = lats[::-1]; data = data[::-1, :]

    # Plot
    fig = plt.figure(figsize=(13, 9))
    ax = plt.axes(projection=proj)
    ax.set_extent(extent, crs=extent_crs)

    # black land & borders
    ax.add_feature(cfeature.LAND, facecolor="none", edgecolor="black", linewidth=0.7)
    ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.6)
    ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)

    xx, yy = np.meshgrid(lon_plot, lats)
    cmap, vmax, vmin = rbtop3()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    im = ax.pcolormesh(xx, yy, data, transform=data_crs, cmap=cmap, norm=norm, shading="auto")

    when = np.datetime_as_string(ds["time"].values, unit="m")
    region_str = "WPAC Basin" if full_wpac else f"Custom ({lon_center:.1f}E, {lat_center:.1f}N)"
    ax.set_title(f"Himawari-8/9 B13 Brightness Temperature (°C)\n{when} UTC • {region_str}", fontsize=12)

    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.06, shrink=0.82)
    cb.set_label("Brightness Temperature (°C)")

    st.pyplot(fig, clear_figure=True)

    # Download PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=175, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        "Download Image",
        data=buf,
        file_name=f"Himawari_B13_{int(year):04d}{int(month):02d}{int(day):02d}{int(hour):02d}_{'WPAC' if full_wpac else 'Custom'}.png",
        mime="image/png",
    )

    st.caption("LUT: repo-bundled HS13.txt → DN→K; DN 0..4095 completed by interpolation. Land & borders in black.")
