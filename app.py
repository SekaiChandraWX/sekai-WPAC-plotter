import os
import io
import re
import bz2
import ftplib
import tarfile
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -------------------- Streamlit page --------------------
st.set_page_config(page_title="Himawari-8/9 (CEReS) B13 Brightness Temp", layout="wide")
st.title("Himawari-8/9 B13 (10.4 μm) Brightness Temperature")

# -------------------- Constants --------------------
FTP_HOST = "hmwr829gr.cr.chiba-u.ac.jp"
BASE_DIR = "/gridded/FD/V20190123"
FAMILY   = "TIR"       # CEReS family
BAND     = "01"        # TIR01 ~ AHI Band 13
MISSING_RAW = 65535

# grid spec (fixed CEReS FD lat/lon grid)
GRID = dict(nx=6000, ny=6000, ddeg=0.02, lon_w=85.0, lat_n=60.0)

# WPAC basin (same flavor you used earlier)
WPAC = dict(w=94.9, e=183.5, s=-14.6, n=56.1)

# -------------------- Colormap --------------------
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
    return newcmp.reversed(), 40, -100

# -------------------- FTP helpers (cache everything) --------------------
@lru_cache(maxsize=1)
def ftp_listdir(path: str):
    items = []
    with ftplib.FTP(FTP_HOST) as ftp:
        ftp.login()  # anonymous
        try:
            ftp.cwd(path)
            ftp.retrlines("NLST", callback=items.append)
        except ftplib.all_errors:
            return []
    # NLST returns names only; join to path for clarity
    return [name.split("/")[-1] for name in items]

@st.cache_data(show_spinner=False, ttl=3600)
def list_available_years():
    # Directories here are YYYYMM; derive unique years
    yyyymms = [d for d in ftp_listdir(BASE_DIR) if re.fullmatch(r"\d{6}", d)]
    years = sorted({int(d[:4]) for d in yyyymms})
    return years

@st.cache_data(show_spinner=False, ttl=3600)
def list_months(year: int):
    yyyymms = [d for d in ftp_listdir(BASE_DIR) if d.startswith(f"{year:04d}")]
    months = sorted({int(d[4:6]) for d in yyyymms})
    return months

@st.cache_data(show_spinner=False, ttl=3600)
def list_days(year: int, month: int):
    yyyymm = f"{year:04d}{month:02d}"
    path = f"{BASE_DIR}/{yyyymm}/{FAMILY}"
    files = ftp_listdir(path)
    # pattern: YYYYMMDDHHMM.tir.01.fld.geoss.bz2
    days = sorted({int(f[8:10]) for f in files if f.endswith(f".tir.{BAND}.fld.geoss.bz2") and re.match(r"\d{12}", f[:12])})
    return days

@st.cache_data(show_spinner=False, ttl=3600)
def list_hours(year: int, month: int, day: int):
    yyyymm = f"{year:04d}{month:02d}"
    path = f"{BASE_DIR}/{yyyymm}/{FAMILY}"
    files = ftp_listdir(path)
    hrs = []
    for f in files:
        if f.endswith(f".tir.{BAND}.fld.geoss.bz2") and re.match(r"\d{12}", f[:12]):
            if int(f[6:8]) == day:  # DD
                # Only expose HH for which the 00-minute scan exists
                if f[10:12] == "00":
                    hrs.append(int(f[8:10]))
    return sorted(set(hrs))

def ftp_download(year, month, day, hour, out_dir: Path) -> Path:
    """Download the .bz2 for YYYY MM DD HH (00 minute) and return path."""
    yyyymm = f"{year:04d}{month:02d}"
    ymdhm  = f"{year:04d}{month:02d}{day:02d}{hour:02d}00"
    fname  = f"{ymdhm}.tir.{BAND}.fld.geoss.bz2"
    src    = f"{BASE_DIR}/{yyyymm}/{FAMILY}/{fname}"
    out_dir.mkdir(parents=True, exist_ok=True)
    local  = out_dir / fname
    if local.exists():
        return local

    with ftplib.FTP(FTP_HOST) as ftp:
        ftp.login()
        with open(local, "wb") as f:
            ftp.retrbinary(f"RETR {src}", f.write)
    return local

# -------------------- LUT (download HT13 or fallback HS13) --------------------
@st.cache_data(show_spinner=False, ttl=24*3600)
def fetch_ht13_txt() -> np.ndarray:
    """
    Try to fetch thermal LUT HT13.txt (H08 then H09). If not, fallback to HS13.txt.
    Return a full 0..4095 Kelvin LUT (float32).
    """
    # helper: fetch file over ftp to string
    def _ftp_get(path):
        data = []
        with ftplib.FTP(FTP_HOST) as ftp:
            ftp.login()
            ftp.retrlines(f"RETR {path}", callback=lambda line: data.append(line))
        return "\n".join(data)

    # 1) HT13 preferred
    for path in (f"{BASE_DIR}/support/LUT_H08/HT13.txt",
                 f"{BASE_DIR}/support/LUT_H09/HT13.txt"):
        try:
            txt = _ftp_get(path)
            vals = []
            for line in txt.splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = [t for t in s.replace(",", " ").split() if t]
                try:
                    vals.append(float(parts[-1]))
                except:
                    pass
            arr = np.asarray(vals, dtype=np.float32)
            if arr.size >= 4096 and np.nanmax(arr) > np.nanmin(arr):
                return arr[:4096]
        except ftplib.all_errors:
            pass

    # 2) HS13 fallback (DN=<int> <K>)
    for path in (f"{BASE_DIR}/support/LUT_H08/HS13.txt",
                 f"{BASE_DIR}/support/LUT_H09/HS13.txt"):
        try:
            txt = _ftp_get(path)
            dn_vals = {}
            for line in txt.splitlines():
                m = re.search(r"DN\s*=\s*(\d+)\s+([+-]?\d+(?:\.\d+)?)", line)
                if m:
                    dn_vals[int(m.group(1))] = float(m.group(2))
            if dn_vals:
                N = max(4096, max(dn_vals)+1)
                lut = np.full(N, np.nan, dtype=np.float32)
                for dn, v in dn_vals.items():
                    if dn < N:
                        lut[dn] = v
                idx = np.arange(N, dtype=np.float32)
                good = ~np.isnan(lut)
                lut = np.interp(idx, idx[good], lut[good]).astype(np.float32)
                return lut[:4096]
        except ftplib.all_errors:
            pass

    raise RuntimeError("Could not obtain a thermal LUT (HT13/HS13) from CEReS support.")

# -------------------- Data I/O --------------------
def decompress_bz2(path_bz2: Path) -> Path:
    raw_path = path_bz2.with_suffix("")  # drop .bz2
    if raw_path.exists():
        return raw_path
    with bz2.open(path_bz2, "rb") as f_in, open(raw_path, "wb") as f_out:
        f_out.write(f_in.read())
    return raw_path

def read_raw_to_dataset(raw_path: Path, valid_dt: datetime) -> xr.Dataset:
    nx, ny, ddeg = GRID["nx"], GRID["ny"], GRID["ddeg"]
    lon_w, lat_n = GRID["lon_w"], GRID["lat_n"]
    lons = lon_w + ddeg * np.arange(nx, dtype=np.float64)
    lats = lat_n - ddeg * np.arange(ny, dtype=np.float64)

    arr = np.fromfile(raw_path, dtype=">u2")
    if arr.size != nx*ny:
        raise RuntimeError(f"Size mismatch: expected {nx*ny}, got {arr.size}")
    arr = arr.reshape((ny, nx))
    data = np.where(arr == MISSING_RAW, np.nan, arr).astype("float32")

    t64 = np.datetime64(valid_dt)
    ds = xr.Dataset(
        {f"tir_{BAND}_counts": (("lat","lon"), data)},
        coords={"lat": lats, "lon": lons, "time": t64},
        attrs={"source": "CEReS Himawari-8/9 FD", "family": "TIR", "band": "13"}
    )
    return ds

def counts_to_celsius(ds: xr.Dataset, lut_K: np.ndarray) -> xr.Dataset:
    counts = ds[f"tir_{BAND}_counts"].values
    c = counts.astype(np.int64)
    c = np.clip(c, 0, len(lut_K)-1)
    tbb_K = lut_K[c].astype(np.float32)
    tbb_K = np.where(np.isnan(counts), np.nan, tbb_K)
    tbb_C = np.clip(tbb_K, 150.0, 350.0) - 273.15
    out = ds.copy()
    out["tbb_K"] = (("lat","lon"), tbb_K); out["tbb_K"].attrs["units"] = "K"
    out["tbb_C"] = (("lat","lon"), tbb_C); out["tbb_C"].attrs["units"] = "degC"
    return out

# -------------------- Map helpers (IDL-safe) --------------------
def mod360(x): return (np.asarray(x) % 360.0 + 360.0) % 360.0

def shortest_arc_mid(lw, le):
    w = mod360(lw); e = mod360(le); d = (e - w) % 360.0
    if d <= 180.0:
        center = (w + d/2.0) % 360.0; w_u, e_u = w, w + d
    else:
        d2 = 360.0 - d; center = (e + d2/2.0) % 360.0; w_u, e_u = w, w + 360.0 - d2
    return center, w_u, e_u

def build_projection_and_extent(lon_w, lon_e, lat_s, lat_n):
    s, n = (lat_s, lat_n) if lat_s <= lat_n else (lat_n, lat_s)
    center, w_u, e_u = shortest_arc_mid(lon_w, lon_e)
    def to_center(lon): return ((lon - center + 180.0) % 360.0) - 180.0
    w_c, e_c = to_center(w_u), to_center(e_u)
    proj = ccrs.PlateCarree(central_longitude=float(center))
    extent_crs = ccrs.PlateCarree(central_longitude=float(center))
    return proj, extent_crs, [w_c, e_c, s, n], float(center)

def to_center_frame_vec(lon_pm180, center_deg):
    return ((lon_pm180 - center_deg + 180.0) % 360.0) - 180.0

# -------------------- UI: time pickers (only valid choices) --------------------
with st.container():
    c1, c2, c3, c4 = st.columns(4)
    years = list_available_years()
    with c1:
        year = st.selectbox("Year", years, index=len(years)-1)
    with c2:
        months = list_months(year)
        month = st.selectbox("Month", months, index=0)
    with c3:
        days = list_days(year, month)
        if not days:
            st.warning("No days available in this month (TIR/01).")
            st.stop()
        day = st.selectbox("Day", days, index=0)
    with c4:
        hours = list_hours(year, month, day)
        if not hours:
            st.warning("No hours (HH:00) available on this day.")
            st.stop()
        hour = st.selectbox("Hour (UTC)", hours, index=0)

# -------------------- WPAC / Custom Zoom toggle --------------------
st.subheader("Region")
b1, b2 = st.columns(2)
with b1:
    full_wpac = st.checkbox("Full WPAC basin", value=True, help="94.9°E–183.5°E, 14.6°S–56.1°N")
with b2:
    custom = st.checkbox("Custom 20×20° zoom", value=False, help="Center lon/lat; must lie within WPAC")

lon_center = None; lat_center = None
if custom:
    full_wpac = False
    c5, c6 = st.columns(2)
    with c5:
        lon_center = st.number_input("Center Longitude (°E, can exceed 180)", value=140.0, step=0.1, format="%.3f")
    with c6:
        lat_center = st.number_input("Center Latitude (°N)", value=20.0, step=0.1, format="%.3f")

generate = st.button("Generate", type="primary")

# -------------------- Run pipeline --------------------
if generate:
    # region checks
    if full_wpac:
        lon_w, lon_e, lat_s, lat_n = WPAC["w"], WPAC["e"], WPAC["s"], WPAC["n"]
    else:
        if lon_center is None or lat_center is None:
            st.error("Enter center coordinates for custom zoom.")
            st.stop()
        # enforce within WPAC
        in_wpac_lon = (mod360(lon_center) >= mod360(WPAC["w"])) and (mod360(lon_center) <= mod360(WPAC["e"]))
        in_wpac_lat = (lat_center >= WPAC["s"]) and (lat_center <= WPAC["n"])
        if not (in_wpac_lon and in_wpac_lat):
            st.error("Center point is outside the WPAC basin. Choose a point within WPAC.")
            st.stop()
        lon_w = lon_center - 10.0; lon_e = lon_center + 10.0
        lat_s = lat_center - 10.0; lat_n = lat_center + 10.0
        # clip lat to globe
        lat_s = max(-60.0, lat_s); lat_n = min(60.0, lat_n)

    # download -> decompress
    try:
        with st.spinner("Fetching CEReS file and LUT..."):
            bz2_path = ftp_download(year, month, day, hour, Path(tempfile.gettempdir())/ "him8_cache")
            raw_path = decompress_bz2(bz2_path)
            ds_counts = read_raw_to_dataset(raw_path, datetime(year, month, day, hour))
            lut_K = fetch_ht13_txt()
            ds = counts_to_celsius(ds_counts, lut_K)
    except Exception as e:
        st.error(f"Data fetch/convert failed: {e}")
        st.stop()

    # map/projection
    proj, extent_crs, extent, center_deg = build_projection_and_extent(lon_w, lon_e, lat_s, lat_n)
    data_crs = ccrs.PlateCarree(central_longitude=center_deg)

    # build lon/lat grid for pcolormesh
    lons = ds["lon"].values
    lats = ds["lat"].values
    lon_pm180 = ((lons + 180.0) % 360.0) - 180.0
    lon_plot = to_center_frame_vec(lon_pm180, center_deg)

    # Ensure increasing axes (pcolormesh requirement)
    data = ds["tbb_C"].values
    if lon_plot[0] > lon_plot[-1]:
        lon_plot = lon_plot[::-1]
        data = data[:, ::-1]
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        data = data[::-1, :]

    # plot
    fig = plt.figure(figsize=(13, 9))
    ax = plt.axes(projection=proj)
    ax.set_extent(extent, crs=extent_crs)

    # black land & borders (your request)
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
        file_name=f"Himawari_B13_{year}{month:02d}{day:02d}{hour:02d}_{'WPAC' if full_wpac else 'Custom'}.png",
        mime="image/png",
    )

    # Quick metadata
    st.caption(
        f"DN→K LUT source: CEReS support (HT13/HS13). "
        f"Domain handled IDL-safe; plotting with custom rbtop3. Land/borders in black."
    )
