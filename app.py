import streamlit as st
import os
import gzip
import tarfile
import ftplib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
from scipy.ndimage import zoom
from datetime import datetime
import tempfile
import shutil
import io
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="GMS 1-4 WPAC Analysis",
    layout="wide"
)

# Constants
FTP_HOST = "gms.cr.chiba-u.ac.jp"
FTP_BASE_PATH = "/pub"
VALID_HOURS = set(range(0, 24, 3))
GMS_RANGES = {
    "GMS1": (datetime(1981, 3, 1, 0), datetime(1981, 12, 21, 0), datetime(1984, 1, 21, 9), datetime(1984, 6, 29, 12)),
    "GMS2": (datetime(1981, 12, 21, 3), datetime(1984, 1, 21, 6)),
    "GMS3": (datetime(1984, 9, 27, 6), datetime(1989, 12, 4, 0)),
    "GMS4": (datetime(1989, 12, 4, 3), datetime(1995, 6, 13, 0))
}
V_MIN_SETTINGS = {"GMS1": -95, "GMS2": -100, "GMS3": -95, "GMS4": -90}

def get_satellite_for_datetime(dt):
    """Determine which GMS satellite covers the given datetime"""
    for sat, ranges in GMS_RANGES.items():
        for time_range in zip(ranges[::2], ranges[1::2]):
            if time_range[0] <= dt <= time_range[1]:
                return sat
    return None

def format_satellite_name(satellite):
    """Format satellite name for display"""
    name_mapping = {
        "GMS1": "GMS 1",
        "GMS2": "GMS 2", 
        "GMS3": "GMS 3",
        "GMS4": "GMS 4"
    }
    return name_mapping.get(satellite, satellite)

def get_available_times(satellite):
    """Get available times based on satellite"""
    if satellite in ["GMS1", "GMS2", "GMS3"]:
        return [0, 3, 6, 9, 12, 15, 18, 21]  # Tri-hourly
    else:  # GMS4
        return list(range(24))  # Hourly

def load_gms_conversion_table():
    """Load GMS conversion table with fallback"""
    try:
        if os.path.exists('gms_conversions.csv'):
            conv_df = pd.read_csv('gms_conversions.csv')
            return dict(zip(conv_df['BRIT'], conv_df['TEMP']))
        else:
            brit_values = np.arange(0, 256)
            temp_values = 180 + (brit_values / 255.0) * (320 - 180)
            return dict(zip(brit_values, temp_values))
    except Exception:
        brit_values = np.arange(0, 256)
        temp_values = 180 + (brit_values / 255.0) * (320 - 180)
        return dict(zip(brit_values, temp_values))

def conv(dat, mapping):
    """Convert data using temperature mapping"""
    for x in range(len(dat)):
        for y in range(len(dat[x])):
            try:
                dat[x][y] = mapping[round(dat[x][y])]
            except:
                dat[x][y] = 0
    return dat

def find_closest_factors(n, target1, target2):
    """Find closest factors for reshaping"""
    factors = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            factors.append((i, n // i))
    return min(factors, key=lambda x: abs(x[0] - target1) + abs(x[1] - target2))

def process_gms_legacy(year, month, day, hour, satellite):
    """Process GMS 1-4 data using original logic"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Download file
        ftp_dir = f"/pub/{satellite}/VISSR/{year}{month:02d}/{day:02d}"
        file_name = f"VISSR_{satellite}_{year}{month:02d}{day:02d}{hour:02d}00.tar"
        local_tar_path = os.path.join(temp_dir, file_name)

        with ftplib.FTP(FTP_HOST, timeout=30) as ftp:
            ftp.login()
            ftp.cwd(ftp_dir)
            with open(local_tar_path, 'wb') as local_file:
                ftp.retrbinary(f"RETR {file_name}", local_file.write)

        # Extract and process
        with tarfile.open(local_tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.startswith("./IR") and member.name.endswith(".gz"):
                    member.name = os.path.basename(member.name)
                    tar.extract(member, path=temp_dir)
                    local_gz_path = os.path.join(temp_dir, member.name)
                    
                    # Original processing logic
                    mapping = load_gms_conversion_table()
                    
                    with gzip.open(local_gz_path, 'rb') as f:
                        decoded_data = f.read()

                    data_array = np.frombuffer(decoded_data, dtype=np.uint16)
                    total_size = data_array.size
                    closest_factors = find_closest_factors(total_size, 2182, 3504)
                    
                    data_array = data_array.reshape(closest_factors)
                    data_array = 255 + data_array / -255
                    data_converted = (conv(data_array, mapping) - 273.15)
                    
                    # Make square
                    rows, cols = data_converted.shape
                    size = max(rows, cols)
                    data_square = zoom(data_converted, (size / rows, size / cols))

                    # Create plot with original logic
                    colors = [
                        (0/140, "#330f2f"), (10/140, "#9b1f94"), (20/140, "#eb6fc0"),
                        (20/140, "#e1e4e5"), (30/140, "#000300"), (40/140, "#fd1917"),
                        (50/140, "#fbff2d"), (60/140, "#00fe24"), (70/140, "#010071"),
                        (80/140, "#05fcfe"), (80/140, "#fffdfd"), (140/140, "#000000")
                    ]
                    rbtop3 = LinearSegmentedColormap.from_list("rbtop3", colors)

                    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
                    vmin = V_MIN_SETTINGS[satellite]
                    im = ax.imshow(data_square, vmin=vmin, vmax=40, cmap=rbtop3,
                                   extent=[100, 180, -60, 60], transform=ccrs.PlateCarree())

                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # Save to temporary file first
                    temp_plot_path = os.path.join(temp_dir, 'satellite_data_plot.jpg')
                    plt.savefig(temp_plot_path, format='jpg', dpi=2000, bbox_inches='tight', pad_inches=0)
                    plt.close()

                    # Open the saved image and stretch it sideways by 75%
                    img = Image.open(temp_plot_path)
                    width, height = img.size
                    new_width = int(width * 1.75)
                    img = img.resize((new_width, height), Image.LANCZOS)

                    # Add watermarks using PIL
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(img)
                    try:
                        font = ImageFont.truetype("arial.ttf", 200)
                    except:
                        try:
                            font = ImageFont.truetype("Arial.ttf", 200) 
                        except:
                            # Fallback to a smaller default font if system fonts fail
                            font = ImageFont.load_default()
                    
                    watermark_text_top = f"GMS Data for {year}-{month:02d}-{day:02d} at {hour:02d}:00 UTC"
                    watermark_text_bottom = "Plotted by Sekai Chandra @Sekai_WX"
                    draw.text((10, 10), watermark_text_top, fill="white", font=font)
                    draw.text((10, height - 250), watermark_text_bottom, fill="red", font=font)

                    # Convert to bytes
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='JPEG', quality=95)
                    img_buffer.seek(0)

                    return img_buffer.getvalue()
                    
        raise Exception("Could not find IR file in archive")
        
    except Exception as e:
        raise Exception(f"GMS processing failed: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# Main UI
st.title("GMS 1-4 WPAC Basin Analysis")
st.write("Satellite data analysis for the Western Pacific basin from 1981-1995")

# Information about coverage
st.info("""
**Coverage Periods:**
- GMS 1: March 1, 1981 - December 21, 1981 and January 21, 1984 - June 29, 1984
- GMS 2: December 21, 1981 - January 21, 1984  
- GMS 3: September 27, 1984 - December 4, 1989
- GMS 4: December 4, 1989 - June 13, 1995

**Data Frequency:**
- GMS 1-3: Every 3 hours (00, 03, 06, 09, 12, 15, 18, 21 UTC)
- GMS 4: Hourly data starting December 4, 1989
""")

# Check for conversion file
if not os.path.exists('gms_conversions.csv'):
    st.warning("⚠️ gms_conversions.csv not found. Using fallback temperature conversion.")

# Date input
col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input(
        "Select Date",
        value=datetime(1990, 6, 15).date(),
        min_value=datetime(1981, 3, 1).date(),
        max_value=datetime(1995, 6, 13).date()
    )

# Convert to datetime for satellite detection
test_datetime = datetime.combine(selected_date, datetime.min.time())
satellite = get_satellite_for_datetime(test_datetime)

if satellite:
    st.success(f"Using data from: {format_satellite_name(satellite)}")
    
    # Get available times
    available_times = get_available_times(satellite)
    
    with col2:
        selected_hour = st.selectbox("Hour (UTC)", available_times)

    if st.button("Generate Satellite Plot", type="primary"):
        with st.spinner(f"Processing {format_satellite_name(satellite)} data..."):
            try:
                year, month, day = selected_date.year, selected_date.month, selected_date.day
                
                image_bytes = process_gms_legacy(year, month, day, selected_hour, satellite)
                st.success("Satellite data processed successfully!")
                st.image(image_bytes, caption=f"{format_satellite_name(satellite)} Satellite Data")
                
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
else:
    st.error("No GMS satellite coverage available for the selected date.")