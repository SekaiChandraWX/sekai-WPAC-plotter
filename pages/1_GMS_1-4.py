import os
import gzip
import tarfile
import ftplib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import streamlit as st
import tempfile
import io

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

def load_conversion_table():
    """Load GMS conversion table, with fallback if not available"""
    try:
        if os.path.exists('gms_conversions.csv'):
            conv_df = pd.read_csv('gms_conversions.csv')
            return dict(zip(conv_df['BRIT'], conv_df['TEMP']))
        else:
            # Fallback linear conversion if CSV not available
            brit_values = np.arange(0, 256)
            temp_values = 180 + (brit_values / 255.0) * (320 - 180)
            return dict(zip(brit_values, temp_values))
    except Exception:
        # Emergency fallback
        brit_values = np.arange(0, 256)
        temp_values = 180 + (brit_values / 255.0) * (320 - 180)
        return dict(zip(brit_values, temp_values))

def fetch_file(year, month, day, hour):
    """Fetch satellite file from FTP server"""
    request_time = datetime(year, month, day, hour)

    # Determine satellite
    satellite = None
    for sat, ranges in GMS_RANGES.items():
        for time_range in zip(ranges[::2], ranges[1::2]):
            if time_range[0] <= request_time <= time_range[1]:
                satellite = sat
                break
        if satellite:
            break

    if not satellite:
        return None, None, "The requested date is out of this dataset's period of coverage!"

    # Check valid hours
    if satellite != "GMS4" and hour not in VALID_HOURS:
        return None, None, "This dataset is only valid every three hours EXCEPT FOR GMS 4!"

    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        ftp_dir = f"{FTP_BASE_PATH}/{satellite}/VISSR/{year}{month:02d}/{day:02d}"
        file_name = f"VISSR_{satellite}_{year}{month:02d}{day:02d}{hour:02d}00.tar"
        local_tar_path = os.path.join(temp_dir, file_name)

        # Download file
        with ftplib.FTP(FTP_HOST, timeout=30) as ftp:
            ftp.login()
            ftp.cwd(ftp_dir)
            with open(local_tar_path, 'wb') as local_file:
                ftp.retrbinary(f"RETR {file_name}", local_file.write)

        # Extract IR file
        with tarfile.open(local_tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.startswith("./IR") and member.name.endswith(".gz"):
                    member.name = os.path.basename(member.name)
                    tar.extract(member, path=temp_dir)
                    local_gz_path = os.path.join(temp_dir, member.name)
                    return local_gz_path, satellite, temp_dir

    except Exception as e:
        # Clean up on error
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        return None, None, f"Failed to download file: {str(e)}"

    return None, None, "Could not find IR file in archive"

def conv_data(data_array, mapping):
    """Convert data using temperature mapping"""
    converted = np.zeros_like(data_array, dtype=float)
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            try:
                converted[i, j] = mapping.get(round(data_array[i, j]), 0)
            except:
                converted[i, j] = 0
    return converted

def find_closest_factors(n, target1, target2):
    """Find closest factors for reshaping"""
    factors = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            factors.append((i, n // i))
    return min(factors, key=lambda x: abs(x[0] - target1) + abs(x[1] - target2))

def process_and_plot(file_path, satellite, temp_dir, year, month, day, hour):
    """Process and plot satellite data"""
    try:
        # Load conversion mapping
        mapping = load_conversion_table()
        
        # Read and process data
        with gzip.open(file_path, 'rb') as f:
            decoded_data = f.read()

        data_array = np.frombuffer(decoded_data, dtype=np.uint16)
        total_size = data_array.size
        closest_factors = find_closest_factors(total_size, 2182, 3504)
        
        data_array = data_array.reshape(closest_factors)
        data_array = 255 + data_array / -255
        
        # Convert to temperature and then Celsius
        data_converted = conv_data(data_array, mapping) - 273.15
        
        # Make square
        rows, cols = data_converted.shape
        size = max(rows, cols)
        data_square = zoom(data_converted, (size / rows, size / cols))

        # Create colormap
        colors = [
            (0/140, "#330f2f"), (10/140, "#9b1f94"), (20/140, "#eb6fc0"),
            (20/140, "#e1e4e5"), (30/140, "#000300"), (40/140, "#fd1917"),
            (50/140, "#fbff2d"), (60/140, "#00fe24"), (70/140, "#010071"),
            (80/140, "#05fcfe"), (80/140, "#fffdfd"), (140/140, "#000000")
        ]
        cmap = LinearSegmentedColormap.from_list("rbtop3", colors)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        vmin = V_MIN_SETTINGS[satellite]
        im = ax.imshow(data_square, vmin=vmin, vmax=40, cmap=cmap)

        # Style plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.colorbar(im, ax=ax, orientation='vertical', label='Temperature (°C)')
        
        # Add title
        dt = datetime(year, month, day, hour)
        title = f'{satellite} Data for {dt.strftime("%B %d, %Y at %H:00 UTC")}'
        plt.title(title, fontsize=14, weight='bold', pad=10)
        plt.figtext(0.5, 0.02, 'Plotted by Sekai Chandra (@Sekai_WX)', 
                   ha='center', fontsize=10, weight='bold')

        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='jpg', bbox_inches='tight', pad_inches=0.1, dpi=150)
        img_buffer.seek(0)
        plt.close()

        return img_buffer.getvalue()

    finally:
        # Clean up
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass

st.title("GMS 1-4 Analysis")

st.write("""
**Coverage Periods:**
- GMS1: March 1, 1981 - December 21, 1981 and January 21, 1984 - June 29, 1984
- GMS2: December 21, 1981 - January 21, 1984  
- GMS3: September 27, 1984 - December 4, 1989
- GMS4: December 4, 1989 - June 13, 1995

**Note:** Data available every 3 hours, except GMS4 which has hourly data starting December 4, 1989.
""")

# Check for conversion file
if not os.path.exists('gms_conversions.csv'):
    st.warning("⚠️ gms_conversions.csv not found. Using fallback temperature conversion.")

col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Year", min_value=1981, max_value=1995, value=1990)
    
with col2:
    month = st.number_input("Month", min_value=1, max_value=12, value=6)
    
with col3:
    day = st.number_input("Day", min_value=1, max_value=31, value=15)

hour = st.selectbox("Hour (UTC)", list(range(24)))

if st.button("Generate Plot"):
    with st.spinner("Downloading and processing satellite data..."):
        try:
            file_path, satellite, temp_dir = fetch_file(year, month, day, hour)
            
            if isinstance(satellite, str) and satellite in GMS_RANGES:
                image_bytes = process_and_plot(file_path, satellite, temp_dir, year, month, day, hour)
                st.success("Plot generated successfully!")
                st.image(image_bytes, caption=f"{satellite} Satellite Data")
            else:
                error_msg = temp_dir if temp_dir else "Unknown error occurred"
                st.error(error_msg)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")