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

def load_conversion_table():
    """Load GMS conversion table, with fallback if not available"""
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

def fetch_file(year, month, day, hour):
    """Fetch satellite file from FTP server"""
    request_time = datetime(year, month, day, hour)

    # Determine the satellite
    satellite = None
    for sat, ranges in GMS_RANGES.items():
        for time_range in zip(ranges[::2], ranges[1::2]):
            if time_range[0] <= request_time <= time_range[1]:
                satellite = sat
                break
        if satellite:
            break

    if not satellite:
        return None, "The requested date is out of this dataset's period of coverage!"

    # Check valid hours for the satellite
    if satellite != "GMS4" and hour not in VALID_HOURS:
        return None, f"This dataset is only valid every three hours EXCEPT FOR GMS 4, which begins on 12/04/1989 at 00:00 UTC!"

    temp_dir = tempfile.mkdtemp()
    
    try:
        # Construct the file path
        ftp_dir = f"{FTP_BASE_PATH}/{satellite}/VISSR/{year}{month:02d}/{day:02d}"
        file_name = f"VISSR_{satellite}_{year}{month:02d}{day:02d}{hour:02d}00.tar"
        local_tar_path = os.path.join(temp_dir, file_name)

        # Download the file using ftplib
        with ftplib.FTP(FTP_HOST, timeout=30) as ftp:
            ftp.login()  # Anonymous login
            ftp.cwd(ftp_dir)
            with open(local_tar_path, 'wb') as local_file:
                ftp.retrbinary(f"RETR {file_name}", local_file.write)

        # Extract the IR file from the tar file
        with tarfile.open(local_tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.startswith("./IR") and member.name.endswith(".gz"):
                    member.name = os.path.basename(member.name)  # Strip the leading directory
                    tar.extract(member, path=temp_dir)
                    local_gz_path = os.path.join(temp_dir, member.name)

                    # Return the gzipped file path and satellite info
                    return local_gz_path, satellite, temp_dir, None

        return None, None, None, "Could not find IR file in tar archive"

    except ftplib.all_errors as e:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, None, f"Failed to download the file: {e}"

    except tarfile.TarError as e:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, None, f"Failed to extract the file: {e}"

def process_and_plot(file_path, satellite, temp_dir, year, month, day, hour):
    """Process and plot GMS satellite data"""
    try:
        mapping = load_conversion_table()

        # Decompress the gzipped file
        with gzip.open(file_path, 'rb') as f:
            decoded_data = f.read()

        # Process the data
        data_array = np.frombuffer(decoded_data, dtype=np.uint16)

        # Find closest factors for reshaping
        total_size = data_array.size
        closest_factors = find_closest_factors(total_size, 2182, 3504)

        # Reshape and normalize the data to 0-255 range
        data_array = data_array.reshape(closest_factors)
        data_array = 255 + data_array / -255

        # Convert the data using the provided CSV mapping
        data_converted = (conv(data_array, mapping) - 273.15)

        # Stretch the data to make it as square as possible
        rows, cols = data_converted.shape
        size = max(rows, cols)
        data_square = zoom(data_converted, (size / rows, size / cols))

        # Define the custom inverted colormap
        colors = [
            (0/140, "#330f2f"),
            (10/140, "#9b1f94"),
            (20/140, "#eb6fc0"),
            (20/140, "#e1e4e5"),
            (30/140, "#000300"),
            (40/140, "#fd1917"),
            (50/140, "#fbff2d"),
            (60/140, "#00fe24"),
            (70/140, "#010071"),
            (80/140, "#05fcfe"),
            (80/140, "#fffdfd"),
            (140/140, "#000000")
        ]
        rbtop3 = LinearSegmentedColormap.from_list("rbtop3", colors)

        # Plot the data using the custom inverted colormap with Cartopy
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        vmin = V_MIN_SETTINGS[satellite]
        im = ax.imshow(data_square, vmin=vmin, vmax=40, cmap=rbtop3,
                       extent=[100, 180, -60, 60], transform=ccrs.PlateCarree())

        # Remove all borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Save the plot as a high-quality JPG image first
        temp_plot_path = os.path.join(temp_dir, 'satellite_data_plot.jpg')
        plt.savefig(temp_plot_path, format='jpg', dpi=2000, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Open the saved image and stretch it sideways by 75%
        from PIL import Image, ImageDraw, ImageFont
        img = Image.open(temp_plot_path)
        width, height = img.size
        new_width = int(width * 1.75)
        img = img.resize((new_width, height), Image.LANCZOS)

        # Add watermarks
        draw = ImageDraw.Draw(img)
        try:
            # Try to use a system font, fallback to default if not available
            font = ImageFont.truetype("arial.ttf", 200)
        except:
            try:
                font = ImageFont.truetype("Arial.ttf", 200) 
            except:
                font = ImageFont.load_default()
        
        watermark_text_top = f"GMS Data for {year}-{month:02d}-{day:02d} at {hour:02d}:00 UTC"
        watermark_text_bottom = "Plotted by Sekai Chandra @Sekai_WX"
        draw.text((10, 10), watermark_text_top, fill="white", font=font)
        draw.text((10, height - 250), watermark_text_bottom, fill="red", font=font)

        # Convert to bytes buffer
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=95)
        img_buffer.seek(0)

        return img_buffer.getvalue()

    finally:
        # Clean up temporary directory
        if temp_dir:
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

# Date and time inputs
col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Year", min_value=1981, max_value=1995, value=1990)

with col2:
    month = st.number_input("Month", min_value=1, max_value=12, value=6)

with col3:
    day = st.number_input("Day", min_value=1, max_value=31, value=15)

# Check which satellite covers this date
test_date = datetime(year, month, day)
covering_satellite = get_satellite_for_datetime(test_date)

if covering_satellite:
    st.success(f"Date covered by: {covering_satellite}")
    
    # Determine available hours based on satellite
    if covering_satellite == "GMS4":
        available_hours = list(range(24))
        st.info("GMS 4 provides hourly data")
    else:
        available_hours = [0, 3, 6, 9, 12, 15, 18, 21]
        st.info(f"{covering_satellite} provides 3-hourly data")
    
    hour = st.selectbox("Hour (UTC)", available_hours)
    
    if st.button("Generate Satellite Plot", type="primary"):
        with st.spinner(f"Processing {covering_satellite} data..."):
            try:
                # Fetch and process the data
                result = fetch_file(year, month, day, hour)
                
                if len(result) == 4 and result[0]:  # Success case
                    file_path, satellite, temp_dir, _ = result
                    image_bytes = process_and_plot(file_path, satellite, temp_dir, year, month, day, hour)
                    
                    st.success("Satellite data processed successfully!")
                    st.image(image_bytes, caption=f"{satellite} Satellite Data - {datetime(year, month, day, hour).strftime('%B %d, %Y at %H:00 UTC')}")
                    
                else:
                    # Error case
                    error_message = result[1] if len(result) > 1 else "Unknown error occurred"
                    st.error(error_message)
                    
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
else:
    st.error(f"No GMS satellite coverage available for {year}-{month:02d}-{day:02d}. Please check the coverage periods above.")