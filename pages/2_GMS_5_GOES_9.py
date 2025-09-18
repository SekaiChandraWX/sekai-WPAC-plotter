import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from satpy import Scene
import warnings
import os
import ftplib
import tarfile
import gzip
from datetime import datetime
import tempfile
import shutil
import struct
from scipy import ndimage
import streamlit as st
import io

FTP_HOST = "gms.cr.chiba-u.ac.jp"
GMS5_START_DATE = datetime(1995, 6, 13, 6)
GMS5_END_DATE = datetime(2003, 5, 22, 0)
GOES9_START_DATE = datetime(2003, 5, 22, 1)
GOES9_END_DATE = datetime(2005, 6, 28, 2)
VERTICAL_STRETCH = 1.35

def create_colormap():
    """Create custom satellite colormap"""
    return mcolors.LinearSegmentedColormap.from_list("", [
        (0 / 140, "#000000"), (60 / 140, "#fffdfd"), (60 / 140, "#05fcfe"),
        (70 / 140, "#010071"), (80 / 140, "#00fe24"), (90 / 140, "#fbff2d"),
        (100 / 140, "#fd1917"), (110 / 140, "#000300"), (120 / 140, "#e1e4e5"),
        (120 / 140, "#eb6fc0"), (130 / 140, "#9b1f94"), (140 / 140, "#330f2f")
    ]).reversed()

def fetch_file(year, month, day, hour):
    """Fetch satellite file from FTP server"""
    temp_dir = tempfile.mkdtemp()

    try:
        request_time = datetime(year, month, day, hour)

        if request_time < GMS5_START_DATE or request_time > GOES9_END_DATE:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None, None, "The requested date is out of this dataset's period of coverage!"

        if GMS5_START_DATE <= request_time <= GMS5_END_DATE:
            ftp_base_path = "/pub/GMS5/VISSR"
            satellite = "GMS5"
        elif GOES9_START_DATE <= request_time <= GOES9_END_DATE:
            ftp_base_path = "/pub/GOES9-Pacific/VISSR"
            satellite = "GOES9"
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None, None, "The requested date is out of this dataset's period of coverage!"

        ftp_dir = f"{ftp_base_path}/{year}{month:02d}/{day:02d}"
        file_name = f"VISSR_{satellite}_{year}{month:02d}{day:02d}{hour:02d}00.tar"
        local_tar_path = os.path.join(temp_dir, file_name)

        # Download file
        with ftplib.FTP(FTP_HOST, timeout=60) as ftp:
            ftp.login()
            ftp.set_pasv(False)
            ftp.cwd(ftp_dir)
            
            with open(local_tar_path, 'wb') as local_file:
                ftp.retrbinary(f"RETR {file_name}", local_file.write)

        # Extract IR1 file
        with tarfile.open(local_tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.endswith("IR1.A.IMG.gz"):
                    tar.extract(member, path=temp_dir)
                    extracted_path = os.path.join(temp_dir, member.name)
                    
                    # Decompress the gzipped file
                    img_path = os.path.join(temp_dir, "IR1.A.IMG")
                    with gzip.open(extracted_path, 'rb') as f_in:
                        with open(img_path, 'wb') as f_out:
                            f_out.write(f_in.read())

                    return img_path, satellite, temp_dir, None

        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, None, "Could not find IR1.A.IMG.gz file in tar archive"

    except ftplib.all_errors as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, None, f"FTP error: {e}"
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, None, f"General error: {e}"

def try_manual_reading(file_path, year, month, day, hour):
    """Fallback manual reading method if Satpy fails"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()

        # Try to extract dimensions from header
        if len(data) > 12:
            try:
                width = struct.unpack('>H', data[8:10])[0]
                height = struct.unpack('>H', data[10:12])[0]
            except:
                width = 2366
                height = 2366
        else:
            width = 2366
            height = 2366

        # Skip header and get image data
        header_size = 352
        if len(data) > header_size + width * height:
            image_data = data[header_size:header_size + width * height]
            image_array = np.frombuffer(image_data, dtype=np.uint8)

            expected_size = width * height
            if len(image_array) >= expected_size:
                image = image_array[:expected_size].reshape(height, width)
            else:
                available_pixels = len(image_array)
                height = available_pixels // width
                image = image_array[:height * width].reshape(height, width)

            # Convert to temperature (approximated calibration)
            temperature = 180.0 + (image.astype(np.float32) / 255.0) * (320.0 - 180.0)

            return temperature
        else:
            return None

    except Exception:
        return None

def process_and_plot(file_path, satellite, temp_dir, year, month, day, hour):
    """Process and plot the satellite data"""
    try:
        warnings.filterwarnings('ignore')
        kelvin_values = None

        # Try different reading methods
        try:
            # Try with Satpy first
            scene = Scene([file_path], reader='gms5-vissr_l1b', reader_kwargs={"mask_space": False})
            scene.load(["IR1"])
            ir1_data = scene["IR1"]
            kelvin_values = ir1_data.values
        except Exception:
            # Try manual reading as fallback
            kelvin_values = try_manual_reading(file_path, year, month, day, hour)
            
            if kelvin_values is None:
                # Final fallback - simple file reading
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                # Simple conversion assuming 8-bit data
                image_data = np.frombuffer(data[352:], dtype=np.uint8)  # Skip header
                if len(image_data) >= 2366*2366:
                    image_data = image_data[:2366*2366].reshape(2366, 2366)
                    kelvin_values = 180.0 + (image_data.astype(np.float32) / 255.0) * (320.0 - 180.0)
                else:
                    raise ValueError("Could not process satellite data")

        # Convert to Celsius
        celsius_values = kelvin_values - 273.15

        # Apply vertical stretch
        if VERTICAL_STRETCH != 1.0:
            celsius_values = ndimage.zoom(celsius_values, (VERTICAL_STRETCH, 1.0), order=1)

        # Create visualization
        custom_cmap = create_colormap()
        vmin = -100
        vmax = 40

        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
        img = ax.imshow(celsius_values, cmap=custom_cmap, vmin=vmin, vmax=vmax)

        ax.grid(False)
        ax.axis('off')

        plt.colorbar(img, ax=ax, orientation='vertical', label='Temperature (°C)')

        dt = datetime(year, month, day, hour)
        title = f'{satellite} West Pacific IR Data - {dt.strftime("%B %d, %Y at %H:00 UTC")}'
        plt.title(title, fontsize=16, weight='bold', pad=10)
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
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass

st.title("GMS 5 & GOES 9 Analysis")

st.write("""
**Coverage Periods:**
- GMS 5: June 13, 1995 - May 22, 2003
- GOES 9: May 22, 2003 - June 28, 2005

**Note:** Hourly data available for both satellites.
""")

col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Year", min_value=1995, max_value=2005, value=2000)
    
with col2:
    month = st.number_input("Month", min_value=1, max_value=12, value=6)
    
with col3:
    day = st.number_input("Day", min_value=1, max_value=31, value=15)

hour = st.selectbox("Hour (UTC)", list(range(24)))

if st.button("Generate Plot"):
    with st.spinner("Downloading and processing satellite data..."):
        try:
            file_path, satellite, temp_dir, error_message = fetch_file(year, month, day, hour)
            
            if error_message:
                st.error(error_message)
            elif file_path and satellite:
                image_bytes = process_and_plot(file_path, satellite, temp_dir, year, month, day, hour)
                st.success("Plot generated successfully!")
                st.image(image_bytes, caption=f"{satellite} Satellite Data")
            else:
                st.error("Failed to download satellite data")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")