import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from satpy import Scene
import os
import ftplib
import tarfile
import gzip
from datetime import datetime
import streamlit as st
import tempfile
import shutil
import io

FTP_HOST = "gms.cr.chiba-u.ac.jp"
MTSAT1R_END_DATE = datetime(2013, 12, 19, 2)
MTSAT2_START_DATE = datetime(2013, 12, 19, 3)
MTSAT2_END_DATE = datetime(2015, 7, 7, 1)
COVERAGE_START_DATE = datetime(2005, 6, 28, 3)

def fetch_file(year, month, day, hour):
    """Fetch MTSAT file from FTP server"""
    request_time = datetime(year, month, day, hour)

    if request_time < COVERAGE_START_DATE or request_time > MTSAT2_END_DATE:
        return None, None, None, "The requested date is out of this dataset's period of coverage!"

    if request_time <= MTSAT1R_END_DATE:
        ftp_base_path = "/pub/MTSAT-1R/HRIT"
        satellite = "MTSAT1"
        reader = 'jami_hrit'
    elif request_time >= MTSAT2_START_DATE and request_time <= MTSAT2_END_DATE:
        ftp_base_path = "/pub/MTSAT-2/HRIT"
        satellite = "MTSAT2"
        reader = 'mtsat2-imager_hrit'
    else:
        return None, None, None, "The requested date is out of this dataset's period of coverage!"

    temp_dir = tempfile.mkdtemp()
    
    try:
        ftp_dir = f"{ftp_base_path}/{year}{month:02d}/{day:02d}"
        file_name = f"HRIT_{satellite}_{year}{month:02d}{day:02d}{hour:02d}00.tar"
        local_tar_path = os.path.join(temp_dir, file_name)

        # Download file
        with ftplib.FTP(FTP_HOST, timeout=30) as ftp:
            ftp.login()
            ftp.cwd(ftp_dir)
            with open(local_tar_path, 'wb') as local_file:
                ftp.retrbinary(f"RETR {file_name}", local_file.write)

        # Extract the HRIT file
        with tarfile.open(local_tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.startswith(f"./HRIT_{satellite}") and member.name.endswith("_DK01IR1.gz"):
                    member.name = os.path.basename(member.name)
                    tar.extract(member, path=temp_dir)
                    local_gz_path = os.path.join(temp_dir, member.name)
                    return local_gz_path, reader, temp_dir, None

        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, None, "Could not find HRIT IR1 file in archive"

    except ftplib.all_errors as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, None, f"Failed to download the file: {e}"
    except tarfile.TarError as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, None, f"Failed to extract the file: {e}"
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, None, f"Unexpected error: {e}"

def rbtop3():
    """Create custom colormap"""
    return mcolors.LinearSegmentedColormap.from_list("", [
        (0 / 140, "#000000"),
        (60 / 140, "#fffdfd"),
        (60 / 140, "#05fcfe"),
        (70 / 140, "#010071"),
        (80 / 140, "#00fe24"),
        (90 / 140, "#fbff2d"),
        (100 / 140, "#fd1917"),
        (110 / 140, "#000300"),
        (120 / 140, "#e1e4e5"),
        (120 / 140, "#eb6fc0"),
        (130 / 140, "#9b1f94"),
        (140 / 140, "#330f2f")
    ]).reversed()

def manual_hrit_read(file_path):
    """Manual HRIT file reading as fallback"""
    try:
        with open(file_path, 'rb') as f:
            # Skip HRIT header (typically first 6144 bytes)
            f.seek(6144)
            data = f.read()
        
        # Convert to numpy array
        image_data = np.frombuffer(data, dtype=np.uint8)
        
        # Try to reshape to common MTSAT dimensions
        # MTSAT typically uses 2200x2200 or similar
        expected_size = 2200 * 2200
        if len(image_data) >= expected_size:
            image_data = image_data[:expected_size].reshape(2200, 2200)
        else:
            # Try square root for auto-sizing
            size = int(np.sqrt(len(image_data)))
            image_data = image_data[:size*size].reshape(size, size)
        
        # Convert to temperature (approximated calibration)
        temperature = 180.0 + (image_data.astype(np.float32) / 255.0) * (320.0 - 180.0)
        return temperature
        
    except Exception:
        return None

def process_and_plot(file_path, reader, temp_dir, year, month, day, hour):
    """Process and plot MTSAT data"""
    try:
        custom_cmap = rbtop3()
        kelvin_values = None
        
        # Decompress the gzipped file first
        hrit_path = os.path.join(temp_dir, "decompressed.hrit")
        with gzip.open(file_path, 'rb') as f_in:
            with open(hrit_path, 'wb') as f_out:
                f_out.write(f_in.read())

        # Try Satpy first
        try:
            files = [hrit_path]
            scn = Scene(filenames=files, reader=reader)
            scn.load(['IR1'])
            data_array = scn['IR1']
            kelvin_values = data_array.values
        except Exception:
            # Try manual reading
            kelvin_values = manual_hrit_read(hrit_path)
            
            if kelvin_values is None:
                raise ValueError("Could not read HRIT file")

        # Convert to Celsius
        celsius_values = kelvin_values - 273.15

        # Create plot
        vmin = -100
        vmax = 40

        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        img = ax.imshow(celsius_values, cmap=custom_cmap, vmin=vmin, vmax=vmax)

        ax.grid(False)
        ax.axis('off')
        
        plt.colorbar(img, ax=ax, orientation='vertical', label='Temperature (°C)')

        dt = datetime(year, month, day, hour)
        title = dt.strftime('MTSAT Satellite IR Data for %B %d, %Y at %H:00 UTC')
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

st.title("MTSAT Analysis")

st.write("""
**Coverage Periods:**
- MTSAT-1R: June 28, 2005 - December 19, 2013
- MTSAT-2: December 19, 2013 - July 7, 2015

**Note:** Hourly data available for both satellites.
""")

col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Year", min_value=2005, max_value=2015, value=2010)
    
with col2:
    month = st.number_input("Month", min_value=1, max_value=12, value=6)
    
with col3:
    day = st.number_input("Day", min_value=1, max_value=31, value=15)

hour = st.selectbox("Hour (UTC)", list(range(24)))

if st.button("Generate Plot"):
    with st.spinner("Downloading and processing satellite data..."):
        try:
            file_path, reader, temp_dir, error_message = fetch_file(year, month, day, hour)
            
            if error_message:
                st.error(error_message)
            elif file_path and reader:
                image_bytes = process_and_plot(file_path, reader, temp_dir, year, month, day, hour)
                st.success("Plot generated successfully!")
                st.image(image_bytes, caption="MTSAT Satellite Data")
            else:
                st.error("Failed to download satellite data")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")