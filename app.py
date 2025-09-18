import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from satpy import Scene
import warnings
import os
import ftplib
import tarfile
import gzip
from datetime import datetime, timedelta
import tempfile
import shutil
import struct
from scipy import ndimage
import io
import requests
import xml.etree.ElementTree as ET
import bz2
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom

# Page configuration
st.set_page_config(
    page_title="WPAC Typhoon Analysis",
    layout="wide"
)

# Constants and date ranges
SATELLITE_RANGES = {
    "GMS1": [(datetime(1981, 3, 1, 0), datetime(1981, 12, 21, 0)), 
             (datetime(1984, 1, 21, 9), datetime(1984, 6, 29, 12))],
    "GMS2": [(datetime(1981, 12, 21, 3), datetime(1984, 1, 21, 6))],
    "GMS3": [(datetime(1984, 9, 27, 6), datetime(1989, 12, 4, 0))],
    "GMS4": [(datetime(1989, 12, 4, 3), datetime(1995, 6, 13, 0))],
    "GMS5": [(datetime(1995, 6, 13, 6), datetime(2003, 5, 22, 0))],
    "GOES9": [(datetime(2003, 5, 22, 1), datetime(2005, 6, 28, 2))],
    "MTSAT1": [(datetime(2005, 6, 28, 3), datetime(2013, 12, 19, 2))],
    "MTSAT2": [(datetime(2013, 12, 19, 3), datetime(2015, 7, 7, 1))],
    "HIMAWARI8": [(datetime(2015, 7, 7, 2), datetime(2022, 12, 13, 4))],
    "HIMAWARI9": [(datetime(2022, 12, 13, 5), datetime.now() + timedelta(days=1))]
}

FTP_HOST = "gms.cr.chiba-u.ac.jp"
VERTICAL_STRETCH = 1.35

def get_satellite_for_datetime(dt):
    """Determine which satellite covers the given datetime"""
    for satellite, ranges in SATELLITE_RANGES.items():
        for start, end in ranges:
            if start <= dt <= end:
                return satellite
    return None

def get_available_times(selected_date, satellite):
    """Get available times for the selected date based on satellite"""
    if satellite in ["GMS1", "GMS2", "GMS3"]:
        return [0, 3, 6, 9, 12, 15, 18, 21]  # Tri-hourly
    elif satellite == "HIMAWARI9":
        times = []
        for h in range(24):
            for m in [0, 10, 20, 30, 40, 50]:
                times.append((h, m))
        return times
    else:
        return list(range(24))  # Hourly

def create_colormap():
    """Create custom satellite colormap"""
    return mcolors.LinearSegmentedColormap.from_list("", [
        (0 / 140, "#000000"), (60 / 140, "#fffdfd"), (60 / 140, "#05fcfe"),
        (70 / 140, "#010071"), (80 / 140, "#00fe24"), (90 / 140, "#fbff2d"),
        (100 / 140, "#fd1917"), (110 / 140, "#000300"), (120 / 140, "#e1e4e5"),
        (120 / 140, "#eb6fc0"), (130 / 140, "#9b1f94"), (140 / 140, "#330f2f")
    ]).reversed()

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

# Processing functions for each satellite type

def process_gms_legacy(year, month, day, hour, satellite):
    """Process GMS 1-4 data"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        ftp_dir = f"/pub/{satellite}/VISSR/{year}{month:02d}/{day:02d}"
        file_name = f"VISSR_{satellite}_{year}{month:02d}{day:02d}{hour:02d}00.tar"
        local_tar_path = os.path.join(temp_dir, file_name)

        with ftplib.FTP(FTP_HOST, timeout=30) as ftp:
            ftp.login()
            ftp.cwd(ftp_dir)
            with open(local_tar_path, 'wb') as local_file:
                ftp.retrbinary(f"RETR {file_name}", local_file.write)

        with tarfile.open(local_tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.startswith("./IR") and member.name.endswith(".gz"):
                    member.name = os.path.basename(member.name)
                    tar.extract(member, path=temp_dir)
                    local_gz_path = os.path.join(temp_dir, member.name)
                    
                    mapping = load_gms_conversion_table()
                    
                    with gzip.open(local_gz_path, 'rb') as f:
                        decoded_data = f.read()

                    data_array = np.frombuffer(decoded_data, dtype=np.uint16)
                    total_size = data_array.size
                    
                    # Find closest factors
                    factors = []
                    for i in range(1, int(np.sqrt(total_size)) + 1):
                        if total_size % i == 0:
                            factors.append((i, total_size // i))
                    closest_factors = min(factors, key=lambda x: abs(x[0] - 2182) + abs(x[1] - 3504))
                    
                    data_array = data_array.reshape(closest_factors)
                    data_array = 255 + data_array / -255
                    
                    # Convert using mapping
                    converted = np.zeros_like(data_array, dtype=float)
                    for i in range(data_array.shape[0]):
                        for j in range(data_array.shape[1]):
                            try:
                                converted[i, j] = mapping.get(round(data_array[i, j]), 0)
                            except:
                                converted[i, j] = 0
                    
                    data_converted = converted - 273.15
                    
                    # Make square
                    rows, cols = data_converted.shape
                    size = max(rows, cols)
                    data_square = zoom(data_converted, (size / rows, size / cols))

                    return create_plot(data_square, satellite, year, month, day, hour, vmin_override={"GMS1": -95, "GMS2": -100, "GMS3": -95, "GMS4": -90}.get(satellite, -100))
                    
        raise Exception("Could not find IR file in archive")
        
    except Exception as e:
        raise Exception(f"GMS Legacy processing failed: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def process_gms5_goes9(year, month, day, hour, satellite):
    """Process GMS 5 or GOES 9 data"""
    temp_dir = tempfile.mkdtemp()

    try:
        if satellite == "GMS5":
            ftp_base_path = "/pub/GMS5/VISSR"
        else:  # GOES9
            ftp_base_path = "/pub/GOES9-Pacific/VISSR"

        ftp_dir = f"{ftp_base_path}/{year}{month:02d}/{day:02d}"
        file_name = f"VISSR_{satellite}_{year}{month:02d}{day:02d}{hour:02d}00.tar"
        local_tar_path = os.path.join(temp_dir, file_name)

        with ftplib.FTP(FTP_HOST, timeout=60) as ftp:
            ftp.login()
            ftp.set_pasv(False)
            ftp.cwd(ftp_dir)
            
            with open(local_tar_path, 'wb') as local_file:
                ftp.retrbinary(f"RETR {file_name}", local_file.write)

        with tarfile.open(local_tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.endswith("IR1.A.IMG.gz"):
                    tar.extract(member, path=temp_dir)
                    extracted_path = os.path.join(temp_dir, member.name)
                    
                    img_path = os.path.join(temp_dir, "IR1.A.IMG")
                    with gzip.open(extracted_path, 'rb') as f_in:
                        with open(img_path, 'wb') as f_out:
                            f_out.write(f_in.read())

                    # Try manual reading
                    with open(img_path, 'rb') as f:
                        data = f.read()
                    
                    header_size = 352
                    if len(data) > header_size + 2366*2366:
                        image_data = np.frombuffer(data[header_size:], dtype=np.uint8)[:2366*2366]
                        image = image_data.reshape(2366, 2366)
                        kelvin_values = 180.0 + (image.astype(np.float32) / 255.0) * (320.0 - 180.0)
                    else:
                        raise ValueError("Insufficient data")

                    celsius_values = kelvin_values - 273.15

                    if VERTICAL_STRETCH != 1.0:
                        celsius_values = ndimage.zoom(celsius_values, (VERTICAL_STRETCH, 1.0), order=1)

                    return create_plot(celsius_values, satellite, year, month, day, hour)
                    
        raise Exception("Could not find IR1.A.IMG.gz file")
        
    except Exception as e:
        raise Exception(f"GMS5/GOES9 processing failed: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def process_mtsat(year, month, day, hour, satellite):
    """Process MTSAT data"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        if satellite == "MTSAT1":
            ftp_base_path = "/pub/MTSAT-1R/HRIT"
        else:  # MTSAT2
            ftp_base_path = "/pub/MTSAT-2/HRIT"

        ftp_dir = f"{ftp_base_path}/{year}{month:02d}/{day:02d}"
        file_name = f"HRIT_{satellite}_{year}{month:02d}{day:02d}{hour:02d}00.tar"
        local_tar_path = os.path.join(temp_dir, file_name)

        with ftplib.FTP(FTP_HOST, timeout=30) as ftp:
            ftp.login()
            ftp.cwd(ftp_dir)
            with open(local_tar_path, 'wb') as local_file:
                ftp.retrbinary(f"RETR {file_name}", local_file.write)

        with tarfile.open(local_tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.startswith(f"./HRIT_{satellite}") and member.name.endswith("_DK01IR1.gz"):
                    member.name = os.path.basename(member.name)
                    tar.extract(member, path=temp_dir)
                    local_gz_path = os.path.join(temp_dir, member.name)
                    
                    hrit_path = os.path.join(temp_dir, "decompressed.hrit")
                    with gzip.open(local_gz_path, 'rb') as f_in:
                        with open(hrit_path, 'wb') as f_out:
                            f_out.write(f_in.read())

                    # Manual HRIT reading
                    with open(hrit_path, 'rb') as f:
                        f.seek(6144)  # Skip HRIT header
                        data = f.read()
                    
                    image_data = np.frombuffer(data, dtype=np.uint8)
                    size = int(np.sqrt(len(image_data)))
                    image_data = image_data[:size*size].reshape(size, size)
                    
                    kelvin_values = 180.0 + (image_data.astype(np.float32) / 255.0) * (320.0 - 180.0)
                    celsius_values = kelvin_values - 273.15

                    return create_plot(celsius_values, satellite, year, month, day, hour)
                    
        raise Exception("Could not find HRIT IR1 file")
        
    except Exception as e:
        raise Exception(f"MTSAT processing failed: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def process_himawari(year, month, day, hour, minute, satellite):
    """Process Himawari-8 or Himawari-9 data"""
    if satellite == "HIMAWARI8":
        bucket_base = "noaa-himawari8"
        sat_code = "H08"
    else:  # HIMAWARI9
        bucket_base = "noaa-himawari9"
        sat_code = "H09"
        
    if satellite == "HIMAWARI8":
        bucket_url = f"https://{bucket_base}.s3.amazonaws.com/?prefix=AHI-L1b-FLDK/{year}/{month:02d}/{day:02d}/{hour:02d}00/"
    else:
        bucket_url = f"https://{bucket_base}.s3.amazonaws.com/?prefix=AHI-L1b-FLDK/{year}/{month:02d}/{day:02d}/{hour:02d}{minute:02d}/"

    # List files
    response = requests.get(bucket_url, timeout=30)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    files = [contents.find('{http://s3.amazonaws.com/doc/2006-03-01/}Key').text 
             for contents in root.findall('{http://s3.amazonaws.com/doc/2006-03-01/}Contents')]

    with tempfile.TemporaryDirectory() as output_folder:
        if satellite == "HIMAWARI8":
            prefix = f"AHI-L1b-FLDK/{year}/{month:02d}/{day:02d}/{hour:02d}00/HS_{sat_code}_{year}{month:02d}{day:02d}_{hour:02d}00_B14"
        else:
            prefix = f"AHI-L1b-FLDK/{year}/{month:02d}/{day:02d}/{hour:02d}{minute:02d}/HS_{sat_code}_{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}_B14"
            
        found_files = [file for file in files if file.startswith(prefix)]

        if not found_files:
            raise Exception("Files not found in AWS bucket")

        # Download and extract files
        extracted_files = []
        for file_name in found_files:
            download_url = f"https://{bucket_base}.s3.amazonaws.com/{file_name}"
            
            # Download
            file_path = os.path.join(output_folder, file_name.split('/')[-1])
            response = requests.get(download_url, timeout=30)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            # Extract
            with open(file_path, 'rb') as f:
                decompressed_data = bz2.decompress(f.read())
                extracted_file_path = os.path.join(output_folder, os.path.splitext(os.path.basename(file_path))[0])
                with open(extracted_file_path, 'wb') as extracted_file:
                    extracted_file.write(decompressed_data)
            
            os.remove(file_path)
            extracted_files.append(extracted_file_path)

        # Process with Satpy
        scn = Scene(filenames=extracted_files, reader='ahi_hsd')
        scn.load(['B14'])
        data_array = scn['B14']
        celsius_values = data_array.values - 273.15

        return create_plot(celsius_values, satellite, year, month, day, hour, minute, 
                         vmin_override=-100, vmax_override=40)

def create_plot(data, satellite, year, month, day, hour, minute=None, vmin_override=None, vmax_override=None):
    """Create standardized plot for any satellite data"""
    custom_cmap = create_colormap()
    vmin = vmin_override if vmin_override is not None else -100
    vmax = vmax_override if vmax_override is not None else 40

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    img = ax.imshow(data, cmap=custom_cmap, vmin=vmin, vmax=vmax)

    ax.grid(False)
    ax.axis('off')
    plt.colorbar(img, ax=ax, orientation='vertical', label='Temperature (°C)')

    # Create title
    if minute is not None:
        dt = datetime(year, month, day, hour, minute)
        time_str = dt.strftime("%B %d, %Y at %H:%M UTC")
    else:
        dt = datetime(year, month, day, hour)
        time_str = dt.strftime("%B %d, %Y at %H:00 UTC")
    
    title = f'{satellite} Satellite IR Data - {time_str}'
    plt.title(title, fontsize=16, weight='bold', pad=10)
    plt.figtext(0.5, -0.02, 'Plotted by Sekai Chandra (@Sekai_WX)', 
               ha='center', fontsize=10, weight='bold')

    # Save to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='jpg', bbox_inches='tight', pad_inches=0.1, dpi=150)
    img_buffer.seek(0)
    plt.close()

    return img_buffer.getvalue()

# Main UI
st.title("WPAC Basin Satellite Data Archive")
st.write("Continuous satellite data of the WPAC basin from 1981 to present")

# Check for conversion file
if not os.path.exists('gms_conversions.csv'):
    st.warning("⚠️ gms_conversions.csv not found. Using fallback temperature conversion for GMS 1-4 data.")

# Date input
col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input(
        "Select Date",
        value=datetime(2018, 6, 15).date(),
        min_value=datetime(1981, 3, 1).date(),
        max_value=datetime.now().date()
    )

# Convert to datetime for satellite detection
test_datetime = datetime.combine(selected_date, datetime.min.time())
satellite = get_satellite_for_datetime(test_datetime)

if satellite:
    st.success(f"Satellite system: {satellite}")
    
    # Get available times
    available_times = get_available_times(selected_date, satellite)
    
    with col2:
        if satellite == "HIMAWARI9":
            time_options = [(f"{h:02d}:{m:02d}", (h, m)) for h, m in available_times]
            selected_time_str = st.selectbox("Select Time (UTC)", [opt[0] for opt in time_options])
            selected_hour, selected_minute = next(opt[1] for opt in time_options if opt[0] == selected_time_str)
        else:
            if isinstance(available_times[0], tuple):
                # This shouldn't happen for non-Himawari9, but just in case
                selected_hour = st.selectbox("Hour (UTC)", [t[0] for t in available_times])
                selected_minute = 0
            else:
                selected_hour = st.selectbox("Hour (UTC)", available_times)
                selected_minute = 0

    if st.button("Generate Satellite Plot", type="primary"):
        with st.spinner(f"Processing {satellite} data..."):
            try:
                year, month, day = selected_date.year, selected_date.month, selected_date.day
                
                # Route to appropriate processing function
                if satellite in ["GMS1", "GMS2", "GMS3", "GMS4"]:
                    image_bytes = process_gms_legacy(year, month, day, selected_hour, satellite)
                elif satellite in ["GMS5", "GOES9"]:
                    image_bytes = process_gms5_goes9(year, month, day, selected_hour, satellite)
                elif satellite in ["MTSAT1", "MTSAT2"]:
                    image_bytes = process_mtsat(year, month, day, selected_hour, satellite)
                elif satellite in ["HIMAWARI8", "HIMAWARI9"]:
                    image_bytes = process_himawari(year, month, day, selected_hour, selected_minute, satellite)
                else:
                    st.error("Unknown satellite system")
                    st.stop()

                st.success("Satellite data processed successfully!")
                st.image(image_bytes, caption=f"{satellite} Satellite Data")
                
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
else:
    st.error("No satellite coverage available for the selected date.")