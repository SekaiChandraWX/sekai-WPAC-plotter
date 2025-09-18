import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from satpy import Scene
from satpy.readers.gms import gms5_vissr_l1b
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
import cartopy.crs as ccrs
from PIL import Image, ImageDraw, ImageFont

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

def format_satellite_name(satellite):
    """Format satellite name for display"""
    name_mapping = {
        "GMS1": "GMS 1", "GMS2": "GMS 2", "GMS3": "GMS 3", "GMS4": "GMS 4",
        "GMS5": "GMS 5", "GOES9": "GOES-9", "MTSAT1": "MTSAT-1R", "MTSAT2": "MTSAT-2",
        "HIMAWARI8": "Himawari-8", "HIMAWARI9": "Himawari-9"
    }
    return name_mapping.get(satellite, satellite)

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

# GMS 1-4 PROCESSING (ORIGINAL LOGIC)
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

        with ftplib.FTP(FTP_HOST) as ftp:
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

                    # Create plot with cartopy
                    v_min_settings = {"GMS1": -95, "GMS2": -100, "GMS3": -95, "GMS4": -90}
                    
                    colors = [
                        (0/140, "#330f2f"), (10/140, "#9b1f94"), (20/140, "#eb6fc0"),
                        (20/140, "#e1e4e5"), (30/140, "#000300"), (40/140, "#fd1917"),
                        (50/140, "#fbff2d"), (60/140, "#00fe24"), (70/140, "#010071"),
                        (80/140, "#05fcfe"), (80/140, "#fffdfd"), (140/140, "#000000")
                    ]
                    rbtop3 = LinearSegmentedColormap.from_list("rbtop3", colors)

                    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
                    vmin = v_min_settings[satellite]
                    im = ax.imshow(data_square, vmin=vmin, vmax=40, cmap=rbtop3,
                                   extent=[100, 180, -60, 60], transform=ccrs.PlateCarree())

                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    plt.colorbar(im, ax=ax, orientation='vertical', label='Temperature (°C)')
                    
                    dt = datetime(year, month, day, hour)
                    title = f'{format_satellite_name(satellite)} Data for {dt.strftime("%B %d, %Y at %H:00 UTC")}'
                    plt.title(title, fontsize=16, weight='bold', pad=10)

                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
                    img_buffer.seek(0)
                    plt.close()

                    return img_buffer.getvalue()
                    
        raise Exception("Could not find IR file in archive")
        
    except Exception as e:
        raise Exception(f"GMS Legacy processing failed: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# GMS5/GOES9 PROCESSING (ORIGINAL LOGIC WITH PATCHES)
def comprehensive_patch_gms5_reader():
    """Comprehensive patch for GMS5 reader"""
    if hasattr(gms5_vissr_l1b.GMS5VISSRFileHandler, '_patched_comprehensive'):
        return

    original_read_image_data = gms5_vissr_l1b.GMS5VISSRFileHandler._read_image_data
    original_get_actual_shape = gms5_vissr_l1b.GMS5VISSRFileHandler._get_actual_shape

    def safe_read_from_file_obj(file_obj, dtype, count, offset=0):
        file_obj.seek(offset)
        remaining_data = file_obj.read()
        actual_bytes = len(remaining_data)
        bytes_needed = dtype.itemsize * count

        if actual_bytes < bytes_needed:
            actual_count = actual_bytes // dtype.itemsize
            if actual_count == 0:
                raise ValueError(f"Not enough data to read even one record of type {dtype}")
        else:
            actual_count = count
            remaining_data = remaining_data[:bytes_needed]

        return np.frombuffer(remaining_data, dtype=dtype, count=actual_count)

    def patched_read_image_data(self):
        try:
            return original_read_image_data(self)
        except (ValueError, struct.error) as e:
            if "buffer is smaller than requested size" in str(e) or "unpack requires" in str(e):
                return self._read_image_data_completely_safe()
            raise e

    def patched_get_actual_shape(self):
        try:
            return original_get_actual_shape(self)
        except Exception:
            return self._get_file_based_shape()

    def _read_image_data_completely_safe(self):
        specs = self._get_image_data_type_specs()
        file_size = os.path.getsize(self._filename)
        available_data = file_size - specs["offset"]
        max_records = available_data // specs["dtype"].itemsize

        from satpy.readers.utils import generic_open
        with generic_open(self._filename, "rb") as file_obj:
            return safe_read_from_file_obj(
                file_obj, dtype=specs["dtype"], count=max_records, offset=specs["offset"]
            )

    def _get_file_based_shape(self):
        try:
            _, nominal_pixels = self._get_nominal_shape()
        except:
            nominal_pixels = 2366

        specs = self._get_image_data_type_specs()
        file_size = os.path.getsize(self._filename)
        available_data = file_size - specs["offset"]

        if specs["dtype"].names:
            sample_record_size = specs["dtype"].itemsize
            max_lines = available_data // sample_record_size
        else:
            bytes_per_pixel = specs["dtype"].itemsize
            total_pixels = available_data // bytes_per_pixel
            max_lines = total_pixels // nominal_pixels

        return max_lines, nominal_pixels

    gms5_vissr_l1b.GMS5VISSRFileHandler._read_image_data = patched_read_image_data
    gms5_vissr_l1b.GMS5VISSRFileHandler._get_actual_shape = patched_get_actual_shape
    gms5_vissr_l1b.GMS5VISSRFileHandler._read_image_data_completely_safe = _read_image_data_completely_safe
    gms5_vissr_l1b.GMS5VISSRFileHandler._get_file_based_shape = _get_file_based_shape
    gms5_vissr_l1b.GMS5VISSRFileHandler._patched_comprehensive = True
    gms5_vissr_l1b.read_from_file_obj = safe_read_from_file_obj

def create_gms5_colormap():
    return mcolors.LinearSegmentedColormap.from_list("", [
        (0 / 140, "#000000"), (60 / 140, "#fffdfd"), (60 / 140, "#05fcfe"),
        (70 / 140, "#010071"), (80 / 140, "#00fe24"), (90 / 140, "#fbff2d"),
        (100 / 140, "#fd1917"), (110 / 140, "#000300"), (120 / 140, "#e1e4e5"),
        (120 / 140, "#eb6fc0"), (130 / 140, "#9b1f94"), (140 / 140, "#330f2f")
    ]).reversed()

def try_manual_reading(file_path, year, month, day, hour):
    """Manual reading fallback"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()

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

            temperature = 180.0 + (image.astype(np.float32) / 255.0) * (320.0 - 180.0)

            import xarray as xr
            ir1_data = xr.DataArray(
                temperature, dims=['y', 'x'],
                attrs={
                    'platform': 'GMS-5', 'sensor': 'VISSR', 'units': 'K',
                    'standard_name': 'brightness_temperature',
                    'start_time': datetime(year, month, day, hour),
                }
            )
            return ir1_data
        else:
            raise ValueError("Insufficient data in file")
    except Exception:
        return None

def process_gms5_goes9(year, month, day, hour, satellite):
    """Process GMS5/GOES9 using original logic"""
    temp_dir = tempfile.mkdtemp()

    try:
        comprehensive_patch_gms5_reader()
        warnings.filterwarnings('ignore')

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
                    
                    satpy_filename = f"VISSR_{year}{month:02d}{day:02d}_{hour:02d}00_IR1.A.IMG"
                    local_img_path = os.path.join(temp_dir, satpy_filename)

                    with gzip.open(extracted_path, 'rb') as f_in:
                        with open(local_img_path, 'wb') as f_out:
                            f_out.write(f_in.read())

                    ir1_data = None
                    try:
                        scene = Scene([local_img_path], reader='gms5-vissr_l1b', reader_kwargs={"mask_space": False})
                        scene.load(["IR1"])
                        ir1_data = scene["IR1"]
                        satellite_name = ir1_data.attrs.get('platform', 'GMS-5')
                    except Exception:
                        ir1_data = try_manual_reading(local_img_path, year, month, day, hour)
                        if ir1_data is None:
                            raise ValueError("Both Satpy and manual reading failed")
                        satellite_name = format_satellite_name(satellite)

                    kelvin_values = ir1_data.values
                    celsius_values = kelvin_values - 273.15

                    if VERTICAL_STRETCH != 1.0:
                        original_height, original_width = celsius_values.shape
                        new_height = int(original_height * VERTICAL_STRETCH)
                        celsius_values = ndimage.zoom(celsius_values, (VERTICAL_STRETCH, 1.0), order=1)

                    custom_cmap = create_gms5_colormap()
                    vmin = -100
                    vmax = 40

                    fig, ax = plt.subplots(figsize=(12, 10), dpi=600)
                    img = ax.imshow(celsius_values, cmap=custom_cmap, vmin=vmin, vmax=vmax)

                    ax.grid(False)
                    ax.axis('off')
                    plt.colorbar(img, ax=ax, orientation='vertical', label='Temperature (°C)')

                    dt = datetime(year, month, day, hour)
                    title = f'{format_satellite_name(satellite)} West Pacific IR Data - {dt.strftime("%B %d, %Y at %H:00 UTC")}'
                    plt.title(title, fontsize=18, weight='bold', pad=10)
                    plt.figtext(0.5, -0.015, 'Plotted by Sekai Chandra (@Sekai_WX)', ha='center', fontsize=12, weight='bold')

                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='jpg', bbox_inches='tight', pad_inches=0.05, dpi=600)
                    img_buffer.seek(0)
                    plt.close()

                    return img_buffer.getvalue()
                    
        raise Exception("Could not find IR1.A.IMG.gz file")
        
    except Exception as e:
        raise Exception(f"GMS5/GOES9 processing failed: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# MTSAT PROCESSING (ORIGINAL LOGIC)
def rbtop3_mtsat():
    return mcolors.LinearSegmentedColormap.from_list("", [
        (0 / 140, "#000000"), (60 / 140, "#fffdfd"), (60 / 140, "#05fcfe"),
        (70 / 140, "#010071"), (80 / 140, "#00fe24"), (90 / 140, "#fbff2d"),
        (100 / 140, "#fd1917"), (110 / 140, "#000300"), (120 / 140, "#e1e4e5"),
        (120 / 140, "#eb6fc0"), (130 / 140, "#9b1f94"), (140 / 140, "#330f2f")
    ]).reversed()

def process_mtsat(year, month, day, hour, satellite):
    """Process MTSAT using original logic"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        if satellite == "MTSAT1":
            ftp_base_path = "/pub/MTSAT-1R/HRIT"
            reader = 'jami_hrit'
        else:  # MTSAT2
            ftp_base_path = "/pub/MTSAT-2/HRIT"
            reader = 'mtsat2-imager_hrit'

        ftp_dir = f"{ftp_base_path}/{year}{month:02d}/{day:02d}"
        file_name = f"HRIT_{satellite}_{year}{month:02d}{day:02d}{hour:02d}00.tar"
        local_tar_path = os.path.join(temp_dir, file_name)

        with ftplib.FTP(FTP_HOST) as ftp:
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
                    
                    local_hrit_path = os.path.join(temp_dir, os.path.basename(local_gz_path).replace(".gz", ""))
                    with gzip.open(local_gz_path, 'rb') as f_in:
                        with open(local_hrit_path, 'wb') as f_out:
                            f_out.write(f_in.read())

                    files = [local_hrit_path]
                    scn = Scene(filenames=files, reader=reader)

                    data_type = 'IR1'
                    scn.load([data_type])

                    data_array = scn[data_type]
                    kelvin_values = data_array.values
                    celsius_values = kelvin_values - 273.15

                    custom_cmap = rbtop3_mtsat()
                    vmin = -100
                    vmax = 40

                    fig, ax = plt.subplots(figsize=(18, 10), dpi=800)
                    img = ax.imshow(celsius_values, cmap=custom_cmap, vmin=vmin, vmax=vmax)

                    ax.grid(False)
                    plt.colorbar(img, ax=ax, orientation='vertical', label='Temperature (°C)')

                    dt = datetime(year, month, day, hour)
                    title = dt.strftime('MTSAT Satellite IR Data for %B %d, %Y at %H:00 UTC')
                    plt.title(title, fontsize=18, weight='bold', pad=10)

                    plt.tight_layout()
                    plt.figtext(0.5, -0.01, 'Plotted by Sekai Chandra (@Sekai_WX)', ha='center', fontsize=15, weight='bold')

                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='jpg', bbox_inches='tight', pad_inches=0.05)
                    img_buffer.seek(0)
                    plt.close()

                    return img_buffer.getvalue()
                    
        raise Exception("Could not find HRIT IR1 file")
        
    except Exception as e:
        raise Exception(f"MTSAT processing failed: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# HIMAWARI PROCESSING (ORIGINAL LOGIC)
def rbtop3_himawari():
    return mcolors.LinearSegmentedColormap.from_list("", [
        (0 / 140, "#000000"), (60 / 140, "#fffdfd"), (60 / 140, "#05fcfe"),
        (70 / 140, "#010071"), (80 / 140, "#00fe24"), (90 / 140, "#fbff2d"),
        (100 / 140, "#fd1917"), (110 / 140, "#000300"), (120 / 140, "#e1e4e5"),
        (120 / 140, "#eb6fc0"), (130 / 140, "#9b1f94"), (140 / 140, "#330f2f")
    ]).reversed()

def download_file(url, output_folder):
    file_name = url.split('/')[-1]
    file_path = os.path.join(output_folder, file_name)
    response = requests.get(url)
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

def extract_bz2(file_path, output_folder):
    with open(file_path, 'rb') as file:
        decompressed_data = bz2.decompress(file.read())
        extracted_file_path = os.path.join(output_folder, os.path.splitext(os.path.basename(file_path))[0])
        with open(extracted_file_path, 'wb') as extracted_file:
            extracted_file.write(decompressed_data)
    os.remove(file_path)
    return extracted_file_path

def list_files(bucket_url):
    response = requests.get(bucket_url)
    if response.status_code == 200:
        xml_content = response.content
        root = ET.fromstring(xml_content)
        files = []
        for contents in root.findall('{http://s3.amazonaws.com/doc/2006-03-01/}Contents'):
            key = contents.find('{http://s3.amazonaws.com/doc/2006-03-01/}Key').text
            files.append(key)
        return files
    else:
        raise Exception("Failed to access the bucket.")

def process_himawari8(year, month, day, hour):
    """Process Himawari-8 using original logic"""
    custom_cmap = rbtop3_himawari()
    
    bucket_url = f"https://noaa-himawari8.s3.amazonaws.com/?prefix=AHI-L1b-FLDK/{year}/{month:02d}/{day:02d}/{hour:02d}00/"

    start_date = datetime(2015, 7, 7, 2)
    end_date = datetime(2019, 12, 31, 23)
    split_start_date = datetime(2020, 1, 1, 0)
    split_end_date = datetime(2022, 12, 13, 4)
    requested_date = datetime(int(year), int(month), int(day), int(hour))

    if requested_date < start_date or (requested_date > end_date and requested_date < split_start_date) or requested_date > split_end_date:
        raise Exception("The requested date is out of this dataset's period of coverage!")

    files = list_files(bucket_url)

    with tempfile.TemporaryDirectory() as output_folder:
        if requested_date <= end_date:
            file_name = f"AHI-L1b-FLDK/{year}/{month:02d}/{day:02d}/{hour:02d}00/HS_H08_{year}{month:02d}{day:02d}_{hour:02d}00_B14_FLDK_R20_S0101.DAT.bz2"
            download_url = f"https://noaa-himawari8.s3.amazonaws.com/{file_name}"
            if file_name in files:
                compressed_file_path = download_file(download_url, output_folder)
                extracted_file_path = extract_bz2(compressed_file_path, output_folder)

                scn = Scene(filenames=[extracted_file_path], reader='ahi_hsd')
                scn.load(['B14'])
                data_array = scn['B14']
                celsius_values = data_array.values - 273.15

                vmin_kelvin = 173.15
                vmax_kelvin = 313.15
                vmin_celsius = vmin_kelvin - 273.15
                vmax_celsius = vmax_kelvin - 273.15

                fig, ax = plt.subplots(figsize=(18, 10), dpi=750)
                img = ax.imshow(celsius_values, cmap=custom_cmap, vmin=vmin_celsius, vmax=vmax_celsius)

                ax.grid(False)
                plt.colorbar(img, ax=ax, orientation='vertical', label='Temperature (°C)')

                dt = datetime(year, month, day, hour)
                title = dt.strftime('Himawari-8 Satellite B14 Data for %B %d, %Y at %H:00 UTC')
                plt.title(title, fontsize=18, weight='bold', pad=10)

                plt.tight_layout()
                plt.figtext(0.5, -0.01, 'Plotted by Sekai Chandra (@Sekai_WX)', ha='center', fontsize=15, weight='bold')

                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='jpg', bbox_inches='tight', pad_inches=0.05)
                img_buffer.seek(0)
                plt.close()

                return img_buffer.getvalue()
            else:
                raise Exception("File not found in AWS bucket")
        else:
            prefix = f"AHI-L1b-FLDK/{year}/{month:02d}/{day:02d}/{hour:02d}00/HS_H08_{year}{month:02d}{day:02d}_{hour:02d}00_B14"
            found_files = [file for file in files if file.startswith(prefix)]

            if found_files:
                extracted_files = []
                for file_name in found_files:
                    download_url = f"https://noaa-himawari8.s3.amazonaws.com/{file_name}"
                    compressed_file_path = download_file(download_url, output_folder)
                    extracted_file_path = extract_bz2(compressed_file_path, output_folder)
                    extracted_files.append(extracted_file_path)

                scn = Scene(filenames=extracted_files, reader='ahi_hsd')
                scn.load(['B14'])
                data_array = scn['B14']
                celsius_values = data_array.values - 273.15

                vmin_kelvin = 173.15
                vmax_kelvin = 313.15
                vmin_celsius = vmin_kelvin - 273.15
                vmax_celsius = vmax_kelvin - 273.15

                fig, ax = plt.subplots(figsize=(18, 10), dpi=500)
                img = ax.imshow(celsius_values, cmap=custom_cmap, vmin=vmin_celsius, vmax=vmax_celsius)

                ax.grid(False)
                plt.colorbar(img, ax=ax, orientation='vertical', label='Temperature (°C)')

                dt = datetime(year, month, day, hour)
                title = dt.strftime('Himawari-8 Satellite B14 Data for %B %d, %Y at %H:00 UTC')
                plt.title(title, fontsize=18, weight='bold', pad=10)

                plt.tight_layout()
                plt.figtext(0.5, -0.01, 'Plotted by Sekai Chandra (@Sekai_WX)', ha='center', fontsize=15, weight='bold')

                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='jpg', bbox_inches='tight', pad_inches=0.05)
                img_buffer.seek(0)
                plt.close()

                return img_buffer.getvalue()
            else:
                raise Exception("Files not found in AWS bucket")

def process_himawari9(year, month, day, hour_min):
    """Process Himawari-9 using original logic"""
    custom_cmap = rbtop3_himawari()
    
    hour = hour_min // 10
    minute = (hour_min % 10) * 10

    bucket_url = f"https://noaa-himawari9.s3.amazonaws.com/?prefix=AHI-L1b-FLDK/{year}/{month:02d}/{day:02d}/{hour:02d}{minute:02d}/"

    start_date = datetime(2022, 12, 13, 5)
    end_date = datetime.now() + timedelta(days=1)
    requested_date = datetime(int(year), int(month), int(day), int(hour), int(minute))

    if requested_date < start_date or requested_date > end_date:
        raise Exception("The requested date is out of this dataset's period of coverage!")

    files = list_files(bucket_url)

    with tempfile.TemporaryDirectory() as output_folder:
        prefix = f"AHI-L1b-FLDK/{year}/{month:02d}/{day:02d}/{hour:02d}{minute:02d}/HS_H09_{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}_B14"
        found_files = [file for file in files if file.startswith(prefix)]

        if found_files:
            extracted_files = []
            for file_name in found_files:
                download_url = f"https://noaa-himawari9.s3.amazonaws.com/{file_name}"
                compressed_file_path = download_file(download_url, output_folder)
                extracted_file_path = extract_bz2(compressed_file_path, output_folder)
                extracted_files.append(extracted_file_path)

            scn = Scene(filenames=extracted_files, reader='ahi_hsd')
            scn.load(['B14'])
            data_array = scn['B14']
            celsius_values = data_array.values - 273.15

            vmin_kelvin = 173.15
            vmax_kelvin = 313.15
            vmin_celsius = vmin_kelvin - 273.15
            vmax_celsius = vmax_kelvin - 273.15

            fig, ax = plt.subplots(figsize=(18, 10), dpi=850)
            img = ax.imshow(celsius_values, cmap=custom_cmap, vmin=vmin_celsius, vmax=vmax_celsius)

            ax.grid(False)
            plt.colorbar(img, ax=ax, orientation='vertical', label='Temperature (°C)')

            dt = datetime(year, month, day, hour, minute)
            title = dt.strftime('Himawari-9 Satellite B14 Data for %B %d, %Y at %H:%M UTC')
            plt.title(title, fontsize=18, weight='bold', pad=10)

            plt.tight_layout()
            plt.figtext(0.5, -0.01, 'Plotted by Sekai Chandra (@Sekai_WX)', ha='center', fontsize=15, weight='bold')

            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='jpg', bbox_inches='tight', pad_inches=0.05)
            img_buffer.seek(0)
            plt.close()

            return img_buffer.getvalue()
        else:
            raise Exception("Files not found in AWS bucket")

# Main UI
st.title("WPAC Basin Satellite Data Archive")
st.write("Continuous satellite data of the WPAC basin from 1981 to present")

if not os.path.exists('gms_conversions.csv'):
    st.warning("⚠️ gms_conversions.csv not found. Using fallback temperature conversion for GMS 1-4 data.")

col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input(
        "Select Date",
        value=datetime(2018, 6, 15).date(),
        min_value=datetime(1981, 3, 1).date(),
        max_value=datetime.now().date()
    )

test_datetime = datetime.combine(selected_date, datetime.min.time())
satellite = get_satellite_for_datetime(test_datetime)

if satellite:
    st.success(f"Using data from: {format_satellite_name(satellite)}")
    
    available_times = get_available_times(selected_date, satellite)
    
    with col2:
        if satellite == "HIMAWARI9":
            time_options = [(f"{h:02d}:{m:02d}", (h, m)) for h, m in available_times]
            selected_time_str = st.selectbox("Select Time (UTC)", [opt[0] for opt in time_options])
            selected_hour, selected_minute = next(opt[1] for opt in time_options if opt[0] == selected_time_str)
        else:
            selected_hour = st.selectbox("Hour (UTC)", available_times)
            selected_minute = 0

    if st.button("Generate Satellite Plot", type="primary"):
        with st.spinner(f"Processing {format_satellite_name(satellite)} data..."):
            try:
                year, month, day = selected_date.year, selected_date.month, selected_date.day
                
                if satellite in ["GMS1", "GMS2", "GMS3", "GMS4"]:
                    image_bytes = process_gms_legacy(year, month, day, selected_hour, satellite)
                elif satellite in ["GMS5", "GOES9"]:
                    image_bytes = process_gms5_goes9(year, month, day, selected_hour, satellite)
                elif satellite in ["MTSAT1", "MTSAT2"]:
                    image_bytes = process_mtsat(year, month, day, selected_hour, satellite)
                elif satellite == "HIMAWARI8":
                    image_bytes = process_himawari8(year, month, day, selected_hour)
                elif satellite == "HIMAWARI9":
                    hour_min = selected_hour * 10 + (selected_minute // 10)
                    image_bytes = process_himawari9(year, month, day, hour_min)
                else:
                    st.error("Unknown satellite system")
                    st.stop()

                st.success("Satellite data processed successfully!")
                st.image(image_bytes, caption=f"{format_satellite_name(satellite)} Satellite Data")
                
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
else:
    st.error("No satellite coverage available for the selected date.")