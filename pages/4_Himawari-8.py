import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from satpy import Scene
import requests
import os
import datetime
import xml.etree.ElementTree as ET
import bz2
import streamlit as st
import tempfile
import io

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

def download_file(url, output_folder):
    """Download file from URL"""
    try:
        file_name = url.split('/')[-1]
        file_path = os.path.join(output_folder, file_name)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, 'wb') as file:
            file.write(response.content)
        return file_path
    except Exception as e:
        raise Exception(f"Download failed: {e}")

def extract_bz2(file_path, output_folder):
    """Extract bz2 compressed file"""
    try:
        with open(file_path, 'rb') as file:
            decompressed_data = bz2.decompress(file.read())
            extracted_file_path = os.path.join(output_folder, os.path.splitext(os.path.basename(file_path))[0])
            with open(extracted_file_path, 'wb') as extracted_file:
                extracted_file.write(decompressed_data)
        os.remove(file_path)
        return extracted_file_path
    except Exception as e:
        raise Exception(f"Extraction failed: {e}")

def list_files(bucket_url):
    """List files in AWS S3 bucket"""
    try:
        response = requests.get(bucket_url, timeout=30)
        response.raise_for_status()
        xml_content = response.content
        root = ET.fromstring(xml_content)
        files = []
        for contents in root.findall('{http://s3.amazonaws.com/doc/2006-03-01/}Contents'):
            key = contents.find('{http://s3.amazonaws.com/doc/2006-03-01/}Key').text
            files.append(key)
        return files
    except Exception as e:
        raise Exception(f"Failed to list bucket files: {e}")

def process_and_plot(year, month, day, hour):
    """Process and plot Himawari-8 data"""
    custom_cmap = rbtop3()
    
    bucket_url = f"https://noaa-himawari8.s3.amazonaws.com/?prefix=AHI-L1b-FLDK/{year}/{month:02d}/{day:02d}/{hour:02d}00/"

    # Check date range
    start_date = datetime.datetime(2015, 7, 7, 2)
    end_date = datetime.datetime(2019, 12, 31, 23)
    split_start_date = datetime.datetime(2020, 1, 1, 0)
    split_end_date = datetime.datetime(2022, 12, 13, 4)
    requested_date = datetime.datetime(int(year), int(month), int(day), int(hour))

    if requested_date < start_date or (requested_date > end_date and requested_date < split_start_date) or requested_date > split_end_date:
        return "The requested date is out of this dataset's period of coverage!"

    # List files
    files = list_files(bucket_url)

    with tempfile.TemporaryDirectory() as output_folder:
        try:
            if requested_date <= end_date:
                # Single file processing
                file_name = f"AHI-L1b-FLDK/{year}/{month:02d}/{day:02d}/{hour:02d}00/HS_H08_{year}{month:02d}{day:02d}_{hour:02d}00_B14_FLDK_R20_S0101.DAT.bz2"
                if file_name in files:
                    download_url = f"https://noaa-himawari8.s3.amazonaws.com/{file_name}"
                    compressed_file_path = download_file(download_url, output_folder)
                    extracted_file_path = extract_bz2(compressed_file_path, output_folder)

                    scn = Scene(filenames=[extracted_file_path], reader='ahi_hsd')
                    scn.load(['B14'])
                    data_array = scn['B14']
                else:
                    return "File not found in AWS bucket"
            else:
                # Multiple file processing
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
                else:
                    return "Files not found in AWS bucket"

            # Convert to Celsius
            celsius_values = data_array.values - 273.15

            # Set temperature range
            vmin_celsius = 173.15 - 273.15
            vmax_celsius = 313.15 - 273.15

            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
            img = ax.imshow(celsius_values, cmap=custom_cmap, vmin=vmin_celsius, vmax=vmax_celsius)

            ax.grid(False)
            ax.axis('off')
            plt.colorbar(img, ax=ax, orientation='vertical', label='Temperature (°C)')

            # Add title
            dt = datetime.datetime(year, month, day, hour)
            title = dt.strftime('Himawari-8 Satellite B14 Data for %B %d, %Y at %H:00 UTC')
            plt.title(title, fontsize=16, weight='bold', pad=10)
            plt.figtext(0.5, 0.02, 'Plotted by Sekai Chandra (@Sekai_WX)', 
                       ha='center', fontsize=10, weight='bold')

            # Save to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='jpg', bbox_inches='tight', pad_inches=0.1, dpi=150)
            img_buffer.seek(0)
            plt.close()

            return img_buffer.getvalue()
            
        except Exception as e:
            return f"Processing error: {e}"

st.title("Himawari-8 Analysis")

st.write("""
**Coverage Period:** July 7, 2015 - December 13, 2022

**Data Format:**
- Single files: July 7, 2015 - December 31, 2019
- Split files: January 1, 2020 - December 13, 2022
- Hourly data available
""")

col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Year", min_value=2015, max_value=2022, value=2018)
    
with col2:
    month = st.number_input("Month", min_value=1, max_value=12, value=6)
    
with col3:
    day = st.number_input("Day", min_value=1, max_value=31, value=15)

hour = st.selectbox("Hour (UTC)", list(range(24)))

if st.button("Generate Plot"):
    with st.spinner("Downloading and processing satellite data..."):
        try:
            result = process_and_plot(year, month, day, hour)
            
            if isinstance(result, bytes):
                st.success("Plot generated successfully!")
                st.image(result, caption="Himawari-8 Satellite Data")
            else:
                st.error(result)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")