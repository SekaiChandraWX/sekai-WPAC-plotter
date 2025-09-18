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
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import streamlit as st
import tempfile
import time

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

def find_closest_factors(n, target1, target2):
    factors = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            factors.append((i, n // i))
    closest_factors = min(factors, key=lambda x: abs(x[0] - target1) + abs(x[1] - target2))
    return closest_factors

def conv(dat, csv_file_path):
    """Convert data using CSV mapping"""
    if not os.path.exists(csv_file_path):
        st.error("GMS conversion CSV file not found. Please ensure 'gms_conversions.csv' is in the app directory.")
        return dat
    
    conv_df = pd.read_csv(csv_file_path)
    mapping = dict(zip(conv_df['BRIT'], conv_df['TEMP']))
    for x in range(len(dat)):
        for y in range(len(dat[x])):
            try:
                dat[x][y] = mapping[round(dat[x][y])]
            except:
                dat[x][y] = 0
    return dat

@st.cache_data(ttl=3600)  # Cache for 1 hour to reduce server load
def fetch_file(year, month, day, hour):
    """Fetch and process satellite data file"""
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
        return None, None, "The requested date is out of this dataset's period of coverage! Please check the coverage dates in the sidebar."

    # Check valid hours for the satellite
    if satellite != "GMS4" and hour not in VALID_HOURS:
        return None, None, f"This dataset is only valid every three hours EXCEPT FOR GMS4, which begins on 12/04/1989 and is hourly!"

    # Create temporary directory for downloads
    temp_dir = tempfile.mkdtemp()
    
    # Construct the file path
    ftp_dir = f"{FTP_BASE_PATH}/{satellite}/VISSR/{year}{month:02d}/{day:02d}"
    file_name = f"VISSR_{satellite}_{year}{month:02d}{day:02d}{hour:02d}00.tar"
    local_tar_path = os.path.join(temp_dir, file_name)

    # Download the file using ftplib
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Connecting to FTP server...")
        
        with ftplib.FTP(FTP_HOST, timeout=30) as ftp:
            ftp.login()  # Anonymous login
            progress_bar.progress(20)
            status_text.text("Connected. Navigating to directory...")
            
            ftp.cwd(ftp_dir)
            progress_bar.progress(40)
            status_text.text("Downloading file...")
            
            with open(local_tar_path, 'wb') as local_file:
                ftp.retrbinary(f"RETR {file_name}", local_file.write)
            progress_bar.progress(60)
            status_text.text("File downloaded. Extracting...")

        # Extract the IRYYMMDD.ZHH.gz file from the tar file
        with tarfile.open(local_tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.startswith("./IR") and member.name.endswith(".gz"):
                    member.name = os.path.basename(member.name)
                    tar.extract(member, path=temp_dir)
                    local_gz_path = os.path.join(temp_dir, member.name)
                    progress_bar.progress(80)
                    status_text.text("File extracted. Processing data...")
                    
                    # Process the extracted file
                    final_image_path = process_and_plot(local_gz_path, year, month, day, hour, satellite, temp_dir)
                    progress_bar.progress(100)
                    status_text.text("Processing complete!")
                    return final_image_path, satellite, None

    except ftplib.all_errors as e:
        return None, None, f"Failed to download the file: {e}"
    except tarfile.TarError as e:
        return None, None, f"Failed to extract the file: {e}"
    except Exception as e:
        return None, None, f"Unexpected error: {e}"

    return None, None, "File not found in the tar archive."

def process_and_plot(file, year, month, day, hour, satellite, temp_dir):
    """Process and plot the satellite data"""
    csv_file_path = "gms_conversions.csv"  # Ensure this file is in your repo
    
    # Decompress the gzipped file
    with gzip.open(file, 'rb') as f:
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
    data_converted = (conv(data_array, csv_file_path) - 273.15)

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
    if "rbtop3" not in plt.colormaps():
        rbtop3 = LinearSegmentedColormap.from_list("rbtop3", colors)
        plt.colormaps.register(name='rbtop3', cmap=rbtop3)

    # Plot the data using the custom inverted colormap with Cartopy
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    vmin = V_MIN_SETTINGS[satellite]
    im = ax.imshow(data_square, vmin=vmin, vmax=40, cmap='rbtop3',
                   extent=[100, 180, -60, 60], transform=ccrs.PlateCarree())

    # Remove all borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the plot as a high-quality image
    plot_path = os.path.join(temp_dir, 'satellite_data_plot.jpg')
    plt.savefig(plot_path, format='jpg', dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Open the saved image and stretch it sideways by 75%
    img = Image.open(plot_path)
    width, height = img.size
    new_width = int(width * 1.75)
    img = img.resize((new_width, height), Image.LANCZOS)

    # Add watermarks
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font if Arial Bold isn't available
    try:
        font = ImageFont.truetype("arial.ttf", 50)  # Smaller size for web display
    except:
        font = ImageFont.load_default()
    
    watermark_text_top = f"GMS Data for {year}-{month:02d}-{day:02d} at {hour:02d}:00 UTC"
    watermark_text_bottom = "Plotted by Sekai Chandra @Sekai_WX"
    draw.text((10, 10), watermark_text_top, fill="white", font=font)
    draw.text((10, height - 70), watermark_text_bottom, fill="red", font=font)

    # Save the final image
    final_image_path = os.path.join(temp_dir, 'final_satellite_data_plot.jpg')
    img.save(final_image_path)

    return final_image_path

def main():
    st.set_page_config(
        page_title="GMS 1-4 Satellite Data Archive (1981-1995)",
        layout="centered"
    )
    
    st.title("GMS Satellite Data Viewer")
    
    # Input form in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        year = st.number_input("Year", min_value=1981, max_value=1995, value=1990)
    with col2:
        month = st.number_input("Month", min_value=1, max_value=12, value=1)
    with col3:
        day = st.number_input("Day", min_value=1, max_value=31, value=1)
    with col4:
        hour = st.number_input("Hour (UTC)", min_value=0, max_value=23, value=0)
    
    # Centered generate button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        generate_clicked = st.button("Generate Image")
    
    if generate_clicked:
        with st.spinner("Processing satellite data..."):
            start_time = time.time()
            
            final_image_path, satellite_used, error_message = fetch_file(
                year, month, day, hour
            )
            
            processing_time = time.time() - start_time
            
            if error_message:
                st.error(f"Error: {error_message}")
            elif final_image_path:
                st.success(f"Image generated successfully in {processing_time:.1f} seconds using {satellite_used}!")
                
                # Display the image
                st.image(final_image_path, caption=f"GMS Satellite Data - {year}-{month:02d}-{day:02d} {hour:02d}:00 UTC")
                
                # Provide download button
                with open(final_image_path, "rb") as file:
                    st.download_button(
                        label="Download Image",
                        data=file.read(),
                        file_name=f"GMS_{year}{month:02d}{day:02d}_{hour:02d}00_UTC.jpg",
                        mime="image/jpeg"
                    )
            else:
                st.error("Failed to generate image. Please try again.")

if __name__ == "__main__":
    main()