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
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, date, time
import tempfile
import shutil

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

def get_satellite_for_date(request_time):
    """Determine which satellite covers the requested time"""
    for sat, ranges in GMS_RANGES.items():
        for time_range in zip(ranges[::2], ranges[1::2]):
            if time_range[0] <= request_time <= time_range[1]:
                return sat
    return None

def validate_hour_for_satellite(satellite, hour):
    """Check if the hour is valid for the given satellite"""
    if satellite != "GMS4" and hour not in VALID_HOURS:
        return False
    return True

def conv(dat, csv_file_path):
    """Convert brightness temperature data using CSV mapping"""
    conv_df = pd.read_csv(csv_file_path)
    mapping = dict(zip(conv_df['BRIT'], conv_df['TEMP']))
    for x in range(len(dat)):
        for y in range(len(dat[x])):
            try:
                dat[x][y] = mapping[round(dat[x][y])]
            except:
                dat[x][y] = 0
    return dat

def find_closest_factors(n, target1, target2):
    """Find the closest factors to target dimensions"""
    factors = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            factors.append((i, n // i))
    closest_factors = min(factors, key=lambda x: abs(x[0] - target1) + abs(x[1] - target2))
    return closest_factors

def fetch_and_process_satellite_data(year, month, day, hour, progress_callback=None):
    """Main function to fetch and process satellite data"""
    request_time = datetime(year, month, day, hour)
    
    # Determine the satellite
    satellite = get_satellite_for_date(request_time)
    if not satellite:
        return None, "The requested date is out of this dataset's period of coverage!"
    
    # Validate hour
    if not validate_hour_for_satellite(satellite, hour):
        closest_hour = min(VALID_HOURS, key=lambda x: abs(x - hour))
        return None, f"This dataset is only valid every three hours EXCEPT FOR GMS 4, which begins on 12/04/1989 at 00:00 UTC! Closest valid hour: {closest_hour:02d}:00"
    
    if progress_callback:
        progress_callback(0.1, "Connecting to FTP server...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Construct the file path
        ftp_dir = f"{FTP_BASE_PATH}/{satellite}/VISSR/{year}{month:02d}/{day:02d}"
        file_name = f"VISSR_{satellite}_{year}{month:02d}{day:02d}{hour:02d}00.tar"
        local_tar_path = os.path.join(temp_dir, file_name)
        
        if progress_callback:
            progress_callback(0.2, "Downloading satellite data...")
        
        # Download the file using ftplib
        with ftplib.FTP(FTP_HOST) as ftp:
            ftp.login()  # Anonymous login
            ftp.cwd(ftp_dir)
            with open(local_tar_path, 'wb') as local_file:
                ftp.retrbinary(f"RETR {file_name}", local_file.write)
        
        if progress_callback:
            progress_callback(0.4, "Extracting data files...")
        
        # Extract the IRYYMMDD.ZHH.gz file from the tar file
        with tarfile.open(local_tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.startswith("./IR") and member.name.endswith(".gz"):
                    member.name = os.path.basename(member.name)  # Strip the leading directory
                    tar.extract(member, path=temp_dir)
                    local_gz_path = os.path.join(temp_dir, member.name)
                    
                    if progress_callback:
                        progress_callback(0.6, "Processing satellite data...")
                    
                    # Process the extracted file
                    final_image_path = process_and_plot(local_gz_path, year, month, day, hour, satellite, temp_dir, progress_callback)
                    return final_image_path, None
        
        return None, "No valid IR data file found in the archive"
        
    except ftplib.all_errors as e:
        return None, f"Failed to download the file: {e}"
    except tarfile.TarError as e:
        return None, f"Failed to extract the file: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"
    finally:
        # Clean up temporary files (except the final image)
        pass  # We'll keep temp files for now since we need to return the image path

def process_and_plot(file_path, year, month, day, hour, satellite, temp_dir, progress_callback=None):
    """Process the satellite data and create the plot"""
    csv_file_path = "gms_conversions.csv"  # Assume this is in the same directory as the app
    
    if progress_callback:
        progress_callback(0.7, "Decompressing data...")
    
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
    
    if progress_callback:
        progress_callback(0.8, "Converting temperature data...")
    
    # Convert the data using the provided CSV mapping
    data_converted = (conv(data_array, csv_file_path) - 273.15)
    
    # Stretch the data to make it as square as possible
    rows, cols = data_converted.shape
    size = max(rows, cols)
    data_square = zoom(data_converted, (size / rows, size / cols))
    
    if progress_callback:
        progress_callback(0.9, "Creating visualization...")
    
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
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
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
    
    # Save the plot as a high-quality JPG image
    plot_path = os.path.join(temp_dir, 'satellite_data_plot.jpg')
    plt.savefig(plot_path, format='jpg', dpi=2000, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Open the saved image and stretch it sideways by 75%
    img = Image.open(plot_path)
    width, height = img.size
    new_width = int(width * 1.75)
    img = img.resize((new_width, height), Image.LANCZOS)
    
    # Add watermarks
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 200)  # Try arial.ttf first
    except:
        try:
            font = ImageFont.truetype("Arial.ttf", 200)  # Try Arial.ttf
        except:
            font = ImageFont.load_default()  # Fallback to default font
    
    watermark_text_top = f"GMS Data for {year}-{month:02d}-{day:02d} at {hour:02d}:00 UTC"
    watermark_text_bottom = "Plotted by Sekai Chandra @Sekai_WX"
    draw.text((10, 10), watermark_text_top, fill="white", font=font)
    draw.text((10, height - 250), watermark_text_bottom, fill="red", font=font)
    
    # Save the final image
    final_image_path = os.path.join(temp_dir, 'final_satellite_data_plot.jpg')
    img.save(final_image_path)
    
    if progress_callback:
        progress_callback(1.0, "Complete!")
    
    return final_image_path

# Streamlit App
def main():
    st.set_page_config(
        page_title="WPAC Satellite Data Plotter (1981-1995)",
        page_icon="🛰️",
        layout="wide"
    )
    
    st.title("🛰️ WPAC Satellite Data Plotter (1981-1995)")
    st.markdown("Generate satellite imagery from GMS 1-4 historical data")
    
    # Sidebar for input controls
    st.sidebar.header("📅 Select Date and Time")
    
    # Date input
    selected_date = st.sidebar.date_input(
        "Date",
        min_value=date(1981, 3, 1),
        max_value=date(1995, 6, 13),
        value=date(1984, 1, 1)
    )
    
    # Time input
    selected_time = st.sidebar.time_input("Time (UTC)", value=time(0, 0))
    
    # Create datetime object
    selected_datetime = datetime.combine(selected_date, selected_time)
    
    # Determine satellite and validate time
    satellite = get_satellite_for_date(selected_datetime)
    
    # Display satellite info
    if satellite:
        st.sidebar.success(f"📡 Satellite: {satellite}")
        
        # Check if hour is valid
        hour = selected_time.hour
        if not validate_hour_for_satellite(satellite, hour):
            if satellite != "GMS4":
                closest_hour = min(VALID_HOURS, key=lambda x: abs(x - hour))
                st.sidebar.warning(f"⚠️ {satellite} data is only available every 3 hours (00, 03, 06, 09, 12, 15, 18, 21 UTC)")
                st.sidebar.info(f"💡 Closest valid time: {closest_hour:02d}:00 UTC")
    else:
        st.sidebar.error("❌ Selected date is outside the dataset coverage period!")
    
    # Display coverage information
    with st.sidebar.expander("📊 Dataset Coverage"):
        st.write("**GMS1:** Mar 1981 - Dec 1981, Jan 1984 - Jun 1984")
        st.write("**GMS2:** Dec 1981 - Jan 1984")
        st.write("**GMS3:** Sep 1984 - Dec 1989")
        st.write("**GMS4:** Dec 1989 - Jun 1995")
        st.write("")
        st.write("**Time Resolution:**")
        st.write("• GMS1-3: Every 3 hours")
        st.write("• GMS4: Every hour")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        # Generate plot button
        if st.button("🚀 Generate Satellite Plot", type="primary", use_container_width=True):
            if not satellite:
                st.error("Please select a valid date within the dataset coverage period.")
                return
            
            if not validate_hour_for_satellite(satellite, hour):
                st.error("Please select a valid time for the chosen satellite.")
                return
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(value, text):
                progress_bar.progress(value)
                status_text.text(text)
            
            # Process the data
            try:
                final_image_path, error_message = fetch_and_process_satellite_data(
                    selected_date.year, 
                    selected_date.month, 
                    selected_date.day, 
                    selected_time.hour,
                    update_progress
                )
                
                if error_message:
                    st.error(f"Error: {error_message}")
                else:
                    st.success("✅ Plot generated successfully!")
                    
                    # Store the image path in session state
                    st.session_state.current_image_path = final_image_path
                    st.session_state.current_image_info = {
                        'date': selected_date,
                        'time': selected_time,
                        'satellite': satellite
                    }
                    
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
            finally:
                progress_bar.empty()
                status_text.empty()
    
    # Display the generated image
    with col1:
        if hasattr(st.session_state, 'current_image_path') and os.path.exists(st.session_state.current_image_path):
            st.subheader("🖼️ Generated Satellite Image")
            
            # Display image info
            info = st.session_state.current_image_info
            st.info(f"📅 **Date:** {info['date'].strftime('%Y-%m-%d')} | "
                   f"⏰ **Time:** {info['time'].strftime('%H:%M')} UTC | "
                   f"🛰️ **Satellite:** {info['satellite']}")
            
            # Display the image
            image = Image.open(st.session_state.current_image_path)
            st.image(image, use_column_width=True)
            
            # Download button
            with open(st.session_state.current_image_path, "rb") as file:
                btn = st.download_button(
                    label="💾 Download High-Resolution Image",
                    data=file.read(),
                    file_name=f"satellite_{info['satellite']}_{info['date'].strftime('%Y%m%d')}_{info['time'].strftime('%H%M')}.jpg",
                    mime="image/jpeg"
                )
        else:
            st.info("👈 Select a date and time, then click 'Generate Satellite Plot' to view the satellite imagery.")

if __name__ == "__main__":
    main()