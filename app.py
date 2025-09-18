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

# Set page config
st.set_page_config(
    page_title="WPAC Satellite Data Plotter (1981-1995)",
    page_icon="🛰️",
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

def process_gms_data(year, month, day, hour):
    """Main function to fetch and process GMS satellite data"""
    request_time = datetime(year, month, day, hour)
    
    # Determine the satellite
    satellite = get_satellite_for_date(request_time)
    if not satellite:
        raise Exception("The requested date is out of this dataset's period of coverage!")
    
    # Validate hour
    if not validate_hour_for_satellite(satellite, hour):
        closest_hour = min(VALID_HOURS, key=lambda x: abs(x - hour))
        raise Exception(f"This dataset is only valid every three hours EXCEPT FOR GMS 4, which begins on 12/04/1989 at 00:00 UTC! Closest valid hour: {closest_hour:02d}:00")
    
    # Construct the file path
    ftp_dir = f"{FTP_BASE_PATH}/{satellite}/VISSR/{year}{month:02d}/{day:02d}"
    file_name = f"VISSR_{satellite}_{year}{month:02d}{day:02d}{hour:02d}00.tar"
    
    # Use temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tar') as tmp_tar:
        local_tar_path = tmp_tar.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.gz') as tmp_gz:
        local_gz_path = tmp_gz.name
    
    try:
        # Download the file using ftplib
        with ftplib.FTP(FTP_HOST) as ftp:
            ftp.login()  # Anonymous login
            ftp.cwd(ftp_dir)
            with open(local_tar_path, 'wb') as local_file:
                ftp.retrbinary(f"RETR {file_name}", local_file.write)
        
        # Extract the IRYYMMDD.ZHH.gz file from the tar file
        extracted = False
        with tarfile.open(local_tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.startswith("./IR") and member.name.endswith(".gz"):
                    # Extract to our temporary gz file
                    f = tar.extractfile(member)
                    if f:
                        with open(local_gz_path, 'wb') as output:
                            output.write(f.read())
                        extracted = True
                        break
        
        if not extracted:
            raise Exception("No valid IR data file found in the archive")
        
        # Process the extracted file
        fig = create_satellite_plot(local_gz_path, year, month, day, hour, satellite)
        return fig
        
    finally:
        # Clean up temporary files
        for tmp_path in [local_tar_path, local_gz_path]:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass  # Ignore cleanup errors

def create_satellite_plot(file_path, year, month, day, hour, satellite):
    """Process the satellite data and create the plot"""
    csv_file_path = "gms_conversions.csv"
    
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
    
    # Add title
    plt.title(f'GMS Data for {year}-{month:02d}-{day:02d} at {hour:02d}:00 UTC ({satellite})', 
              fontsize=16, weight='bold', pad=20)
    
    # Add attribution
    fig.text(0.5, 0.02, 'Plotted by Sekai Chandra (@Sekai_WX)', 
             ha='center', fontsize=12, weight='bold')
    
    return fig

def create_stretched_image_with_watermarks(fig, year, month, day, hour):
    """Create the PIL processed image with stretching and watermarks"""
    # Save matplotlib figure to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        fig.savefig(tmp_file.name, format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
        temp_plot_path = tmp_file.name
    
    try:
        # Open the saved image and stretch it sideways by 75%
        img = Image.open(temp_plot_path)
        width, height = img.size
        new_width = int(width * 1.75)
        img = img.resize((new_width, height), Image.LANCZOS)
        
        # Add watermarks
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 100)
        except:
            try:
                font = ImageFont.truetype("Arial.ttf", 100)
            except:
                font = ImageFont.load_default()
        
        watermark_text_top = f"GMS Data for {year}-{month:02d}-{day:02d} at {hour:02d}:00 UTC"
        watermark_text_bottom = "Plotted by Sekai Chandra @Sekai_WX"
        draw.text((10, 10), watermark_text_top, fill="white", font=font)
        draw.text((10, height - 120), watermark_text_bottom, fill="red", font=font)
        
        # Save the final image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as final_tmp:
            img.save(final_tmp.name, quality=95)
            return final_tmp.name
            
    finally:
        # Clean up the temporary matplotlib save
        if os.path.exists(temp_plot_path):
            try:
                os.remove(temp_plot_path)
            except:
                pass

# Streamlit UI
st.title("🛰️ WPAC Satellite Data Plotter (1981-1995)")
st.markdown("### Generate satellite imagery from GMS 1-4 historical data")

st.markdown("""
This tool allows you to generate satellite imagery from historical GMS (Geostationary Meteorological Satellite) data 
covering the Western Pacific region from 1981-1995.
""")

# Create columns for better layout
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📅 Select Date & Time")
    
    # Date input
    selected_date = st.date_input(
        "Date",
        min_value=date(1981, 3, 1),
        max_value=date(1995, 6, 13),
        value=date(1984, 1, 1)
    )
    
    # Time input - show valid hours based on satellite
    selected_datetime = datetime.combine(selected_date, time(0, 0))
    satellite = get_satellite_for_date(selected_datetime)
    
    if satellite == "GMS4":
        # GMS4 is hourly
        hour_options = list(range(24))
        hour_labels = [f"{h:02d}:00" for h in range(24)]
    else:
        # GMS1-3 are tri-hourly
        hour_options = [0, 3, 6, 9, 12, 15, 18, 21]
        hour_labels = [f"{h:02d}:00" for h in hour_options]
    
    selected_hour = st.selectbox(
        "Hour (UTC)",
        options=hour_options,
        format_func=lambda x: f"{x:02d}:00"
    )
    
    # Display satellite info
    if satellite:
        st.success(f"📡 Satellite: {satellite}")
        if satellite != "GMS4":
            st.info("ℹ️ This satellite provides data every 3 hours")
        else:
            st.info("ℹ️ This satellite provides hourly data")
    else:
        st.error("❌ Selected date is outside the dataset coverage period!")
    
    # Generate button
    generate_button = st.button("🚀 Generate Satellite Image", type="primary")

with col2:
    st.subheader("🛰️ Satellite Image")
    
    if generate_button:
        if not satellite:
            st.error("Please select a valid date within the dataset coverage period.")
        else:
            try:
                with st.spinner("Downloading and processing satellite data... This may take a few minutes."):
                    fig = process_gms_data(
                        selected_date.year,
                        selected_date.month,
                        selected_date.day,
                        selected_hour
                    )
                
                # Display the matplotlib figure
                st.pyplot(fig, use_container_width=True)
                
                # Create and offer download of the processed PIL image
                with st.spinner("Creating high-resolution image with watermarks..."):
                    processed_image_path = create_stretched_image_with_watermarks(
                        fig, selected_date.year, selected_date.month, 
                        selected_date.day, selected_hour
                    )
                
                # Clean up matplotlib figure
                plt.close(fig)
                
                # Offer download
                try:
                    with open(processed_image_path, "rb") as file:
                        st.download_button(
                            label="💾 Download High-Resolution Image",
                            data=file.read(),
                            file_name=f"satellite_{satellite}_{selected_date.strftime('%Y%m%d')}_{selected_hour:02d}00.jpg",
                            mime="image/jpeg"
                        )
                    st.success("✅ Image generated successfully!")
                    st.info("💡 The download includes the stretched, watermarked version as shown in your original script.")
                finally:
                    # Clean up processed image file
                    if os.path.exists(processed_image_path):
                        try:
                            os.remove(processed_image_path)
                        except:
                            pass
                            
            except Exception as e:
                st.error(f"❌ Error generating image: {str(e)}")

# Information sections
with st.expander("📊 Dataset Coverage"):
    st.markdown("""
    **GMS1:** March 1981 - December 1981, January 1984 - June 1984  
    **GMS2:** December 1981 - January 1984  
    **GMS3:** September 1984 - December 1989  
    **GMS4:** December 1989 - June 1995  
    
    **Time Resolution:**
    - GMS1-3: Every 3 hours (00, 03, 06, 09, 12, 15, 18, 21 UTC)
    - GMS4: Every hour (00-23 UTC)
    """)

with st.expander("ℹ️ About this data"):
    st.markdown("""
    **GMS (Geostationary Meteorological Satellite)** data provides infrared brightness temperature measurements 
    over the Western Pacific region.
    
    - **Spatial Coverage**: Western Pacific (100°E-180°E, 60°S-60°N)
    - **Data Source**: Chiba University FTP Archive
    - **Processing**: Custom brightness temperature conversion and meteorological colormap
    - **Output**: High-resolution images with horizontal stretching and watermarks
    
    The images show cloud-top temperatures and surface temperatures where clouds are absent. 
    Colder temperatures (blues/purples) typically indicate higher cloud tops, while warmer 
    temperatures (reds/yellows) indicate lower clouds or surface features.
    """)

st.markdown("---")
st.markdown("*Created by Sekai Chandra (@Sekai_WX)*")