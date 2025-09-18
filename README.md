# 🛰️ WPAC Satellite Data Plotter (1981-1995)

A Streamlit web application for generating satellite imagery from historical GMS (Geostationary Meteorological Satellite) data covering the Western Pacific region from 1981-1995.

## 🌟 Features

- **Interactive Web Interface**: Easy-to-use date and time picker
- **Historical Coverage**: Access to GMS 1-4 satellite data (1981-1995)
- **Automatic Satellite Selection**: App automatically determines which satellite to use based on date
- **Time Validation**: Ensures selected times are valid for each satellite
- **High-Quality Output**: Generates publication-ready satellite imagery
- **Real-time Processing**: Downloads and processes data on-demand
- **Custom Colormap**: Uses specialized meteorological color scheme

## 🛰️ Satellite Coverage

| Satellite | Coverage Period | Time Resolution |
|-----------|----------------|-----------------|
| **GMS1** | Mar 1981 - Dec 1981<br>Jan 1984 - Jun 1984 | Every 3 hours |
| **GMS2** | Dec 1981 - Jan 1984 | Every 3 hours |
| **GMS3** | Sep 1984 - Dec 1989 | Every 3 hours |
| **GMS4** | Dec 1989 - Jun 1995 | Every hour |

**Valid Times for GMS1-3**: 00, 03, 06, 09, 12, 15, 18, 21 UTC  
**Valid Times for GMS4**: Every hour (00-23 UTC)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd wpac-satellite-plotter
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have the conversion file**
   Make sure `gms_conversions.csv` is in the root directory of the project. This file contains the brightness temperature conversion mappings.

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   The app will automatically open in your browser at `http://localhost:8501`

## 📊 How to Use

1. **Select a Date**: Choose any date within the satellite coverage periods using the date picker
2. **Select a Time**: Choose a valid time (the app will warn you if the time is invalid for the selected satellite)
3. **Generate Plot**: Click the "Generate Satellite Plot" button
4. **View Results**: The high-resolution satellite image will appear in the main panel
5. **Download**: Use the download button to save the image locally

## 🔧 Technical Details

### Data Source
- **FTP Server**: `gms.cr.chiba-u.ac.jp`
- **Data Format**: VISSR (Visible and Infrared Spin Scan Radiometer) infrared data
- **File Format**: Gzipped binary data in TAR archives
- **Spatial Coverage**: Western Pacific (100°E-180°E, 60°S-60°N)

### Image Processing Pipeline
1. **Download**: Fetches compressed satellite data from FTP server
2. **Extract**: Uncompresses TAR and GZIP archives
3. **Convert**: Applies brightness temperature conversion using CSV lookup
4. **Reshape**: Processes raw binary data into 2D arrays
5. **Enhance**: Applies custom meteorological colormap
6. **Post-process**: Stretches image horizontally by 75% and adds watermarks
7. **Export**: Saves as high-resolution JPEG

### Key Features of Generated Images
- **Resolution**: 2000 DPI for publication quality
- **Projection**: Plate Carrée (Geographic)
- **Color Scheme**: Custom "rbtop3" meteorological colormap
- **Watermarks**: Date/time stamp and attribution
- **Format**: JPEG with horizontal stretching for optimal viewing

## 📁 File Structure
```
wpac-satellite-plotter/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── gms_conversions.csv   # Temperature conversion lookup table
└── README.md             # This file
```

## ⚠️ Important Notes

- **Internet Connection Required**: The app downloads data in real-time from the FTP server
- **Processing Time**: Initial downloads may take 1-3 minutes depending on connection speed
- **File Size**: Generated images are high-resolution and may be several MB in size
- **Temporary Files**: The app creates temporary files during processing that are automatically cleaned up

## 🐛 Troubleshooting

### Common Issues

**"Date is out of coverage period"**
- Check that your selected date falls within the satellite coverage periods listed above

**"Invalid time for satellite"**
- GMS1-3: Only available every 3 hours (00, 03, 06, 09, 12, 15, 18, 21 UTC)
- GMS4: Available every hour

**"Failed to download file"**
- Check your internet connection
- The FTP server may be temporarily unavailable
- Try a different date/time

**Font errors in generated images**
- The app will fallback to default fonts if system fonts are not available
- On Linux systems, you may need to install additional font packages

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📜 License

This project is for educational and research purposes. Please respect the data source and provide appropriate attribution when using the generated imagery.

## 🙏 Acknowledgments

- **Data Source**: Chiba University for maintaining the GMS historical archive
- **Original Processing**: Based on Sekai Chandra's processing methodology
- **Cartopy**: For geographic projections and mapping functionality