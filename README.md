# 🛰️ GMS Satellite Data Viewer

A Streamlit web application for viewing historical Geostationary Meteorological Satellite (GMS) infrared imagery from 1981-1995.

## 🌟 Features

- **Historical Coverage**: Access GMS1-GMS4 satellite data spanning March 1981 to June 1995
- **Interactive Interface**: Simple date and time selection with automatic satellite detection
- **High-Quality Visualization**: Custom colormap for infrared temperature data
- **Download Capability**: Save generated images as high-resolution JPEGs
- **Caching**: Built-in caching to reduce server load and improve performance

## 📊 Satellite Coverage

| Satellite | Coverage Period | Data Frequency |
|-----------|----------------|----------------|
| **GMS1** | March 1, 1981 - June 29, 1984 | Every 3 hours |
| **GMS2** | December 21, 1981 - January 21, 1984 | Every 3 hours |
| **GMS3** | September 27, 1984 - December 4, 1989 | Every 3 hours |
| **GMS4** | December 4, 1989 - June 13, 1995 | Every hour |

*Note: Tri-hourly data is available at 00, 03, 06, 09, 12, 15, 18, 21 UTC. GMS4 provides hourly data (00-23 UTC).*

## 🚀 Deployment

### Deploy on Streamlit Cloud

1. **Fork this repository** to your GitHub account

2. **Add the required CSV file**: You need to add a `gms_conversions.csv` file to the repository root. This file should contain brightness-to-temperature conversion data with columns:
   - `BRIT`: Brightness values
   - `TEMP`: Temperature values

3. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your forked repository
   - Set the main file path to `app.py`
   - Click "Deploy"

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/gms-satellite-viewer.git
cd gms-satellite-viewer

# Install dependencies
pip install -r requirements.txt

# Add your gms_conversions.csv file to the repository root

# Run the application
streamlit run app.py
```

## 📋 Required Files

Make sure your repository contains these files:

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `gms_conversions.csv` - Brightness-to-temperature conversion data (you need to provide this)
- `README.md` - This documentation

## 💡 Usage

1. **Select Date**: Choose any date within the satellite coverage period (1981-1995)
2. **Select Hour**: Pick an available hour (tri-hourly for GMS1-3, hourly for GMS4)
3. **Generate Image**: Click the "Generate Satellite Image" button
4. **Download**: Save the generated high-resolution image

## ⚡ Performance Notes

- **Caching**: Data is cached for 1 hour to reduce FTP server load
- **Processing Time**: Initial requests may take 30-60 seconds due to FTP downloads
- **Rate Limiting**: Built-in delays help manage server resources
- **File Sizes**: Generated images are high-resolution and may be several MB

## 🛠️ Technical Details

### Data Source
- **FTP Server**: gms.cr.chiba-u.ac.jp
- **Format**: Compressed tar files containing gzipped VISSR data
- **Processing**: Data is decompressed, reshaped, and converted to temperature values

### Visualization
- **Colormap**: Custom "rbtop3" colormap optimized for infrared imagery
- **Projection**: Plate Carrée projection covering 100°E-180°E, 60°S-60°N
- **Resolution**: Images are generated at 300 DPI for high quality

### Libraries Used
- **Streamlit**: Web application framework
- **Matplotlib + Cartopy**: Geospatial visualization
- **NumPy + SciPy**: Data processing
- **Pillow**: Image manipulation
- **Pandas**: CSV data handling

## 🔧 Troubleshooting

### Common Issues

**"GMS conversion CSV file not found"**
- Ensure `gms_conversions.csv` is in your repository root
- Check that the file has the correct column names (`BRIT`, `TEMP`)

**"Failed to download the file"**
- The FTP server may be temporarily unavailable
- Check your internet connection
- Try a different date/time

**"Date is out of coverage period"**
- Verify the date is within 1981-1995
- Check the satellite coverage table above

### Font Issues
The application will fall back to default fonts if Arial is not available on the deployment server.

## 📝 License

This project is for educational and research purposes. Please respect the data source and cite appropriately when using generated images.

## 👨‍💻 Credits

- **Original Script**: Sekai Chandra (@Sekai_WX)
- **Data Source**: Chiba University GMS Archive
- **Streamlit Conversion**: [Your name here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

*For questions or support, please open an issue on GitHub.*