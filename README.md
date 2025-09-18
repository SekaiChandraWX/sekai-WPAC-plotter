# WPAC Typhoon Analysis - Streamlit Application

A comprehensive satellite data analysis tool for the Western Pacific basin, supporting multiple satellite systems from 1981 to present.

## File Structure

```
your-app
├── app.py                    # Main application file
├── pages
│   ├── 1_GMS_1-4.py         # GMS 1-4 satellite analysis
│   ├── 2_GMS_5_GOES_9.py    # GMS 5 & GOES 9 analysis
│   ├── 3_MTSAT.py           # MTSAT satellite analysis
│   ├── 4_Himawari-8.py      # Himawari-8 analysis
│   └── 5_Himawari-9.py      # Himawari-9 analysis
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Satellite Coverage

- GMS 1-4 1981-1995 (3-hourly, hourly for GMS4 after Dec 1989)
- GMS 5 & GOES 9 1995-2005 (hourly)
- MTSAT 2005-2015 (hourly)
- Himawari-8 2015-2022 (hourly)
- Himawari-9 2022-Present (10-minute intervals)

## Deployment

### Local Development

1. Clonedownload all files
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application
   ```bash
   streamlit run app.py
   ```

### Cloud Deployment (Streamlit Cloud)

1. Upload all files to a GitHub repository
2. Connect your GitHub repo to Streamlit Cloud
3. Deploy with the main file set to `app.py`

## Important Notes & Limitations

### Missing Dependencies for Full Functionality

1. GMS Conversion CSV The GMS 1-4 processor requires a `gms_conversions.csv` file for temperature calibration. This file is not included and needs to be provided.

2. Font Files Scripts reference `arialbd.ttf` which may not be available in cloud environments. The app will work but without custom fonts.

### Potential Cloud Deployment Issues

1. FTP Connections GMS 1-4, GMS 5GOES 9, and MTSAT pages use FTP to download data from `gms.cr.chiba-u.ac.jp`. Cloud environments may block FTP connections.

2. Memory Usage Satellite data processing is memory-intensive. Large files may cause memory issues on free cloud tiers.

3. Processing Time Data download and processing can take several minutes, which may timeout on some cloud platforms.

4. File Permissions Some cloud environments have restricted file system access.

### Data Sources

- GMSMTSAT Data from Chiba University FTP server
- Himawari-8 NOAA AWS S3 bucket (noaa-himawari8)
- Himawari-9 NOAA AWS S3 bucket (noaa-himawari9)

## Usage

1. Select a satellite system from the sidebar navigation
2. Choose your desired date and time within the coverage period
3. Click Generate Plot to download and process the data
4. View the generated full-disk satellite imagery

## Troubleshooting

- FTP Timeouts Try different times or check if the FTP server is accessible
- AWS S3 Errors Verify the date is within the available range
- Memory Errors Try using smaller time periods or restart the application
- Font Warnings Ignore font-related warnings - they won't affect functionality

## Dependencies

See `requirements.txt` for the complete list of Python packages required.