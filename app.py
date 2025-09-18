import streamlit as st

st.set_page_config(
    page_title="WPAC Typhoon Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("WPAC Basin Typhoon Analysis")

st.header("Western Pacific Basin Typhoon Analysis System")

st.write("""
This application provides full-disk satellite imagery analysis from multiple satellite systems 
spanning several decades of meteorological observations in the Western Pacific basin.
""")

st.header("Available Satellite Datasets")

col1, col2 = st.columns(2)

with col1:
    st.subheader("GMS 1-4")
    st.write("**Period:** 1977-1995")
    st.write("Japan's first generation geostationary meteorological satellites")
    st.write("Full-disk imagery of Western Pacific region")
    
    st.subheader("GMS 5 & GOES 9")
    st.write("**Period:** 1995-2003")
    st.write("Enhanced capabilities with improved spatial and temporal resolution")
    st.write("Advanced full-disk imaging with better typhoon tracking")

with col2:
    st.subheader("MTSAT")
    st.write("**Period:** 2003-2015")
    st.write("Multi-functional Transport Satellite with advanced meteorological sensors")
    st.write("High-resolution imagery with multiple spectral channels")
    
    st.subheader("Himawari Modern")
    st.write("**Period:** 2015-Present")
    st.write("Latest generation with unprecedented temporal resolution (10-minute full-disk)")
    st.write("Ultra-high resolution imagery with advanced atmospheric analysis")

st.header("Navigation")

st.write("""
Use the sidebar to navigate between different satellite datasets. Each page provides:

- Satellite-specific data processing
- Full-disk plot generation  
- Typhoon analysis tools
- Meteorological visualizations

**Quick Start:**
1. Select a satellite system from the sidebar
2. Choose your date and time
3. Generate full-disk plots
4. Analyze typhoon patterns
""")

st.header("System Status")
st.success("All satellite processors online")
st.info("Ready for data analysis")

st.markdown("---")
st.markdown("**WPAC Typhoon Analysis System | Meteorological Data Processing Platform**")
st.markdown("Select a satellite dataset from the sidebar to begin analysis")

st.sidebar.markdown("---")
st.sidebar.info("Select a satellite dataset from the pages above to start your analysis!")
st.sidebar.markdown("---")
st.sidebar.markdown("**System Info:**")
st.sidebar.markdown("- Real-time processing")
st.sidebar.markdown("- Multi-satellite support") 
st.sidebar.markdown("- Advanced typhoon tracking")