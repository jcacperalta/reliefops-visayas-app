import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point
from mpl_toolkits.basemap import Basemap


st.set_page_config(page_title='ReliefOps Visayas Transport Network Efficiency Simulator')

def plot_gdf_on_basemap(gdf, ax, m, color='C1', markersize=5, zorder=0):
    for geometry in gdf.geometry:
        if geometry.geom_type == 'Polygon':
            coords = list(zip(*geometry.exterior.xy))
            x, y = m([lon for lon, lat in coords], [lat for lon, lat in coords])
            ax.fill(x, y, color='#f5ebce',edgecolor='#999999', linewidth=0.25, zorder= zorder)
        elif geometry.geom_type == 'MultiPolygon':
            for polygon in geometry:
                coords = list(zip(*polygon.exterior.xy))
                x, y = m([lon for lon, lat in coords], [lat for lon, lat in coords])
                ax.fill(x, y, color='#f5ebce',edgecolor='#999999', zorder= zorder)
        else:
            x, y = m(geometry.x, geometry.y)
            ax.plot(x, y, color=color, markersize=markersize, marker='o', zorder= zorder)

def plot_baseline_visayas():
    visayas_area = gpd.read_file('data/visayas_provinces.geojson')
    town_centers = gpd.read_file('data/visayas_town_centers_2016.geojson')
    food_hubs = gpd.read_file('data/visayas_food_hubs_2016.geojson')

    fig = plt.figure(figsize=[8,9])
    ax = fig.add_subplot(111)

    lon=[121.017,126.408]
    lat=[8.495,13.134]
    clon=np.mean(lon)
    clat=np.mean(lat)
    mymap = Basemap(llcrnrlon=lon[0], llcrnrlat=lat[0],
                    urcrnrlon=lon[1], urcrnrlat=lat[1],
                    resolution='h', projection='tmerc', lat_0 = clat, lon_0 = clon )
    mymap.fillcontinents(color='#f5f5f5', lake_color='#ffffff')

    plot_gdf_on_basemap(visayas_area, ax, mymap, zorder = 98)
    plot_gdf_on_basemap(food_hubs, ax, mymap, zorder = 99)
    plot_gdf_on_basemap(town_centers, ax, mymap, color='k', markersize=0.35, zorder = 99)
    return fig

st.title('ReliefOps Visayas Transport Network Efficiency Simulator')
st.markdown("---")
st.markdown("### Try the tool")
SIMULATION_TYPES = ["No typhoon (baseline)", "With Typhoon"]

this_simulation = st.selectbox('Select simulation', SIMULATION_TYPES)
if this_simulation == "No typhoon (baseline)":
    fig = plot_baseline_visayas()
    st.pyplot(fig)
st.markdown('')
st.markdown("### Interpret the results")