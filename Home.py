import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point
import fiona
from simplification.cutil import simplify_coords_vw 
from mpl_toolkits.basemap import Basemap
import time
import re


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

def plot_food_hubs_on_basemap(gdf, ax, m, colors, markersize=8, zorder=0):
    for geometry, color in zip(gdf.geometry, colors):
        x, y = m(geometry.x, geometry.y)
        ax.plot(x, y, c=color, markersize=markersize, marker='o', zorder=zorder)

def is_match_hub_psgc(hub,psgc):
    return int(re.search(r'\d+', hub).group())==psgc//10000000

def get_nodes_table(Gw):
    nodes = pd.DataFrame(Gw.nodes(data=True), columns=['node','data'])
    nodes['lon'] = nodes['data'].apply(lambda r:r['x'])
    nodes['lat'] = nodes['data'].apply(lambda r:r['y'])
    nodes = nodes[['node','lon','lat']]
    return nodes

def routing_visayas(Gw, food_hubs, town_centers):

    print("\nStarting routing from food hubs...")
    fh = food_hubs['name'].values
    od_matrix = []
    for i in np.arange(len(fh)): 
        print(f"Food Hub: {fh[i]}")
        src_list = food_hubs['nearest_node'].values[i]
        dest_list = town_centers['nearest_node'].values
        for src in [src_list]:
            t=time.time()
            print("Calculating dest distances for src="+str(src))
            for dest in dest_list:
                ttime=0
                path=[]
                try:
                    # path=nx.shortest_path(Gw, source=src,target=dest,weight='travel_time') 
                    ttime,path = nx.bidirectional_dijkstra(Gw, source=src,target=dest,weight='travel_time')
                    # path=nx.astar_path(Gw, source=src,target=dest,weight='travel_time') 
                    # ttime,path=nx.bidirectional_dijkstra(Gw, source=src,target=dest, weight='travel_time')
                    # path_edges=pairwise(path)
                    # for pe in path_edges:
                    #    ttime=ttime + Gw.get_edge_data(pe[0],pe[1])[0]['travel_time']
                except:
                    ttime=-1
                od_matrix.append([src,dest,fh[i],ttime,path])

            toc=time.time()
            print(f"Routing from {fh[i]} done in {toc-t:0.2f} secs")

    rdf = pd.DataFrame(od_matrix, columns=['src_node','dest_node','food_hub','travel_time','path'])
    rdf = rdf.merge(town_centers[['nearest_node','PSGC','POP_2015','POP_2020']], left_on='dest_node', right_on='nearest_node').rename(columns={'PSGC':'dest_psgc'})
    pwrdf = rdf[rdf[['food_hub','dest_psgc']].apply(lambda x: is_match_hub_psgc(x['food_hub'],x['dest_psgc']), axis=1)]
    pwrdf = pwrdf.sort_values(by=['dest_psgc', 'travel_time'])
    pwrdf = pwrdf.groupby('dest_psgc').head(1).reset_index(drop=True)
    pwrdf = pwrdf.rename(columns={'travel_time':'min_travel_time','src_psgc':'food_hub_assigned'})
    
    return pwrdf


def plot_baseline_visayas():
    visayas_area = gpd.read_file('data/visayas_provinces.geojson')
    town_centers = gpd.read_file('data/visayas_town_centers.geojson')
    food_hubs = gpd.read_file('data/visayas_food_hubs_2016.geojson')

    Gw = nx.read_gml("data/visayas_transport_network.gml")
    Gw = nx.relabel_nodes(Gw, lambda x: int(x))
    nodes = get_nodes_table(Gw)
    pwrdf = routing_visayas(Gw, food_hubs, town_centers)
    hub_to_color = dict(zip(food_hubs['name'],[f'C{i}' if i not in [1,2] else 'g' for i in food_hubs.index]))
    pwrdf['route_color'] = pwrdf['food_hub'].apply(lambda x: hub_to_color[x])

    ################################
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

    plot_gdf_on_basemap(visayas_area, ax, mymap, zorder = 1)
    plot_food_hubs_on_basemap(food_hubs, ax, mymap, colors=list(hub_to_color.values()), markersize=5, zorder = 99)
    plot_gdf_on_basemap(town_centers, ax, mymap, color='k', markersize=0.35, zorder = 99)

    path_data = pwrdf[pwrdf['min_travel_time']>-1]
    for fh_color,path in path_data[['route_color','path']].values:
        path_df = pd.DataFrame(path, columns=['node'])
        path_df = path_df.merge(nodes, on='node', how='left')
        simplified_path_coords = simplify_coords_vw(list(zip(path_df['lon'].values,path_df['lat'].values)), 0.0001)
        x,y = mymap([p[0] for p in simplified_path_coords],[p[1] for p in simplified_path_coords])
        mymap.plot(x,y,linestyle='-',color=fh_color,lw=0.7, alpha=0.35, zorder=2)
    return fig

st.title('ReliefOps Visayas Transport Network Efficiency Simulator')
st.markdown("---")
st.markdown("### Try the tool")
SIMULATION_TYPES = ["No typhoon (baseline)", "With Typhoon"]

this_simulation = st.selectbox('Select simulation', SIMULATION_TYPES)
start_run = st.button("Run")
if start_run:
    tic=time.time()
    with st.spinner("Run in progress..."):
        if this_simulation == "No typhoon (baseline)":
            fig = plot_baseline_visayas()
            st.pyplot(fig)
            toc=time.time()
    st.success(f"Run completed in {toc-tic:.1f} secs!")
st.markdown('')
st.markdown("### Interpret the results")