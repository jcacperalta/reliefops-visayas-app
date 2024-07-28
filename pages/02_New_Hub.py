import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import re
import ast

st.set_page_config(page_title="New Hub Simulator")
town_centers = pd.read_csv("data/visayas_town_centers_2024.csv")

def get_hub_assignment(routes_df, capital_psgc, new_hub_psgc=None):
    # Get minimum travel time
    rdf = routes_df[routes_df['src_psgc'].isin([capital_psgc, new_hub_psgc])]
    prdf = rdf.loc[rdf.groupby('dest_psgc')['travel_time'].idxmin()]
    # remove case where 2 hubs are OD
    if new_hub_psgc:
        hub_rows = ((prdf['dest_psgc']==capital_psgc)&(prdf['src_psgc']==new_hub_psgc))|\
                    ((prdf['dest_psgc']==new_hub_psgc)&(prdf['src_psgc']==capital_psgc))
        prdf = prdf[~hub_rows]
    prdf['travel_time'] = prdf['travel_time']/3600
    prdf = prdf.sort_values(by=['src_psgc','travel_time']).reset_index()
    return prdf

def time_to_readable(seconds):
    seconds = seconds*3600
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}mins"
    else:
        return f"{minutes} mins"

def colorize(val, num_col):
    norm = (num_col - num_col.min()) / (num_col.max() - num_col.min())
    color = plt.cm.YlGn(norm)
    return [f'background-color: rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, 0.5)' for c in color]

########################################################################
########################################################################
########################################################################
st.title('ReliefOps Visayas Transport Network Model - New Hub Simulator')
st.markdown("---")
st.write("This tool simulates adding a new food hub in a province and estimates how well it can reduce relief delivery times to all cities and municipalities. ")

st.markdown("""**Technical note**: The relief delivery times mentioned are estimated travel times to the LGUs from the food hub nearest to them. In this idealized simulation:
- Each LGU has one dedicated delivery vehicle, i.e. the vehicles do not do stopovers in towns they would pass by, but only on its destination.
- Deliveries to all LGUs start simultaneously from the food hubs
- All land and sea routes are considered fully passable.

These assumptions will not hold in real-life scenarios. Still, the maps generated by the simulation can serve to illustrate the potential impact of activating new food hubs to more efficient relief distribution for nearby areas and may be used as a starting point for more detailed projections. """)
st.markdown("---")

region_lookup = {"6 - WESTERN VISAYAS":'6', "7 - CENTRAL VISAYAS":'7', "8 - EASTERN VISAYAS":'8', "NIR - NEGROS ISLAND REGION":"NIR"}

# Declare states
if "region_picked" not in st.session_state:
    st.session_state["region_picked"] = False
if "province_picked" not in st.session_state:
    st.session_state["province_picked"] = False
if "municity_picked" not in st.session_state:
    st.session_state["municity_picked"] = False

def disable():
    st.session_state["disabled"] = True

region_pick = st.selectbox(
        label = "Choose region of focus:",
        options = ("", "6 - WESTERN VISAYAS", "7 - CENTRAL VISAYAS", "8 - EASTERN VISAYAS", "NIR - NEGROS ISLAND REGION"),
        format_func=lambda x: 'Select an option' if x == '' else x,
        disabled= st.session_state["region_picked"]
        )

if region_pick:
    region = region_lookup[region_pick]
    st.session_state["region_picked"] = True    
    
##################
if st.session_state["region_picked"]:
    provinces = [""]+sorted([p for p in town_centers[town_centers['region']==region]['province'].unique()])
    province = st.selectbox(
            label = "Choose province of focus:",
            options = provinces,
            format_func=lambda x: 'Select an option' if x == '' else x,
            disabled= st.session_state["province_picked"]
        )
    prov_str = province.lower().replace(' ','_')
    if province:
        st.session_state["province_picked"] = True

if st.session_state["province_picked"]:
    capital_name = town_centers[(town_centers['province']==province)&(town_centers['is_capital']=='Y')]['municipality'].values[0]
    capital_psgc = town_centers[(town_centers['province']==province)&(town_centers['is_capital']=='Y')]['psgc'].values[0]
    st.write(f'You picked province {province}, where the currently set food hub is located at its capital, {capital_name}.')
    municities = [""]+sorted([p for p in town_centers[(town_centers['province']==province)]['municipality'].unique()])
    municities = [m for m in municities if m !=capital_name]
    municity_name = st.selectbox(
            label = "Choose city/municipality to serve as NEW food hub:",
            options = municities,
            format_func=lambda x: 'Select an option' if x == '' else x,
            disabled= st.session_state["municity_picked"]
        )
    if municity_name:
        st.session_state["municity_picked"] = True

if st.session_state["municity_picked"]:
    st.write(f"You chose **{municity_name}, {province}** as the location of the new food hub.")
    st.write("Upon verifying that this is your desired new hub location, click the button below to run the model. Otherwise, refresh the page to input again to pick a different hub.")
    start_run = st.button("Run model")

    if start_run:
        with st.spinner(f"Calculating routes from {capital_name} and {municity_name}..."):
            new_hub_psgc = town_centers[(town_centers['province']==province)&(town_centers['municipality']==municity_name)]['psgc'].values[0]
            routes_df = pd.read_csv(f'data/routes/reg{region}_{prov_str}_routes.csv')
            baseline_prdf = get_hub_assignment(routes_df, capital_psgc).rename(columns = {'travel_time':'baseline_travel_time'})
            new_prdf = get_hub_assignment(routes_df, capital_psgc, new_hub_psgc).rename(columns = {'travel_time':'new_travel_time'})
            time.sleep(2)

        st.write('### Food hub assignment')        
        col1, col2 = st.columns(2)
        with col1:
            st.image(f"results/baseline/region{region}_{prov_str}_baseline.png", caption="Baseline")
        with col2:
            st.image(f"results/new_hub/region{region}_{prov_str}_newhub_{new_hub_psgc}.png" , caption=" With new hub")

        st.write('### Travel time') 
        #baseline
        baseline_tmax = baseline_prdf['baseline_travel_time'].max().round(1)
        scenario_tmax = new_prdf['new_travel_time'].max().round(1)
        reduced_tmax_pct = 100*(baseline_tmax-scenario_tmax)/baseline_tmax
        ############################
        # Compute travel time 
        bprdf  = baseline_prdf[['dest_municipality','baseline_travel_time']]
        nprdf  = new_prdf[['dest_municipality','src_municipality','new_travel_time']]
        result_table = bprdf.merge(nprdf, on=['dest_municipality'])
        result_table['difference'] = result_table['baseline_travel_time'] - result_table['new_travel_time']
        result_table = result_table.sort_values(by=['difference','baseline_travel_time'], ascending=[False,False]).reset_index(drop=True)
        result_table['Baseline travel time'] = result_table['baseline_travel_time'].map(time_to_readable)
        result_table['New travel time'] = result_table['new_travel_time'].map(time_to_readable)
        result_table['Saved time'] = result_table['difference'].map(time_to_readable)
        result_table =  result_table.rename(columns={'src_municipality':'Assigned Food Hub',\
                                                    'dest_municipality':'LGU destination'})
        ###################                                            
        reduced_tmean_hrs = result_table[result_table['difference']>0]['new_travel_time'].mean().round(1)
        efficiency_per_dest = result_table[result_table['difference']>0]['difference']/result_table['baseline_travel_time']
        efficiency = 100*(efficiency_per_dest.mean())

        st.write(f"With a new food hub in **{municity_name},{province}**, travel time is reduced to its assigned LGUs by an average of **{reduced_tmean_hrs} hrs**, which is **{efficiency:0.1f}% of the baseline**. ")
        if reduced_tmax_pct>1 :
            st.write(f"Given this setup, it will take only **{scenario_tmax} hrs** to reach all LGUs, which is **{reduced_tmax_pct:0.1f}%** reduction from the baseline ({baseline_tmax} hrs). ")
        else:
            st.write(f"Given this setup, it will take **{scenario_tmax} hr** to reach all LGUs, which is the same as the baseline. ")

        st.write('The table below compares the baseline to the new (2-hub) travel time across all LGU destinations. The topmost rows show the LGUs which would benefit the most from shortened travel times.')

        #result_table = result_table[['LGU destination','Assigned Food Hub','Baseline travel time','New travel time','Saved time']]                                           
        col = 'difference'
        norm = (result_table[col] - result_table[col].min()) / (result_table[col].max() - result_table[col].min())
        # Generate colors and styles
        colors = plt.cm.Greens(norm * 0.8)
        styles = [f'background-color: rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, 0.5)' for c in colors]
        # Apply styles to the DataFrame
        styled_result_table = result_table.style.apply(lambda x: styles, subset=['Saved time'], axis=0)
        # Display dataframe
        st.dataframe(styled_result_table, use_container_width=True, column_config={'baseline_travel_time': None, 'new_travel_time': None, 'difference':None})
        # ############################
        # st.write('### Population reached')       
        # st.write(f"With a new food hub in **{municity_name},{province}** , it will now take only **X,Y,Z hours** to reach 50%, 75%, and 95% of the population which is **E%** faster than baseline")
        
# st.write("Enter new hub coordinates (up to 4 decimal places)")
# st.caption("You may check OpenStreetMap or other mapping services to get accurate coordinates.")
# st.caption("You may also like to try these locations: Catarman (12.4994, 124.6405) or Kalibo (11.6967, 122.3684)")
# with st.form(key='hub_locs'):
#     # Create input widgets for latitude and longitude
#     new_hub_lat = st.number_input("Enter Latitude:", 8.4, 13.0, new_hub_lat, key='new_hub_lat', format="%.4f",
#                                     help="Enter value between 8.4 and 13.0")
#     new_hub_lon = st.number_input("Enter Longitude:", 121.0, 126.5, new_hub_lon, key='new_hub_lon', format="%.4f",
#                                     help="Enter value between 121.0 and 126.5")
#     submit_hub = st.form_submit_button(label="Submit new hub") 

# Initialize button states
# if "Submit new hub" not in st.session_state:
#     st.session_state["Submit new hub"] = False
# if "Run routing model" not in st.session_state:
#     st.session_state["Run routing model"] = False  

# if submit_hub:
#     st.session_state["Submit new hub"] = not st.session_state["Submit new hub"] 

# if st.session_state["Submit new hub"]:
#     st.write(f"You entered new hub located at coordinate: ({new_hub_lat}, {new_hub_lon})")
#     st.write("Upon verifying that this is your desired new hub location, click the button below to run the model.")
#     st.write("Run takes about 10 mins so please be patient!")
#     start_run = st.button("Run routing model")

    # if start_run:
    #     st.session_state["Run routing model"] = not st.session_state["Run routing model"] 
    #     print(st.session_state["Run routing model"],st.session_state["Submit new hub"])
    # if st.session_state["Submit new hub"] and st.session_state["Run routing model"]:
    #     tic=time.time()
    #     with st.spinner("Initializing..."):
    #         initialize()
    #         st.write(f"Tool initialized!")
    #     routes_fig = plot_routing_visayas((new_hub_lon,new_hub_lat))
    #     st.markdown("### Results")
    #     st.write('1. Routing map')
    #     st.pyplot(routes_fig)
    #     toc=time.time()
    #     st.success(f"Routing model run completed in {toc-tic:.1f} secs!")
    #     st.write(f"Please refresh the page if you wish to run the model again.")
# 

# 1) user selects region (6,7,or 8) via drop-down menu

# 2) display (baseline) relief delivery time map of region with location(s) of DSWD food hub — this can be preloaded* 

# 3) display text: With [X] DSWD food hub(s) as suppliers of relief goods, it will take up to [Y] hours to reach all town centers in Region [Z] and [A,B,C] hours to reach 50%, 75%, and 95% of the region’s population (these values are also preloaded) 

# 3) user selects location where to place a new active food hub via drop-down menu for province, city/municipality — we place the food hub in the town center** 

# 4) display updated relief delivery time map with location of additional food hub 

# 5) display text: With an additional active food hub in [city/municipality, province], it will now take only [Y] hours to reach all town centers in Region [Z] and [A,B,C] hours to reach 50%, 75%, and 95% of the region’s population. 

# Let us also add a technical note— draft:




