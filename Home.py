import streamlit as st

st.set_page_config(page_title="ReliefOps Visayas Transport Network Efficiency Simulator")
st.title("ReliefOps Visayas Transport Network Efficiency Simulator")
st.markdown("---")
st.markdown("### About the Project")
st.write("This prototype app simulates the Visayas transport network during relief delivery operation. \
    We demonstrate how a network model of Visayas towns connected with its roads and sea routes can be constructed using available open data.")
st.markdown("---")
st.markdown("### Datasets Used")
st.write("We used OpenStreetMap (OSM) data to construct the road network. Sea routes connecting the islands of Visayas were identified from three sources in 2018: \
    ferry routes in (1) HOTOSM (2) Google Maps, and (3) the Visayas General Logistics Planning Map \
    created by Logistics Cluster of the United Nations World Food Programme (UN-WFP), which is the output of humanitarian operations and long-term engagements with LGUs in the aftermath of Typhoon Haiyan hitting the region in November 2013 (Logistics Cluster 2014). \
        All in all, we obtained 38 sea routes from HotOSM, 27 from Google Maps, and 23 from UNWFP.")