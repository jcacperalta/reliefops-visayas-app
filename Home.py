import streamlit as st

st.set_page_config(page_title="ReliefOps Visayas Transport Network Efficiency Simulator")
st.title("ReliefOps Visayas Transport Network Efficiency Simulator")
st.markdown("---")
st.markdown("### About the Project")
st.write("This prototype app simulates the Visayas transport network during relief delivery operation. \
    We demonstrate how a network model of Visayas towns connected with its roads and sea routes can be constructed using available open data.")
st.markdown("---")
st.markdown("### Datasets Used")
st.write("We used OpenStreetMap (OSM) data to construct the road network. Sea routes connecting the islands of Visayas were identified from (1) OpenStreetMap and (2) documented ferry and boat routes")
