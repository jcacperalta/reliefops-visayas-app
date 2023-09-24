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

st.title('Typhoon Scenario Simulation')

TYPHOON_SELECT = ["Yolanda (Haiyan) 2014", "Odette (Rai) 2021", "Uring (Thelma) 1991"]
this_typhoon = st.selectbox('Select typhoon', TYPHOON_SELECT)
