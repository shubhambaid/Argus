import arcgis
import cv2
from arcgis.gis import GIS
from arcgis.features.analyze_patterns import interpolate_points
from arcgis.geocoding import geocode
from arcgis.features.find_locations import trace_downstream
from arcgis.features.use_proximity import create_buffers

gis = GIS(username="baidman123",password="shubham111")


chennai_pop_map = gis.map("Chennai")
