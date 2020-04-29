import numpy
import pygplates

#Creat the target grid
target_lats=numpy.linspace(-40,10,10) 
target_lons=numpy.linspace(-80,-50,10) 

latlat,lonlon=numpy.meshgrid(target_lats,target_lons)
arr=numpy.stack((numpy.ravel(latlat),numpy.ravel(lonlon)),axis=-1)

#Now assign plate ids
#Set the rotaion file/static polygons.
input_rotation_filename = "PLATEMODELS/Muller2016/Global_EarthByte_230-0Ma_GK07_AREPS.rot"
polygons_filename="PLATEMODELS/Muller2016/Shapefiles/StaticPolygons/Global_EarthByte_GPlates_PresentDay_StaticPlatePolygons_2015_v1.shp"

#Create a rotation model to rotate the points back in time
file_registry = pygplates.FeatureCollectionFileFormatRegistry()
rotation_feature_collection = file_registry.read(input_rotation_filename)
rotation_model = pygplates.RotationModel([rotation_feature_collection])

#Assign plate ids to full list of points
point_features = []
for point in arr:
    latlonPoint = pygplates.LatLonPoint(point[0],point[1])
    point_to_rotate = pygplates.convert_lat_lon_point_to_point_on_sphere(latlonPoint)
    point_feature = pygplates.Feature()
    point_feature.set_geometry(point_to_rotate)

    point_features.append(point_feature)
    
assigned_point_features = pygplates.partition_into_plates(
    polygons_filename,
    rotation_model,
    point_features,
    properties_to_copy = [pygplates.PartitionProperty.reconstruction_plate_id])

#Now create the final with lons, lats, ages, and plateids
probgrid=[]
for age in range(0,231):
    print(age)
    for idx,i in enumerate(assigned_point_features):
        #Get the details from the GPlates features, and convert to lat/lon.
        plateid = i.get_reconstruction_plate_id()
        point_to_rotate=i.get_geometry()
        point = point_to_rotate.to_lat_lon_array()      
        probgrid.append([point[0,1],point[0,0],age,plateid])
        
probgrid=numpy.array(probgrid)

#And save out the grid for coregistering
numpy.savetxt('MLtest.csv',probgrid,delimiter=',')