#!/usr/bin/env python3

#start with the main() function and follow the code
#the input file is defined in parameters.py and has the same length with the output files.

from netCDF4 import Dataset
import scipy.spatial
from scipy.signal import decimate
from scipy.interpolate import griddata
import numpy as np
import pandas as pd
import time, math, pickle, os, urllib, csv, glob
import pygplates
#from matplotlib import pyplot as plt
import shapefile
import cv2

from parameters import parameters as param
import Utils

start_time = param["time"]["start"]
end_time = param["time"]["end"]

#construct the grid tree
grid_x, grid_y = np.mgrid[-90:91, -180:181]
grid_points = [pygplates.PointOnSphere((float(row[0]), float(row[1]))).to_xyz() for row in zip(grid_x.flatten(), grid_y.flatten())]
grid_tree = scipy.spatial.cKDTree(grid_points)

rotation_files = Utils.get_files(param["rotation_files"])
rotation_model = pygplates.RotationModel(rotation_files) #load rotation model

def save_data(data, filename):
    row_len=0
    for row in data:
        if len(row)>row_len:
            row_len = len(row)
    with open(filename,"w+") as f:
        for row in data:
            if row:
                f.write(','.join(['{:.2f}'.format(i) for i in row]))
                if len(row)<row_len: #keep the length of rows the same
                    f.write(',')
                    f.write(','.join(['nan']*(row_len-len(row))))
            else:
                f.write('NO_DATA')
            f.write('\n')
    
    
#input: 
    #sample_points: 2D array. The columns are index, lon, lat, time and plate id.
    #vector_file: The name of the data file from which to extract attributes for sample points.
    #             The vector_file contains a set of points and each point associates with a set of attributes.
    #region: region of interest(in degree)
    
#output:
    #2D array(the same length as the sample_points). The columns are reconstructed lon, lat, distance and
    #the attributes copied from vector_file
    
#For each point(row) in a 2D array sample_points, search the nearest point from vector_file within the region
#of interest. If the nearest point is found, copy its attributes to the input point.
def query_vector(sample_points, vector_file, region):
    #prepare the list for result data and insert indices for input points 
    ret=[]
    indices_bak = []
    with open(vector_file.format(time=start_time), 'r') as f:
        column_num = len(f.readline().split(','))
    
    for i in range(len(sample_points)):
        ret.append([np.nan, np.nan])
        indices_bak.append(sample_points[i][0]) #keep a copy of the original indices
        sample_points[i][0] = i
        
    #sort and group by time to improve performance
    sorted_points = sorted(sample_points, key = lambda x: int(x[3])) #sort by time
    from itertools import groupby
    for t, group in groupby(sorted_points, lambda x: int(x[3])):  #group by time
        if t>end_time or t<start_time:
            continue
        #print('querying '+vector_file.format(time=t))
        # build the points tree at time t
        data=np.loadtxt(vector_file.format(time=t), skiprows=1, delimiter=',') 
     
        #assume first column is lon and second column is lat
        points_3d = [pygplates.PointOnSphere((row[1],row[0])).to_xyz() for row in data]
        points_tree = scipy.spatial.cKDTree(points_3d)

        # reconstruct the points
        rotated_points = []
        grouped_points = list(group)#must make a copy, the items in "group" will be gone after first iteration
        for point in grouped_points:
            point_to_rotate = pygplates.PointOnSphere((point[2], point[1]))
            finite_rotation = rotation_model.get_rotation(point[3], int(point[4]))#time, plate_id
            geom = finite_rotation * point_to_rotate
            rotated_points.append(geom.to_xyz())
            idx = point[0]
            ret[idx][1], ret[idx][0] = geom.to_lat_lon()
                   
        # query the tree of points 
        dists, indices = points_tree.query(
            rotated_points, k=1, distance_upper_bound=Utils.degree_to_straight_distance(region)) 

        for point, dist, idx in zip(grouped_points, dists, indices):
            if idx < len(data):
                ret[point[0]] = ret[point[0]] + [dist] + list(data[idx])
    
    for i in range(len(ret)):
        if len(ret[i]) == 2:
            ret[i] = ret[i] + [np.nan]*(1+column_num)
            
    #restore original indices
    for i in range(len(indices_bak)):
        sample_points[i][0] = indices_bak[i]
    return ret
   
    

#input: 
    #sample_points: 2D array. The columns are index, lon, lat, time and plate id.
    #grid_file: The name of the grid file from which to extract data for sample points.
    #region: region of interest(in degree)
    
#output:
    #2D array(the same length as the sample_points). 
    #The columns are reconstructed lon, lat, region and grid mean value.
    
#For each point(row) in a 2D array sample_points, calculate the mean value of the grid data within the region
#of interest. Attach the mean value to the input point.
def query_grid(sample_points, grid_file, region):
    #prepare the list for result data and insert indices for input points 
    ret=[]
    indices_bak = []
    for i in range(len(sample_points)):
        ret.append([np.nan, np.nan])
        indices_bak.append(sample_points[i][0]) #keep a copy of the original indices
        sample_points[i][0] = i
        
    #sort and group by time to improve performance
    sorted_points = sorted(sample_points, key = lambda x: int(x[3])) #sort by time
    from itertools import groupby
    for t, group in groupby(sorted_points, lambda x: int(x[3])):  #group by time
        if t>end_time or t<start_time:
            continue
        #print('querying '+grid_file.format(time=t))
        age_grid_fn = grid_file.format(time=t)
        
        # reconstruct the points
        rotated_points_lons = []
        rotated_points_lats = []
        grouped_points = list(group)#must make a copy, the items in "group" will be gone after first iteration
        for point in grouped_points:
            point_to_rotate = pygplates.PointOnSphere((point[2], point[1]))
            finite_rotation = rotation_model.get_rotation(point[3], int(point[4]))#time, plate_id
            geom = finite_rotation * point_to_rotate
            lat, lon = geom.to_lat_lon()
            rotated_points_lons.append(lon)
            rotated_points_lats.append(lat)
            idx = point[0]
            ret[idx][1], ret[idx][0] = geom.to_lat_lon()
                   
        # query the grid tree
        values = Utils.query_raster(age_grid_fn, rotated_points_lons,rotated_points_lats, region, True)
        for point, value in zip(grouped_points,values):
            ret[point[0]] = ret[point[0]] + [region, value]
     
    for i in range(len(ret)):
        if len(ret[i]) == 2:
            ret[i] = ret[i] + [np.nan]*2
            
    #restore original indices
    for i in range(len(indices_bak)):
        sample_points[i][0] = indices_bak[i]
    return ret

        
def run():
    #create output dir
    out_dir=param['coreg_output_dir']
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    tic=time.time()
    
    for input_file in Utils.get_files(param['coreg_input_files']):
        print(f'processing {input_file} ***********************************')
        results=[]
        seed_data = pd.read_csv(input_file) 
        #print(seed_data)
        columns=[]
        count=0
        #query the vector data
        for in_file in param['vector_files']:
            print(f'querying {in_file}')
            with open(in_file.format(time=start_time), 'r') as f:
                header = f.readline()
            
            input_data = seed_data.values.tolist()
            
            if count == 0:
                results.append(input_data)
                columns=columns+['index', 'lon', 'lat', 'age', 'plate_id', 'recon_lon','recon_lat','distance']+header.split(',')
            else:
                columns=columns+['distance_'+str(count)]+header.split(',')
            
            result=[np.nan]*len(input_data)
            for region in sorted(param['regions']):
                print('region of interest: {}'.format(region))
                print('the length of input data is: {}'.format(len(input_data)))

                ret = query_vector(input_data, in_file, region)

                new_input_data=[]
                for i in range(len(input_data)):
                    result[int(input_data[i][0])] = ret[i] #save the result
                    if np.isnan(ret[i]).any():
                        new_input_data.append(input_data[i])#prepare the input data to query again with a bigger region

                input_data = new_input_data
             
            if count > 0:
                for r in result:
                    r.pop(0)
                    r.pop(0)
            results.append(result)
            count+=1
   
        count=0
        #query the grids 
        for in_file in param['grid_files']:
            print(f'querying {in_file}')
            input_data = seed_data.values.tolist()
            if count == 0:
                columns=columns+['recon_lon','recon_lat','region', in_file[1]]
            else:
                columns=columns+['region_'+str(count), in_file[1]]
            
            result=[np.nan]*len(input_data)
            for region in sorted(param['regions']):
                print('region of interest: {}'.format(region))
                print('the length of input data is: {}'.format(len(input_data)))

                ret = query_grid(input_data, in_file[0], region)

                new_input_data=[]
                for i in range(len(input_data)):
                    result[int(input_data[i][0])] = ret[i] #save the result
                    if np.isnan(ret[i]).any():
                        new_input_data.append(input_data[i])#prepare the input data to query again with a bigger region

                input_data = new_input_data
            if count > 0:
                for r in result:
                    r.pop(0)
                    r.pop(0)
            results.append(result)
            count+=1
        
        results = np.concatenate(results, axis=1)  
        df = pd.DataFrame(results, columns=columns)
        df = df.drop(columns=['index'])
        print(len(columns))
        print(df.shape)
        df.to_csv(out_dir+'/'+os.path.basename(input_file), float_format='%.3f', index=False) 
        
    toc=time.time()
    print(f'The coregistration output data have been saved in folder {out_dir} successfully!')
    print("Time taken:", toc-tic, " seconds")
  
if __name__ == "__main__":
    run()
