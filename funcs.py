#!/usr/bin/python3


# Created on Wed Nov 15 17:22:46 2023

# @author: Lorenzo Corgnati
# e-mail: lorenzo.corgnati@sp.ismar.cnr.it


# This application reads from the EU HFR NODE EU HFR NODE database the information about 
# radial and total HFR files (both Codar and WERA) pushed by the data providers,
# combines radials into totals, generates HFR radial and total data to netCDF 
# files according to the European standard data model for data distribution towards
# EMODnet Physics portal and  generates HFR radial and total data to netCDF 
# files according to the Copernicus Marine Service data model for data distribution towards
# Copernicus Marine Service In Situ component.

# This application works on historical data, i.e. it processes HFR data within time intervals
# specified by the user. It does not work for Near Real Time operations.

# When calling the application it is possible to specify if all the networks have to be processed
# or only the selected one, the time interval to be processed and if the generation of HFR radial 
# and total netCDF data files according to the Copernicus Marine Service data model has to
# be performed.

# This application implements parallel computing by launching a separate process 
# per each HFR network to be processed (in case of processing multiple networks).

import os
import sys
import getopt
import glob
import datetime as dt
import numpy as np
import pandas as pd
from radials import Radial, buildEHNradialFolder, buildEHNradialFilename, convertEHNtoINSTACradialDatamodel, buildINSTACradialFolder, buildINSTACradialFilename
from totals import Total, buildEHNtotalFolder, buildEHNtotalFilename, combineRadials, convertEHNtoINSTACtotalDatamodel, buildINSTACtotalFolder, buildINSTACtotalFilename, buildUStotal
from calc import createLonLatGridFromBB, createLonLatGridFromBBwera, createLonLatGridFromTopLeftPointWera
from common import addBoundingBoxMetadata
import pickle

######################
# PROCESSING FUNCTIONS
######################

def modifyNetworkDataFolders(ntwDF,dataFolder):
    """
    This function replaces the data folder paths in the total_input_folder_path and in the 
    total_HFRnetCDF_folder_path fields of the DataFrame containing information about the network
    according to the data folder path specified by the user.
    
    INPUT:
        ntwDF: Series containing the information of the network
        dataFolder: full path of the folder containing network data
        
    OUTPUT:
        ntwDF: Series containing the information of the network
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    mfErr = False
    
    try:
        # Check if the total_input_folder_path field is specified
        if ntwDF.loc['total_input_folder_path']:
            # Modify the total_input_folder_path
            ntwDF.loc['total_input_folder_path'] = os.path.join(dataFolder,ntwDF.loc['network_id'],ntwDF.loc['total_input_folder_path'].split('/')[-1])
            
        # Check if the total_HFRnetCDF_folder_path field is specified
        if ntwDF.loc['total_HFRnetCDF_folder_path']:
            # Modify the total_HFRnetCDF_folder_path
            ntwDF.loc['total_HFRnetCDF_folder_path'] = os.path.join(dataFolder,ntwDF.loc['network_id'],'Totals_nc')
        
    except Exception as err:
        mfErr = True
    
    return ntwDF

def modifyStationDataFolders(staDF,dataFolder):
    """
    This function replaces the data folder paths in the radial_input_folder_path and in the 
    radial_HFRnetCDF_folder_path fields of the DataFrame containing information about the radial
    stations according to the data folder path specified by the user.
    
    INPUT:
        staDF: Series containing the information of the radial station
        dataFolder: full path of the folder containing network data
        
    OUTPUT:
        staDF: Series containing the information of the radial station
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    mfErr = False
    
    try:
        # Check if the radial_input_folder_path field is specified
        if staDF.loc['radial_input_folder_path']:
            # Modify the radial_input_folder_path
            staDF.loc['radial_input_folder_path'] = os.path.join(dataFolder,staDF.loc['network_id'],staDF.loc['radial_input_folder_path'].split('/')[-2],staDF.loc['radial_input_folder_path'].split('/')[-1])
            
        # Check if the radial_HFRnetCDF_folder_path field is specified
        if staDF.loc['radial_HFRnetCDF_folder_path']:
            # Modify the radial_HFRnetCDF_folder_path
            staDF.loc['radial_HFRnetCDF_folder_path'] = os.path.join(dataFolder,staDF.loc['network_id'],'Radials_nc')
        
    except Exception as err:
        mfErr = True
    
    return staDF

def performRadialCombination(combRad,networkData):
    """
    This function performs the least square combination of the input Radials and creates
    a Total object containing the resulting total current data. 
    The Total object is also saved as .ttl file via pickle binary serialization.
    The function creates a DataFrame containing the resulting Total object along with 
    related information.
    The function inserts information about the created netCDF file into the EU HFR NODE database.
    
    INPUTS:
        combRad: DataFrame containing the Radial objects to be combined with the related information
        networkData: DataFrame containing the information of the network to which the radial site belongs
        numActiveStations: number of operational radial sites
        
    OUTPUTS:
        combTot = DataFrame containing the Total object obtained via the least square combination 
                  with the related information
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    crErr = False
    
    # Create the output DataFrame
    combTot = pd.DataFrame(columns=['Total', 'NRT_processed_flag'])
    
    # Check if the combination is to be performed
    if networkData.iloc[0]['radial_combination'] == 1:
        # Check if the radials were already combined
        if ((networkData.iloc[0]['network_id'] != 'HFR-WesternItaly') and (0 in combRad['NRT_combined_flag'].values)) or ((networkData.iloc[0]['network_id'] == 'HFR-WesternItaly') and (0 in combRad['NRT_processed_flag_integrated_network'].values)):
            # Get the lat/lons of the bounding box
            lonMin = networkData.iloc[0]['geospatial_lon_min']
            lonMax = networkData.iloc[0]['geospatial_lon_max']
            latMin = networkData.iloc[0]['geospatial_lat_min']
            latMax = networkData.iloc[0]['geospatial_lat_max']

            # Get the grid resolution in meters
            gridResolution = networkData.iloc[0]['grid_resolution'] * 1000      # Grid resolution is stored in km in the EU HFR NODE database

            # Create the geographical grid
            exts = combRad.extension.unique().tolist()
            if (len(exts) == 1):
                if exts[0] == '.ruv':
                    gridGS = createLonLatGridFromBB(lonMin, lonMax, latMin, latMax, gridResolution)
                elif exts[0] == '.crad_ascii':
                    gridGS = createLonLatGridFromBBwera(lonMin, lonMax, latMin, latMax, gridResolution)
            else:
                gridGS = createLonLatGridFromBB(lonMin, lonMax, latMin, latMax, gridResolution)

            # Scale velocities and variances of WERA radials in case of combination with CODAR radials
            if (len(exts) > 1):
                for idx in combRadcombRad.loc[combRad['extension'] == '.crad_ascii'].loc[:]['Radial'].index:
                    combRad.loc[idx]['Radial'].data.VELO *= 100
                    combRad.loc[idx]['Radial'].data.HCSS *= 10000

            # Get the combination search radius in meters
            searchRadius = networkData.iloc[0]['combination_search_radius'] * 1000      # Combination search radius is stored in km in the EU HFR NODE database

            # Get the timestamp
            timeStamp = dt.datetime.strptime(str(combRad.iloc[0]['datetime']),'%Y-%m-%d %H:%M:%S')

            # Generate the combined Total
            T, warn = combineRadials(combRad,gridGS,searchRadius,gridResolution,timeStamp)

            # Add metadata related to bounding box
            T = addBoundingBoxMetadata(T,lonMin,lonMax,latMin,latMax,gridResolution/1000)

            # Update is_combined attribute
            T.is_combined = True

            # Add is_wera attribute
            if (len(exts) == 1):
                if exts[0] == '.ruv':
                    T.is_wera = False
                elif exts[0] == '.crad_ascii':
                    T.is_wera = True
            else:
                T.is_wera = False
                 
    return T


def selectRadials(networkID,stationData):
    """
    This function lists the input radial files pushed by the HFR data providers 
    that falls into the processing time interval and creates the DataFrame containing 
    the information needed for the combination of radial files into totals and for the
    generation of the radial and total data files into the European standard data model.
    
    INPUTS:
        networkID: network ID of the network to be processed
        stationData: DataFrame containing the information of the stations belonging 
                     to the network to be processed
        
    OUTPUTS:
        radialsToBeProcessed: DataFrame containing all the radials to be processed for the input 
                              network with the related information
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    sRerr = False
    
    # Create output total Series
    radialsToBeProcessed = pd.DataFrame(columns=['filename', 'filepath', 'network_id', 'station_id', \
                                                 'timestamp', 'datetime', 'reception_date', 'filesize', 'extension', \
                                                 'NRT_processed_flag', 'NRT_processed_flag_integrated_network', 'NRT_combined_flag'])
    
    #####
    # List radials from stations
    #####
    
    # Scan stations
    for st in range(len(stationData)):
        try:   
            # Get station id
            stationID = stationData.iloc[st]['station_id']
            # Trim heading and trailing whitespaces from input folder path string
            inputFolder = stationData.iloc[st]['radial_input_folder_path'].strip()
            # Check if the input folder is specified
            if(not inputFolder):
                print('The radial input folder for station ' + networkID + '-' + stationID + ' does not exist.')
            else:
                # Check if the input folder path exists
                if not os.path.isdir(inputFolder):
                    print('The radial input folder for station ' + networkID + '-' + stationID + ' does not exist.')
                else:
                    # Get the input file type (based on manufacturer)
                    manufacturer = stationData.iloc[st]['manufacturer'].lower()
                    if 'codar' in manufacturer:
                        fileTypeWildcard = '**/*.ruv'
                    elif 'wera' in manufacturer:
                        fileTypeWildcard = '**/*.crad_ascii'    
                    elif 'lera' in manufacturer:
                        fileTypeWildcard = '**/*.crad_ascii' 
                    # List all radial files
                    inputFiles = [file for file in glob.glob(os.path.join(inputFolder,fileTypeWildcard), recursive = True)]                    
                    for inputFile in inputFiles:
                        try:
                            # Get file parts
                            filePath = os.path.dirname(inputFile)
                            fileName = os.path.basename(inputFile)
                            fileExt = os.path.splitext(inputFile)[1]
                            
                            # Get file timestamp
                            radial = Radial(inputFile)
                            timeStamp = radial.time.strftime("%Y %m %d %H %M %S")                    
                            dateTime = radial.time.strftime("%Y-%m-%d %H:%M:%S")  
                            
                            # Get file size in Kbytes
                            fileSize = os.path.getsize(inputFile)/1024 
                            
    #####
    # Insert radial information into the output DataFrame
    #####
    
                            # Prepare data to be inserted into the output DataFrame
                            dataRadial = {'filename': [fileName], 'filepath': [filePath], 'network_id': [networkID], \
                                          'station_id': [stationID], 'timestamp': [timeStamp], 'datetime': [dateTime], \
                                          'reception_date': [dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")], \
                                          'filesize': [fileSize], 'extension': [fileExt], 'NRT_processed_flag': [0], \
                                          'NRT_processed_flag_integrated_network': [0], 'NRT_combined_flag': [0]}
                            dfRadial = pd.DataFrame(dataRadial)

                            # Insert into the output DataFrame
                            radialsToBeProcessed = pd.concat([radialsToBeProcessed, dfRadial],ignore_index=True)

                        except Exception as err:
                            sRerr = True
                        
        except Exception as err:
            sRerr = True
    
    return radialsToBeProcessed
