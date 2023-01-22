# Lidar_QL_radsat (Lidar Quicklook Plot with Radar and Satellite for Context)

This respository contains python code to create quicklook plots of lidar vertical profile data including:
- Vertical Velocity
- Backscatter
- Wind speed and direction (in Hodograph and/or barb format)

Furthermore, these plots include radar reflectivity and high-resolution (0.5km) visible satellite
at the time of the corresponding lidar vertical profile to provide context to the data

Example of radsat plot from 22 March 2022 during the PERiLS 2022 Field Campaign:

  *Red star indicates location of lidar profiler which in this case is CLAMPS2 which was located near Yazoo City, MS
<img width="1128" alt="Screen Shot 2023-01-21 at 6 19 57 PM" src="https://user-images.githubusercontent.com/67449088/213899891-9946d837-ce77-4c53-b5ff-04cd78151897.png">

Important Note: This Program assumes the following files are downloaded to your local system:
- dlvad NetCDF file i.e. clampsdlvadC2.c1.20220322.000000.nc
- dlfp NetCDF file i.e. clampsdlfpC2.b1.20220322.000000.nc
- ABI GOES-16 Channel 2 satellite NetCDF file i.e. OR_ABI-L1b_g16_meso_20220413_18.nc
