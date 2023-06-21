# Hyperspectral Datasets
As can be seen in the Google Sheets spreadsheet, there aren't a lot of Multi Spectral CT Datasets. For now, the one we will use (until a better alternative can be found) is MUSIC 2D and MUSIC 3D. This dataset can be downloaded from this [link](http://easi-cil.compute.dtu.dk/index.php/datasets/music/)

## MUSIC 2D
### Structure
```
MUSIC2D_HDF5
│   README.md    
│
└─── sample name
│   │   
│   └─── fullspectrum
|   |   │   
│   |   └─── projections
│   |   │   │   sinogram.h5
│   |   │   
│   |   └─── reconstruction
│   |   │   │   reconstruction.h5
│   |   │   
│   |   └─── segmentation
│   |       |   
│   |       └─── FAMS
│   |       |   │   sfams_k220.h5
│   |       |   
│   |       └─── FAMS
│   |           │   graphCut_double_5.h5
│   |      
│   └─── manualSegmentation
│   |   │   manualSegmentation.h5
│   │   
│   └─── reducedspectrum
│   |   │   
│   |   └─── reconstruction
│   |   │   │   reconstruction.h5
│   |   │   
│   |   └─── segmentation
│   |       |   
│   |       └─── FAMS
│   |       |   │   sfams_k220.h5
│   |       |   
│   |       └─── FAMS
│   |           │   graphCut_double_5.h5
|  
...   
```
- Projections/Sinograms: Contain sinograms of a sample with 128 different energy levels. Their dimensions are (128, 256, 370)
- Reconstructions: Slices for each energy level. Their dimensions are (100, 100, 1, 128)
- Segmentations: Segmentations done with different methods (FAMS and Graph Cut). The output is a single slice with dimensions (100, 100).
- Manual Segmentation: Manual segmentation of ROI based on the projections. This segmentations have dimensions of (100, 100) for each sample.

NOTE: The difference between the Full Spectrum and the Reduced Spectrum is the number of energy levels used. In the reduced spectrum data, the reconstructions are done for only 10 energy levels. Moreover, the segmentation is only done based on the data from 10 energy levels.

### How to visualize
To visualize the data from the dataset simply run:
```
python MUSIC_2D_Visualization.py -d <PATH/TO/MUSIC2D_HDF5>
```