# Hyperspectral_CT_Recon
Build a prototyping pipeline to test different hyperspectral reconstruction and segmentation approaches leveraging public data and/or simulated data. 

## Research Papers and Datasets Documentation

In this [Google Sheets spreadsheet](https://docs.google.com/spreadsheets/d/1jJbZ0b8knY3XQfHnoKXy_cuw0ZG5uWUUDZ_9mqVxh0Y/edit#gid=0) you can find the papers and datasets we have found and which we believe might be useful for the project. Feel free to add any information you might have.

Additionally (if you haven't already), it might also be useful to check the [Project Proposal's](https://wiki.tum.de/display/mlmi/MLMI+Summer+2023?preview=/1375273096/1474757233/MLMI_Proposal_Hyperspectral%20CT%20Reconstruction.pdf) references. More specifically, I would recommend:
- Multi-Spectral Imaging via Computed Tomography (MUSIC)- Comparing Unsupervised Spectral Segmentations for Material Differentiation
- Spectral computed tomography:fundamental principles and recent developments

## TO DOs
Ideally we have the issues and just update them accordingly to know whose doing what and in which state we currently are.
Nevertheless here are some important deadlines:
- [ ] mid-presentation: 06.07.2023. For this date we are supposed to have a baseline script/model as well as the presentation.
- [ ] final presentation: 21.09.2023
- [ ] Adapt a method, like Deepsdf, to do Hyperspectral Segmentation and 3D reconstruction

## BASELINE PROPOSAL:
1. Approach 1:
    1. Slice Reconstruction from Histograms (use FBP/ART TV/any other similar method)
    2. Slice Segmentation and Classification ([based on Automatic multi-organ segmentation in dual-energy CT (DECT) with dedicated 3D fully convolutional DECT networks](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.13950?af=R))
    3. Find Method to Stack and visualize reconstruction
2. Approach:
    1. Do Slice Reconstruction from Histograms using Learning-based approach (some methods can be found in the spreadsheet)
    2. Do direct reconstruction and segmentation using Learning-based approach.


## DATASET

Download the [MUSIC 2D and Music 3D Spectral datasets](http://easi-cil.compute.dtu.dk/index.php/datasets/music/) and set them in the root of the project.

## Docker Container
Install [docker](https://docs.docker.com/engine/install/ubuntu/) (if you don't already have it).

After that, go to the root folder of the repo and simply run:
 ```
docker build -t mlmi -f Dockerfile ..
 ```
Once the docker file has been compiled you can simply run by executing:
 ```
 sudo docker run --rm -v <DATASET FOLDER PATH>:/workspace/dataset -v <REPO FOLDER PATH>:/workspace --gpus all -it  mlmi
  ```
  From there, you can execute as you would normally do. 
  
  To exit the container simply run `exit`.

