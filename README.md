# Graph-based Object Classification for Neuromorphic Vision Sensing

## Summary
This is the implemtation code and proposed dataset(ASL-DVS) for the following paper:

[1] Yin Bi, Aaron Chadha, Alhabib Abbas, Eirina Bourtsoulatze and Yiannis Andreopoulos, 'Graph-based Object Classification for Neuromorphic Vision Sensing', IEEE Conference on Computer Vision (ICCV), Oct.17 - Nov,2, 2019, Seoul, Korea

## Dataset: ASL-DVS 
We source one of the largest neuromorphic vision dataset acquired under real-world conditions, and make it available to the research community at the link: 

<img height="318" src="https://github.com/PIX2NVS/Graph2NVS/blob/master/images/ASL.JPG">                  <img height="318" src="https://github.com/PIX2NVS/Graph2NVS/blob/master/images/Dataset.JPG">


ASL-DVS contains 24 classes correspond to 24 letters (A-Y, excluding J) from the American Sign Language (ASL). The ASL-DVS was recorded with an iniLabs DAVIS240c NVS camera set up in an office environment with low environmental noise and constant illumination. Five subjects were asked to pose the different static handshapes relative to the camera in order to introduce natural variance into the dataset. For each letter, we collected 4,200 samples (total of 100,800 samples) and each sample lasts for approximately 100 milliseconds.


## Code Implementation
### Requirements:
     Python 2.7 
     Pytorch 
     pytorch_geometric
     
     
### Running examples:
    cd code
    python Test.py   # running file for G-CNN 


## Cite
Please cite our paper if you use this code or dataset in your own work. 

@InProceedings{Bi2019Graph,

author = {Yin Bi, Aaron Chadha, Alhabib Abbas, Eirina Bourtsoulatze and Yiannis Andreopoulos},

title = {Graph-based Object Classification for Neuromorphic Vision Sensing},

booktitle = {The IEEE International Conference on Computer Vision (ICCV)},

month = {Oct},

year = {2019}

}

## Contact 
For any questions or bug reports, please contact Yin Bi at yin.bi.16@ucl.ac.uk .
