# Graph-based Object Classification for Neuromorphic Vision Sensing

## Summary
This is the implemtation code and proposed dataset(ASL-DVS) for the following paper. Please cite following paper if you use this code or dataset in your own work. 

MLA:
    
   [1] Yin Bi, Aaron Chadha, Alhabib Abbas, Eirina Bourtsoulatze and Yiannis Andreopoulos, 'Graph-based Object Classification for Neuromorphic Vision Sensing', IEEE Conference on Computer Vision (ICCV), Oct.17 - Nov,2, 2019, Seoul, Korea
    
BibTex:
    
    @inproceedings{bi2019graph,
    title={Graph-based Object Classification for Neuromorphic Vision Sensing},
    author={Bi, Y and Chadha, A and Abbas, A and and Bourtsoulatze, E and Andreopoulos, Y},
    booktitle={2019 IEEE International Conference on Computer Vision (ICCV)},
    year={2019},
    organization={IEEE}
    }

## Dataset: ASL-DVS 
We source one of the largest neuromorphic vision dataset acquired under real-world conditions, and make it available to the research community at the link: 
https://www.dropbox.com/sh/ibq0jsicatn7l6r/AACNrNELV56rs1YInMWUs9CAa?dl=0


<img height="318" src="https://github.com/PIX2NVS/Graph2NVS/blob/master/images/ASL.JPG">                  <img height="318" src="https://github.com/PIX2NVS/Graph2NVS/blob/master/images/Dataset.JPG">


ASL-DVS contains 24 classes correspond to 24 letters (A-Y, excluding J) from the American Sign Language (ASL). The ASL-DVS was recorded with an iniLabs DAVIS240c NVS camera set up in an office environment with low environmental noise and constant illumination. Five subjects were asked to pose the different static handshapes relative to the camera in order to introduce natural variance into the dataset. For each letter, we collected 4,200 samples (total of 100,800 samples) and each sample lasts for approximately 100 milliseconds.

## Framework of Graph-Based Object Classification 

Our goal is to represent the stream of spike events from neuromorphic vision sensors as a graph and perform convolution on the graph for object classification. Our model is visualized in following figure: a non-uniform sampling strategy is firstly used to obtain a small set of neuromorphic events for computationally and memory-efficient processing; then sampling events are constructed into a radius neighborhood graph, which is processed by our proposed residual-graph CNNs for object classification.

<img height="360" width='800' src="https://github.com/PIX2NVS/NVS2Graph/blob/master/images/framework.JPG">

## Code Implementation
### Requirements:
     Python 2.7 
     Pytorch 1.0.1.post2
     pytorch_geometric 1.1.2
     
### Preparations:
    Training graphs are saved in './data/Traingraph/raw/' folder.
    Testing graphs are saved in './data/Testgraph/raw/' folder.
    Each sample should contains feature of nodes, edge, pseudo adresses and label.
    
### Running examples:
    cd code
    python Test.py   # running file for G-CNN 
    
    #The results can be found in the 'Results' folder.



## Contact 
For any questions or bug reports, please contact Yin Bi at yin.bi.16@ucl.ac.uk .
