# Predicting 2D and 3D shape of objects and liquids inside transparent vessels from a single image using neural net for XYZ map generation


Given an image containing transparent containers, predict the 3d model and 2D segmentation of the vessel content, the vessel itself, and the vessel opening plane.
The prediction is made in the form of an XYZ map that predicts X,Y,Z coordinates per pixel. 
The 3D model prediction is independent of the image source and camera used.
The 3D prediction is scale and translation invariant which means that the 3D shape of the object is predicted, but the scale of the model and its position (translation) is arbitrary.
See paper [Predicting 3D shapes, masks, and properties of materials, liquids, and objects inside transparent containers, using the TransProteus CGI datase]
(https://arxiv.org/pdf/2109.07577.pdf) for more details

The same code with train model included (run out of the box) could be download from: [1](https://icedrive.net/0/42jdtddE6S), [2](https://e.pcloud.link/publink/show?code=XZEI55ZNk2cL139W78o1FMk35VeG5j9Jzck), [3](https://zenodo.org/record/5696254#.YZKKQ7tyZH4)

These code was trained using the TransProteus dataset that can be downloaded from: [Full Dataset 1](https://e.pcloud.link/publink/show?code=kZfx55Zx1GOrl4aUwXDrifAHUPSt7QUAIfV),  [Full DataSet Link2](https://icedrive.net/1/6cZbP5dkNG), [Subset](https://zenodo.org/record/5508261#.YUGsd3tE1H4)
![](/Figure1.jpg)
Figure 1. Structure of the net for predicting 3D and 2D segments from the image. XYZ maps are represented as and BGR image with blue, red, and green channels corresponds to the X,Y,Z coordinates of the pixel.

## Videos of the net results:

https://www.youtube.com/watch?v=EbVvEYespII

https://www.youtube.com/watch?v=zWJJyjmsBko

https://www.youtube.com/watch?v=JC55VmYHB_s

https://zenodo.org/record/5697212#.YZKKOLtyZH4


  
# Requirements
## Hardware
For using the trained net for prediction, no specific hardware is needed, but the net will run much faster on Nvidia GPU.

For training the net, an Nvidia GPU is needed (the net was trained on RTX 3090)

## Setup
Create a environment with the required dependencies ([Pytorch](https://pytorch.org/), torchvision, scipy and OpenCV, Open3D): *conda env create -f environment.yml*

## Software:
This network was run with Python 3.88 [Anaconda](https://www.anaconda.com/download/) with  [Pytorch 1.8](https://pytorch.org/) and OpenCV* package.
* Installing opencv for conda can usually be done using: pip install opencv-python or conda install opencv

# Prediction: running the trained net on  a single image

1. Train net or download code with pre-trained net weight from [1](https://icedrive.net/0/42jdtddE6S), [2](https://e.pcloud.link/publink/show?code=XZEI55ZNk2cL139W78o1FMk35VeG5j9Jzck).
2. Open RunOnImage.py
3. Set image path to InputImage parameter
4. Set the path to the trained net weights  file in: Trained_model_path  (If you downloaded the code with the pre-train network from, the model path is already set) 
5. Run script to get prediction display on the screen.
Additional optional parameters: 
UseGPU: decide whether to use GPU hardware (True/False).
Other optional parameters are in the input parameters section of the script.

# For training and evaluating download TransProteus and LabPics

1. Download and extract the TransProteus dataset from:[Full Dataset 1](https://e.pcloud.link/publink/show?code=kZfx55Zx1GOrl4aUwXDrifAHUPSt7QUAIfV),  [Full DataSet Link2](https://icedrive.net/1/6cZbP5dkNG), [Subset](https://zenodo.org/record/5508261#.YUGsd3tE1H4)

3. Optional: download and extract the LabPics dataset from [here](https://zenodo.org/record/4736111#.YTkdcFtE1H4) or [here](https://www.kaggle.com/sagieppel/labpics-chemistry-labpics-medical)

## Training

1. Open Train.py
3. Set the path to TransProteus train folders in the dictionary "TransProteusFolder" in the input parameter section (the dictionary keys names don't matter). 
Note that this dictionary can get several folders, and each folder can be added more than once. If a folder appears twice, it will be used during training twice as much.
(By default, this parameter point to the example folder supplied with the code)

4. If you wish to use the LabPics dataset during training, set UseLabPicsDataSet=True  else set it to False

5. If you use the LabPics dataset: Set the path LabPics train folders in the dictionary  "LabPicsFolder" in the input parameter section (the dictionary keys don't matter).  Note that this dictionary can get several folders, and each folder can be added more than once. If a folder appears twice, it will be used during training twice as much.
(By default, this parameter point to the example folder supplied with the code)

6. Run the script
7. The trained net weight will appear in the folder defined in the  logs_dir 
8. For other optional training parameters, see the "Other training paramters" section in Train.py script

## Evaluating 

1. Train net or download code with pre-trained net weight from: [1](https://icedrive.net/0/42jdtddE6S), [2](https://e.pcloud.link/publink/show?code=XZEI55ZNk2cL139W78o1FMk35VeG5j9Jzck).
2. Open EvaluateModel.py
3. Set a path to the trained net weights  file in: Trained_model_path  (If you downloaded the code with the pre-train network from  the model path is already set) 
4. Set Test data folder  path to the  TestFolder parameter (This can be either the RealSense real images or one of the TransProteus virtual test datasets, both supply with TransProteus)
5. If using the RealSense (real photos) test set,  set parameter UseRealSenseDataset=True else set it to False
6. Run the script

For other parameters, see the Input parameters section.


## More Info 
See paper []() For more details



![](/Figure2.jpg)
Figure 2. Results of the net on various examples. XYZ maps are represented as and BGR image with blue, red, and green channels corresponds to the X,Y,Z coordinates of the pixel.
