# Run net on folder of images and display results

import numpy as np
import FCN_NetModel as NET_FCN # The net Class
import torch
import Visuallization as vis
import cv2

#------------------input parameters-------------------------------------------------------------------------------
InputImage=r"Example/Test.jpg" # Input image file
Trained_model_path =  "logs/Defult.torch" # Train model to use
UseGPU=False
DisplayXYZPointCloud=True # Show Point cloud
DisplayVesselOpeningOnPointCloud=True # Add vessel opening to point cloud
MinSize=600#min image size (height or width)
MaxSize=1000 # Max image sie (height or width)
#************************************Masks and XYZ maps to predict********************************************************************************************************
MaskClasses =  {}
XYZList = ["VesselXYZ","ContentXYZ","VesselOpening_XYZ"] # XYZ maps to predict
MaskList = ["VesselMask","ContentMaskClean","VesselOpeningMask"] # Segmentation mask to predict
XYZ2Mask={"VesselXYZ":"VesselMask","ContentXYZ":"ContentMaskClean","VesselOpening_XYZ":"VesselOpeningMask"} # Connect XYZ map to segmentation mask
XYZ2Color={"VesselXYZ":[255,0,0],"ContentXYZ":[0,255,0],"VesselOpening_XYZ":[0,0,255]} # Colors where eac object will appear on XYZ point cloud
#******************************Create and Load neural net**********************************************************************************************************************

Net=NET_FCN.Net(MaskList=MaskList,XYZList=XYZList) # Create net and load pretrained
Net.load_state_dict(torch.load(Trained_model_path))
Net=Net.cuda()
Net.eval()
#*********************************Read image and resize*******************************************************************************

Im=cv2.imread(InputImage)
Im=vis.ResizeToMaxSize(Im,MaxSize)

Im=np.expand_dims(Im,axis=0)
###############################Run Net and make prediction###########################################################################
with torch.no_grad():
    PrdXYZ, PrdProb, PrdMask = Net.forward(Images=Im,TrainMode=False,UseGPU=UseGPU) # Run net inference and get prediction

#----------------------------Convert Prediction to numpy-------------------------------------------
Prd={}
for nm in PrdXYZ:
    Prd[nm]=(PrdXYZ[nm].transpose(1,2).transpose(2, 3)).data.cpu().numpy()
for nm in PrdMask:
    Prd[nm]=(PrdMask[nm]).data.cpu().numpy()
#-----------------------------Display 3d point cloud----------------------------------------------------------
if DisplayXYZPointCloud:
    import open3d as o3d
    xyz = np.zeros([20000,3],np.float32) # list of points for XYZ point cloud
    colors = np.zeros([20000,3],np.float32) # Point color for above list

    xyzMap={} # Maps from which points will be sampled
    h,w=Im.shape[1:3]
    xyzMap["VesselXYZ"] = Prd["VesselXYZ"][0]
    xyzMap["ContentXYZ"] = Prd["ContentXYZ"][0]
    xyzMap["VesselOpening_XYZ"] =Prd["VesselOpening_XYZ"][0]


    vis.show(Im[0].astype(np.uint8)," Close window to continue")

    tt=0
    while True: # Sample points for point cloud
           print("collecting points", tt)
           nm = list(xyzMap)[np.random.randint(len(list(xyzMap)))]
           #nm = "VesselOpening_XYZ"
           x = np.random.randint(xyzMap[nm].shape[1])
           y = np.random.randint(xyzMap[nm].shape[0])
           print("dddddd")
           if (Prd[XYZ2Mask[nm]][0,y,x])>0.95:
                     print("EEEEEEEe")
                     if np.abs(xyzMap[nm][y,x]).sum()>0:# and (GT[nm.replace("XYZ","Mask")][0,y,x]*GT["ROI"][0,y,x])>0.6:
                           print("ffffffffff")
                           xyz[tt]=xyzMap[nm][y,x]
                           colors[tt]=XYZ2Color[nm]
                           tt+=1
                           if tt>=xyz.shape[0]: break
    #...................Display point cloud.........................................................................................
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd],"Red=Vessel, Green=Content, Blue=Opening")
#-----------------------------Display XYZ map and images-----------------------------------------
for nm in Prd:
   # if "XYZ" in nm: continue

         print(nm, Prd[nm][0].max(),Prd[nm][0].min())
         #---------------Normalize value to the range of RGB image 0-255--------------------------------------------
         tmIm = Prd[nm][0].copy()
         if Prd[nm][0].max()>255 or Prd[nm][0].min()<0 or np.ndim(Prd[nm][0])==2:
             if tmIm.max()>tmIm.min(): #
                 tmIm[tmIm>1000]=0
                 tmIm = tmIm-tmIm.min()
                 tmIm = tmIm/tmIm.max()*255
             print(nm,"New", tmIm.max(), tmIm.min())
             if np.ndim(tmIm)==2:
                 tmIm=cv2.cvtColor(tmIm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
             # Remove region out side of the object mask from the XYZ mask
             if nm in XYZ2Mask:
                 print(nm)
                 for i in range(tmIm.shape[2]):
                     tmIm[:,:,i][Prd[XYZ2Mask[nm]][0]==0]=0
   #--------------------------------display------------------------------------------------------------
         im=cv2.resize(Im[0].astype(np.uint8),(tmIm.shape[1],tmIm.shape[0]))
         vis.show(np.hstack([tmIm,im]),nm+ " Max=" + str(Prd[nm][0].max()) + " Min=" + str(Prd[nm][0].min())+ " Close window to continue")