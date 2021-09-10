# reade display Depth map  XYZ map from the real sense dataset
import os
import open3d as o3d
import Visuallization as vis
import numpy as np
import cv2



#########################################################################################################################
class Reader:
# Initiate reader and define the main parameters for the data reader
    def __init__(self,DataSetFolder):

        self.AllFolders=[] # All anotation folder
        self.itr=0
        self.epoch=0
        for dname in os.listdir(DataSetFolder): # Read list of annotations
            sbdir=DataSetFolder+"//"+dname+"//"
            if os.path.isdir(sbdir):
                   self.AllFolders.append(sbdir)
#########################Read next data###################################################################################
    def LoadSingle(self): # load single image and annotation

        data = {}
     #-------------Select next image---------------------------------------
        if self.itr>=len(self.AllFolders):
            self.itr=0
            self.epoch+=1
#---
        sbdir=self.AllFolders[self.itr]
        print(self.itr,sbdir)
        self.itr+= 1
        #------------------Read files------------------------------------------------------------------------------------------
     #   data["All"]=cv2.imread(sbdir+"//All.jpg")
        data["VesselWithContentRGB"]=cv2.imread(sbdir+"/VesselImg.png")
        data["ContentRGB"]=cv2.imread(sbdir+"/ContentImg.png")
      #  data["VesselOverlay"] = cv2.imread(sbdir + "/VesselOverlay.png")
       # data["ContentOverlay"] = cv2.imread(sbdir + "/ContentOverlay.png")

        data["VesselMask"] = cv2.imread(sbdir + "/VesselMask.png", 0)>0
        data["ContentMask"] = cv2.imread(sbdir + "/ContentMask.png",0)>0
        data["VesselXYZMask"] = cv2.imread(sbdir + "/Vessel_ValidDepthMask.png", 0)>0
        data["ContentXYZMask"] = cv2.imread(sbdir + "/Content_ValidDepthMask.png", 0)>0
        data["VesselXYZ"] = cv2.imread(sbdir + "/Vessel_XYZMap.exr",  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH )
        data["ContentXYZ"] = cv2.imread(sbdir + "/Content_XYZMap.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        data["ROI"]=np.ones(data["VesselMask"].shape)

        #-------------------------------------Expand dim to create batch like shape----------------------------------------------------------------------
        for nm in data:
            data[nm]=np.expand_dims(data[nm],axis=0).astype(np.float32)

        #---------------------------------return----------------------------------------------------------------------------
        return data
