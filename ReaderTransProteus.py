

import numpy as np
import os
import cv2
import json
import threading
import Visuallization as vis

MapsAndDepths= {"VesselMask":1, # List of maps to use and  their depths (layers)
               "VesselOpening_Depth":1,
               "VesselWithContentRGB":3,
               "VesselWithContentNormal":3,
               #"VesselWithContentDepth":1,
               "EmptyVessel_Depth":1,
               "ContentNormal":3,
               "ContentDepth":1,
               "ContentMask":3,
               "ContentMaskClean":1,
               "VesselOpeningMask":1,
               "ROI":1,
               "VesselXYZ":3,
               "ContentXYZ":3,
               "VesselOpening_XYZ":3}

##############################################################################################
#########################################################################################################################
class Reader:
# Initiate reader and define the main parameters for the data reader
    def __init__(self, MainDir=r"", MaxBatchSize=1,MinSize=250,MaxSize=900,MaxPixels=800*800*5,TrainingMode=True):
        self.MaxBatchSize=MaxBatchSize # Max number of image in batch
        self.MinSize=MinSize # Min image width and hight in pixels
        self.MaxSize=MaxSize #Max image width and hight in pixels
        self.MaxPixels=MaxPixels # Max number of pixel in all the batch (reduce to solve oom out of memory issues)
        self.epoch = 0 # Training Epoch
        self.itr = 0 # Training iteratation
        self.epoch = 0
# ----------------------------------------Create list of annotations with maps and dicitionaries per annotation--------------------------------------------------------------------------------------------------------------
        self.AnnList = [] # Image/annotation list
        print("Creating annotation list for reader this might take a while")
        for AnnDir in os.listdir(MainDir):
            AnnDir=MainDir+"//"+AnnDir+"//"
            Ent={}
            if os.path.isfile(AnnDir+"//ContentMaterial.json"): Ent["ContentMaterial"]=AnnDir+"//ContentMaterial.json"
            if os.path.isfile(AnnDir + "//VesselMaterial.json"): Ent["VesselMaterial"] = AnnDir + "//VesselMaterial.json"
            if os.path.isfile(AnnDir + "//CameraParameters.json"): Ent["CameraParameters"] = AnnDir + "//CameraParameters.json"
            Ent["VesselMask"] = AnnDir + "//VesselMask.png"
            Ent["VesselOpening_Depth"] = AnnDir + "//VesselOpening_Depth.exr"
            Ent["EmptyVessel_RGB"] = AnnDir + "//EmptyVessel_Frame_0_RGB.jpg"
            Ent["EmptyVessel_Normal"] = AnnDir + "//EmptyVessel_Frame_0_Normal.exr"
            Ent["EmptyVessel_Depth"] = AnnDir + "//EmptyVessel_Frame_0_Depth.exr"
            Ent["MainDir"]=AnnDir
            for nm in os.listdir(AnnDir):
                filepath=AnnDir+"/"+nm
                if ("VesselWithContent" in nm) and ("_RGB.jpg" in nm):
                    EntTemp=Ent.copy()
                    EntTemp["VesselWithContentRGB"]=filepath
                    EntTemp["VesselWithContentNormal"] = filepath.replace("_RGB.jpg","_Normal.exr")
                    EntTemp["VesselWithContentDepth"] =  filepath.replace("_RGB.jpg", "_Depth.exr")

                    EntTemp["ContentRGB"] = EntTemp["VesselWithContentRGB"].replace("VesselWithContent_", "Content_")
                    EntTemp["ContentNormal"] = EntTemp["VesselWithContentNormal"].replace("VesselWithContent_", "Content_")
                    EntTemp["ContentDepth"] = EntTemp["VesselWithContentDepth"].replace("VesselWithContent_", "Content_")
                    EntTemp["ContentMask"] =  EntTemp["ContentDepth"].replace("_Depth.exr","_Mask.png")
                    self.AnnList.append(EntTemp)
#------------------------------------Check list for errors----------------------------------------------------------------------------------------------------
        for Ent in self.AnnList:
            for nm in Ent:
                print(Ent[nm])
                if (".exr" in Ent[nm]) or (".png" in Ent[nm]) or (".jpg" in Ent[nm]):
                    if os.path.exists(Ent[nm]):
                        print("Confirmed")
                    else:
                        print("Missing")
                        exit()
#------------------------------------------------------------------------------------------------------------

        print("done making file list Total=" + str(len(self.AnnList)))
        if TrainingMode:
            self.StartLoadBatch()
        self.AnnData=False
#############################################################################################################################
#############################################################################################################################
# Crop and resize image and mask and ROI to feet batch size

    def CropResize(self, Maps, Hb, Wb):
            # ========================resize image if it too small to the batch size==================================================================================

            h, w = Maps["ROI"].shape
            Bs = np.min((h / Hb, w / Wb))
            if (Bs < 1 or Bs>3 or np.random.rand()<0.2):  # Resize image and mask to batch size if mask is smaller then batch or if segment bounding box larger then batch image size
                h = int(h / Bs) + 1
                w = int(w / Bs) + 1
                for nm in Maps:
                    if hasattr(Maps[nm], "shape"):# check if array
                                    if "RGB" in nm: Maps[nm] = cv2.resize(Maps[nm], dsize=(w, h), interpolation=cv2.INTER_LINEAR)
                                    else: Maps[nm] = cv2.resize(Maps[nm], dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            # =======================Crop image to fit batch size===================================================================================

            if w > Wb:
                X0 = int((w - Wb)/2-0.1)#np.random.randint(w - Wb)
            else:
                X0 = 0
            if h > Hb:
                Y0 = int((h - Hb)/2-0.1)#np.random.randint(h - Hb)
            else:
                Y0 = 0

            for nm in Maps:
                if hasattr(Maps[nm], "shape"):  # check if array
                       Maps[nm] = Maps[nm][Y0:Y0 + Hb, X0:X0 + Wb]

            #-------------------If still not batch size resize again-------------------------------
            for nm in Maps:
                if hasattr(Maps[nm], "shape"): # check if array
                    if not (Maps[nm].shape[0] == Hb and Maps[nm].shape[1] == Wb):
                         Maps[nm] = cv2.resize(Maps[nm], dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)

            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
            return Maps

######################################################Augmented Image##################################################################################################################################

    def Augment(self,Maps):
        for nm in Maps:
            if "RGB" in nm:
                if np.random.rand() < 0.1: # Gaussian blur
                            Maps[nm] = cv2.GaussianBlur(Maps[nm], (5, 5), 0)

                if np.random.rand() < 0.1:  # Dark light
                            Maps[nm] =  Maps[nm] * (0.5 + np.random.rand() * 0.65)
                            Maps[nm][ Maps[nm]>255]=255

                if np.random.rand() < 0.1:  # GreyScale
                            Gr=Maps[nm].mean(axis=2)
                            r=np.random.rand()

                            Maps[nm][:, :, 0] = Maps[nm][:, :, 0] * r + Gr * (1 - r)
                            Maps[nm][:, :, 1] = Maps[nm][:, :, 1] * r + Gr * (1 - r)
                            Maps[nm][:, :, 2] = Maps[nm][:, :, 2] * r + Gr * (1 - r)


        return Maps
############################################################################################################################################################

#                                        Convert Depth map to xyz map

##########################################################################################################################################################
    def Depth2XYZ(self,DepthMap,CameraParameters):
# resolution_x resolutiopn in pixels
# shift_x It seems to be in percentage or fraction of the render size (in pixels). That means that a lens shift of 1 shifts the camera exactly one frame unit in a certain direction, making it frame exactly outside the previous default framed area. ng in Full HD, that is 1920 x 1080 pixel image; a frame shift if 1 unit will shift exactly 1920 pixels in any direction, that is up/down/left/right.
#sensor_fit  Method to fit image and field of view angle inside the sensor AUTO Auto, Fit to the sensor width or height depending on image resolution.Type:	enum in [‘AUTO’, ‘HORIZONTAL’, ‘VERTICAL’], default ‘AUTO’
# sensor_height Vertical size of the image sensor area in millimeters Type:	float in [1, inf], default 0.0
# sensor_width Horizontal size of the image sensor area in millimeters probably the only one that is actually used
                  #    print(CameraParameters)
                      hpx,wpx=DepthMap.shape
                      MaxDimPix=np.max([hpx,wpx]) # Size of the image in pixels
                      shift_x=shift_y=0
                      sensor_width=36.0
                      if  'shift_x' in CameraParameters: shift_x=MaxDimPix*CameraParameters['shift_x'] # How much the center of of the image moved compared to center of the image
                      if  'shift_x' in CameraParameters: shift_y=MaxDimPix*CameraParameters['shift_y'] # How much the center of of the image moved compared to center of the image
                      if  'sensor_width' in CameraParameters: sensor_width=CameraParameters['sensor_width'] # Size of the image in mm


                      GridY = (list(range(hpx))-shift_y-hpx/2)*sensor_width/MaxDimPix
                      GridY = np.transpose(np.tile(GridY, (hpx,1)))


                      GridX = (list(range(wpx))+shift_x-wpx/2)*sensor_width/MaxDimPix # Might be +shift x https://www.rojtberg.net/1601/from-blender-to-opencv-camera-and-back/
                      GridX = np.tile(GridX, (wpx, 1))




                      R = np.sqrt(GridX**2+GridY**2+CameraParameters['Focal Length']**2)+0.0000001


                      XYZ=np.zeros([hpx,wpx,3],dtype=np.float32)
                      XYZ[:,:,2] = DepthMap * CameraParameters['Focal Length'] / R
                      XYZ[:,:,1] = DepthMap * GridY / R
                      XYZ[:,:,0] = DepthMap * GridX / R

                      return XYZ





##################################################################################################################################################################
# ==========================Read image annotation and data===============================================================================================
    def LoadNext(self, pos, Hb, Wb):
# ----------------------Select random example from the batch-------------------------------------------------
            AnnInd = np.random.randint(len(self.AnnList))
            Ann=self.AnnList[AnnInd]
            Maps = {} # List of all maps to read


            with open(Ann["CameraParameters"]) as f:
                     Maps["CameraParameters"] = json.load(f)

            for nm in MapsAndDepths: #Load segmentation maps and XYZ maps
                if not nm in Ann: continue
                Path = Ann[nm]
                Depth =  MapsAndDepths[nm]
                if ".exr" in Path: # Depth maps and normal maps
                    I=cv2.imread(Path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH )
                    if np.ndim(I)>=3 and Depth==1:
                        I=I[:,:,0]
                else: # Segmentation mask
                    if   Depth==1:
                        I=cv2.imread(Path,  0)
                    else:
                        I = cv2.imread(Path)
                Maps[nm] = I.astype(np.float32)

 #------------Process segmentation mask------------------------------------------------------------------------------
            Maps["VesselMask"][Maps["VesselMask"]>0]=1
            Maps["VesselOpeningMask"] = (Maps["VesselOpening_Depth"] < 5000).astype(np.float32)
            Maps["ContentMaskClean"]=(Maps["ContentMask"].sum(2)>0).astype(np.float32)
            Maps["ROI"]=np.ones(Maps["VesselMask"].shape,dtype=np.float32)


            IgnoreMask = Maps["ContentMask"][:, :, 2]  # Undistorted content not viewed trough the vessel walls is ignored (leaking)
            IgnoreMask[Maps["ContentMask"][:, :, 1] > 0] = 0  # Contet viewed trough vessel opening is not ignored
            IgnoreMask[(Maps["ContentMask"][:, :, 1] * Maps["ContentMask"][:, :, 0]) > 0] = 1  # areas where the content is viewd trough the vessel floor are ignored
            Maps["ROI"][IgnoreMask > 0] = 0  # Region of interest where the annotation is well defined
#----------------------------------Generate XYZ map---------------------------------------------------------------------------------------------------
            Maps["EmptyVessel_Depth"][Maps["EmptyVessel_Depth"]>5000]=0 # Remove far away background points
            Maps["VesselOpening_Depth"][Maps["VesselOpening_Depth"]>5000]=0 # Remove far away background points
            Maps["ContentDepth"][Maps["ContentDepth"]>5000]=0 # Remove far away background points

            Maps["VesselXYZ"]=self.Depth2XYZ(Maps["EmptyVessel_Depth"], Maps["CameraParameters"]) # Convert depth to XYZ
            Maps["ContentXYZ"] = self.Depth2XYZ(Maps["ContentDepth"], Maps["CameraParameters"])# Convert depth to XYZ
            Maps["VesselOpening_XYZ"] = self.Depth2XYZ(Maps["VesselOpening_Depth"], Maps["CameraParameters"])# Convert depth to XYZ

#-----------------------------------Augment Crop and resize-----------------------------------------------------------------------------------------------------
          #  self.before = Maps['VesselWithContentRGB'].copy()
            Maps = self.Augment(Maps)
            if Hb!=-1:
               Maps = self.CropResize(Maps, Hb, Wb)


#-----------------Put maps in the batch-----------------------------------------------------------------------------------------------------------
            for nm in Maps:
                if nm in  self.Maps:
                     self.Maps[nm][pos]=Maps[nm]

############################################################################################################################################################
# Start load batch of images (multi  thread the reading will occur in background and will will be ready once waitLoad batch as run
    def StartLoadBatch(self):
        # =====================Initiate batch=============================================================================================
        while True:
            Hb =np.random.randint(low=self.MinSize, high=self.MaxSize)  # Batch hight #900
            Wb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # batch  width #900
            if Hb*Wb<self.MaxPixels: break
        BatchSize =  np.int(np.min((np.floor(self.MaxPixels / (Hb * Wb)), self.MaxBatchSize)))
        #====================Create empty maps batch==========================================================
        self.Maps={}
        for nm in MapsAndDepths:
            if MapsAndDepths[nm]>1:
                self.Maps[nm]= np.zeros([BatchSize, Hb, Wb,MapsAndDepths[nm]], dtype=np.float32)
            else:
                self.Maps[nm] = np.zeros([BatchSize, Hb, Wb], dtype=np.float32)
        #===================Start reading data multithreaded====================================================
        self.thread_list = []
        for pos in range(BatchSize):
            th=threading.Thread(target=self.LoadNext,name="threadReader"+str(pos),args=(pos,Hb,Wb))
            self.thread_list.append(th)
            th.start()
###########################################################################################################
#Wait until the data batch loading started at StartLoadBatch is finished
    def WaitLoadBatch(self):
            for th in self.thread_list:
                 th.join()

########################################################################################################################################################################################
    def LoadBatch(self):
# Load batch for training (muti threaded  run in parallel with the training proccess)
# return previously  loaded batch and start loading new batch
            self.WaitLoadBatch()
            Maps=self.Maps
        #    self.before2 = self.before.copy()
        #    self.after2 = self.after.copy()
            self.StartLoadBatch()
            return Maps

##################################################################################################################################################################
# ==========================Read single image annotation and data with no augmentation for testing===============================================================================================
    def LoadSingle(self, MaxSize=1000):
#  pick the next image in the list
            if self.itr>=len(self.AnnList):
                self.itr=0
                self.epoch+=1

            Ann=self.AnnList[self.itr]
            self.itr+=1
# -------------Read data-----------------------------
            Maps = {} # Annotaion maps


            with open(Ann["CameraParameters"]) as f:
                Maps["CameraParameters"] = json.load(f)

            for nm in MapsAndDepths: # Read depth maps and segmentation mask
                if not nm in Ann: continue
                Path = Ann[nm]
                Depth = MapsAndDepths[nm]
                if ".exr" in Path: #  read Depth and normals
                    I = cv2.imread(Path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    if np.ndim(I) >= 3 and Depth == 1:
                        I = I[:, :, 0]
                else: # Read  masks and image
                    if Depth == 1:
                        I = cv2.imread(Path, 0)
                    else:
                        I = cv2.imread(Path)

                Maps[nm] = I.astype(np.float32)

#---------------Proccess  segmentation masks--------------------------------------------------------
            Maps["ContentRGB"]=cv2.imread(Ann["ContentRGB"])
            Maps["VesselMask"][Maps["VesselMask"] > 0] = 1
            Maps["VesselOpeningMask"] = (Maps["VesselOpening_Depth"] < 5000).astype(np.float32)
            Maps["ContentMaskClean"] = (Maps["ContentMask"].sum(2) > 0).astype(np.float32)
            Maps["ROI"] = np.ones(Maps["VesselMask"].shape, dtype=np.float32)

            IgnoreMask = Maps["ContentMask"][:, :, 2]  # Undistorted content not viewed trough the vessel walls is ignored (leaking)
            IgnoreMask[Maps["ContentMask"][:, :, 1] > 0] = 0  # Contet viewed trough vessel opening is not ignored
            IgnoreMask[(Maps["ContentMask"][:, :, 1] * Maps["ContentMask"][:, :, 0]) > 0] = 1  # areas where the content is viewd trough the vessel floor are ignored
            Maps["ROI"][IgnoreMask > 0] = 0  # Region of interest where the annotation is well defined
#----------------------------------Generate XYZ maps from depth maps--------------------------------------------------------------------------------------------------
            Maps["EmptyVessel_Depth"][Maps["EmptyVessel_Depth"]>5000]=0 # Remove far away background points
            Maps["VesselOpening_Depth"][Maps["VesselOpening_Depth"]>5000]=0 # Remove far away background points
            Maps["ContentDepth"][Maps["ContentDepth"]>5000]=0 # Remove far away background points

            Maps["VesselXYZ"]=self.Depth2XYZ(Maps["EmptyVessel_Depth"], Maps["CameraParameters"]) # convert depth to XYZ
            Maps["ContentXYZ"] = self.Depth2XYZ(Maps["ContentDepth"], Maps["CameraParameters"]) # convert depth to XYZ
            Maps["VesselOpening_XYZ"] = self.Depth2XYZ(Maps["VesselOpening_Depth"], Maps["CameraParameters"]) # convert depth to XYZ

#-----------------------More  changes  to match specific key format expected by evaluting function---------------------------------------
            del Maps["CameraParameters"]

            Maps["ContentMask"] = Maps["ContentMaskClean"]
            Maps["VesselXYZMask"] = Maps["VesselMask"].copy()
            Maps["ContentXYZMask"] =  Maps["ContentMask"].copy()
#-----------------------Resize if too big---------------------------------------------------------------------------------------------------------------------------------
            h,w=Maps["VesselMask"].shape
            r=np.min([MaxSize/h,MaxSize/w])
            if r<1:
                for nm in Maps:
                     Maps[nm] = cv2.resize(Maps[nm], dsize=(int(r*w), (r*w)), interpolation=cv2.INTER_NEAREST)
#--------------------------Expand dimension to create batch like array expected by the net------------------------------------
            for nm in Maps: Maps[nm]=np.expand_dims(Maps[nm],axis=0)
#------------------Return------------------------------------------------------
            return Maps
##################################################################################################################################################################
