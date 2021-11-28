## Reader for labpics dataset


import numpy as np
import os
import cv2
import json
import threading
#import ClassesGroups
import Visuallization as vis

MapsAndDepths= {"VesselMask":1, # Depth/Layers
               "VesselWithContentRGB":3,
               "ContentMaskClean":1,
               "ROI":1}
#########################################################################################################################
class Reader:
    # Initiate reader and define the main parameters for the data reader
    def __init__(self, MainDir=r"", MaxBatchSize=100, MinSize=250, MaxSize=1000, MaxPixels=800 * 800 * 5):

        self.MaxBatchSize = MaxBatchSize  # Max number of image in batch
        self.MinSize = MinSize  # Min image width and height in pixels
        self.MaxSize = MaxSize  # Max image width and height in pixels
        self.MaxPixels = MaxPixels  # Max number of pixel in all the batch (reduce to solve  out of memory issues)
        self.epoch = 0  # Training Epoch
        self.itr = 0  # Training iteratation
        # ----------------------------------------Create list of annotations arranged by class--------------------------------------------------------------------------------------------------------------
        self.AnnList = []  # Image/annotation list
        self.AnnByCat = {}  # Image/annotation list by class

        print("Creating annotation list for reader this might take a while")
        for AnnDir in os.listdir(MainDir): # List of all example
            self.AnnList.append(MainDir + "/" + AnnDir)

        # ------------------------------------------------------------------------------------------------------------

        print("done making file list Total=" + str(len(self.AnnList)))

        self.StartLoadBatch()  # Start loading semantic maps batch (multi threaded)
        self.AnnData = False


#############################################################################################################################

# Crop and resize image and mask and ROI to feet batch size

#############################################################################################################################
# Crop and resize image and maps and ROI to feet batch size
    def CropResize(self, Maps, Hb, Wb):
            # ========================resize image if it too small to the batch size==================================================================================
            h, w = Maps["ROI"].shape
            Bs = np.min((h / Hb, w / Wb))
            if (
                    Bs < 1 or Bs > 3 or np.random.rand() < 0.2):  # Resize image and mask to batch size if mask is smaller then batch or if segment bounding box larger then batch image size
                h = int(h / Bs) + 1
                w = int(w / Bs) + 1
                for nm in Maps:
                    if hasattr(Maps[nm], "shape"):  # check if array
                        if "RGB" in nm:
                            Maps[nm] = cv2.resize(Maps[nm], dsize=(w, h), interpolation=cv2.INTER_LINEAR)
                        else:
                            Maps[nm] = cv2.resize(Maps[nm], dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            # =======================Crop image to fit batch size===================================================================================

            if w > Wb:
                X0 = int(w - Wb)/2#np.random.randint(w - Wb)
            else:
                X0 = 0
            if h > Hb:
                Y0 = int(h - Hb)/2#np.random.randint(h - Hb)
            else:
                Y0 = 0

            for nm in Maps:
                if hasattr(Maps[nm], "shape"):  # check if array
                    Maps[nm] = Maps[nm][Y0:Y0 + Hb, X0:X0 + Wb]

            # -------------------If still not batch size resize again--------------------------------------------
            for nm in Maps:
                if hasattr(Maps[nm], "shape"):  # check if array
                    if not (Maps[nm].shape[0] == Hb and Maps[nm].shape[1] == Wb):
                        Maps[nm] = cv2.resize(Maps[nm], dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)

            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
            return Maps


######################################################Augmented Image##################################################################################################################################

    def Augment(self, Maps):
            if np.random.rand() < 0.5:  # flip left right
                for nm in Maps:
                    if hasattr(Maps[nm], "shape"):
                        Maps[nm] = np.fliplr(Maps[nm])
            for nm in Maps:
                if "RGB" in nm:
                    if np.random.rand() < 0.1:  # Gaussian blur
                        Maps[nm] = cv2.GaussianBlur(Maps[nm], (5, 5), 0)

                    if np.random.rand() < 0.1:  # Dark light
                        Maps[nm] = Maps[nm] * (0.5 + np.random.rand() * 0.65)
                        Maps[nm][Maps[nm] > 255] = 255

                    if np.random.rand() < 0.1:  # GreyScale
                        Gr = Maps[nm].mean(axis=2)
                        r = np.random.rand()

                        Maps[nm][:, :, 0] = Maps[nm][:, :, 0] * r + Gr * (1 - r)
                        Maps[nm][:, :, 1] = Maps[nm][:, :, 1] * r + Gr * (1 - r)
                        Maps[nm][:, :, 2] = Maps[nm][:, :, 2] * r + Gr * (1 - r)

            return Maps

     ##################################################################################################################################################################

    # Read single image and annotation into batch

    def LoadNext(self, pos, Hb, Wb):
        # -----------------------------------select image-----------------------------------------------------------------------------------------------------
        AnnInd = np.random.randint(len(self.AnnList))
       # #AnnInd=1220
       # print(AnnInd)
        InPath = self.AnnList[AnnInd]
       # print(InPath)
       # print("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{")
      #  data = json.load(open(InPath + '/Data.json', 'r'))
        # print(Ann)
        Img = cv2.imread(InPath + "/Image.jpg")  # Load Image
        if (Img.ndim == 2):  # If grayscale turn to rgb
            Img = np.expand_dims(Img, 3)
            Img = np.concatenate([Img, Img, Img], axis=2)
        Img = Img[:, :, 0:3]  # Get first 3 channels in case there are more

        # -----------------------------Read All segmentation mask if exists else set to zero----------------------------------------------------------------------------------------
        SemanticDir = InPath + r"/SemanticMaps/FullImage/"

        VesselMask = np.zeros(Img.shape)
        FilledMask = np.zeros(Img.shape)
        PartsMask = np.zeros(Img.shape)
        Ignore = np.zeros(Img.shape[0:2])
        MaterialScattered = np.zeros(Img.shape)

        if os.path.exists(SemanticDir+"//Transparent.png"): VesselMask =cv2.imread(SemanticDir+"//Transparent.png")
        if os.path.exists(SemanticDir + "//Filled.png"): FilledMask = cv2.imread(SemanticDir + "//Filled.png")
        if os.path.exists(SemanticDir + "//PartInsideVessel.png"): PartsMask = cv2.imread(SemanticDir + "//PartInsideVessel.png")
        if os.path.exists(SemanticDir + "//MaterialScattered.png"):
            MaterialScattered = cv2.imread(SemanticDir + "//MaterialScattered.png")
            print("Reading material scattered")
        if os.path.exists(InPath + "//Ignore.png"): Ignore = cv2.imread(InPath+ "//Ignore.png",0)

        Msk={}

        Msk["VesselWithContentRGB"] = Img
        Msk["VesselMask"] = (VesselMask[:,:,0]>0).astype(np.float32)
        Msk["VesselMask"][PartsMask[:,:,0]>0] = 1
        Msk["ROI"] = (1 - Ignore).astype(np.float32)
        Msk["ROI"][FilledMask[:,:,2]>15] = 0
        Msk["ROI"][MaterialScattered[:,:,2]>0] = 0
        Msk["ContentMaskClean"] = (FilledMask[:, :, 0]>0).astype(np.float32)*Msk["VesselMask"]
        Msk["ContentMaskClean"] = (PartsMask[:,:,0] > 0).astype(np.float32) * Msk["VesselMask"]

        # -----------------------------------Augment Crop and resize-----------------------------------------------------------------------------------------------------
        #  self.before = Maps['VesselWithContentRGB'].copy()
        Maps = self.Augment(Msk)
        if Hb != -1:
            Maps = self.CropResize(Maps, Hb, Wb)

        #   self.after=np.hstack([cv2.resize(self.before,(Wb,Hb)),Maps['VesselWithContentRGB'].copy()])
        # ----------------------Generate forward and background segment mask-----------------------------------------------------------------------------------------------------------
        for nm in Maps:
            if nm in self.Maps:
                self.Maps[nm][pos] = Maps[nm]
############################################################################################################################################################
# Start load batch of images (multi  thread the reading will occur in background and will will be ready once waitLoad batch as run
    def StartLoadBatch(self):
        # =====================Initiate batch=============================================================================================
        while True:
            Hb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # Batch hight #900
            Wb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # batch  width #900
            if Hb * Wb < self.MaxPixels: break
        BatchSize = np.int(np.min((np.floor(self.MaxPixels / (Hb * Wb)), self.MaxBatchSize)))
        # ===================Create empty batch ===========================================================
        self.Maps = {}
        for nm in MapsAndDepths: # Create enoty
            if MapsAndDepths[nm] > 1:
                self.Maps[nm] = np.zeros([BatchSize, Hb, Wb, MapsAndDepths[nm]], dtype=np.float32)
            else:
                self.Maps[nm] = np.zeros([BatchSize, Hb, Wb], dtype=np.float32)
        # ====================Start reading data multithreaded===================================================
        self.thread_list = []
        for pos in range(BatchSize):
            th = threading.Thread(target=self.LoadNext, name="threadReader" + str(pos), args=(pos, Hb, Wb))
            self.thread_list.append(th)
            th.start()

    ###########################################################################################################
    # Wait until the data batch loading started at StartLoadBatch is finished
    def WaitLoadBatch(self):
        for th in self.thread_list:
            th.join()

    ########################################################################################################################################################################################
    def LoadBatch(self):
        # Load batch for training (muti threaded  run in parallel with the training proccess)
        # return previously  loaded batch and start loading new batch
        self.WaitLoadBatch()
        Maps = self.Maps
        #    self.before2 = self.before.copy()
        #    self.after2 = self.after.copy()
        self.StartLoadBatch()
        return Maps
