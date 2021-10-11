# Train net that predict XYZ map and segmentation of vessel, vessel content, and vessel opening
#...............................Imports..................................................................
import os
import numpy as np
import FCN_NetModel as NET_FCN # The net Class
import torch
import torch.nn as nn
import torch.nn.functional as F
import ReaderTransProteus as DepthReader
import LossFunctions
import Visuallization as vis
import cv2
import ReaderLabPics

##################################Input paramaters#########################################################################################
#.................................Main Input parametrs...........................................................................................

UseLabPicsDataSet=True # Use the LabPics dataset for training True/False

#----------------TransProteus dataset folders (note that each folder will be used in the proprtions by which it occur (hence same folder appearing twice mean it will be used twice as much in training)

TransProteusFolder={}
TransProteusFolder["Liquid"]=r"Example/Train/TranProteus//"
# TransProteusFolder["ObjectContent"]=r"TranProteus/Training/ObjectContent/"
# TransProteusFolder["ObjectContent2"]=r"TranProteus/Training/SingleObjectContent/"
# TransProteusFolder["LiquidContent"]=r"TranProteus/Training/LiquidContent/"
# TransProteusFolder["LiquidFlat"]=r"TranProteus/Training/FlatSurfaceLiquids/"


#--------------LabPics dataset folder (if LabPics used for trainig)----------------------------------------------------------
if UseLabPicsDataSet:
    LabPicsFolder = {}
    LabPicsFolder["Chemistry"] = r"Example/Train/LabPics//"
    # LabPicsFolder["Medical"] = r"LabPics/LabPicsMedical/LabPicsMedical/Train//"
    # LabPicsFolder["Chemical"] = r"LabPics/LabPicsChemistry/LabPicsChemistry/Train//"
    # LabPicsFolder["Chemical2"] = r"LabPics/LabPicsChemistry/LabPicsChemistry/Train//"
    # LabPicsFolder["Chemical3"] = r"LabPics/LabPicsChemistry/LabPicsChemistry/Train//"

#...............Other training paramters..............................................................................


MinSize=270 # Min image dimension (height or width)
MaxSize=1000# Max image dimension (height or width)
MaxPixels=800*800*2# Max pixels in a batch (not in image), reduce to solve out if memory problems
MaxBatchSize=6# Max images in batch

Trained_model_path="" # Path of trained model weights If you want to return to trained model, else if there is no pretrained mode this should be =""
Learning_Rate=1e-5 # intial learning rate
TrainedModelWeightDir="logs/" # Output Folder where trained model weight and information will be stored

TrainLossTxtFile=TrainedModelWeightDir+"TrainLoss.txt" #Where train losses statitics will be writen
Weight_Decay=4e-5# Weight for the weight decay loss function
MAX_ITERATION = int(100000010) # Max  number of training iteratio

##################List of XYZ maps and segmentation Mask to predict###################################################################################################################3
MaskClasses =  {}
XYZList = ["VesselXYZ","ContentXYZ","VesselOpening_XYZ"] # List of XYZ maps to predict
MaskList = ["VesselMask","ContentMaskClean","VesselOpeningMask"] # List of segmentation Masks to predict
XYZ2Mask={"VesselXYZ":"VesselMask","ContentXYZ":"ContentMaskClean","VesselOpening_XYZ":"VesselOpeningMask"} # Dictionary connecting XYZ maps and segmentation maps of same object
XYZ2LossFactor={"VesselXYZ":1,"ContentXYZ":1.,"VesselOpening_XYZ":0.4} # Weight of loss of XYZ prediction per object (some object will contribute less to the loss function)

#=========================Load net weights====================================================================================================================
InitStep=1
if os.path.exists(TrainedModelWeightDir + "/Defult.torch"):
    Trained_model_path=TrainedModelWeightDir + "/Defult.torch"
if os.path.exists(TrainedModelWeightDir+"/Learning_Rate.npy"):
    Learning_Rate=np.load(TrainedModelWeightDir+"/Learning_Rate.npy")
if os.path.exists(TrainedModelWeightDir+"/itr.npy"): InitStep=int(np.load(TrainedModelWeightDir+"/itr.npy"))

############################CREATE XYZ loss class that will be used to calculate XYZ loss########################################################################################################

LossXYZ=LossFunctions.Loss()

####################Create and Initiate net and create optimizer##########################################################################################3

Net=NET_FCN.Net(MaskList=MaskList,XYZList=XYZList) # Create net and load pretrained

#--------------------if previous model exist load it--------------------------------------------------------------------------------------------
if Trained_model_path!="": # Optional initiate full net by loading a pretrained net
    Net.load_state_dict(torch.load(Trained_model_path))
Net=Net.cuda()
#--------------------------------Optimizer--------------------------------------------------------------------------------------------
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay) # Create adam optimizer

#----------------------------------------Create reader for data sets--------------------------------------------------------------------------------------------------------------
Readers={} #Transproteus readers
for nm in TransProteusFolder:
    Readers[nm]=DepthReader.Reader(TransProteusFolder[nm],MaxBatchSize,MinSize,MaxSize,MaxPixels,TrainingMode=True)

if UseLabPicsDataSet:
    LPReaders={}# Labpics Reader (if used)
    for nm in LabPicsFolder:
              LPReaders[nm] = ReaderLabPics.Reader(LabPicsFolder[nm], MaxBatchSize, MinSize, MaxSize, MaxPixels)


#--------------------------- Create logs files for saving loss during training----------------------------------------------------------------------------------------------------------

if not os.path.exists(TrainedModelWeightDir): os.makedirs(TrainedModelWeightDir) # Create folder for trained weight
torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + "test" + ".torch") # test saving to see the everything is fine

f = open(TrainLossTxtFile, "w+")# Training loss log file
f.write("Iteration\tloss\t Learning Rate=")
f.close()
#-------------------Loss Parameters--------------------------------------------------------------------------------
PrevAvgLoss=0 # Average loss in the past (to compare see if loss as been decrease)
AVGCatLoss={} # Average loss for each prediction


############################################################################################################################
#..............Start Training loop: Main Training....................................................................
print("Start Training")
for itr in range(InitStep,MAX_ITERATION): # Main training loop
    print("------------------------------" , itr , "------------------------------------------------")

    #***************************Reading batch ******************************************************************************
    Mode="Virtual" # Transproteus data
    if UseLabPicsDataSet and np.random.rand()<0.33: Mode="LabPics" # Selecting datset
    if Mode=="Virtual": # Read transproteus
        readertype=list(Readers)[np.random.randint(len(list(Readers)))]  # Pick reader (folder)
        print(readertype)
        GT = Readers[readertype].LoadBatch() # Read batch

    if Mode=="LabPics": # Read Labpics data
        readertype = list(LPReaders)[np.random.randint(len(list(LPReaders)))]  # Pick reader (Folder
        print("Reader type",readertype)
        GT = LPReaders[readertype].LoadBatch() # Read batch

    #***************************************************************************************************
    # batchSize=GT["VesselWithContentRGB"].shape[0]
    # for i in range(batchSize):
    #    for nm in GT:
    #
    #          print(nm, GT[nm][i].max(),GT[nm][i].min())
    #          tmIm = GT[nm][i].copy()
    #          if GT[nm][i].max()>255 or GT[nm][i].min()<0 or np.ndim(GT[nm][i])==2:
    #              if tmIm.max()>tmIm.min():
    #                  tmIm[tmIm>1000]=0
    #                  tmIm = tmIm-tmIm.min()
    #                  tmIm = tmIm/tmIm.max()*255
    #              print(nm,"New", tmIm.max(), tmIm.min())
    #              if np.ndim(tmIm)==2: tmIm=cv2.cvtColor(tmIm, cv2.COLOR_GRAY2BGR)
    #          vis.show(np.hstack([tmIm,GT["VesselWithContentRGB"][i].astype(np.uint8)]) ,nm+ " Max=" + str(GT[nm][i].max()) + " Min=" + str(GT[nm][i].min()))
#*****************************************************************************

    print("RUN PREDICITION")

    PrdXYZ, PrdProb, PrdMask = Net.forward(Images=GT["VesselWithContentRGB"]) # Run net inference and get prediction
    Net.zero_grad()

#------------------------Calculating loss---------------------------------------------------------------------

    CatLoss = {} # will store the Category loss per object

 #**************************************XYZ Map Loss*************************************************************************************************************************

    if Mode == "Virtual": # XYZ loss is calculated only for transproteus
        print("Calculating loss xyz")
        TGT={} # GT XYZ in torch format



        NormConst=[] # Scale constant to normalize the predicted XYZ map
        for nm in XYZList:
           #------------------------ROI Punish XYZ prediction only within  the object mask, resize  ROI to prediction size (prediction map is shrink version of the input image)----------------------------------------------------
                   ROI = torch.autograd.Variable(torch.from_numpy(GT[XYZ2Mask[nm]] * GT['ROI']).unsqueeze(1).cuda(),requires_grad=False) # ROI to torch
                   ROI = nn.functional.interpolate(ROI, tuple((PrdXYZ[nm].shape[2], PrdXYZ['VesselXYZ'].shape[3])),mode='bilinear', align_corners=False) #ROI to output scale
                   ROI[ROI < 0.9]=0 # Resize have led to some intirmidiate values ignore them
                   ROI[ROI > 0.9] = 1 # Resize have led to some intirmidiate values ignore them

           #Convert GT XYZ maps to torch format

                   TGT[nm] = torch.from_numpy(GT[nm]).cuda().transpose(1,3).transpose(2,3) ### GT XYZ to troce
                   TGT[nm].requires_grad = False # Is this do anything ?
                   TGT[nm] = nn.functional.interpolate(TGT[nm], tuple((PrdXYZ[nm].shape[2],PrdXYZ[nm].shape[3])), mode='bilinear', align_corners=False)
                   if len(NormConst)==0: # Calculate relative scale and loss for the main XYZ map, calculate  relative scale constants between predicted anf GT ZYZ
                          print("Calculating Normaliztion constant", nm)
                          CatLoss[nm], NormConst= LossXYZ.DiffrentialLoss(PrdXYZ[nm], TGT[nm], ROI) # Calculate XYZ loss and scalling normalization constants (relative scale
                          VesselGT_XYZ= TGT[nm] # Use this maps as XYZ anchor  to calculate relative translation to other objects (otherwise each object  will different translation)
                          VesselPrd_XYZ = PrdXYZ[nm] # Same as above for predicted XYZ map
                   else: # Calculate depth loss using the difference with the vessel XYZ assuming that the relative scale is all ready known and given by NormConst
                          # Calculating relative Loss
                          print("Predicitng Hierarchical loss ",nm)
                          CatLoss[nm], NormConst = LossXYZ.DiffrentialLoss(PrdXYZ[nm], TGT[nm], ROI, NormConst) #Calculate  XYZ loss within ROI and given known relative scale (NormConstant)
                          CatLoss[nm]*=XYZ2LossFactor[nm] # Loss factor reduce loss for less important objects
                          #if itr>10000: # Start Predicting content only after training the vessel optional
                          #---------------------Loss for translation between different objects, caclulated relative to an anchor object  (otherwise every object will have different translation)-------------------------------------------------------------
                          difPrd = (VesselPrd_XYZ - PrdXYZ[nm]) # Predicted translation between objects to vessel
                          difGT = (VesselGT_XYZ - TGT[nm]) # GT translation between object to vessel
                          for i in range(len(NormConst)): # Check difference between map and points in related maps
                              if NormConst[i] > 0.0001:
                                  CatLoss[nm] += (torch.abs(difGT[i] - difPrd[i] / NormConst[i]).mean(0)*ROI[i][0]).mean()*0.4*XYZ2LossFactor[nm] # Translation loss between objects

###############################################################################################################################################
#******************Segmentation Mask Loss************************************************************************************************************************************
#-----------------------------ROI---------------------------------------------------------------------------
    ROI = torch.autograd.Variable(torch.from_numpy( GT['ROI']).unsqueeze(1).cuda(),requires_grad=False) # Region of interest in the image where loss is calulated
    ROI = nn.functional.interpolate(ROI, tuple((PrdProb[MaskList[0]].shape[2], PrdProb[MaskList[0]].shape[3])),mode='bilinear', align_corners=False) # Resize ROI to prediction
#-------------------
    print("Calculating Mask Loss")

    for nm in MaskList:
             if nm in GT:
                 TGT = torch.autograd.Variable(torch.from_numpy(GT[nm]).cuda(),requires_grad=False).unsqueeze(1) #Convert GT segmentation mask to pytorch
                 TGT = nn.functional.interpolate(TGT, tuple((PrdProb[nm].shape[2], PrdProb[nm].shape[3])),mode='bilinear', align_corners=False) # Resize GT mask to predicted image size (prediction is scaled down version of the image)
                 CatLoss[nm] = -torch.mean(TGT[:,0] * torch.log(PrdProb[nm][:,1]+0.00001) * ROI[:,0]) - torch.mean((1-TGT[:,0]) * torch.log(PrdProb[nm][:,0]+0.0000001) * ROI[:,0]) # Calculate cross entropy loss

             # for ii in range(GT[nm].shape[0]):
             #        prdmsk=cv2.resize(PrdMask[nm][ii].data.cpu().numpy().astype(np.uint8),(GT[nm][ii].shape[1],GT[nm][ii].shape[0]))
             #        vis.show(np.hstack([GT[nm][ii].astype(np.uint8),prdmsk])*255)
#==========================================================================================================================
#---------------Calculate Total Loss  and average loss by using the sum of all objects losses----------------------------------------------------------------------------------------------------------
    print("Calculating Total Loss")
    fr = 1 / np.min([itr - InitStep + 1, 2000])
    TotalLoss=0 # will be use for backptop
    AVGCatLoss["XYZ"]=0  #will be use to collect statitics
    AVGCatLoss["Mask"]=0#will be use to collect statitics
    AVGCatLoss["Total"] = 0#will be use to collect statitics
    for nm in CatLoss: # Go over all object losses and sum them
        if not nm in AVGCatLoss: AVGCatLoss[nm]=0
        if  CatLoss[nm]>0:
                AVGCatLoss[nm]=(1 - fr) * AVGCatLoss[nm] + fr * CatLoss[nm].data.cpu().numpy()
        TotalLoss+=CatLoss[nm]

        if "XYZ" in nm: AVGCatLoss["XYZ"]+=AVGCatLoss[nm]
        if "Mask" in nm: AVGCatLoss["Mask"] += AVGCatLoss[nm]
        AVGCatLoss["Total"] +=AVGCatLoss[nm]
#--------------Apply backpropogation-----------------------------------------------------------------------------------
    print("Back Prop")
    TotalLoss.backward() # Backpropogate loss
    optimizer.step() # Apply gradient descent change to weight
    print("save")



###############################################################################################################################
#===================Display, Save and update learning rate======================================================================================
#########################################################################################################################33

# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
    if itr % 300 == 0:# and itr>0: #Save model weight once every 300 steps, temp file
        print("Saving Model to file in "+TrainedModelWeightDir+"/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/DefultBack.torch")
        print("model saved")
        np.save(TrainedModelWeightDir+"/Learning_Rate.npy",Learning_Rate)
        np.save(TrainedModelWeightDir+"/itr.npy",itr)
    if itr % 60000 == 0 and itr>0: #Save model weight once every 60k steps permenant file
        print("Saving Model to file in "+TrainedModelWeightDir+"/"+ str(itr) + ".torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + str(itr) + ".torch")
        print("model saved")
#......................Write and display train loss..........................................................................
    if itr % 2==0: # Display train loss and write to statics file
        txt="\n"+str(itr)
        for nm in AVGCatLoss:
            txt+="\tAverage Cat Loss["+nm+"] "+str(AVGCatLoss[nm])+"  "
        print(txt)
        #Write train loss to file
        with open(TrainLossTxtFile, "a") as f:
            f.write(txt)
            f.close()
# #----------------Update learning rate -------------------------------------------------------------------------------
    if itr%20000==0:
        if "TotalPrevious" not in AVGCatLoss:
            AVGCatLoss["TotalPrevious"]=AVGCatLoss["Total"]
        elif AVGCatLoss["Total"]*0.95<AVGCatLoss["TotalPrevious"]: # If average loss havent decrease in the last 20k steps update training loss
            Learning_Rate*=0.9 # Reduce learning rate
            if Learning_Rate<=3e-7: # If learning rate to small increae it
                Learning_Rate=5e-6
            print("Learning Rate="+str(Learning_Rate))
            print("======================================================================================================================")
            optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate,weight_decay=Weight_Decay)  # Create adam optimizer with new learning rate
            torch.cuda.empty_cache()  # Empty cuda memory to avoid memory leaks
        AVGCatLoss["TotalPrevious"]=AVGCatLoss["Total"]+0.0000000001 # Save current average loss for later comparison



