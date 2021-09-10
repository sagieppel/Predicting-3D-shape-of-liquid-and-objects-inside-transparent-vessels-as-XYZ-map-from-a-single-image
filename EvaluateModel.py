# Evaluate next XYZ and mask predictions
#...............................Imports..................................................................
import os
import ChamferDistance
import numpy
import numpy as np
import FCN_NetModel as NET_FCN # The net Class
import torch
import torch.nn as nn
import torch.nn.functional as F
import LossFunctions

import Visuallization as vis
import cv2



############################CREATE XYZ loss class ########################################################################################################

LossXYZ=LossFunctions.Loss()

##################################Input paramaters#########################################################################################
UseRealSenseDataset=False # Use RealSense Test set (True) or Virtual set (False), The realsense real photo dataset of TransProteus have different reader


#.................................Main Input parametrs...........................................................................................
TestFolder = r"Example/Train/TranProteus//" # input folder
#TestFolder = r"Datasets/TranProteus/RealSense/Data//"
#TestFolder = r"TranProteus/Training/FlatLiquidAll/"
Trained_model_path =  r"logs//Defult.torch" # Trained model path
OutputStatisticsFile = "Statics.txt"

MaxSize=1000# max image dimension
UseChamfer=True # Evaluate chamfer distance (this takes lots of time)
#SetNormalizationUsing="ContentXYZ"
SetNormalizationUsing="VesselXYZ" # Normalize prediction scale to GT scale by matching the vessel scale
#************************************Group of classes to predict and use********************************************************************************************************
MaskClasses =  {}
XYZList = ["VesselXYZ","ContentXYZ","VesselOpening_XYZ"] # XYZ maps names
MaskList = ["VesselMask","ContentMask","VesselOpeningMask"] # Segmentation Masks names
XYZ2Mask={"VesselXYZ":"VesselXYZMask","ContentXYZ":"ContentXYZMask","VesselOpening_XYZ":"VesselOpeningMask"} # XYZ map and corresponding segmentation maks
Statics={""}
#------------------------------------------------------------------------------------------------------------
if UseRealSenseDataset:
    import ReadRealSense as ReaderData
else:
    import ReaderTransProteus as ReaderData

#===========================================================================================================================================

# https://arxiv.org/pdf/1406.2283.pdf





#=========================Load net weights====================================================================================================================

#---------------------Create and Initiate net and load net------------------------------------------------------------------------------------
Net=NET_FCN.Net(MaskList=MaskList,XYZList=XYZList) # Create net and load pretrained
Net.load_state_dict(torch.load(Trained_model_path))
Net=Net.cuda().eval()

#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------

Reader=ReaderData.Reader(TestFolder)


#-------------------------------Create Evaluation statistics dictionary for XYZ--------------------------------------------------------------------
# https://arxiv.org/pdf/1406.2283.pdf
##https://cseweb.ucsd.edu//~haosu/papers/SI2PC_arxiv_submit.pdf
EvalTypes=["RMSE","MAE","TSS",r"MeanError//GTMaxDist","MeanError//stdv","MeanError//MAD","SumPixels"] # TSS total sum of squares, RSS Sum of Squares residuals
if UseChamfer: EvalTypes+=["ChamferDist//GT_MaxDst","ChamferDist//GT_STDV","ChamferDist//GT_Max_Distance"]
StatXYZ={} # Sum All statistics across
for nm in XYZList:
    StatXYZ[nm]={}
    for et in EvalTypes:
        StatXYZ[nm][et]=0
#---------------------------------------Create Statistics dictionaries for Mask IOU for segmentation-------------------------------------------------------------

MaskEvalType=["InterSection","Union","SumGT","SumPrd"]
StatMask={}
for nm in MaskList:
    StatMask[nm]={}
    for et in MaskEvalType:
        StatMask[nm][et]=0
#
# #---------------------Create reader------------------------------------------------------------------------------
#
# GT0 = Reader.LoadSingle()

#-----------------Start evaluation---------------------------------------------------------------------------------
while (Reader.epoch==0 and Reader.itr<100): # Test 100 example or one epoch
    GT = Reader.LoadSingle() # Load example

    print("------------------------------", Reader.itr, "------------------------------------------------")
    #***************************************************************************************************
#    batchSize=GT["VesselWithContentRGB"].shape[0]
 #    for i in range(batchSize):
 #       for nm in GT:
 #
 #             print(nm, GT[nm][i].max(),GT[nm][i].min())
 #             tmIm = GT[nm][i].copy()
 #             if GT[nm][i].max()>255 or GT[nm][i].min()<0 or np.ndim(GT[nm][i])==2:
 #                 if tmIm.max()>tmIm.min():
 #                     tmIm[tmIm>1000]=0
 #                     tmIm = tmIm-tmIm.min()
 #                     tmIm = tmIm/tmIm.max()*255
 #                 print(nm,"New", tmIm.max(), tmIm.min())
 #                 if np.ndim(tmIm)==2: tmIm=cv2.cvtColor(tmIm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
 #             vis.show(np.hstack([tmIm,GT["VesselWithContentRGB"][i].astype(np.uint8)]) ,nm+ " Max=" + str(GT[nm][i].max()) + " Min=" + str(GT[nm][i].min()))
 # #  *****************************************************************************
    print("RUN PREDICITION")

    with torch.no_grad():
          PrdXYZ, PrdProb, PrdMask = Net.forward(Images=GT["VesselWithContentRGB"]) # Run net inference and get prediction
 #    MasksForDisplay=[]
 #    XYZForDisplay=[]
    NormConst=[] # Scale normalization constant
######################################Put the object u want to  first in the list this object will be use for fiding  relativescale############################################################
    SortLst=list(XYZList)
    for fff in range(len(SortLst)):
        if SortLst[fff]==SetNormalizationUsing:
            SortLst[fff] =  SortLst[0]
    SortLst[0]=SetNormalizationUsing
####################################################################################################
    for nm in SortLst:
       #------------------------ROI Punish depth prediction only within mask----------------------------------------------------
               if not nm in GT: continue
               ROI = torch.autograd.Variable(torch.from_numpy(GT[XYZ2Mask[nm]] * GT['ROI']).unsqueeze(1).cuda(),requires_grad=False)
              # ROI = nn.functional.interpolate(ROI, tuple((PrdXYZ[nm].shape[2], PrdXYZ[nm].shape[3])),mode='bilinear', align_corners=False)
               ROI[ROI < 0.9] = 0
               ROI[ROI > 0.9] = 1
       #-------------------------------Convert GT XYZ map to torch -----------------------------------------------------------------------------------------------------
               TGT={}
               TGT[nm] = torch.from_numpy(GT[nm]).cuda().transpose(1,3).transpose(2,3) ###.unsqueeze(1)
               PrdXYZ[nm] = nn.functional.interpolate(PrdXYZ[nm], tuple((TGT[nm].shape[2], TGT[nm].shape[3])), mode='bilinear', align_corners=False)
       #----------------Caclulate relative scale and translation (if not previously calculated----------------------------------------
               if len(NormConst)==0:
                    #  if ApplyRelativeScaleTranslation:#not HierarchicalMode:
                          print("Calculating Normaliztion constant", nm)
                          with torch.no_grad():
                                    CatLoss, NormConst= LossXYZ.DiffrentialLoss(PrdXYZ[nm], TGT[nm], ROI) # Calculate relative scqale
                          VesselGT_XYZ = TGT[nm][0]
                          VesselPrd_XYZ = PrdXYZ[nm][0]
                          ROI=torch.cat([ROI[0],ROI[0],ROI[0]],0)
                          Translation = ((VesselGT_XYZ-VesselPrd_XYZ/NormConst[0])*ROI).sum(1).sum(1)/ROI.sum(1).sum(1) # Calculate translation between object
                      # else: # in case of predicting 3D model of the content using 3D model of the vessel
                      #     NormConst=[1]
                      #     Translation=np.array([0,0,0])
#######################################################################################################################################################3
               print("Normalization Constant",NormConst)
               PXYZ =nn.functional.interpolate(PrdXYZ[nm], tuple((GT['ROI'].shape[1],GT['ROI'].shape[2])), mode='bilinear', align_corners=False) # predicted XYZ map
               PXYZ = PXYZ[0].cpu().detach().numpy()
               PXYZ=np.moveaxis(PXYZ,[0,1,2],[2,0,1])#swapaxes(PXYZ,[2,0,1])
               GXYZ= GT[nm][0] # GT XYZ map
               ROI= GT['ROI'][0]*GT[XYZ2Mask[nm]][0] # Mask of the object evaluation will be done only for pixels belonging to this mask
               SumPixels=ROI.sum()
               if ROI.max()==0: continue


  #-----------------------------------------------------------------------------------------
               #print(NormConst[0])
               # MasksForDisplay.append(ROI)
               # # MasksForDisplay.append(ROI)
               # XYZForDisplay.append(GXYZ)
               # # XYZForDisplay.append(PXYZ)

  #######################Calculatee distances##############################################################################
    # if NormConst[0] < 0.01 or NormConst[0] > 1000:
    #     xx = 0
               for i in range(3):
                   PXYZ[:, :, i] = (PXYZ[:, :, i] / NormConst[0] + Translation.tolist()[i])*ROI # Rescale predicted XYZ map and translate to match GT match (only in the ROI region)
                   GXYZ[:, :, i] *= ROI # Remove predictions outside of ROI region

               #PXYZ=np.fliplr(PXYZ)
               # if np.isnan(gt).any() or np.isnan(prd).any():
               #     xx=0
               dif=np.sum((PXYZ- GXYZ)**2,2)**0.5 # Pygtagorian distance between XYZ points in the sampe pixel
               #dif = np.sum(np.abs(PXYZ - GXYZ), 2)
            #   dif = np.abs(PXYZ[:,:,0] - GXYZ[:,:,0])
               tmpGXYZ=GXYZ.copy()
            #   vis.show(ROI*100,nm+" ROI")
               for i in range(3): tmpGXYZ[:,:,i]= (GXYZ[:,:,i]- (GXYZ[:,:,i].sum()/SumPixels))*ROI # Substract the mean to get deviation from center

               StatXYZ[nm]["TSS"] += ((tmpGXYZ)**2).sum()# Total sum of sqr distance from the mean
               mdv = np.abs(tmpGXYZ).sum() / SumPixels # mean absulote deviation
               sdv = np.sqrt((tmpGXYZ**2).sum(2)).sum() / SumPixels # Standart deviation

               # if np.isnan( StatXYZ[nm]["TSS"]).any():
               #     xx = 0

               StatXYZ[nm]["SumPixels"] += SumPixels
               StatXYZ[nm]["RMSE"]+=(dif**2).sum()
               #StatXYZ[nm]["RMSElog"][i] += (np.log(prd/(GXYZ**2+0.000001))**2).sum()
               SumDif= dif.sum()
               StatXYZ[nm]["MAE"]+=SumDif # Mean absoulte error
               dst=0

               #---------------max distance between ppont in the moder
               for i in range(3): #
                   if GXYZ[:, :, i].max() > 0:
                       GXYZ[:, :, i][ROI == 0] = GXYZ[:, :, i].mean()  # Prevent zero from being the minimum since zero mean out of ROI
                   dst += (GXYZ[:,:,i].max()-GXYZ[:,:,i].min())**2
               dst=dst**0.5 # Max distance between points in the model


               StatXYZ[nm]["MeanError//GTMaxDist"] += SumDif/  (dst+ 0.00000001)
               StatXYZ[nm]["MeanError//stdv"] += SumDif/ (sdv + 0.00000001)
               StatXYZ[nm]["MeanError//MAD"] += SumDif /  (mdv + 0.00000001)
               if UseChamfer: # Calculate chamfer distace
                   AbsChamferDist, SqrChamferDist = ChamferDistance.ChamferDistance(GXYZ, PXYZ, ROI)
                   AbsChamferDist*=SumPixels
                   SqrChamferDist*=SumPixels
                   StatXYZ[nm]["ChamferDist//GT_MaxDst"] += AbsChamferDist / (dst+0.00001) # normalize chamfer distance by max distance between points in the mdoe
                   StatXYZ[nm]["ChamferDist//GT_STDV"] += AbsChamferDist / (sdv+0.00001)# normalize chamfer distance by standart deviation in GT XYZ model
                   StatXYZ[nm]["ChamferDist//GT_Max_Distance"] += AbsChamferDist / (mdv + 0.00000001) # normalize chamfer distance by meandeviation in GT XYZ model

#******************************************************************************************************************************************
               # if nm=="ContentXYZ":
               #     MasksForDisplay.append(ROI)
               #     # MasksForDisplay.append(ROI)
               #     XYZForDisplay.append(GXYZ)
               #     # XYZForDisplay.append(PXYZ)
               #     MasksForDisplay.append((dif<(dst)/3)*ROI)
               #     MasksForDisplay.append((dif > (dst) / 3) * ROI)
               #     XYZForDisplay.append(PXYZ)
               #     XYZForDisplay.append(PXYZ)
               #    print(StatXYZ[nm]["AbsRelDif"][i] / StatXYZ[nm]["SumPixels"][i].sum())
   # vis.DisplayPointClouds2(GT["VesselWithContentRGB"][0],XYZForDisplay,MasksForDisplay)
    # https://arxiv.org/pdf/1406.2283.pdf
###################### Predict segmentation Mask Accuracy IOU#######################################################################
for nm in MaskList:
  if nm in GT:
      ROI=GT['ROI'][0]
      Pmask= nn.functional.interpolate(PrdProb[nm], tuple((GT['ROI'].shape[1], GT['ROI'].shape[2])),mode='bilinear', align_corners=False)
      Pmask=(Pmask[0][1]>0.5).squeeze().cpu().detach().numpy().astype(np.float32) # Predicted mask
      Pmask*=ROI # Limit to the region of interse
      Gmask=GT[nm][0]*ROI # GT mask  limite to region of ineterstr (ROI)
      #****************************************************
      # Im=GT["VesselWithContentRGB"][0].copy()
      # Im[:,:,0][Gmask>0]=0
      # Im[:, :, 1][Pmask > 0] = 0
      # vis.show(Im)
      #***************Calculate IOU********************************************
      InterSection=(Pmask*Gmask).sum()
      Union = (Pmask + Gmask).sum()-InterSection

      StatMask[nm]["Union"]+=Union
      StatMask[nm]["InterSection"] += InterSection
      StatMask[nm]["SumGT"] += (Gmask).sum()
      StatMask[nm]["SumPrd"] += (Pmask).sum()


######################Display final statistics XYZ#################################################################################################

print("\n\n\n########################   3D XYZ statitics ################################\n\n\n")
for nm in XYZList:
    print("\n\n==========================="+nm+"===========================================================\n\n")
    # FinalError={}
    # for ev in range(EvalTypes):
    #     FinalError[ev]=0
    if StatXYZ[nm]["SumPixels"] == 0: continue
    # for i,xyz in enumerate(['x','y','z']):
    #     print("\n---------"+xyz+"-----------------\n")
    SumPix=StatXYZ[nm]["SumPixels"]

    for et in EvalTypes:
        if et=="SumPixels": continue
        Pr=1
        if "RMSE" in et: Pr=0.5
        print("\n",nm,"\t",et,"\t=\t",(StatXYZ[nm][et]/SumPix)**Pr)

    Rsqr=1 - StatXYZ[nm]["RMSE"]/StatXYZ[nm]["TSS"]
    print("\n",nm,"\tR square\t=\t",Rsqr)
    print("\n---------All-----------------\n")

#======================Display Segmentation statistics==============================================================================
print("\n\n\n########################   Segmentation statitics ################################\n\n\n")
for nm in MaskList:
   if StatMask[nm]["Union"]==0: continue
   IOU=StatMask[nm]["InterSection"]/StatMask[nm]["Union"]
   Precision =StatMask[nm]["InterSection"] /(StatMask[nm]["SumPrd"]+0.0001)
   Recall = StatMask[nm]["InterSection"] / (StatMask[nm]["SumGT"]+0.0001)
   print(nm,"\tIOU=\t",IOU,"\tPrecission=\t",Precision,"\tRecall=\t",Recall)