# Loss function for XYZ maps
import torch
import copy
import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import Visuallization as vis
######################################################################################################################333
class Loss(nn.Module):
########################################################################################################################
    def __init__(self): # Create class for Loss function for XYZ maps
           super(Loss, self).__init__()
           self.DifLayers={}
           self.ROILayers={}
           ##Dif  filters use to find the distance between two points on the XYZ map
           # ROI filters use on the ROI map to find that  the points is within the ROI (otherwise the distance is invalid)
           self.DifLayers['Horizontal']=torch.from_numpy(np.array(  [[0,0,0] , [0,1,-1], [0,0,0]],dtype=np.float32)).cuda().unsqueeze(0).unsqueeze(0)
           self.ROILayers['Horizontal'] = torch.from_numpy(np.array([[0,0,0] , [0,0,1],  [0,0,0]],dtype=np.float32)).cuda().unsqueeze(0).unsqueeze(0)
           self.DifLayers['Vertical'] = torch.from_numpy(np.array(  [[0,0,0] , [0,1,0],  [0,-1,0]],dtype=np.float32)).cuda().unsqueeze(0).unsqueeze(0)
           self.ROILayers['Vertical']= torch.from_numpy(np.array(   [[0,0,0] , [0,0,0],  [0,1,0]],dtype=np.float32)).cuda().unsqueeze(0).unsqueeze(0)
           self.DifLayers['Diagonal']= torch.from_numpy(np.array(   [[0,0,0] , [0,1,0],  [0,0,-1]], dtype=np.float32)).cuda().unsqueeze(0).unsqueeze(0)
           self.ROILayers['Diagonal'] = torch.from_numpy(np.array(  [[0,0,0] , [0,0,0],  [0,0,1]], dtype=np.float32)).cuda().unsqueeze(0).unsqueeze(0)

#######################################################################################################################################################
    def DiffrentialLoss(self, PredXYZ, GTXYZ,ROIMask, ConstNP=[]):
        # Calculate L1 loss using distances between pair of points in XYZ maps (where predicted XYZ map is scaled to match GT map_
        # If not given in ConstNp the relative scale is also calculated.
        ReCalculateNormalizationConst = (len( ConstNP) == 0) # ConstNP is the the relative scale between the predicted and GT XYZ model. This can be either given or be calculated within the function.
       # print(self.DifLayers['Horizontal'])
        MaxScale=np.max(ROIMask.shape[2:])# Max Scale of the dilated convolution layer (max distance between pixels in the image that will use to calculate distance between points)
        MaxScale=np.min([200,MaxScale])
        MinScale=1
        step=3
        NumLayers=int(np.ceil((MaxScale-MinScale)/step)*3*len(list(self.DifLayers))) # distance between points will be calculated on the X Y and Z axis seperately using dilated convolution with [1,-1] structure
        difPrd=torch.autograd.Variable(torch.zeros(PredXYZ.shape[0],NumLayers,PredXYZ.shape[2],PredXYZ.shape[3]).cuda(),requires_grad=False) # Will GT contain the distance between pairs of points in different distances in the X,Y,Z axis
        difGT=torch.autograd.Variable(torch.zeros(PredXYZ.shape[0],NumLayers,PredXYZ.shape[2],PredXYZ.shape[3]).cuda(),requires_grad=False) # Will Predicted contain the distance between pairs of points in different distances in the X,Y,Z axis

        i=-1
        for scale in range(1,MaxScale,3): # Go over all scales (distance between pixels)
            for l in range(3): # Go over X,Y,Z axis
                for nm in self.DifLayers: # Go over filters (relative location of points pairs, horizontal/vertical/diagonal)
                     i+=1
                     ROI = ROIMask * F.conv2d(ROIMask, self.ROILayers[nm], bias=None, stride=1, padding=scale,dilation=scale) # Check that both points are within the ROI
                     difPrd[:,i:i+1] = ROI*F.conv2d(PredXYZ[:,l:l+1,:,:], self.DifLayers[nm], bias=None, stride=1, padding=scale, dilation=scale)# Find distance between two points on the predicted XYZ model
                     difGT[:,i:i+1] = ROI*F.conv2d(GTXYZ[:,l:l+1,:,:], self.DifLayers[nm], bias=None, stride=1, padding=scale, dilation=scale) # Find distance between two points on the GT XYZ model
       # print("i=",i,"NumLayers=",NumLayers)
##============================Calculate relative scale between predicted and GT maps
        # ---------This part should NOT transmit gradient-----------------------------
        if  ReCalculateNormalizationConst: # If normalization  scale constants are not pregiven calculate them
            Rat = (difPrd/(difGT+0.00001)) #Ratios of difference  bewtween ground truth and predicted  distances between points
            Rat = F.relu(Rat) # Only postive ratios can be used when calculating loss

        #---------- Const minimize ratio with larger difference contribute more. Basically absolute sum of GT distances divided by absolute sum of predicted distances-------------------------
        # NormConst is the relative scale between GT and predicted XYZ (one number per image)
            NormConst=(torch.abs(difGT)*Rat).sum((1,2,3))/((torch.abs(difGT)*(Rat>0)).sum((1,2,3))+0.0001) # Weighted Average of Rat were the weight are the difference
            ConstNP=NormConst.data.cpu().numpy() # Convert to numpy to block gradient (yes there are more efficent ways but its one number per image so it take little time)
        print("ConstNP=",ConstNP)
        #----------------This part should transmit grdient-------------------------
        Loss=0
        for i in range(len(ConstNP)):
 #=====================Loss is the absolute difference between predicted and GT XYZ maps, where the prediction is scaled by the scale constant
          # print("ScaleDif",ScaleDif)
           if ConstNP[i]>0.0001: # If scale constant too small ignore
              Loss+=torch.abs(difGT[i]-difPrd[i]/ConstNP[i]).mean() # Calculate loss


#----------------make sure predictions will not be too large or small basically punish to small or too big scale constants ----------------------------------------------
           if ReCalculateNormalizationConst: # Check that the constant are not too large or small
               ROISum = ROIMask[i].sum()
               if ROISum>200: # Make sure ROI is not too small to create reliable statitics
                   MeanPrdDif = torch.abs(difPrd[i]).sum()/(torch.abs(difGT[i]>0).sum()) # The mean average distances between points. difGT[i]>0 term is simply the number of valid distances
                   if  MeanPrdDif>30 and ConstNP[i]>10: # Punish relative scale if it too large
                        Loss+=(MeanPrdDif-30)

                   if  MeanPrdDif<2 and ConstNP[i]<0.1: # Punish relative scale if it too small
                       Fact = 0.1 / (ConstNP[i] + 0.001)
                       print("MeanPrdDif",MeanPrdDif)
                       Loss += (0.2-MeanPrdDif)*Fact
       # Loss/=ROIMask.shape[0]
        return Loss,ConstNP # return loss and normalization scale constant

########################Find the difference between the vessel and content mask using known normalization constants=========================================









