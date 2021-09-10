# Calculate Chamfer distance between point clouds
import os
import numpy as np
#https://cseweb.ucsd.edu//~haosu/papers/SI2PC_arxiv_submit.pdf
def ChamferDistance(xyz1,xyz2,mask,step=40):
    # sum of minimal distance between  every point in xyz1 to  the closest point in xyz2 and vice versa only inside the mask region take one of step pixel for reducing calculation time

    AbsDst=0
    SqrDst=0
    NumPoints=0
    for x in range(0,mask.shape[1],step):
        for y in range(0,mask.shape[0],step): #
          #  print(x,y)
            if mask[y,x]>0: #Go over all pixels in the map find minimal distance
                  NumPoints+=1
                  p1=xyz1[y,x]
                  dst=np.zeros(xyz2.shape)
                  for i in range(3): dst[:,:,i]=(mask*(xyz2[:,:,i]-p1[i]))**2
                  dst=dst.sum(2)
                  dst[dst==0] = dst.max()
                  mindist = dst.min()
                  AbsDst += np.sqrt(mindist)
                  SqrDst += mindist

                  p1 = xyz2[y, x]
                  dst = np.zeros(xyz1.shape)
                  for i in range(3): dst[:, :, i] = (mask * (xyz1[:, :, i] - p1[i])) ** 2
                  dst = dst.sum(2)
                  dst[dst == 0] = dst.max()
                  mindist = dst.min()
                  AbsDst += np.sqrt(mindist)
                  SqrDst += mindist
    return AbsDst/(NumPoints+0.0001),SqrDst/(NumPoints+0.0001)



