# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:33:48 2024

@author: juan
"""

#res2dmon fordward model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
sys.path.insert(0, '../../../Source/')

import emwtinv_lib as emwt
import Groundwater2D

fid=open('real_model.mod','wt')
dy=[42.66666667,21.33333333,10.66666667,5.333333333,2.666666667,1.333333333,0.666666667,0.333333333,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667]#0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667]

ne=48#96
es=2#1
nl=0
nr=len(dy)#100#29
dh=0.025#125#0.025#125
beta=-0.5
#thk=[dh for i in np.arange(nr)]
thk=dy[::-1]#[dh*(1.02)**i for i in np.arange(nr)]
depths=np.cumsum(thk)

x=np.arange(ne*4)*es/4
z=np.concatenate([[0],depths[:-1]])
res=np.ones([ne*4,nr])

zmax=2
silt_zmin=2#1.5
silt_zmax=3#1.5
silt_xmin=34
silt_xmax=44
water_level=0.7
rho_clay=5
rho_water=50
rho_gravel0=400
rho_gravel1=400
rho0=rho_gravel0
rho1=rho_water
rho2=rho_clay
dr=(rho_gravel1-rho_gravel0)/4
rho_bk0=rho_gravel0
dz=zmax/4
z1=dz
rho_bk1=rho_gravel0+dr
z2=dz*2
rho_bk2=rho_gravel0+dr*2
z3=dz*3
rho_bk3=rho_gravel0+dr*3
z4=dz*4
rho_bk4=rho_gravel0+dr*4
#rho_bk_lst=[rho_bk0,rho_bk1,rho_bk2,rho_bk3,rho_bk4]
rho_lst=[rho_bk0,rho_bk1,rho_bk2,rho_bk3,rho_bk4,rho1,rho2]
rho_bk_lst=[rho_bk0,rho_bk1,rho_bk2,rho_bk3,rho_bk4,rho1,rho2]

rho_0=1/(1/rho1-1/rho0)
#rho_wt_lst=[np.nan,rho_0,rho_0]
rho_wt_lst=[rho_bk0,rho_bk1,rho_bk2,rho_bk3,rho_bk4,rho1,rho2]

for ix in np.arange(len(x)):
    for iy in np.arange(len(z)):
        if (z[iy]>=0)&(z[iy]<z1):
            res[ix,iy]=rho_bk_lst[0]
        if (z[iy]>=z1)&(z[iy]<z2):
            res[ix,iy]=rho_bk_lst[1]
        if (z[iy]>=z2)&(z[iy]<z3):
            res[ix,iy]=rho_bk_lst[2]
        if (z[iy]>=z3)&(z[iy]<z4):
            res[ix,iy]=rho_bk_lst[3]
        if (z[iy]>=z4):
            res[ix,iy]=rho_bk_lst[4]

for ix in np.arange(len(x)):
    for iy in np.arange(len(z)):
#        if z[iy]>=water_level:
#            res[ix,iy]=1#rho_water
#            
        if (x[ix]>=silt_xmin) & (x[ix]<silt_xmax) & (z[iy]>=silt_zmin) & (z[iy]<silt_zmax):
            res[ix,iy]=rho_bk_lst[6]#rho_clay
res_bk=res.copy()

WTfwd=Groundwater2D.water_level()
mesh=emwt.mesh()
dx=np.diff(x)
dx=np.concatenate([dx,[dx[-1]]])
mesh.dX=dx
dy=np.diff(z)
dy=np.concatenate([dy,[dy[-1]]])
mesh.dY=-dy
mesh.X=x+dx/2
mesh.Y=-z+dy/2
mesh.Xe=np.concatenate([x,[x[-1]+dx[-1]]])
mesh.Ye=np.concatenate([z,[z[-1]+dy[-1]]])
WTfwd.set_mesh(mesh)
#print(beta)
#rho_lst[int(ires)
K=-beta*(np.log(res)/0.8)**0.8 #for ires in np.reshape(res.T,[len(x)*len(z)])] 
K=np.exp(K)
water_level=-WTfwd.WTmodeling1(np.array(K),-water_level)

for ix in np.arange(len(x)):
    for iy in np.arange(len(z)):
#        res[ix,iy]=0#rho_grave
        if z[iy]>=water_level[ix]:
            res[ix,iy]=1/(1/rho_water+1/res[ix,iy])

res_lst=np.unique(np.sort(np.reshape(res,len(res[:,0])*len(res[0,:]))))
res_bk_lst=np.unique(res)

total_res=res.copy()
for ix in np.arange(len(x)):
    for iy in np.arange(len(z)):
        res[ix,iy]=np.argmin(np.abs(np.array(res_lst)-total_res[ix,iy]))

flag=True
if flag==True:

    fid.write('model\n')
    fid.write(str(ne)+'\n') #number of electrode
    fid.write(str(nl)+'\n') #pseudo levels
    fid.write(str(0)+'\n') #underwater flag
    fid.write(str(es)+'\n') #electrode spacing
    fid.write(str(2)+'\n') #user defined flag
    fid.write(str(0)+',') #offset
    fid.write(str(ne*4)+',') #number of block??
    fid.write(str(len(res_lst))+'\n') #number of resistivity values
    fid.write(str(4)+'\n') #nodes per electrode spacing
    strtmp=''
    for i in np.arange(len(res_lst)-1):
        strtmp+=str(res_lst[i])+','
    strtmp+=str(res_lst[-1])
    fid.write(strtmp+'\n')
    
#    fid.write(str(rho0)+','+str(rho1)+','+str(rho2)+'\n') #resistivitiy values
    fid.write(str(nr)+'\n') #number of rows
    for i in np.arange(nr-1):
        fid.write(str(depths[i])+',') #number of rows
    fid.write(str(depths[-1])+'\n')

    for i in np.arange(nr):
        for k in np.arange(len(res[:,])):
            fid.write(str(int(res[k,i])))
        fid.write('\n')
    fid.write('1\n')
    fid.write('0\n')
    fid.write('0\n')
    fid.write('0\n')
    fid.write('0\n')
    fid.write('0\n')
    fid.write('0\n')
    fid.write('0\n')
    fid.close()
    
    dx=[128,64,32,16,8,4,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,4,8,16,32,64,128]
    dy=[42.66666667,21.33333333,10.66666667,5.333333333,2.666666667,1.333333333,0.666666667,0.333333333,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667,0.166666667]
    x0=np.cumsum(np.concatenate([[0],dx]))+-256
    y0=np.cumsum(np.concatenate([[0],dy]))+-92
    
    #silt_z=1.5
    #silt_xmin=24
    #silt_xmax=55
    nx=len(x0)-1
    ny=len(y0)-1
    bkmodel=np.ones(nx*ny)*rho_gravel0
    k=0
    for iny in np.arange(ny):
        for inx in np.arange(nx):
            if (x0[inx]>=silt_xmin) & (x0[inx]<silt_xmax) & (y0[iny]<-silt_zmin) & (y0[iny]>-silt_zmax):
                bkmodel[k]=rho_clay
            if 0<=y0[iny]:
                bkmodel[k]=1e8
                
            k=k+1
    np.savetxt('bkmodel.mod',bkmodel)
    np.savetxt('model_bk_[0]_1000.0_100.mod',bkmodel)
    np.savetxt('model_wt_[0]_1000.0_hat__100.mod',bkmodel)

#Plots
colorzon=np.loadtxt('resis.zon')
newcolor=np.zeros([len(colorzon),3])
newcolor[:,0]=1-colorzon[:,1]/255
newcolor[:,1]=1-colorzon[:,2]/255
newcolor[:,2]=1-colorzon[:,3]/255
vmin=np.log10(colorzon[0,0])
vmax=np.log10(colorzon[-2,0])

#newcolors = []

temms = LinearSegmentedColormap.from_list('temms', newcolor)

plt.subplot(3,1,1)
#Background model
plt.title('Background Resistivity Model')
model=[ires for ires in np.reshape(res_bk.T,[len(x)*len(z)])] 
plt.pcolor(x,-z,np.reshape(np.log10(model),[len(z),len(x)]),cmap=temms,vmin=vmin,vmax=vmax)
plt.xlabel('Distance(m)')
plt.ylabel('Depth(m)')
plt.ylim([-2.5,0])

#Groundwater model
plt.subplot(3,1,2)
plt.title('Groundwater Resistivity Model')
model=[ires for ires in np.reshape(res_bk.T,[len(x)*len(z)])] 
plt.pcolor(x,-z,np.reshape(np.log10(model),[len(z),len(x)]),cmap=temms,vmin=vmin,vmax=vmax)
plt.xlabel('Distance(m)')
plt.ylabel('Depth(m)')
plt.ylim([-2.5,0])

#Total model
plt.subplot(3,1,3)
plt.title('Total Resistivity Model')
model=[ires for ires in np.reshape(total_res.T,[len(x)*len(z)])] 
plt.pcolor(x,-z,np.reshape(np.log10(model),[len(z),len(x)]),cmap=temms,vmin=vmin,vmax=vmax)
plt.xlabel('Distance(m)')
plt.ylabel('Depth(m)')
plt.ylim([-2.5,0])

plt.tight_layout()

