# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 22:16:46 2021

@author: juan
"""
import discretize
from pymatsolver import SolverLU,Solver
import numpy as np
import matplotlib.pyplot as plt
import scipy



class water_level:
# Class to determine the water table in an unconfined aquifer using the saturated groundwater flow equation.

    def __init__(self):
        None

    def set_mesh(self,meshin):
        mesh = discretize.TensorMesh([meshin.dX, meshin.dY], origin=[meshin.X0,meshin.Y0])
        self.mesh=mesh        

    def plot_stream(self,Phi):
        mesh=self.mesh
        x0=mesh.cell_centers[:,0]
        y0=mesh.cell_centers[:,1]
        dx=mesh.average_face_x_to_cell*(mesh.cell_gradient_x*Phi)
        dy=mesh.average_face_y_to_cell*(mesh.cell_gradient_y*Phi)
        plt.streamplot(x0,y0,dx,dy)

    def streamline(self,h):
        mesh=self.mesh
        head=self.head
        x0=mesh.cell_centers[:,0]
        y0=mesh.cell_centers[:,1]
        maxN=10*len(mesh.cell_centers_x)
        xx0=0
        dh=np.min(mesh.h[0])
        dhh=dh/2
        Nindx=np.arange(len(mesh.cell_centers_x))[dh==mesh.h[0]]
        indx=mesh.cell_centers_x[dh==mesh.h[0]]
        indy=mesh.cell_centers_y
        Nini=len(mesh.cell_centers_x)
        Xini=indx[0]
        NXini=Nindx[0]
        NXfin=Nindx[-1]
        Xfin=indx[-1]
        Yini=indy[0]
        Yfin=indy[-1]

        dx=mesh.average_face_x_to_cell*(mesh.cell_gradient_x*head)
        dy=mesh.average_face_y_to_cell*(mesh.cell_gradient_y*head)
        V=(dx**2+dy**2)**0.5+1e-190  #Avoiding divide by zero
        dx=-dx/V
        dy=-dy/V

        stream=np.zeros([maxN+Nini,2])
        k=0
        
        #water level is h before the geophysical data 
        for i in np.arange(Nini):
            stream[i,0]=mesh.cell_centers_x[i]
            stream[i,1]=h
            k=k+1
            if stream[i,0]>=xx0:
                break

        #water level modeling using K and the water head 
        for i in np.arange(0,maxN):
            n = discretize.utils.closestPoints(mesh, np.c_[stream[k-1,0],stream[k-1,1]])[0]                
            stream[k,0]=stream[k-1,0]+dx[n]*dhh
            stream[k,1]=stream[k-1,1]+dy[n]*dhh
            if(stream[k,0]>Xfin)or(stream[k,0]<=Xini):
                break
            if(stream[k,1]<=Yini)or(stream[k,1]>=Yfin):
                break            
            k=k+1

        #water level is stream[k-1,1] after the geophysical data 
        for i in np.arange(NXfin,Nini):
            stream[k,0]=mesh.cell_centers_x[i]
            stream[k,1]=stream[k-1,1]
            k=k+1
        
        f = scipy.interpolate.interp1d(stream[0:k,0],stream[0:k,1])
        wt=f(mesh.cell_centers_x)
            
        return wt



    def streamline2(self,h):
        mesh=self.mesh
        head=self.head
        x0=mesh.cell_centers[:,0]
        y0=mesh.cell_centers[:,1]
        maxN=10*len(mesh.cell_centers_x)
        xx0=0
        Nx=len(mesh.cell_centers_x)
        Ny=len(mesh.cell_centers_y)
        nodeGradx=np.reshape(mesh.cell_gradient_x*head,[Ny,Nx+1])
        nodeGradx[:,0]=nodeGradx[:,2]
        nodeGradx[:,1]=nodeGradx[:,2]
        nodeGradx[:,Nx-1]=nodeGradx[:,Nx-2]
        nodeGradx[:,Nx-1]=nodeGradx[:,Nx-2]
        nodeGradx=np.reshape(nodeGradx,Ny*(Nx+1))

        nodeGrady=np.reshape(mesh.cell_gradient_y*head,[Ny+1,Nx])
        nodeGrady=np.reshape(nodeGrady,(Ny+1)*Nx)

    
        dx=mesh.average_face_x_to_cell*nodeGradx#(mesh.cell_gradient_x*head)
        dy=mesh.average_face_y_to_cell*nodeGrady#(mesh.cell_gradient_y*head)
        V=(dx**2+dy**2)**0.5+1e-190  #Avoiding divide by zero
        dx=-dx/V
        dy=-dy/V
        hx=(mesh.h[0][:-1]+mesh.h[0][1:])/2
        hy=(mesh.h[1][:-1]+mesh.h[1][1:])/2
        X=(mesh.cell_centers_x)
        Y=(mesh.cell_centers_y)
        kk=5
        stream=np.zeros([Nx*kk*2,2])
        k=0
        
        V2D=np.reshape(V,[Ny,Nx])
        dx2D=np.reshape(dx,[Ny,Nx])
        dy2D=np.reshape(dy,[Ny,Nx])
        nx0=int(len(X)/2)-1
        ny0=np.argmin(np.abs(h-Y))
        xx=X[nx0]
        yy=Y[ny0]
        s=np.zeros(8)
        d=np.zeros(8)
        breakt=False
        
        dx=1
        dy=0
        
        for i in np.arange(0,len(stream[:,0])-1):

            if nx0<=0:
                x1=X[0]-hx[0]
                x0=X[0]
                nx0=0
            else:
                x1=X[nx0-1]                
                x0=X[nx0]

            if nx0>=Nx-1:
                x2=X[Nx-1] #+ hx[Nx-1]               
                nx0=Nx-1
            else:
                x2=X[nx0+1]
            
            if ny0<=0:
                y1=Y[0]-hy[0]
                y0=Y[0]
                ny0=0
            else:
                y1=Y[ny0-1]
                y0=Y[ny0]
            
            if ny0>=Ny-1:
                y2=Y[Ny-1] #+ hy[Ny-1]
                ny0=Ny-1
            else:
                y2=Y[ny0+1]

            d[0]=((x2-xx)**2+(y1-yy)**2)**0.5
            d[1]=((x2-xx)**2+(y0-yy)**2)**0.5
            d[2]=((x2-xx)**2+(y2-yy)**2)**0.5
            d[3]=((x0-xx)**2+(y2-yy)**2)**0.5
            d[4]=((x1-xx)**2+(y2-yy)**2)**0.5
            d[5]=((x1-xx)**2+(y0-yy)**2)**0.5
            d[6]=((x1-xx)**2+(y1-yy)**2)**0.5
            d[7]=((x0-xx)**2+(y1-yy)**2)**0.5

            dd=np.mean(d)/kk

            if(nx0<0):
                nx0=0
                breakt=True
            if(nx0>=Nx-1):
                nx0=Nx-1
                breakt=True
            if(ny0<0):
                ny0=0
                breakt=True
            if(ny0>=Ny-1):                
                ny0=Ny-1
                breakt=True
                
            stream[k,0]=xx+dx*dd
            stream[k,1]=yy+dy*dd
            xx=stream[k,0]
            yy=stream[k,1]
            nx0=np.argmin(np.abs(X-xx))
            ny0=np.argmin(np.abs(Y-yy))
            k=k+1
            dx=dx2D[ny0,nx0]
            dy=dy2D[ny0,nx0]
                        
            if xx < X[0]:
                break

            if yy > X[-1]:
                break

            if breakt:
                break

            if k >= len(stream)-1:
                break
            
        nx0=int(len(X)/2)-1
        ny0=np.argmin(np.abs(h-Y))
        xx=X[nx0]
        yy=Y[ny0]
        nx0=np.argmin(np.abs(X-xx))
        ny0=np.argmin(np.abs(Y-yy))
        stream2=stream[0:k,:].copy()
        k=k+1
        dx=dx2D[ny0,nx0]
        dy=dy2D[ny0,nx0]

        kk=5
        stream=np.zeros([Nx*kk*2,2])
        k=0
        breakt=False

        for i in np.arange(0,len(stream[:,0])-1):

            if nx0<=0:
                x1=X[0]-hx[0]
            else:
                x1=X[nx0-1]                
            x0=X[nx0]
            if nx0>=Nx-1:
                x2=X[Nx-1] #+ hx[Nx-1]               
            else:
                x2=X[nx0+1]
            
            if nx0<=0:
                y1=Y[0]-hy[0]
            else:
                y1=Y[ny0-1]
            y0=Y[ny0]
            if ny0>=Ny-1:
                y2=Y[Ny-1] #+ hy[Ny-1]
            else:
                y2=Y[ny0+1]

            d[0]=((x2-xx)**2+(y1-yy)**2)**0.5
            d[1]=((x2-xx)**2+(y0-yy)**2)**0.5
            d[2]=((x2-xx)**2+(y2-yy)**2)**0.5
            d[3]=((x0-xx)**2+(y2-yy)**2)**0.5
            d[4]=((x1-xx)**2+(y2-yy)**2)**0.5
            d[5]=((x1-xx)**2+(y0-yy)**2)**0.5
            d[6]=((x1-xx)**2+(y1-yy)**2)**0.5
            d[7]=((x0-xx)**2+(y1-yy)**2)**0.5
            dd=np.mean(d)/kk

            if(nx0<=0):
                nx0=0
                breakt=True
            if(nx0>Nx-1):
                nx0=Nx-1
                breakt=True
            if(ny0<=0):
                ny0=0
                breakt=True
            if(ny0>Ny-1):                
                ny0=Ny-1
                breakt=True
                
            stream[k,0]=xx-dx*dd
            stream[k,1]=yy-dy*dd
            xx=stream[k,0]
            yy=stream[k,1]

            nx0=np.argmin(np.abs(X-xx))
            ny0=np.argmin(np.abs(Y-yy))
            k=k+1
            dx=dx2D[ny0,nx0]
            dy=dy2D[ny0,nx0]
            
            if xx < X[0]:
                break

            if yy > X[-1]:
                break

            if breakt:
                break
            if k > len(stream)-1:
                break

        if stream[-1,0]<X[-1]:
            stream[-1,0]=X[-1]

        #water level is stream[k-1,1] after the geophysical data 
        stream=stream[0:k,:]
        mm=np.concatenate([stream[::-1,:],stream2])
        f = scipy.interpolate.interp1d(mm[:,0],mm[:,1],fill_value='extrapolate')
        wt=f(mesh.cell_centers_x)            
        return wt

    def fields(self,K):
        mesh=self.mesh
        G=mesh.cell_gradient
        MsigI = mesh.get_face_inner_product(K, invert_model=True, invert_matrix=True)
        A = G.T * MsigI * G 
                
        nx_left =mesh.cell_centers[:,0]==mesh.cell_centers_x[1]
        nx_right=mesh.cell_centers[:,0]==mesh.cell_centers_x[-1]
        nvar = ~(nx_left | nx_right)
        
        x=np.zeros(mesh.nC)
        x[nx_left]=1
        
        subA=A[nvar,:][:,nvar]
        subB=-(A*x)[nvar]
        if np.sum(A.diagonal()==0)>0:
            return np.nan
        else:
            subAinv=Solver(subA)
            x[nvar]=subAinv*subB
        self.head=x
        return x
    
    def WTmodeling(self,K,h1,h2):

        if np.isfinite(np.sum(K)):
            if np.isfinite(np.sum(self.fields(K))):
                a=self.streamline2(h1)
                b=self.streamline2(h2)
            else:
                a=np.zeros(len(self.mesh.cell_centers_x))
                b=np.zeros(len(self.mesh.cell_centers_x))
        else:
            a=np.zeros(len(self.mesh.cell_centers_x))
            b=np.zeros(len(self.mesh.cell_centers_x))
        return a,b

    def WTmodeling1(self,K,h1):

        if np.isfinite(np.sum(K)):
            if np.isfinite(np.sum(self.fields(K))):
                a=self.streamline2(h1)
            else:
                a=np.zeros(len(self.mesh.cell_centers_x))
        else:
            a=np.zeros(len(self.mesh.cell_centers_x))
        return a
        