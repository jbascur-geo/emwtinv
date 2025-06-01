# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 20:56:28 2025

@author: juan
"""
import numpy as np
import matplotlib.pyplot as plt

def read_mesh(meshfile):

    fid=open(meshfile,'rt')
    tmp=fid.readline().split()
    Nx=int(tmp[0])
    Ny=int(tmp[1])
    tmp=fid.readline().split()
    X0=float(tmp[0])
    Y0=float(tmp[1])
    tmp=fid.readline().split()
    dX=np.zeros(Nx)
    for i in np.arange(len(tmp)):
        dX[i]=float(tmp[i])
    tmp=fid.readline().split()
    dY=np.zeros(Ny)
    for i in np.arange(len(tmp)):
        dY[i]=float(tmp[i])
    X=np.cumsum(dX)-dX/2+X0
    Y=np.cumsum(dY)-dY/2+Y0
    return X,Y,dX,dY

def get_PZwt(filename,meshfile):
    X,Y,dX,dY=read_mesh(meshfile)
    Nx=len(X)
    Ny=len(Y)
    arr=np.loadtxt(filename) 
    P = np.reshape(arr,[Ny,Nx])
    Zwt_hat = np.zeros(Nx)
    Zwt_sigma = np.zeros(Nx)
    Zwt_pos = np.zeros(Nx)
    Zwt_neg = np.zeros(Nx)
    for i in np.arange(Nx):

        Zwt_hat[i]=np.sum(P[:,i]*(Y))/np.sum(P[:,i])
        Zwt_sigma[i]=np.sum(P[:,i]*(Y)**2)/np.sum(P[:,i])

        Zwt_sigma[i]=(Zwt_sigma[i]-Zwt_hat[i]**2)**0.5
        Zwt_pos[i]=Zwt_hat[i]+Zwt_sigma[i]
        Zwt_neg[i]=Zwt_hat[i]-Zwt_sigma[i]

    P=P/np.sum(P)/np.min(abs(dY))/np.min(abs(dX))

    plt.plot(X,Zwt_hat,'k')
    plt.plot(X,Zwt_pos,'k--')
    plt.plot(X,Zwt_neg,'k--')
    return P,Zwt_hat,Zwt_sigma,Zwt_neg