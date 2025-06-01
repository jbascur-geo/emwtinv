# -*- coding: utf-8 -*-
"""
Created on Tue May 14 03:27:14 2024

@author: juan
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
#sys.path.append('../../../Source/')
sys.path.insert(0, '../Source/')
import emwtinv_lib as em
import scipy
import glob 
import matplotlib.pyplot as plt

background_model_dir='\\Background models'
groundwater_model_dir='\\Expected Groundwater Model'
PDFs_dir='\\Posterior PDFs'
iteration_dir='\\Iterations'

def plot_models(path,iters,estep_samples=2000):
    
    bk_model_file=path+background_model_dir+'\\model_bk_[0]_'+str(estep_samples)+'.0_'+str(iters)+'.mod'
    wt_model_file=path+groundwater_model_dir+'\\model_wt_[0]_'+str(estep_samples)+'.0_hat__'+str(iters)+'.mod'
    mesh_file=path+'\\emwtinv.0.msh'
    bk_model=em.model()
    bk_model.read(bk_model_file, mesh_file)
    plt.subplot(3,1,1)

    logs=load_logs(path+iteration_dir)
    if len(logs[0])>3:
        plt.title(
            r'iter '+str(logs[iters][0])+' '
            r'$P(\sigma_{bk} \mid d)=-$'+'{:.2f}'.format(logs[iters][1])+' '+
            r'$P(\sigma_{wt} \mid d,\sigma_{bk})=-$'+'{:.2f}'.format(logs[iters][2])+' '
            r'$P(\sigma_{bk},\widehat{\sigma_{wt}} \mid d)=-$'+'{:.2f}'.format(logs[iters][3]))
    else:
        plt.title(
            r'iter '+str(logs[iters][0])+' '
            r'$P(\sigma_{bk} \mid d)=-$'+'{:.2f}'.format(logs[iters][1])+' '+
            r'$P(\sigma_{wt} \mid d,\sigma_{bk})=-$'+'{:.2f}'.format(logs[iters][2])+' ')

    bk_model.plot2D(log10=True,cmap='jet',xlim=[0,96],ylim=[-2.77,1],vlim=[np.log10(10),np.log10(1000)]); #plt.gca().yaxis.set_inverted(True)
    wt_model=em.model()
    plt.subplot(3,1,2)
    wt_model.read(wt_model_file, mesh_file)
    wt_model.plot2D(log10=True,cmap='jet',xlim=[0,96],ylim=[-2.77,1],vlim=[np.log10(10),np.log10(1000)]); #plt.gca().yaxis.set_inverted(True)
    plt.subplot(3,1,3)
    total_model=wt_model.model_sum(wt_model,bk_model)
    total_model.plot2D(log10=True,cmap='jet',xlim=[0,96],ylim=[-2.77,1],vlim=[np.log10(10),np.log10(1000)]); #plt.gca().yaxis.set_inverted(True)
    return bk_model,wt_model,total_model

def plot_density_models(path,iters,estep_samples=2000):
    ns=estep_samples
#    iters=
#    path=path
    wt_model_file=path+PDFs_dir+'\\model_wt__'+str(estep_samples)+'.0_Zwtmap_'+str(iters)+'.0mod'
    # model_wt__1000.0_Zbsmap_0.0mod
    mesh_file=path+'\\emwtinv.0.msh'
    wt_model=em.model()
    plt.subplot(3,1,2)
    wt_model.read(wt_model_file, mesh_file)
    wt_model.plot2D(cmap='jet')#,xlim=[0,100],ylim=[-2.77,0])#,vlim=[np.log10(10),np.log10(600)]); #plt.gca().yaxis.set_inverted(True)
    return wt_model

def save_model_xyz(file,model):    
    dat = model.get_linear_vals()
    Ny=len(model.mesh.dY) 
    Nx=len(model.mesh.dX) 
    R=model.get_linear_vals()
    Rmap = np.reshape(R, [Ny, Nx])
    
    fid = open(file,'wt')
    for i in np.arange(Nx):
        for j in np.arange(Ny):
            fid.write(str(model.mesh.X[i])+','+str(model.mesh.Y[j])+','+str(Rmap[j,i])+'\n')
    fid.close()
    
def load_logs(path):
    files=glob.glob(path+'/logfile*.*')
    postpdf=np.zeros([len(files),3])
    for i in np.arange(len(files)):
#        A=np.loadtxt(files[i])[-1,:]
        C=open(files[i],'rt')
        B=C.readlines()
        A=B[-1].split()
        n=int(files[i].split('logfile')[-1][:-4])
        postpdf[n,0]=float(A[0])
        postpdf[n,2]=float(A[3])
        postpdf[n,1]=float(A[2])
    return postpdf

def get_rms(model,em):
    ind_active=model.get_linear_vals()<1e7
    em.e_step.data[0].method.set_mesh(model.mesh,ind_active)
    em.e_step.data[0].method.set_model(model.get_linear_vals())
    data=em.e_step.data[0].method.data
    resp=em.e_step.data[0].method.fwd()
    print('RMS(%):',100*np.mean( (data-resp)**2/data**2 )**0.5)
    return resp