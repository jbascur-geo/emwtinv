# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:41:37 2024

@author: juan
"""

import utils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy import interpolate
import sys 
sys.path.insert(0, '../../Source/')

import emwtinv_lib as emwt
import emwtinv_setup 
import multi_prop_dc2d as prop

background_model_dir='\\Background models'
groundwater_model_dir='\\Expected Groundwater Model'
PDFs_dir='\\Posterior PDFs'
iteration_dir='\\Iterations'

def paramplots(path,iters,estep_samples=2000):
    
    def calc_porosity(pdfs,path):
        logrho=pdfs.param_pdf[0][0]
        likerho=pdfs.param_pdf[1][0]#/np.sum(pdfs.param_pdf[1][0])
        logrho_fine=np.linspace(logrho[0],logrho[-1],501)
        PP=interpolate.make_smoothing_spline(logrho, likerho, lam=1e-3)
        AAA=PP(logrho_fine)
        AAA[AAA<0]=0
        #plt.plot(logrho,likerho,'ok')
        likerho=AAA
        logrho=logrho_fine
        #plt.plot(logrho,likerho,'--r')
        p=prop.param_proposal([emwt.model()],path+'/sigmawt_setup.txt') 
        likephi_,Phi,Phi_e=p.get_conditional_like([np.exp(np.array(logrho))])

        likephi=np.zeros(len(Phi))
        
        for i in np.arange(len(Phi)):#logrho)):
            #*likerho
            likephi[i]=np.sum(likerho*likephi_[i,:])/np.sum(likerho)
            
        return likephi,Phi
    
    
    pdf=utils.parampdf()
    file_path0=path+PDFs_dir+'\\model_wt__'+str(estep_samples)+'.0_parpdfs_'+str(iters)+'.txt'
    file_path=path+PDFs_dir+'\\model_wt__'+str(estep_samples)+'.0_parpdfs_'+str(iters)+'_ed.txt'
    fid=open(file_path0,'rt')
    lines=fid.readlines()
    lines=[lines[6],lines[7],lines[4],lines[5]]
    fid_out=open(file_path,'wt')
    for line in lines:
        fid_out.write(line)
    fid_out.close()

    pdf.read(file_path)
    likephi,Phi=calc_porosity(pdf,path)
    pdf.param_pdf[0].append(Phi)
    pdf.param_pdf[1].append(likephi)
    
    pdf.param_name[0]='log10 Resistivity[Ohm-m]'
    pdf.param_pdf[0][0] = np.concatenate([[-0.5],np.log10(np.exp(pdf.param_pdf[0][0]))])
    pdf.param_pdf[1][0] = np.concatenate([[0],pdf.param_pdf[1][0]])
    
    pdf.param_name[0]='log10 Resistivity[Ohm-m]'
    pdf.param_name[1]='Beta'
    pdf.param_name[2]='Porosity'
    plt.figure(figsize=[10,3])
    pdf.plots(color='red',linewidth=2)
        
    plt.subplot(1,3,1)
    plt.xlim([-0.5,4])
    
    plt.subplot(1,3,2)
    plt.ylim([0,6])

    plt.subplot(1,3,3)
    plt.tight_layout()    
    return pdf
    
