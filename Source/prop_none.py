# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 10:29:37 2022

@author: juan
"""

import numpy as np
import scipy.stats as stats 

import sigma_prob_basic as sm
import Groundwater2D 
import matplotlib.pyplot as plt

class param_proposal:
# This class provides a transparent E-Step without groundwater model proposals generation. 
# Useful when HBI is applied to multiple geophysical datasets and one dataset is used solely 
# for background model inversion, not for estimating the groundwater model.


    # Groundwater model parameters:
    # Zmin: Minimum water table depth
    # Zmax: Maximum water table depth
    # log_sigma_min: Minimum electrical conductivity of the groundwater aquifer
    # log_sigma_max: Maximum electrical conductivity of the groundwater aquifer
    # beta_min: Minimum beta parameter of sigma-K realationship function (beta:0 constant K)
    # beta_max: Maximum beta parameter of sigma-K realationship function (beta:0 constant K)
    Zmin=0
    Zmax=200
    dZ=1
    log_sigma_min=np.log(0.1)
    log_sigma_max=np.log(1000)
    beta_min=0
    beta_max=3    

    #=============================================================================
    #Archie pdf parameters.
    #=============================================================================
    m1=1.5 #minimum cementation  factor
    m2=2.5 #maximum cementation  factor
    logA1=np.log(0.1) #minimum log-fluid resistivity (including tortuosity factor)
    logA2=np.log(30) #maximum log-fluid resistivity (including tortuosity factor)

    por_array=np.linspace(0.01,0.99,21)#0.01,0.99,10)#np.array([0.01,0.99])#np.linspace(0.01,0.99,50)
    last_params=np.array([Zmax/3,2*Zmax/3,0.5])
    last_log_likehood=-999

    sigma0_m=np.log(100)        
    sigma0_var=1        
    
    def read_setup(self,setup_file,text=True):
        fid=open(setup_file,'rt')
        lines=fid.readlines()
        k=0
        self.Zmin=float(lines[k].split(' ')[-1].split('\n')[0])
        k=k+1
        self.Zmax=float(lines[k].split(' ')[-1].split('\n')[0])
        k=k+1
        self.dZ=float(lines[k].split(' ')[-1].split('\n')[0])
        k=k+1
        self.log_sigma_min=float(lines[k].split(' ')[-1].split('\n')[0])
        k=k+1
        self.log_sigma_max=float(lines[k].split(' ')[-1].split('\n')[0])
        k=k+1
        self.beta_min=float(lines[k].split(' ')[-1].split('\n')[0])
        k=k+1
        self.beta_max=float(lines[k].split(' ')[-1].split('\n')[0])
        k=k+1
        self.por_min=float(lines[k].split(' ')[-1].split('\n')[0])
        k=k+1
        self.por_max=float(lines[k].split(' ')[-1].split('\n')[0])
        k=k+1
        self.por_array=np.linspace(self.por_min,self.por_max,21)

        self.m1=float(lines[k].split(' ')[-1].split('\n')[0])
        k=k+1
        self.m2=float(lines[k].split(' ')[-1].split('\n')[0])
        k=k+1
        self.logA1=float(lines[k].split(' ')[-1].split('\n')[0])
        k=k+1
        self.logA2=float(lines[k].split(' ')[-1].split('\n')[0])
        k=k+1
        if len(lines[k].split(' ')[1].split('\n')[0])>0:
            self.sigma0_m=float(lines[k].split(' ')[1].split('\n')[0])
        else:
            self.sigma0_m=[]
            
        k=k+1
        if len(lines[k].split(' ')[1].split('\n')[0])>0:
            self.sigma0_var=float(lines[k].split(' ')[1].split('\n')[0])
        else:
            self.sigma0_var=[]
        k=k+1
        if len(lines[k].split(' ')[1].split('\n')[0])>0:
            self.magiclmd=float(lines[k].split(' ')[1].split('\n')[0])
        else:
            self.magiclmd=[]


            
        if text == True:
 
            print('-----------------------------------------------------')
            print('Conditional Petrophysical and Hydrological Parameters')
            print('-----------------------------------------------------')            
            print('Zmin: ' + str(self.Zmin))#=float(lines[k].split(' ')[-1].split('\n')[0])
            print('Zmax: ' + str(self.Zmax))#=float(lines[k].split(' ')[-1].split('\n')[0])
            print('ZdZ: ' + str(self.dZ))#=float(lines[k].split(' ')[-1].split('\n')[0])
            print('log_sigma_min: ' + str(self.log_sigma_min))#=float(lines[k].split(' ')[-1].split('\n')[0])
            print('log_sigma_max: ' + str(self.log_sigma_max))#=float(lines[k].split(' ')[-1].split('\n')[0])
            print('beta_min: ' + str(self.beta_min))#=float(lines[k].split(' ')[-1].split('\n')[0])
            print('beta_max: ' + str(self.beta_max))#=float(lines[k].split(' ')[-1].split('\n')[0])
            print('por_min: ' + str(self.por_min))#=float(lines[k].split(' ')[-1].split('\n')[0])
            print('por_max: ' + str(self.por_max))#=float(lines[k].split(' ')[-1].split('\n')[0])
            print('m1: ' + str(self.m1))#=float(lines[k].split(' ')[-1].split('\n')[0])
            print('m2: ' + str(self.m2))#=float(lines[k].split(' ')[-1].split('\n')[0])
            print('logA1: ' + str(self.logA1))#=float(lines[k].split(' ')[-1].split('\n')[0])
            print('logA2: ' + str(self.logA2))#=float(lines[k].split(' ')[-1].split('\n')[0])
            print('sigma0_m: ' + str(self.sigma0_m))#=float(lines[k].split(' ')[1].split('\n')[0])
            print('sigma0_var: ' + str(self.sigma0_var))#=float(lines[k].split(' ')[1].split('\n')[0])
            print('magic_lmd: ' + str(self.magiclmd))#=float(lines[k].split(' ')[1].split('\n')[0])
            print('-----------------------------------------------------')            
            
            

    def __init__(self,model,setup_file,seed=0):
        self.read_setup(setup_file)
        self.proposal_pdf_list=[]
        self.model=model
        #Calculate possible sets for Zwt & Zbs 
        Z=[]
        Nz=int((np.ceil(self.Zmax-self.Zmin)/self.dZ))+1
        for i in np.arange(Nz):
            for j in np.arange(Nz):
                if j>=i:
                    Z.append([self.dZ*i+self.Zmin,self.dZ*j+self.Zmin])
        self.Nz=len(Z)
        self.Z=Z

        np.random.seed(seed)
        
        #Pdf list of proposed parameters
        self.proposal_pdf_list.append(stats.triang(c=0,loc=self.Zmin,scale=self.Zmax-self.Zmin))#stats.uniform(0,self.Nz)) #Z_bk
        self.proposal_pdf_list.append(stats.uniform(0,1))

        #Target pdf list for sampled parameters 
        self.target_pdf_list=[]                
        if self.sigma0_m:
            self.target_pdf_list.append(sm.SigmaProbMarginal(self.m1,self.m2,self.logA1,self.logA2,self.por_array,pordist='uniform',sigmapdf=[self.sigma0_m,self.sigma0_var]))
        else:
            self.target_pdf_list.append(sm.SigmaProbMarginal(self.m1,self.m2,self.logA1,self.logA2,self.por_array,pordist='uniform',sigmapdf=[],sigma_range=[self.log_sigma_min,self.log_sigma_max]))
        
        self.last_log_likehood=self.target_pdf(self.last_params)

    def target_pdf(self,params):
        target=self.target_pdf_list[0].logpdf(params[-1])
        return target
    
    def get_paramrange(self):
        minpar=[self.Zmin,self.Zmin,self.log_sigma_min]
        maxpar=[self.Zmax,self.Zmax,self.log_sigma_max]
        Npar=[20,20,20]
        return minpar,maxpar,Npar

    def get_many_param_proposal(self,N):
        prop=[]
        for i in np.arange(N):
            prop.append(self.get_param_proposal())
        return np.array(prop)            

    def get_param_proposal(self):
        N=3
        params=np.zeros(N)
        params[0]=self.proposal_pdf_list[0].rvs()
        params[1]=self.proposal_pdf_list[1].rvs()*(self.Zmax-params[0])+params[0]
        params[2]=self.target_pdf_list[-1].random()
        logpdf=self.target_pdf(params)

        self.last_params=params
        return self.last_params

    def param2model(self,model_lst,invmodel_ref_lst,param):
        return model_lst

    def get_conditional_like(self,Rlist):
        R=Rlist[0]
        Phi_e=self.por_array
        Phi=(Phi_e[:-1]+Phi_e[1:])/2
        phiprob=sm.PhiProb(self.m1, self.m2, self.logA1, self.logA2, R[0])        
        likephi=np.zeros([len(Phi),len(R)])
        for ir in np.arange(len(R)):        
            phiprob.setparams([self.m1, self.m2, self.logA1, self.logA2, np.log(R[ir])])
            for ip in np.arange(len(Phi)):        
                likephi[ip,ir]=np.exp(phiprob.logpdf(np.log(Phi[ip])))
        return likephi,Phi,Phi_e


    def prior_likehood(self,param,model_lst,modelinv_ref):
        model=model_lst[0]
        Zwt0=param[0]
        Zbs0=param[1]
        pdf=self.get_multipdf(model)
        Zm=(model.mesh.Y)        
        pi=pdf(np.mean(model.get_log_vals()[(Zwt0<=Zm)&(Zbs0>=Zm)]))
        return pi
    
    def get_invmodelref(self,model):
        return self.param2model(model,model,[76,100,3])
        

    def get_multipdf(self,model,n_layers=2):
        
        vals=model.get_log_vals()
        
        #initiazing Gaussian mixture parameters
        m0=np.min(vals)
        m1=np.max(vals)
        m=np.array([(m1-m0)/(n_layers-1)*i+m0 for i in np.arange(n_layers)])
        sigma2=np.array([np.abs(m1-m0)/n_layers/20 for i in np.arange(n_layers)])**2
        pi=np.array([0.1,0.1])
    
        for iter_ in np.arange(3):
    
            p=[(vals-m[i])**2/sigma2[i] for i in np.arange(len(m))]
            cf=np.sum([p[i][:]*pi[i] for i in np.arange(len(m))])
            p=np.array(p)
            n=np.argmin(p,axis=0)
            pi=[np.sum(n==i) for i in np.arange(len(m))]
            pi = np.array(pi)/len(vals) 
            m=np.array([np.mean(vals[n==i]) for i in np.arange(len(m))])
            sigma2=np.array([np.var(vals[n==i]) for i in np.arange(len(m))])
       
        nmin= np.argmin(m)
        if np.isnan(m[nmin]):
            def f(x):
                return 0
            pdf = f
        else:
            pdf = stats.norm(loc=m[nmin],scale=sigma2[nmin]**0.5).logpdf
        return pdf
            