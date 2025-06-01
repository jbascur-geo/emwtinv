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
# Class for generating groundwater resistivity model proposals based on unconfined aquifer modeling.
# Uses saturated groundwater flow modeling and employs conditional probability distributions as petrophysical relationships,
# including a poorly calibrated Archie’s law and an uncalibrated CR–k (CK) relationship.

    # Groundwater model parameters:
    # Zmin: Minimum water table depth
    # Zmax: Maximum water table depth
    # log_sigma_min: Minimum electrical conductivity of the groundwater aquifer
    # log_sigma_max: Maximum electrical conductivity of the groundwater aquifer
    # beta_min: Minimum beta parameter of sigma-K realationship function (beta:0 constant K)
    # beta_max: Maximum beta parameter of sigma-K realationship function (beta:0 constant K)
    Zmin=-20
    Zmax=0
    dZ=1
    log_sigma_min=np.log(0.1)
    log_sigma_max=np.log(1000)
    beta_min=0
    beta_max=3    
    logsum=False#True
    curve=False
    #=============================================================================
    #Archie pdf parameters.
    #=============================================================================
    m1=1.5 #minimum cementation  factor
    m2=2.5 #maximum cementation  factor
    logA1=np.log(0.1) #minimum log-fluid resistivity (including tortuosity factor)
    logA2=np.log(30) #maximum log-fluid resistivity (including tortuosity factor)

    por_array=np.linspace(0,1,21)#0.01,0.99,10)#np.array([0.01,0.99])#np.linspace(0.01,0.99,50)
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

        text=False            
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


    def __init__(self,model_lst,setup_file,seed=0,params_hat=None):
        model=model_lst[0]
        
        self.read_setup(setup_file)
        self.proposal_pdf_list=[]
        self.model=model

        #Calculate possible sets for Zwt & Zbs 
        Z=[]
        Nz=int((self.Zmax-self.Zmin)/self.dZ)
        for i in np.arange(Nz):
             Z.append([self.Zmin+self.dZ*i,self.Zmax])#self.Zmin+self.dZ*j])
        self.Nz=len(Z)
        self.Z=Z

        np.random.seed(seed)

        #Pdf list of proposed parameters
        self.proposal_pdf_list.append(stats.uniform(0,self.Nz)) #Z_bk

        #Target pdf list for sampled parameters 
        self.target_pdf_list=[]                
        np.random.seed(seed)
        self.target_pdf_list.append(stats.uniform(self.beta_min,self.beta_max-self.beta_min))        

        if self.sigma0_m:
            self.target_pdf_list.append(sm.SigmaProbMarginal(self.m1,self.m2,self.logA1,self.logA2,self.por_array,pordist='uniform',sigmapdf=[self.sigma0_m,self.sigma0_var]))
        else:
            self.target_pdf_list.append(sm.SigmaProbMarginal(self.m1,self.m2,self.logA1,self.logA2,self.por_array,pordist='uniform',sigmapdf=[]))
        
        self.last_log_likehood=self.target_pdf(self.last_params)

        #Grounwater level modelling
        WTfwd=Groundwater2D.water_level()
        WTfwd.set_mesh(model.mesh)
        self.WTfwd=WTfwd
        self.WTmodeling=WTfwd.WTmodeling1

        #Range of Bulk resistivity        
        self.R=[]
        self.R.append(np.logspace(self.log_sigma_min,self.log_sigma_max,51))

        #model_wt maps
        self.modelwtmap=[]
        self.Zwt_map=[]
        self.Zbs_map=[]
        n=len(model_lst[0].get_linear_vals())
        ni=len(self.R[0])-1
        self.modelwtmap.append(np.zeros([n,ni]))
        self.Zwt_map.append(np.zeros([n]))
        self.Zbs_map.append(np.zeros([n]))


    def target_pdf(self,params):
#        print(params)
        target=self.target_pdf_list[0].logpdf(params[-1])
        return target
    
    def get_paramrange(self):
        minpar=[self.Zmin,self.Zmin,self.beta_min,self.log_sigma_min]#,self.beta_min]
        maxpar=[self.Zmax,self.Zmax,self.beta_max,self.log_sigma_max]#,self.beta_max]
        Npar=[20,20,10,20]#[10,10,10]#,10]
        return minpar,maxpar,Npar

    def get_many_param_proposal(self,N):
        prop=[]
        for i in np.arange(N):
            prop.append(self.get_param_proposal())
        return np.array(prop)            


    def get_param_proposal(self,modelbk_lst):
#        N=3#len(self.proposal_pdf_list)
#        params=np.zeros(N)
#        Zindex=int(self.proposal_pdf_list[0].rvs())
#        params[0]=self.Z[Zindex][0]
#        params[1]=self.Z[Zindex][1]
#        params[2]=self.target_pdf_list[-1].random()
#        logpdf=self.target_pdf(params)
        N=4#len(self.proposal_pdf_list)
        params=np.zeros(N)
        Zindex=int(self.proposal_pdf_list[0].rvs())
        params[0]=self.Z[Zindex][0]
#        print('proposal',params[0])
        params[1]=np.min((modelbk_lst[0].mesh.Y))#self.Z[Zindex][1]
        params[2]=self.target_pdf_list[-2].rvs()
        params[3]=self.target_pdf_list[-1].random()
#        print(params)
        logpdf=self.target_pdf(params)

        self.last_log_likehood=logpdf           
        self.last_params=params
                    
        
        return self.last_params

#    def param2model(self,model,param):
#        mod=model.copy_model()
#        Zwt=param[0]
#        Zbs=param[1]
#        alfa=0.5
#        k=0
#        for j in np.arange(len(mod.mesh.Y)):
#            for i in np.arange(len(mod.mesh.X)):                
#                if((Zwt<=mod.mesh.Ye[j])&(Zbs>=mod.mesh.Ye[j+1])):
#                    mod.vals[k]=param[-1]
#                else:
#                    mod.vals[k]=13#np.inf
#                       
#                k+=1
#        return mod
#        mod=model.copy_model()
#        Zwt0=param[0]
#        Zbs0=param[1]
#        #Zwt0,Zbs0=self.get_wt_level(model,param)
#        k=0
#        
#        for j in np.arange(len(mod.mesh.Y)):
#            for i in np.arange(len(mod.mesh.X)):                
#                Zm=(mod.mesh.Ye[j]+mod.mesh.Ye[j+1])/2
#                if((Zwt0<=Zm)&(Zbs0>=Zm)):
#                    mod.vals[k]=param[-1]
#                else:
#                    mod.vals[k]=13#np.inf                        
#                k+=1
#        return mod

    def param2model(self,model_lst,invmodel_ref_lst,param):
#        print(param,self.magiclmd)
        mod_lst=[]
#        if isinstance(self.magiclmd,list):
#            log_m=0
#        else: 
#            log_m=-np.mean(np.log(self.por_array))*(self.m2+self.m1)/2+(self.logA2+self.logA1)/2

#        print(model_lst[0])
        for im,model  in enumerate(model_lst):
            mod=model.copy_model()
            Zwt=param[0]
            Zbs=param[1]

            Ny=len(mod.mesh.Y)
            Nx=len(mod.mesh.X)
#            model_im=np.reshape(model.get_log_vals(),[Ny,Nx])
            Zm=(mod.mesh.Ye[:-1]+mod.mesh.Ye[1:])/2

#            Zwt0,Zbs0=self.get_wt_level(model_lst[im],param)
            Zwt0=self.get_wt_level(model_lst[im],param)
#             print(Zwt0,Zbs0)
#             if isinstance(self.magiclmd,list):
#                 log_modelbk = 0
#             else:
#                 k=0
#                 for j in np.arange(len(mod.mesh.Y)):
#                     for i in np.arange(len(mod.mesh.X)):                
#                         Zm=(mod.mesh.Ye[j]+mod.mesh.Ye[j+1])/2
#                         if((Zwt0[i]>=Zm)&(Zbs0[i]<=Zm)):
#         #                if((Zwt0[i]<=mod.mesh.Ye[j])&(Zbs0[i]>=mod.mesh.Ye[j+1])):
#                             if model.vals[k]<np.log(1e8):
#                                 mod.vals[k]=1
#                             else:
#                                 mod.vals[k]=0#13#np.inf                                                    
#                         else:
#                             mod.vals[k]=0#13#np.inf                        
#                         k+=1
#                 log_modelbk=np.median(model.get_log_vals()[mod.vals==1])
# #                print(log_modelbk)
#                 log_modelbk = log_modelbk+np.log(self.magiclmd)
#            print(log_modelbk)            
#            Zwt0,Zbs0=self.get_wt_level(model_lst[im],param)
            k=0      
            if self.logsum: 
                empty_value=0
            else:     
                empty_value=np.log(1e8)
            for j in np.arange(len(mod.mesh.Y)):
                for i in np.arange(len(mod.mesh.X)):                
                    Zm=mod.mesh.Ye[j]#(mod.mesh.Ye[j]+mod.mesh.Ye[j+1])/2
                    if(Zwt0[i]>=Zm):#&(Zbs0[i]<=Zm)):
     #                if((Zwt0[i]<=mod.mesh.Ye[j])&(Zbs0[i]>=mod.mesh.Ye[j+1])):
                        if model.vals[k]<np.log(1e8):
                            None
                            mod.vals[k]=param[-1]#+log_modelbk-log_m
                        else:
                            mod.vals[k]=empty_value #np.log(1e8)#13#np.inf                                                    
                    else:
                        mod.vals[k]=empty_value #np.log(1e8)#13#np.inf                        
                    k+=1
            mod_lst.append(mod)      
        return mod_lst

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

                
    def get_K(self,model,beta):
        log_sigma=model.vals
        self.logA1/2
        #vlim=[1,4]
        logK=-beta*(np.abs(log_sigma)**0.8)/0.8 #
        logK2D=np.reshape(logK,[len(model.mesh.dY),len(model.mesh.dX)])
        dXmin=np.min(model.mesh.dX)
        xini=np.arange(len(model.mesh.dX))[model.mesh.dX==dXmin][0]
        xfin=np.arange(len(model.mesh.dX))[model.mesh.dX==dXmin][-1]
        for i in np.arange(0,xini+1):
            logK2D[:,i]=logK2D[:,xini]

        for i in np.arange(xfin,len(model.mesh.dX)):
            logK2D[:,i]=logK2D[:,xfin]
        logK=np.reshape(logK2D,len(model.mesh.dX)*len(model.mesh.dY))
        return np.exp(logK)               


    def get_wt_level(self,model,params):
        
        Zwtx=params[0]
        #Zbsx=params[1]
        beta=params[2]
        K=self.get_K(model,beta)
        Zwt=self.WTmodeling(K,Zwtx)#,Zbsx)#,Zbs
        #Zbs=Zbs*0+Zbsx
        return Zwt#,Zbs


#    def get_conditional_like(self,R):
#        Phi_e=self.por_array
#        Phi=(Phi_e[:-1]+Phi_e[1:])/2
#        phiprob=sm.PhiProb(self.m1, self.m2, self.logA1, self.logA2, R[0])        
#        likephi=np.zeros([len(Phi),len(R)])
#        for ir in np.arange(len(R)):        
#            phiprob.setparams([self.m1, self.m2, self.logA1, self.logA2, np.log(R[ir])])
#            for ip in np.arange(len(Phi)):        
#                likephi[ip,ir]=np.exp(phiprob.logpdf(np.log(Phi[ip])))
#        return likephi,Phi,Phi_e


    def get_proposal_logpdf(self,params,model_lst=[]):
        logpdf=np.log(1/(self.Zmax-self.Zmin))
        logpdf+=self.target_pdf_list[-2].logpdf(params[-2])
        logpdf+=self.target_pdf_list[-1].logpdf(params[-1])
        return logpdf

    def get_prior_logpdf(self,params,model_lst=[]):
        logpdf=np.log(1/(self.Zmax-self.Zmin))
        logpdf+=self.target_pdf_list[-2].logpdf(params[-2])
        logpdf+=self.target_pdf_list[-1].logpdf(params[-1])
            
        return logpdf

    def set_last_param(self,params):
        self.last_params=params.copy()

    def update_modelwtmap(self, modelwt):
        """!Update the probability desity map of the sampled \f$m_{wt}\f$ models.          
        @param model latest sampled \f$m_{wt}\f$ model
        """    
        
        if isinstance(modelwt, list):
            for ir in np.arange(len(modelwt)):
                mod = modelwt[ir].get_linear_vals()
                N = len(mod)
                for i in np.arange(N):
                    if mod[i] < modelwt[ir].airval:
                        n = (self.R[ir][:-1] < mod[i]
                             ) & (self.R[ir][1:] >= mod[i])
                        self.modelwtmap[ir][i, n] += 1

                self.Zwt_map = self.Zwt_map[:] + modelwt[ir].get_Zwt(curve=self.curve)[:]  # ?????
                self.Zbs_map = self.Zbs_map[:] + modelwt[ir].get_Zbs(curve=self.curve)[:]  # ?????
        else:
            resp = modelwt.get_linear_vals()
            N = len(resp)
            for i in np.arange(N):
                if resp[i] < modelwt[ir].airval:
                    n = (self.R[:-1] < resp[i]) & (self.R[1:] >= resp[i])
                    self.modelwtmap[i, n] += 1
            self.Zwt_map = self.Zwt_map[:]+modelwt.get_Zwt(curve=self.curve)[:]
            self.Zbs_map = self.Zbs_map[:]+modelwt.get_Zbs(curve=self.curve)[:]

    def get_fine_mesh_limits(self,model):
        mesh=model.mesh

        #Get X fine mesh limits
        dX=np.abs(mesh.dX)
        dXmin=np.min(dX)
        index=(dX==dXmin)
        Xmin=np.min(mesh.X[index])
        Xmax=np.max(mesh.X[index])

        #Get Y fine mesh limits
#        dY=np.abs(mesh.dY)
#        dYmin=np.min(dY)
#        index=(dY==dYmin)
#        Ymin=np.min(mesh.Y[index])
#        Ymax=np.max(mesh.Y[index])

        Ymin=self.Zmin
        Ymax=self.Zmax

        return Xmin,Xmax,Ymin,Ymax


    def Zwt_map_plot(self,modelwtmap,model):
                   
        Xmin,Xmax,Ymin,Ymax=self.get_fine_mesh_limits(model)

        dY = model.mesh.dY[0]
        Y = model.mesh.Ye  # -dY
        dX = np.min(model.mesh.dX[0])
        X = model.mesh.Xe  # -dY

        likewtmap2d = np.reshape(modelwtmap/np.sum(modelwtmap), self.model.get_N2d())
        plt.pcolor(X, Y, likewtmap2d/dY/dX, cmap='ocean_r')

        plt.xlim([Xmin,Xmax])
        plt.ylim([Ymin, Ymax])
        plt.gca().yaxis.set_inverted(False)
               
        plt.colorbar()


    def modelwtmap_plot(self,modelwtmap,model):
                   
        Xmin,Xmax,Ymin,Ymax=self.get_fine_mesh_limits(model)

        dY = model.mesh.dY[0]
        Y = model.mesh.Ye  # -dY
        dX = np.min(model.mesh.dX[0])
        X = model.mesh.Xe  # -dY

        if np.sum(modelwtmap)>0:
            likewtmap = np.sum(
            modelwtmap, axis=1)/np.sum(modelwtmap)
        else:
            likewtmap = np.zeros(len(model.get_linear_vals()))

        likewtmap2d = np.reshape(likewtmap, self.model.get_N2d())
        plt.pcolor(X, Y, likewtmap2d/dY/dX, cmap='ocean_r')

        plt.xlim([Xmin,Xmax])
        plt.ylim([Ymin, Ymax])
        plt.gca().yaxis.set_inverted(False)
               
        plt.colorbar()
 
 
    def modelbk_plot(self,modelbk):
        Xmin,Xmax,Ymin,Ymax=self.get_fine_mesh_limits(modelbk)
        
        modelbk.plot2D(log10=True,cmap='jet',vlim=[1,3])
        plt.xlim([Xmin,Xmax])
        plt.ylim([Ymin, Ymax])
        plt.gca().yaxis.set_inverted(False)              
#        plt.colorbar()
        
    def total_model_plot(self,total_model):
        Xmin,Xmax,Ymin,Ymax=self.get_fine_mesh_limits(total_model)
        total_model.plot2D(log10=True,cmap='jet',vlim=[1,3])
        plt.xlim([Xmin,Xmax])
        plt.ylim([Ymin, Ymax])
        plt.gca().yaxis.set_inverted(False)              
#        plt.colorbar()
        
    def modelplot(self, file, modelwtmap,model_wt_hat,bk_model):#vlim=[]):
        """!Plot and save the respmap (density probability of sampled \f$m_{bk}\f$ formward responses).          
        @param file(string) filename to save the respmap plot 
        @return None
        """
        for i in np.arange(len(model_wt_hat)):
            plt.figure(figsize=(15, 6))
            plt.subplot(3,1,1)
            self.Zwt_map_plot(modelwtmap[i],bk_model[i])
            plt.subplot(3,1,2)
            self.modelbk_plot(bk_model[i])
            plt.subplot(3,1,3)
            self.total_model_plot(bk_model[i].model_sum(bk_model[i],model_wt_hat[i]))
            plt.savefig(file+'_'+str(i)+'.jpg')
        
        
        


