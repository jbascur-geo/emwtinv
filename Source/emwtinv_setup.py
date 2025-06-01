# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 12:49:47 2023

@author: juan
"""
import numpy as np

class setup:
    
    def __init__(self):

        self.data_list='dc1ddata.dat'
        self.method='DC1D'

        #M-Step (Classic Geophysical Inversion) Default Parameters
        self.externalmodel=False
        self.external_constant_model=False
        self.externalmodel_ref=False
        self.model_file="modeltest.model"
        self.model_mesh="modeltest.mesh"        
        self.modelref_file='./Case I/Ref/dc1dmod_bk.txt'
        self.modelref_mesh='./Case I/Ref/dc1dmesh.txt'
        self.invert_initial_bkmodel = True
        self.reference_value=10
        self.sub_lmd=10
        self.emmax_iter=10
        self.invparameters0=[]
        self.invparameters0.append(1) #Colling rate
        self.invparameters0.append(1) #Beta
        self.invparameters0.append(0.0001) #Minbeta
        self.invparameters0.append(1.5) #Colling factor
        self.invparameters0.append(0.2) #alpha_x
        self.invparameters0.append(1) #alpha_y
        self.invparameters0.append(1) #alpha_z
        self.invparameters0.append(0.01) #alpha_s
        self.invparameters0.append(0.1) #chifact

        self.invparameters=[]
        self.invparameters.append(1) #Colling rate
        self.invparameters.append(1) #Beta
        self.invparameters.append(0.01) #Minbeta
        self.invparameters.append(1.5) #Colling factor
        self.invparameters.append(1) #alpha_x
        self.invparameters.append(0.2) #alpha_y
        self.invparameters.append(1) #alpha_z
        self.invparameters.append(0.01) #alpha_s
        self.invparameters.append(0.1) #chifact

        #E-Step (Bayesian Inversion) Default Parameters
        self.estep_niter=10000 #number of Metropolis hasting samples
        self.estep_beta=0 #number of Metropolis hasting samples
        self.estep_beta_factor=1 #number of Metropolis hasting samples
        self.Nparams=0
        self.pdfpar_min_max_N=[]
        self.e_step_prop='2D'
        self.vmin=[]
        self.vmax=[]

    def get_string(self,line):
        sepchars=' '
        n=line.find(sepchars)
        return line[(n+1):].split('\n')[0]
        
        
    def write_default(self,file):
        fid=open(file,'wt')
        fid.write('GEOPHYSICAL DATA\n')
        fid.write('Data_list: '+'dc1ddata.dat'+'\n')
        fid.write('Method: '+'DC1D'+'\n')

        fid.write('M-STEP (CLASSIC GEOPHYSICAL INVERSION)\n')
        fid.write('externalmodel: '+'False'+'\n')
        fid.write('external_constant_model: '+'False'+'\n')
        fid.write('externalmodel_ref: '+'False'+'\n')
        fid.write('model_file: '+'modeltest.model'+'\n')
        fid.write('model_mesh: '+'modeltest.mesh' + '\n')        
        fid.write('modelref_file: '+'./Case I/Ref/dc1dmod_bk.txt'+'\n')
        fid.write('modelref_mesh: '+'./Case I/Ref/dc1dmesh.txt'+'\n')
        fid.write('invert_initial_bkmodel: ' + 'True' + '\n')
        fid.write('reference_value: ' + str(1000) +'\n')
        fid.write('sub_lmd: ' + str(10) + '\n')
        fid.write('emmax_iter: ' + str(10) + '\n')
        fid.write('invparameters0.Collingrate: ' + str(self.invparameters0[0])+'\n')
        fid.write('invparameters0.Beta: ' + str(self.invparameters0[1])+'\n') 
        fid.write('invparameters0.Minbeta: ' + str(self.invparameters0[2])+'\n')
        fid.write('invparameters0.Collingfactor: ' + str(self.invparameters0[3])+'\n')
        fid.write('invparameters0.alpha_x: ' + str(self.invparameters0[4])+'\n')
        fid.write('invparameters0.alpha_y: ' + str(self.invparameters0[5])+'\n')
        fid.write('invparameters0.alpha_z: ' + str(self.invparameters0[6])+'\n')
        fid.write('invparameters0.alpha_s: ' + str(self.invparameters0[7])+'\n')
        fid.write('invparameters0.chifact: ' + str(self.invparameters0[8])+'\n')

        fid.write('invparameters.Collingrate: ' + str(self.invparameters[0])+'\n')
        fid.write('invparameters.Beta: ' + str(self.invparameters[1])+'\n')
        fid.write('invparameters.Minbeta: ' + str(self.invparameters[2])+'\n')
        fid.write('invparameters.Collingfactor: ' + str(self.invparameters[3])+'\n')
        fid.write('invparameters.alpha_x: ' + str(self.invparameters[4])+'\n')
        fid.write('invparameters.alpha_y: ' + str(self.invparameters[5])+'\n')
        fid.write('invparameters.alpha_z: ' + str(self.invparameters[6])+'\n')
        fid.write('invparameters.alpha_s: ' + str(self.invparameters[7])+'\n')
        fid.write('invparameters.chifact: ' + str(self.invparameters[8])+'\n')
        
        fid.write('E-STEP (BAYESIAN INVERSION)\n')
        fid.write('estep_niter: '+ str(self.estep_niter))
        fid.write('estep_beta: '+ str(self.estep_beta))
        fid.write('estep_beta_factor: '+ str(self.estep_beta_factor))
        fid.write('Nparams: '+ str(self.Nparams))
        fid.write('pdf_params_min_max_N: '+ str(self.pdf_params_min_max_N))
        fid.write('e_step_proc: '+ str(self.e_step_prop))
       
    def read(self,file):
        fid=open(file,'rt')

        #Geophysical Data
        lines=fid.readlines()
        k=0
        k=k+1
        self.data_list=self.get_string(lines[k])
        k=k+1
        self.method=self.get_string(lines[k])

        #M-step (Classic Geophysical Inversion)
        k=k+1
        k=k+1
        self.externalmodel=(self.get_string(lines[k])=='True')
        k=k+1
        self.external_constant_model= (self.get_string(lines[k])=='True')
        k=k+1
        self.externalmodel_ref= (self.get_string(lines[k])=='True')
        k=k+1
        self.model_file=self.get_string(lines[k])
        k=k+1
        self.model_mesh=self.get_string(lines[k])        
        k=k+1
        self.modelref_file=self.get_string(lines[k])
        k=k+1
        self.modelref_mesh=self.get_string(lines[k])
        k=k+1
        self.invert_initial_bkmodel = (self.get_string(lines[k])=='True')
        k=k+1
        self.reference_value= float(self.get_string(lines[k]))
        k=k+1
        self.sub_lmd=float(self.get_string(lines[k]))
        k=k+1
        self.emmax_iter=int(self.get_string(lines[k]))
        k=k+1
        self.invparameters0=[]
        self.invparameters0.append(float(self.get_string(lines[k]))) #Colling rate
        k=k+1
        self.invparameters0.append(float(self.get_string(lines[k]))) #Beta
        k=k+1
        self.invparameters0.append(float(self.get_string(lines[k]))) #Minbeta
        k=k+1
        self.invparameters0.append(float(self.get_string(lines[k]))) #Colling factor
        k=k+1
        self.invparameters0.append(float(self.get_string(lines[k]))) #alpha_x
        k=k+1
        self.invparameters0.append(float(self.get_string(lines[k]))) #alpha_y
        k=k+1
        self.invparameters0.append(float(self.get_string(lines[k]))) #alpha_z
        k=k+1
        self.invparameters0.append(float(self.get_string(lines[k]))) #alpha_s
        k=k+1
        self.invparameters0.append(float(self.get_string(lines[k]))) #chifact

        k=k+1
        self.invparameters=[]
        self.invparameters.append(float(self.get_string(lines[k]))) #Colling rate
        k=k+1
        self.invparameters.append(float(self.get_string(lines[k]))) #Beta
        k=k+1
        self.invparameters.append(float(self.get_string(lines[k]))) #Minbeta
        k=k+1
        self.invparameters.append(float(self.get_string(lines[k]))) #Colling factor
        k=k+1
        self.invparameters.append(float(self.get_string(lines[k]))) #alpha_x
        k=k+1
        self.invparameters.append(float(self.get_string(lines[k]))) #alpha_y
        k=k+1
        self.invparameters.append(float(self.get_string(lines[k]))) #alpha_z
        k=k+1
        self.invparameters.append(float(self.get_string(lines[k]))) #alpha_s
        k=k+1
        self.invparameters.append(float(self.get_string(lines[k]))) #chifact

        #E-step (Bayesian Inversion)
        k=k+1
        k=k+1
        self.estep_niter=float(self.get_string(lines[k]))
        k=k+1
        self.estep_beta=float(self.get_string(lines[k]))
        k=k+1
        self.estep_beta_factor=float(self.get_string(lines[k]))
#        print('estep beta factor:',self.estep_beta_factor)
        k=k+1
        self.e_step_prop=self.get_string(lines[k])
        try:
            k=k+1
            self.vmin=float(self.get_string(lines[k]))
            k=k+1
            self.vmax=float(self.get_string(lines[k]))
        except:
            None
        fid.close()

