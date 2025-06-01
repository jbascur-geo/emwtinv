# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 21:27:57 2021

@author: juan
"""
import numpy as np
import subprocess
import os
exelink = 'occam1d/'
datlink = 'occam1d/MT/'

class mt():
    def __init__(self):
        self.rapp_err=0.05*100
        self.papp_err=0.5*100
        self.iterations=1
        self.target_misfit=0.01
        self.roughness_type=1
        self.step_size=8
        self.model_limits=[]
        self.model_value_steps=[]
        self.debug_level=1
        self.iter=0
        self.lagrange=5
        self.roughness=1e10
        self.param_count=20
        
    def run_occam1d(self):

        subprocess.run(['occam1d\\OCCAM1DCSEM_JEJE.exe','occam1d\\MT\\startup'],capture_output=True, text=True)
        print(self.read_logfile())
        return self.read_model()
        
    def set_setup(self,modelini):
        fid=open(datlink+'startup','wt')
        fid.write('Format:             OCCAMITER_FLEX\n')
        fid.write('Description:        test\n')
        fid.write('Model File:         occam1d/MT/model\n')
        fid.write('Data File:          occam1d/MT/data\n')
        fid.write('Date/Time:          12/05/2008 10:04:06.795\n')
        fid.write('Iterations to run:  '+str(self.iterations)+'\n')                                               
        fid.write('Target Misfit:      '+str(self.target_misfit)+'\n')                                          
        fid.write('Roughness Type:     '+str(self.roughness_type)+'\n')                                                 
        fid.write('Stepsize Cut Count: '+str(self.step_size)+'\n')                                                 
        fid.write('!Model Limits:       min,max\n')
        fid.write('!Model Value Steps:  stepsize (e.g. 0.2 or 0.1)\n')
        fid.write('Debug Level:        '+str(self.debug_level)+'\n')                                                 
        fid.write('Iteration:          '+str(self.iter)+'\n')                                                 
        fid.write('Lagrange Value:     '+str(self.lagrange)+'\n')                                          
        fid.write('Roughness Value:    '+str(self.roughness)+'\n')                                     
        fid.write('Misfit Value:       1000.000\n')                                          
        fid.write('Misfit Reached:     0\n')                                                 
        fid.write('Param Count:       '+str(len(modelini))+'\n')         
        for i in np.arange(len(modelini)):
            fid.write(str(modelini[i])+' ')         
            if np.mod(i,4)==3:
                fid.write('\n')

    def set_model(self,thk,model,model_ref):
        fid=open(datlink+'model','wt')
        fid.write('Format: Resistivity1DMod_1.0\n')
        fid.write('#LAYERS:    '+str(int(len(model)+2))+'\n')
        fid.write('! Layer block in file is:\n')
        fid.write('! [top_depth 	resistivity  penalty	prejudice   prej_penalty]  ! note that top_depth and penalty of 1st layer ignored\n')
        fid.write('   -100000       1d12          0          0             0          ! Air, fixed layer\n')
        fid.write('      0          1d12           0          0             0          ! Sea, fixed layer\n') 
        Y=np.concatenate([[0],np.cumsum(thk)])
        for i in np.arange(len(model)):
            strtmp=str(Y[i]+1000)+' '+'?'+' '+'1'+' '+str(model_ref[i])+' '+'0'+'\n' #str(model_ref[i])
            fid.write(strtmp)
        self.param_count=len(model)
        fid.close()

    def set_data(self,data,F):
        fid=open(datlink+'data','wt')
        fid.write('Format:  EMData_1.1\n')
        fid.write('! This is a synthetic data file generated from Dipole1D output\n')
        fid.write('! Enoisefloor: 1e-15 V/m/Am \n') 
        fid.write('! Bnoisefloor: 1e-18 T/Am \n') 
        fid.write('! Percent Noise added: 2 % \n')
        fid.write('! Data have been rotated (theta,alpha,beta):      0,      0,      0 degrees         \n')
        fid.write('\n')
        fid.write('# Transmitters:   41\n')
        fid.write('!            X            Y            Z      Azimuth          Dip \n')
        for i in np.arange(41):
            strtmp=str(0)+' '+str(i*500)+' '+str(0)+' '+' '+str(90)+' '+str(0)
            fid.write(strtmp+'\n')
        fid.write('# Frequencies:    '+str(len(F))+'\n')
        for i in np.arange(len(F)):
            fid.write(str(F[i])+'\n')

        fid.write('    Phase Convention: lag\n')
        fid.write('# Receivers:      1\n')
        fid.write('!            X            Y            Z        Theta        Alpha         Beta\n')
        fid.write('             0            0           1000            0            0            0\n') 
        fid.write('# Data:       '+str(2*len(F))+'\n')
        fid.write('!         Type        Freq#        Tx#           Rx#       Data 	  Std_Error\n')
        
        for i in np.arange(len(F)):
            Rapp=np.abs(data[i])**2/(2*np.pi*F[i]*4*np.pi*1e-7)
            Papp=np.angle(data[i])/np.pi*180#+90#np.abs(np.angle(data[i])/np.pi*180)
            strtmp=str(103)+' '+str(i+1)+' '+str(0)+' '+str(1)+' '+str(Rapp)+' '+str(self.rapp_err*Rapp)
            fid.write(strtmp+'\n')
            strtmp=str(104)+' '+str(i+1)+' '+str(0)+' '+str(1)+' '+str(Papp)+' '+str(self.papp_err)
            fid.write(strtmp+'\n')
        fid.close()
            
    def read_model(self):
        log=self.read_logfile()
        N=int(np.max(np.array(log)[:,0]))
        fid=open('ITER_'+str(N)+'.iter','rt')#exelink+
        lines=fid.readlines()
        for k in np.arange(len(lines)):
            linedata=lines[k].split()
            if linedata:
                if(linedata[0]=='Param'):
                    Nm=int(linedata[-1])      
                    break
        model=np.zeros(Nm)
        index=0
        for i in np.arange(Nm):
            modeldata=lines[i+k+1].split()
            for n in np.arange(len(modeldata)):
                model[index]=10**(float(modeldata[n]))
                index=index+1
                if index > Nm-1:
                    break
            if index > Nm-1:
                break
        fid.close()
        return model
                        
    
    def read_logfile(self):
        #Read LogFile with inversion results
        #output: list with iteration and RMS error

        fid=open('ITER.logfile','rt')#exelink
        lines=fid.readlines()
        iterations=[]
        iterx=0
        index=0
        for line in lines:
            linedata=line.split()
            if linedata:
                if(linedata[0]=='Starting'):       
                    iterations.append([iterx,float(linedata[-1])])
                    iterx=iterx+1
                if(linedata[0]=='and'):       
                    lastRMS=float(linedata[-1])
            
        if(len(linedata)>1):
            if(linedata[1]=='problems'): 
                None
            else:
                iterations.append([iterx,lastRMS])
        fid.close()
        return iterations
