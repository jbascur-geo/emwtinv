# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 18:04:08 2021

@author: juan
"""
#import glidertools as gt
#from cmocean import cm as cmo  # we use this for colormaps
from scipy.interpolate import Rbf
import scipy
import simpeg.electromagnetics.natural_source as NSEM

import matplotlib.pyplot as plt
from simpeg import discretize 
from simpeg import (
    maps,
    data,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
    utils,
)
from simpeg.electromagnetics.static import resistivity as dc, utils as dcutils
from simpeg.utils import model_builder, surface2inds#_topo
from simpeg.utils import sdiag
from simpeg import SolverLU as Solver
from discretize.utils import mkvc, refine_tree_xyz
import simpeg.electromagnetics.natural_source as NS
import link_occam1d
import simpeg.electromagnetics.time_domain as tdem 
#import time_domain_1d
#import setup 
from simpeg.simulation import BaseSimulation
from scipy.sparse import csr_matrix
from simpeg.potential_fields import gravity
from simpeg.electromagnetics import time_domain


import numpy as np


class tdem_class_1d:
    def __init__(self,loop_size=200,times=[],ramp=1e-7,TF=2,conf='cl'):
         
        #Survey Parameters
#        self.iter_counter=0
#        self.iterindex=0
#        self.loop_size=loop_size
#        self.ramp=ramp
#        self.TF=TF
#        self.conf=conf
#        self.times=times
        None
        self.iterindex=0
        
    def set_modelref(self,model):
        self.model_ref = 1/model

    def set_modelini(self,model):
        self.model_ini = 1/model

    def set_model(self,model):
        self.model= 1/model

    def set_data(self,arr):
        
        #Set survey parameters
        self.loop_size=arr[0][0]        
        self.ramp=arr[0][1]
        perc=arr[0][2]/100 #noise level (%)
        if self.ramp < 1e-7: #if a lower ramp is set, the forward modeling is unstable in SimPEG
            self.ramp = 1e-7
        
        #Set TEM data
        self.data=arr[1][:,1]  #dB/dZ
        self.times=arr[1][:,0] #times
        self.sigma=perc*np.abs(self.data)+1e-20 #error floor

        #Define receivers
        L=self.loop_size/2
        rx=time_domain.receivers.PointMagneticFieldTimeDerivative(locations=[0,0,0],times=self.times,orientation="z")

        #Define a source
        waveform = time_domain.sources.RampOffWaveform(off_time=self.ramp)
        self.waveform=waveform
        source_current=1.0
        #Square Loop Tranmitter
        source_location=np.zeros([5,3])
        source_location[0,:]=np.array([-L, L, 0])
        source_location[1,:]=np.array([ L, L, 0])
        source_location[2,:]=np.array([ L,-L, 0])
        source_location[3,:]=np.array([-L,-L, 0])
        source_location[4,:]=np.array([-L, L, 0])
        source_list = [
            time_domain.sources.LineCurrent(
                receiver_list=[rx],
                location=source_location,
                waveform=waveform,
                current=source_current,
                )
            ]
        self.survey = time_domain.Survey(source_list)
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)
        
    def set_mesh(self,m,ind_active=[]):
        self.thk=m.dY
        self.model=np.zeros(len(m.dY))
        self.mesh = discretize.TensorMesh([(np.r_[self.thk])], "0")#, self.thk[-1]
        self.model_mapping = maps.ExpMap(nP=len(self.thk))
        self.simulation = time_domain.Simulation1DLayered(
            survey=self.survey, thicknesses=self.thk[:-1], sigmaMap=self.model_mapping
        )

    def set_mesh_dY(self,dY):
        self.thk=dY
        self.model=np.zeros(len(dY))

    def set_halfmodel(self,r0):
        self.model=np.ones(len(self.thk))*(1/r0)
        
    def fwd(self):        
        dpred = self.simulation.dpred(np.log(self.model))
        return dpred

    def loglike(self):
        x=self.fwd()
        d=self.data
        sigma=self.sigma
        return -0.5*(np.sum((np.array(x-d)**2)/(sigma*sigma))/len(self.data))
    
    def get_appres(self,EM):
        EMvolt = EM/(4*np.pi)*10/1e6 #tovolt
        mu = 4e-7*np.pi
        Q = self.loop_size**2
        q = 1
        T = self.times**(-5/2)
        return (mu/np.pi)*(np.pi*mu*Q*q/EMvolt*T)**(2/3)

    def set_bkmapping(self,wtmodel):
        
        class HybridSum(maps.IdentityMap):
            """
            Apply the model sum of Hybrid Decomposition
            .. math::

                \rho_t = 1/(1/\rho_wt + 1/\rho_bk)
            """

            def __init__(self, mesh=None, nP=None, wt_val=None, ind_active_wt=None, **kwargs):
                self.wt_val=wt_val
                self.ind_active_wt=ind_active_wt
                super(HybridSum, self).__init__(mesh=mesh, nP=nP, **kwargs)

            def _transform(self, m):
                cond = np.exp(mkvc(m))#-
                cond_wt=1/self.wt_val#1/
                cond[self.ind_active_wt]+=cond_wt
                return cond

            def deriv(self, m):#, v=None):
                mkvc_m=mkvc(m)
                cond_bk = np.exp(mkvc_m)#-
                cond_wt = 1/self.wt_val#1/
                cond_t = cond_bk.copy()
                cond_t[self.ind_active_wt]+=cond_wt
                return sdiag(cond_bk)#/(cond_t**2))#cond_bk/

            def inverse(self, m):
                cond = mkvc(m)#1/
                cond_wt = 1/self.wt_val#1/
                cond[self.ind_active_wt]-=cond_wt
                return np.log(cond)#1/


        val_wt=np.min(wtmodel)
        ind_active_wt=(val_wt==wtmodel)
        nC=self.mesh.nC
        
        HS=HybridSum(self.mesh,nP=nC,wt_val=val_wt,ind_active_wt=ind_active_wt)
        mapping= HS
        
        return mapping


    def invert(self,W=[]):
        #Data Misfit        
        dmis = data_misfit.L2DataMisfit(simulation=self.simulation, data=self.data_object)

        #Initial and reference models
        model_ini = np.r_[self.model_ini]
        model_ref= np.r_[self.model_ref]

        #Setting Regularization  
        if(len(W)>0):          
            reg = regularization.WeightedLeastSquares(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x,weights=W,reference_model=np.log(model_ref))
        else:
            reg = regularization.WeightedLeastSquares(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x,reference_model=np.log(model_ref))

        reg.referenced_model_in_smooth=True
        #reg.mrefInSmooth = True #Forcing that mref must be used
        self.reg=reg
        opt = optimization.InexactGaussNewton(maxIter=1, maxIterCG=300)
        update_sensitivity_weights = directives.UpdateSensitivityWeights()   

        # Schedule for beta updating during inversion
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=self.beta)
        beta_schedule = directives.BetaSchedule(coolingFactor=self.coolingFactor, coolingRate=self.coolingRate)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=0.1)

        #inversion object
        self.directives_list = [
             update_sensitivity_weights,
             starting_beta,
             beta_schedule,
             target_misfit,
         ]
        self.inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

        #Forcing Smooth Reg to use the reference model.
        #self.reg.objfcts[1].mrefInSmooth=True
        #self.reg.objfcts[1].referenced_model_in_smooth=True
        #self.reg.objfcts[1].reference_model=self.reg.objfcts[0].reference_model #forcing that mref must be used in smoothness

        #Run Inversion
        inv = inversion.BaseInversion(self.inv_prob, directiveList=self.directives_list)
        recovered_model = inv.run(-np.log(model_ini))
        
        return np.exp(-recovered_model)


    def invert_step(self,W=[]):
        #Data Misfit        
        dmis = data_misfit.L2DataMisfit(simulation=self.simulation, data=self.data_object)

        #Initial and reference models
        model_ini = np.r_[self.model_ini]
        model_ref= np.r_[self.model_ref]

        #Setting Regularization  
        if(len(W)>0):          
            reg = regularization.WeightedLeastSquares(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x,weights=W,reference_model=-np.log(model_ref))
        else:
            reg = regularization.WeightedLeastSquares(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x,reference_model=-np.log(model_ref))

        #reg.mrefInSmooth = True #Forcing that mref must be used
        #reg.referenced_model_in_smooth=True
        self.reg=reg
        opt = optimization.InexactGaussNewton(maxIter=1, maxIterCG=300)
        update_sensitivity_weights = directives.UpdateSensitivityWeights()   

        # Schedule for beta updating during inversion
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=self.beta)
        beta_schedule = directives.BetaSchedule(coolingFactor=1.0, coolingRate=1.0)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=0.1)

        #inversion object
        self.directives_list = [
             update_sensitivity_weights,
             starting_beta,
             beta_schedule,
             target_misfit,
         ]
        self.inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

        #Forcing Reg to use the reference model.
        #self.reg.objfcts[1].mrefInSmooth=True
        #self.reg.objfcts[1].referenced_model_in_smooth=True
        #self.reg.objfcts[1].reference_model=self.reg.objfcts[0].reference_model

        #Run Inversion
        inv = inversion.BaseInversion(self.inv_prob, directiveList=self.directives_list)
        recovered_model = inv.run(-np.log(model_ini))
        
        self.iterindex+=1
        return np.exp(-recovered_model)

    def invert_step2(self,modelwt,magic_lmd,W=[]):
        
        #create data object
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)

        #setting a mesh
        layer_thicknesses = np.r_[self.thk[:-1]]#[:]#,[self.thk[-1]]])
        self.mesh = discretize.TensorMesh([np.r_[layer_thicknesses, layer_thicknesses[-1]]], "0")#[(np.r_[layer_thicknesses])], "0")  

        model_ini = np.r_[self.model_ini]

        #Setting foward modeling        
        mapping=self.set_bkmapping(modelwt)
        #model_map = maps.IdentityMap(nP=len(model_ini)) * maps.ExpMap()
        simulation = time_domain.Simulation1DLayered(
            survey=self.survey, thicknesses=self.thk[:-1], sigmaMap=mapping
        )

#        simulation = dc.simulation_1d.Simulation1DLayers(
#            survey=self.survey,
#            rhoMap=mapping,
#            thicknesses=layer_thicknesses
#        )

        #Inversion settings 
        model_ref= np.r_[self.model_ref]#np.concatenate([self.model_ref,[self.model_ref[-1]]])
        dmis = data_misfit.L2DataMisfit(simulation=simulation, data=self.data_object)

        #Setting Regularization  
        if(len(W)>0):          
            reg = regularization.WeightedLeastSquares(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x,reference_model=np.log(model_ref),weights=W)
        else:
            reg = regularization.WeightedLeastSquares(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x,reference_model=np.log(model_ref))

        reg.reference_model_in_smooth = True
        #reg.mrefInSmooth = True
        self.reg=reg
        opt = optimization.InexactGaussNewton(maxIter=10, maxIterCG=300)
        update_sensitivity_weights = directives.UpdateSensitivityWeights()   

        # Schedule for beta updating during inversion
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=self.beta)
        beta_schedule = directives.BetaSchedule(coolingFactor=1.0, coolingRate=1.0)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=0.1)


        #inversion object
        self.directives_list = [
            update_sensitivity_weights,
            starting_beta,
            beta_schedule,
            target_misfit,
        ]
        self.inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

        #Forcing Smooth Reg to use the reference model.
        self.reg.objfcts[1].reference_model_in_smooth=True
        self.reg.objfcts[1].reference_model=self.reg.objfcts[0].reference_model

        #Run Inversion
        inv = inversion.BaseInversion(self.inv_prob, directiveList=self.directives_list)
        recovered_model = inv.run(np.log(model_ini))#np.log(
            
        self.iterindex+=1
#        print(np.exp(-recovered_model))
        return np.exp(-recovered_model)


class mt_class_1d_par2:
    def __init__(self,frec=[]):
        None
        self.frec = frec
        self.iterindex=0                   

    def set_bkmapping(self,wtmodel):
        class mapping_rho(maps.IdentityMap):
            def __init__(self,R0, mesh=None, nP=None, **kwargs):
                super(mapping_rho, self).__init__(mesh=mesh, nP=nP, **kwargs)
                self.R0=R0
        
            def _transform(self, m):
                R1=np.exp(m[0])
                Rwt=1/self.R0+R1
                R2=np.exp(m[1])
                rho=np.array([R1,Rwt,R2])
                print('transform',rho)
                return rho
        
            def deriv(self, m):#, v=None):
                R1=np.exp(m[0])
                Rwt=1/self.R0+R1
                R2=np.exp(m[1])
                grad=np.zeros([2,3])
                grad[0,0]=R1
                grad[0,1]=R1
                grad[0,2]=0
                grad[1,0]=0
                grad[1,1]=0
                grad[1,2]=R2
                print(grad,m)
                return grad.T
        
            def inverse(self, m):
                R1=m[0]
                R2=m[2]
                logrho=np.array([np.log(R1),np.log(R2)])
#                print('logrho',logrho)
                return logrho    
        
 
        vals=wtmodel
        R0=np.min(vals)
        mapping=mapping_rho(R0)
        return mapping

    def set_modelref(self,model):
        vals = 1/model.copy()
        self.model_ref=np.array([np.log(vals[0]),np.log(vals[-1])])

    def set_modelini(self,model):
        vals = 1/model.copy()
        self.model_ini=np.array([np.log(vals[0]),np.log(vals[-1])])

    def set_model(self,res):
        self.model=1/res

    def set_data(self,arr):        
        self.frec=arr[0][0]
        self.perc=arr[0][1]
        self.data=arr[1]
        self.sigma=np.abs(self.perc*self.data/100)
        
        Ns=len(self.frec)#24 #numero de frecuencias
        
        freq=self.frec#np.logspace(4, 0, Ns)
        receiver_list = []
        rx_loc=[[0,0,0],[0,0,0]]
        for rx_orientation in ["xy"]:
            receiver_list.append(NSEM.Rx.PointNaturalSource(rx_loc, rx_orientation, "apparent_resistivity"))
            receiver_list.append(NSEM.Rx.PointNaturalSource(rx_loc, rx_orientation, "phase"))
        
        # Source list
        source_list = [
            NSEM.Src.PlanewaveXYPrimary(receiver_list, ifreq)
            for ifreq in freq
            ]
        # Survey MT
        survey = NSEM.Survey(source_list)
        self.survey=survey
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)

        
    def fwd(self):        
        Nl=len(self.thk)
        self.model_mapping = maps.ExpMap(nP=len(self.thk))
        self.simulation = NSEM.Simulation1DRecursive(
        survey=self.survey,
        sigmaMap=self.model_mapping,
        thicknesses=self.thk[:-1])
        dpred = self.simulation.dpred(np.log(self.model))
        return dpred

    def set_mesh(self,m,ind_active=[]):
        self.thk=m.dY
        self.model=np.zeros(len(m.dY))
        self.mesh = discretize.TensorMesh([(np.r_[self.thk])], "0")

    def set_mesh_dY(self,dY):
        self.thk=dY
        self.model=np.zeros(len(dY))
        self.mesh = discretize.TensorMesh([(np.r_[self.thk])], "0")#, self.thk[-1]
            
    def invert(self,W=[]):
 #       pars=self.mod2par(modelwt)
#        mod_thk=np.array([pars[0],pars[1]-pars[0]])
        mod_thk=np.array([50])
        
        #create data object
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)


        model_ini = self.model_ini
        model_ref= self.model_ref
        Nl=2#int((len(model_ini)+1)/2)

        #Setting foward modeling        
#        mapping=self.set_bkmapping(modelwt)
        self.mapping2= maps.ExpMap(nP=Nl)


        sim = NSEM.Simulation1DRecursive(
        survey=self.survey,
        sigmaMap=self.mapping2,
        thicknesses=mod_thk)

#        sim = time_domain.Simulation1DLayered(
#            survey=self.survey, thicknesses=mod_thk, sigmaMap=self.mapping2
#        )

#        self.sim2=sim
        #Inversion settings 
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=self.data_object)

        #model regularization
        mesh_r=discretize.TensorMesh([Nl])
        reg_rho=regularization.WeightedLeastSquares(
            mesh_r, alpha_s=1, alpha_x=0)#, mapping=wire_map.rho)
        
        reg=reg_rho#+reg_thk
        self.reg=reg

    
        starting_beta = directives.BetaEstimateMaxDerivative(beta0_ratio=0.01)#BetaEstimate_ByEig(beta0_ratio=0.01)        
                        
        starting_beta = starting_beta
        beta_schedule = directives.BetaSchedule(coolingFactor=1.0, coolingRate=1.0)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=0.1)
        self.directives_list = [
            starting_beta,
            beta_schedule,
            target_misfit,
        ]

        #Run Inversion
        opt = optimization.InexactGaussNewton(maxIter=10, maxIterCG=300)
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
        inv = inversion.BaseInversion(inv_prob, directiveList=self.directives_list)


        self.inv=inv
        imodel = inv.run(model_ini)
        
        Z=np.concatenate([[0],np.cumsum(self.thk[:-1])])

        recovered_model=np.zeros(len(self.thk))
        for i in np.arange(len(self.thk)):
            if Z[i]>=mod_thk[0]:
                recovered_model[i] = imodel[1]             
            else:
                recovered_model[i] = imodel[0]             
        self.iterindex+=1
        return np.exp(-recovered_model)
    
    def invert_step(self,W=[]):
        None

    def mod2par(self,wtmodel):
        vals=wtmodel
        R0=np.min(vals)
        index=(vals==R0)
        Z1=np.array(np.concatenate([[0],np.cumsum(self.thk[:-1])]))
        Z2=np.array(np.cumsum(self.thk))
        if np.sum(index)<=1:
            Zwt=Z1[index]
            Zbs=Z2[index]
        else:
            Zwt=Z1[index][0]
            Zbs=Z2[index][-1]
        pars=np.array([float(Zwt),float(Zbs),float(R0)])
        return pars

        
    def invert_step2(self,modelwt,W=[],magic_lmd=[]):

        pars=self.mod2par(modelwt)
        print('pars',pars)
        mod_thk=np.array([pars[0],pars[1]-pars[0]])
        #pars[0],
        #create data object
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)


        model_ini = self.model_ini
        model_ref= self.model_ref
        Nl=2#int((len(model_ini)+1)/2)

        #Setting foward modeling        
        mapping=self.set_bkmapping(modelwt)
        self.mapping2=mapping
#Simulation1DRecursive
        sim = NSEM.Simulation1DRecursive(survey=self.survey,sigmaMap=self.mapping2,thicknesses=mod_thk)

#        sim = time_domain.Simulation1DLayered(
#            survey=self.survey, thicknesses=mod_thk, sigmaMap=self.mapping2
#        )

        self.sim2=sim
        #Inversion settings 
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=self.data_object)

        #model regularization
        mesh_r=discretize.TensorMesh([Nl])
        reg_rho=regularization.WeightedLeastSquares(
            mesh_r, alpha_s=1, alpha_x=0)#, mapping=wire_map.rho)
        
        reg=reg_rho#+reg_thk
        self.reg=reg

    
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=0.01)        
                        
        starting_beta = starting_beta
        beta_schedule = directives.BetaSchedule(coolingFactor=1.0, coolingRate=1.0)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=0.1)
        self.directives_list = [
            starting_beta,
            beta_schedule,
            target_misfit,
        ]

        #Run Inversion
#        print(dir(optimization))
        opt = optimization.InexactGaussNewton(maxIter=10, maxIterCG=300)
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
        inv = inversion.BaseInversion(inv_prob, directiveList=self.directives_list)

        self.inv=inv
        
#        print('inv.invProb.dmisfit.deriv2(model_ini.T,model_ini.T)')
#        print(inv.invProb.dmisfit.deriv2(model_ini.T,model_ini.T))
        imodel = inv.run(model_ini)
        
        Z=np.concatenate([[0],np.cumsum(self.thk[:-1])])

        recovered_model=np.zeros(len(self.thk))
        for i in np.arange(len(self.thk)):
            if Z[i]>=pars[1]:
                recovered_model[i] = imodel[1]             
            else:
                recovered_model[i] = imodel[0]             
        self.iterindex+=1
        return np.exp(-recovered_model)


class mt_class_1d:
    def __init__(self,frec=[]):
        None
        self.frec = frec
        self.iterindex=0
        
    def set_modelref(self,model):
        self.model_ref = 1/model#np.log(model)

    def set_modelini(self,model):
        self.model_ini = 1/model#np.log(model)

    def set_model(self,model):
        self.model= 1/model#np.log(model)

    def set_data(self,arr):
        self.frec=arr[0][0]
        self.perc=arr[0][1]
        self.data=arr[1]
        self.sigma=np.abs(self.perc*self.data/100)
        
        Ns=len(self.frec)#24 #numero de frecuencias
        
        freq=self.frec#np.logspace(4, 0, Ns)
        receiver_list = []
        rx_loc=[[0,0,0],[0,0,0]]
        for rx_orientation in ["xy"]:
            receiver_list.append(NSEM.Rx.PointNaturalSource(rx_loc, rx_orientation, "apparent_resistivity"))
            receiver_list.append(NSEM.Rx.PointNaturalSource(rx_loc, rx_orientation, "phase"))
        
        # Source list
        source_list = [
            NSEM.Src.PlanewaveXYPrimary(receiver_list, ifreq)
            for ifreq in freq
            ]
        # Survey MT
        survey = NSEM.Survey(source_list)
        self.survey=survey
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)
        
    def set_mesh(self,m,ind_active=[]):
        self.thk=m.dY
        Nl=len(self.thk)
        self.model=np.zeros(len(m.dY))
        self.mesh = discretize.TensorMesh([(np.r_[self.thk])], "0")#, self.thk[-1]
        self.model_mapping = maps.ExpMap(nP=len(self.thk))
        self.simulation = NSEM.Simulation1DRecursive(
        survey=self.survey,
        sigmaMap=maps.ExpMap(nP=Nl),
        thicknesses=self.thk[:-1])

#        rhoMap=maps.ExpMap(nP=Nl),


    def set_mesh_dY(self,dY):
        self.thk=dY
        self.model=np.zeros(len(dY))

    def set_halfmodel(self,r0):
        self.model=np.ones(len(self.thk))*(1/r0)
        
    def fwd(self):        
        dpred = self.simulation.dpred(np.log(self.model))
        return dpred

    def loglike(self):
        x=self.fwd()
        d=self.data
        sigma=self.sigma
        return -0.5*(np.sum((np.array(x-d)**2)/(sigma*sigma))/len(self.data))
    
    # def set_bkmapping(self,wtmodel):
        
        
#     def set_bkmapping(self,wtmodel):
        
#         class HybridSum(maps.IdentityMap):
#             """
#             Apply the model sum of Hybrid Decomposition
#             .. math::

#                 \rho_t = 1/(1/\rho_wt + 1/\rho_bk)
#             """

#             def __init__(self, mesh=None, nP=None, wt_val=None, ind_active_wt=None, **kwargs):
#                 self.wt_val=wt_val
#                 self.ind_active_wt=ind_active_wt
#                 super(HybridSum, self).__init__(mesh=mesh, nP=nP, **kwargs)

#             def _transform(self, m):
#                 cond = np.exp(-mkvc(m))
#                 cond_wt=1/self.wt_val
#                 cond[self.ind_active_wt]+=cond_wt
#                 return 1/cond

#             def deriv(self, m):#, v=None):
#                 mkvc_m=mkvc(m)
#                 cond_bk = np.exp(-mkvc_m)
#                 cond_wt = 1/self.wt_val
#                 cond_t = cond_bk.copy()
#                 cond_t[self.ind_active_wt]+=cond_wt
# #                return sdiag(cond_bk/(cond_t**2))
#                 return sdiag(cond_bk/(cond_t))

#             def inverse(self, m):
#                 cond = 1/mkvc(m)
#                 cond_wt = 1/self.wt_val
#                 cond[self.ind_active_wt]-=cond_wt
#                 return np.log(1/cond)


        # val_wt=np.min(wtmodel)
        # ind_active_wt=(val_wt==wtmodel)
        # nC=self.mesh.nC
        
        # HS=HybridSum(self.mesh,nP=nC,wt_val=val_wt,ind_active_wt=ind_active_wt)
        # mapping= HS
        
        # return mapping
    def set_bkmapping(self,wtmodel):
        
        class HybridSum(maps.IdentityMap):
            """
            Apply the model sum of Hybrid Decomposition
            .. math::

                \rho_t = 1/(1/\rho_wt + 1/\rho_bk)
            """

            def __init__(self, mesh=None, nP=None, wt_val=None, ind_active_wt=None, **kwargs):
                self.wt_val=wt_val
                self.ind_active_wt=ind_active_wt
                super(HybridSum, self).__init__(mesh=mesh, nP=nP, **kwargs)

            def _transform(self, m):
                cond = np.exp(-mkvc(m))
                cond_wt=1/self.wt_val
                cond[self.ind_active_wt]+=cond_wt
                return 1/cond

            def deriv(self, m):#, v=None):
                mkvc_m=mkvc(m)
                cond_bk = np.exp(-mkvc_m)
                cond_wt = 1/self.wt_val
                cond_t = cond_bk.copy()
                cond_t[self.ind_active_wt]+=cond_wt
                return sdiag(cond_bk/(cond_t)**2)#sdiag(cond_bk/(cond_t**2))

            def inverse(self, m):
                cond = 1/mkvc(m)
                cond_wt = 1/self.wt_val
                cond[self.ind_active_wt]-=cond_wt
                return np.log(1/cond)

        val_wt=np.min(wtmodel)
        ind_active_wt=(val_wt==wtmodel)
        nC=self.mesh.nC
        
        HS=HybridSum(self.mesh,nP=nC,wt_val=val_wt,ind_active_wt=ind_active_wt)
        mapping= HS        
        return mapping

    def invert(self,W=[]):
        #Data Misfit        
        dmis = data_misfit.L2DataMisfit(simulation=self.simulation, data=self.data_object)

        #Initial and reference models
        model_ini = np.r_[self.model_ini]
        model_ref= np.r_[self.model_ref]

        #Setting Regularization  
        if(len(W)>0):
            reg = regularization.WeightedLeastSquares(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x,weights=W,reference_model=np.log(model_ref))
        else:
            reg = regularization.WeightedLeastSquares(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x,reference_model=np.log(model_ref))

        reg.referenced_model_in_smooth=True
        #reg.mrefInSmooth = True #Forcing that mref must be used
        self.reg=reg
        opt = optimization.InexactGaussNewton(maxIter=1, maxIterCG=300)
        update_sensitivity_weights = directives.UpdateSensitivityWeights()   

        # Schedule for beta updating during inversion
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=self.beta)
        beta_schedule = directives.BetaSchedule(coolingFactor=1.0, coolingRate=1.0)#coolingFactor=self.coolingFactor, coolingRate=self.coolingRate)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=0.1)

        #inversion object
        self.directives_list = [
             update_sensitivity_weights,
             starting_beta,
             beta_schedule,
             target_misfit,
         ]
        self.inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

        #Forcing Smooth Reg to use the reference model.
        #self.reg.objfcts[1].mrefInSmooth=True
        #self.reg.objfcts[1].referenced_model_in_smooth=True
        #self.reg.objfcts[1].reference_model=self.reg.objfcts[0].reference_model #forcing that mref must be used in smoothness

        #Run Inversion
        inv = inversion.BaseInversion(self.inv_prob, directiveList=self.directives_list)
        recovered_model = inv.run(-np.log(model_ini))
        
        return np.exp(-recovered_model)


    def invert_step(self,W=[]):
        #Data Misfit        
        dmis = data_misfit.L2DataMisfit(simulation=self.simulation, data=self.data_object)

        #Initial and reference models
        model_ini = np.r_[self.model_ini]
        model_ref= np.r_[self.model_ref]

        #Setting Regularization  
        if(len(W)>0):          
            reg = regularization.WeightedLeastSquares(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x,weights=W,reference_model=-np.log(model_ref))
        else:
            reg = regularization.WeightedLeastSquares(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x,reference_model=-np.log(model_ref))

        #reg.mrefInSmooth = True #Forcing that mref must be used
        #reg.referenced_model_in_smooth=True
        self.reg=reg
        opt = optimization.InexactGaussNewton(maxIter=1, maxIterCG=300)
        update_sensitivity_weights = directives.UpdateSensitivityWeights()   

        # Schedule for beta updating during inversion
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=self.beta)
        beta_schedule = directives.BetaSchedule(coolingFactor=1.0, coolingRate=1.0)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=0.1)

        #inversion object
        self.directives_list = [
             update_sensitivity_weights,
             starting_beta,
             beta_schedule,
             target_misfit,
         ]
        self.inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

        #Forcing Reg to use the reference model.
        #self.reg.objfcts[1].mrefInSmooth=True
        #self.reg.objfcts[1].referenced_model_in_smooth=True
        #self.reg.objfcts[1].reference_model=self.reg.objfcts[0].reference_model

        #Run Inversion
        inv = inversion.BaseInversion(self.inv_prob, directiveList=self.directives_list)
        recovered_model = inv.run(-np.log(model_ini))
        
        self.iterindex+=1
        return np.exp(-recovered_model)

    def invert_step2(self,modelwt,magic_lmd,W=[]):
        
        #create data object
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)

        #setting a mesh
        layer_thicknesses = np.r_[self.thk[:-1]]#[:]#,[self.thk[-1]]])
        self.mesh = discretize.TensorMesh([np.r_[layer_thicknesses, layer_thicknesses[-1]]], "0")#[(np.r_[layer_thicknesses])], "0")  

        model_ini = np.r_[self.model_ini]

        #Setting foward modeling        
        mapping=self.set_bkmapping(modelwt)
        #model_map = maps.IdentityMap(nP=len(model_ini)) * maps.ExpMap()
        simulation = NSEM.Simulation1DRecursive(
                survey=self.survey,
                sigmaMap=mapping,#maps.ExpMap(nP=len(self.thk)),
                thicknesses=self.thk[:-1]
#                mapping=
            )
#rhoMap=mapping,
        #Inversion settings 
        model_ref= np.r_[self.model_ref]#np.concatenate([self.model_ref,[self.model_ref[-1]]])
        dmis = data_misfit.L2DataMisfit(simulation=simulation, data=self.data_object)

        #Setting Regularization  
        if(len(W)>0):          
            reg = regularization.WeightedLeastSquares(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x,reference_model=np.log(model_ref),weights=W)
        else:
            reg = regularization.WeightedLeastSquares(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x,reference_model=np.log(model_ref))

        reg.reference_model_in_smooth = True
        #reg.mrefInSmooth = True
        self.reg=reg
        opt = optimization.InexactGaussNewton(maxIter=10, maxIterCG=300)
        update_sensitivity_weights = directives.UpdateSensitivityWeights()   

        # Schedule for beta updating during inversion
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=self.beta)
        beta_schedule = directives.BetaSchedule(coolingFactor=1.0, coolingRate=1.0)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=0.1)


        #inversion object
        self.directives_list = [
            update_sensitivity_weights,
            starting_beta,
            beta_schedule,
            target_misfit,
        ]
        self.inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

        #Forcing Smooth Reg to use the reference model.
        self.reg.objfcts[1].reference_model_in_smooth=True
        self.reg.objfcts[1].reference_model=self.reg.objfcts[0].reference_model

        #Run Inversion
        inv = inversion.BaseInversion(self.inv_prob, directiveList=self.directives_list)
        recovered_model = inv.run(np.log(model_ini))#np.log(
            
        self.iterindex+=1
        return np.exp(-recovered_model)



class dc_class_1d:
    def __init__(self,a=[],b=[],m=[],n=[]):
        self.a=a
        self.b=b
        self.n=n
        self.m=m
        self.data=[]
        self.sigma=[]
        self.background_res=[]
        self.background_thk=[]
        self.background_Z=[]

        #regulation schedule for inversion
        self.iterindex=0
        self.collingRate=1
        self.beta=1
        self.collingFactor=1
        self.alpha_x=1
        self.alpha_y=1
        self.alpha_z=1
        self.alpha_s=0.001

        source_list = []
        for i in np.arange(len(self.a)):
            A_location = np.r_[self.a[i], 0.0, 0.0]
            B_location = np.r_[self.b[i], 0.0, 0.0]
            M_location = np.r_[self.m[i], 0.0, 0.0]
            N_location = np.r_[self.n[i], 0.0, 0.0]    
            receiver_list = dc.receivers.Dipole(M_location, N_location)
            receiver_list = [receiver_list]
            source_list.append(dc.sources.Dipole(receiver_list, A_location, B_location))
        self.survey = dc.Survey(source_list)
                   

    def set_bkmapping(self,wtmodel):
        
        class HybridSum(maps.IdentityMap):
            """
            Apply the model sum of Hybrid Decomposition
            .. math::

                \rho_t = 1/(1/\rho_wt + 1/\rho_bk)
            """

            def __init__(self, mesh=None, nP=None, wt_val=None, ind_active_wt=None, **kwargs):
                self.wt_val=wt_val
                self.ind_active_wt=ind_active_wt
                super(HybridSum, self).__init__(mesh=mesh, nP=nP, **kwargs)

            def _transform(self, m):
                cond = np.exp(-mkvc(m))
                cond_wt=1/self.wt_val
                cond[self.ind_active_wt]+=cond_wt
                return 1/cond

            def deriv(self, m):#, v=None):
                mkvc_m=mkvc(m)
                cond_bk = np.exp(-mkvc_m)
                cond_wt = 1/self.wt_val
                cond_t = cond_bk.copy()
                cond_t[self.ind_active_wt]+=cond_wt
#                return sdiag(cond_bk/(cond_t**2))
                return sdiag(cond_bk/(cond_t))

            def inverse(self, m):
                cond = 1/mkvc(m)
                cond_wt = 1/self.wt_val
                cond[self.ind_active_wt]-=cond_wt
                return np.log(1/cond)


        val_wt=np.min(wtmodel)
        ind_active_wt=(val_wt==wtmodel)
        nC=self.mesh.nC
        
        HS=HybridSum(self.mesh,nP=nC,wt_val=val_wt,ind_active_wt=ind_active_wt)
        mapping= HS
        
        return mapping


    def set_modelref(self,model):
        self.model_ref = model.copy()

    def set_modelini(self,model):
        self.model_ini = model.copy()

    def set_model(self,res):
        self.model=res

    def save_data(self,filename):
        fid=open(filename,'wt')
        for i in np.arange(len(self.a)):
            fid.write(str(self.a[i])+' '+str(self.b[i])+' '+str(self.m[i])+' '+str(self.n[i])+' '+str(self.data[i])+'\n')
        fid.close()            

    def set_data(self,arr):
        self.a=arr[:,0]
        self.b=arr[:,1]
        self.m=arr[:,2]
        self.n=arr[:,3]
        self.data=arr[:,4]
        self.sigma=0.05*np.abs(self.data)
        source_list = []
        for i in np.arange(len(self.a)):
            A_location = np.r_[self.a[i], 0.0, 0.0]
            B_location = np.r_[self.b[i], 0.0, 0.0]
            M_location = np.r_[self.m[i], 0.0, 0.0]
            N_location = np.r_[self.n[i], 0.0, 0.0]    
            if self.m[i] == self.n[i]:
                receiver_list = dc.receivers.Pole(M_location,data_type="apparent_resistivity")            
            else:            
                receiver_list = dc.receivers.Dipole(M_location, N_location,data_type="apparent_resistivity")
        
            receiver_list = [receiver_list]
            
            if self.a[i] == self.b[i]:
                source_list.append(dc.sources.Pole(receiver_list, A_location,data_type="apparent_resistivity"))
            else:
                source_list.append(dc.sources.Dipole(receiver_list, A_location, B_location,data_type="apparent_resistivity"))
        self.survey = dc.Survey(source_list)

    def set_mesh(self,m,ind_active=[]):
        self.thk=m.dY
        self.model=np.zeros(len(m.dY))

    def set_mesh_dY(self,dY):
        self.thk=dY
        self.model=np.zeros(len(dY))

    def set_halfmodel(self,r0):
        self.model=np.ones(len(self.thk))*r0

    def plot_pseudo(self,obs=[],resp1=[],resp2=[]):
        #if plt.gcf().get_axes():
        plt.plot(np.abs(self.a-self.b)/2,obs,'k')
        plt.plot(np.abs(self.a-self.b)/2,resp1,'r')
        plt.plot(np.abs(self.a-self.b)/2,resp2,'b')
#            plt.legend(['observed','best model resp.'])
#        else:
        plt.plot(np.abs(self.a-self.b)/2,self.fwd(),'k')
        plt.legend(['observed','best model resp.','mean model resp'])
        plt.ylabel('Resistivity(ohm-m)')
        plt.xlabel('AB/2(m)')
        plt.title('DC responses')
        
    def read_data(self,filename):
        arr=np.loadtxt(filename)
        self.a=arr[:,0]
        self.b=arr[:,1]
        self.m=arr[:,2]
        self.n=arr[:,3]
        self.data=arr[:,4]
        self.sigma=0.05*np.abs(self.data)
        return arr

    def fwd(self):        
        #create data object
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)

        #setting a mesh
        layer_thicknesses = np.r_[self.thk[:-1]]#[:]#,[self.thk[-1]]])

        self.mesh = discretize.TensorMesh([np.r_[layer_thicknesses, layer_thicknesses[-1]]], "0")#[(np.r_[layer_thicknesses])], "0")  

        model_map = maps.IdentityMap(nP=len(self.model)) * maps.ExpMap()
        simulation = dc.simulation_1d.Simulation1DLayers(
            survey=self.survey,
            rhoMap=model_map,
            thicknesses=layer_thicknesses,
#            data_type="apparent_resistivity",
        )

        dpred = simulation.dpred(np.log(self.model))
        return dpred

    def loglike(self):
        x=self.fwd()
        d=self.data
        sigma=self.sigma
        return -0.5*(np.sum((np.array(x-d)**2)/(sigma*sigma))/len(self.data))
    
    def invert(self,W=[]):

        #create data object
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)

        #setting a mesh
        layer_thicknesses = np.r_[self.thk[:-1]]
        self.mesh = discretize.TensorMesh([np.r_[layer_thicknesses, layer_thicknesses[-1]]], "0")

        model_ini = np.r_[self.model_ini]

        #Setting foward modeling        
        mapping = maps.IdentityMap(nP=len(self.model)) * maps.ExpMap()
        simulation = dc.simulation_1d.Simulation1DLayers(
            survey=self.survey,
            rhoMap=mapping,
            thicknesses=layer_thicknesses
        )

        #Inversion settings 
        model_ref= np.r_[self.model_ref]
        dmis = data_misfit.L2DataMisfit(simulation=simulation, data=self.data_object)

        #Setting Regularization  
        if(len(W)>0):          
            reg = regularization.WeightedLeastSquares(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x,reference_model=np.log(model_ref),weights=W)
        else:
            reg = regularization.WeightedLeastSquares(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x,reference_model=np.log(model_ref))
        
        reg.reference_model_in_smooth = True
        #reg.mrefInSmooth = True
        self.reg=reg
        opt = optimization.InexactGaussNewton(maxIter=10, maxIterCG=300)
#        update_sensitivity_weights = directives.UpdateSensitivityWeights()   

        # Schedule for beta updating during inversion
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=self.beta)
        beta_schedule = directives.BetaSchedule(coolingFactor=1.0, coolingRate=1.0)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=0.01)


        #inversion object
#        self.directives_list = [
#            update_sensitivity_weights,
#            starting_beta,
#            beta_schedule,
#            target_misfit,
#        ]
        self.directives_list = [
            starting_beta,
            beta_schedule,
            target_misfit,
        ]
 
        
        self.inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt) #+reg2

        #Forcing Smooth Reg to use the reference model.
        self.reg.objfcts[1].reference_model_in_smooth=True
        self.reg.objfcts[1].reference_model=self.reg.objfcts[0].reference_model

        #Run Inversion
        inv = inversion.BaseInversion(self.inv_prob, directiveList=self.directives_list)
        recovered_model = inv.run(np.log(model_ini))
            
        return np.exp(recovered_model)

    def invert_step(self,W=[]):
        
        #create data object
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)

        #setting a mesh
        layer_thicknesses = np.r_[self.thk[:-1]]#[:]#,[self.thk[-1]]])
        self.mesh = discretize.TensorMesh([np.r_[layer_thicknesses, layer_thicknesses[-1]]], "0")#[(np.r_[layer_thicknesses])], "0")  

        model_ini = np.r_[self.model_ini]

        #Setting foward modeling        
        model_map = maps.IdentityMap(nP=len(model_ini)) * maps.ExpMap()
        simulation = dc.simulation_1d.Simulation1DLayers(
            survey=self.survey,
            rhoMap=model_map,
            thicknesses=layer_thicknesses
        )

        #Inversion settings 
        model_ref= np.r_[self.model_ref]#np.concatenate([self.model_ref,[self.model_ref[-1]]])
        dmis = data_misfit.L2DataMisfit(simulation=simulation, data=self.data_object)

        #Setting Regularization  
        if(len(W)>0):          
            reg = regularization.WeightedLeastSquares(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x,reference_model=np.log(model_ref),weights=W)
        else:
            reg = regularization.WeightedLeastSquares(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x,reference_model=np.log(model_ref))

        reg.reference_model_in_smooth = True
        #reg.mrefInSmooth = True
        self.reg=reg
        opt = optimization.InexactGaussNewton(maxIter=1, maxIterCG=300)
        update_sensitivity_weights = directives.UpdateSensitivityWeights()   

        # Schedule for beta updating during inversion
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=self.beta)
        beta_schedule = directives.BetaSchedule(coolingFactor=1.0, coolingRate=1.0)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=0.1)


        #inversion object
        self.directives_list = [
            update_sensitivity_weights,
            starting_beta,
            beta_schedule,
            target_misfit,
        ]
        self.inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

        #Forcing Smooth Reg to use the reference model.
        self.reg.objfcts[1].reference_model_in_smooth=True
        self.reg.objfcts[1].reference_model=self.reg.objfcts[0].reference_model

        #Run Inversion
        inv = inversion.BaseInversion(self.inv_prob, directiveList=self.directives_list)
        recovered_model = inv.run(np.log(model_ini))
            
        self.iterindex+=1
        return np.exp(recovered_model)

    def invert_step2(self,modelwt,W=[],magic_lmd=[]):
        
        #create data object
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)

        #setting a mesh
        layer_thicknesses = np.r_[self.thk[:-1]]#[:]#,[self.thk[-1]]])
        self.mesh = discretize.TensorMesh([np.r_[layer_thicknesses, layer_thicknesses[-1]]], "0")#[(np.r_[layer_thicknesses])], "0")  

        model_ini = np.r_[self.model_ini]

        #Setting foward modeling        
        mapping=self.set_bkmapping(modelwt)
        simulation = dc.simulation_1d.Simulation1DLayers(
            survey=self.survey,
            rhoMap=mapping,
            thicknesses=layer_thicknesses
        )

        #Inversion settings 
        model_ref= np.r_[self.model_ref]#np.concatenate([self.model_ref,[self.model_ref[-1]]])
        dmis = data_misfit.L2DataMisfit(simulation=simulation, data=self.data_object)

        #Setting Regularization  
        if(len(W)>0):          
            reg = regularization.WeightedLeastSquares(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x,reference_model=np.log(model_ref),weights=W)
        else:
            reg = regularization.WeightedLeastSquares(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x,reference_model=np.log(model_ref))
        
        index_wt=(modelwt==np.min(modelwt))
        modelref2=model_ini.copy()
        N=np.sum(index_wt)
        W2=np.zeros(len(modelref2))
        if isinstance(magic_lmd,list):
            reg = reg 
        else:
            #print("magic_lmd:",magic_lmd)
            #magic_lmd=2.0
#            W2[index_wt]=magic_lmd/(1*np.abs(modelref2[index_wt]/4)*(N**0.5))
#            modelref2[index_wt]=np.min(modelwt)/magic_lmd
#            reg2 = regularization.WeightedLeastSquares(self.mesh, alpha_s=1.0, alpha_x=0,reference_model=np.log(modelref2),weights=W2)
#            reg = reg + reg2
#            reg2.reference_model_in_smooth = True
            None

        reg.reference_model_in_smooth = True
        #reg.mrefInSmooth = True
        self.reg=reg
        opt = optimization.InexactGaussNewton(maxIter=1, maxIterCG=300)
        #update_sensitivity_weights = directives.UpdateSensitivityWeights()   

        # Schedule for beta updating during inversion
#        starting_beta=self.beta
#        if self.iterindex==0:
        self.starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=self.beta)        
                        
        starting_beta=self.starting_beta
        beta_schedule = directives.BetaSchedule(coolingFactor=1.0, coolingRate=1.0)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=0.01)


        #inversion object
        self.directives_list = [
#            update_sensitivity_weights,
            starting_beta,
            beta_schedule,
            target_misfit,
        ]
        self.inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt) #+reg2

        #Forcing Smooth Reg to use the reference model.
        self.reg.objfcts[1].reference_model_in_smooth=True
        self.reg.objfcts[1].reference_model=self.reg.objfcts[0].reference_model

        #Run Inversion
        inv = inversion.BaseInversion(self.inv_prob, directiveList=self.directives_list)
        self.inv=inv
        recovered_model = inv.run(np.log(model_ini))
            
        self.iterindex+=1
        return np.exp(recovered_model)


class dc_class_1d_par:
    def __init__(self,a=[],b=[],m=[],n=[]):
        self.a=a
        self.b=b
        self.n=n
        self.m=m
        self.data=[]
        self.sigma=[]
##        self.background_res=[]
#        self.background_thk=[]
#        self.background_Z=[]

        #regulation schedule for inversion
        self.iterindex=0
        self.collingRate=1
        self.beta=1
        self.collingFactor=1
        self.alpha_x=1
        self.alpha_y=1
        self.alpha_z=1
        self.alpha_s=0.001

        source_list = []
        for i in np.arange(len(self.a)):
            A_location = np.r_[self.a[i], 0.0, 0.0]
            B_location = np.r_[self.b[i], 0.0, 0.0]
            M_location = np.r_[self.m[i], 0.0, 0.0]
            N_location = np.r_[self.n[i], 0.0, 0.0]    
            receiver_list = dc.receivers.Dipole(M_location, N_location)
            receiver_list = [receiver_list]
            source_list.append(dc.sources.Dipole(receiver_list, A_location, B_location))
        self.survey = dc.Survey(source_list)
                   

    def set_bkmapping(self,wtmodel):
        class mapping_rho(maps.IdentityMap):
            def __init__(self,R0, mesh=None, nP=None, **kwargs):
                super(mapping_rho, self).__init__(mesh=mesh, nP=nP, **kwargs)
                self.R0=R0
        
            def _transform(self, m):
                R1=np.exp(m[0])
                Rwt=1/(1/self.R0+1/R1)
                R2=np.exp(m[1])
                rho=np.array([R1,Rwt,R2])
                return rho
        
            def deriv(self, m):#, v=None):
                R1=np.exp(m[0])
                Rwt=1/(1/self.R0+1/R1)
                R2=np.exp(m[1])
                grad=np.zeros([2,3])
                grad[0,0]=R1
                grad[0,1]=Rwt**2/R1
                grad[0,2]=0
                grad[1,0]=0
                grad[1,1]=0
                grad[1,2]=R2
                return grad.T
        
            def inverse(self, m):
                R1=m[0]
                R2=m[2]
                logrho=np.array([np.log(R1),np.log(R2)])
                return logrho    
        
        class mapping_thk(maps.IdentityMap):
            def __init__(self, Zwt, mesh=None, nP=None, **kwargs):
                super(mapping_thk, self).__init__(mesh=mesh, nP=nP, **kwargs)
                self.Zwt=Zwt
        
            def _transform(self, m):
                if m[0]>self.Zwt:
                    thk=np.array([self.Zwt,m[0]-self.Zwt])
                else:
                    thk=np.array([self.Zwt,0])                    
                return thk 
        
            def deriv(self, m):#, v=None):
                grad=np.zeros([1,2])
                grad[0,0]=0
                grad[0,1]=1
                return grad.T
        
            def inverse(self, m):
                thk=np.array([m[1]+self.Zwt])
                return thk
        vals=wtmodel
        R0=np.min(vals)
        Z=0
        R1=vals[0]
        for i in np.arange(len(vals)):
            if vals[i] == R1:
                Z+=self.thk[i]
            else:
                break
                
            
        mapping=[mapping_rho(R0),mapping_thk(Z)]
        return mapping

    def set_modelref(self,model):
        vals = model.copy()
        Z=np.sum(self.thk[vals == vals[0]])
        self.model_ref=np.array([np.log(vals[0]),np.log(vals[-1]),Z])
        

    def set_modelini(self,model):
        vals = model.copy()
        Z=np.sum(self.thk[vals == vals[0]])
        self.model_ini=np.array([np.log(vals[0]),np.log(vals[-1]),Z])


    def set_model(self,res):
        self.model=res
 #       vals = res.copy()
 #       Z=np.sum(self.thk[vals == vals[0]])
 #       self.model=np.array([np.log(vals[0]),np.log(vals[-1]),Z])



    def set_data(self,arr):
        self.a=arr[:,0]
        self.b=arr[:,1]
        self.m=arr[:,2]
        self.n=arr[:,3]
        self.data=arr[:,4]
        self.sigma=0.05*np.abs(self.data)
        source_list = []
        for i in np.arange(len(self.a)):
            A_location = np.r_[self.a[i], 0.0, 0.0]
            B_location = np.r_[self.b[i], 0.0, 0.0]
            M_location = np.r_[self.m[i], 0.0, 0.0]
            N_location = np.r_[self.n[i], 0.0, 0.0]    
            if self.m[i] == self.n[i]:
                receiver_list = dc.receivers.Pole(M_location,data_type="apparent_resistivity")            
            else:            
                receiver_list = dc.receivers.Dipole(M_location, N_location,data_type="apparent_resistivity")
        
            receiver_list = [receiver_list]
            
            if self.a[i] == self.b[i]:
                source_list.append(dc.sources.Pole(receiver_list, A_location,data_type="apparent_resistivity"))
            else:
                source_list.append(dc.sources.Dipole(receiver_list, A_location, B_location,data_type="apparent_resistivity"))
        self.survey = dc.Survey(source_list)

    def set_mesh(self,m,ind_active=[]):
        self.thk=m.dY
        self.model=np.zeros(len(m.dY))

    def set_mesh_dY(self,dY):
        self.thk=dY
        self.model=np.zeros(len(dY))

        
    def fwd(self):        
        #create data object
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)

        #setting a mesh
        layer_thicknesses = np.r_[self.thk[:-1]]#[:]#,[self.thk[-1]]])

        self.mesh = discretize.TensorMesh([np.r_[layer_thicknesses, layer_thicknesses[-1]]], "0")#[(np.r_[layer_thicknesses])], "0")  

        model_map = maps.IdentityMap(nP=len(self.model)) * maps.ExpMap()
        simulation = dc.simulation_1d.Simulation1DLayers(
            survey=self.survey,
            rhoMap=model_map,
            thicknesses=layer_thicknesses,
#            data_type="apparent_resistivity",
        )

        dpred = simulation.dpred(np.log(self.model))
        return dpred
    
    def invert(self,W=[]):
        #create data object
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)
        model_ini = self.model_ini
        model_ref= self.model_ini
        Nl=int((len(model_ini)+1)/2)
        

        #Setting foward modeling        
        
        wire_map = maps.Wires(("rho", Nl), ("thk", Nl-1))   #Clase de alambrado

        sim=dc.Simulation1DLayers(
            survey=self.survey,
            rhoMap=maps.ExpMap(nP=Nl)*wire_map.rho,                   #Aqui incluye la funcion de mapeo de 
                                              #las resistividades de las capas
            thicknessesMap=maps.ExpMap(nP=Nl-1)*wire_map.thk)           #Aqui incluye la funcion de mapeo 
                                              #de los espesores de las capas


        #Inversion settings 
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=self.data_object)

        #model regularization
        mesh_r=discretize.TensorMesh([Nl])
        reg_rho=regularization.WeightedLeastSquares(
            mesh_r, alpha_s=0.0001, alpha_x=0, mapping=wire_map.rho)
        
        mesh_t=discretize.TensorMesh([Nl-1])
        reg_thk=regularization.WeightedLeastSquares(
            mesh_t, alpha_s=1e-12, alpha_x=0, mapping=wire_map.thk)

        reg=reg_rho+reg_thk
        self.reg=reg

    
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=0.001)        
                        
        starting_beta = starting_beta
        beta_schedule = directives.BetaSchedule(coolingFactor=1.0, coolingRate=1.0)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=0.1)
        self.directives_list = [
            starting_beta,
            beta_schedule,
            target_misfit,
        ]

        #Run Inversion
        opt = optimization.InexactGaussNewton(maxIter=30, maxIterCG=300)
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
        inv = inversion.BaseInversion(inv_prob, directiveList=self.directives_list)


        self.inv=inv
        model_ini = np.array(np.concatenate([[np.log(100),np.log(100)],[50]]))

        imodel = inv.run(model_ini)
        Z=np.concatenate([[0],np.cumsum(self.thk)])
        recovered_model=np.zeros(len(self.thk))
        for i in np.arange(len(self.thk)):
            if Z[i]>=imodel[2]:            
                recovered_model[i] = imodel[1]             
            else:
                recovered_model[i] = imodel[0]             
        self.iterindex+=1
        return np.exp(recovered_model)

    def invert_step(self,W=[]):
        None
        
    def invert_step2(self,modelwt,W=[],magic_lmd=[]):
        
        #create data object
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)
        model_ini = self.model_ini
        model_ref= self.model_ref
        Nl=int((len(model_ini)+1)/2)
        

        #Setting foward modeling        
        mapping=self.set_bkmapping(modelwt)
        self.mapping2=mapping
        wire_map = maps.Wires(("rho", Nl), ("thk", Nl-1))   #Clase de alambrado

        sim=dc.Simulation1DLayers(
            survey=self.survey,
            rhoMap=mapping[0]*wire_map.rho,                   #Aqui incluye la funcion de mapeo de 
                                              #las resistividades de las capas
            thicknessesMap=mapping[1]*wire_map.thk)           #Aqui incluye la funcion de mapeo 
                                              #de los espesores de las capas

        self.sim2=sim
        #Inversion settings 
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=self.data_object)

        #model regularization
        mesh_r=discretize.TensorMesh([Nl])
        reg_rho=regularization.WeightedLeastSquares(
            mesh_r, alpha_s=1, alpha_x=0, mapping=wire_map.rho)
        
        mesh_t=discretize.TensorMesh([Nl-1])
        reg_thk=regularization.WeightedLeastSquares(
            mesh_t, alpha_s=1, alpha_x=0, mapping=wire_map.thk)
#1e-12
        reg=reg_rho+reg_thk
        self.reg=reg

    
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=0.01)        
                        
        starting_beta = starting_beta
        beta_schedule = directives.BetaSchedule(coolingFactor=1.0, coolingRate=1.0)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=0.1)
        self.directives_list = [
            starting_beta,
            beta_schedule,
            target_misfit,
        ]

        #Run Inversion
        opt = optimization.InexactGaussNewton(maxIter=10, maxIterCG=300)
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
        inv = inversion.BaseInversion(inv_prob, directiveList=self.directives_list)


        self.inv=inv
#        model_ini = np.array(np.concatenate([[np.log(100),np.log(100)],[50]]))

        imodel = inv.run(model_ini)
        #print("imodel:",imodel)
        Z=np.concatenate([[0],np.cumsum(self.thk)])
        recovered_model=np.zeros(len(self.thk))
        for i in np.arange(len(self.thk)):
            if Z[i]>=imodel[2]:            
                recovered_model[i] = imodel[1]             
            else:
                recovered_model[i] = imodel[0]             
        self.iterindex+=1
        return np.exp(recovered_model)


class dc_class_1d_par2:
    def __init__(self,a=[],b=[],m=[],n=[]):
        self.a=a
        self.b=b
        self.n=n
        self.m=m
        self.data=[]
        self.sigma=[]

        #regulation schedule for inversion
        self.iterindex=0
        self.collingRate=1
        self.beta=1
        self.collingFactor=1
        self.alpha_x=1
        self.alpha_y=1
        self.alpha_z=1
        self.alpha_s=0.001

        source_list = []
        for i in np.arange(len(self.a)):
            A_location = np.r_[self.a[i], 0.0, 0.0]
            B_location = np.r_[self.b[i], 0.0, 0.0]
            M_location = np.r_[self.m[i], 0.0, 0.0]
            N_location = np.r_[self.n[i], 0.0, 0.0]    
            receiver_list = dc.receivers.Dipole(M_location, N_location)
            receiver_list = [receiver_list]
            source_list.append(dc.sources.Dipole(receiver_list, A_location, B_location))
        self.survey = dc.Survey(source_list)
                   

    def set_bkmapping(self,wtmodel):
        class mapping_rho(maps.IdentityMap):
            def __init__(self,R0, mesh=None, nP=None, **kwargs):
                super(mapping_rho, self).__init__(mesh=mesh, nP=nP, **kwargs)
                self.R0=R0
        
            def _transform(self, m):
                R1=np.exp(m[0])
                Rwt=1/(1/self.R0+1/R1)
                R2=np.exp(m[1])
                rho=np.array([R1,Rwt,R2])
                return rho
        
            def deriv(self, m):#, v=None):
                R1=np.exp(m[0])
                Rwt=1/(1/self.R0+1/R1)
                R2=np.exp(m[1])
                grad=np.zeros([2,3])
                grad[0,0]=R1
                grad[0,1]=Rwt**2/R1
                grad[0,2]=0
                grad[1,0]=0
                grad[1,1]=0
                grad[1,2]=R2
                return grad.T
        
            def inverse(self, m):
                R1=m[0]
                R2=m[2]
                logrho=np.array([np.log(R1),np.log(R2)])
                return logrho    
        
        # class mapping_thk(maps.IdentityMap):
        #     def __init__(self, Zwt, Zbs, mesh=None, nP=None, **kwargs):
        #         super(mapping_thk, self).__init__(mesh=mesh, nP=nP, **kwargs)
        #         self.Zwt=Zwt
        #         self.Zbs=Zbs
        
        #     def _transform(self, m):
        #         if m[0]>self.Zwt:
        #             thk=np.array([self.Zwt,m[0]-self.Zwt])
        #         else:
        #             thk=np.array([self.Zwt,0])                    
        #         return thk 
        
        #     def deriv(self, m):#, v=None):
        #         grad=np.zeros([1,2])
        #         grad[0,0]=0
        #         grad[0,1]=1
        #         return grad.T
        
        #     def inverse(self, m):
        #         thk=np.array([m[1]+self.Zwt])
        #         return thk
        vals=wtmodel
        R0=np.min(vals)
#        Z=0
#        R1=vals[0]
#        for i in np.arange(len(vals)):
#            if vals[i] == R1:
#                Z+=self.thk[i]
#            else:
#                break
                
            
        mapping=mapping_rho(R0)
        return mapping

    def set_modelref(self,model):
        vals = model.copy()
#        Z=np.sum(self.thk[vals == vals[0]])
        self.model_ref=np.array([np.log(vals[0]),np.log(vals[-1])])
        

    def set_modelini(self,model):
        vals = model.copy()
#        Z=np.sum(self.thk[vals == vals[0]])
        self.model_ini=np.array([np.log(vals[0]),np.log(vals[-1])])


    def set_model(self,res):
        self.model=res
 #       vals = res.copy()
 #       Z=np.sum(self.thk[vals == vals[0]])
 #       self.model=np.array([np.log(vals[0]),np.log(vals[-1]),Z])



    def set_data(self,arr):
        self.a=arr[:,0]
        self.b=arr[:,1]
        self.m=arr[:,2]
        self.n=arr[:,3]
        self.data=arr[:,4]
        self.sigma=0.05*np.abs(self.data)
        source_list = []
        for i in np.arange(len(self.a)):
            A_location = np.r_[self.a[i], 0.0, 0.0]
            B_location = np.r_[self.b[i], 0.0, 0.0]
            M_location = np.r_[self.m[i], 0.0, 0.0]
            N_location = np.r_[self.n[i], 0.0, 0.0]    
            if self.m[i] == self.n[i]:
                receiver_list = dc.receivers.Pole(M_location,data_type="apparent_resistivity")            
            else:            
                receiver_list = dc.receivers.Dipole(M_location, N_location,data_type="apparent_resistivity")
        
            receiver_list = [receiver_list]
            
            if self.a[i] == self.b[i]:
                source_list.append(dc.sources.Pole(receiver_list, A_location,data_type="apparent_resistivity"))
            else:
                source_list.append(dc.sources.Dipole(receiver_list, A_location, B_location,data_type="apparent_resistivity"))
        self.survey = dc.Survey(source_list)

    def set_mesh(self,m,ind_active=[]):
        self.thk=m.dY
        self.model=np.zeros(len(m.dY))

    def set_mesh_dY(self,dY):
        self.thk=dY
        self.model=np.zeros(len(dY))

        
    def fwd(self):        
        #create data object
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)

        #setting a mesh
        layer_thicknesses = np.r_[self.thk[:-1]]#[:]#,[self.thk[-1]]])

        self.mesh = discretize.TensorMesh([np.r_[layer_thicknesses, layer_thicknesses[-1]]], "0")#[(np.r_[layer_thicknesses])], "0")  

        model_map = maps.IdentityMap(nP=len(self.model)) * maps.ExpMap()
        simulation = dc.simulation_1d.Simulation1DLayers(
            survey=self.survey,
            rhoMap=model_map,
            thicknesses=layer_thicknesses,
#            data_type="apparent_resistivity",
        )

        dpred = simulation.dpred(np.log(self.model))
        return dpred
    
    def invert(self,W=[]):
        #create data object
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)
        model_ini = np.concatenate([self.model_ini,[np.sum(self.thk)/2]])
        model_ref= self.model_ini
        Nl=int((len(model_ini)+1)/2)
        

        #Setting foward modeling        
        
        wire_map = maps.Wires(("rho", Nl), ("thk", Nl-1))   #Clase de alambrado

        sim=dc.Simulation1DLayers(
            survey=self.survey,
            rhoMap=maps.ExpMap(nP=Nl)*wire_map.rho,                   #Aqui incluye la funcion de mapeo de 
                                              #las resistividades de las capas
            thicknessesMap=maps.ExpMap(nP=Nl-1)*wire_map.thk)           #Aqui incluye la funcion de mapeo 
                                              #de los espesores de las capas


        #Inversion settings 
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=self.data_object)

        #model regularization
        mesh_r=discretize.TensorMesh([Nl])
        reg_rho=regularization.WeightedLeastSquares(
            mesh_r, alpha_s=0.0001, alpha_x=0, mapping=wire_map.rho)
        
        mesh_t=discretize.TensorMesh([Nl-1])
        reg_thk=regularization.WeightedLeastSquares(
            mesh_t, alpha_s=1e-12, alpha_x=0, mapping=wire_map.thk)

        reg=reg_rho+reg_thk
        self.reg=reg

    
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=0.001)        
                        
        starting_beta = starting_beta
        beta_schedule = directives.BetaSchedule(coolingFactor=1.0, coolingRate=1.0)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=0.1)
        self.directives_list = [
            starting_beta,
            beta_schedule,
            target_misfit,
        ]

        #Run Inversion
        opt = optimization.InexactGaussNewton(maxIter=30, maxIterCG=300)
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
        inv = inversion.BaseInversion(inv_prob, directiveList=self.directives_list)


        self.inv=inv
        model_ini = np.array(np.concatenate([[np.log(100),np.log(100)],[50]]))

        imodel = inv.run(model_ini)
        Z=np.concatenate([[0],np.cumsum(self.thk)])
        recovered_model=np.zeros(len(self.thk))
        for i in np.arange(len(self.thk)):
            if Z[i]>=imodel[2]:            
                recovered_model[i] = imodel[1]             
            else:
                recovered_model[i] = imodel[0]             
        self.iterindex+=1
        return np.exp(recovered_model)

    def invert_step(self,W=[]):
        None

    def mod2par(self,wtmodel):
        vals=wtmodel
        R0=np.min(vals)
        index=(vals==R0)
        Z1=np.array(np.concatenate([[0],np.cumsum(self.thk[:-1])]))
        Z2=np.array(np.cumsum(self.thk))
        if np.sum(index)<=1:
            Zwt=Z1[index]
            Zbs=Z2[index]
        else:
            Zwt=Z1[index][0]
            Zbs=Z2[index][-1]
        pars=np.array([float(Zwt),float(Zbs),float(R0)])
        return pars

        
    def invert_step2(self,modelwt,W=[],magic_lmd=[]):

        pars=self.mod2par(modelwt)
        mod_thk=np.array([pars[0],pars[1]-pars[0]])
        
        #create data object
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)
        model_ini = self.model_ini
        model_ref= self.model_ref
        Nl=2#int((len(model_ini)+1)/2)

        #Setting foward modeling        
        mapping=self.set_bkmapping(modelwt)
        self.mapping2=mapping
#        wire_map = maps.Wires(("rho", Nl), ("thk", Nl-1))   #Clase de alambrado

        sim=dc.Simulation1DLayers(
            survey=self.survey,
            rhoMap=mapping,                   #Aqui incluye la funcion de mapeo de 
                                              #las resistividades de las capas
            thicknesses=mod_thk)
#            thicknessesMap=mapping[1]*wire_map.thk)           #Aqui incluye la funcion de mapeo 
                                              #de los espesores de las capas

        self.sim2=sim
        #Inversion settings 
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=self.data_object)

        #model regularization
        mesh_r=discretize.TensorMesh([Nl])
        reg_rho=regularization.WeightedLeastSquares(
            mesh_r, alpha_s=1, alpha_x=0)#, mapping=wire_map.rho)
        
#        mesh_t=discretize.TensorMesh([Nl-1])
#        reg_thk=regularization.WeightedLeastSquares(
#            mesh_t, alpha_s=1, alpha_x=0, mapping=wire_map.thk)
#1e-12
        reg=reg_rho#+reg_thk
        self.reg=reg

    
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=0.01)        
                        
        starting_beta = starting_beta
        beta_schedule = directives.BetaSchedule(coolingFactor=1.0, coolingRate=1.0)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=0.1)
        self.directives_list = [
            starting_beta,
            beta_schedule,
            target_misfit,
        ]

        #Run Inversion
        opt = optimization.InexactGaussNewton(maxIter=10, maxIterCG=300)
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
        inv = inversion.BaseInversion(inv_prob, directiveList=self.directives_list)


        self.inv=inv
#        model_ini = np.array(np.concatenate([[np.log(100),np.log(100)],[50]]))

        imodel = inv.run(model_ini)
        #print("imodel:",imodel)
        
        Z=np.concatenate([[0],np.cumsum(self.thk[:-1])])

        recovered_model=np.zeros(len(self.thk))
        for i in np.arange(len(self.thk)):
            if Z[i]>=pars[1]:            
                recovered_model[i] = imodel[1]             
            else:
                recovered_model[i] = imodel[0]             
        self.iterindex+=1
        return np.exp(recovered_model)


class tem_class_1d_par:
    def __init__(self,a=[],b=[],m=[],n=[]):
        self.a=a
        self.b=b
        self.n=n
        self.m=m
        self.data=[]
        self.sigma=[]
        #regulation schedule for inversion
        self.iterindex=0
        self.collingRate=1
        self.beta=1
        self.collingFactor=1
        self.alpha_x=1
        self.alpha_y=1
        self.alpha_z=1
        self.alpha_s=0.001

        source_list = []
        for i in np.arange(len(self.a)):
            A_location = np.r_[self.a[i], 0.0, 0.0]
            B_location = np.r_[self.b[i], 0.0, 0.0]
            M_location = np.r_[self.m[i], 0.0, 0.0]
            N_location = np.r_[self.n[i], 0.0, 0.0]    
            receiver_list = dc.receivers.Dipole(M_location, N_location)
            receiver_list = [receiver_list]
            source_list.append(dc.sources.Dipole(receiver_list, A_location, B_location))
        self.survey = dc.Survey(source_list)
                   

    def set_bkmapping(self,wtmodel):
        class mapping_rho(maps.IdentityMap):
            def __init__(self,R0, mesh=None, nP=None, **kwargs):
                super(mapping_rho, self).__init__(mesh=mesh, nP=nP, **kwargs)
                self.R0=R0
        
            def _transform(self, m):
                R1=np.exp(m[0])
                Rwt=1/self.R0+R1
                R2=np.exp(m[1])
                rho=np.array([R1,Rwt,R2])
                return rho
        
            def deriv(self, m):#, v=None):
                R1=np.exp(m[0])
                Rwt=1/self.R0+R1
                R2=np.exp(m[1])
                grad=np.zeros([2,3])
                grad[0,0]=R1
                grad[0,1]=R1
                grad[0,2]=0
                grad[1,0]=0
                grad[1,1]=0
                grad[1,2]=R2
                return grad.T
        
            def inverse(self, m):
                R1=m[0]
                R2=m[2]
                logrho=np.array([np.log(R1),np.log(R2)])
                return logrho    
        
 
        vals=wtmodel
        R0=np.min(vals)
        mapping=mapping_rho(R0)
        return mapping

    def set_modelref(self,model):
        vals = 1/model.copy()
        self.model_ref=np.array([np.log(vals[0]),np.log(vals[-1])])

    def set_modelini(self,model):
        vals = 1/model.copy()
        self.model_ini=np.array([np.log(vals[0]),np.log(vals[-1])])

    def set_model(self,res):
        self.model=1/res

    def set_data(self,arr):        
        #Set survey parameters
        self.loop_size=arr[0][0]        
        self.ramp=arr[0][1]
        perc=arr[0][2]/100 #noise level (%)
        if self.ramp < 1e-7: #if a lower ramp is set, the forward modeling is unstable in SimPEG
            self.ramp = 1e-7
        
        #Set TEM data
        self.data=arr[1][:,1]  #dB/dZ
        self.times=arr[1][:,0] #times
        self.sigma=perc*np.abs(self.data)+1e-20 #error floor

        #Define receivers
        L=self.loop_size/2
        rx=time_domain.receivers.PointMagneticFieldTimeDerivative(locations=[0,0,0],times=self.times,orientation="z")

        #Define a source
        waveform = time_domain.sources.RampOffWaveform(off_time=self.ramp)
        self.waveform=waveform
        source_current=1.0
        #Square Loop Tranmitter
        source_location=np.zeros([5,3])
        source_location[0,:]=np.array([-L, L, 0])
        source_location[1,:]=np.array([ L, L, 0])
        source_location[2,:]=np.array([ L,-L, 0])
        source_location[3,:]=np.array([-L,-L, 0])
        source_location[4,:]=np.array([-L, L, 0])
        source_list = [
            time_domain.sources.LineCurrent(
                receiver_list=[rx],
                location=source_location,
                waveform=waveform,
                current=source_current,
                )
            ]
        self.survey = time_domain.Survey(source_list)
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)

    def set_mesh(self,m,ind_active=[]):
        self.thk=m.dY
        self.model=np.zeros(len(m.dY))
        self.mesh = discretize.TensorMesh([(np.r_[self.thk])], "0")#, self.thk[-1]

    def set_mesh_dY(self,dY):
        self.thk=dY
        self.model=np.zeros(len(dY))
        self.mesh = discretize.TensorMesh([(np.r_[self.thk])], "0")#, self.thk[-1]
        
    def fwd(self):        
#        self.thk=m.dY
#        self.model=np.zeros(len(m.dY))
        self.model_mapping = maps.ExpMap(nP=len(self.thk))
        self.simulation = time_domain.Simulation1DLayered(
            survey=self.survey, thicknesses=self.thk[:-1], sigmaMap=self.model_mapping
        )

#         #create data object
#         self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)

#         #setting a mesh
#         layer_thicknesses = np.r_[self.thk[:-1]]#[:]#,[self.thk[-1]]])

#         self.mesh = discretize.TensorMesh([np.r_[layer_thicknesses, layer_thicknesses[-1]]], "0")#[(np.r_[layer_thicknesses])], "0")  

#         model_map = maps.IdentityMap(nP=len(self.model)) * maps.ExpMap()
#         simulation = dc.simulation_1d.Simulation1DLayers(
#             survey=self.survey,
#             rhoMap=model_map,
#             thicknesses=layer_thicknesses,
# #            data_type="apparent_resistivity",
#         )

        dpred = self.simulation.dpred(np.log(self.model))
        return dpred
    
    def invert(self,W=[]):
        #create data object
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)
        model_ini = np.concatenate([self.model_ini,[np.sum(self.thk)/2]])
        model_ref= self.model_ini
        Nl=int((len(model_ini)+1)/2)
        
        #Setting foward modeling        
        
        wire_map = maps.Wires(("rho", Nl), ("thk", Nl-1))   #Clase de alambrado

        sim = time_domain.Simulation1DLayered(
            survey=self.survey,
            rhoMap=maps.ExpMap(nP=Nl)*wire_map.rho,                   #Aqui incluye la funcion de mapeo de 
            thicknessesMap=maps.ExpMap(nP=Nl-1)*wire_map.thk)           #Aqui incluye la funcion de mapeo 
#    )
#            survey=self.survey, thicknesses=mod_thk, sigmaMap=self.mapping2
#        )

#        sim=dc.Simulation1DLayers(
#            survey=self.survey,
#            rhoMap=maps.ExpMap(nP=Nl)*wire_map.rho,                   #Aqui incluye la funcion de mapeo de 
#            thicknessesMap=maps.ExpMap(nP=Nl-1)*wire_map.thk)           #Aqui incluye la funcion de mapeo 
                                              #de los espesores de las capas


        #Inversion settings 
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=self.data_object)

        #model regularization
        mesh_r=discretize.TensorMesh([Nl])
        reg_rho=regularization.WeightedLeastSquares(
            mesh_r, alpha_s=0.0001, alpha_x=0, mapping=wire_map.rho)
        
        mesh_t=discretize.TensorMesh([Nl-1])
        reg_thk=regularization.WeightedLeastSquares(
            mesh_t, alpha_s=1e-12, alpha_x=0, mapping=wire_map.thk)

        reg=reg_rho+reg_thk
        self.reg=reg

    
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=0.001)        
                        
        starting_beta = starting_beta
        beta_schedule = directives.BetaSchedule(coolingFactor=1.0, coolingRate=1.0)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=0.1)
        self.directives_list = [
            starting_beta,
            beta_schedule,
            target_misfit,
        ]

        #Run Inversion
        opt = optimization.InexactGaussNewton(maxIter=30, maxIterCG=300)
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
        inv = inversion.BaseInversion(inv_prob, directiveList=self.directives_list)


        self.inv=inv
        model_ini = np.array(np.concatenate([[np.log(100),np.log(100)],[50]]))

        imodel = inv.run(model_ini)
        Z=np.concatenate([[0],np.cumsum(self.thk)])
        recovered_model=np.zeros(len(self.thk))
        for i in np.arange(len(self.thk)):
            if Z[i]>=imodel[2]:            
                recovered_model[i] = imodel[1]             
            else:
                recovered_model[i] = imodel[0]             
        self.iterindex+=1
        return np.exp(-recovered_model)

    def invert_step(self,W=[]):
        None

    def mod2par(self,wtmodel):
        vals=wtmodel
        R0=np.min(vals)
        index=(vals==R0)
        Z1=np.array(np.concatenate([[0],np.cumsum(self.thk[:-1])]))
        Z2=np.array(np.cumsum(self.thk))
        if np.sum(index)<=1:
            Zwt=Z1[index]
            Zbs=Z2[index]
        else:
            Zwt=Z1[index][0]
            Zbs=Z2[index][-1]
        pars=np.array([float(Zwt),float(Zbs),float(R0)])
        return pars

        
    def invert_step2(self,modelwt,W=[],magic_lmd=[]):

        pars=self.mod2par(modelwt)
        mod_thk=np.array([pars[0],pars[1]-pars[0]])
        
        #create data object
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)
        model_ini = self.model_ini
        model_ref= self.model_ref
        Nl=2#int((len(model_ini)+1)/2)

        #Setting foward modeling        
        mapping=self.set_bkmapping(modelwt)
        self.mapping2=mapping
#        wire_map = maps.Wires(("rho", Nl), ("thk", Nl-1))   #Clase de alambrado

        sim = time_domain.Simulation1DLayered(
            survey=self.survey, thicknesses=mod_thk, sigmaMap=self.mapping2
        )

#        sim = dc.Simulation1DLayers(
#            survey=self.survey,
#            rhoMap=mapping,                   #Aqui incluye la funcion de mapeo de 
#                                              #las resistividades de las capas
#            thicknesses=mod_thk)
#            thicknessesMap=mapping[1]*wire_map.thk)           #Aqui incluye la funcion de mapeo 
                                              #de los espesores de las capas

        self.sim2=sim
        #Inversion settings 
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=self.data_object)

        #model regularization
        mesh_r=discretize.TensorMesh([Nl])
        reg_rho=regularization.WeightedLeastSquares(
            mesh_r, alpha_s=1, alpha_x=0)#, mapping=wire_map.rho)
        
#        mesh_t=discretize.TensorMesh([Nl-1])
#        reg_thk=regularization.WeightedLeastSquares(
#            mesh_t, alpha_s=1, alpha_x=0, mapping=wire_map.thk)
#1e-12
        reg=reg_rho#+reg_thk
        self.reg=reg

    
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=0.01)        
                        
        starting_beta = starting_beta
        beta_schedule = directives.BetaSchedule(coolingFactor=1.0, coolingRate=1.0)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=0.1)
        self.directives_list = [
            starting_beta,
            beta_schedule,
            target_misfit,
        ]

        #Run Inversion
        opt = optimization.InexactGaussNewton(maxIter=10, maxIterCG=300)
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
        inv = inversion.BaseInversion(inv_prob, directiveList=self.directives_list)


        self.inv=inv
#        model_ini = np.array(np.concatenate([[np.log(100),np.log(100)],[50]]))

        imodel = inv.run(model_ini)
        #print("imodel:",imodel)
        
        Z=np.concatenate([[0],np.cumsum(self.thk[:-1])])

        recovered_model=np.zeros(len(self.thk))
        for i in np.arange(len(self.thk)):
            if Z[i]>=pars[1]:            
                recovered_model[i] = imodel[1]             
            else:
                recovered_model[i] = imodel[0]             
        self.iterindex+=1
        return np.exp(-recovered_model)


class tem_class_1d_par2:
    def __init__(self,a=[],b=[],m=[],n=[]):
        self.a=a
        self.b=b
        self.n=n
        self.m=m
        self.data=[]
        self.sigma=[]
        #regulation schedule for inversion
        self.iterindex=0
        self.collingRate=1
        self.beta=1
        self.collingFactor=1
        self.alpha_x=1
        self.alpha_y=1
        self.alpha_z=1
        self.alpha_s=0.001

        source_list = []
        for i in np.arange(len(self.a)):
            A_location = np.r_[self.a[i], 0.0, 0.0]
            B_location = np.r_[self.b[i], 0.0, 0.0]
            M_location = np.r_[self.m[i], 0.0, 0.0]
            N_location = np.r_[self.n[i], 0.0, 0.0]    
            receiver_list = dc.receivers.Dipole(M_location, N_location)
            receiver_list = [receiver_list]
            source_list.append(dc.sources.Dipole(receiver_list, A_location, B_location))
        self.survey = dc.Survey(source_list)
                   

    def set_bkmapping(self,wtmodel):
        class mapping_rho(maps.IdentityMap):
            def __init__(self,R0, mesh=None, nP=None, **kwargs):
                super(mapping_rho, self).__init__(mesh=mesh, nP=nP, **kwargs)
                self.R0=R0
        
            def _transform(self, m):
                R1=np.exp(m[0])
                Rwt=1/self.R0+R1
                R2=np.exp(m[1])
                rho=np.array([R1,Rwt,R2])
                return rho
        
            def deriv(self, m):#, v=None):
                R1=np.exp(m[0])
                Rwt=1/self.R0+R1
                R2=np.exp(m[1])
                grad=np.zeros([2,3])
                grad[0,0]=R1
                grad[0,1]=R1
                grad[0,2]=0
                grad[1,0]=0
                grad[1,1]=0
                grad[1,2]=R2
                return grad.T
        
            def inverse(self, m):
                R1=m[0]
                R2=m[2]
                logrho=np.array([np.log(R1),np.log(R2)])
                return logrho    
        
 
        vals=wtmodel
        R0=np.min(vals)
        mapping=mapping_rho(R0)
        return mapping

    def set_modelref(self,model):
        vals = 1/model.copy()
        self.model_ref=np.array([np.log(vals[0]),np.log(vals[-1])])

    def set_modelini(self,model):
        vals = 1/model.copy()
        self.model_ini=np.array([np.log(vals[0]),np.log(vals[-1])])

    def set_model(self,res):
        self.model=1/res

    def set_data(self,arr):        
        #Set survey parameters
        self.loop_size=arr[0][0]        
        self.ramp=arr[0][1]
        perc=arr[0][2]/100 #noise level (%)
        if self.ramp < 1e-7: #if a lower ramp is set, the forward modeling is unstable in SimPEG
            self.ramp = 1e-7
        
        #Set TEM data
        self.data=arr[1][:,1]  #dB/dZ
        self.times=arr[1][:,0] #times
        self.sigma=perc*np.abs(self.data)+1e-20 #error floor

        #Define receivers
        L=self.loop_size/2
        rx=time_domain.receivers.PointMagneticFieldTimeDerivative(locations=[0,0,0],times=self.times,orientation="z")

        #Define a source
        waveform = time_domain.sources.RampOffWaveform(off_time=self.ramp)
        self.waveform=waveform
        source_current=1.0
        #Square Loop Tranmitter
        source_location=np.zeros([5,3])
        source_location[0,:]=np.array([-L, L, 0])
        source_location[1,:]=np.array([ L, L, 0])
        source_location[2,:]=np.array([ L,-L, 0])
        source_location[3,:]=np.array([-L,-L, 0])
        source_location[4,:]=np.array([-L, L, 0])
        source_list = [
            time_domain.sources.LineCurrent(
                receiver_list=[rx],
                location=source_location,
                waveform=waveform,
                current=source_current,
                )
            ]
        self.survey = time_domain.Survey(source_list)
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)

    def set_mesh(self,m,ind_active=[]):
        self.thk=m.dY
        self.model=np.zeros(len(m.dY))
        self.mesh = discretize.TensorMesh([(np.r_[self.thk])], "0")#, self.thk[-1]

    def set_mesh_dY(self,dY):
        self.thk=dY
        self.model=np.zeros(len(dY))
        self.mesh = discretize.TensorMesh([(np.r_[self.thk])], "0")#, self.thk[-1]
        
    def fwd(self):        
#        self.thk=m.dY
#        self.model=np.zeros(len(m.dY))
        self.model_mapping = maps.ExpMap(nP=len(self.thk))
        self.simulation = time_domain.Simulation1DLayered(
            survey=self.survey, thicknesses=self.thk[:-1], sigmaMap=self.model_mapping
        )

#         #create data object
#         self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)

#         #setting a mesh
#         layer_thicknesses = np.r_[self.thk[:-1]]#[:]#,[self.thk[-1]]])

#         self.mesh = discretize.TensorMesh([np.r_[layer_thicknesses, layer_thicknesses[-1]]], "0")#[(np.r_[layer_thicknesses])], "0")  

#         model_map = maps.IdentityMap(nP=len(self.model)) * maps.ExpMap()
#         simulation = dc.simulation_1d.Simulation1DLayers(
#             survey=self.survey,
#             rhoMap=model_map,
#             thicknesses=layer_thicknesses,
# #            data_type="apparent_resistivity",
#         )

        dpred = self.simulation.dpred(np.log(self.model))
        return dpred
    
    def invert(self,W=[]):
        #create data object
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)
        model_ini = np.concatenate([self.model_ini,[np.sum(self.thk)/2]])
        model_ref= self.model_ini
        Nl=int((len(model_ini)+1)/2)
        
        #Setting foward modeling        
        
        wire_map = maps.Wires(("rho", Nl), ("thk", Nl-1))   #Clase de alambrado

        sim = time_domain.Simulation1DLayered(
            survey=self.survey,
            rhoMap=maps.ExpMap(nP=Nl)*wire_map.rho,                   #Aqui incluye la funcion de mapeo de 
            thicknessesMap=maps.ExpMap(nP=Nl-1)*wire_map.thk)           #Aqui incluye la funcion de mapeo 
#    )
#            survey=self.survey, thicknesses=mod_thk, sigmaMap=self.mapping2
#        )

#        sim=dc.Simulation1DLayers(
#            survey=self.survey,
#            rhoMap=maps.ExpMap(nP=Nl)*wire_map.rho,                   #Aqui incluye la funcion de mapeo de 
#            thicknessesMap=maps.ExpMap(nP=Nl-1)*wire_map.thk)           #Aqui incluye la funcion de mapeo 
                                              #de los espesores de las capas


        #Inversion settings 
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=self.data_object)

        #model regularization
        mesh_r=discretize.TensorMesh([Nl])
        reg_rho=regularization.WeightedLeastSquares(
            mesh_r, alpha_s=0.0001, alpha_x=0, mapping=wire_map.rho)
        
        mesh_t=discretize.TensorMesh([Nl-1])
        reg_thk=regularization.WeightedLeastSquares(
            mesh_t, alpha_s=1e-12, alpha_x=0, mapping=wire_map.thk)

        reg=reg_rho+reg_thk
        self.reg=reg

    
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=0.001)        
                        
        starting_beta = starting_beta
        beta_schedule = directives.BetaSchedule(coolingFactor=1.0, coolingRate=1.0)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=0.1)
        self.directives_list = [
            starting_beta,
            beta_schedule,
            target_misfit,
        ]

        #Run Inversion
        opt = optimization.InexactGaussNewton(maxIter=30, maxIterCG=300)
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
        inv = inversion.BaseInversion(inv_prob, directiveList=self.directives_list)


        self.inv=inv
        model_ini = np.array(np.concatenate([[np.log(100),np.log(100)],[50]]))

        imodel = inv.run(model_ini)
        Z=np.concatenate([[0],np.cumsum(self.thk)])
        recovered_model=np.zeros(len(self.thk))
        for i in np.arange(len(self.thk)):
            if Z[i]>=imodel[2]:            
                recovered_model[i] = imodel[1]             
            else:
                recovered_model[i] = imodel[0]             
        self.iterindex+=1
        return np.exp(-recovered_model)

    def invert_step(self,W=[]):
        None

    def mod2par(self,wtmodel):
        vals=wtmodel
        R0=np.min(vals)
        index=(vals==R0)
        Z1=np.array(np.concatenate([[0],np.cumsum(self.thk[:-1])]))
        Z2=np.array(np.cumsum(self.thk))
        if np.sum(index)<=1:
            Zwt=Z1[index]
            Zbs=Z2[index]
        else:
            Zwt=Z1[index][0]
            Zbs=Z2[index][-1]
        pars=np.array([float(Zwt),float(Zbs),float(R0)])
        return pars

        
    def invert_step2(self,modelwt,W=[],magic_lmd=[]):

        pars=self.mod2par(modelwt)
        mod_thk=np.array([pars[0],pars[1]-pars[0]])
        
        #create data object
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)
        model_ini = self.model_ini
        model_ref= self.model_ref
        Nl=2#int((len(model_ini)+1)/2)

        #Setting foward modeling        
        mapping=self.set_bkmapping(modelwt)
        self.mapping2=mapping
#        wire_map = maps.Wires(("rho", Nl), ("thk", Nl-1))   #Clase de alambrado

        sim = time_domain.Simulation1DLayered(
            survey=self.survey, thicknesses=mod_thk, sigmaMap=self.mapping2
        )

#        sim = dc.Simulation1DLayers(
#            survey=self.survey,
#            rhoMap=mapping,                   #Aqui incluye la funcion de mapeo de 
#                                              #las resistividades de las capas
#            thicknesses=mod_thk)
#            thicknessesMap=mapping[1]*wire_map.thk)           #Aqui incluye la funcion de mapeo 
                                              #de los espesores de las capas

        self.sim2=sim
        #Inversion settings 
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=self.data_object)

        #model regularization
        mesh_r=discretize.TensorMesh([Nl])
        reg_rho=regularization.WeightedLeastSquares(
            mesh_r, alpha_s=1, alpha_x=0)#, mapping=wire_map.rho)
        
#        mesh_t=discretize.TensorMesh([Nl-1])
#        reg_thk=regularization.WeightedLeastSquares(
#            mesh_t, alpha_s=1, alpha_x=0, mapping=wire_map.thk)
#1e-12
        reg=reg_rho#+reg_thk
        self.reg=reg

    
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=0.01)        
                        
        starting_beta = starting_beta
        beta_schedule = directives.BetaSchedule(coolingFactor=1.0, coolingRate=1.0)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=0.1)
        self.directives_list = [
            starting_beta,
            beta_schedule,
            target_misfit,
        ]

        #Run Inversion
        opt = optimization.InexactGaussNewton(maxIter=10, maxIterCG=300)
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
        inv = inversion.BaseInversion(inv_prob, directiveList=self.directives_list)


        self.inv=inv
#        model_ini = np.array(np.concatenate([[np.log(100),np.log(100)],[50]]))

        imodel = inv.run(model_ini)
        #print("imodel:",imodel)
        
        Z=np.concatenate([[0],np.cumsum(self.thk[:-1])])

        recovered_model=np.zeros(len(self.thk))
        for i in np.arange(len(self.thk)):
            if Z[i]>=pars[1]:            
                recovered_model[i] = imodel[1]             
            else:
                recovered_model[i] = imodel[0]             
        self.iterindex+=1
        return np.exp(-recovered_model)

class dc_class:
    
    Rho_air = 1e8
    def __init__(self):
        None
        self.iterindex=0
        self.model_ref=[]
        self.model_inv=[]
        self.model_ini=[]
        self.collingRate=2
        self.beta=1000
        self.minbeta=0.1
        self.collingFactor=2
        self.approx1D=False
        self.logsum=False#True
    
    def set_topo(self):
        self.topo=[]
        
    def set_mesh(self,meshin,ind_active=[]):
        X0=meshin.X0
        Y0=meshin.Y0
        self.mesh = discretize.TensorMesh([meshin.dX, meshin.dY], origin=[X0,Y0])
        
        if len(ind_active)==0:# == []:
            self.mapping = maps.ExpMap(self.mesh)
        else:            
            self.ind_active=ind_active
            active_map = maps.InjectActiveCells(self.mesh, self.ind_active, self.Rho_air)
            self.mapping =  active_map*maps.ExpMap()

#        air_resistivity = Rho_air#np.log(1e8)
#        background_conductivity = np.median(-np.log(data.dobs))#toappres(data.dobs,data)))#np.log(1e-2)
        
#        if len(topo)>0:
#            topo[:,1]=topo[:,1]
#            ind_activex = surface2ind_topo(mesh, topo)
#            ind_active = np.ones(mesh.n_cells)==0
#            ind_active[ind_activex]=True
#        else:
#            ind_active = mesh.cell_centers[:,1]>0
        
        self.survey.drape_electrodes_on_topography(self.mesh,ind_active,option="top")
        active_map = maps.InjectActiveCells(self.mesh, ind_active, self.Rho_air)
        nC = int(ind_active.sum())
        resistivity_map = active_map * maps.ExpMap()
#        starting_conductivity_model = background_conductivity * np.ones(nC)
        
        self.simulation = dc.simulation_2d.Simulation2DNodal(
            self.mesh, survey=self.survey, rhoMap=resistivity_map, solver=Solver,nky=11, storeJ=True
        )
#    return simulation,starting_conductivity_model,ind_active


#        self.simulation = dc.Simulation2DNodal(
#            self.mesh, survey=self.survey, rhoMap=self.mapping, solver=Solver, bc_type="Neumann")
        N=self.mesh.n_cells
        self.model=np.zeros(N)

        return self.mesh

    def set_bkmapping(self,wtmodel):
        
        class HybridSum(maps.IdentityMap):
          
            """
            Apply the model sum of Hybrid Decomposition
            .. math::

                \rho_t = 1/(1/\rho_wt + 1/\rho_bk)
            """
            logsum=self.logsum
            
            def __init__(self, mesh=None, nP=None, wt_val=None, ind_active_wt=None, **kwargs):
                self.wt_val=wt_val
                self.ind_active_wt=ind_active_wt
                self.logsum=False#True
                super(HybridSum, self).__init__(mesh=mesh, nP=nP, **kwargs)

            def _transform(self, m):
                if self.logsum:
                    logcond = -mkvc(m)
                    logcond_wt=-np.log(self.wt_val) #1/self.wt_val
                    logcond[self.ind_active_wt]+=logcond_wt
                    cond=np.exp(logcond)
                else:
                    cond = np.exp(-mkvc(m))
                    cond_wt=1/self.wt_val
                    cond[self.ind_active_wt]+=cond_wt
                return 1/cond

            def deriv(self, m):#, v=None):
                if self.logsum:
                    mkvc_m=mkvc(m)
#                    logcond_bk = -mkvc_m
#                    logcond_wt = -np.log(self.wt_val)
#                    logcond_t = logcond_bk.copy()
#                    logcond_t[self.ind_active_wt]+=logcond_wt
                    out = sdiag(np.ones(len(mkvc_m)))#sdiag(cond_bk/(cond_t**2))
                else:
                    mkvc_m=mkvc(m)
                    cond_bk = np.exp(-mkvc_m)
                    cond_wt = 1/self.wt_val
                    cond_t = cond_bk.copy()
                    cond_t[self.ind_active_wt]+=cond_wt
                    out = sdiag(cond_bk/(cond_t)**2)#sdiag(cond_bk/(cond_t**2))
                return out

            def inverse(self, m):
                if self.logsum:
                    logcond = -np.log(mkvc(m))
                    logcond_wt = -np.log(self.wt_val)
                    logcond[self.ind_active_wt]-=logcond_wt
                    out=logcond
                else:
                    cond = 1/mkvc(m)
                    cond_wt = 1/self.wt_val
                    cond[self.ind_active_wt]-=cond_wt
                    out=np.log(1/cond)
                return out


        val_wt=np.min(wtmodel)
        ind_active_wt=(val_wt==wtmodel)
        nC=self.mesh.nC
        
        HS=HybridSum(self.mesh,nP=nC,wt_val=val_wt,ind_active_wt=ind_active_wt)
        mapping= HS        
        return mapping

    def set_data(self,datax_unsort,sigma_per=0.05,floor_error=1e-4):                
    
        C1=np.array(datax_unsort[:,0])
        C2=np.array(datax_unsort[:,2])
        P1=np.array(datax_unsort[:,4])
        P2=np.array(datax_unsort[:,6])
        res=np.array(datax_unsort[:,8])
        volt=res
        err=0.05*volt+floor_error

        #Assigning receivers to each transmitted source
        nC1=np.unique(C1)
        nC2=np.unique(C2)
        dobs=[]
        std=[]
        data_type='apparent_resistivity'
        count=0
        source_list=[]
        for i in np.arange(len(nC1)):
            for j in np.arange(len(nC2)):
                index=(nC1[i]==C1)&(nC2[j]==C2)
                nP1=P1[index]
                nP2=P2[index]
                dobs=np.concatenate([dobs,volt[index]])
                std=np.concatenate([std,err[index]])
                
                if len(nP1)>0:
                    Rx=(dc.receivers.Dipole(locations_m=np.array([nP1,0*nP1]).T,locations_n=np.array([nP2,0*nP2]).T, data_type=data_type))
                    count+=len(nP1)
                    if nC1[i]==nC2[j]:
                        source_list.append(dc.sources.Pole([Rx], [nC1[i],0]))
                    else:
                        source_list.append(dc.sources.Dipole([Rx], [nC1[i],0],[nC2[j],0]))
    
        survey = dc.survey.Survey(source_list)
        self.GF=survey.set_geometric_factor()
        self.survey=survey
        data_out = data.Data(survey=survey,dobs=dobs, standard_deviation=std)
        self.data=dobs                                            
        self.data_object=data_out                                            
        return self.data
                
    def get_GF(self):
        return self.GF.dobs

    def set_modelref(self,model):
        self.modelref=np.log(model)

    def set_modelini(self,model):
        self.modelini=np.log(model)

    def set_model(self,model):
#        if len(self.ind_active)>0:
        self.model=np.log(model)
#        else
#        None    

#    def getfwd(self): #Es necesario??
#        return self.dpred

    def fwd(self):
        if len(self.ind_active) > 0:
            resp=self.simulation.dpred(self.model[self.ind_active])
        else:
            resp=self.simulation.dpred(self.model)
        return resp

    def loglike(self): #Aqui??
        None

    def invert(self,W=[]):
        dc_data_misfit = data_misfit.L2DataMisfit(data=self.data_object, simulation=self.simulation)
        
        # Define the regularization (model objective function)
#        alpha_s=1
        #12.5
#        alpha_x=(3)**2*alpha_s
#        alpha_y=(1)**2*alpha_s

        alpha_s=self.alpha_s#1
        #12.5
        alpha_x=self.alpha_x#(1)**2*alpha_s
        alpha_y=self.alpha_y#(1000)**2*alpha_s #1

#        mesh=simulation.mesh
        starting_resistivity_model=self.modelini[self.ind_active]
        dc_regularization = regularization.WeightedLeastSquares(
            self.mesh,
            active_cells=self.ind_active,
            reference_model=starting_resistivity_model,
            alpha_s=alpha_s,
            alpha_x=alpha_x,
            alpha_y=alpha_y,
        #    length_scale_x=400,
        #    length_scale_y=400
        )
        
        dc_optimization = optimization.InexactGaussNewton(maxIter=5)
        dc_inverse_problem = inverse_problem.BaseInvProblem(
            dc_data_misfit, dc_regularization, dc_optimization
        )
        # Apply and update sensitivity weighting as the model updates
        update_sensitivity_weighting = directives.UpdateSensitivityWeights()
        
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=self.beta)
        beta_schedule = directives.BetaSchedule(coolingFactor=self.collingFactor, coolingRate=self.collingRate)

#        beta_schedule = directives.BetaSchedule(coolingFactor=1.5, coolingRate=2)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        
        # Setting a stopping criteria for the inversion.
        target_misfit = directives.TargetMisfit(chifact=1)
        update_jacobi = directives.UpdatePreconditioner()
        
        directives_list = [
        #    update_sensitivity_weighting,
            starting_beta,
            beta_schedule,
            save_iteration,
            target_misfit,
            update_jacobi,
        ]
        
        # Here we combine the inverse problem and the set of directives
        dc_inversion = inversion.BaseInversion(
            dc_inverse_problem, directiveList=directives_list
        )

        recovered_model = dc_inversion.run(starting_resistivity_model)

        return self.mapping*recovered_model

    def invert_step2(self,modelwt,W=[],magic_lmd=[]):

        mapping_back=self.mapping
        simulation_back=self.simulation
        self.mapping=self.set_bkmapping(modelwt)*maps.InjectActiveCells(self.mesh, self.ind_active, np.log(self.Rho_air))
        simulation = dc.Simulation2DNodal(
        self.mesh, survey=self.survey, rhoMap=self.mapping, solver=Solver, bc_type="Neumann")
        self.simulation=simulation

        dc_data_misfit = data_misfit.L2DataMisfit(data=self.data_object, simulation=self.simulation)
        
        # Define the regularization (model objective function)        
        alpha_s=self.alpha_s
        alpha_x=self.alpha_x
        alpha_y=self.alpha_y
        starting_resistivity_model=self.modelini[self.ind_active]
        dc_regularization = regularization.WeightedLeastSquares(
            self.mesh,
            active_cells=self.ind_active,
            reference_model=starting_resistivity_model,
            alpha_s=alpha_s,
            alpha_x=alpha_x,
            alpha_y=alpha_y,
        )
        
        dc_optimization = optimization.InexactGaussNewton(maxIter=5)
        dc_inverse_problem = inverse_problem.BaseInvProblem(
            dc_data_misfit, dc_regularization, dc_optimization
        )
        # Apply and update sensitivity weighting as the model updates
        #        update_sensitivity_weighting = directives.UpdateSensitivityWeights()
        
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=self.beta)#1000
#self.collingFactor, coolingRate=self.collingRate
#        beta_schedule = directives.BetaSchedule(coolingFactor=1,coolingRate=1)
        beta_schedule = directives.BetaSchedule(coolingFactor=1.5, coolingRate=2)
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        
        # Setting a stopping criteria for the inversion.
        target_misfit = directives.TargetMisfit(chifact=1)
        update_jacobi = directives.UpdatePreconditioner()
        
        directives_list = [
        #    update_sensitivity_weighting,
            starting_beta,
            beta_schedule,
            save_iteration,
            target_misfit,
            update_jacobi,
        ]
        
        # Here we combine the inverse problem and the set of directives
        dc_inversion = inversion.BaseInversion(
            dc_inverse_problem, directiveList=directives_list
        )

        recovered_model = dc_inversion.run(starting_resistivity_model)
        recovered_model = mapping_back * recovered_model
        self.inv_model = recovered_model
        
        self.dpred=dc_inversion.invProb.dpred
        self.iterindex+=1

        self.mapping = mapping_back
        self.simulation = simulation_back
        
        return recovered_model

class grav2d:

    flagfwd=1

    def __init__(self):
        self.dpred=[]
        self.model_ref=[]
        self.model_inv=[]
        self.model_ini=[]
        self.iterindex=0
        self.collingRate=1
        self.beta=5
        self.minbeta=0.1
        self.collingFactor=1

    def set_topo(self):
        self.topo=[]

    def plot_model(self,model,vmin=10,vmax=1000):
        
        fig, axes = plt.subplots(1, 1)
        ax,=self.mesh.plotImage(model,grid=True, ax=axes)
        ax.set_cmap('jet')
        plt.colorbar(ax)
        ax.axes.invert_yaxis()
        
    def plot_invmodel(self,vmin=10,vmax=1000):
        fig, axes = plt.subplots(1, 1)
        
        ax,=self.mesh.plotImage(self.inv_model,grid=True, ax=axes)
        ax.set_cmap('jet')
        ax.set_clim([np.log(vmin),np.log(vmax)])
        ax.axes.invert_yaxis()        
        return ax
        
    def set_mesh(self,meshin,ind_active=[],topo=[]):
        X0=np.array(meshin.X0)
        Y0=np.array(meshin.Y0)
        #dY=np.array([np.sum(meshin.dX)*4])
        dY0=np.sum(meshin.dX)*1000
        dY=[(dY0,1)]
        #-dY/2
        self.mesh = discretize.TensorMesh([meshin.dX,dY,meshin.dY], origin=[X0,-dY0/2,Y0])
        
#        # Assign magnetization values
        nC = len(meshin.dX)*len(meshin.dY)
        activeCells=ind_active
#        activeCells=np.ones(nC)>0
        self.ind_active=ind_active        
        self.background=0
#        active_map = maps.InjectActiveCells(self.mesh, self.ind_active, self.background)
        nC=int(np.sum(ind_active))
#        active_map*
        idnmap =  maps.IdentityMap(nP=nC)
        self.mapping = idnmap 

        # Create reduced identity map
                
        # Create the forward simulation for the global dataset
        self.simulation = gravity.simulation.Simulation3DIntegral(
            survey=self.survey, mesh=self.mesh, rhoMap=idnmap, ind_active=self.ind_active
        )

        N=self.mesh.n_cells
        self.model=np.zeros(N)

    def set_data(self,datax,sigma=0.01):                
        source_list = []
        self.sigma=[]
        self.data=[]
        X=[]
        Y=[]
        Z=[]
        for ii in range(0, len(datax)):
        
            X.append(datax[ii][0])
            Y.append(datax[ii][1])
            Z.append(datax[ii][2])
            self.data.append(datax[ii][3])
        
        self.rxLoc=np.c_[np.array(X),np.array(Y),np.array(Z)]
        self.data=np.array(self.data)
        receivers = gravity.receivers.Point(self.rxLoc)
        srcField = gravity.sources.SourceField([receivers])
        self.survey = gravity.survey.Survey(srcField)
        self.sigma=sigma+self.data*0
        self.data_object = data.Data(self.survey, dobs=self.data, standard_deviation=self.sigma)
        
    def set_halfmodel(self,dens):
        self.model=np.ones(len(self.model))*dens
        
    def set_modelref(self,dens):
        self.model_ref = np.array(dens)

    def set_modelini(self,dens):
        self.model_ini = np.array(dens)

    def set_model(self,dens):
        self.model = np.array(dens)
    
    def getfwd(self):
        return self.dpred

    def pre_set_fwd(self,subNindex):

        self.simulation.dpred(np.ones(len(self.model)))        
        A = self.simulation.getJ(np.ones(len(self.model)))
        self.subNindex=subNindex
        nd=len(self.data)
        nm=len(self.subNindex)
        subJ = np.zeros([nd,nm])
        for i in np.arange(nd):
            for j in np.arange(nm):
                print(A[i,j].data[0][0])
                subJ[i,j]=A[i,j].data[0][0]
        self.subJ=subJ
        self.subNindex = subNindex
        self.flagfwd=1

    def fwd(self):
        if self.flagfwd==2:
            return self.fwd2()
        else:
            return self.fwd1()
            

    def fwd2(self):
        simdata = np.zeros(len(self.data))
#        v = self.model[self.subNindex]
        return np.dot(self.subJ,self.model[self.subNindex])
            
            
#        return self. fwd1()        

    def fwd1(self):

        if len(self.ind_active)>0:
            model=self.model[self.ind_active]
        else:
            model=self.model

        d = self.simulation.fields(model)
        return d 

    def loglike(self):
        self.dpred=self.fwd()
        err = -0.5*np.sum((self.data-self.dpred)**2/self.sigma)/len(self.data)
        return err

    def invert(self,W=[]):
        
        dmis = data_misfit.L2DataMisfit(data=self.data_object, simulation=self.simulation)
        self.ind_active=np.ones(len(self.model))>0

        starting_model=self.model_ini
        mref=self.model_ref

        #Setting Regularization  
#        if(len(W)>0):          
#            reg = regularization.Tikhonov(self.mesh, active_cells=self.ind_active, mapping=self.mapping)
#            reg = regularization.Simple(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x, alpha_y=self.alpha_y,mref=mref,weights=W)
        reg = regularization.WeightedLeastSquares(
                self.mesh,
                active_cells=self.ind_active,
                reference_model=starting_model,
                alpha_s=self.alpha_s,
                alpha_x=self.alpha_x,
                alpha_y=self.alpha_y,
            #    length_scale_x=400,
            #    length_scale_y=400
            )

#        else:
#            reg = regularization.Tikhonov(self.mesh, active_cells=self.ind_active, mapping=self.mapping)
            #            reg = regularization.Simple(self.mesh, alpha_s=self.alpha_s, alpha_x=self.alpha_x, alpha_y=self.alpha_y,mref=mref)

        reg.reference_model_in_smooth = True
        self.reg=reg

        opt = optimization.ProjectedGNCG(
            maxIter=100, lower=-1.0, upper=1.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3
        )
 
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt) 
        
        update_sensitivity_weighting = directives.UpdateSensitivityWeights(every_iteration=False)

        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=self.beta)  #skip simpeg regularization (only is using 1 iteration) 
        beta_schedule = directives.BetaSchedule(coolingFactor=self.collingFactor, coolingRate=self.collingRate)
        
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=self.chifact)
        
        update_jacobi = directives.UpdatePreconditioner()        
        
        # Here is where the norms are applied
        # Use a threshold parameter empirically based on the distribution of
        # model parameters
        update_IRLS = directives.Update_IRLS(
            f_min_change=1e-4,
            max_irls_iterations=0,
            coolEpsFact=1.5,
           beta_tol=1e-2,
        )
 
        directives_list =[update_IRLS, update_sensitivity_weighting, beta_schedule, update_jacobi, save_iteration]

#        saveDict = directives.SaveOutputEveryIteration(save_txt=False)
#        update_Jacobi = directives.UpdatePreconditioner()
#        sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)
        inv = inversion.BaseInversion(
            inv_prob,
            directiveList=directives_list
        )
        
        #Forcing Smooth Reg to use the reference model.
        self.reg.objfcts[1].reference_model_in_smooth=True
        self.reg.objfcts[1].reference_model=self.reg.objfcts[0].reference_model

        # Run the inversion
        recovered_model = inv.run(starting_model)

        self.inv_model=recovered_model-starting_model
        self.dpred=inv.invProb.dpred
        return recovered_model

    def get_regularization2(self):
        alpha_s=1
        mref=self.model_ref
        reg = regularization.WeightedLeastSquares(
                self.mesh,
                active_cells=self.ind_active,
                reference_model=mref,
                alpha_s=alpha_s*0.1,
                alpha_x=alpha_s*1,
                alpha_y=alpha_s*1,
            )
        reg.reference_model_in_smooth = True#False#True#False#True#False  # Reference model in smoothness term
        
        return self.reg

    def get_regularization(self,model):
        tmp=0
        for objfcts in self.reg.objfcts:
            tmp+=objfcts(model)
        return tmp


    def invert_step2(self,modelwt,W=[],magic_lmd=[]):

        mapping_back=self.mapping
        simulation_back=self.simulation
        self.mapping=self.set_bkmapping(modelwt)*maps.InjectActiveCells(self.mesh, self.ind_active, 0)

        simulation = gravity.simulation.Simulation3DIntegral(
            survey=self.survey, mesh=self.mesh, rhoMap=self.mapping, ind_active=self.ind_active)

        self.simulation=simulation

        dmis = data_misfit.L2DataMisfit(data=self.data_object, simulation=self.simulation)
        self.ind_active=np.ones(len(self.model))>0

        starting_model=self.model_ini
        mref=self.model_ref

        #Setting Regularization  
        reg = regularization.WeightedLeastSquares(
                self.mesh,
                active_cells=self.ind_active,
                reference_model=starting_model,
                alpha_s=self.alpha_s,
                alpha_x=self.alpha_x,
                alpha_y=self.alpha_y,
            )

        reg.reference_model_in_smooth = True
        self.reg=reg

        opt = optimization.ProjectedGNCG(
            maxIter=100, lower=-1.0, upper=1.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3
        )
 
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt) 
        
        update_sensitivity_weighting = directives.UpdateSensitivityWeights(every_iteration=False)

        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=self.beta)  #skip simpeg regularization (only is using 1 iteration) 
        beta_schedule = directives.BetaSchedule(coolingFactor=self.collingFactor, coolingRate=self.collingRate)
        
        save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
        target_misfit = directives.TargetMisfit(chifact=self.chifact)
        
        update_jacobi = directives.UpdatePreconditioner()        
        
        # Here is where the norms are applied
        # Use a threshold parameter empirically based on the distribution of
        # model parameters
        update_IRLS = directives.Update_IRLS(
            f_min_change=1e-4,
            max_irls_iterations=0,
            coolEpsFact=1.5,
           beta_tol=1e-2,
        )
 
        directives_list =[update_IRLS, update_sensitivity_weighting, beta_schedule, update_jacobi, save_iteration]

#        saveDict = directives.SaveOutputEveryIteration(save_txt=False)
#        update_Jacobi = directives.UpdatePreconditioner()
#        sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)
        inv = inversion.BaseInversion(
            inv_prob,
            directiveList=directives_list
        )
        
        #Forcing Smooth Reg to use the reference model.
        self.reg.objfcts[1].reference_model_in_smooth=True
        self.reg.objfcts[1].reference_model=self.reg.objfcts[0].reference_model

        # Run the inversion
        recovered_model = inv.run(starting_model)

        recovered_model = mapping_back * recovered_model
        self.inv_model = recovered_model
        
        self.dpred=inv.invProb.dpred
        self.iterindex+=1

        self.mapping = mapping_back
        self.simulation = simulation_back

        return recovered_model


    def get_regularization2(self):
        alpha_s=1
        mref=self.model_ref
        reg = regularization.WeightedLeastSquares(
                self.mesh,
                active_cells=self.ind_active,
                reference_model=mref,
                alpha_s=alpha_s*0.1,
                alpha_x=alpha_s*1,
                alpha_y=alpha_s*1,
            )
        reg.reference_model_in_smooth = True#False#True#False#True#False  # Reference model in smoothness term
        
        return self.reg

    def set_bkmapping(self,wtmodel):
        
        class HybridSum(maps.IdentityMap):
            """
            Apply the model sum of Hybrid Decomposition
            .. math::

                \rho_t = 1/(1/\rho_wt + 1/\rho_bk)
            """

            def __init__(self, mesh=None, nP=None, wt_val=None, ind_active_wt=None, **kwargs):
                self.wt_val=wt_val
                self.ind_active_wt=ind_active_wt
                super(HybridSum, self).__init__(mesh=mesh, nP=nP, **kwargs)

            def _transform(self, m):
                return m + wtmodel

            def deriv(self, m):#, v=None):
                return sdiag(np.ones(len(m)))#sdiag(cond_bk/(cond_t**2))

            def inverse(self, m):
                return m-wtmodel


        val_wt=np.min(wtmodel)
        ind_active_wt=(val_wt==wtmodel)
        nC=self.mesh.nC
        
        HS=HybridSum(self.mesh,nP=nC,wt_val=val_wt,ind_active_wt=ind_active_wt)
        mapping= HS        
        return mapping
