# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 08:59:12 2023

@author: Juan Bascur
"""
import sys
sys.path.insert(0, '../../../Source/')

import numpy as np
from mpi4py import MPI
import emwtinv_lib as emwt
import emwtinv_setup 


setup_file=['emwtinv_setup.txt']
proposal_file='sigmawt_setup.txt'

sigma_bk_file='model_bk_'
sigma_wt_file='model_wt_'
log_file='logfile'


#Multiprocessing
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

#Loading Setup Inversion File.

#Create EM Object
if isinstance(setup_file,list):
    setup=[]
    bk_model=[]
    for i,isetup in enumerate(setup_file):
        setup.append(emwtinv_setup.setup())
        setup[i].read(isetup)   #Setup parameters for E and M Steps

    em=emwt.em(setup,rank=rank)

    for i,isetup in enumerate(setup_file):
        bk_model.append(em.m_step.model[i].copy_model())
    nprint=int(setup[0].estep_niter/10)
    emmax_iter=setup[0].emmax_iter
    estep_niter=setup[0].estep_niter
#    setup=setup[0] #Setup[0] is used for most of e_step parameters.
else:
    setup=emwtinv_setup.setup()
    setup.read(setup_file)   #Setup parameters for E and M Steps
    em=emwt.em(setup,rank=rank)
    bk_model=em.m_step.model.copy_model()
    nprint=int(setup.estep_niter/10)
    emmax_iter=setup.emmax_iter
    estep_niter=setup.estep_niter

# Broadcast BK model from Rank0 to the other Ranks Processes
for im in em.m_step.invmodel_ref:
    tmp_invmodel_ref=np.array(im.get_linear_vals())
    tmp_invmodel_ref = comm.bcast(tmp_invmodel_ref, root=0)
    im.set_linear_vals(tmp_invmodel_ref)

for im in em.e_step.invmodel_ref:
    tmp_invmodel_ref=np.array(im.get_linear_vals())
    tmp_invmodel_ref = comm.bcast(tmp_invmodel_ref, root=0)
    im.set_linear_vals(tmp_invmodel_ref)

#Running Bayesian Hybrid Inversion
priorPbk=0
params_hat=[]
rms_error1=[np.inf]
rms_error2=[np.inf]

for mstep_i in np.arange(emmax_iter):    

    ########################################################################
    ############################## E-Step ##################################
    ########################################################################
    if rank == 0:    
        logfid=open(log_file+str(mstep_i)+'.txt','wt')

    #Printing E-Step Header
    if rank == 0:
        print("0.0 ESTEP_BETA:",em.e_step.estep_beta)
        print("Iteration " + str(mstep_i))    
        print("=====================")    
        #====================================================
        #E-Step
        #====================================================
        print("= 1.- E-Step:")    
        print(" % ====== Sigma_bk_Post_Like ====== Data_Log_Post_Like ====== Totalmodel_Data_Misfit ")    
    
    #E-step Resetting
    em.e_step.init(em.m_step.model,em.e_step.data,rank+nprocs*mstep_i,params_hat)

    for niter in np.arange(estep_niter):
        
        #Messages of MPI E-step processes 
        if ((np.mod(niter,nprint) == 0) or (niter==estep_niter-1)) and (niter>0):
            
            #Printing E-Step Progress (only Rank0)
            if rank == 0:

                #*********************RANK0**********************************
                #Collecting E-step results of MPI parallel processes in rank0
                estep_rcp,index_rcp=em.e_step.get_recipient()
                estep_list_m=[em.e_step.get_list()[0]]
                
                #Receiving E-step partial results from each MPI parallel process 
                for ip in np.arange(1,nprocs):
                    estep_list_ip=[]
                    ntag=0
                    for nrcp in np.arange(len(estep_rcp)):
                        if index_rcp[nrcp] == 1:
                            comm.Recv(estep_rcp[nrcp],ip,tag=ntag)
                            ntag=ntag+1
                            estep_list_ip.append(estep_rcp[nrcp].copy())
                        else:
                            rnn_list=[]
                            for nn in np.arange(index_rcp[nrcp]): 
                                rnn=estep_rcp[nrcp][nn].copy()
                                comm.Recv(rnn,ip,tag=ntag)
                                ntag=ntag+1
                                rnn_list.append(rnn)
                            estep_list_ip.append(rnn_list)

                    estep_list_m.append(estep_list_ip)
                
                em.e_step.set_list(estep_list_m)

                #Printting progress status
                print(str(int(niter/estep_niter*100))+'%',np.abs(2*np.mean(em.e_step.target_function_m)+(priorPbk))**0.5,np.abs(2*np.mean(em.e_step.target_function_data_m))**0.5,rms_error1)
                logfid.write(str(mstep_i)+' '+str(int(niter/estep_niter*100))+' '+str(np.abs(2*np.mean(em.e_step.target_function_m-priorPbk))**0.5)+' '+str((np.abs(np.mean(em.e_step.target_function_data_m)*2))**0.5)+' '+str(rms_error1)+' '+str(rms_error2)+'\n')

            else:
                #*********************RANK_N**********************************
                #Sending preliminary E-step results to rank0 
                estep_list_m,index_rcp=em.e_step.get_list()                
                ntag2=0

                for nrcp in np.arange(len(estep_list_m)):
                    if index_rcp[nrcp] == 1:
                        comm.Send(estep_list_m[nrcp], dest=0, tag=ntag2)                     
                        ntag2+=1
                    else:
                        for nn in np.arange(index_rcp[nrcp]): 
                            comm.Send(estep_list_m[nrcp][nn],0,tag=ntag2)
                            ntag2+=1
    
        #Update E_step
        em.e_step.update(em.m_step.model,em.e_step.invmodel_ref)    
    
    if rank == 0:
        logfid.close()
        #Calculating Expected-Value
        em.e_step.m2local()
        print("Expected Sigma_wt Parameters")
        print(em.e_step.param_hat_m)
        if isinstance(em.m_step.model,list):
            i_bkmodel = em.e_step.param2model(em.m_step.model,em.m_step.invmodel_ref,em.e_step.param_hat_m)
            model_wt_hat = [i_bkmodel[i] for i,im in enumerate(em.m_step.model)] #im.model_sum(im,i_bkmodel[i])
        else:
            model_wt_hat = em.e_step.param2model(em.m_step.model,em.m_step.invmodel_ref,em.e_step.param_hat_m)#em.m_step.model.model_sum(em.m_step.model,em.e_step.param2model(em.m_step.model,em.m_step.invmodel_ref,em.e_step.param_hat_m))
        
    
        #Plotting E-Step Results
#        try:
#        modelwtmap=
#        model_wt_hat=model_wt_hat
#        model = em.e_step.model 
        em.e_step.p.modelplot(sigma_wt_file+'_model_'+str(estep_niter)+'_'+str(mstep_i)+'model.jpg',[em.e_step.Zwt_map],model_wt_hat,bk_model)
#        except:            
#            em.e_step.modelplot(sigma_wt_file+'_model_'+str(estep_niter)+'_'+str(mstep_i)+'model.jpg',vlim=[0,2])
        
        fig=em.e_step.pdfplot()
        fig.savefig(sigma_wt_file+'_parampdfs_'+str(estep_niter)+'_'+str(mstep_i)+'.jpg')
        em.e_step.save_pdfparams(sigma_wt_file+'_'+str(estep_niter)+'_parpdfs_'+str(mstep_i)+'.txt')

        #Generating plot and output files
        if isinstance(em.m_step.model,list):
            for im in np.arange(len(em.m_step.model)):
                em.m_step.model[im].save_model(sigma_bk_file+'['+str(im)+']_'+str(estep_niter)+'_'+str(mstep_i)+'.mod')
                model_wt_hat[im].save_model(sigma_wt_file+'['+str(im)+']_'+str(estep_niter)+'_hat_'+'_'+str(mstep_i)+'.mod')
                em.e_step.save_modelwtmap(sigma_wt_file+'_'+str(estep_niter)+'_densitymap_'+str(mstep_i)+'.mod')
                em.m_step.model[im].mesh.save_mesh('emwtinv.'+str(im)+'.msh')
                em.e_step.save_Zwtmap(sigma_wt_file+'_'+str(estep_niter)+'_Zwt_'+str(mstep_i)+'.mod')
        else:
            em.m_step.model.save_model(sigma_bk_file+'_'+str(estep_niter)+'_'+str(mstep_i)+'.mod')
            em.m_step.model.mesh.save_mesh('emwtinv.msh')
            model_wt_hat.save_model(sigma_wt_file+str(estep_niter)+'_hat_'+'_'+str(mstep_i)+'.mod')
            em.e_step.save_modelwtmap(sigma_wt_file+'_'+str(estep_niter)+'_densitymap_'+str(mstep_i)+'.mod')
            em.e_step.save_Zwtmap(sigma_wt_file+'_'+str(estep_niter)+'_Zwt_'+str(mstep_i)+'.mod')

        em.e_step.save_paramsmap(sigma_wt_file+'_'+str(estep_niter)+'_parmap_'+str(mstep_i)+'.mod')        
        np.savetxt(sigma_wt_file+'_'+str(estep_niter)+'_hatpars_'+str(mstep_i)+'.txt',em.e_step.param_hat_m)
        
                
        ########################################################################
        ############################## M-Step ##################################
        ########################################################################
     
        # M-Step Initialization
        if mstep_i == 0:
            if isinstance(setup,list):
                em.m_step.init(em.m_step.data,model_wt_hat,model_wt_hat,em.m_step.model,em.m_step.invmodel_ref,em.m_step.invmodel_ref,invparameters=[isetup.invparameters for isetup in setup])   #Setting inversion parameters
            else:
                em.m_step.init(em.m_step.data,model_wt_hat,model_wt_hat,em.m_step.model,em.m_step.invmodel_ref,em.m_step.invmodel_ref,invparameters=setup.invparameters)   #Setting inversion parameters
        else:
            if isinstance(setup,list):
                em.m_step.init(em.m_step.data,model_wt_hat,model_wt_hat,em.m_step.model,em.m_step.invmodel_ref,em.m_step.invmodel_ref,invparameters=[[] for isetup in setup])   #Setting inversion parameters
            else:
                em.m_step.init(em.m_step.data,model_wt_hat,model_wt_hat,em.m_step.model,em.m_step.invmodel_ref,em.m_step.invmodel_ref,invparameters=[])   #Setting inversion parameters

        # Inversion step
        bk_model,rms_error1,rms_error2,priorPbk=em.m_step.run([isetup.sub_lmd for isetup in setup]) 
        priorPbk=np.sum(priorPbk)
        print("= 2.- M-Step:")    
        print("=   * Inversion R.M.S = " + str(rms_error1)+ " " + str(rms_error2) + " " + str(priorPbk))#error1: inverted model, error2: bkmodel    
         
         
    
    # Broadcasting BK model from Rank0 to the other Ranks Processes
    if isinstance(bk_model,list):
        for ibk_model in bk_model:
            tmp_bk_model=np.array(ibk_model.get_linear_vals())
            tmp_bk_model = comm.bcast(tmp_bk_model, root=0)
            ibk_model.set_linear_vals(tmp_bk_model)
        em.m_step.set_bk_model(bk_model) 
    else:
        tmp_bk_model=np.array(bk_model.get_linear_vals())
        tmp_bk_model = comm.bcast(tmp_bk_model, root=0)
        bk_model.set_linear_vals(tmp_bk_model)
        em.m_step.set_bk_model(bk_model)
    if rank==0:
        params_hat=np.array(em.e_step.param_hat_m)
    
    comm.bcast(params_hat,root=0)
    