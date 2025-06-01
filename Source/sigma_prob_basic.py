# -*- coding: utf-8 -*-
"""
Created on Thu May 27 06:09:25 2021

@author: ja-ba
"""
#from pymc3.distributions.distribution import Continuous,Discrete, draw_values, generate_samples
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


class PhiProb():
# Conditional probability distribution of log porosity given log electrical bulk resistivity, 
# according to Archie’s law with calibration parameters (m and A) following uniform distributions.

    def __init__(self, m1, m2, logA1, logA2,log_rho,shape=1,**kwargs):
        self.m1=m1
        self.m2=m2
        self.logA1=logA1
        self.logA2=logA2
        self.rho=np.exp(log_rho)
        self.log_rho=log_rho
        self.p11=(logA1-log_rho)/m1
        self.p12=(logA1-log_rho)/m2
        self.p21=(logA2-log_rho)/m1
        self.p22=(logA2-log_rho)/m2
        pp=np.sort(np.array([self.p11,self.p12,self.p21,self.p22]))
        self.p11=pp[0]
        self.p12=pp[1]
        self.p21=pp[2]
        self.p22=pp[3]
        self.shape=shape
        super().__init__(**kwargs)

    def setparams(self,params):
        self.m1=params[0]
        self.m2=params[1]
        self.logA1=params[2]
        self.logA2=params[3]
        self.log_rho=params[4]
        self.rho=np.exp(params[4])
        
        self.p11=(self.logA1-self.log_rho)/self.m1
        self.p12=(self.logA1-self.log_rho)/self.m2
        self.p21=(self.logA2-self.log_rho)/self.m1
        self.p22=(self.logA2-self.log_rho)/self.m2
        pp=np.sort(np.array([self.p11,self.p12,self.p21,self.p22]))
        self.p11=pp[0]
        self.p12=pp[1]
        self.p21=pp[2]
        self.p22=pp[3]

    def logpdf(self, value):

        K=self.m2*(self.logA2-self.logA1)+self.m1*(self.logA1-self.logA2)
        log_por=value
        if value < self.p11:
            like=0
        if((value < self.p12) & (value >= self.p11)):
            like=(-1+self.p11**2/value**2)*self.m1**2/2 
        if((value < self.p21) & (value >= self.p12)):
            like=(self.m2**2-self.m1**2)/2
        if((value < self.p22) & (value >= self.p21)):
            like=(self.m2**2-self.m1**2)/2-(-1+self.p21**2/value**2)*self.m1**2/2
        if value >= self.p22:
            like=0
        
        if like >0:
            ret=np.log(like/K)
        else:
            ret=-np.inf
        return ret
        
    def _distr_parameters_for_repr(self):
        return ["m1", "m2","logA1","logA2","por"]
    

#Continuous
class SigmaProb():
# Conditional probability distribution of log bulk electrical resistivity given log porosity, 
# according to Archie’s law with calibration parameters (m and A) following uniform distributions.

    def __init__(self, m1, m2, logA1, logA2,por,shape=1,**kwargs):
        self.m1=m1
        self.m2=m2
        self.logA1=logA1
        self.logA2=logA2
        self.por=por
        if por == 0:
            self.log_por=-np.Inf
        else:
            self.log_por=np.log(por)

        A0=logA1-self.log_por*m2
        A1=logA1-self.log_por*m1
        A2=logA2-self.log_por*m2
        A3=logA2-self.log_por*m1
        arr=np.sort(np.array([A0,A1,A2,A3]))
        self.r11=arr[1]
        self.r12=arr[0]
        self.r21=arr[3]
        self.r22=arr[2]
            
        self.pm=1/(m2-m1)
        self.pA=1/np.abs(logA2-logA1)
        self.shape=shape
        super().__init__(**kwargs)

    def setparams(self,params):
        self.m1=params[0]
        self.m2=params[1]
        self.logA1=params[2]
        self.logA2=params[3]
        self.por=np.exp(params[4])
        self.log_por=params[4]
        
        A0=self.logA1-self.log_por*self.m2
        A1=self.logA1-self.log_por*self.m1
        A2=self.logA2-self.log_por*self.m2
        A3=self.logA2-self.log_por*self.m1
#        if(A1<A0)&(A0<A3)&(A3<A2):
#            self.r11=A0
#            self.r12=A1
#            self.r21=A2
#            self.r22=A3
#        else:

        arr=np.sort(np.array([A0,A1,A2,A3]))
        self.r11=arr[1]
        self.r12=arr[0]
        self.r21=arr[3]
        self.r22=arr[2]
        
    def random(self, point=None, size=None):
                
        r11=self.r11
        r12=self.r12
        r21=self.r21
        r22=self.r22
        c=(r11-r12)/(r21-r12); 
        d=(r22-r12)/(r21-r12); 
        loc=r12; 
        scale=(r21-r12)
        return stats.trapz.rvs(c=c,d=d,loc=loc,scale=scale,size=size)#generate_samples(

    def logpdf(self, value):
        if self.r11 == np.inf:
            return -np.inf #??
        else:
            c=(self.r11-self.r12)/(self.r21-self.r12); 
            d=(self.r22-self.r12)/(self.r21-self.r12); 
            loc=self.r12; 
            scale=(self.r21-self.r12)            
        return stats.trapz.logpdf(value,c=c,d=d,loc=loc,scale=scale)
        
    def _distr_parameters_for_repr(self):
        return ["m1", "m2","logA1","logA2","por"]


class SigmaProbMarginal():
# Prior probability function of log bulk electrical resistivity, 
# derived from a prior uniform or Gaussian distribution of log-porosity, 
# based on Archie’s law with calibration parameters (m and A) following uniform distributions.
    

    def __init__(self, m1, m2, logA1, logA2,por_array,pordist='uniform',sigmapdf=[],shape=1,sigma_range=[],**kwargs):
        if pordist=='uniform':
            self.LikePor=stats.uniform(0,1)
        else:
            self.LikePor=stats.halfnorm(0,0.4)

        self.SigmaProbList=[]
        self.por_array=por_array
        for i in np.arange(len(por_array)):
            self.SigmaProbList.append(SigmaProb(m1,m2,logA1,logA2,por_array[i]))

        #################################################
        if sigmapdf:
            self.proposal = stats.norm(sigmapdf[0],sigmapdf[1])            
        else:
            if len(sigma_range)>0:
                self.proposal = stats.uniform(sigma_range[0],sigma_range[1]-sigma_range[0])
            else:
                self.proposal = stats.uniform(np.log(0.1),np.log(1E6))
        #################################################

        if por_array[0]==0:
            self.last = np.Inf
        else:
            self.last = logA1-m1*np.log(por_array[0])
        self.like_last = self.logpdf(self.last)
            
    def logpdf(self, value):
        out=0
        N=len(self.SigmaProbList)
        ss=0
        for i in np.arange(N):
            p=self.LikePor.pdf(self.por_array[i])
            s=np.exp(self.SigmaProbList[i].logpdf(value))*p
            if ((np.isinf(s)) or (np.isnan(s))):
                None
            else:                
                out+=s
                ss+=p
        out=out/ss
        if out == 0:
            out = -np.inf
        else:
            out = np.log(out)
        return out
        
    def random(self, point=None, size=None):
        proposal=self.proposal.rvs()
        logpdf = self.logpdf(proposal)
        if logpdf>=self.like_last:
             self.like_last=logpdf           
             self.last=proposal
             None
        else:
            test=np.random.rand()
            if test > 0:
                if test <= np.exp(logpdf)*10:
                    self.like_last=logpdf
                    self.last=proposal                
        return self.last


