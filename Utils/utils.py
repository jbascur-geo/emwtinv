# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:21:18 2023

@author: juan
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../Source/')
import emwtinv_lib as em
import scipy

def model1D_plot(path,conductivity=False):
    bk_model_file=path+'dc1dmod_bk.txt'
    true_model_file=path+'dc1dmod_true.txt'
    mesh_file=path+'dc1dmesh.txt'
    true_model=em.model()
    true_model.read(true_model_file,mesh_file)
    bk_model=em.model()
    bk_model.read(bk_model_file,mesh_file)
    true_model.plot1D(color='--b',conductivity=conductivity)
    bk_model.plot1D(conductivity=conductivity)


class densitymap():
    X=[]
    Y=[]
    im=[]
    def read_pdfmap(self,file):
        fid = open(file,'rt')
        N=fid.readline().split()
        #1D Case
        if len(N) < 3:
            Nx=int(N[0])
            Ny=int(N[1])
        
            X=np.array(fid.readline().split()).astype(float)
            Y=np.array(fid.readline().split()).astype(float)
            
            Y=Y-Y[0]#???
            
            im=np.zeros([Ny,Nx])
            k=0
            for j in np.arange(len(Y)):    
                for i in np.arange(len(X)):
                    im[j,i]=float(fid.readline())
                    k+=1
            fid.close()
        self.im=im
        self.X=X
        self.Y=Y
        
        return im,X,Y

    def marginal(self,axis=0):
        return np.sum(self.im,axis=axis)/np.sum(self.im)

    def mapplot(self,lim=[]):
        R=self.X
        Y=self.Y
        modelwtmap=self.im
        dYY=np.abs(np.diff(Y))
        dlogR=np.abs(np.diff(np.log10(R))[0])
        dY=dYY[0]
        if len(lim)==0:
            ax=plt.pcolor(np.log10(R),-Y,modelwtmap/np.sum(modelwtmap)/dY/dlogR,cmap='ocean_r');        
        else:
            ax=plt.pcolor(np.log10(R),-Y,modelwtmap/np.sum(modelwtmap)/dY/dlogR,cmap='ocean_r');
            plt.xlim([lim[0],lim[1]])                
            plt.ylim([-lim[3],-lim[2]])                
        Z=np.concatenate([[0],np.cumsum(dYY)])

    def RMS(self,map1,map2):
        im1=map1.im/np.sum(map1.im)
        im2=map2.im/np.sum(map2.im)
        alpha = 0.05
        im_min = 5e-4
        rms=np.mean((im1-im2)**2/(alpha*np.abs(im1)+im_min)**2)
        return rms

class parampdf():
    param_name=['Depth [m]','Depth [m]','Rho [ohm-m]','Porosity']
    color=['k','r','b','g','m']
    param_pdf=[]
 
    def plots(self,subplot=0,exp_value=[],symbol='-',alpha=0.5,linewidth=0.5,markersize=1,color='k',fig=[],legend=[],*kwargs):#pdf,
        pdf=self.param_pdf
        #n=len(np.unique(self.param_name))
        n=len(self.param_pdf[0])
        ant_name=self.param_name[0]
        n_ini=0
        n_fin=1
        s=0
        for i in np.arange(n):
            if i == n-1:
                plt.subplot(1,np.max([n,subplot]),s+1)
                self.plot(pdf,np.arange(n_ini,n_fin),exp_value=exp_value,symbol=symbol,alpha=alpha,linewidth=linewidth,markersize=markersize,color=color,*kwargs)
            else:
                plt.subplot(1,np.max([n,subplot]),s+1)
                self.plot(pdf,np.arange(n_ini,n_fin),exp_value=exp_value,symbol=symbol,alpha=alpha,linewidth=linewidth,markersize=markersize,color=color,*kwargs)
                s=s+1
                n_ini=i+1
                n_fin=i+2
        if not(isinstance(legend,list)):
            plt.legend(legend)
            

    def plot(self,pdf,n,exp_value=[],symbol='-',alpha=0.5,linewidth=0.5,markersize=1,color=color,*kwargs):        
        if isinstance(n,int):
            self.singleplot(pdf,n,expval=False,var=False,exp_value=exp_value,symbol=symbol,alpha=alpha,linewidth=linewidth,markersize=markersize,color=color,*kwargs)
        else:
            for i in np.arange(len(n)):
                self.singleplot(pdf,n[i],expval=False,var=False,exp_value=exp_value,symbol=symbol,alpha=alpha,linewidth=linewidth,markersize=markersize,color=color,*kwargs)
        plt.gca().set_facecolor('xkcd:light gray')
        plt.grid(color='w',linestyle='--')

                    
    def log2log10(self,pdf,n):
        pdf[0][n]=np.log10(np.exp(pdf[0][n]))

    def rho2sigma(self,pdf,n):
        pdf[0][n]=-pdf[0][n]

    def singleplot(self,pdf,n,expval=False,var=False,exp_value=[],symbol='-',alpha=0.5,linewidth=0.5,markersize=1,color=[],*kwargs):
        
        dx=(np.max(pdf[0][n])-np.min(pdf[0][n]))/20
        m_fontsize=int(10)
        v_fontsize=int(10)
        dx=np.min(np.abs(np.diff(pdf[0][n])))#[1]-pdf[0][n][0])
        npdf=pdf[1][n]/np.sum(pdf[1][n])/dx
        plt.plot(pdf[0][n],npdf,symbol,alpha=alpha,linewidth=linewidth,markersize=markersize,color=color,*kwargs)
        plt.xlabel(self.param_name[n])
        plt.gcf().subplots_adjust(wspace=0.5)
        if n == 0:
            plt.ylabel('Density')      
        
        if expval==True:
            if len(exp_value)>0:
                m=exp_value[n]
            else:
                m=self.calc_expected_value(pdf[0][n],npdf)
            nm=np.argmin(np.abs(pdf[0][n]-m))
            plt.plot([m,m],[0,npdf[nm]],'--k')
            plt.text(m+dx,npdf[nm]/2,'m='+format(m,'4.1f'),weight='bold',fontsize=m_fontsize)
        if var==True:
            v=self.calc_var(pdf[0][n],npdf)
            plt.text(m+dx,npdf[nm]/3,'sd='+format(v**0.5,'4.1f'),weight='bold',fontsize=v_fontsize)

    def calc_expected_value(self,param,dens):
        p=np.array(param)
        d=np.array(dens)
        return np.sum(p*d)/np.sum(d)

    def calc_expected_values(self,pdfset):
        hatv=[]
        for i in np.arange(len(pdfset[0])):
            p=np.array(pdfset[0][i])
            d=np.array(pdfset[1][i])
            hatv.append(np.sum(p*d)/np.sum(d))
            
        return hatv

    def calc_vars(self,pdfset):
        hatv=[]
        for i in np.arange(len(pdfset[0])):
            p=np.array(pdfset[0][i])
            d=np.array(pdfset[1][i])
            m=self.calc_expected_value(p,d)
            hatv.append(np.sum((p**2)*d)/np.sum(d)-m**2)
        return hatv

    def calc_var(self,param,dens):
        p=np.array(param)
        d=np.array(dens)
        m=self.calc_expected_value(param,dens)
        return np.sum((p**2)*d)/np.sum(d)-m**2 

    def read(self,file):
        fid = open(file,'rt')
        param_pdf=[]    
        param_ticks=[]
        lines=fid.readlines()
        n = int(len(lines)/2)
        for i in np.arange(n):
            
            data=lines[2*i].split(' ')
            ticks=np.zeros(len(data)-1)
            for j in np.arange(len(data)-1):
                ticks[j]=float(data[j])
            data=lines[2*i+1].split(' ')
            hist=np.zeros(len(data)-1)
            for j in np.arange(len(data)-1):
                hist[j]=float(data[j])
            #print(i,ticks)
            param_ticks.append(ticks)    
            param_pdf.append(hist)    
    
        for j in np.arange(len(param_ticks)):
            if len(param_ticks[j])==len(param_pdf[j]):
                None
            else:
                param_ticks[j]=(param_ticks[j][0:-1]+param_ticks[j][1:])/2

        self.param_pdf=[param_ticks,param_pdf]
    
    def old_read(self,file):
        fid = open(file,'rt')
        N=fid.readline().split()
        #1D Case
        if len(N) < 3:
            Nx=int(N[0])
            Ny=int(N[1])
        
            X=np.array(fid.readline().split()).astype(float)
            Y=np.array(fid.readline().split()).astype(float)
            Y=Y-Y[0]
            
            im=np.zeros([Ny,Nx])
            for j in np.arange(len(Y)):    
                for i in np.arange(len(X)):
                    im[j,i]=float(fid.readline())
            fid.close()
        return im,X,Y

    def RMS(self,pdf0,pdf1):
        n=np.min([len(pdf0[0][:]),len(pdf1[0][:])])
        perc=0.10
        rms=[]
        nn=2
        for i in [nn]:
            a=np.concatenate([[0],np.cumsum(np.diff(pdf0[0][i]))])
            b=np.concatenate([a,[2*a[-1]-a[-2]]])
            dx0=np.abs(np.diff(b))            

            a=np.concatenate([[0],np.cumsum(np.diff(pdf1[0][i]))])
            b=np.concatenate([a,[2*a[-1]-a[-2]]])
            dx1=np.abs(np.diff(b))            
            
            f0=np.sum(pdf0[1][i]*dx0)
            f1=np.sum(pdf1[1][i]*dx1)

            f=scipy.interpolate.interp1d(pdf1[0][i],pdf1[1][i]/f1,fill_value='extrapolate')
            y=f(pdf0[0][i])
            minm=0.001
            rms.append(np.mean((y-pdf0[1][i]/f0)**2/(perc*np.abs(pdf0[1][i]/f0)+minm)**2)**0.5)
        return rms,y,pdf0[1][nn]/f0
        