# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 20:48:45 2022

@author: juan
"""
"""! @brief Defines the emwtinv classes."""
##
# @file emwtinv_lib.py
#
# @brief Defines the emwtinv classes.
#
# @section description_emwtinv Description
# Defines base classes for implementing the hybrid Bayesian inversion.
# - data
# - model
# - mesh
# - em (base class) 
# - e_step
# - m_step
# - multi_m_step
# - none_class
#
# @section libraries_emwitinv Libraries/Modules
# - numpy 
# - matplotlib.pyplot
# - scipy
# - link_simpeg: connects simpeg package with emwtlib 
#       -dc_class
#       -dc_class_1d_par
#       -dc_class_1d_par2
#       -tem_class_1d_par2
#       -dc_class_1d
#       -mt_class_1d
#       -mt_class_1d_par2
#       -tdem_class_1d
#       -grav2d
#
# @section notes_emwitinv Notes
# - Comments are Doxygen compatible.
#
# @section todo_emwitinv TODO
# - None.
#
# @section author_sensors Author(s)
# - Created by Juan Bascur on 11/17/2024.
#
# Copyright (c) 2024 Juan A. Bascur.  All rights reserved.

import numpy as np
import matplotlib.pyplot as plt
import scipy
from link_simpeg import dc_class, dc_class_1d_par,dc_class_1d_par2,tem_class_1d_par2, dc_class_1d, mt_class_1d, mt_class_1d_par2, tdem_class_1d, grav2d
#tdem_class_2d, 

class data:
    ind_active = []
    minR = [] #min value of data samples
    maxR = [] #max value of data samples
    ND = [] #number of samples of dataset
    def __init__(self, method):
        # selecting reading function for geophysical data
        if method == 'NONE':
            print('Geophysical method: NONE ')
            self.read = self.readdc1d
            self.set_model_from_data = self.set_model_from_dc1ddata
            self.method = none_class()
            self.misfit = self.misfit_none
            self.linear=False
            self.perc = 5
        if method == 'DC2D':
            print('Geophysical method: DC2D ')
            self.read = self.readres2dinv
            self.set_model_from_data = self.set_model_from_ertdata
            self.method = dc_class()
            self.misfit = self.misfit_DC2D
            self.linear=False
            self.perc = 5
        if method == 'DC1D':
            print('Geophysical method: DC1D ')
            self.read = self.readdc1d
            self.method = dc_class_1d_par2()
            self.set_model_from_data = self.set_model_from_dc1ddata
            self.misfit = self.misfit_DC1D
            self.linear=False
            self.perc = 5
        if method == 'TEM1D':
            print('Geophysical method: TEM1D ')
            self.method = tem_class_1d_par2()
            self.read = self.readtem1d
            self.set_model_from_data = self.set_model_from_tem1ddata
            self.misfit = self.misfit_TEM1D
            self.linear=False
#            self.perc = 5
        if method == 'MT1D':
            print('Geophysical method: MT1D ')
            self.read = self.readmt1d
            self.method = mt_class_1d_par2()
            self.set_model_from_data = self.set_model_from_mt1ddata
            self.misfit = self.misfit_MT1D
            self.linear=False
#            self.perc = 5

        if method == 'GRAV2D':
            print('Geophysical method: GRAV2D ')
            self.read = self.readgrav2d
            self.set_model_from_data = self.set_model_from_grav2ddata
            self.method = grav2d()
            self.misfit = self.misfit_GRAV2D
            self.perc = 5
            self.linear=True

    def init_method(self, model):
        self.method.set_data(self.data)
        if self.linear==True:
            model.linear=True
        if len(self.ind_active) == 0:#[]:
            self.method.set_mesh(model.mesh)
        else:
            #print('Topography modeling actived')
            self.method.set_mesh(model.mesh, ind_active=self.ind_active)

        self.forward = self.method.fwd

    def readnone(self, file):
        None

    def readres2dinv(self, file):
        # Read a .dat files with RES2DINV format.
        fid = open(file, 'rt')
        lines = fid.readlines()

        # Reading header
        name = lines[0].split()
        esp = float(lines[1])
        array = int(lines[2])
        NN = lines[3]
        NN = lines[4]
        NN = lines[5]
        Ndat = int(lines[6])
        dist_type = int(lines[7])
        NN = int(lines[8])
        k = 9
        self.data = []

        for i in np.arange(Ndat):

            dat = lines[k].split()
            Nel = int(dat[0])

            # 3 electrodes (Pole-Dipole)
            if Nel == 3:
                C1x = float(dat[1])
                C1y = float(dat[2])
                C2x = float(dat[1])
                C2y = float(dat[2])
                P1x = float(dat[3])
                P1y = float(dat[4])
                P2x = float(dat[5])
                P2y = float(dat[6])
                R = float(dat[7])

            # 3 electrodes (Dipole-Dipole)
            if Nel == 4:
                C1x = float(dat[1])
                C1y = float(dat[2])
                C2x = float(dat[3])
                C2y = float(dat[4])
                P1x = float(dat[5])
                P1y = float(dat[6])
                P2x = float(dat[7])
                P2y = float(dat[8])
                R = float(dat[9])
            self.data.append([C1x, -C1y, C2x, -C2y, P1x, -P1y, P2x, -P2y, R])
            k = k+1
        self.data = np.array(self.data)
        minX = np.min(np.array(self.data)[:, [0, 2, 4, 6]])
        maxX = np.max(np.array(self.data)[:, [0, 2, 4, 6]])
        self.minX = minX
        self.maxX = maxX
        self.minY = 0
        self.maxY = 0.2*(maxX-minX)
        return self.data

    def readtem1d(self, file):
        fid = open(file,'rt')
        lines = fid.readlines()
        ramp=float(lines[0].split(':')[1])
        loopsize=float(lines[1].split(':')[1]) #Tx square loop side, Area of Rx is 1  
        error=float(lines[2].split(':')[1]) #Error floor (percentage)           
        self.perc=error
        #lines[3] -> Header
        data=[]
        for line in lines[4:]: #first column: time in sec; snd column EM in uvolt 
            strdat=line.split()
            data.append([float(strdat[0]),float(strdat[1])])
        datatem=np.array(data)
        datatem[:, 1] = datatem[:, 1]*10/(4*np.pi) #from uV to SIMPEG

        #Survey parameters
        survey_params=[loopsize,ramp,error]

        self.data=[survey_params,datatem]
        
        self.minR = 1 #range of resistivities 
        self.maxR = 1000
        self.ND = len(datatem[:,-1])
        return self.data  # Z,T

    def readdc1d(self, file):
        self.data = np.loadtxt(file)
        return self.data

    def readmt1d(self, file):
        fid = open(file,'rt')
        lines = fid.readlines()
        self.perc=float(lines[0].split(':')[1])
        #lines[1] -> Header
        data=[]
        F=[]
        for line in lines[1:]: #first column: time in sec; snd column EM in uvolt 
            strdat=line.split()
            data.append(float(strdat[1]))
            data.append(float(strdat[2]))
            F.append(float(strdat[0]))
        datamt=np.array(data)
        F=np.array(F)
        #Survey parameters
        survey_params=[F,self.perc]

        self.data=[survey_params,datamt]
        
        self.minR = 1 #range of resistivities 
        self.maxR = 1000
        self.ND = len(datamt)
        return self.data

    def readgrav2d(self, file):
        # Read a .dat files with GRAV format.
        fid = open(file, 'rt')
        lines = fid.readlines()
        data = []
        for line in lines:
            tmp = line.split()
            X = float(tmp[0])
            Y = float(tmp[1])
            Z = float(tmp[2])
            gravity = float(tmp[3])
            data.append([X, Y, Z, gravity])
        self.data = np.array(data)
        return self.data

    def get_GF(self, a, b, m, n):
        ram = np.linalg.norm(a-m, axis=1)
        ran = np.linalg.norm(a-n, axis=1)
        rbm = np.linalg.norm(b-m, axis=1)
        rbn = np.linalg.norm(b-n, axis=1)
        if np.linalg.norm(a-b) == 0:
            GF = 1/(2*np.pi)*(1/ram-1/ran)
        else:
            GF = 1/(2*np.pi)*(1/ram-1/ran-1/rbm+1/rbn)
        return GF

    def set_model_from_none(self):
        None

    def set_model_from_ertdata(self):
        cells_by_dipole=1
        datax=self.data
        af=datax[:,0:2]
        bf=datax[:,2:4]
        mf=datax[:,4:6]
        nf=datax[:,6:8]
        electrodes=np.unique(np.array(np.concatenate([af,bf,mf,nf])),axis=0) #unique_electrode_locations
        topo=np.sort(electrodes,axis=0)
        dh=np.min(np.abs(np.diff(np.sort(electrodes[:,0]))))/cells_by_dipole
        maxx=np.max(electrodes[:,0])
        minx=np.min(electrodes[:,0])
        dx=maxx-minx
        Ne=1
        hxb = np.concatenate([np.ones(Ne)*dh,2**(np.arange(8))*dh])    
        hx = np.concatenate([hxb[::-1],np.ones(int(dx/dh)+2)*dh,hxb])
    
        a=af[:,0]
        b=bf[:,0]
        m=mf[:,0]
        n=nf[:,0]
        
        dxx=np.max([np.max(np.abs(m-a)),np.max(np.abs(n-a)),np.max(np.abs(m-b)),np.max(np.abs(n-b)),np.max(np.abs(n-m))])
        dhz=dh*cells_by_dipole/6
    
        f=2
        maxtopo=np.max(topo[:,1])
        mintopo=np.min(topo[:,1])
        Nby=8
        #dxx/dhz
        air_cells=np.ones(int(((maxtopo-mintopo)/dhz)))*dhz
        target_cells=np.ones(int(dxx/dhz*2/3))*dhz
        boundary_cells=f**(np.arange(Nby))*dhz*2
        hz = np.concatenate([air_cells,target_cells,boundary_cells])
        hz = hz[::-1]
        xo=minx-np.sum(hxb)
        
        mesh_self = mesh()
        mesh_self.dX=hx
        mesh_self.dY=hz
        Y0 = -np.sum(hz)+maxtopo+dh
        X0 = xo
        mesh_self.X = np.cumsum(mesh_self.dX)-mesh_self.dX/2+X0
        mesh_self.Y = np.cumsum(mesh_self.dY)-mesh_self.dY/2+Y0
        mesh_self.Xe = np.concatenate([[X0],np.cumsum(mesh_self.dX)+X0])
        mesh_self.Ye = np.concatenate([[Y0],np.cumsum(mesh_self.dY)+Y0])

        mesh_self.X0 = xo
        mesh_self.Y0 = Y0
        
        #defining model space
        Rho_air = 1e8
        dobs=datax[:,8]
        background_resistivity = np.exp(np.median(np.log(dobs)))#toappres(data.dobs,data)))#np.log(1e-2)
        
         # ===============================
         # air_cells
         # ===============================
        vals=np.ones(len(mesh_self.Y)*len(mesh_self.X))*background_resistivity
        k = 0
        topo_interp = np.interp(mesh_self.X, topo[:,0], topo[:,1])
        model_topo = 9999.00+topo_interp*0  # maxY+topo*0
        self.Rho_air = Rho_air
        for j in np.arange(len(mesh_self.Y)):
            for i in np.arange(len(mesh_self.X)):
                xo = mesh_self.X[i]
                yo = mesh_self.Y[j]
                if yo > topo_interp[i]:
                    vals[k] = Rho_air
                else:
                    if yo < model_topo[i]:
                        model_topo[i] = mesh_self.Ye[j]  # yo-mesh_self.dY[j]/2
                k = k+1

        # Drape electrodes to surface
        for i in np.arange(len(self.data)):
            c1x = datax[i][0]
            c1y = datax[i][1]
            c2x = datax[i][2]
            c2y = datax[i][3]
            p1x = datax[i][4]
            p1y = datax[i][5]
            p2x = datax[i][6]
            p2y = datax[i][7]

            nx = np.argmin(np.abs(mesh_self.Xe-c1x))
            c1x = mesh_self.Xe[nx]
            c1y = model_topo[nx]
            nx = np.argmin(np.abs(mesh_self.Xe-c2x))
            c2x = mesh_self.Xe[nx]
            c2y = model_topo[nx]
            nx = np.argmin(np.abs(mesh_self.Xe-p1x))
            p1x = mesh_self.Xe[nx]
            p1y = model_topo[nx]
            nx = np.argmin(np.abs(mesh_self.Xe-p2x))
            p2x = mesh_self.Xe[nx]
            p2y = model_topo[nx]

            datax[i][0:8] = [c1x, c1y, c2x, c2y, p1x, p1y, p2x, p2y]

        self.data = datax
        self.ind_active = np.array(vals < Rho_air)

        return mesh_self, vals

    def set_model_from_dc1ddata(self):
        data = self.data
        NL = 30  # Number of layers
        a = data[:, 0]
        b = data[:, 1]
        m = data[:, 2]
        n = data[:, 3]
        r = data[:, 4]
        maxab = np.max([np.abs(a-b), np.abs(a-m), np.abs(a-n)])
        dz = (0.25*maxab)/NL
        meshx = mesh()
        meshx.dY = np.ones(NL)*dz
        meshx.Y = np.cumsum(meshx.dY)-meshx.dY/2
        meshx.Ye = np.concatenate([[0], np.cumsum(meshx.dY)])
        meshx.dX = np.ones(1)*(dz)
        meshx.X = np.ones(1)*(dz)-meshx.dX/2
        meshx.Xe = np.concatenate([[0], np.cumsum(meshx.dX)])
        vals = np.exp(np.mean(np.log(r)))*np.ones(NL)
        self.ind_active = (np.ones(NL) == 1)
        return meshx, vals

    def set_model_from_tem1ddata(self):
        NL = 30  
        dz = (200)/NL
        meshx = mesh()
        meshx.dY = np.ones(NL)*dz
        meshx.Y = np.cumsum(meshx.dY)-meshx.dY/2
        meshx.Ye = np.concatenate([[0], np.cumsum(meshx.dY)])
        meshx.dX = np.ones(1)*(dz)
        meshx.X = np.ones(1)*(dz)-meshx.dX/2
        meshx.Xe = np.concatenate([[0], np.cumsum(meshx.dX)])
        vals = np.ones(NL)*100  # np.exp(np.mean(np.log(r)))*np.ones(NL)
        self.ind_active = (np.ones(NL) == 1)
        return meshx, vals

    def set_model_from_grav2ddata(self):

        data = self.data
        N = len(data)
        X = np.sort(data[:, 0])
        Y = np.sort(data[:, 2])
        minX = min(X)
        dX = abs(np.diff(X))
        dX = np.min(dX[dX > 1e-3])

        # Getting X range
        minX = np.min(X)
        maxX = np.max(X)
        minY = np.min(Y)
        maxY = np.max(Y)
        self.minX = minX
        self.maxX = maxX
        self.minY = minY
        self.maxY = 0.5*(maxX-minX)+minY

        # Discretization
        Nx = int((maxX-minX)/dX)
        dYmin = 0.3*dX
        Ny_topo = int(np.ceil(np.abs(maxY-minY)/(dYmin)))
        Ny = 35  # ...#25 #Number vertical cells
        Nb = 5  # Horizontal boundary cells

        # mesh definition
        mesh_self = mesh()

        # X border cells dimensions
        mesh_self.dX = np.ones(Nx)*dX
        bdX = dX*(1.9**(np.arange(Nb)))
        mesh_self.dX = np.concatenate([bdX[::-1], mesh_self.dX, bdX])
        X0 = np.sum(bdX)

        mesh_self.X = np.cumsum(mesh_self.dX)-mesh_self.dX/2-X0+minX
        mesh_self.Xe = np.concatenate(
            [mesh_self.X-mesh_self.dX/2, [mesh_self.X[-1]+mesh_self.dX[-1]/2]])
        mesh_self.dY = np.concatenate(
            [np.ones(Ny_topo)*dYmin, dX*0.15*((1.1)**np.arange(Ny))])
        Y0 = minY
        mesh_self.Y = np.cumsum(mesh_self.dY)-mesh_self.dY/2+Y0
        mesh_self.Ye = np.concatenate(
            [mesh_self.Y-mesh_self.dY/2, [mesh_self.Y[-1]+mesh_self.dY[-1]/2]])

        # setting an array with the model cells values with the mean apparent resistivity
        vals = np.zeros(len(mesh_self.dX)*len(mesh_self.dY))  # np.log(R)

        # Reference cordinates for model mesh (left upper corner)
        mesh_self.X0 = mesh_self.X[0]-mesh_self.dX[0]/2
        mesh_self.Y0 = mesh_self.Y[0]-mesh_self.dY[0]/2

        # ===============================
        # air_cells
        # ===============================
        k = 0
        topo = np.interp(mesh_self.X, X, Y)
        model_topo = 9999.00+topo*0  # maxY+topo*0
        back_density = 0
        self.back_density = back_density
        for j in np.arange(len(mesh_self.Y)):
            for i in np.arange(len(mesh_self.X)):
                xo = mesh_self.X[i]
                yo = mesh_self.Y[j]
                if yo < topo[i]:
                    vals[k] = back_density
                else:
                    if yo < model_topo[i]:
                        model_topo[i] = mesh_self.Ye[j]  # yo-mesh_self.dY[j]/2
                k = k+1

        # Drape electrodes to surface
        for i in np.arange(len(self.data)):
            Xi = data[i][0]
            Yi = data[i][2]

            nx = np.argmin(np.abs(mesh_self.Xe-Xi))
            Xd = mesh_self.Xe[nx]
            Yd = model_topo[nx]

            data[i][0] = Xd
            data[i][2] = Yd

        self.data = data
        self.ind_active = np.ones(len(vals))==1#np.array(vals < back_density)
        return mesh_self, vals

    def set_model_from_mt1ddata(self):
        NL = 30  # Number of layers
        dz = (200)/NL
        meshx = mesh()
        meshx.dY = np.ones(NL)*dz
        meshx.Y = np.cumsum(meshx.dY)-meshx.dY/2
        meshx.Ye = np.concatenate([[0], np.cumsum(meshx.dY)])
        meshx.dX = np.ones(1)*(dz)
        meshx.X = np.ones(1)*(dz)-meshx.dX/2
        meshx.Xe = np.concatenate([[0], np.cumsum(meshx.dX)])
        vals = np.ones(NL)*100
        self.ind_active = (np.ones(NL) == 1)
        return meshx, vals

    def misfit_MT1D(self, resp):
        obs_rapp = self.data[1]
        calc_rapp = resp
        dev = obs_rapp*self.perc/100
        N = len(resp)
        T = -0.5*np.dot((obs_rapp-calc_rapp)/dev, (obs_rapp-calc_rapp)/dev)/N
        return T

    def misfit_TEM1D(self, resp):
        obs_rapp = self.data[1][:, -1]
        calc_rapp = resp
        dev = obs_rapp*self.perc/100
        N = len(resp)
        T = -0.5*np.dot((obs_rapp-calc_rapp)/dev, (obs_rapp-calc_rapp)/dev)/N
        return T

    def misfit_DC2D(self, resp):
        obs_rapp = np.array(self.method.data)/self.method.get_GF()  # [:,-1]

        calc_rapp = resp/self.method.get_GF()
        dev = obs_rapp*self.perc/100
        N = len(resp)
        logK = 0  
        T = -0.5*np.dot((obs_rapp-calc_rapp)/dev, (obs_rapp-calc_rapp)/dev)/N
        return logK+T

    def misfit_GRAV2D(self, resp):
        obs_rapp = np.array(self.data)[:, -1]
        calc_rapp = resp
        dev = obs_rapp*self.perc/100
        N = len(resp)
        T = -0.5*np.dot((obs_rapp-calc_rapp)/dev, (obs_rapp-calc_rapp)/dev)/N
        return T

    def misfit_DC1D(self, resp):
        obs_rapp = np.array(self.data)[:, -1]
        calc_rapp = resp
        dev = obs_rapp*self.perc/100
        N = len(resp)
        logK = 0
        T = -0.5*np.dot((obs_rapp-resp)/dev, (obs_rapp-resp)/dev)/(N)
        return logK+T

    def misfit_none(self, resp):
        return -np.sum(resp)*10



class model:
    def __init__(self):
        self.mesh = mesh()
        self.vals = []  # log-values
        self.airval=1e9
        self.linear=False
        self.logsum=False  # This flag enables hybrid decomposition in log-conductivity space, 
                            # where background and groundwater conductivities are multiplied 
                            # instead of added (i.e., log-domain superposition).

    def read(self, model_file, mesh_file):
        self.mesh.read_mesh(mesh_file)
        self.read_model(model_file)

    def set_model_from_data(self, data):
        self.mesh, y = data.set_model_from_data()
        self.set_linear_vals(y)

    def read_mesh(self, file):
        self.mesh.read_mesh(file)

    def set_linear_vals(self, vals):
        if self.linear:
            self.vals = vals
        else:
            self.vals = np.log(vals)

    def set_log_vals(self, vals):
        if self.linear:
            self.vals = np.exp(vals)
        else:
            self.vals = vals

    def get_linear_vals(self):
        if self.linear:
            tmp = self.vals
        else:
            tmp = np.exp(self.vals)
        return tmp

    def get_log_vals(self):
        if self.linear:
            tmp = np.log(self.vals)
        else:
            tmp = self.vals
        return tmp

    def get_Zwt(self, curve=False):
        Nx = len(self.mesh.dX)
        Ny = len(self.mesh.dY)
        mod = np.reshape(self.get_linear_vals(), [Ny, Nx])
        res = np.zeros([Ny, Nx])
        Zwt_curve = np.zeros(Nx)
        k = 0
        indexj=np.argsort(self.mesh.Y)[::-1]
        for i in np.arange(Nx):
            k = Ny*i
            for j in indexj:
                if mod[j, i] < self.airval/10:
                    Zwt_curve[i] = self.mesh.Y[j]
                    res[j, i] = 1
                    break
                k = k+1
        if curve == False:
            res = np.reshape(res, [Ny*Nx])
        else:
            res = Zwt_curve
        return res  

    def get_Zbs(self, curve=False):
        Nx = len(self.mesh.dX)
        Ny = len(self.mesh.dY)
        mod = np.reshape(self.get_linear_vals(), [Ny, Nx])
        Zbs_curve = np.zeros(Nx)
        res = np.zeros([Ny, Nx])
        k = 0
        for i in np.arange(Nx):
            for j in np.arange(Ny)[::-1]:
                if mod[j, i] < self.airval:
                    Zbs_curve[i] = self.mesh.Ye[j]
                    res[j, i] = 1
                    break
        if curve == False:
            res = np.reshape(res, [Ny*Nx])
        else:
            res = Zbs_curve
        return res 

    def read_model(self, file):
        # Read a generic model file
        fid = open(file, 'rt')

        # Header with number of cells in each direction
        Nx = len(self.mesh.dX)
        Ny = len(self.mesh.dY)
        Nz = len(self.mesh.dZ)
        self.vals = []
        for i in np.arange(Nx):
            for j in np.arange(Ny):
                if Nz == 0:
                    # Reading 2D model values
                    tmp = fid.readline()
                    if self.linear:
                        tmp = float(tmp)
                    else:
                        tmp = np.log(float(tmp))                        
                    self.vals.append(tmp)
                else:
                    # Reading 3D model values
                    for k in np.arange(Nz):
                        if self.linear:
                            tmp = float(tmp)
                        else:
                            tmp = np.log(float(tmp))                        
                        self.vals.append(tmp)

        self.vals = np.array(self.vals)

    def save_model(self, file):
        # Read a generic model file
        fid = open(file, 'wt')
        # Header with number of cells in each direction
        Nx = len(self.mesh.dX)
        Ny = len(self.mesh.dY)
        Nz = len(self.mesh.dZ)
        n = 0
        for i in np.arange(Nx):
            for j in np.arange(Ny):
                if Nz == 0:
                    # Reading 2D model values
                    if self.linear:
                        tmp = self.vals[n]
                    else:
                        tmp = np.exp(self.vals[n])                        
                    fid.write(str(tmp)+'\n')
                    n = n+1
                else:
                    # Reading 3D model values
                    for k in np.arange(Nz):
                        if self.linear:
                            tmp = self.vals[n]
                        else:
                            tmp = np.exp(self.vals[n])                        
                        fid.write(str(tmp)+'\n')
                        n = n+1

    def copy_model(self):
        # Create a copy of the model object
        modeltmp = model()
        modeltmp.mesh.Nx = self.mesh.Nx
        modeltmp.mesh.Ny = self.mesh.Ny
        modeltmp.mesh.Nz = self.mesh.Nz
        modeltmp.mesh.X = self.mesh.X.copy()
        modeltmp.mesh.Y = self.mesh.Y.copy()
        modeltmp.mesh.Z = self.mesh.Z.copy()
        modeltmp.mesh.Xe = self.mesh.Xe.copy()
        modeltmp.mesh.Ye = self.mesh.Ye.copy()
        modeltmp.mesh.Ze = self.mesh.Ze.copy()
        modeltmp.mesh.dX = self.mesh.dX.copy()
        modeltmp.mesh.dY = self.mesh.dY.copy()
        modeltmp.mesh.dZ = self.mesh.dZ.copy()
        modeltmp.mesh.X0 = self.mesh.X0
        modeltmp.mesh.Y0 = self.mesh.Y0
        modeltmp.mesh.Z0 = self.mesh.Z0
        modeltmp.vals = self.vals.copy()
        modeltmp.linear = self.linear
        modeltmp.airval = self.airval
        modeltmp.logsum = self.logsum

        return modeltmp

    def model_sum(self, model1, model2):
        if self.logsum:
            modelout = self.model_log_sum(model1, model2)
        else:
            if self.linear:
                vals = model1.get_linear_vals() + model2.get_linear_vals()
            else:
                vals = 1/(1/np.exp(model1.get_log_vals()) +
                      1/np.exp(model2.get_log_vals()))
            modelout = model1.copy_model()
            modelout.set_linear_vals(vals)
        return modelout

    def model_diff(self, model1, model2):
        if self.linear:
            vals = model1.get_linear_vals()-model2.get_linear_vals()
        else:
            off = 1e-20
            vals = 1/(1/model1.get_linear_vals()-1/model2.get_linear_vals()+off)

        modelout = model1.copy_model()
        modelout.set_linear_vals(vals)
        return modelout

    def model_log_diff(self, model1, model2):
        vals = model1.get_log_vals()-model2.get_log_vals()
        modelout = model1.copy_model()
        modelout.set_log_vals(vals)
        return modelout

    def model_log_sum(self, model1, model2):
        vals = model1.get_log_vals()+model2.get_log_vals()
        modelout = model1.copy_model()
        modelout.set_log_vals(vals)
        return modelout

    def set_constant_model(self, value):
        modelout = self.copy_model()
        modelout.set_linear_vals(np.ones(len(modelout.vals))*value)
        return modelout

    def plot1D(self, color='k', conductivity=False, labels=True,linewidth=1.0,alpha=1.0):

        if conductivity:
            yant = 0  # self.mesh.Y0
            logR = np.log10(self.get_linear_vals())
            logRant = logR[0]
            logRR=[]
            YY=[]
            for i in np.arange(len(self.mesh.dY)):

                if i > 0:
                    logRR.append(logRant)
                    logRR.append(logR[i])
                    YY.append(np.abs(yant))
                    YY.append(np.abs(yant))
                logRR.append(logR[i])
                logRR.append(logR[i])
                YY.append(np.abs(yant))
                YY.append(np.abs(yant+self.mesh.dY[i]))
                
                
                yant = yant+self.mesh.dY[i]
                logRant = logR[i]
                
            plt.plot(-logRR,YY, color,linewidth=linewidth,alpha=alpha)
            plt.gca().yaxis.set_inverted(True)
            if labels == True:
                plt.xlabel('Conductivity [log10 S/m]')
                plt.ylabel('Depth [m]')
        else:
            yant = 0  # self.mesh.Y0
            logR = np.log10(self.get_linear_vals())
            logRant = logR[0]
            logRR=[]
            YY=[]
            for i in np.arange(len(self.mesh.dY)):
                if i > 0:
                    logRR.append(logRant)
                    logRR.append(logR[i])
                    YY.append(np.abs(yant))
                    YY.append(np.abs(yant))
                logRR.append(logR[i])
                logRR.append(logR[i])
                YY.append(np.abs(yant))
                YY.append(np.abs(yant+self.mesh.dY[i]))

                yant = yant+self.mesh.dY[i]
                logRant = logR[i]
            
            plt.plot(logRR,YY, color,linewidth=linewidth,alpha=alpha)
            plt.gca().yaxis.set_inverted(True)

            if labels == True:
                plt.xlabel('Resistivity [log10 Ohm.m]')
                plt.ylabel('Depth [m]')

    def plot2D(self, conductivity=False, log10=False, density=False, cmap='ocean_r', xlim=[], ylim=[], colorbar_label=[], contour=False, vlim=[]):

        if conductivity:
            if log10:
                R = -np.log10(np.exp(self.get_log_vals()))
            else:
                R = 1/self.get_linear_vals()

        else:
            if log10:
                R = np.log10(np.exp(self.get_log_vals()))
            else:
                R = self.get_linear_vals()

        dY = self.mesh.dY
        dX = self.mesh.dX
        Y = self.mesh.Ye
        X = self.mesh.Xe
        Nx = len(dX)
        Ny = len(dY)

        NNX, NNY = np.meshgrid(np.arange(Nx), np.arange(Ny))
        areamap = dX[NNX]*dY[NNY]

        if density == 1:
            Rmap = np.reshape(R, [Ny, Nx])/areamap/np.sum(R)
        else:
            Rmap = np.reshape(R, [Ny, Nx])

        if contour:
            XX, YY = np.meshgrid(self.mesh.X, self.mesh.Y)
            if vlim:
                plt.contourf(XX, YY, Rmap, cmap=cmap,
                             vmin=vlim[0], vmax=vlim[1])
            else:
                plt.contourf(XX, YY, Rmap, cmap=cmap)
        else:
            if vlim:
                plt.pcolor(X, Y, Rmap, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
            else:
                plt.pcolor(X, Y, Rmap, cmap=cmap)

        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)

        clb = plt.colorbar()
        if colorbar_label:
            clb.set_label(colorbar_label)
        plt.gca().yaxis.set_inverted(False)

    def get_N2d(self):
        return [len(self.mesh.Y), len(self.mesh.X)]

    def get_vals2d(self):
        return np.reshape(self.get_linear_vals(), [len(self.mesh.Y), len(self.mesh.X)])

    def set_from1D(self, model1D):
        Nx = len(self.mesh.dX)
        Ny = len(self.mesh.dY)
        vals = np.zeros([Nx, Ny])
        for i in np.arange(Nx):
            for j in np.arange(Ny):
                vals[i, j] = model1D[j]
        self.set_linear_vals(vals)


class mesh:
    def __init__(self):
        self.dX = []
        self.dY = []
        self.dZ = []
        self.X = []
        self.Y = []
        self.Z = []
        self.Xe = []
        self.Ye = []
        self.Ze = []
        self.X0 = 0
        self.Y0 = 0
        self.Z0 = 0
        self.Nx = 0
        self.Ny = 0
        self.Nz = 0

    def save_mesh(self, file):

        # Save a generic mesh file
        fid = open(file, 'wt')
        if self.Nz > 0:
            fid.write(str(len(self.dX))+' '+str(len(self.dY)) +
                      ' '+str(len(self.dZ))+'\n')
            fid.write(str(self.X0)+' '+str(self.Y0)+' '+str(self.Z0)+'\n')
        else:
            fid.write(str(len(self.dX))+' '+str(len(self.dY))+'\n')
            fid.write(str(self.X0)+' '+str(self.Y0)+'\n')

        fid.write(str(self.dX[0]))
        for i in np.arange(1, len(self.dX)):
            fid.write(' ' + str(self.dX[i]))

        fid.write('\n')
        fid.write(str(self.dY[0]))
        for i in np.arange(1, len(self.dY)):
            fid.write(' ' + str(self.dY[i]))

        fid.write('\n')
        if self.Nz > 0:
            fid.write(str(self.dZ[0]))
            for i in np.arange(1, len(self.dZ)):
                fid.write(' ' + str(self.dZ[i]))

        fid.close()

    def read_mesh(self, file):
        # Read a generic mesh file
        fid = open(file, 'rt')
        lines = fid.readlines()
        k = 0
        self.Nx = lines[k].split()[0]
        self.Ny = lines[k].split()[1]
        if len(lines[k].split()) > 2:
            self.Nz = int(lines[k].split()[2])
        else:
            self.Nz = 0
        k = k+1
        self.X0 = float(lines[k].split()[0])
        self.Y0 = float(lines[k].split()[1])
        if self.Nz > 0:
            self.Z0 = float(lines[k].split()[2])
        else:
            self.Z0 = 0
        k = k+1
        self.dX = np.array(lines[k].split(), dtype=float)
        k = k+1
        self.dY = np.array(lines[k].split(), dtype=float)
        k = k+1
        if self.Nz > 0:
            self.dZ = np.array(lines[k].split(), dtype=float)
        else:
            self.dZ = np.array([])
        # Coordinates at the center of the cells
        self.X = np.cumsum(self.dX)-self.dX/2+self.X0
        self.Y = np.cumsum(self.dY)-self.dY/2+self.Y0
        self.Z = np.cumsum(self.dZ)-self.dZ/2+self.Z0
        self.Xe = np.concatenate([[self.X0], np.cumsum(self.dX)+self.X0])
        self.Ye = np.concatenate([[self.Y0], np.cumsum(self.dY)+self.Y0])
        self.Ze = np.concatenate([[self.Z0], np.cumsum(self.dZ)+self.Z0])


class em:
    def __init__(self, setup, rank=0):
        if isinstance(setup, list):
            self.data = []
            for isetup in setup:
                tmp = data(isetup.method)
                tmp.read(isetup.data_list)
                self.data.append(tmp)
            self.m_step = multi_m_step(setup, self.data, rank=rank)

        else:
            self.data = data(setup.method)
            self.data.read(setup.data_list)
            self.m_step = m_step(setup, self.data, rank=rank)

        self.e_step = e_step(setup, self.data, self.m_step.model, rank)
        
        if isinstance(self.m_step.model,list):
            for i in np.arange(len(self.m_step.model)):
                self.m_step.m_step[i].magic_lmd=self.e_step.p.magiclmd
        else:
            self.m_step.m_step[i].magic_lmd=self.e_step.p.magiclmd


    def read(self, filename):
        # read configration file of em algorithm
        fid = open(filename, 'rt')
        # Initial background model, 0: constant value, 1: from external files
        model_flag = fid.readline().split('//')[0].split()[0]

    def set_bk_model(self, model):
        self.bk_model = model.copy_model()
        return self.bk_model

    def set_wt_model(self, model):
        self.wt_model = model.copy_model()
        return self.wt_model


class e_step:
    """!Class for implementing the E-step in the Hybrid Bayesian Inversion. The E-step aims at estimating the posterior PDF \f$P(m_{wt}|m_{bk}(n),d)\f$\n\n
    with,\n
    \f$m_{wt}\f$: Groundwater model.    
    \f$m_{bk}\f$: Background model.    
    \f$d\f$: Geophysical Observations    
    """
    # PDFs
    # - respmap: Histograms that estimates the PDF of the forward responses. (discretization defined in setup file)
    # - params_hat: exepected value of model_wt parameters.
    # - parpdf: Histograms that estimates the PDFs of the model_wt parameters.
    # - sigma_wt_pdf: Estimated pdf of model_wt.
    # PDF Maps in model_wt space
    # - Zbsmap: Map for representing the Zbs uncertatinty.
    # - Zwtmap: Map for representing the Zwt uncertatinty.
    # - sigma_wt_map: Map for representing the model_wt uncertatinty.
    #
    #

    model_wt_hat = []
    pdfmap = []
    pre_pdfmap = []
    target_function = []
    prior_function = []

    get_param_proposal = []
    data = []
    model = []
    Niter = 1000
    param_pdf_list = []
    params = []  # last params used
    params2model = []
    last_log_pdf_d = []
    nforward = 0
    invmodel_ref = []

    def __init__(self, datax, modelx):
        """!Simple E-Step constructor (used?). 
        
        @param datax(class) Data object including data observations, geophysical method modelling functions, misfit definition.            
        @param modelx(class) Background model or list of Backgound models.

        @return
            None
        """    
        self.data = datax
        self.model = modelx
        self.estep_beta = 100
        self.estep_betafactor = 1.5

    def __init__(self, setup, data, model, rank):
        """!E-Step constructor. 

        @param setup(class) Setup object with EM algorithm parameters.            
        @param data(class) Data object including data observations, geophysical method modelling functions, misfit definition.
        @param model(class) Background model or list of Backgound models.
        @param rank(int) Designation of parallel process or thread. Used as seed in random number generators.

        @return
            None
        """    
        self.data = data
        self.model = model
        if isinstance(setup, list):
            method = setup[0].e_step_prop  # method
            if method.find('REF') >= 0:
                import multi_prop_dc2d_constrained as prop
                self.curve=False

            if method.find('2D') >= 0:
                import multi_prop_dc2d as prop
                self.curve=False

            if method.find('1D') >= 0:
                import prop_dc1d as prop
                self.curve=False


            if method.find('NONE') >= 0:
                import prop_none as prop
                self.curve=False

# Implemented but not thoroughly tested — remove before sharing.
#            if method.find('SED') >= 0:
#                import prop_sed as prop
#                self.curve=False
#            if method.find('rich') >= 0:
#                import multi_prop_site as prop
#                self.curve=False
#            if method.find('MOD1D') >= 0:
#                import prop_dc1d_multimodel as prop
#                self.curve=False
#            if method.find('LOG') >= 0:
#                import multi_prop_dc2d_log as prop
#                self.curve=False

            setup = setup[0]
        else:
            method = setup.e_step_prop
#            method=setup.method
            if method.find('2D') > 0:
                import prop_dc2d as prop
            if method.find('1D') > 0:
                import prop_dc1d as prop
            if method.find('NONE') >= 0:
                import prop_none as prop
# Implemented but not thoroughly tested — remove before sharing.
#            if method.find('rich') >= 0:
#                import multi_prop_site as prop
#            if method.find('MOD1D') >= 0:
#               import prop_dc1d_multimodel as prop

        self.prop=prop
        self.p = prop.param_proposal(
            self.model, 'sigmawt_setup.txt', seed=rank)
        
        self.set_param_proposal(self.p.get_param_proposal)
        self.set_param2model(self.p.param2model)
        try:
            self.update_modelwtmap=self.update_modelwtmap1
        except:
            self.update_modelwtmap=self.update_modelwtmap_old

        try:
            self.modelplot=self.p.modelplot
        except:
            self.modelplot=self.modelplot_old

            
        if setup.estep_beta > 0:
            self.invmodel_ref = self.p.get_invmodelref(self.model)
        else:
            if isinstance(self.model, list):
                self.invmodel_ref = [imodel.copy_model()
                                     for imodel in self.model]
            else:
                self.invmodel_ref = self.model.copy_model()

        self.estep_beta = setup.estep_beta
        self.estep_beta_factor = setup.estep_beta_factor
        self.sub_lmd = setup.sub_lmd

        self.last_proposed_total_model = []
        self.param_pdf_list = []
        self.params = []  # last params used
        self.params2model = []
        self.last_log_pdf_d = []
        self.nforward = 0
        self.model_wt_hat = []
        self.pdfmap = []
        self.target_function = []
        self.target_function_data = []
        self.target_function_model = []
        self.target_function_sigma_bk = []

        self.prior_function = []
        self.modelmap = []
        self.modelwtmap = []
        self.parammap = []
        self.N = 0
        self.super_target_function = []

        minpar, maxpar, Npar = self.p.get_paramrange()
        self.init_histogram(minpar, maxpar, Npar)
        if method.find('SED')>=0:
            print('Set forward operator')
            self.data[0].method.pre_set_fwd(self.p.index_ref)



    def init(self, modelx, datax, rank,params_hat=[]):
        """!E-Step update (used?). 

        @param modelx(class) latest estimated background model or list of backgound models.
        @param datax(class) Data object including data observations, geophysical method modelling functions, misfit definition.
        @param rank(int) Designation of parallel process or thread. Used as seed in random number generators.
        @param params_hat(array or list) Expected values of groundwater model parameters.  

        @return
            None
        """    

        self.data = datax
        self.model = modelx
        self.parameters = []#parameters
        self.last_proposed_total_model = []
        self.param_pdf_list = []
        self.params = []  # last params used
        self.params2model = []
        self.last_log_pdf_d = []
        self.last_log_pdf_data = []
        self.last_log_pdf_model = []
        self.nforward = 0
        self.model_wt_hat = []
        self.pdfmap = []
        self.target_function = []
        self.target_function_data = []
        self.target_function_sigma_bk = []
        self.prior_function = []
        self.modelmap = []
        self.modelwtmap = []
        self.parammap = []
        self.N = 0
        self.super_target_function = []
        minpar, maxpar, Npar = self.p.get_paramrange()
        self.init_histogram(minpar, maxpar, Npar)
        self.estep_beta = self.estep_beta/self.estep_beta_factor
        self.p = self.prop.param_proposal(
            self.model, 'sigmawt_setup.txt', seed=rank, params_hat=params_hat)



    def get_recipient(self):
        """!Retrieve a list of preliminary results from an e_step thread. Used for multi-threaded execution of e_step.  

        @param None
        
        @return
        recipient = [sigma_wt_hat_rcp,
                          sigma_wt_max_rcp,
                          sigma_wt_pdfmap_rcp,
                          target_function_rcp,
                          target_function_data_rcp,
                          max_target_function_rcp,
                          sigma_wt_map_rcp,
                          Zwt_map_rcp,
                          Zbs_map_rcp,
                          respmap_rcp]
        @return
        index_list    number of lists included in each component of recipient
        """    

        index_list = {}

        # Create recipients to receive info from MPI processes
        # expected value of model_wt parameters
        sigma_wt_hat_rcp = np.array(self.get_expectation()).copy()
        index_list['sigma_wt_hat'] = 1  # array

        # model_wt parameters with the highest likelihood
        sigma_wt_max_rcp = np.array(self.get_max_param()).copy()
        index_list['sigma_wt_max'] = 1  # array

        tmp = self.get_pdfmap()
        index_list['pdfmap'] = len(tmp)
        sigma_wt_pdfmap_rcp = []
        for i in np.arange(index_list['pdfmap']):
            sigma_wt_pdfmap_rcp.append(np.array(tmp[i]).copy())

        target_function_rcp = np.array(self.get_target_function()).copy()
        index_list['target_function'] = 1

        target_function_data_rcp = np.array(
            self.get_target_function_data()).copy()
        # may be it could a list for multimethod
        index_list['target_function_data'] = 1

        max_target_function_rcp = np.array(
            self.get_max_target_function()).copy()
        index_list['max_target_function'] = 1

        sigma_wt_map_rcp = np.array(self.get_modelwtmap()).copy()
        if isinstance(sigma_wt_map_rcp, list):
            index_list['sigma_wt_map'] = len(
                sigma_wt_map_rcp)  # list for multimethod
        else:
            index_list['sigma_wt_map'] = 1  # list for multimethod

        Zwt_map_rcp = self.get_Zwtmap().copy()
        index_list['Zwt_map'] = 1  # In the meantime, it is an array

        Zbs_map_rcp = self.get_Zbsmap().copy()
        index_list['Zbs_map'] = 1  # In the meantime, it is an array

        respmap_rcp = self.get_respmap().copy()
        if isinstance(respmap_rcp, list):
            index_list['respmap'] = len(respmap_rcp)
        else:
            index_list['respmap'] = 1
        self.index_list = index_list  # list of lists!!!

        self.recipient = [sigma_wt_hat_rcp,
                          sigma_wt_max_rcp,
                          sigma_wt_pdfmap_rcp,
                          target_function_rcp,
                          target_function_data_rcp,
                          max_target_function_rcp,
                          sigma_wt_map_rcp,
                          Zwt_map_rcp,
                          Zbs_map_rcp,
                          respmap_rcp]
        return self.recipient, [self.index_list[key] for key in self.index_list.keys()]

    def get_list(self):
        """!Same as get_recipient.  
        """    

        self.xlist = []
        self.xlist = self.get_recipient()

        return self.xlist

    def set_list(self, list_mm):
        """!Consolidate the temporary results from all e_step threads into a single e_step object.\n  
        @param list_mm list of recepients from e_step threads.
        """    

        rcp_m = {key: [] for key in self.index_list.keys()}
        n = 0
        for n in np.arange(len(list_mm)):  # list_m in list_mm[n]:
            list_m = list_mm[n]
            k = 0
            for key in self.index_list.keys():
                rcp_m[key].append(list_m[k])
                k = k+1

        self.target_function_m = np.mean(np.mean(rcp_m['target_function']))
        self.target_function_data_m = np.mean(
            np.mean(rcp_m['target_function_data']))

        self.pdfmap_m = rcp_m['pdfmap'][0].copy()
        for pdfmapi in rcp_m['pdfmap'][1:]:
            for k in np.arange(len(self.pdfmap_m)):
                self.pdfmap_m[k] += pdfmapi[k]

        self.param_hat_m = np.average(rcp_m['sigma_wt_hat'], axis=0)
        nmax = np.argmax(self.target_function_m)
        self.param_max_m = rcp_m['sigma_wt_max'][nmax]

        tmp = rcp_m['sigma_wt_map']
        self.sigma_wt_map_m = tmp[0].copy()
        for irank in np.arange(1, len(tmp)):
            for imethod in np.arange(len(tmp[0])):
                self.sigma_wt_map_m[imethod] += tmp[irank][imethod]

        tmp = rcp_m['Zwt_map']
        self.Zwt_map_m = tmp[0].copy()
        for irank in np.arange(1, len(tmp)):
            for imethod in np.arange(len(tmp[0])):
                self.Zwt_map_m[imethod] += tmp[irank][imethod]

        tmp = rcp_m['Zbs_map']
        self.Zbs_map_m = tmp[0].copy()
        for irank in np.arange(1, len(tmp)):
            for imethod in np.arange(len(tmp[0])):
                self.Zbs_map_m[imethod] += tmp[irank][imethod]

    def m2local(self):
        """!Set the temporary results (_m) of the e_step as the final results. Used to consolidate all thread results after the multi-threaded execution of e_step. 
        @param None
        @return None
        """    

        self.pdfmap = self.pdfmap_m
        self.param_hat = self.param_hat_m
        self.param_max = self.param_max_m
        self.sigma_wt_map = self.sigma_wt_map_m
        self.modelwtmap = self.sigma_wt_map_m
        self.Zwt_map = self.Zwt_map_m
        self.Zbs_map = self.Zbs_map_m

    def init_histogram(self, minpar, maxpar, Npar):
        """!Set a histogram for each \f$m_{wt}\f$ parameter using N bins between minpar and maxpar. 
        @param minpar List of minimum values for the parameter histograms.   
        @param maxpar List of maximum values for the parameter histograms.   
        @param Npar List specifying the number of bins for each parameter histogram.   
        """    
        self.histticks = []
        self.pdfmap = []
        self.pre_pdfmap = []
        for i in np.arange(len(minpar)):
            dP = (maxpar[i]-minpar[i])/Npar[i]
            self.histticks.append(np.arange(Npar[i]+1)*dP+minpar[i])
            self.pre_pdfmap.append(np.zeros(Npar[i]))
            self.pdfmap.append(np.zeros(Npar[i]))
        self.pre_pdfmap = self.pdfmap.copy()  # np.array()
#        self.pdfmap=self.pdfmap#np.array()

        if isinstance(self.data, list):
            self.Rd = []
            self.R = []
            self.respmap = []
            self.respmap_ND = []
            self.respmap_NR = []
            self.modelmap = []
            self.modelwtmap = []

            for i in np.arange(len(self.data)):
                if isinstance(self.data[i].minR,list):
                    minR = np.min(self.data[i].data[:, -1])/2
                else:
                    minR = self.data[i].minR
                if isinstance(self.data[i].maxR,list):
                    maxR = np.max(self.data[i].data[:, -1])*2
                else:
                    maxR = self.data[i].maxR
                NN = 51
                if isinstance(self.data[i].ND,list):
                    ND = len(self.data[i].data)
                else:
                    ND = self.data[i].ND
                
                
                if self.model[i].linear:
                    self.Rd.append(np.linspace(minR, maxR, NN))                    
                else:
                    self.Rd.append(np.logspace(np.log10(minR), np.log10(maxR), NN))
                self.respmap.append(np.zeros([ND, NN-1]))
                self.respmap_ND.append(ND)  # Is it used?
                self.respmap_NR.append(NN-1)  # Is it used?


                if isinstance(self.data[i].minR,list):
                    minR = np.min(self.data[i].data[:, -1])/100
                else:
                    minR = self.data[i].minR
                if isinstance(self.data[i].maxR,list):
                    maxR = np.max(self.data[i].data[:, -1])*100
                else:
                    maxR = self.data[i].maxR
                NN = 51
                ND = len(self.model[0].vals)
                self.Y = self.model[0].mesh.Y
#                minR = np.min(self.data[i].data[:, -1])/100
#                maxR = np.max(self.data[i].data[:, -1])*100

                if self.model[i].linear:
                    self.R.append(np.linspace(minR, maxR, NN))
                else:
                    self.R.append(np.logspace(np.log10(minR), np.log10(maxR), NN))
                    
                self.modelmap.append(np.zeros([ND, NN-1]))
                self.modelwtmap.append(np.zeros([ND, NN-1]))

                # Extraño???
                self.Zwt_map = np.zeros(ND)
                self.Zbs_map = np.zeros(ND)

        else:
            if isinstance(self.data.minR,list):
                minR = np.min(self.data.data[:, -1])/2
            else:
                minR = self.data.minR
            if isinstance(self.data.maxR,list):
                maxR = np.max(self.data.data[:, -1])*2
            else:
                maxR = self.data.maxR
            if isinstance(self.data.ND,list):
                ND = len(self.data.data)
            else:
                ND = self.data.ND

            NN = 51
            self.Rd = np.logspace(np.log10(minR), np.log10(maxR), NN)

            self.respmap = np.zeros([ND, NN-1])
            self.respmap_ND = ND
            self.respmap_NR = NN-1


            if isinstance(self.data.minR,list):
                minR = np.min(self.data.data[:, -1])/100
            else:
                minR = self.data.minR
            if isinstance(self.data.maxR,list):
                maxR = np.max(self.data.data[:, -1])*100
            else:
                maxR = self.data.maxR
            NN = 51
            ND = len(self.model.vals)
            self.Y = self.model.mesh.Y

            self.R = np.logspace(np.log10(minR), np.log10(maxR), NN)
            self.modelmap = np.zeros([ND, NN-1])
            self.modelwtmap = np.zeros([ND, NN-1])

            self.Zwt_map = np.zeros(ND)
            self.Zbs_map = np.zeros(ND)

    def update_respmap(self, resp):
        """!Update the probability desity map of the forward responses from the sampled \f$m_{wt}\f$ models.          
        @param resp foward response of latest \f$m_{wt}\f$ model
        """
        if isinstance(resp, list):
            for ir in np.arange(len(resp)):
                N = len(resp[ir])
                for i in np.arange(N):
                    n = (self.Rd[ir][:-1] <= resp[ir][i]
                         ) & (self.Rd[ir][1:] > resp[ir][i])
                    self.respmap[ir][i, n] += 1
        else:
            N = len(resp)
            for i in np.arange(N):

                n = (self.Rd[:-1] <= resp[i]) & (self.Rd[1:] > resp[i])
                self.respmap[i, n] += 1

    def update_modelmap(self, model):
        """!Update the probability desity map of the sampled \f$m_{t}\f$ models.          
        @param model latest sampled \f$m_{t}\f$ model
        """    

        if isinstance(model, list):
            for ir in np.arange(len(model)):
                mod = model[ir].get_linear_vals()
                N = len(mod)
                for i in np.arange(N):
                    n = (self.R[ir][:-1] <= mod[i]) & (self.R[ir][1:] > mod[i])
                    self.modelmap[ir][i, n] += 1
        else:
            mod = model.get_linear_vals()
            N = len(mod)
            for i in np.arange(N):
                n = (self.R[:-1] <= mod[i]) & (self.R[1:] > mod[i])
                self.modelmap[i, n] += 1

    def update_modelwtmap_old(self, modelwt):
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

    def update_histogram(self, params):
        """!Update the histrograms of the sampled \f$m_{wt}\f$ parameters.          
        @param model latest sampled \f$m_{wt}\f$ parameters set.
        """            

        for i in np.arange(len(params)):
            histticks = self.histticks[i]
            nmax = len(histticks)
            maxH=histticks[nmax-1]
            nindex = (histticks[0:nmax-1] < params[i]) & (histticks[1:nmax] >= params[i])
            self.pdfmap[i][nindex] += 1

    def update_pre_histogram(self, params):
        None

    def update_expected_value(self, params):
        """!Update the expected values of the sampled \f$m_{wt}\f$ parameters.
        @param model latest sampled \f$m_{wt}\f$ parameters set.
        """            
        if(self.N < 1):
            self.expected_value = params.copy()
            self.max_value = params.copy()
            self.max_log_pdf = self.last_log_pdf_d

        else:
            N=self.N
            self.expected_value = (
                self.expected_value*N+params.copy())/(N+1)
            if self.last_log_pdf_d > self.max_log_pdf:
                self.max_value = params.copy()
                self.max_log_pdf = self.last_log_pdf_d

    def pdfplot(self):
        """!Plot \f$m_{wt}\f$ parameter Histograms.          
        @param None
        @return None
        """            

        plt.figure()
        N = int(np.ceil(len(self.params)/2))
        for i in np.arange(len(self.params)):
            plt.subplot(N, int(2), int(i+1))
            histticks = self.histticks[i]
            labels = []
            for j in np.arange(len(histticks)-1):
                labels.append(((histticks[j]+histticks[j+1])/2))
            plt.bar(labels, self.pdfmap[i], width=histticks[1]-histticks[0])
        return plt.gcf()

    def pre_pdfplot(self):
        N = int(np.ceil(len(self.params)/2))
        for i in np.arange(len(self.params)):
            plt.subplot(N, int(2), int(i+1))
            histticks = self.histticks[i]
            labels = []
            for j in np.arange(len(histticks)-1):
                labels.append(((histticks[j]+histticks[j+1])/2))
            plt.bar(labels, self.pre_pdfmap[i],
                    width=histticks[1]-histticks[0])

    def custom_cmap(self):
        """!Get a customized colormap.          
        @param None
        @return colormap(class)
        """            

        from matplotlib.colors import LinearSegmentedColormap
        colors = [(1, 1, 1), (0.8, 1, 0.8), (0.7, 0.7, 1),
                  (0.6, 0.6, 1), (0, 0, 1)]  # W -> G -> B
        n_bin = [3, 6, 10, 100]  # Discretizes the interpolation into bins
        cmap_name = 'my_list'
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=10)
        return cmap

    def hatplot(self, file):
        """!Plot and save the respmap (density probability of sampled \f$m_{bk}\f$ fordward responses).          
        @param file(string) filename to save the respmap plot 
        @return None
        """            

        if len(self.model.mesh.dX) > 1:
            print('FATAL ERROR CUCUCUCUCU!!!')
        else:
            # 1D Case
            plt.clf()
            datax = np.log10(self.data.data[:, -1])
            ddb = np.diff(datax)[0]/2
            dde = np.diff(datax)[-1]/2
            cmap = self.custom_cmap()
            respmap = np.reshape(
                self.respmap, [self.respmap_ND, self.respmap_NR])
            plt.pcolor(np.concatenate([[0], np.arange(len(datax))+1]), np.log10(
                self.Rd), np.log10(respmap.T/np.sum(respmap)), cmap=cmap)  # 'ocean_r')
            plt.plot(np.log10(self.data.data[:, -1]), 'k-')
            clb = plt.colorbar(orientation="horizontal",
                               fraction=0.08, aspect=20)
            clb.set_alpha(100)
            clb.set_label('Log10 Density')
            plt.xlabel('Data index')
            plt.ylabel('Log10 Apparent Resisivity [Ohm.m]')
            plt.savefig(file)

    def modelplot_old(self, file, vlim=[]):
        """!Plot and save the respmap (density probability of sampled \f$m_{bk}\f$ fordward responses).          
        @param file(string) filename to save the respmap plot 
        @return None
        """            

        for im in np.arange(len(self.model)):

            # ==========================================
            # P(Pars/sigma_bk,d) Plot
            # ==========================================
            if len(self.model[im].mesh.dX) > 1:
                None
            else:
                None

            # ==========================================
            # P(sigma_wt/sigma_bk,d) Plot
            # ==========================================

            if len(self.model[im].mesh.dX) > 1:
                plt.figure(figsize=(15, 6))

                # ==========================================
                # Mesh 2D
                # ==========================================
                if not(vlim):
                    Rmax = 2
                    Rmin = 4
                else:
                    Rmin = vlim[1]
                    Rmax = vlim[0]

                # ============ WT MODEL ================================
                plt.subplot(3, 1, 1)
                if self.model[im].linear:
                    dlogR = np.abs(np.diff(self.R[im])[0])
                else:
                    dlogR = np.abs(np.diff(np.log10(self.R[im]))[0])
                    
                dY = self.model[im].mesh.dY[0]
                Y = self.model[im].mesh.Ye  # -dY
                dX = np.min(self.model[im].mesh.dX[0])
                X = self.model[im].mesh.Xe  # -dY
                opt = 1
                if opt == 1:
                    if np.sum(self.modelwtmap)>0:
                        likewtmap = np.sum(
                            self.modelwtmap[0], axis=1)/np.sum(self.modelwtmap)
                    else: 
                        likewtmap=0
                else:
                    likewtmap = self.Zwt_map+self.Zbs_map
                    likewtmap = likewtmap/np.sum(likewtmap)

                likewtmap2d = np.reshape(
                    likewtmap, [len(self.model[im].mesh.dY), len(self.model[im].mesh.dX)])
                plt.pcolor(X, Y, likewtmap2d/dY/dX, cmap='ocean_r')
                plt.xlim([self.data[im].minX, self.data[im].maxX])

                self.flagplotydir=True
                if self.flagplotydir:
                    plt.ylim([self.data[im].maxY, self.data[im].minY])
                    plt.gca().invert_yaxis()
                    plt.gca().yaxis.set_inverted(True)
                else:
                    plt.ylim([-self.data[im].maxY, -self.data[im].minY])
                    plt.gca().yaxis.set_inverted(False)
                

                plt.colorbar()

                # ============ BK MODEL ================================
                plt.subplot(3, 1, 2)
                if self.model[im].linear:
                    dlogR = np.abs(np.diff(self.R[im])[0])
                else:
                    dlogR = np.abs(np.diff(np.log10(self.R[im]))[0])

                dY = self.model[im].mesh.dY[0]
                Y = self.model[im].mesh.Ye  # -dY
                dX = np.min(self.model[im].mesh.dX[0])
                X = self.model[im].mesh.Xe  # -dY
                modvals = self.model[im].get_linear_vals().copy()
                modvals[modvals >= 1e8] = np.nan

                if self.model[im].linear:
                   plt.pcolor(X, Y, np.reshape(modvals,
                          [len(Y)-1, len(X)-1]), cmap='jet', vmin=Rmin, vmax=Rmax)                
                else:
                    plt.pcolor(X, Y, np.log10(np.reshape(modvals,
                           [len(Y)-1, len(X)-1])), cmap='jet', vmin=Rmin, vmax=Rmax)

                plt.xlim([self.data[im].minX, self.data[im].maxX])
                if self.flagplotydir:
                    plt.ylim([self.data[im].maxY, self.data[im].minY])
                    plt.gca().yaxis.set_inverted(True)
                else:
                    plt.ylim([-self.data[im].maxY, -self.data[im].minY])
                    plt.gca().yaxis.set_inverted(False)

                plt.colorbar()

                # ============ BK MODEL + E(WT MODEL)================================
                plt.subplot(3, 1, 3)
                if self.model[im].linear:
                    dlogR = np.abs(np.diff(self.R[im])[0])
                else:
                    dlogR = np.abs(np.diff(np.log10(self.R[im]))[0])

                dY = self.model[im].mesh.dY[0]
                Y = self.model[im].mesh.Ye  # -dY
                dX = np.min(self.model[im].mesh.dX[0])
                X = self.model[im].mesh.Xe  # -dY

                self.model_wt_hat = self.param2model(
                    self.model, self.invmodel_ref, self.expected_value)
                modvals = self.model[im].model_sum(
                    self.model[im], self.model_wt_hat[im]).get_linear_vals().copy()
                if self.model[im].linear:
                    modvals[modvals > self.model[im].airval] = np.nan
                    plt.pcolor(X, Y, np.reshape(modvals,
                           [len(Y)-1, len(X)-1]), cmap='jet', vmin=Rmin, vmax=Rmax)
                else:
                    modvals[modvals >= 1e8] = np.nan
                    plt.pcolor(X, Y, np.log10(np.reshape(modvals,
                           [len(Y)-1, len(X)-1])), cmap='jet', vmin=Rmin, vmax=Rmax)
                plt.xlim([self.data[im].minX, self.data[im].maxX])
                if self.flagplotydir:
                    plt.ylim([self.data[im].maxY, self.data[im].minY])
                    plt.gca().yaxis.set_inverted(True)
                else:
                    plt.ylim([-self.data[im].maxY, -self.data[im].minY])
                    plt.gca().yaxis.set_inverted(False)

                plt.colorbar()

                plt.savefig(file+'_'+str(im)+'.jpg')

            else:
                # ==========================================
                # mesh 1D
                # ==========================================
                plt.figure(figsize=(6, 6))
                for im in np.arange(len(self.model)):
                    dlogR = np.abs(np.diff(np.log10(self.R[im]))[0])
                    dY = self.model[im].mesh.dY[0]
                    Y = self.model[im].mesh.Ye  

                    plt.pcolor(np.log10(self.R[im]), -Y, self.sigma_wt_map_m[im]/np.sum(
                        self.sigma_wt_map_m[im])/dY/dlogR, cmap='ocean_r')
                    Z = np.concatenate(
                        [[0], np.cumsum(self.model[im].mesh.dY)])

                    # Plot Background model
                    rho = (self.model[im].get_linear_vals())
                    last = []
                    for i in np.arange(len(Z)-1):
                        MZ = (-Z[i:i+2])
                        MR = (np.log10(np.array([rho[i], rho[i]])))
                        if i > 0:
                            plt.plot(
                                np.log10(np.array([last, rho[i]])), -np.array([Z[i], Z[i]]), 'k')
                        last = 10**MR[0]
                        plt.plot(MR, MZ, 'k')

                    # Plot model-wt expectation
                    axclb = plt.colorbar()
                    axclb.set_label('Density')
                    plt.ylabel('Depth(m)')
                    plt.xlabel('Log10 Resisivity [Ohm.m]')
                    plt.ylim([-np.max(Y), -np.min(self.model[im].mesh.Y)])
                    plt.title('target function: ' + str(self.target_function))

                    # ==========================================
                    # ==========================================
                    plt.savefig(file+'_'+str(im)+'.jpg')

    def pdfmap_average(self, pdfmap_list):
        N = len(pdfmap_list)
        av_pdfmap = []
        for j in np.arange(len(pdfmap_list[0])):
            av = np.zeros(len(pdfmap_list[0][j]))
            for pdfmap in pdfmap_list:
                av = av+pdfmap[j]
            av_pdfmap.append(av)
        return av_pdfmap

    def sigma_wt_hat_average(self, sigma_wt_hat_m):
        av = np.zeros(np.shape(sigma_wt_hat_m[0]))
        for sigma_wt_hat in sigma_wt_hat_m:
            av = av+sigma_wt_hat
        return av

    def set_param_proposal(self, function):
        self.get_param_proposal = function

    def set_param2model(self, function):
        self.param2model = function

    def update_modelwtmap1(self,modelwt):
        self.p.update_modelwtmap(modelwt)
        self.modelwtmap=self.p.modelwtmap
        self.Zwt_map=self.p.Zwt_map
        self.Zbs_map=self.p.Zbs_map
                    

    def set_parampdf_discretization(self, minpar, maxpar, Npar):
        self.init_histogram(minpar, maxpar, Npar)

    def is_model_changed(self, last_model, new_model):
        if isinstance(last_model, list):
            return False
        if len(last_model.vals) != (len(new_model.vals)):
            return False
        if np.sum(last_model.vals != new_model.vals) > 0:
            return True
        return False


    def MetropolisHasting_SaveProposal(self,log_pdf_d,log_pdf_data,log_pdf_model,params,proposed_model,proposed_model_wt):
        self.last_log_pdf_d = log_pdf_d
        self.params = params
        if isinstance(proposed_model, list):
            self.last_total_model = [im.copy_model()
                                             for im in proposed_model]
            self.last_model_wt = [im.copy_model()
                                          for im in proposed_model_wt]
        else:
            self.last_total_model = proposed_model.copy_model()
            self.last_model_wt = proposed_model_wt.copy_model()
        self.last_resp = self.resp.copy()
        self.last_log_pdf_data = log_pdf_data
        self.last_log_pdf_model = log_pdf_model

    def MetropolisHastingTest(self, log_pdf_data, log_pdf_model, log_pdf_proposal, params, proposed_model, proposed_model_wt):
        # Metropolis-hasting algorithm test to select the next model sample
        alfa = 1
        if (log_pdf_model < 0) & np.isinf(log_pdf_model):
            log_pdf_d=-np.inf
        else:
            if (log_pdf_proposal <0 ) & np.isinf(log_pdf_proposal):
                log_pdf_d=-np.inf
            else:
                log_pdf_d=log_pdf_data+log_pdf_model-log_pdf_proposal
        
        if not(self.last_log_pdf_d == []):
            
            diff = (log_pdf_d-self.last_log_pdf_d)/alfa
            if(diff > 0):
                self.MetropolisHasting_SaveProposal(log_pdf_d,log_pdf_data,log_pdf_model,params,proposed_model,proposed_model_wt)
            else:
                rand = np.random.rand()
                if rand > 0:
                    if diff > np.log(rand):
                        self.MetropolisHasting_SaveProposal(log_pdf_d,log_pdf_data,log_pdf_model,params,proposed_model,proposed_model_wt)
        else:
            self.MetropolisHasting_SaveProposal(log_pdf_d,log_pdf_data,log_pdf_model,params,proposed_model,proposed_model_wt)

        return self.last_log_pdf_d


    def update(self, model, invmodel_ref):

        #########################################################
        #0. Setup
        #########################################################
        self.model = model
        self.invmodel_ref = invmodel_ref
        data = self.data

        #########################################################
        #1. Getting Param Proposal
        #########################################################
        # get a rando..m parameter list for selecting a proposed model.
        params = self.get_param_proposal(self.model)
        self.proposed_model_wt = self.p.param2model(model, invmodel_ref, params)
        prop_logpdf = self.p.get_proposal_logpdf(params,model_lst=model)
        prior_logpdf = self.p.get_prior_logpdf(params,model_lst=model)

        #########################################################
        #2. Convert Params to a Sigma_wt model
        #########################################################
#        log_sum=False
#        if log_sum==False:
        if isinstance(self.proposed_model_wt, list):
            self.proposed_total_model = []
            for im in np.arange(len(self.proposed_model_wt)):                
                self.proposed_total_model.append(self.model[im].model_sum(
                self.proposed_model_wt[im], model[im]))  # proposed_model + background_model
        else:
            self.proposed_total_model = self.model.model_sum(
                self.proposed_model_wt, model)  # proposed_model + background_model
        # else:
        #     if isinstance(self.proposed_model_wt, list):
        #         self.proposed_total_model = []
        #         for im in np.arange(len(self.proposed_model_wt)):                
        #             self.proposed_total_model.append(self.model[im].model_log_sum(
        #                 self.proposed_model_wt[im], model[im]))  # proposed_model + background_model
        #     else:
        #         self.proposed_total_model = self.model.model_log_sum(
        #             self.proposed_model_wt, model)  # proposed_model + background_model


        #########################################################
        #3. Metropolis-Hasting Algorithm.
        #########################################################
        log_pdf_data = self.log_post_pdf_d(self.proposed_total_model, self.data)#prior_logpdf-prop_logpdf
        self.log_pdf_data = log_pdf_data

        # Metropolis hasting testing to accept/reject the proposed model
        self.MetropolisHastingTest(log_pdf_data, prior_logpdf, prop_logpdf, params, self.proposed_total_model, self.proposed_model_wt)


        #########################################################
        #4. Evaluating Target Function.
        #########################################################
        # The last_proposed_model is saved to avoid calulating many times the same forward response.

        if isinstance(self.target_function, list):
            self.target_function = 0
        if isinstance(self.target_function_data, list):
            self.target_function_data = 0
        if isinstance(self.target_function_model, list):
            self.target_function_model = 0
        if isinstance(self.target_function_sigma_bk, list):
            self.target_function_sigma_bk = 0

        self.target_function_data = (
                self.target_function_data*self.N+self.last_log_pdf_data)/(self.N+1)
    
        self.target_function_model = (
                self.target_function_model*self.N+self.last_log_pdf_model)/(self.N+1)
    
        self.target_function = self.target_function_data + self.target_function_model

        if log_pdf_data < -1E6:
            None
        else:
            self.target_function_sigma_bk = np.log((np.exp(self.target_function_sigma_bk)*self.N + np.exp(log_pdf_data))/(self.N+1))

        #########################################################
        #5. Updating the histograms, expected values, maps, etc..
        #########################################################
        self.p.set_last_param(self.params)
        self.update_expected_value(self.params)
        self.update_histogram(self.params)
        self.update_respmap(self.last_resp)
        self.update_modelmap(self.last_total_model)
        self.update_modelwtmap(self.last_model_wt)
        self.N += 1
        return

    def modelnorm(self, total_model, modelinv_ref):
        if isinstance(total_model, list):
            result = 0
            for i in np.arange(len(total_model)):
                result += - \
                    np.mean((total_model[i].get_log_vals() -
                            modelinv_ref[i].get_log_vals())**2)
        else:
            result = -np.mean((total_model.get_log_vals() -
                              modelinv_ref.get_log_vals())**2)
        return result

    def log_post_pdf_d(self, proposed_model, data):
        #
        # Date:    Nov/2023
        # Description: Compute the log of the likelihood function P(data|model) as:
        #    data.misfit(data,forward(proposed_model))
        #
        # Parameters : proposed_model: model object
        #              data: data object
        #
        # Return     : (float)log-likelihood
        #

        if isinstance(data, list):  # Multi-data or Multi-method
            self.resp = []
            self.err = 0

            for i, data in enumerate(self.data):
                data.method.set_model(proposed_model[i].get_linear_vals())
                self.resp.append(data.forward())
                self.err += data.misfit(self.resp[i])
        else:  # Single method
            self.data.method.set_model(proposed_model.get_linear_vals())
            self.resp = self.data.forward()
            self.err = data.misfit(self.resp)
        self.nforward += 1
        return self.err



    def get_max_param(self):
     #       self.model_wt_max=self.param2model(self.model,np.array(self.max_value))
     #       return self.model_wt_max.vals
        return self.max_value

    def get_expectation(self):
        return self.expected_value

    def get_pdfmap(self):
        return self.pdfmap

    def get_modelwtmap(self):
        return self.modelwtmap

    def get_Zwtmap(self):
        return self.Zwt_map

    def get_Zbsmap(self):
        return self.Zbs_map

    def save_modelwtmap(self, file):
        if isinstance(self.model, list):
            for im in np.arange(len(self.model)):
                fid = open(file[:-4]+'['+str(im)+']'+file[-4:], 'wt')
                Rm = np.exp(
                    (np.log(self.R[im][0:-1])+np.log(self.R[im][1:]))/2)
                if len(self.model[im].mesh.dX) > 1:
                    fid.write(str(len(
                        Rm))+" "+str(len(self.model[im].mesh.X))+" "+str(len(self.model[im].mesh.Y))+'\n')
                else:
                    fid.write(str(len(Rm))+" " +
                              str(len(self.model[im].mesh.Y))+'\n')

                tmp = ''
                for i in np.arange(len(Rm)):
                    tmp += str(Rm[i])+' '
                fid.write(tmp+'\n')

                # Detecting if is a 2D or a 1D case.
                if len(self.model[im].mesh.dX) > 1:
                    # 2D
                    tmp = ''
                    for i in np.arange(len(self.model[im].mesh.X)):
                        tmp += str(self.model[im].mesh.X[i])+' '
                    fid.write(tmp+'\n')

                    tmp = ''
                    for i in np.arange(len(self.model[im].mesh.Y)):
                        tmp += str(self.model[im].mesh.Y[i])+' '
                    fid.write(tmp+'\n')
                else:
                    # 1D
                    tmp = ''
                    for i in np.arange(len(self.model[im].mesh.Y)):
                        tmp += str(self.model[im].mesh.Y[i])+' '
                    fid.write(tmp+'\n')

                tmp = ''
                for i in np.arange(len(self.modelwtmap[im][:, 0])):
                    for j in np.arange(len(self.modelwtmap[im][0, :])):
                        fid.write(str(self.modelwtmap[im][i, j])+'\n')

                fid.close()

        else:
            fid = open(file, 'wt')
            Rm = np.exp((np.log(self.R[0:-1])+np.log(self.R[1:]))/2)
            if len(self.model.mesh.dX) > 1:
                fid.write(str(len(Rm))+" "+str(len(self.model.mesh.X)
                                               )+" "+str(len(self.model.mesh.Y))+'\n')
            else:
                fid.write(str(len(Rm))+" "+str(len(self.model.mesh.Y))+'\n')

            tmp = ''
            for i in np.arange(len(Rm)):
                tmp += str(Rm[i])+' '
            fid.write(tmp+'\n')

            # Detecting if is a 2D or a 1D case.
            if len(self.model.mesh.dX) > 1:
                # 2D
                tmp = ''
                for i in np.arange(len(self.model.mesh.X)):
                    tmp += str(self.model.mesh.X[i])+' '
                fid.write(tmp+'\n')

                tmp = ''
                for i in np.arange(len(self.model.mesh.Y)):
                    tmp += str(self.model.mesh.Y[i])+' '
                fid.write(tmp+'\n')
            else:
                # 1D
                tmp = ''
                for i in np.arange(len(self.model.mesh.Y)):
                    tmp += str(self.Y[i])+' '
                fid.write(tmp+'\n')

            tmp = ''
            for i in np.arange(len(self.modelwtmap[:, 0])):
                for j in np.arange(len(self.modelwtmap[0, :])):
                    fid.write(str(self.modelwtmap[i, j])+'\n')

            fid.close()
        return

    def save_Zwtmap(self, file):
        if isinstance(self.model, list):
            inx = 0
            for imodel in self.model:
                modeltmp = imodel.copy_model()
                modeltmp.set_linear_vals(self.Zwt_map[inx])
                modeltmp.save_model(file[:-3]+str(inx)+file[-3:])
                inx = inx+1
        else:
            modeltmp = self.model.copy_model()
            modeltmp.set_linear_vals(self.Zwt_map)
            modeltmp.save_model(file)
        return

    def save_Zbsmap(self, file):
        if isinstance(self.model, list):
            inx = 0
            for imodel in self.model:
                modeltmp = imodel.copy_model()
                modeltmp.set_linear_vals(self.Zbs_map[inx])
                modeltmp.save_model(file[:-3]+str(inx)+file[-3:])
                inx = inx+1
        else:
            modeltmp = self.model.copy_model()
            modeltmp.set_linear_vals(self.Zbs_map)
            modeltmp.save_model(file)
        return

    def save_paramsmap(self, file):
        if self.p.get_conditional_like==[]:
            return
         
        self.model2param(self.p.get_conditional_like)
        if isinstance(self.model, list):
            for ix, Phi in enumerate(self.Phi):
                fid = open(file[:-4]+'.'+str(ix)+file[-4:], 'wt')
                Phim = (Phi[0:-1]+Phi[1:])/2  # ????
                fid.write(str(len(Phim))+" "+str(len(self.Y))+'\n')
                tmp = ''
                for i in np.arange(len(Phim)):
                    tmp += str(Phim[i])+' '
                fid.write(tmp+'\n')

                tmp = ''
                for i in np.arange(len(self.Y)):
                    tmp += str(self.Y[i])+' '
                fid.write(tmp+'\n')

                tmp = ''

                for i in np.arange(len(self.parammap[ix][:, 0])):
                    for j in np.arange(len(self.parammap[ix][0, :])):
                        fid.write(str(self.parammap[ix][i, j])+'\n')
                fid.close()
        else:
            fid = open(file, 'wt')
            Phim = (self.Phi[0:-1]+self.Phi[1:])/2
            fid.write(str(len(Phim))+" "+str(len(self.Y))+'\n')
            tmp = ''
            for i in np.arange(len(Phim)):
                tmp += str(Phim[i])+' '
            fid.write(tmp+'\n')

            tmp = ''
            for i in np.arange(len(self.Y)):
                tmp += str(self.Y[i])+' '
            fid.write(tmp+'\n')

            tmp = ''
            for i in np.arange(len(self.parammap[:, 0])):
                for j in np.arange(len(self.parammap[0, :])):
                    fid.write(str(self.parammap[i, j])+'\n')
            fid.close()
        return

    def get_super_target_function(self):
        return self.super_target_function

    def get_target_function(self):
        return self.target_function_sigma_bk

    def get_target_function_data(self):
        return self.target_function_data

    def get_max_target_function(self):
        return self.max_log_pdf

    def model2param(self, likephi_function):
        if isinstance(self.model, list):
            self.Phi = []
            self.parammap = []
            RR = []
            for i, iR in enumerate(self.R):
                RR.append(np.exp((np.log(iR[:-1])+np.log(iR[1:]))/2))
            likephi, Phi, Phi_e = likephi_function(RR)
            self.Phi.append(Phi_e)
            self.likephi = likephi

            for i, iR in enumerate(self.R):
                modelmap = self.modelwtmap[i]
                a = np.shape(modelmap)
                CC = np.zeros([a[0], len(likephi[:, 0])])
                for n in np.arange(a[0]):
                    for ir in np.arange(len(RR[i])):
                        for ip in np.arange(len(likephi[:, 0])):
                            if not(np.isinf(modelmap[n, ir]*likephi[ip, ir])):
                                if not(np.isnan(modelmap[n, ir]*likephi[ip, ir])):
                                    CC[n, ip] = CC[n, ip] + \
                                        modelmap[n, ir]*likephi[ip, ir]
                self.parammap.append(CC)
        else:
            RR = np.exp((np.log(self.R[:-1])+np.log(self.R[1:]))/2)
            likephi, Phi, Phi_e = likephi_function(RR)
            self.Phi = Phi_e
            modelmap = self.modelwtmap
            a = np.shape(modelmap)
            CC = np.zeros([a[0], len(likephi[:, 0])])
            for n in np.arange(a[0]):
                for ir in np.arange(len(RR)):
                    for ip in np.arange(len(likephi[:, 0])):
                        if not(np.isinf(modelmap[n, ir]*likephi[ip, ir])):
                            if not(np.isnan(modelmap[n, ir]*likephi[ip, ir])):
                                CC[n, ip] = CC[n, ip] + \
                                    modelmap[n, ir]*likephi[ip, ir]
            self.parammap = CC
        return

    def smoothgrid(self, X, Y, gridpoints, colormap, vmin=0, vmax=1):
        points = []
        values = []
        for i in np.arange(len(X)):
            for j in np.arange(len(Y)):
                if gridpoints[j, i] > 0:
                    points.append([X[i], Y[j]])
                    values.append(gridpoints[j, i])
        interp = scipy.interpolate.CloughTocher2DInterpolator(
            points, values, fill_value=np.nan, tol=1e-6, maxiter=400, rescale=True)

        XXX = np.linspace(np.min(X), np.max(X), 200)
        YYY = np.linspace(np.min(Y), np.max(Y), 300)

        XX, YY = np.meshgrid(XXX, YYY)
        ZZ = interp(XX, YY)
        plt.pcolormesh(XX, YY, ZZ, shading='auto',
                       cmap=colormap, vmin=vmin)
        ax = plt.colorbar()
        return ax

    def save_pdfparams(self, file):
        fid = open(file, 'wt')
        for i in np.arange(len(self.histticks)):
            tmp = ''
            for val in self.histticks[i]:
                tmp += str(val)+' '
            fid.write(tmp+'\n')
            tmp = ''
            for val in self.pdfmap[i]:
                tmp += str(val)+' '
            fid.write(tmp+'\n')
        fid.close()

    def get_respmap(self):
        rcp = []
        if isinstance(self.respmap, list):
            self.respmap_ND = [len(irespmap[:, 0])
                               for irespmap in self.respmap]
            self.respmap_NR = [len(irespmap[0, :])
                               for irespmap in self.respmap]
            rcp = [np.reshape(self.respmap[ir], self.respmap_ND[ir]*self.respmap_NR[ir])
                   for ir in np.arange(len(self.respmap))]
            if len(rcp) == 1:
                rcp = rcp[0]
        else:
            self.respmap_ND = len(self.respmap[:, 0])
            self.respmap_NR = len(self.respmap[0, :])
            rcp = np.reshape(self.respmap, self.respmap_ND*self.respmap_NR)
        return rcp

    def get_model_wt_hat():
        None


class m_step:
    method = []
    data = []
    magic_lmd = []

    def __init__(self, setup, datax, rank=0):

        self.data = datax
        self.model = model()
        if self.data.linear:
            self.model.linear=True
        self.setup = setup

        if setup.externalmodel_ref:
            model_file = setup.modelref_file
            model_mesh = setup.model_mesh
            self.invmodel_ref = model()
            if self.data.linear:
                self.invmodel_ref.linear=True                
            self.invmodel_ref.read(model_file, model_mesh)
            if setup.externalmodel:
                model_file = setup.model_file
                model_mesh = setup.model_mesh
                self.model.read(model_file, model_mesh)
                self.data.ind_active = self.model.get_linear_vals()<1e8                
            else:
                self.model = self.invmodel_ref.set_constant_model(
                    setup.reference_value)
        else:
            if setup.externalmodel:
                model_file = setup.model_file
                model_mesh = setup.model_mesh
                self.model.read(model_file, model_mesh)
                self.data.ind_active = self.model.get_linear_vals()<1e8
            else:
                self.model.set_model_from_data(self.data)

            self.invmodel_ref = self.model.copy_model()

        if setup.external_constant_model==True:
            vals=self.invmodel_ref.get_linear_vals()
            vals[self.data.ind_active]=setup.reference_value
            self.invmodel_ref.set_linear_vals(vals)
            self.model = self.invmodel_ref.copy_model()

        self.data.init_method(self.model)
        self.method = self.data.method

        #if rank == 0:
        if (setup.invert_initial_bkmodel == True) and (rank==0):
                self.method.collingRate = setup.invparameters0[0]
                self.method.beta = setup.invparameters0[1]
                self.method.minbeta = setup.invparameters0[2]
                self.method.collingFactor = setup.invparameters0[3]
                self.method.alpha_x = setup.invparameters0[4]
                self.method.alpha_y = setup.invparameters0[5]
                self.method.alpha_z = setup.invparameters0[6]
                self.method.alpha_s = setup.invparameters0[7]
                self.method.chifact = setup.invparameters0[8]
                self.method.set_modelini(self.model.get_linear_vals())
                self.method.set_modelref(self.model.get_linear_vals())
                self.model = self.invert()[0]                
        else:
                self.method.collingRate = setup.invparameters[0]
                self.method.beta = setup.invparameters[1]
                self.method.minbeta = setup.invparameters[2]
                self.method.collingFactor = setup.invparameters[3]
                self.method.alpha_x = setup.invparameters[4]
                self.method.alpha_y = setup.invparameters[5]
                self.method.alpha_z = setup.invparameters[6]
                self.method.alpha_s = setup.invparameters[7]
                self.method.chifact = setup.invparameters[8]
                self.method.set_modelini(self.model.get_linear_vals())
                self.method.set_modelref(self.model.get_linear_vals())
                #self.invmodel_ref = self.invert()[0]

        self.model0 = self.invmodel_ref.copy_model()
            
    def init(self, data, model_wt_hat, model_wt_max, model, ref, invmodel_ref, invparameters=[]):

        self.data = data
        self.method = data.method
        self.model = model
        self.model_wt = model_wt_hat
        self.modelini = model.copy_model()

        self.linsolution_flag = True
        self.logsolution_flag = False
        
        if isinstance(ref, float) or isinstance(ref, int):
            self.model0 = model.set_constant_model(ref)
        else:            
            self.model0 = ref

        self.modelref = self.model0.copy_model()
            
        self.method.set_modelini(self.modelini.get_linear_vals())
        self.method.set_modelref(self.modelref.get_linear_vals())

        if len(invparameters) > 0:
            self.method.collingRate = invparameters[0]
            self.method.beta = invparameters[1]
            self.method.minbeta = invparameters[2]
            self.method.collingFactor = invparameters[3]
            self.method.alpha_x = invparameters[4]
            self.method.alpha_y = invparameters[5]
            self.method.alpha_z = invparameters[6]
            self.method.alpha_s = invparameters[7]
            self.method.chifact = invparameters[8]

    def setW(self, model_wt):
        #Not used anymore
        
        B = np.diff(model_wt.get_log_vals())
        C = B/np.dot(B, B)**0.5
        M = np.multiply(np.matrix(C).T, C)

        NA = len(B)
        MD = np.zeros([NA, NA+1])
        ind = np.arange(NA)
        MD[ind[:-1], ind[:-1]] = -1
        MD[ind[:-1], ind[1:]] = 1
        MD = np.matrix(MD)
        W = np.matmul(MD.T, np.matmul(M, MD))
        return W

    def invert(self):
        nbeta = 1
        nmid = (nbeta-1)/2
        alfa = 3
        beta = self.method.beta
        bk_model_list = []
        for i in np.arange(nbeta):
            beta_i = beta*alfa**(i-nmid)
            print("BETA_I:", beta_i)
            self.method.beta = beta_i
            W = []
            res = self.method.invert(W=W)
            invmodel = self.model.copy_model()
            invmodel.set_linear_vals(res)
            self.invmodel = invmodel

            self.method.set_model(invmodel.get_linear_vals())
            dpred = self.method.fwd()
            rms = self.data.misfit(dpred)
            print('RMS:', rms)

        return invmodel, rms

    def run(self, lmd=0):
        """
        Compute an M-Step iteration to determine the background model $\sigma_{bk}^{n+1}$ that maximizes the following target function.\\  
        \[
        \max_{\sigma_{bk}^{n+1}} \lVert d - F(\sigma_{bk}^{n+1} + \hat{\sigma}_{wt}) \rVert^2_{C_d^{-1}} + 
        \lambda \lVert \sigma_{bk}^{n+1} - \sigma_{ref} \rVert^2_{C_m^{-1}} 
        \]
        where:\\
        $\sigma_{bk}$: Background model
        $\sigma_{ref}$: Background reference model
        $\hat{\sigma_{wt}}$: Expected groundwater model
        $\lambda$: Regularizaton weighting factor
        $C_d$: Data Covariance 
        $C_m$: Background Model Covariance

        Args:
            lmd (float): not used (included only for call compatibility)
    
        Returns:
            bk_model (class): Model class containing the computed background model. 
            rms1 (float): misfit between data and the responses of the total model (background model + expected groundwater model).  
            rms2 (float): misfit between data and the responses of the background model.
            priorPDF (float): Value of the background model prior probability density.
    
        """

        beta = self.method.beta
        W = []

        self.method.set_modelini(self.model.get_linear_vals())#self.model.get_linear_vals())#np.concatenate(
        self.method.set_modelref(self.model0.get_linear_vals())#*0+(model_0))#np.concatenate(
        res = self.method.invert_step2(self.model_wt.get_linear_vals(),W=W,magic_lmd=self.magic_lmd)

        invmodel = self.model.copy_model()
        invmodel.set_linear_vals(res)
        self.invmodel = invmodel
        bk_model = invmodel

        self.method.set_model(bk_model.get_linear_vals())
        dpred = self.method.fwd()
        rms2 = self.data.misfit(dpred)

        self.method.set_model(bk_model.model_sum(bk_model, self.model_wt).get_linear_vals())
        dpred = self.method.fwd()
        rms1 = self.data.misfit(dpred)

        self.priorPbk = 0#self.method.inv.invProb.beta*phi_m*np.abs(rms1/phi_d)

        self.method.beta = self.method.beta/self.method.collingFactor
        if self.method.beta < self.method.minbeta:
            self.method.beta = self.method.minbeta
            
        return bk_model, rms1, rms2, self.priorPbk


    def set_bk_model(self, model):
        self.model = model.copy_model()

class multi_m_step:
    #
    # Date:    Nov/2023
    # Description: Class to define a multi method m_step.
    #
    #

    def __init__(self, setup_lst, data_lst, rank=0):
        self.m_step = [m_step(setup_lst[i], data_lst[i], rank=rank)
                       for i in np.arange(len(setup_lst))]
        self.model = [self.m_step[i].model for i in np.arange(len(setup_lst))]
        self.data = data_lst
        self.invmodel_ref = [self.m_step[i].invmodel_ref for i in np.arange(len(setup_lst))]#self.model
        self.model0 = [self.m_step[i].model0 for i in np.arange(len(setup_lst))]#self.model

    def init(self, data_lst, model_wt_hat_lst, model_wt_max_lst, model_lst, ref_lst, invmodel_ref_lst, invparameters=[]):
        self.model = []
        self.model0 = []
        self.invmodel_ref = []
        for i in np.arange(len(self.m_step)):
            if invparameters:
                self.m_step[i].init(data_lst[i], model_wt_hat_lst[i], model_wt_max_lst[i],
                                    model_lst[i], ref_lst[i], invmodel_ref_lst[i], invparameters=invparameters[i])
            else:
                self.m_step[i].init(data_lst[i], model_wt_hat_lst[i], model_wt_max_lst[i], 
                                    model_lst[i], ref_lst[i], invmodel_ref_lst[i])

            self.model.append(self.m_step[i].model)
            self.invmodel_ref.append(self.m_step[i].invmodel_ref)
            self.model0.append(self.m_step[i].model0)

    def invert(self):
        return [im_step.invert() for im_step in self.m_step]

    def run(self, lmd=0):
        result = []
        if isinstance(lmd, list):
            result = [self.m_step[i].run(lmd[i])
                      for i in np.arange(len(self.m_step))]
        else:
            result = [self.m_step[i].run(lmd)
                      for i in np.arange(len(self.m_step))]
        return [ir[0] for ir in result], [ir[1] for ir in result], [ir[2] for ir in result], [ir[3] for ir in result]


    def set_bk_model(self, model_lst):
        self.model = []
        for i in np.arange(len(self.m_step)):
            self.m_step[i].set_bk_model(model_lst[i])
            self.model.append(self.m_step[i].model)

class none_class:
    
    def __init__(self):
        self.model_ref=[]
        self.model_inv=[]
        self.model_ini=[]

    def set_topo(self):
        self.topo=[]

    def set_mesh(self,meshin,ind_active=[]):
        None

    def set_bkmapping(self,wtmodel):
        None

    def set_data2(self,datax,data2,sigma_per=0.05):                
        self.data=datax[:,-1]
        
    def set_data(self,datax_unsort,sigma_per=0.05):                
        self.data=datax_unsort[:,-1]
        
    def set_halfmodel(self,Rho):
        None

    def set_modelref(self,model):
        None

    def set_modelini(self,model):
        None

    def set_model(self,model):
        self.model=np.log(model)

    def getfwd(self):
        return self.data

    def fwd(self):
        N=len(self.model)
        bk_ref=np.zeros(N)+4.63
        bk_ref[:int(N/5)]=1
        return self.data*0+np.sum(np.abs(bk_ref-self.model))/len(self.data)/len(self.model)
    
    def invert(self,W=[]):
        return 1

    def get_regularization2(self):
        return 1

    def get_regularization(self,model):
        return 1

    def invert_step(self,W=[]):
        return 1

    def invert_step2(self,wtmodel,W=[]):
        return 1
    