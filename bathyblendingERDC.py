# -*- coding: utf-8 -*-
"""
 - 
@author: Jonghyun Harry Lee 
@version 0.1 
@date 10/17/2023

This is a template script to be used with the Bathy Blending App. The model object should be able to do three things when called upon in the app:
    - load in the data in a cbathy format by use of the _load_data() 
    - make predictions and saves them by use of the inv() method
    - plot results

Tested with Python 3.10.12, Numpy 1.23.5 and Scipy 1.11.3
"""

import numpy as np
from numpy import log, log2, tanh, cosh
import scipy as sp
from scipy.io import loadmat
from scipy.sparse import spdiags, eye, kron, diags
from scipy.sparse.linalg import cg, LinearOperator
from scipy.optimize import root_scalar
import timeit
import inspect
from scipy.sparse import csr_matrix
from scipy.optimize import fminbound
import matplotlib.pyplot as plt
from os.path import join

plt.rcParams.update({'font.size': 12})

class BathyBlending:
    def __init__(self, cbathy_dir = './cbathy', pbt_dir = './pbt', survey_dir ='./survey', output_dir = './', params = None):
        """

        Parameters
        ----------
        
        output_dir : str
            directory where output/result files are located 
        
        Returns
        -------
        None.

        """
        self.cbathy_dir = cbathy_dir 
        self.pbt_dir = pbt_dir 
        self.survey_dir = survey_dir 
        self.output_dir = output_dir
        self.use_testdata = None

        # if parameters are not defined, use default values
       
        # Inverse modeling parameters
        self.Lambda = 1.0 # regularization parameter related to inverse covariance/precision matrix
        self.obs_std = 0.05 # obs std

        # GN (outer loop)- CG (inner loop) optimization method default parameters 
        self.max_gn_iter = 30
        self.max_cg_iter = 500
        self.cg_tol = 1.e-6
        self.g_tol = 1.e-4
        
        # cBathy data screen parameters; default values suggested by CBathy and users can make changes
        self.min_skill = 0.5  
        self.min_lam1 = 10.0 
        self.hfB_max = 10.
        self.hfB_min = 0.25
        self.h_min = None
        self.h_max = None

        self.h_pbt = None
        self.x_pbt = None
        self.y_pbt = None

        self.h_cbathy = None
        self.x_cbathy = None
        self.y_cbathy = None

        self.h_survey = None
        self.x_survey = None
        self.y_survey = None 

        self.k = None
        self.fB = None

        self.mydate = None
        self.mydatetext = None
        
        self.h_prior = None
        self.res_init = None
        self.res_prior = None
        self.h_direct = None

        self.h_cur = None
        self.k_cur = None
        self.k_cur_init = None

        self.rmse = None
        self.rmse_init = None
        
        self.obs_std_variable = None
        
        self.kErr = None
        self.KErr_multiplier = None # KErr multiplied by KErr_multiplier

        self.hTempErr = None

        self.skill = None
        self.lam1 = None

        # read/overwrite default parameters
        for key, value in params.items(): 
            setattr(self, key, value)

        self._load_data()

        # construct Sqrt Inverse Covariance/Precision Matrix 
        self.L = self._ConstructSqrtPrecisionMat()

    def _load_data(self):
        """
            Read CBathy Parameters. For now, we use downloaded 
            Domain configuration and k-fB data screening
            - k-fB data chosen based on cbathy data quality control criteria (see below) as well as cbathy estimate locations        
        

        """
        print("## 1. Initialize BathyBlending object")
        print("\t- Loading Data... ")
        if self.use_testdata is not None:
            self.use_testdata = int(self.use_testdata)
            # choose the date
            testdate = ('03272017','11222017','10062019')
            textdate = ('3/27/2017','11/22/2017','10/6/2019')
            
            if 0 <= self.use_testdata <= 2:
                pass
            else:
                self.use_testdata = 0

            date_filename = testdate[self.use_testdata]
            
            self.mydate = testdate[self.use_testdata]
            self.mydatetext = textdate[self.use_testdata]
        else:
            # ii = 0 # for now
            # date = ('032717','042916','112217')
            # self.mydate = date[ii]
            if self.mydate is None:
                self.mydate = None # for now
            date_filename = ''
            #self.mydate
            #self.mydatetext


        # Read estimates from pbt, cbathy, and survey
        self.h_pbt = loadmat(join(self.pbt_dir,date_filename + 'pbt_h_land.mat'))['h']
        self.x_pbt = loadmat(join(self.pbt_dir,date_filename + 'pbt_x_land.mat'))['x']
        self.y_pbt = loadmat(join(self.pbt_dir,date_filename + 'pbt_y_land.mat'))['y']

        self.h_cbathy = loadmat(join(self.cbathy_dir, date_filename + 'cbathyv2_h.mat'))['h']
        self.x_cbathy = loadmat(join(self.cbathy_dir, date_filename + 'cbathyv2_x.mat'))['xm']
        self.y_cbathy = loadmat(join(self.cbathy_dir, date_filename + 'cbathyv2_y.mat'))['ym']
        self.tide = loadmat(join(self.cbathy_dir, date_filename + 'cbathyv2_tide.mat'))['zt'][0][0]
        print('\t- tide is set to ' + str(self.tide) + ' m')

        # file exist check
        self.kErr = loadmat(join(self.cbathy_dir, date_filename + 'cbathyv2_kErr.mat'))['kErr'] #KErr
        self.hTempErr = loadmat(join(self.cbathy_dir, date_filename + 'cbathyv2_hTemp.mat'))['hTemp'] #hTemp
        self.skill = loadmat(join(self.cbathy_dir, date_filename + 'cbathyv2_skill.mat'))['skill']
        self.lam1 = loadmat(join(self.cbathy_dir, date_filename + 'cbathyv2_lam1.mat'))['lam1']

        # read observation from cbathy
        self.k = loadmat(join(self.cbathy_dir, date_filename + 'cbathyv2_k.mat'))['k']
        self.fB = loadmat(join(self.cbathy_dir, date_filename + 'cbathyv2_fB.mat'))['fB']
    
        # survey data
        self.h_survey = loadmat(join(self.survey_dir, date_filename + 'survey_h.mat'))['h']
        self.x_survey = loadmat(join(self.survey_dir, date_filename + 'survey_x.mat'))['x']
        self.y_survey = loadmat(join(self.survey_dir, date_filename + 'survey_y.mat'))['y']

        if ((self.x_pbt - self.x_cbathy).sum() != 0.) or ((self.y_pbt - self.y_cbathy).sum() != 0.):
            #print('pbt and cbathy grids are different!')
            ValueError('pbt and cbathy grids are different! Please check the data and make them consistent.')

        self.x, self.y = np.copy(self.x_pbt), np.copy(self.y_pbt)

        # replace offshore NaN values to nanmax(h_pbt) - this should be optional
        # self.h_pbt[(self.h_pbt==0.) & (self.x > 400)] = np.nanmax(self.h_pbt)  

        # nearest neighbor interpolation to fill in NaN value
        from scipy.interpolate import NearestNDInterpolator
        mask = np.where(~np.isnan(self.h_pbt))
        interp = NearestNDInterpolator(np.transpose(mask), self.h_pbt[mask])
        self.h_pbt = interp(*np.indices(self.h_pbt.shape))
        
        # pbt - tide subtracted; this may become a future issue and needs to be fixed.
        self.h_pbt = self.h_pbt + self.tide
        ## h_cbathy : f_combined needs to be subtracted by tide for visualization/comparison 
        self.h_cbathy = self.h_cbathy - self.tide

        # Domain configuration and k-fB data screening
        # k-fB data chosen based on cbathy data quality control criteria (see below) as well as cbathy estimate locations

        """- Cbathy param examples for data quality control:
                    MINDEPTH: 0.2500
                        QTOL: 0.5000 % reject skill below this in csm
                        minLam: 10

        **Observation screening based on the direct solution if skill_ij < 0.5, lam1 < 10. or h_ij < 0.25** (included h_ij > 10 MAXDEPTH: 10.0 as well)
        """

        # domain parameters 
        self.nx, self.ny = self.h_cbathy.shape[1], self.h_cbathy.shape[0] # 73, 81
        self.nh = self.nx*self.ny # number of unknown h points
        self.nfB = self.fB.shape[2] # number of frequncies default (cBathy): 4
        
        if self.h_min is None:
            self.h_min = 0.001 # define minimum h to evaluate 

        if self.h_max is None:
            self.h_max = 10. # define minimum h to evaluate 

        # read indices  - find locations where 1) k values exist (~isnan(k)) AND 2) h_cbathy exist
        self.obsidx_k = ~np.isnan(self.k)
        
        print("\t- # of wave observations from cBathy phase 1: %d" % np.sum(self.obsidx_k))

        # Screening 

        #self.nobs = self.obsidx.sum()
        #self.obs = self.k[self.obsidx]

        # if skill < 0.5 or np.nan, do not use k-fB
        print("\t- screening the wave obs with skill parameter (skill > %f)" % self.min_skill)
        self.obsidx_cbathy_skill = np.full((self.ny,self.nx,self.nfB),False)
        self.obsidx_cbathy_skill[self.skill >= self.min_skill] = True

        # if lam1 < 10.0 or np.nan, don't use k-fB
        print("\t- screening the wave obs with lam1 parameter (lam1 > %f)" % self.min_lam1)
        self.obsidx_cbathy_lam1 = np.full((self.ny,self.nx,self.nfB),False)
        self.obsidx_cbathy_lam1[self.lam1 > self.min_lam1] = True

        # default if direct h < hfB_min (0.25 m) or h > hfB_max (10 m), don't use k-fB
        #self.hfB_max = 10.
        #self.hfB_min = 0.25

        # additional analysis (direct inversion from available data using least square)
        print('\t- screening the wave obs whose direct inversion of the depth h is between hfB_min %f and hfB_max %f' % (self.hfB_min, self.hfB_max))
        print('\t\t- perform direct, point/freq-wise bathy inversion from cBathy wave data using least square')
        
        self.h_direct_nfB = self._directInv_nfB()
        #self.h_direct_nfB = self.h_direct_nfB - self.tide

        self.obsidx_cbathy_hlimit = np.full((self.ny,self.nx,self.nfB),False)
        self.obsidx_cbathy_hlimit[(self.h_direct_nfB > self.hfB_min) & (self.h_direct_nfB < self.hfB_max)] = True

        # additional screening based on cbathy estimate NaN positions
        self.obsidx_cbathy = np.zeros_like(self.obsidx_k)
        for i in range(self.obsidx_cbathy.shape[2]):
            self.obsidx_cbathy[:,:,i] = ~np.isnan(self.h_cbathy)

        # observations used in the inversion
        # observatin index, obs, nobs
        self.obsidx = (self.obsidx_k & self.obsidx_cbathy_hlimit & self.obsidx_cbathy_skill & self.obsidx_cbathy_lam1 & self.obsidx_cbathy)
        self.obs = self.k[self.obsidx]
        self.nobs = self.obs.shape[0]
        #print(nobs)
        print("\t- # of wave observations used in this run after screening: %d" % self.nobs)

        print('\t- (optional) perform direct, point-wise bathy inversion from cBathy wave data using least square with equal weights over the nfB freqs (similar to cBathy Phase 2)')
        self.h_direct = self._directInv()
        #self.h_direct = self.h_direct - self.tide # tide

    def blend(self, h_prior = None, **kwargs):
        '''
        perform blending
        '''

        print("## 2. Perform blending ")
        # prior parameter assignment
        # prior mean 
        if h_prior is None:
            h_prior = np.copy(self.h_pbt).reshape(-1,1)
        else:
            h_prior = h_prior.reshape(-1,1)

        h_prior_h_min = np.copy(h_prior)
        h_prior_h_min[np.isnan(h_prior_h_min)] = self.h_min
        h_prior_h_min[h_prior_h_min < self.h_min] = self.h_min
        self.h_prior = h_prior

        # h_prior[np.isnan(h_prior)] = self.h_min
        # h_prior[h_prior < self.h_min] = self.h_min

        # obs_std_variable == True to assign variable observation error level
        if self.obs_std_variable is None:
            if self.kErr is None:
                self.obs_std_variable = False
            else:
                self.obs_std_variable = True
        
        if self.KErr_multiplier is None:
            self.KErr_multiplier = 5.0 # KErr multiplied by KErr_multiplier

        # Construct Error matrix
        if self.obs_std_variable:
            print('\t- observation error determined from cBathy Phase 1\'s kErr (obs_std_variable = %s)\n' % ("True" if self.obs_std_variable else "False"))
            obs_std2d = self.KErr_multiplier*self.kErr
            obs_std_KErr = obs_std2d[self.obsidx]
            # Error matrix
            self.R = diags(obs_std_KErr**2,offsets=0,shape=(self.nobs,self.nobs))
            self.invR = diags((1./obs_std_KErr)**2,offsets=0,shape=(self.nobs,self.nobs))
        else: # constant error 
            #obs_std2d = obs_std*np.ones((self.ny,self.nx,self.nfB))
            # # Error matrix
            #self.R = sp.sparse.diags(self.obs_std**2,offsets=0,shape=(self.nobs,self.nobs))
            #self.invR = sp.sparse.diags((1./self.obs_std)**2,offsets=0,shape=(self.nobs,self.nobs))
            print('\t- observation error set to %f (obs_std_variable = %s)\n' % (self.obs_std, "True" if self.obs_std_variable else "False"))
            self.R = (self.obs_std)**2 * eye(self.nobs,self.nobs)
            self.invR = (1./self.obs_std)**2 * eye(self.nobs,self.nobs)

        # compute k_prior 
        k_prior = self._forward(h_prior,self.fB,self.obsidx)
        res_prior = k_prior - self.obs
        self.res_prior = res_prior
        # Start CG (inner iterations)-GN (outer iterations) method
        tic = timeit.default_timer()
       
        h_cur = np.copy(h_prior)
        self.h_cur = h_cur

        for i in range(self.max_gn_iter):

            h_cur_h_min = np.copy(h_cur)
            h_cur_h_min[h_cur < self.h_min] = self.h_min

            # evaluate k-fB with h_cur >= h_min. if h_cur < h_min, h_cur = h_min
            k_cur = self._forward(h_cur_h_min,self.fB,self.obsidx)
            res = k_cur - self.obs

            if i == 0:
                self.k_cur_init = np.copy(k_cur)
                res_init = np.copy(res)
                self.res_init = res_init

            invRres = self.invR @ res
            misfit = np.dot(res.T, invRres)
            hshift = h_cur - self.h_prior
            reg = np.dot(hshift.T, (self.L @ hshift))
            obj_cur = 0.5*misfit + 0.5*self.Lambda*reg
            rmse = np.sqrt((res**2.).sum()/self.nobs)

            print(' GN iteration %d, misfit: %f, reg: %f, obj: %f, rmse :%f' % (i+1,misfit,reg,obj_cur,rmse))
                
            # for CG linearoperator: 
            h_cur_ = h_cur.reshape(-1)
            k_cur_ = k_cur.reshape(-1)
                
            if misfit <= 1.05*self.nobs and i > 0:
                print(' * misfit/tolerance discrency principle achieved')
                converged = True
                break

            def mv(v):
                '''
                Jacobian-vector product
                '''
                h_kloc_cur = self._h_to_h_kloc(h_cur,self.obsidx)
                v_kloc_cur = self._h_to_h_kloc(v,self.obsidx)

                # you can use explicitly constructed (sparse) Jacobian 
                #tic = timeit.default_timer()
                # J = self._Jac(k_cur_,h_kloc_cur)
                # Jv = J @ v
                #toc = timeit.default_timer()
                #print('Elapsed: %s' % (toc-tic))

                Jv = np.multiply(self._dkdh(k_cur_,h_kloc_cur),v_kloc_cur)

                return Jv

            def rmv(v):
                '''
                Jacobian transpose-vector product
                '''
                h_kloc_cur = self._h_to_h_kloc(h_cur,self.obsidx)
                # you can use Jacobian-vec
                #tic = timeit.default_timer()
                #J = self._Jac(k_cur_,h_kloc_cur)
                #print(J.T)
                #JTv1 = J.T @ v
                #toc = timeit.default_timer()
                #print('Elapsed: %s' % (toc-tic))
                JTV_ele = np.multiply(self._dkdh(k_cur_,h_kloc_cur),v) # (nobs,)

                cidx, _ = self._gen_index(self.obsidx)
                k = 0
                JTv = np.zeros(self.nh,)
                for i in cidx:
                    JTv[i] = JTv[i] + JTV_ele[k]
                    k = k + 1

                return JTv
            
            def Hv(v):
                '''
                Gauss-Newton Hessian-vector product
                '''
                Jv = mv(v)

                invRJv = self.invR @ Jv 

                JTJv = rmv(invRJv)

                return JTJv + self.Lambda*(self.L @ v)

            H = LinearOperator((self.nh,self.nh), matvec=Hv, rmatvec=Hv)
            J = LinearOperator((self.nobs,self.nh), matvec=mv, rmatvec = rmv)
            g = J.rmatvec(invRres).reshape(-1,1) + self.Lambda*(self.L @ (h_cur -h_prior))
                
            if i == 0:
                g_init = g
                g_tol_rel = np.linalg.norm(g_init) * self.g_tol
                
            # gradient termination criterion
            if np.linalg.norm(g) < g_tol_rel:
                print(' * gradient criterion achieved norm(g): %f, g_tol_rel: %f' % (np.linalg.norm(g),g_tol_rel))
                converged = True
                break 

            def report(xk):
                '''
                    report cg iterations 
                '''
                frame = inspect.currentframe().f_back
                # x, iter_, resid, info, ndx1, ndx2, sclr1, sclr2, ijob
                #print(frame.f_locals)
                if frame.f_locals['iter_'] % 100 == 0:
                    print('  inner cg iter %d, rel. res. (norm(res)/norm(g)): %10.4e' % (frame.f_locals['iter_'], frame.f_locals['resid']/np.linalg.norm(frame.f_locals['b']) )) 
                #print()

            # for inexact cg-gn, cgtol = np.min(0.5, np.sqrt(np.linalg.norm(np.solve(M,g))/g_init))
            dh, info = cg(H,-g, tol=self.cg_tol, maxiter = self.max_cg_iter, callback= report)
            #print(info)

            # line search (can be replaced with wolf condition-based LS)
            h_cur_old = np.copy(h_cur)

            def linesearch(alpha):
                h_cur_ = h_cur + alpha*dh.reshape(-1,1)
                h_cur_[h_cur_ < self.h_min] = self.h_min

                return self._obj(h_cur_,h_prior)
                
            alpha_opt, obj_opt, ierr, numfunc = fminbound(linesearch, -0.1, 1.5, args=(), xtol=1e-05, maxfun=500, full_output=True, disp=1)
            print(' - Linear Search, alpha : %f, obj: %f, ierr: %d, nfunc: %d' % (alpha_opt,obj_opt,ierr,numfunc))

            h_cur = h_cur + alpha_opt*dh.reshape(-1,1)
            
            
            if np.abs(alpha_opt) < 0.001:
                print('Converged')
                break
        
        # End of CG-GN steps

        toc = timeit.default_timer()
        print('\n#### Inversion completed in %9.3f secs          ####' % (toc-tic))
        
        # Result Summary 
        self.k_cur = self._rmse_report(h_cur)
        print('#######################################################\n')


        # Tide correction
        h_prior = h_prior - self.tide 
        h_cur = h_cur - self.tide
        
        self.h_cur = h_cur
        self.h_prior = h_prior
        self.h_direct = self.h_direct - self.tide
        
        print(' - Blending results were corrected with tide and saved')
        
        self.rmse_blending, self.rmse_cbathy, self.rmse_prior, self.bias_blending, self.bias_cbathy, self.bias_prior = self.compute_error()

        return h_cur # for now return only h_cur - will do uncertainty


    def _rmse_report(self,h_cur,h_prior=None):
        '''
            report RMSE
        '''
        if h_prior is None:
            h_prior = self.h_prior
            if h_prior is None:
                h_prior = np.copy(self.h_pbt).reshape(-1,1)
        else:
            h_prior = h_prior.reshape(-1,1)

        h_cur_h_min = np.copy(h_cur)
        h_cur_h_min[h_cur < self.h_min] = self.h_min

        # evaluate k-fB with h_cur >= h_min. if h_cur < h_min, h_cur = h_min
        k_cur = self._forward(h_cur_h_min,self.fB,self.obsidx)
        #k_cur = self._forward(h_cur,self.fB,self.obsidx)
        
        res = k_cur - self.obs
        invRres = self.invR @ res
        misfit = np.dot(res.T, invRres)
        hshift = h_cur - h_prior
        reg = np.dot(hshift.T, (self.L @ hshift))
        obj_cur = 0.5*misfit + 0.5*self.Lambda*reg

        rmse = np.sqrt((res**2.).sum()/self.nobs)

        
        print('## misfit: %f, reg: %f, obj: %f' % (misfit,reg,obj_cur))
        
        if self.res_init is not None:
            rmse_init = np.sqrt((self.res_init**2.).sum()/self.nobs)
            print('## init res. rmse: %f' % ( rmse_init))
        #if self.res_prior is not None:
        #    rmse_prior = np.sqrt((self.res_prior**2.).sum()/self.nobs)
        
        print('## final res. rmse: %f' % ( rmse ))

        self.rmse = rmse
        self.rmse_init = rmse_init

        return k_cur

    def _ConstructSqrtPrecisionMat(self):
        '''
        - Kitanidis (1999) Generalized covariance functions associated with 
        the Laplace equation and their use in interpolation and inverse problems, WRR
        '''
        
        er = np.ones((self.nx,)) # nx = nr
        ec = np.ones((self.ny,)) # ny = nc
        datar = np.vstack([-er,2.*er,-er])
        datac = np.vstack([-ec,2.*ec,-ec])
        diags = np.array([-1,0,1])
        
        L1r = spdiags(datar, diags, self.nx, self.nx)
        L1r = L1r.tocsc()
        L1r[0,0] = 1.
        L1r[self.nx-1,self.nx-1] = 1.
        I1r = eye(self.ny,self.ny)

        L1c = spdiags(datac, diags, self.ny, self.ny)
        L1c = L1c.tocsc()
        L1c[0,0] = 1.
        L1c[self.ny-1,self.ny-1] = 1.
        I1c = eye(self.nx,self.nx)
        L = kron(I1r,L1r) + kron(L1c,I1c) # invQ = np.dot(L.T , L)

        return L


    def _sech(self,x):
        '''
            sech function
        '''
        return 1./cosh(x)

    def _disp_res(self,k,sigma,h):
        '''
            implicit forward model: dispersion relationship
        '''
        g = 9.81
        return sigma**2. - g*k*tanh(k*h)
        
    def _dkdh(self,k,h):
        '''
            compute gradient dk/dh 
        '''
        hk = h*k
        #sechhk2 = sech(hk)**2
        tanhhx = tanh(hk)
        sechhk2 = 1. - tanhhx**2.
        return (-k**2 * sechhk2) / (tanhhx + hk*sechhk2)

    def _h_to_h_kloc(self,h,obsidx):
        '''
            find a subset of h corresponding to k observation locations 
        '''
        h2d = h.reshape(self.ny,self.nx)
        h3d = np.zeros((self.ny,self.nx,self.nfB))
        for i in range(4):
            h3d[:,:,i] = h2d
        h_kloc = h3d[obsidx]
        return h_kloc

    def _gen_index(self,obsidx):
        '''
            generate indices for Jac-vec operations
        '''
        idx_all = np.linspace(0,self.nh-1,self.nh,dtype=np.int16)
        idx_all2d = idx_all.reshape(self.ny,self.nx)
        idx_all3d = np.zeros((self.ny,self.nx,self.nfB),dtype=np.int16)
        for i in range(4):
            idx_all3d[:,:,i] = idx_all2d
        cidx = idx_all3d[obsidx]
        ridx = np.linspace(0,cidx.shape[0]-1,cidx.shape[0],dtype=np.int16)
        return cidx, ridx 

    def _forward(self,h,fB,obsidx):
        '''
            forward model: find k given h and fB using implicit dispersion equation
        '''
        h_kloc = self._h_to_h_kloc(h,obsidx)
        k_res = np.zeros(h_kloc.shape[0])
        fB = fB[obsidx]
        for i in range(h_kloc.shape[0]):
            if h_kloc[i] < 0.1*self.h_min:
                k_res[i] = 1.E+10
            else:
                sol = root_scalar(self._disp_res, args=(fB[i]*2.*np.pi, h_kloc[i]), method='brentq', bracket=[1e-4, 1000])

                if sol.converged:
                    k_res[i] = sol.root
                else:
                    print('not converged at %d, run again\n' % (i))
                    sol = root_scalar(self._disp_res, args=(fB[i]*2.*np.pi, h_kloc[i]), method='toms748', bracket=[1e-6, 100000])
                    if sol.converged:
                        k_res[i] = sol.root
                    else:
                        print('not converged again at %d\n' % (i))
                        k_res[i] = np.nan
        return k_res

    # def _report(self,xk):
    #     '''
    #         report cg iterations 
    #     '''
    #     frame = inspect.currentframe().f_back
    #     # x, iter_, resid, info, ndx1, ndx2, sclr1, sclr2, ijob
    #     #print(frame.f_locals)
    #     if frame.f_locals['iter_'] % 100 == 0:
    #         print('  inner cg iter %d, rel. res. (norm(res)/norm(g)): %10.4e' % (frame.f_locals['iter_'], frame.f_locals['resid']/np.linalg.norm(frame.f_locals['b']) )) 
        
    
    def _Jac(self,k,h):
        '''
            explicit construction of (sparse) Jacobian; for debugging purpose
        '''
        cidx, ridx = self._gen_index(self.obsidx)
        data = self._dkdh(k,h)
        return csr_matrix((data, (ridx, cidx)), shape=(self.nobs, self.nh))
        
    def _obj(self,h,h_prior):
        '''
            compute objective function
        '''
        res = self._forward(h,self.fB,self.obsidx) - self.obs
        hshift = h - h_prior
        return 0.5*np.dot( res.T, ( self.invR @ res ) ) + 0.5*self.Lambda* np.dot(hshift.T, self.L @ hshift)
        
    def _maxres(self,k,h,fB):
        # given k, h, fB, compute max residual of dispersion relationship
        return np.abs(self._disp_res(k.reshape(-1),2.*np.pi*fB[self.obsidx],self._h_to_h_kloc(h,self.obsidx))).max()

    def _directInv(self):
        '''
            use Least Square to compute "point-wise" bathy from k-fB data
            equal weights assgined over nfB (=4 by default) frequencies
        '''
        h_direct = np.zeros((self.ny,self.nx))
        for i in range(self.ny):
            for j in range(self.nx):
                if (~np.isnan(self.k[i,j,:])).any():
                    def direct_obj(h):
                        mse = 0.0
                        for l in range(self.nfB):
                            if ~np.isnan(self.k[i,j,l]):
                                mse_tmp = self._disp_res(self.k[i,j,l], 2.*self.fB[i,j,l]*np.pi, h)
                                mse = mse + np.sum(mse_tmp**2.)
                        return mse
                    
                    res = sp.optimize.minimize_scalar(direct_obj, bracket=[0.1*self.h_min,5.*self.h_max])
                    if res.success:
                        h_direct[i,j] = res.x
                    else:
                        print('not converged (grid %d, %d)' % (i,j))
                        h_direct[i,j] = np.nan
                else:
                    h_direct[i,j] = np.nan
        return h_direct

    def _directInv_nfB(self):
        '''
            use Least Square to compute "point/frequency-wise" bathy from k-fB data
        '''
        h_direct_nfB = np.zeros((self.ny,self.nx,self.nfB))

        for i in range(self.ny):
            for j in range(self.nx):
                for l in range(self.nfB):
                    if ~np.isnan(self.k[i,j,l]):
                        def direct_obj(h):
                            err = self._disp_res(self.k[i,j,l], 2.*self.fB[i,j,l]*np.pi, h)
                            return err**2.
                        
                        res = sp.optimize.minimize_scalar(direct_obj, bracket=[0.1*self.hfB_min,5.*self.hfB_max])
                        #res = sp.optimize.minimize_scalar(direct_obj, bracket=[0.001, 30.0])
                        if res.success:
                            h_direct_nfB[i,j,l] = res.x
                        else:
                            print('not converged (grid %d, %d, %d)' % (i,j,l))
                            h_direct_nfB[i,j,l] = np.nan
                    else:
                        h_direct_nfB[i,j,l] = np.nan
        return h_direct_nfB

    def plot_obslocs(self):
        '''   
            plot observation locations
        '''
        YY, XX = np.meshgrid(self.x,self.y)
        kobs_loc = np.copy(self.k)
        kobs_loc[kobs_loc>0] = 1.0
        kobs_loc[kobs_loc<=0] = np.nan
        kobs_loc[np.isnan(self.k)] = np.nan
        
        cbathy_loc = np.copy(self.h_cbathy)
        cbathy_loc[cbathy_loc>0] = 1.0
        cbathy_loc[cbathy_loc<=0] = np.nan
        cbathy_loc[np.isnan(cbathy_loc)] = np.nan
        #print((cbathy_loc).dtype)
        #print(cbathy_loc)
        
        hobs_loc = np.zeros_like(self.k)
        hobs_loc[self.obsidx_cbathy_hlimit] = 1.0
        hobs_loc[~self.obsidx_cbathy_hlimit] = np.nan
        #print((hobs_loc).dtype)
        
        lam1_loc = np.zeros_like(self.k)
        lam1_loc[self.obsidx_cbathy_lam1] = 1.0
        lam1_loc[~self.obsidx_cbathy_lam1] = np.nan
        #print((lam1_loc).dtype)
        
        skill_loc = np.zeros_like(self.k)
        skill_loc[self.obsidx_cbathy_skill] = 1.0
        skill_loc[~self.obsidx_cbathy_skill] = np.nan

        obs_loc = np.zeros_like(self.k)
        obs_loc[self.obsidx] = 1.0
        obs_loc[~self.obsidx] = np.nan
        
        fig, ax = plt.subplots(5,self.nfB+1,figsize=(16,20))
        
        if self.mydate != None:
            plt.suptitle('observation locations ' + self.mydatetext, fontsize=15)
        else:
            plt.suptitle('observation locations')
        
        for i in range(self.nfB):
            ax[0,i].set_title('Original K-fB #%d'% (i+1))
            ax[0,i].set_xticks([])
            if i == 0:
                ax[0,i].set_yticks([-500,0,500,1000,1500])
            ax[0,i].set_yticks([])
            im0 = ax[0,i].pcolormesh(YY,XX,kobs_loc[:,:,i])
            #fig.colorbar(im0, ax=ax[0,i])

            ax[1,i].set_title('cbathy h > %5.2f & h < %5.2f #%d'% (self.hfB_min,self.hfB_max,i+1))
            ax[1,i].set_xticks([80,800])
            if i == 0:
                ax[1,i].set_yticks([-500,0,500,1000,1500])
                ax[1,i].set_yticks([])
            
            im1 = ax[1,i].pcolormesh(YY,XX,hobs_loc[:,:,i])
            #fig.colorbar(im1, ax=ax[1,i])
            
            ax[2,i].set_title('skill > 0.5 #%d'% (i+1))
            ax[2,i].set_xticks([80,800])
            if i == 0:
                ax[2,i].set_yticks([-500,0,500,1000,1500])
                ax[2,i].set_yticks([])
            
            im2 = ax[2,i].pcolormesh(YY,XX,skill_loc[:,:,i])
            #fig.colorbar(im2, ax=ax[2,i])

            ax[3,i].set_title('lam1 > 10.0 #%d'% (i+1))
            ax[3,i].set_xticks([80,800])
            if i == 0:
                ax[3,i].set_yticks([-500,0,500,1000,1500])
                ax[3,i].set_yticks([])
            im3 = ax[3,i].pcolormesh(YY,XX,lam1_loc[:,:,i])
            #fig.colorbar(im3, ax=ax[3,i])

            ax[4,i].set_title('final obs loc #%d'% (i+1))
            ax[4,i].set_xticks([80,800])
            if i == 0:
                ax[4,i].set_yticks([-500,0,500,1000,1500])
                ax[4,i].set_yticks([])
            
            im4 = ax[4,i].pcolormesh(YY,XX,obs_loc[:,:,i])

        im5=ax[0,i+1].pcolormesh(YY,XX,cbathy_loc)
        ax[0,i+1].set_title('cbathy')
        ax[0,i+1].set_xticks([80,800])
        ax[0,i+1].set_yticks([])
        #fig.colorbar(im5, ax=ax[0,i+1])

        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        plt.show()

    def plot_cbathy_errors(self):
        '''   
            plot kErr and hTempErr
        '''  
        YY, XX = np.meshgrid(self.x,self.y)
        
        fig, ax = plt.subplots(2,self.nfB,figsize=(16,12))
        if self.mydatetext is None:
            plt.suptitle('Cbathy Error', fontsize=15)
        else:
            plt.suptitle('Cbathy Error ' + self.mydatetext, fontsize=15)
        
        for i in range(self.nfB): 
            im = ax[0,i].pcolormesh(YY,XX,self.kErr[:,:,i],cmap='jet')
            ax[0,i].set_title('kErr #%d' % (i))
            ax[0,i].set_xticks([80,800])
            ax[0,i].set_yticks([])
            fig.colorbar(im,ax=ax[0,i])
        
        for i in range(self.nfB): 
            im = ax[1,i].pcolormesh(YY,XX,self.hTempErr[:,:,i],cmap='jet')
            ax[1,i].set_title('hTempErr #%d' % (i))
            ax[1,i].set_xticks([80,800])
            ax[1,i].set_yticks([])
            fig.colorbar(im,ax=ax[1,i])
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
        
        plt.show()

    def plot_bathy(self,h_cur=None,h_prior=None,h_cbathy = None, h_survey=None, hmin=-2., hmax=None, pierFRF=((80,530),(515,515)), xFRF=(80,800),  yFRF=(-500,1500), pierFRFhatch=None):
        '''
            plot results
            more details will be added here
        '''    
        nx, ny = self.nx, self.ny

        if h_cur is None:
            h_cur = self.h_cur
        if h_prior is None:
            h_prior = self.h_prior
        if h_survey is None:
            h_survey = self.h_survey
        if h_cbathy is None:
            h_cbathy = self.h_cbathy

        if hmax is None:
            hmax = np.ceil(-h_survey[~np.isnan(h_survey)].min()) # np.ceil(np.max((h_prior,h_cur)))

        column_ratio = (yFRF[1] - yFRF[0])/2000.
        
        from mpl_toolkits.axes_grid1 import ImageGrid

        # Set up figure and image grid
        fig = plt.figure(figsize=(10., 5.*(column_ratio*0.5 + 0.5)))

        ax = ImageGrid(fig, 111,          # as in plt.subplot(111)
                        nrows_ncols=(1,4),
                        axes_pad=0.40,
                        share_all=True,
                        cbar_location="right",
                        cbar_mode="single",
                        cbar_size="7%",
                        cbar_pad=0.20,
                        label_mode = "L",
                        )

        YY, XX = np.meshgrid(self.x,self.y)

        # fig, ax = plt.subplots(1,4,figsize=(16,6)) # 20,6
        fig.suptitle('Bathymetry Estimation (%s)' % (self.mydatetext))

        mylevels = (0.,1., 2., 3., 4.)

        im0 = ax[0].pcolormesh(YY,XX, h_prior.reshape(ny,nx),cmap='jet_r',shading='auto',vmin=hmin, vmax = hmax)
        CS0 = ax[0].contour(YY, XX, h_prior.reshape(ny,nx), levels = mylevels, colors='black', linewidths=0.5)
        ax[0].clabel(CS0, CS0.levels[::2], inline=True, fmt='%d', fontsize=9, colors='black')

        #ax[0].set_title('Prior (PBT) ' + date[ii])
        #ax[0].set_title('Prior (PBT) ' + date[ii])
        ax[0].set_title('(a) Prior (PBT)', fontsize=14)
        ax[0].set_xticks([80,500,800])
        ax[0].set_yticks([-500,0,500,1000,1500])
        #fig.colorbar(im0, ax=ax[0])
        im1 = ax[1].pcolormesh(YY,XX,h_cbathy,cmap='jet_r',shading='auto',vmin=hmin, vmax = hmax)
        #ax[1].set_title('Estimate ' + date[ii])
        ax[1].set_title('(b) cBathy', fontsize=14)
        CS1 = ax[1].contour(YY, XX, h_cbathy, levels = mylevels, colors='black', linewidths=0.5)
        ax[1].clabel(CS1, CS1.levels[::2], inline=True, fmt='%d', fontsize=9, colors='black')


        ax[1].set_xticks([80,500,800])
        ax[1].set_yticks([-500,0,500,1000,1500])
        im2 = ax[2].pcolormesh(YY,XX, h_cur.reshape(ny,nx),cmap='jet_r',shading='auto',vmin=hmin, vmax = hmax)
        CS2 = ax[2].contour(YY, XX, h_cur.reshape(ny,nx), levels = mylevels, colors='black', linewidths=0.5)
        ax[2].clabel(CS2, CS2.levels[::2], inline=True, fmt='%d', fontsize=9, colors='black')

        ax[2].set_title('(c) Blending', fontsize=14)
        ax[2].set_xticks([80,500,800])
        ax[2].set_yticks([-500,0,500,1000,1500])
        im3 = ax[3].pcolormesh(YY,XX,-h_survey,cmap='jet_r',shading='auto',vmin=hmin, vmax = hmax)
        CS3 = ax[3].contour(YY, XX, -h_survey, levels = mylevels, colors='black', linewidths=0.5)
        ax[3].clabel(CS3, CS3.levels[::2], inline=True, fmt='%d', fontsize=9, colors='black')

        ax[3].set_title('(d) Survey', fontsize=14)
        ax[3].set_xticks([80,500,800])
        ax[3].set_yticks([-500,0,500,1000,1500])

        cbar = ax[3].cax.colorbar(im3)
        ax[3].cax.toggle_label(True)
        cbar.ax.set_title('depth [m]',fontsize=10)

        if pierFRF is not None:
            for i in range(len(ax.axes_all)):
                ax[i].plot(pierFRF[0],pierFRF[1],'k',linewidth=2)
                if pierFRFhatch is not None:
                    ax[i].fill_between((xFRF), (pierFRFhatch[0],pierFRFhatch[0]), (pierFRFhatch[1],pierFRFhatch[1]), facecolor="none", hatch="xx", edgecolor="k", linewidth=0.0)

        if yFRF is not None:
            for i in range(len(ax.axes_all)):
                ax[i].set_ylim(yFRF[0],yFRF[1])

        fig.text(0.5, 0.0, 'xFRF [m]', va='center', ha='center', fontsize=plt.rcParams['axes.labelsize'])
        fig.text(0.1, 0.5, 'yFRF [m]', va='center', ha='center', rotation='vertical', fontsize=plt.rcParams['axes.labelsize'])

        fig.tight_layout()
        fig.subplots_adjust(top=0.87)

        plt.savefig(join(self.output_dir,'estimate_' + str(self.mydate) + '.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
    def plot_bathy_error(self,pierFRF=((80,530),(515,515)), yFRF=(-500,1500), xFRF=(80,800), pierFRFhatch=None, h_cur=None,h_prior=None,h_cbathy = None, h_survey=None):

        from mpl_toolkits.axes_grid1 import ImageGrid

        nx, ny = self.nx, self.ny

        if h_cur is None:
            h_cur = self.h_cur
        if h_prior is None:
            h_prior = self.h_prior
        if h_survey is None:
            h_survey = self.h_survey
        if h_cbathy is None:
            h_cbathy = self.h_cbathy

        column_ratio = (yFRF[1] - yFRF[0])/2000.
    
        # Set up figure and image grid
        fig = plt.figure(figsize=(10., 5.*(column_ratio*0.5 + 0.5)))

        ax = ImageGrid(fig, 111,          # as in plt.subplot(111)
                        nrows_ncols=(1,3),
                        axes_pad=0.40,
                        share_all=True,
                        cbar_location="right",
                        cbar_mode="single",
                        cbar_size="7%",
                        cbar_pad=0.20,
                        label_mode = "L",
                        )

        YY, XX = np.meshgrid(self.x,self.y)

        # fig, ax = plt.subplots(1,4,figsize=(16,6)) # 20,6
        fig.suptitle('Difference between Estimate and Survey %s' % (self.mydatetext))

        prior_diff = -h_survey-h_prior.reshape(ny,nx)
        cur_diff = -h_survey-h_cur.reshape(ny,nx)
        cbathy_diff = -h_survey-h_cbathy.reshape(ny,nx)

        # print(np.nanmin(prior_diff),np.nanmax(prior_diff))
        # print(np.nanmin(cur_diff),np.nanmax(cur_diff))
        # print(np.nanmin(cbathy_diff),np.nanmax(cbathy_diff))

        #mylevels = (-4., -3, -2., -1, 0., 1, 2., 3., 4.)
        #mylevels = (-3, -2., -1, 0., 1, 2., 3.)
        mylevels = (-4,-3,-2,-1, -0.5, 0., 0.5, 1,2,3,4)

        cmap = plt.get_cmap('RdBu', 16)    # 11 discrete colors

        hmin = -4. #np.min((np.min((h_prior,h_cur)),0.))
        hmax = 4. # np.ceil(np.max((h_prior,h_cur)))
        #im0 = ax[0].pcolormesh(YY,XX,prior_diff,cmap='RdBu',shading='auto',vmin=hmin, vmax = hmax)
        im0 = ax[0].pcolormesh(YY,XX,prior_diff,cmap=cmap,shading='auto',vmin=hmin, vmax = hmax)

        CS0 = ax[0].contour(YY, XX, prior_diff, levels = mylevels, colors='black', linewidths=0.5)
        ax[0].clabel(CS0, CS0.levels, inline=True, fmt='%d', fontsize=9, colors='black')

        #ax[0].set_title('Prior (PBT) ' + date[ii])
        #ax[0].set_title('Prior (PBT) ' + date[ii])
        ax[0].set_title('(a) PBT', fontsize=14)
        ax[0].set_xticks([80,500,800])
        ax[0].set_yticks([-500,0,500,1000,1500])
        #fig.colorbar(im0, ax=ax[0])
        #im1 = ax[1].pcolormesh(YY,XX,cbathy_diff,cmap='RdBu',shading='auto',vmin=hmin, vmax = hmax)
        im1 = ax[1].pcolormesh(YY,XX, cbathy_diff,cmap=cmap,shading='auto',vmin=hmin, vmax = hmax)
        CS1 = ax[1].contour(YY, XX, cbathy_diff, levels = mylevels, colors='black', linewidths=0.5)
        ax[1].clabel(CS1, CS1.levels, inline=True, fmt='%d', fontsize=9, colors='black')


        #ax[1].set_title('Estimate ' + date[ii])
        ax[1].set_title('(b) cBathy', fontsize=14)
        ax[1].set_xticks([80,500,800])
        ax[1].set_yticks([-500,0,500,1000,1500])
        #fig.colorbar(im1, ax=ax[1])
        #im2 = ax[2].pcolormesh(YY,XX,cur_diff,cmap='RdBu',shading='auto',vmin=hmin, vmax = hmax)
        im2 = ax[2].pcolormesh(YY,XX,cur_diff,cmap=cmap,shading='auto',vmin=hmin, vmax = hmax)
        CS2 = ax[2].contour(YY, XX, cur_diff, levels = mylevels, colors='black', linewidths=0.5)
        ax[2].clabel(CS2, CS2.levels, inline=True, fmt='%3.1f', fontsize=9, colors='black')

        #ax[2].set_title('Cbathy ' + date[ii])
        ax[2].set_title('(c) Blending', fontsize=14)
        ax[2].set_xticks([80,500,800])
        ax[2].set_yticks([-500,0,500,1000,1500])

        cbar = ax[2].cax.colorbar(im2)
        ax[2].cax.toggle_label(True)
        cbar.ax.set_title('diff [m]',fontsize=10)

        if pierFRF is not None:
            for i in range(len(ax.axes_all)):
                ax[i].plot(pierFRF[0],pierFRF[1],'k',linewidth=2)
                if pierFRFhatch is not None:
                    ax[i].fill_between((xFRF), (pierFRFhatch[0],pierFRFhatch[0]), (pierFRFhatch[1],pierFRFhatch[1]), facecolor="none", hatch="xx", edgecolor="k", linewidth=0.0)

        if yFRF is not None:
            for i in range(len(ax.axes_all)):
                ax[i].set_ylim(yFRF[0],yFRF[1])

        fig.text(0.5, 0.0, 'xFRF [m]', va='center', ha='center', fontsize=plt.rcParams['axes.labelsize'])
        fig.text(0.1, 0.5, 'yFRF [m]', va='center', ha='center', rotation='vertical', fontsize=plt.rcParams['axes.labelsize'])

        # ax[0].text(150, -300,'RMSE: %5.2f m' % (self.rmse_prior))
        # ax[0].text(150, -400,'Bias: %5.2f m' % (self.bias_prior))
        # ax[1].text(150, -300,'RMSE: %5.2f m' % (self.rmse_cbathy))
        # ax[1].text(150, -400,'Bias: %5.2f m' % (self.bias_cbathy))
        # ax[2].text(150, -300,'RMSE: %5.2f m' % (self.rmse_blending))
        # ax[2].text(150, -400,'Bias: %5.2f m' % (self.bias_blending))

        ax[0].text(150, yFRF[0]+200,'RMSE: %5.2f m' % (self.rmse_prior))
        ax[0].text(150, yFRF[0]+100,'Bias: %5.2f m' % (self.bias_prior))
        ax[1].text(150, yFRF[0]+200,'RMSE: %5.2f m' % (self.rmse_cbathy))
        ax[1].text(150, yFRF[0]+100,'Bias: %5.2f m' % (self.bias_cbathy))
        ax[2].text(150, yFRF[0]+200,'RMSE: %5.2f m' % (self.rmse_blending))
        ax[2].text(150, yFRF[0]+100,'Bias: %5.2f m' % (self.bias_blending))
        fig.tight_layout()
        fig.subplots_adjust(top=0.87)

        plt.savefig(join(self.output_dir,'estimate_diff_' + str(self.mydate) + '.png'), dpi=150, bbox_inches='tight')
        plt.show()

    def plot_bathy_fitting_all(self):
        '''
            Compute RMSE and BIAS between survey and estimate
        '''
        from mpl_toolkits.axes_grid1 import ImageGrid

        idx_iswatersurvey = np.logical_and(~np.isnan(self.h_survey),(self.h_survey < 0.))
        #n_h_survey_water = np.sum(idx_iswatersurvey)
        #print(n_h_survey_water)

        YY, XX = np.meshgrid(self.x,self.y)
        
        idx_is_non_pier = np.logical_or((XX <= 400.),(XX >= 600.))
        #n_h_non_pier = np.sum(idx_is_non_pier)
        
        prior_diff = -self.h_survey-self.h_prior.reshape(self.ny,self.nx)
        # prior_diff = prior_diff[~np.isnan(prior_diff)] # entire area
        # prior_diff = prior_diff[np.logical_and(idx_iswatersurvey,~np.isnan(prior_diff))] # underwater area
        idx_prior = np.logical_and(np.logical_and(idx_iswatersurvey,~np.isnan(prior_diff)),idx_is_non_pier)
        prior_diff = prior_diff[idx_prior] # underwater non-pier area 
        n_prior_diff = prior_diff.shape[0]
        totsumsqr = ((-self.h_survey[idx_prior] - np.mean(-self.h_survey[idx_prior]))**2).sum()
        
        cur_diff = -self.h_survey-self.h_cur.reshape(self.ny,self.nx)
        # cur_diff = cur_diff[~np.isnan(cur_diff)] # entire area 
        # cur_diff = cur_diff[np.logical_and(idx_iswatersurvey,~np.isnan(cur_diff))] # underwater area
        idx_cur = np.logical_and(np.logical_and(idx_iswatersurvey,~np.isnan(cur_diff)),idx_is_non_pier)
        cur_diff = cur_diff[idx_cur] # underwater non-pier area 
        n_cur_diff = cur_diff.shape[0]
        
        cbathy_diff = -self.h_survey-self.h_cbathy.reshape(self.ny,self.nx)
        # cbathy_diff = cbathy_diff[~np.isnan(cbathy_diff)] # entire area
        # cbathy_diff = cbathy_diff[np.logical_and(idx_iswatersurvey,~np.isnan(cbathy_diff))] # underwater area
        idx_cbathy = np.logical_and(np.logical_and(idx_iswatersurvey,~np.isnan(cbathy_diff)),idx_is_non_pier)
        cbathy_diff = cbathy_diff[idx_cbathy]  # underwater non-pier area 
        n_cbathy_diff = cbathy_diff.shape[0]
        
        #print(totsumsqr)
        r2_prior = 1. - (((prior_diff)**2).sum()/totsumsqr)
        r2_cur = 1. - (((cur_diff)**2).sum()/totsumsqr)
        r2_cbathy = 1. - (((cbathy_diff)**2).sum()/totsumsqr)

        fig = plt.figure(figsize=(5, 5))

        ax = ImageGrid(fig, 111,          # as in plt.subplot(111)
                        nrows_ncols=(1,1),
                        axes_pad=0.40,
                        share_all=True,
                        #  cbar_location="right",
                        #  cbar_mode="single",
                        #  cbar_size="7%",
                        #  cbar_pad=0.20,
                        label_mode = "L",
                        )

        YY, XX = np.meshgrid(self.x,self.y)

        ax[0].set_title('Estimate vs. Survey %s' % (self.mydatetext))
        ax[0].scatter(-self.h_survey[idx_cur], self.h_cur.reshape(self.ny,self.nx)[idx_cur], s = 0.5, color='blue',label='$R^2$ (Blending): %5.2f' % (r2_cur))
        ax[0].scatter(-self.h_survey[idx_prior], self.h_prior.reshape(self.ny,self.nx)[idx_prior], s = 0.5, color='green',label='$R^2$ (PBT): %5.2f' % (r2_prior))
        ax[0].scatter(-self.h_survey[idx_cbathy], self.h_cbathy.reshape(self.ny,self.nx)[idx_cbathy], s = 0.5, color='red',label='$R^2$ (cBathy): %5.2f' % (r2_cbathy))

        ax[0].set_ylabel('estimate [m]')
        ax[0].plot([0.,8.0],[0.0,8.0],'b-')
        # ax.set_aspect('equal', adjustable='box')
        # plt.axis([0.01,1,0.01,1])
        # plt.xticks([0.01, 0.5 ,1.0])
        # plt.yticks([0.01, 0.5 ,1.0])
        #plt.plot([0.0,0.5],[0.0,0.5],'r-')
        ax[0].set_aspect('equal', adjustable='box')
        #plt.axis([0.0,8.0,0.0,0.5])
        ax[0].set_xticks([0, 4 , 8])
        ax[0].set_yticks([0, 4 , 8])
        ax[0].set_xlim([0,8])
        ax[0].set_ylim([0,8])
        l = ax[0].legend(markerscale=6.0,scatteryoffsets=[+0.75])#,handletextpad=0.0)#,handletextpad=-0.5)
        #legend(scatterpoints=1,scatteryoffsets=[0],handletextpad=-0.5)
        #for t in l.get_texts(): t.set_va('center_baseline')
        #for t in l.get_texts(): t.set_va('center')
        plt.savefig(join(self.output_dir,'bathy_fitting_' + str(self.mydate) + '_all.png'), dpi=150, bbox_inches='tight')
        plt.show()    

    def plot_bathy_fitting(self):
        '''
            Compute RMSE and BIAS between survey and estimate
        '''
        from mpl_toolkits.axes_grid1 import ImageGrid

        idx_iswatersurvey = np.logical_and(~np.isnan(self.h_survey),(self.h_survey < 0.))
        #n_h_survey_water = np.sum(idx_iswatersurvey)
        #print(n_h_survey_water)

        YY, XX = np.meshgrid(self.x,self.y)
        
        idx_is_non_pier = np.logical_or((XX <= 400.),(XX >= 600.))
        #n_h_non_pier = np.sum(idx_is_non_pier)
        
        prior_diff = -self.h_survey-self.h_prior.reshape(self.ny,self.nx)
        # prior_diff = prior_diff[~np.isnan(prior_diff)] # entire area
        # prior_diff = prior_diff[np.logical_and(idx_iswatersurvey,~np.isnan(prior_diff))] # underwater area
        idx_prior = np.logical_and(np.logical_and(idx_iswatersurvey,~np.isnan(prior_diff)),idx_is_non_pier)
        prior_diff = prior_diff[idx_prior] # underwater non-pier area 
        n_prior_diff = prior_diff.shape[0]
        totsumsqr = ((-self.h_survey[idx_prior] - np.mean(-self.h_survey[idx_prior]))**2).sum()
        
        cur_diff = -self.h_survey-self.h_cur.reshape(self.ny,self.nx)
        # cur_diff = cur_diff[~np.isnan(cur_diff)] # entire area 
        # cur_diff = cur_diff[np.logical_and(idx_iswatersurvey,~np.isnan(cur_diff))] # underwater area
        idx_cur = np.logical_and(np.logical_and(idx_iswatersurvey,~np.isnan(cur_diff)),idx_is_non_pier)
        cur_diff = cur_diff[idx_cur] # underwater non-pier area 
        n_cur_diff = cur_diff.shape[0]
        
        cbathy_diff = -self.h_survey-self.h_cbathy.reshape(self.ny,self.nx)
        # cbathy_diff = cbathy_diff[~np.isnan(cbathy_diff)] # entire area
        # cbathy_diff = cbathy_diff[np.logical_and(idx_iswatersurvey,~np.isnan(cbathy_diff))] # underwater area
        idx_cbathy = np.logical_and(np.logical_and(idx_iswatersurvey,~np.isnan(cbathy_diff)),idx_is_non_pier)
        cbathy_diff = cbathy_diff[idx_cbathy]  # underwater non-pier area 
        n_cbathy_diff = cbathy_diff.shape[0]
        
        #print(totsumsqr)
        r2_prior = 1. - (((prior_diff)**2).sum()/totsumsqr)
        r2_cur = 1. - (((cur_diff)**2).sum()/totsumsqr)
        r2_cbathy = 1. - (((cbathy_diff)**2).sum()/totsumsqr)

        # Set up figure and image grid
        fig = plt.figure(figsize=(8, 3))

        ax = ImageGrid(fig, 111,          # as in plt.subplot(111)
                        nrows_ncols=(1,3),
                        axes_pad=0.40,
                        share_all=True,
                        #  cbar_location="right",
                        #  cbar_mode="single",
                        #  cbar_size="7%",
                        #  cbar_pad=0.20,
                        label_mode = "L",
                        )

        # fig, ax = plt.subplots(1,4,figsize=(16,6)) # 20,6
        fig.suptitle('Estimate vs. Survey (%s)' % (self.mydatetext))

        ax[0].set_title('Prior (PBT)')
        ax[0].scatter(-self.h_survey[idx_prior], self.h_prior.reshape(self.ny,self.nx)[idx_prior], s = 0.5)
        ax[0].text(0.5,7.,'$R^2$: %5.2f' % (r2_prior))
        #ax[0].text(0.,6.,'bias: %5.2f m' % (bias_prior))
        #ax[0].set_xlabel('survey')
        ax[0].set_ylabel('estimate [m]')
        ax[0].plot([0.,8.0],[0.0,8.0],'r-')
        # ax.set_aspect('equal', adjustable='box')
        # plt.axis([0.01,1,0.01,1])
        # plt.xticks([0.01, 0.5 ,1.0])
        # plt.yticks([0.01, 0.5 ,1.0])
        #plt.plot([0.0,0.5],[0.0,0.5],'r-')
        ax[0].set_aspect('equal', adjustable='box')
        #plt.axis([0.0,8.0,0.0,0.5])
        ax[0].set_xticks([0, 4 , 8])
        ax[0].set_yticks([0, 4 , 8])
        ax[0].set_xlim([0,8])
        ax[0].set_ylim([0,8])

        ax[1].set_title('cBathy')
        ax[1].scatter(-self.h_survey[idx_cbathy], self.h_cbathy.reshape(self.ny,self.nx)[idx_cbathy], s = 0.5)
        #plt.scatter(k_cur, obs, 'black', alpha_arr, s=1,label='Blending (RMSE: %7.3f [$m^{-1}$])' % (rmse_cur))
        ax[1].text(0.5,7.,'$R^2$: %5.2f' % (r2_cbathy))
        #ax[1].text(0.,6.,'bias: %5.2f m' % (bias_cbathy))
        ax[1].set_xlabel('survey [m]')
        ax[1].plot([0.,8.0],[0.0,8.0],'r-')
        # ax.set_aspect('equal', adjustable='box')
        # plt.axis([0.01,1,0.01,1])
        # plt.xticks([0.01, 0.5 ,1.0])
        # plt.yticks([0.01, 0.5 ,1.0])
        #plt.plot([0.0,0.5],[0.0,0.5],'r-')
        ax[1].set_aspect('equal', adjustable='box')
        #plt.axis([0.0,0.5,0.0,0.5])
        ax[1].set_xticks([0, 4 , 8])
        ax[1].set_yticks([0, 4 , 8])
        ax[1].set_xlim([0,8])
        ax[1].set_ylim([0,8])
        ax[2].set_title('Blending')
        ax[2].scatter(-self.h_survey[idx_cur], self.h_cur.reshape(self.ny,self.nx)[idx_cur], s = 0.5)
        ax[2].text(0.5,7.,'$R^2$: %5.2f' % (r2_cur))
        ax[2].set_xlim([0,8])
        ax[2].set_ylim([0,8])
        #ax[2].text(0.,6.,'bias: %5.2f m' % (bias_cur))
        #plt.scatter(k_cur, obs, 'black', alpha_arr, s=1,label='Blending (RMSE: %7.3f [$m^{-1}$])' % (rmse_cur))
        #ax[2].set_xlabel('survey')
        ax[2].plot([0.,8.0],[0.0,8.0],'r-')
        # ax.set_aspect('equal', adjustable='box')
        # plt.axis([0.01,1,0.01,1])
        # plt.xticks([0.01, 0.5 ,1.0])
        # plt.yticks([0.01, 0.5 ,1.0])
        #plt.plot([0.0,0.5],[0.0,0.5],'r-')
        ax[2].set_aspect('equal', adjustable='box')
        #plt.axis([0.0,0.5,0.0,0.5])
        ax[2].set_xticks([0, 4 , 8])
        ax[2].set_yticks([0, 4 , 8])


        plt.savefig(join(self.output_dir,'bathy_fitting_' + str(self.mydate) + '_separate.png'), dpi=150, bbox_inches='tight')
        plt.show()

    def plot_bathy_fitting_wavenumber(self):
        '''
            plot (a) bathy_fitting and (b) wavenumber
        '''

        from mpl_toolkits.axes_grid1 import ImageGrid
        idx_iswatersurvey = np.logical_and(~np.isnan(self.h_survey),(self.h_survey < 0.))
        #n_h_survey_water = np.sum(idx_iswatersurvey)
        #print(n_h_survey_water)

        YY, XX = np.meshgrid(self.x,self.y)
        
        idx_is_non_pier = np.logical_or((XX <= 400.),(XX >= 600.))
        #n_h_non_pier = np.sum(idx_is_non_pier)
        
        prior_diff = -self.h_survey-self.h_prior.reshape(self.ny,self.nx)
        # prior_diff = prior_diff[~np.isnan(prior_diff)] # entire area
        # prior_diff = prior_diff[np.logical_and(idx_iswatersurvey,~np.isnan(prior_diff))] # underwater area
        idx_prior = np.logical_and(np.logical_and(idx_iswatersurvey,~np.isnan(prior_diff)),idx_is_non_pier)
        prior_diff = prior_diff[idx_prior] # underwater non-pier area 
        n_prior_diff = prior_diff.shape[0]
        totsumsqr = ((-self.h_survey[idx_prior] - np.mean(-self.h_survey[idx_prior]))**2).sum()
        
        cur_diff = -self.h_survey-self.h_cur.reshape(self.ny,self.nx)
        # cur_diff = cur_diff[~np.isnan(cur_diff)] # entire area 
        # cur_diff = cur_diff[np.logical_and(idx_iswatersurvey,~np.isnan(cur_diff))] # underwater area
        idx_cur = np.logical_and(np.logical_and(idx_iswatersurvey,~np.isnan(cur_diff)),idx_is_non_pier)
        cur_diff = cur_diff[idx_cur] # underwater non-pier area 
        n_cur_diff = cur_diff.shape[0]
        
        cbathy_diff = -self.h_survey-self.h_cbathy.reshape(self.ny,self.nx)
        # cbathy_diff = cbathy_diff[~np.isnan(cbathy_diff)] # entire area
        # cbathy_diff = cbathy_diff[np.logical_and(idx_iswatersurvey,~np.isnan(cbathy_diff))] # underwater area
        idx_cbathy = np.logical_and(np.logical_and(idx_iswatersurvey,~np.isnan(cbathy_diff)),idx_is_non_pier)
        cbathy_diff = cbathy_diff[idx_cbathy]  # underwater non-pier area 
        n_cbathy_diff = cbathy_diff.shape[0]
        
        #print(totsumsqr)
        r2_prior = 1. - (((prior_diff)**2).sum()/totsumsqr)
        r2_cur = 1. - (((cur_diff)**2).sum()/totsumsqr)
        r2_cbathy = 1. - (((cbathy_diff)**2).sum()/totsumsqr)

        fig, ax = plt.subplots(1,2,figsize=(10, 25))
        plt.subplots_adjust(wspace=0.3)

        ax[0].set_title('(a) Estimate vs. Survey %s' % (self.mydatetext))
        # ax[0].scatter(-h_survey[idx_cur], h_cur.reshape(ny,nx)[idx_cur], s = 0.5, color='blue',label='Blending ($R^2$: %5.2f)' % (r2_cur))
        # ax[0].scatter(-h_survey[idx_prior], h_prior.reshape(ny,nx)[idx_prior], s = 0.5, color='green',label='PBT ($R^2$: %5.2f)' % (r2_prior))
        # ax[0].scatter(-h_survey[idx_cbathy], h_cbathy.reshape(ny,nx)[idx_cbathy], s = 0.5, color='red',label='cBathy ($R^2$: %5.2f)' % (r2_cbathy))
        ax[0].scatter(-self.h_survey[idx_cur], self.h_cur.reshape(self.ny,self.nx)[idx_cur], s = 0.5, color='blue',label='$R^2$ (Blending): %5.2f' % (r2_cur))
        ax[0].scatter(-self.h_survey[idx_prior], self.h_prior.reshape(self.ny,self.nx)[idx_prior], s = 0.5, color='green',label='$R^2$ (PBT): %5.2f' % (r2_prior))
        ax[0].scatter(-self.h_survey[idx_cbathy], self.h_cbathy.reshape(self.ny,self.nx)[idx_cbathy], s = 0.5, color='red',label='$R^2$ (cBathy): %5.2f' % (r2_cbathy))


        ax[0].set_ylabel('estimate [m]')
        ax[0].set_xlabel('survey [m]')
        ax[0].plot([0.,8.0],[0.0,8.0],'k-')
        # ax.set_aspect('equal', adjustable='box')
        # plt.axis([0.01,1,0.01,1])
        # plt.xticks([0.01, 0.5 ,1.0])
        # plt.yticks([0.01, 0.5 ,1.0])
        #plt.plot([0.0,0.5],[0.0,0.5],'r-')
        ax[0].set_aspect('equal', adjustable='box')
        #plt.axis([0.0,8.0,0.0,0.5])
        ax[0].set_xticks([0, 4 , 8])
        ax[0].set_yticks([0, 4 , 8])
        ax[0].set_xlim([0,8])
        ax[0].set_ylim([0,8])
        l = ax[0].legend(markerscale=3,scatteryoffsets=[+0.75], fontsize = 10)#,handletextpad=0.0)#,handletextpad=-0.5)
        #legend(scatterpoints=1,scatteryoffsets=[0],handletextpad=-0.5)
        #for t in l.get_texts(): t.set_va('center_baseline')
        #for t in l.get_texts(): t.set_va('center')


        h_min_ = 0.00001
        # final result
        h_cur_kB = np.zeros((self.ny,self.nx,self.nfB))

        for j in range(self.nfB):
            h_cur_kB[:,:,j] = self.h_cur.reshape(self.ny,self.nx)

        obsidx_cur_hmin = np.copy(self.obsidx)
        #obsidx_cur_hmin[h_cur_kB < h_min] = False
        obsidx_cur_hmin[h_cur_kB < h_min_] = False

        #np.logical_and(~np.isnan(h_survey),(h_survey < 0.))

        k_cur = self._forward(self.h_cur,self.fB,obsidx_cur_hmin)
        obs_cur = self.k[obsidx_cur_hmin]

        rmse_cur = np.sqrt(((k_cur-obs_cur)**2.).sum()/obs_cur.shape[0])

        # prior result
        h_prior_kB = np.zeros((self.ny,self.nx,self.nfB))

        for i in range(self.nfB):
            h_prior_kB[:,:,i] = self.h_prior.reshape(self.ny,self.nx)

        obsidx_prior_hmin = np.copy(self.obsidx)
        obsidx_prior_hmin[h_prior_kB < h_min_] = False

        k_prior = self._forward(self.h_prior,self.fB,obsidx_prior_hmin)
        obs_prior = self.k[obsidx_prior_hmin]

        rmse_prior = np.sqrt(((k_prior-obs_prior)**2.).sum()/obs_prior.shape[0])

        if self.k_cur.reshape(-1).shape[0] > 1000000:
            ValueError('Too many observations to plot > 1 M')

        k_cur_obs = np.vstack([k_cur,obs_cur])
        k_cur_init_obs = np.vstack([k_prior,obs_prior])
        from scipy.stats import gaussian_kde
        z = gaussian_kde(k_cur_obs)(k_cur_obs)
        z_init = gaussian_kde(k_cur_init_obs)(k_cur_init_obs)

        # this needs to be user-defined..    
        alpha_arr = z/z.max()
        alpha_arr = alpha_arr*0.7 + 0.3

        alpha_arr_init = z_init/z_init.max()
        alpha_arr_init = alpha_arr_init*0.7 + 0.3
        
        from matplotlib.colors import to_rgb #, to_rgba
        
        def scatter(x, y, color, alpha_arr, **kwarg):
        # Sort the points by density, so that the densest points are plotted last
            idx = alpha_arr.argsort()
            x, y, alpha_arr = x[idx], y[idx], alpha_arr[idx]

            r, g, b = to_rgb(color)
            # r, g, b, _ = to_rgba(color)
            color = [(r, g, b, alpha) for alpha in alpha_arr]
            ax[1].scatter(x, y, c=color, **kwarg)


        ax[1].set_title('(b) Wave number $k_{f_B}$ fitting %s' % (self.mydatetext))

        scatter(obs_prior, k_prior, 'green', alpha_arr_init, s=1, label='PBT (RMSE: %7.3f [m$^{-1}$])' % (rmse_prior))
        scatter(obs_cur, k_cur, 'blue', alpha_arr, s=1, label='Blending (RMSE: %7.3f [m$^{-1}$])' % (rmse_cur))

        ax[1].set_xlabel('simulated wave number [m$^{-1}$]')
        ax[1].set_ylabel('observed wave number (cBathy Phase 1) [m$^{-1}$]')
        # plt.plot([0.01,1.0],[0.01,1.0],'r-')
        # ax.set_aspect('equal', adjustable='box')
        # plt.axis([0.01,1,0.01,1])
        # plt.xticks([0.01, 0.5 ,1.0])
        # plt.yticks([0.01, 0.5 ,1.0])
        ax[1].plot([0.0,0.5],[0.0,0.5],'r-')
        ax[1].set_aspect('equal', adjustable='box')
        #ax[1].axis([0.0,0.5,0.0,0.5])
        ax[1].set_xlim([0,0.5])
        ax[1].set_ylim([0,0.5])
        ax[1].set_xticks([0.0, 0.25 ,0.5])
        ax[1].set_yticks([0.0, 0.25 ,0.5])

        # leg = ax[1].legend(markerscale=3., fontsize = 10)
        # for lh in leg.legendHandles:
        #     lh.set_alpha(1)


        handles, labels = ax[1].get_legend_handles_labels()
        leg = ax[1].legend(handles[::-1], labels[::-1], loc='upper left', markerscale=3., fontsize = 10)

        for lh in leg.legendHandles:
            lh.set_alpha(1)

        plt.savefig('bathy_fitting_' + str(self.mydate) + '.png', dpi=150, bbox_inches='tight')
        plt.show()


    def plot_transects(self,yloc=None, rmste_xstart=None, rmste_xend=None):
        '''
            plot transect 
        '''
        if yloc is None:
            ValueError('loc is not defined')

        nx, ny = self.nx, self.ny
        x,y = self.x, self.y
        dim = len(yloc)
        
        #print(dim)
        fig = plt.figure(figsize=(4*dim,3*2))
        for i in range(dim):
            ax = fig.add_subplot(2,dim,i+1)
            ax.set_title(' yFRF = %d m' % (y[0,yloc[i]]))

            idx_iswatersurvey = np.logical_and(~np.isnan(self.h_survey[yloc[i],:]),(self.h_survey[yloc[i],:] < 0.))
            cur_diff = -self.h_survey[yloc[i],:] -self.h_cur.reshape(self.ny,self.nx)[yloc[i],:]
            cur_diff = cur_diff[np.logical_and(idx_iswatersurvey,~np.isnan(cur_diff))]
            n_cur_diff = cur_diff.shape[0]

            prior_diff = -self.h_survey[yloc[i],:] -self.h_prior.reshape(self.ny,self.nx)[yloc[i],:]
            prior_diff = prior_diff[np.logical_and(idx_iswatersurvey,~np.isnan(prior_diff))]
            n_prior_diff = prior_diff.shape[0]

            cbathy_diff = -self.h_survey[yloc[i],:] -self.h_cbathy[yloc[i],:]
            cbathy_diff = cbathy_diff[np.logical_and(idx_iswatersurvey,~np.isnan(cbathy_diff))]
            n_cbathy_diff = cbathy_diff.shape[0]

            rmse_prior = np.sqrt(( (prior_diff) **2.).sum()/n_prior_diff)
            rmse_blending = np.sqrt(( (cur_diff)**2.).sum()/n_cur_diff)
            rmse_cbathy = np.sqrt( ((cbathy_diff)**2.).sum()/n_cbathy_diff)
            
            rmste_blending = self.compute_rmste(x.reshape(-1), - self.h_cur.reshape(self.ny,self.nx)[yloc[i],:], self.h_survey[yloc[i],:],xstart = rmste_xstart, xend=rmste_xend)
            rmste_prior = self.compute_rmste(x.reshape(-1), -self.h_prior.reshape(self.ny,self.nx)[yloc[i],:],self.h_survey[yloc[i],:],xstart = rmste_xstart,xend=rmste_xend)
            rmste_cbathy = self.compute_rmste(x.reshape(-1),-self.h_cbathy.reshape(self.ny,self.nx)[yloc[i],:],self.h_survey[yloc[i],:],xstart = rmste_xstart,xend=rmste_xend)
            # print('rmse_prior: %f' % (rmse_prior))
            # print('rmse_blending: %f' % (rmse_blending))
            # print('rmse_cbathy: %f' % (rmse_cbathy))

            ax.plot(self.x.reshape(-1), self.h_survey[yloc[i],:],color='black',label='Survey (RMSE/RMSTE)')
            ax.plot(self.x.reshape(-1),-self.h_cur.reshape(self.ny,self.nx)[yloc[i],:],color='blue',label='Blending (%4.2f m/ %5.2f m$^2$)' % (rmse_blending,rmste_blending))
            ax.plot(self.x.reshape(-1),-self.h_prior.reshape(self.ny,self.nx)[yloc[i],:],'--',color='green',label='PBT (%4.2f m/ %5.2f m$^2$)' % (rmse_prior,rmste_prior))
            ax.plot(self.x.reshape(-1),-self.h_cbathy[yloc[i],:],color='red',label='cBathy (%4.2f m/ %5.2f m$^2$)' % (rmse_cbathy,rmste_cbathy))
            
            if i > 0:
                ax.set_yticks([])
            else:
                ax.set_yticks(range(-8,3,2))
                ax.set_ylabel('elevation [m]')

            ax.set_xticks([80, 200 , 400, 600, 800])
            ax.set_xlim([80,800])
            ax.set_ylim([-10,4])
            ax.legend(fontsize=9, markerscale = 0.75, handlelength = 1.25, handletextpad= 0.4, labelspacing = 0.1)
        
        #fig.suptitle('Bathymetry transect (11/22/2017)') # or plt.suptitle('Main title')

        for i in range(dim):
            ax = fig.add_subplot(2,dim,i+1+dim)
            #ax.set_title(' yFRF = %d m' % (y[0,loc[i]]))
            ax.plot(self.x.reshape(-1),np.abs(self.h_survey[yloc[i],:] + self.h_cur.reshape(ny,nx)[yloc[i],:]),color='blue',label='Blending abs. diff.')
            ax.plot(self.x.reshape(-1),np.abs(self.h_survey[yloc[i],:] + self.h_prior.reshape(ny,nx)[yloc[i],:]),'--',color='green',label='PBT abs. diff.')
            ax.plot(self.x.reshape(-1),np.abs(self.h_survey[yloc[i],:] + self.h_cbathy[yloc[i],:]),color='red',label='cBathy abs. diff.')
            if i > 0:
                ax.set_yticks([])
            else:
                ax.set_yticks(range(0,5,1))
                ax.set_ylabel('abs. error [m]')

            if i == 1:
                ax.set_xlabel('xFRF [m]')
            
            ax.set_xticks([80, 200 , 400, 600, 800])
            ax.set_xlim([80,800])
            ax.set_ylim([0,3])
            plt.legend(fontsize=9.5)

        fig.suptitle('Bathymetry transect (top) and absolute difference w.r.t survey (bottom) %s' % (self.mydatetext)) # or plt.suptitle('Main title')

        fig.tight_layout()
        fig.subplots_adjust(top=0.90)
        plt.savefig(join(self.output_dir,'transect_' + str(self.mydate) + '.png'), dpi=150, bbox_inches='tight')
        plt.show()

    def plot_obs_fitting(self):
        h_min_ = 0.00001
        # final result
        h_cur_kB = np.zeros((self.ny,self.nx,self.nfB))

        for j in range(self.nfB):
            h_cur_kB[:,:,j] = self.h_cur.reshape(self.ny,self.nx)

        obsidx_cur_hmin = np.copy(self.obsidx)
        #obsidx_cur_hmin[h_cur_kB < h_min] = False
        obsidx_cur_hmin[h_cur_kB < h_min_] = False

        #np.logical_and(~np.isnan(h_survey),(h_survey < 0.))

        k_cur = self._forward(self.h_cur,self.fB,obsidx_cur_hmin)
        obs_cur = self.k[obsidx_cur_hmin]

        rmse_cur = np.sqrt(((k_cur-obs_cur)**2.).sum()/obs_cur.shape[0])

        # prior result
        h_prior_kB = np.zeros((self.ny,self.nx,self.nfB))

        for i in range(self.nfB):
            h_prior_kB[:,:,i] = self.h_prior.reshape(self.ny,self.nx)

        obsidx_prior_hmin = np.copy(self.obsidx)
        #obsidx_prior_hmin[h_prior_kB < h_min] = False
        obsidx_prior_hmin[h_prior_kB < h_min_] = False

        k_prior = self._forward(self.h_prior,self.fB,obsidx_prior_hmin)
        obs_prior = self.k[obsidx_prior_hmin]

        rmse_prior = np.sqrt(((k_prior-obs_prior)**2.).sum()/obs_prior.shape[0])

        #print(rmse_cur,rmse_prior)
        #print(self.obs.shape,obs_prior.shape,k_prior.shape)
        #print(k_prior.shape)
        #print(np.sum(self.obsidx),np.sum(obsidx_prior_hmin),np.sum(obsidx_cur_hmin))

        # can be time-consuming for larger observations because it computes density for visualization
        if self.k_cur.reshape(-1).shape[0] > 1000000:
            ValueError('Too many observations to plot > 1 M')

        k_cur_obs = np.vstack([k_cur,obs_cur])
        k_cur_init_obs = np.vstack([k_prior,obs_prior])
        from scipy.stats import gaussian_kde
        z = gaussian_kde(k_cur_obs)(k_cur_obs)
        z_init = gaussian_kde(k_cur_init_obs)(k_cur_init_obs)

        # this needs to be user-defined..    
        alpha_arr = z/z.max()
        alpha_arr = alpha_arr*0.7 + 0.3

        alpha_arr_init = z_init/z_init.max()
        alpha_arr_init = alpha_arr_init*0.7 + 0.3
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('wave number $k_{f_B}$ fitting %s' % (self.mydatetext))

        from matplotlib.colors import to_rgb #, to_rgba
        
        def scatter(x, y, color, alpha_arr, **kwarg):
        # Sort the points by density, so that the densest points are plotted last
            idx = alpha_arr.argsort()
            x, y, alpha_arr = x[idx], y[idx], alpha_arr[idx]

            r, g, b = to_rgb(color)
            # r, g, b, _ = to_rgba(color)
            color = [(r, g, b, alpha) for alpha in alpha_arr]
            ax.scatter(x, y, c=color, **kwarg)

        scatter(obs_prior, k_prior, 'green', alpha_arr_init, s=1, label='PBT (RMSE: %7.3f [$m^{-1}$])' % (rmse_prior))
        scatter(obs_cur, k_cur, 'blue', alpha_arr, s=1,label='Blending (RMSE: %7.3f [$m^{-1}$])' % (rmse_cur))
        plt.xlabel('simulated wave number [m$^{-1}$]')
        plt.ylabel('observed wave number (cBathy Phase 1) [m$^{-1}$]')
        # plt.plot([0.01,1.0],[0.01,1.0],'r-')
        # ax.set_aspect('equal', adjustable='box')
        # plt.axis([0.01,1,0.01,1])
        # plt.xticks([0.01, 0.5 ,1.0])
        # plt.yticks([0.01, 0.5 ,1.0])
        plt.plot([0.0,0.5],[0.0,0.5],'r-')
        ax.set_aspect('equal', adjustable='box')
        plt.axis([0.0,0.5,0.0,0.5])
        plt.xticks([0.0, 0.25 ,0.5])
        plt.yticks([0.0, 0.25 ,0.5])
        
        # leg = plt.legend(markerscale=3., fontsize = 10)
        # for lh in leg.legendHandles:
        #     lh.set_alpha(1)
        
        handles, labels = ax.get_legend_handles_labels()
        leg = plt.legend(handles[::-1], labels[::-1], loc='upper left', markerscale=3., fontsize = 10)

        for lh in leg.legendHandles:
            lh.set_alpha(1)

        plt.savefig(join(self.output_dir,'k_fitting_' + str(self.mydate) + '.png'), dpi=150, bbox_inches='tight')
        plt.show()

    def compute_error(self):
        '''
            Compute RMSE and BIAS between survey and estimate
        '''
        idx_iswatersurvey = np.logical_and(~np.isnan(self.h_survey),(self.h_survey < 0.))
        #n_h_survey_water = np.sum(idx_iswatersurvey)
        #print(n_h_survey_water)

        YY, XX = np.meshgrid(self.x,self.y)
        
        idx_is_non_pier = np.logical_or((XX <= 400.),(XX >= 600.))
        #n_h_non_pier = np.sum(idx_is_non_pier)

        prior_diff = -self.h_survey-self.h_prior.reshape(self.ny,self.nx)
        # prior_diff = prior_diff[~np.isnan(prior_diff)] # entire area
        # prior_diff = prior_diff[np.logical_and(idx_iswatersurvey,~np.isnan(prior_diff))] # underwater area
        prior_diff = prior_diff[np.logical_and(np.logical_and(idx_iswatersurvey,~np.isnan(prior_diff)),idx_is_non_pier)] # underwater non-pier area 
        n_prior_diff = prior_diff.shape[0]
        
        cur_diff = -self.h_survey-self.h_cur.reshape(self.ny,self.nx)
        # cur_diff = cur_diff[~np.isnan(cur_diff)] # entire area 
        # cur_diff = cur_diff[np.logical_and(idx_iswatersurvey,~np.isnan(cur_diff))] # underwater area
        cur_diff = cur_diff[np.logical_and(np.logical_and(idx_iswatersurvey,~np.isnan(cur_diff)),idx_is_non_pier)] # underwater non-pier area 
        n_cur_diff = cur_diff.shape[0]
        
        cbathy_diff = -self.h_survey-self.h_cbathy.reshape(self.ny,self.nx)
        # cbathy_diff = cbathy_diff[~np.isnan(cbathy_diff)] # entire area
        # cbathy_diff = cbathy_diff[np.logical_and(idx_iswatersurvey,~np.isnan(cbathy_diff))] # underwater area
        cbathy_diff = cbathy_diff[np.logical_and(np.logical_and(idx_iswatersurvey,~np.isnan(cbathy_diff)),idx_is_non_pier)]  # underwater non-pier area 
        n_cbathy_diff = cbathy_diff.shape[0]

        rmse_prior = np.sqrt(( (prior_diff) **2.).sum()/n_prior_diff)
        rmse_blending = np.sqrt(( (cur_diff)**2.).sum()/n_cur_diff)
        rmse_cbathy = np.sqrt( ((cbathy_diff)**2.).sum()/n_cbathy_diff)

        print('rmse blending, cbathy, prior : %f, %f, %f' % (rmse_blending, rmse_cbathy, rmse_prior))
        
        bias_prior = np.mean(prior_diff)
        bias_blending = np.mean(cur_diff)
        bias_cbathy = np.mean(cbathy_diff)

        print('bias blending, cbathy, prior : %f, %f, %f' % (bias_blending, bias_cbathy, bias_prior))
        
        return rmse_blending, rmse_cbathy, rmse_prior, bias_blending, bias_cbathy, bias_prior


    def compute_rmste(self, x, h_pred, h_true, xstart=None, xend=None):
        '''
            Compute RMSTE between survey and estimate
        '''
        from scipy import integrate
        from scipy.integrate import solve_bvp
        from scipy.interpolate import interp1d
        
        src = h_pred - h_true

        if x.ndim != 1 or src.ndim != 1:
            ValueError('x and src must be 1D array')

        if x.shape[0] != src.shape[0]:
            ValueError('x.dim and src.dim are different')

        # find the first non-NaN value         
        if xstart is None:
            xstart = x[np.where(~np.isnan(src))[0][0]]

    # we compute RMSTE only underwater area
        xstart_h_true = x[np.where(h_true < 0.)[0][0]]
        
        if xstart_h_true > xstart:
            xstart = xstart_h_true
        
        # find the last non-NaN value         
        if xend is None:
            xend = x[np.where(~np.isnan(src))[0][-1]]

        m = 100 # initial pt
        x0 = np.linspace(xstart,xend,m)
        y0 = np.zeros((2,m))

        idx = ~np.isnan(src)
        src_interp = interp1d(x[idx], src[idx])
            
        def f(x, y):
            "right side of ODE y=[phi, q], dphi/dx = q, dq/dx = k"
            return np.vstack([y[1,:], src_interp(x)]) # 

        def bc(ya, yb): # ya = y[:,0], yb = y[:,-1]
            "boundary condition"
            return np.array([ya[-1],yb[0]]) # q[0] = 0 (closed boundary), phi[-1] = 0 (unconstrained boundary)
            #return np.array([ya[0],yb[0]]) # phi[0] = 0 (unconstrained boundary), phi[-1] = 0 (unconstrained boundary)
            
        # solve bvp
        sol = solve_bvp(f, bc, x0, y0)

        sol_q_func_square = lambda x: (sol.sol.__call__(x)[1])**2.
        result = np.sqrt(integrate.quad(sol_q_func_square, xstart,xend, limit=200))
        rmste = result[0]/np.sqrt(xend - xstart) 
        return rmste

    def plot_cbathy_errors(self, x = None, y = None, kErr = None, hTempErr = None, nfB = None, mydatetext = None):
        '''
            plot kErr and hTempErr
        '''
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if kErr is None:
            kErr = self.kErr
        if hTempErr is None:
            hTempErr = self.hTempErr
        if nfB is None:
            nfB = self.nfB

        YY, XX = np.meshgrid(x,y)

        fig, ax = plt.subplots(2,nfB,figsize=(16,12))
        plt.suptitle('Cbathy Error ' + str(mydatetext), fontsize=15)

        for i in range(nfB):
            im = ax[0,i].pcolormesh(YY,XX,kErr[:,:,i],cmap='jet')
            ax[0,i].set_title('kErr #%d' % (i))
            ax[0,i].set_xticks([80,800])
            ax[0,i].set_yticks([])
            fig.colorbar(im,ax=ax[0,i])

        for i in range(nfB):
            im = ax[1,i].pcolormesh(YY,XX,hTempErr[:,:,i],cmap='jet')
            ax[1,i].set_title('hTempErr #%d' % (i))
            ax[1,i].set_xticks([80,800])
            ax[1,i].set_yticks([])
            fig.colorbar(im,ax=ax[1,i])

        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
        plt.show()

if __name__ == '__main__':
    import bathyblendingERDC as bathyinv
    import numpy as np
    params = {'use_testdata':1} # use 042916 

    DuckBathy = bathyinv.BathyBlending(params=params)
    DuckBathy.blend()
    #DuckBathy.plot_bathy()
    #DuckBathy.plot_slice()
