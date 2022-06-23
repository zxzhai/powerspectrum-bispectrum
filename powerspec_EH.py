import sys
import numpy as np
import scipy as sp
from scipy.integrate import quad, trapz
from scipy.integrate import simps
from scipy import optimize
import math
from scipy.misc import derivative
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from astropy.coordinates import SkyCoord
from astropy import units as u

class powerspec:
    def __init__(self, Omegab=0.05, Omegac=0.25, h=0.7, ns=0.965, sigma8=0.8, T2=2.7255/2.7, matter_cf=False, matter_cf_z=0.0):
        self.Omegab = Omegab
        self.Omegac = Omegac
        self.h = h
        self.ns = ns
        self.sigma8 = sigma8
        self.T2 = T2
        self.matter_cf = matter_cf
        self.matter_cf_z = matter_cf_z

        self.Omegam = self.Omegab+self.Omegac
        self.OmegaL = 1.0-self.Omegam

        self.zeq = 2.50*10.0**4*self.Omegam*self.h*self.h*self.T2**(-4.0)  # eq(2)
        self.keq = 7.46*0.01*self.Omegam*self.h*self.h*self.T2**(-2.0)   # eq(3)

        self.b1 = 0.313*(self.Omegam*self.h*self.h)**(-0.419)*(1.0+0.607*(self.Omegam*self.h*self.h)**(0.674))
        self.b2 = 0.238*(self.Omegam*self.h*self.h)**(0.223)
        self.zd= 1291.0*(self.Omegam*self.h*self.h)**(0.251)/(1.0+0.659*(self.Omegam*self.h*self.h)**(0.828))*(1.0+self.b1*(self.Omegab*self.h*self.h)**self.b2)  # eq(4)

        self.Rd =  31.5*self.Omegab*self.h*self.h*self.T2**(-4.0)*(1000.0/self.zd)
        self.Req = 31.5*self.Omegab*self.h*self.h*self.T2**(-4.0)*(1000.0/self.zeq)

        self.s = 2.0/3.0/self.keq*((6.0/self.Req)**0.5)*math.log(((1.0+self.Rd)**0.5+(self.Rd+self.Req)**0.5)/(1.0+self.Req**0.5)) #  eq(6)
        self.ks = 1.6*(self.Omegab*self.h*self.h)**0.52*(self.Omegam*self.h*self.h)**0.73*(1.0+(10.4*self.Omegam*self.h*self.h)**(-0.95))

        self.a1 = (46.9*self.Omegam*self.h*self.h)**(0.670)*(1.0+(32.1*self.Omegam*self.h*self.h)**(-0.532))
        self.a2 = (12.0*self.Omegam*self.h*self.h)**(0.424)*(1.0+(45.0*self.Omegam*self.h*self.h)**(-0.582))
        self.alphac = self.a1**(-self.Omegab/self.Omegam)*self.a2**(-(self.Omegab/self.Omegam)**3)  # eq(11)

        self.bb1 = 0.944*(1.0+(458.0*self.Omegam*self.h*self.h)**(-0.708))**(-1.0)
        self.bb2 = (0.395*self.Omegam*self.h*self.h)**(-0.0266)
        self.betac = 1.0/(1.0+self.bb1*((self.Omegac/self.Omegam)**self.bb2-1.0))  # eq(12)

        self.y = (1.0+self.zeq)/(1.0+self.zd)
        self.G = self.y*(-6.0*(1.0+self.y)**(0.5)+(2.0+3.0*self.y)*math.log(((1.0+self.y)**0.5+1.0)/((1.0+self.y)**0.5-1.0))) # eq(15)
        self.alphab = 2.07*self.keq*self.s*(1.0+self.Rd)**(-0.75)*self.G  # eq(14)

        self.beta_n = 8.41*(self.Omegam*self.h*self.h)**0.435

        self.betab = 0.5+self.Omegab/self.Omegam+(3.0-2.0*self.Omegab/self.Omegam)*((17.2*self.Omegam*self.h*self.h)*2.0+1.0)**0.5
        
        self.A = self.sigma8**2.0/quad(self.A_int, 0, np.infty)[0]
    
        if self.matter_cf == True:
            self.matter_cf_xx, self.matter_cf_yy = self.xi_matter2(z=self.matter_cf_z)

    def T_tilt_0(self, k, alphac, betac):
    
        q = k/(13.41*self.keq)  #  eq(10)
        f = 1.0/(1.0+(k*self.s/5.4)**4)
        C = 14.2/self.alphac+386.0/(1.0+69.9*q**1.08)
        return math.log(math.e+1.8*self.betac*q)/(math.log(math.e+1.8*self.betac*q)+C*q*q)

    def T(self, k):
        q = k/(13.41*self.keq)
        f = 1.0/(1.0+(k*self.s/5.4)**4.0)
        C = 14.2/self.alphac+386.0/(1.0+69.9*q**1.08)
        ss = self.s/(1.0+(self.beta_n/k/self.s)**3.0)**(1.0/3.0)
    
        Tc = f*self.T_tilt_0(k,1.0,self.betac)+(1.0-f)*self.T_tilt_0(k,self.alphac,self.betac)
        Tb = (self.T_tilt_0(k,1.0,1.0)/(1.0+(k*self.s/5.2)**2.0)+self.alphab/(1.0+(self.betab/k/self.s)**3.0)*math.e**(-(k/self.ks)**1.4))*np.sin(k*ss)/(k*ss)
        return self.Omegab/self.Omegam*Tb+self.Omegac/self.Omegam*Tc
    
    
    def T0(self, k):
        q = k/(13.41*self.keq)
        C0 = 14.2 + 731.0/(1.0+62.5*q)
        L0 = np.log(2.0*np.e+1.8*q)
        return L0/(L0+C0*q*q)
    
    def W(self, k, R):
        return 3.0/((k*R)**3.0)*(math.sin(k*R)-(k*R)*math.cos(k*R))

    def A_int(self, k):
        return 1.0/2.0/math.pi**2.0*(k**(2.0+self.ns))*self.T(k)**2*self.W(k,8.0/self.h)**2.0

    def P_t(self, k, z):
        P_lin = k**self.ns*self.T(k)**2.0*self.D(z)**2.0
    
        return self.A*P_lin
    
    def D(self, z):
        Omegak = 1.0-self.Omegam-self.OmegaL
        Omegaz = self.Omegam*(1.0+z)**3.0/(self.OmegaL+Omegak*(1.0+z)**2.0+self.Omegam*(1.0+z)**3.0)
        OmegaLz = self.OmegaL/(self.OmegaL+Omegak*(1.0+z)**2.0+self.Omegam*(1.0+z)**3.0)
        D1_z = (1.0+z)**(-1.0)*5.0*Omegaz/2.0*(Omegaz**(4.0/7.0)-OmegaLz+(1.0+Omegaz/2.0)*(1.0+OmegaLz/70.0))**(-1.0)
        D1_0 = 5.0*self.Omegam/2.0*(self.Omegam**(4.0/7.0)-self.OmegaL+(1.0+self.Omegam/2.0)*(1.0+self.OmegaL/70.0))**(-1.0)
        return D1_z/D1_0
    
    def D_nonnorm(self, z):
        Omegak = 1.0-self.Omegam-self.OmegaL
        Omegaz = self.Omegam*(1.0+z)**3.0/(self.OmegaL+Omegak*(1.0+z)**2.0+self.Omegam*(1.0+z)**3.0)
        OmegaLz = self.OmegaL/(self.OmegaL+Omegak*(1.0+z)**2.0+self.Omegam*(1.0+z)**3.0)
        D1_z = (1.0+z)**(-1.0)*5.0*Omegaz/2.0*(Omegaz**(4.0/7.0)-OmegaLz+(1.0+Omegaz/2.0)*(1.0+OmegaLz/70.0))**(-1.0)
        return D1_z
    
    def f(self, z):
        return -derivative(self.D, z, dx=1e-6)*(1.0+z)/self.D(z)

    def sigma_sqr_int(self, k, R, z=0):
        return 1.0/2.0/np.pi**2.0*k*k*self.P_t(k, z)*self.W(k, R)**2.0

    def sigma_sqr(self, R, z=0):
        f = lambda k: self.sigma_sqr_int(k, R, z)
        return quad(f, 0.0, np.infty)[0]
    
    
    def sigma_sqr_lnk_int(self, k, R, z=0):
        return 1.0/2.0/np.pi**2.0*k*k*self.P_t(k, z)*self.W(k, R)**2.0*np.log(k)

    def sigma_sqr_lnk(self, R, z=0):
        f = lambda k: self.sigma_sqr_lnk_int(k, R, z)
        return quad(f, 0.0, np.infty)[0]
    
    

    def Delta_sqrt(self, k, z=0):  # not very relavent,  dimensionless power spectrum
        return k**3.0*self.P_t(k, z)/2.0/np.pi**2.0

    def sigma_sqr_dR(self, R, z=0):
        f = lambda R: self.sigma_sqr(R, z)
        return derivative(f, R, dx=1e-6)

    def xi_matter_int(self, k, r, z=0):
        return self.Delta_sqrt(k, z)*np.sin(k*r)/(k*r)/k

    def xi_matter(self, r, z=0):
        f = lambda k: self.xi_matter_int(k, r, z)
        return quad(f, 0.0, np.infty)[0]
        #return quad(f, 0.0, 10)[0]

    def xi_matter2(self, z=0):
        rhi = 210.0
        rlo = 0.001
        n = 300
        tolerance=1.0e-6
        npint = 16.0
        dlogr = (np.log(rhi)-np.log(rlo))/(n-1.0)
        x = np.empty(n)
        y = np.empty(n)
        for i in range(int(n)):
            klo = 1.0e-6
            r_g4 = np.exp((i-1.0)*dlogr)*rlo
            f = lambda k: self.xi_matter_int(k, r_g4, z)
            x[i] = np.exp((i-1.0)*dlogr)*rlo
            j=1.0
            kkx = np.linspace(klo, j/r_g4, npint)
            kky = map(f, kkx)
            s1 = trapz(kky, kkx)
            s2 = s1
            klo = j/r_g4
            while abs(s1)>tolerance*abs(s2):
                j+=16.0;
                kkx = np.linspace(klo, j/r_g4, npint)
                kky = map(f, kkx)
                s1 = trapz(kky, kkx)
                s2 += s1;
                klo = j/r_g4;
            y[i] = s2
        return x, y

    def xi_matter_interp(self, r, z=0):
        f = InterpolatedUnivariateSpline(self.matter_cf_xx, self.matter_cf_yy, k=2)
        return f(r)


class powerspecNL:
    def __init__(self, Omegab=0.05, Omegac=0.25, h=0.7, ns=0.965, sigma8=0.8, T2=2.7255/2.7, matter_cf=False, matter_cf_z=0.0):
        self.Omegab = Omegab
        self.Omegac = Omegac
        self.h = h
        self.ns = ns
        self.sigma8 = sigma8
        self.T2 = T2
        self.matter_cf = matter_cf
        self.matter_cf_z = matter_cf_z
        
        self.Omegam = self.Omegab+self.Omegac
        self.OmegaL = 1.0-self.Omegam
        
        self.zeq = 2.50*10.0**4*self.Omegam*self.h*self.h*self.T2**(-4.0)  # eq(2)
        self.keq = 7.46*0.01*self.Omegam*self.h*self.h*self.T2**(-2.0)   # eq(3)
        
        self.b1 = 0.313*(self.Omegam*self.h*self.h)**(-0.419)*(1.0+0.607*(self.Omegam*self.h*self.h)**(0.674))
        self.b2 = 0.238*(self.Omegam*self.h*self.h)**(0.223)
        self.zd= 1291.0*(self.Omegam*self.h*self.h)**(0.251)/(1.0+0.659*(self.Omegam*self.h*self.h)**(0.828))*(1.0+self.b1*(self.Omegab*self.h*self.h)**self.b2)  # eq(4)
        
        self.Rd =  31.5*self.Omegab*self.h*self.h*self.T2**(-4.0)*(1000.0/self.zd)
        self.Req = 31.5*self.Omegab*self.h*self.h*self.T2**(-4.0)*(1000.0/self.zeq)
        
        self.s = 2.0/3.0/self.keq*((6.0/self.Req)**0.5)*math.log(((1.0+self.Rd)**0.5+(self.Rd+self.Req)**0.5)/(1.0+self.Req**0.5)) #  eq(6)
        self.ks = 1.6*(self.Omegab*self.h*self.h)**0.52*(self.Omegam*self.h*self.h)**0.73*(1.0+(10.4*self.Omegam*self.h*self.h)**(-0.95))
        
        self.a1 = (46.9*self.Omegam*self.h*self.h)**(0.670)*(1.0+(32.1*self.Omegam*self.h*self.h)**(-0.532))
        self.a2 = (12.0*self.Omegam*self.h*self.h)**(0.424)*(1.0+(45.0*self.Omegam*self.h*self.h)**(-0.582))
        self.alphac = self.a1**(-self.Omegab/self.Omegam)*self.a2**(-(self.Omegab/self.Omegam)**3)  # eq(11)
        
        self.bb1 = 0.944*(1.0+(458.0*self.Omegam*self.h*self.h)**(-0.708))**(-1.0)
        self.bb2 = (0.395*self.Omegam*self.h*self.h)**(-0.0266)
        self.betac = 1.0/(1.0+self.bb1*((self.Omegac/self.Omegam)**self.bb2-1.0))  # eq(12)
        
        self.y = (1.0+self.zeq)/(1.0+self.zd)
        self.G = self.y*(-6.0*(1.0+self.y)**(0.5)+(2.0+3.0*self.y)*math.log(((1.0+self.y)**0.5+1.0)/((1.0+self.y)**0.5-1.0))) # eq(15)
        self.alphab = 2.07*self.keq*self.s*(1.0+self.Rd)**(-0.75)*self.G  # eq(14)
        
        self.beta_n = 8.41*(self.Omegam*self.h*self.h)**0.435
        
        self.betab = 0.5+self.Omegab/self.Omegam+(3.0-2.0*self.Omegab/self.Omegam)*((17.2*self.Omegam*self.h*self.h)*2.0+1.0)**0.5
        
        self.A = self.sigma8**2.0/quad(self.A_int, 0, np.infty)[0]
        
        if self.matter_cf == True:
            self.matter_cf_xx, self.matter_cf_yy = self.xi_matter2(z=self.matter_cf_z)

        self.R_nl = optimize.brentq(self.k_nl_equ, 0.01, 20)
        self.k_nl = 1.0/self.R_nl
        self.neff = -3.0-self.dlnS2dlnR(np.log(self.R_nl), z=0)
        self.Ccur = -self.dlnS2dlnR_2(np.log(self.R_nl), z=0)
            
        self.an=10.0**(1.4861 + 1.8369*self.neff + 1.6762*self.neff**2.0 + 0.7940*self.neff**3.0 + 0.1670*self.neff**4.0 - 0.6202*self.Ccur)
        self.bn=10.0**(0.9463 + 0.9466*self.neff + 0.3084*self.neff**2.0 - 0.9400*self.Ccur)
                    
        self.cn= 10.0**(-0.2807 + 0.6669*self.neff + 0.3214*self.neff**2.0 - 0.0793*self.Ccur)
                        
        self.gamma =  0.8649 + 0.2989*self.neff + 0.1631*self.Ccur
        self.alpha =  1.3884 + 0.3700*self.neff - 0.1452*self.neff**2.0
        self.beta  =  0.8291 + 0.9854*self.neff + 0.3401*self.neff**2.0
        self.mu    = 10.0**(-3.5442 + 0.1908*self.neff)
        self.nu    = 10.0**(0.9589 + 1.2857*self.neff)
            
        self.f1b = self.Omegam**(-0.0307)
        self.f2b = self.Omegam**(-0.0585)
        self.f3b = self.Omegam**(+0.0743)

    def T_tilt_0(self, k, alphac, betac):
    
        q = k/(13.41*self.keq)  #  eq(10)
        f = 1.0/(1.0+(k*self.s/5.4)**4)
        C = 14.2/self.alphac+386.0/(1.0+69.9*q**1.08)
        return math.log(math.e+1.8*self.betac*q)/(math.log(math.e+1.8*self.betac*q)+C*q*q)
    
    def T(self, k):
        q = k/(13.41*self.keq)
        f = 1.0/(1.0+(k*self.s/5.4)**4.0)
        C = 14.2/self.alphac+386.0/(1.0+69.9*q**1.08)
        ss = self.s/(1.0+(self.beta_n/k/self.s)**3.0)**(1.0/3.0)
        
        Tc = f*self.T_tilt_0(k,1.0,self.betac)+(1.0-f)*self.T_tilt_0(k,self.alphac,self.betac)
        Tb = (self.T_tilt_0(k,1.0,1.0)/(1.0+(k*self.s/5.2)**2.0)+self.alphab/(1.0+(self.betab/k/self.s)**3.0)*math.e**(-(k/self.ks)**1.4))*np.sin(k*ss)/(k*ss)
        return self.Omegab/self.Omegam*Tb+self.Omegac/self.Omegam*Tc
    
    
    def T0(self, k):
        q = k/(13.41*self.keq)
        C0 = 14.2 + 731.0/(1.0+62.5*q)
        L0 = np.log(2.0*np.e+1.8*q)
        return L0/(L0+C0*q*q)
    
    def W(self, k, R):
        return np.exp(-k*k*R*R/2.0)
    
    def W1(self, k, R):
        return 3.0/((k*R)**3.0)*(math.sin(k*R)-(k*R)*math.cos(k*R))
    
    def A_int(self, k):
        return 1.0/2.0/math.pi**2.0*(k**(2.0+self.ns))*self.T(k)**2.0*self.W1(k, 8.0/self.h)**2.0
    
    def P_t(self, k, z):
        P_lin = k**self.ns*self.T(k)**2.0*self.D(z)**2.0
        
        return self.A*P_lin
    
    def D(self, z):
        Omegak = 1.0-self.Omegam-self.OmegaL
        Omegaz = self.Omegam*(1.0+z)**3.0/(self.OmegaL+Omegak*(1.0+z)**2.0+self.Omegam*(1.0+z)**3.0)
        OmegaLz = self.OmegaL/(self.OmegaL+Omegak*(1.0+z)**2.0+self.Omegam*(1.0+z)**3.0)
        D1_z = (1.0+z)**(-1.0)*5.0*Omegaz/2.0*(Omegaz**(4.0/7.0)-OmegaLz+(1.0+Omegaz/2.0)*(1.0+OmegaLz/70.0))**(-1.0)
        D1_0 = 5.0*self.Omegam/2.0*(self.Omegam**(4.0/7.0)-self.OmegaL+(1.0+self.Omegam/2.0)*(1.0+self.OmegaL/70.0))**(-1.0)
        return D1_z/D1_0
    
    def D_nonnorm(self, z):
        Omegak = 1.0-self.Omegam-self.OmegaL
        Omegaz = self.Omegam*(1.0+z)**3.0/(self.OmegaL+Omegak*(1.0+z)**2.0+self.Omegam*(1.0+z)**3.0)
        OmegaLz = self.OmegaL/(self.OmegaL+Omegak*(1.0+z)**2.0+self.Omegam*(1.0+z)**3.0)
        D1_z = (1.0+z)**(-1.0)*5.0*Omegaz/2.0*(Omegaz**(4.0/7.0)-OmegaLz+(1.0+Omegaz/2.0)*(1.0+OmegaLz/70.0))**(-1.0)
        return D1_z
    
    def f(self, z):
        return -derivative(self.D, z, dx=1e-6)*(1.0+z)/self.D(z)
    
    def sigma_sqr_int(self, k, R, z=0):
        return 1.0/2.0/np.pi**2.0*k*k*self.P_t(k, z)*self.W(k, R)**2.0
    
    def sigma_sqr(self, R, z=0):
        f = lambda k: self.sigma_sqr_int(k, R, z)
        return quad(f, 0.0, np.infty)[0]
    
    def sigma_sqr_lnk_int(self, k, R, z=0):
        return 1.0/2.0/np.pi**2.0*k*k*self.P_t(k, z)*self.W(k, R)**2.0*np.log(k)
    
    def sigma_sqr_lnk(self, R, z=0):
        f = lambda k: self.sigma_sqr_lnk_int(k, R, z)
        return quad(f, 0.0, np.infty)[0]
    
    
    
    def sigma_sqr_int2(self, k, R, z=0):
        return 1.0/2.0/np.pi**2.0*k*k*self.P_nl(k, z)*self.W1(k, R)**2.0
    
    def sigma_sqr2(self, R, z=0):
        f = lambda k: self.sigma_sqr_int2(k, R, z)
        return quad(f, 0.0, np.infty)[0]
    
    def sigma_sqr_lnk_int2(self, k, R, z=0):
        return 1.0/2.0/np.pi**2.0*k*k*self.P_nl(k, z)*self.W1(k, R)**2.0*np.log(k)
    
    def sigma_sqr_lnk2(self, R, z=0):
        f = lambda k: self.sigma_sqr_lnk_int2(k, R, z)
        return quad(f, 0.0, np.infty)[0]
    
    
    
    
    def Delta_sqrt(self, k, z=0):  # not very relavent,  dimensionless power spectrum
        return k**3.0*self.P_t(k, z)/2.0/np.pi**2.0
    
    def sigma_sqr_dR(self, R, z=0):
        f = lambda R: self.sigma_sqr(R, z)
        return derivative(f, R, dx=1e-6)
    
    def xi_matter_int(self, k, r, z=0):
        return self.Delta_sqrt(k, z)*np.sin(k*r)/(k*r)/k
    
    def xi_matter(self, r, z=0):
        f = lambda k: self.xi_matter_int(k, r, z)
        return quad(f, 0.0, np.infty)[0]
    
    def xi_matter2(self, z=0):
        rhi = 210.0
        rlo = 0.001
        n = 300.0
        tolerance=1.0e-6
        npint = 16.0
        dlogr = (np.log(rhi)-np.log(rlo))/(n-1.0)
        x = np.empty(n)
        y = np.empty(n)
        for i in range(int(n)):
            klo = 1.0e-6
            r_g4 = np.exp((i-1.0)*dlogr)*rlo
            f = lambda k: self.xi_matter_int(k, r_g4, z)
            x[i] = np.exp((i-1.0)*dlogr)*rlo
            j=1.0
            kkx = np.linspace(klo, j/r_g4, npint)
            kky = map(f, kkx)
            s1 = trapz(kky, kkx)
            s2 = s1
            klo = j/r_g4
            while abs(s1)>tolerance*abs(s2):
                j+=16.0;
                kkx = np.linspace(klo, j/r_g4, npint)
                kky = map(f, kkx)
                s1 = trapz(kky, kkx)
                s2 += s1;
                klo = j/r_g4;
            y[i] = s2
        return x, y
    
    def xi_matter_interp(self, r, z=0):
        f = InterpolatedUnivariateSpline(self.matter_cf_xx, self.matter_cf_yy, k=2)
        return f(r)
    
    
    def DeltaLin(self, k, z=0):
        return k**3.0/(2.0*np.pi*np.pi)*self.P_t(k, z)
                                  
    def k_nl_equ(self, R):
        return self.sigma_sqr(R, z=0.0)**0.5-1.0

    def dlnS2dlnR(self, lnR, z=0):
        f = lambda lnR: np.log(self.sigma_sqr(np.exp(lnR), z))
        return derivative(f, lnR, dx=1e-6, n=1)

    def dlnS2dlnR_2(self, lnR, z=0):
        f = lambda lnR: np.log(self.sigma_sqr(np.exp(lnR), z))
        return derivative(f, lnR, dx=1e-6, n=2)
    
    def Delta_NL(self, k, z=0):
        y = k/self.k_nl
        fy = y/4.0 + y*y/8.0
        DeltaQ = self.DeltaLin(k, z=0)*(1.0+self.DeltaLin(k,z=0))**self.beta/(1.0+self.alpha*self.DeltaLin(k,z=0))*np.exp(-fy)
        DeltaHp = self.an*y**(3.0*self.f1b)/(1.0+self.bn*y**self.f2b+(self.cn*self.f3b*y)**(3.0-self.gamma))
        DeltaH = DeltaHp/(1.0+self.mu/y+self.nu/(y*y))
        return (DeltaQ+DeltaH)*self.D(z)**2.0

    def P_nl(self, k, z=0):
        return self.Delta_NL(k, z)*(2.0*np.pi**2.0)/(k**3.0)



class BP_model:
    def __init__(self, P_L, s=np.array([0, 0, 1]), cosmo=None):
        self.P_L = P_L
        self.s = s
        self.cosmo = cosmo

    def func_f(self, z):  #  the temporary fitting formula, more accurate results should be from camb
        return self.cosmo.Om(z)**0.55

    def Z1(self, mu, b1, f):
        return b1 + f*mu**2

    def mij(self, ki, kj):
        magki = np.sqrt(ki.dot(ki))
        magkj = np.sqrt(kj.dot(kj))
        return ki.dot(kj)/(magki*magkj)

    def S2(self, ki, kj):
        return self.mij(ki, kj)**2.0 - 1.0/3.0

    def F2(self, ki, kj):
        magki = np.sqrt(ki.dot(ki))
        magkj = np.sqrt(kj.dot(kj))
        return 5.0/7.0 + self.mij(ki, kj)/2.0*(magki/magkj + magkj/magki) + 2.0/7.0 * self.mij(ki, kj)**2.0

    def G2(self, ki, kj):
        magki = np.sqrt(ki.dot(ki))
        magkj = np.sqrt(kj.dot(kj))
        return 3.0/7.0 + self.mij(ki, kj)/2.0*(magki/magkj + magkj/magki) + 4.0/7.0 * self.mij(ki, kj)**2.0

    def Z2(self, ki, kj, b1, b2, bs2, f):
        kij = ki+kj
        magki = np.sqrt(ki.dot(ki))
        magkj = np.sqrt(kj.dot(kj))
        magkij = np.sqrt(kij.dot(kij))
    
        mui = ki.dot(self.s)/magki
        muj = kj.dot(self.s)/magkj
        muij = kij.dot(self.s)/magkij
        return b2/2.0 + b1*self.F2(ki, kj) + f*muij**2.0*self.G2(ki, kj) + f*muij*magkij/2.0*(mui/magki*self.Z1(muj, b1, f)+muj/magkj*self.Z1(mui, b1, f)) + bs2/2.0*self.S2(ki, kj)

    def P(self, k, mu, sigmap, z, b1, f):
        return self.Z1(mu, b1, f)**2.0 * self.P_L(k, z)*np.exp(-(k*mu*sigmap)**2.0 / 2.0)

    def B(self, k1, k2, k3, z, b1, b2, bs2, f, sigmap):
        magk1 = np.sqrt(k1.dot(k1))
        magk2 = np.sqrt(k2.dot(k2))
        magk3 = np.sqrt(k3.dot(k3))
        mu1 = k1.dot(self.s)/magk1
        mu2 = k2.dot(self.s)/magk2
        mu3 = k3.dot(self.s)/magk3
        return 2.0*( self.Z2(k1, k2, b1, b2, bs2, f)*self.Z1(mu1, b1, f)*self.Z1(mu2, b1, f)*self.P_L(magk1, z)*self.P_L(magk2, z) + self.Z2(k2, k3, b1, b2, bs2, f)*self.Z1(mu2, b1, f)*self.Z1(mu3, b1, f)*self.P_L(magk2, z)*self.P_L(magk3, z) + self.Z2(k1, k3, b1, b2, bs2, f)*self.Z1(mu1, b1, f)*self.Z1(mu3, b1, f)*self.P_L(magk1, z)*self.P_L(magk3, z) )*np.exp(-(magk1**2.0*mu1**2.0 + magk2**2.0*mu2**2.0 + magk3**2.0*mu3**2.0)*sigmap**2.0/2.0)

    def Ptilt(self, k, mu, sigmap, z, b1, f, ng):
        return self.P(k, mu, sigmap, z, b1, f) + 1./ng

    def Btilt(self, k1, k2, k3, z, b1, b2, bs2, f, sigmap, ng):
        magk1 = np.sqrt(k1.dot(k1))
        magk2 = np.sqrt(k2.dot(k2))
        magk3 = np.sqrt(k3.dot(k3))
        mu1 = k1.dot(self.s)/magk1
        mu2 = k2.dot(self.s)/magk2
        mu3 = k3.dot(self.s)/magk3

        return self.B(k1, k2, k3, z, b1, b2, bs2, f, sigmap) + ( self.P(magk1, mu1, sigmap, z, b1, f) + self.P(magk2, mu2, sigmap, z, b1, f) + self.P(magk3, mu3, sigmap, z, b1, f))/ng + 1./ng**2.0


    def B_real(self, k1, k2, k3, z, b1, b2, bs2, f, sigmap):   # should be wrong, not tested!!!!!!!
        magk1 = np.sqrt(k1.dot(k1))
        magk2 = np.sqrt(k2.dot(k2))
        magk3 = np.sqrt(k3.dot(k3))
        mu1 = k1.dot(self.s)/magk1
        mu2 = k2.dot(self.s)/magk2
        mu3 = k3.dot(self.s)/magk3

        return 2.0*( self.F2(k1, k2)*self.P_L(magk1, z)*self.P_L(magk2, z) + self.F2(k2, k3)*self.P_L(magk2, z)*self.P_L(magk3, z) + self.F2(k1, k3)*self.P_L(magk1, z)*self.P_L(magk3, z))


