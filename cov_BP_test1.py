import numpy as np
import scipy as sp
from powerspec_EH import *
from astropy.cosmology import FlatLambdaCDM
from numpy.linalg import inv
import matplotlib.pyplot as plt
import camb
from camb import model, initialpower
from scipy.interpolate import interp1d
from scipy.integrate import quad, trapz

Oc0 = 0.2596
Ob0 = 0.0484
h = 0.6781
ns = 0.9677
s8 = 0.8149

Om0 = Oc0 + Ob0

pows = powerspec(Omegab=Ob0, Omegac=Oc0, h=h, ns=ns, sigma8=s8)
cosmo = FlatLambdaCDM(H0=h*100, Om0=Om0, Tcmb0=2.725)


pars = camb.CAMBparams()
pars.set_cosmology(H0=h*100, ombh2=Ob0*h*h, omch2=Oc0*h*h, mnu=0.06, omk=0)
pars.InitPower.set_params(As=2.139e-9, ns=ns, r=0)
pars.set_dark_energy(w=-1.0)


area = 15000.0
areatotal = 41252.96

z_edge = np.linspace(0.65, 2.05, 15)
z_c = (z_edge[1:]+z_edge[0:-1])/2.0

Eu_ng = np.array([2.76, 2.04, 1.53, 1.16, 0.88, 0.68, 0.52, 0.38, 0.26, 0.20, 0.15, 0.11, 0.09, 0.07]) * 1e-3
Eu_b1 = np.array([1.18, 1.22, 1.26, 1.30, 1.34, 1.38, 1.42, 1.46, 1.50, 1.54, 1.58, 1.62, 1.66, 1.70])
Eu_b2 = np.array([-0.76, -0.76, -0.75, -0.74, -0.72, -0.70, -0.68, -0.66, -0.63, -0.60, -0.57, -0.53, -0.49, -0.45])
Eu_bs2 = np.array([-0.10, -0.13, -0.15, -0.17, -0.19, -0.22, -0.24, -0.26, -0.29, -0.31, -0.33, -0.35, -0.38, -0.40])
Eu_sigmap = np.array([4.81, 4.72, 4.62, 4.51, 4.39, 4.27, 4.15, 4.03, 3.92, 3.81, 3.70, 3.61, 3.49, 3.40])
V = np.empty(len(z_c))

for i in range(len(z_edge)-1):
    zl = z_edge[i]
    zh = z_edge[i+1]
    Dl = cosmo.comoving_distance(zl).value
    Dh = cosmo.comoving_distance(zh).value
    V[i] = 4.0*np.pi/3.0 *(Dh**3.0 - Dl**3.0) * area/areatotal

V = V*1e-9 * h**3  #  to be consistent with the paper table 1

V = V*1e9   # in unit of (h^-1 Mpc)^3

kmax = 0.15

mubin = np.linspace(0, 1, 2+1)
mubinC = (mubin[0:-1]+mubin[1:])/2.0
deltamu = mubin[1]- mubin[0]

Np = 4
Na = 2
costh_ed = np.linspace(0, 1.0, Np+1)
phi_ed = np.linspace(np.pi/2.0, 1.5*np.pi, Na+1)

phi_C = (phi_ed[0:-1]+phi_ed[1:])/2.0

costh_C = (costh_ed[0:-1]+costh_ed[1:])/2.0
sinth_C = np.sqrt(1.0-costh_C**2.0)
cosphi_C = np.cos(phi_C)
sinphi_C = np.sin(phi_C)

symmetric_factor = 4.0

#for i in range(len(z_c)):
for i in range(1):
    z = z_c[i]
    Vol = V[i]
    deltak = 2.0*np.pi/Vol**(1.0/3.0) * 1
    kf = deltak
    kedge = np.arange(0, kmax, deltak)
    kedge = np.hstack((kedge, kmax))

    kbinC = (kedge[0:-1]+kedge[1:])/2.0
    print(len(kbinC))
    sigmap = Eu_sigmap[i]
    b1 = Eu_b1[i]
    b2 = Eu_b2[i]
    bs2 = Eu_bs2[i]
    
    ng = Eu_ng[i]
    
    pars.set_matter_power(redshifts=[z], kmax=2.0)
    results = camb.get_results(pars)
    kh, z1, pk = results.get_matter_power_spectrum(minkh=1e-6, maxkh=1.0, npoints=200)
    
    def P_L(k, z):
        f = interp1d(kh, pk)
        return f(k)
    
    BPmodel = BP_model(P_L=P_L, cosmo=cosmo)
    
    f = BPmodel.func_f(z)
    
    def NB(k1, k2, k3):
        return Vol**2/(8.0*np.pi**4)*k1*k2*k3*(deltak)**3 * 1.0 /(Np*Na)
    
    def NP(k):
        return  (Vol/(2.0*np.pi)**2.0 *k**2.0*deltak*deltamu)
    

    def CBB(k1, mu1, k2, mu2, k3, mu3):
        if np.digitize(k1, kedge) == np.digitize(k2, kedge) and np.digitize(k1, kedge) == np.digitize(k3, kedge):
            sB = 6
            #print("666")
        elif np.digitize(k1, kedge) == np.digitize(k2, kedge) or np.digitize(k2, kedge) == np.digitize(k3, kedge) or np.digitize(k1, kedge) == np.digitize(k3, kedge):
            sB = 2
            #print("222")
        else:
            sB = 1
            #print("111")
        
        return sB*Vol*BPmodel.Ptilt(k1, mu1, sigmap, z, b1, f, ng)*BPmodel.Ptilt(k2, mu2, sigmap, z, b1, f, ng)*BPmodel.Ptilt(k3, mu3, sigmap, z, b1, f, ng)/ NB(k1, k2, k3)

    def FUNCB(k1, k2, k3):
        return BPmodel.B(k1, k2, k3, z, b1, b2, bs2, f, sigmap)
    
    C_diag_BB = []
    Signal_BB = []
    ss = 0

    ss2 = 0
    
    ss3 = 0
    
    kindex = np.atleast_2d(np.array([-1, -1, -1]))
    for bk1 in range(len(kbinC)):
        k1_mag = kbinC[bk1]
        for bk2 in range(len(kbinC)):
            k2_mag = kbinC[bk2]
            for bk3 in range(len(kbinC)):
                k3_mag = kbinC[bk3]
                if k1_mag<=(k2_mag+k3_mag) and k1_mag>=abs(k2_mag-k3_mag):
                    ss2 = ss2+1
                    pass
                else:
                    continue
                ksort = np.sort([k1_mag, k2_mag, k3_mag])
                kl_mag = ksort[-1]
                km_mag = ksort[-2]
                ks_mag = ksort[-3]
                
                kindex_test = np.sort(np.array([np.digitize(k1_mag, kedge), np.digitize(k2_mag, kedge), np.digitize(k3_mag, kedge)]))
                    
                if list(kindex_test) in kindex.tolist():
                    continue
                else:
                    ss3 = ss3+1
                    kindex = np.vstack((kindex, kindex_test))

                cosxi12 = (kl_mag**2 + km_mag**2 - ks_mag**2)/(2.0*kl_mag*km_mag)
                xi12 = np.pi-np.arccos(cosxi12)
                sinxi12 = np.sin(xi12)

                for bcosth in range(len(costh_C)):
                    costhe = costh_C[bcosth]
                    sinthe = np.sqrt(1.0-costhe**2.0)
                    for bphi in range(len(phi_C)):
                        phi = phi_C[bphi]
                        cosphi = np.cos(phi)
                        sinphi = np.sin(phi)
                                
                        cosw = sinthe*cosphi
                        sinw = np.sqrt(1.0-sinthe**2*cosphi**2)
                        cosx = -sinthe*sinphi/np.sqrt(1.0-sinthe**2*cosphi**2)
                        sinx = costhe/np.sqrt(1.0-sinthe**2*cosphi**2)
                        
                        x = np.arccos(cosx)
                        k1 = kl_mag*np.array([cosx, -cosw*sinx, sinw*sinx])
                        k2 = km_mag*np.array([np.cos(x-xi12), -cosw*np.sin(x-xi12), sinw*np.sin(x-xi12)])
                        k3 = np.array([0.0, 0.0, 0.0])-(k1+k2)
                        #k3_mag = np.sqrt(k3.dot(k3))
                        k3_mag = ks_mag
                        
                        mu_1 = k1.dot(BPmodel.s)/kl_mag
                        mu_2 = k2.dot(BPmodel.s)/km_mag
                        mu_3 = k3.dot(BPmodel.s)/ks_mag

                        #print(kl_mag, km_mag, ks_mag)
                        C_diag_BB.append(CBB(kl_mag, mu_1, km_mag, mu_2, ks_mag, mu_3))
                        Signal_BB.append(FUNCB(k1, k2, k3))
                                
                        ss = ss+1

    C_diag_BB = np.array(C_diag_BB).flatten()
    Signal_BB = np.array(Signal_BB).flatten()
    icov = np.diag(1./C_diag_BB)
    S2N_sqr = symmetric_factor*np.dot(Signal_BB, np.dot(icov, Signal_BB))

    plt.figure(1)
    plt.scatter(z, S2N_sqr**0.5)


plt.figure(1)
plt.ylim(0, 320)

plt.show()

