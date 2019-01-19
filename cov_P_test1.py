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

symmetric_factor = 1.0    # mu is symmetric: mu -> -mu

for i in range(len(z_c)):
#for i in range(1):
    z = z_c[i]
    Vol = V[i]
    deltak = 2.0*np.pi/Vol**(1.0/3.0)
    kf = deltak
    kedge = np.arange(0, kmax, deltak)
    kedge = np.hstack((kedge, kmax))

    kbinC = (kedge[0:-1]+kedge[1:])/2.0
    print(len(kbinC))
    sigmap = Eu_sigmap[i]
    b1 = Eu_b1[i]
    ng = Eu_ng[i]
    
    pars.set_matter_power(redshifts=[z], kmax=2.0)
    results = camb.get_results(pars)
    kh, z1, pk = results.get_matter_power_spectrum(minkh=1e-6, maxkh=1.0, npoints=200)
    ss8 = np.array(results.get_sigma8())
    
    def P_L(k, z):
        f = interp1d(kh, pk)
        return f(k)
    
    BPmodel = BP_model(P_L=P_L, cosmo=cosmo)
    f = BPmodel.func_f(z)
    
    f1 = lambda k: P_L(k, z)
    sigmav2 = f**2/(6*np.pi**2)*quad(f1, 1e-6, 1.0)[0]
    print(sigmap, np.sqrt(2*sigmav2))

    def CPP(k, mu):
        return 2*BPmodel.Ptilt(k, mu, sigmap, z, b1, f, ng)**2.0 / (Vol/(2.0*np.pi)**2.0 *k**2.0*deltak*deltamu)

    def FUNCP(k, mu):
        return BPmodel.P(k, mu, sigmap, z, b1, f)

    C_diag_PP = np.empty((len(kbinC)*len(mubinC)))
    Signal_PP = np.empty((len(kbinC)*len(mubinC)))

    ss = 0
    for j in range(len(kbinC)):
        for k in range(len(mubinC)):
            C_diag_PP[ss] = CPP(kbinC[j], mubinC[k])
            Signal_PP[ss] = FUNCP(kbinC[j], mubinC[k])
            ss = ss+1

    Cov = np.diag(C_diag_PP)
    icov = inv(Cov)
    S2N_sqr = symmetric_factor*np.dot(Signal_PP, np.dot(icov, Signal_PP))  # probably my result should be increased by a factor of 2 !!!!!



    plt.figure(1)
    plt.scatter(z, S2N_sqr**0.5)


plt.figure(1)
plt.ylim(0, 320)



plt.show()

