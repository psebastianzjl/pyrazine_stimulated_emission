import os, sys
import numpy as np
from twod_tr_spectrum import plot_signal

if len(sys.argv) <= 1:
    print("enter Tw as 1st argument")
    exit()
tw = int(sys.argv[1]) 

NIU = [0.1]
w_pu = 5.2 
w_pr = 5.2
resolution = 400
W_T = np.linspace(1.5,
                   7.0,
                   resolution)
TAU = [0.1]#, 5]
TRAJ = [50] # trajectory number
TSTEP = 401 # time step of dynamics
SE_TRIGGER = 'yes' # get stimulated emission

def fs2au(fs): #femtosecond to atomic unit of time
    aut = fs * 41.341374575751
    return aut

def eV2Ha(eV):
    Ha = eV / 27.211386245
    return Ha

def Ha2eV(Ha):
    eV = Ha * 27.211386245
    return eV

def envelope(w, tau):
    E = np.exp(-(w * tau) ** 2 / 4.) * tau
    return E

def signal(eV, Ha, Ha_0, f, fs, delta=False): #Doorway-Function-wise
    laser = eV2Ha(eV)
    tau = fs2aut(fs)
    if f<0:
        f=0
    w_in = Ha - Ha_0
    dw = laser - w_in
    v2 = f / 2. * 3 / w_in
    if not delta:
        return envelope(dw, tau) ** 2. * v2
    else:
        return 1. * v2

def signal_sum(eV, Ha, Ha_0, f, fs, delta=False):
    laser = eV2Ha(eV)
    tau = fs2aut(fs)
    S = 0
    for (i, val) in enumerate(Ha):
        w_in = val - Ha_0
        dw = laser - w_in
        v2 = f[i] / 2. * 3 / w_in
        if not delta:
            S += envelope(dw, tau) ** 2. * v2
        else:
            S += 1. * v2
    return S 

def dissignal(w_in, Ha, Ha_0, f, taufs, niueV, w_preV): # input w=eV and tau=fs
    w_t = eV2Ha(w_in)
    w_pr = eV2Ha(w_preV)
    niu = eV2Ha(niueV)
    tau = fs2aut(taufs)
    w_ge = Ha - Ha_0
    dw = w_t - w_ge
    v2 = f / 2. * 3 / w_ge
    return envelope(w_t - w_pr, tau) ** 2. * v2 * niu / (niu ** 2 + dw ** 2)

def dissignal_sum(w_in, Ha, Ha_0, f, taufs, niueV, w_preV): # input w=eV and tau=fs
    w_t = eV2Ha(w_in)
    w_pr = eV2Ha(w_preV)
    niu = eV2Ha(niueV)
    tau = fs2aut(taufs)
    S = 0
    for (i, val) in enumerate(Ha):
        w_ge = val - Ha_0
        dw = w_t - w_ge
        v2 = f[i] / 2. * 3 / w_ge
        S += envelope(w_t - w_pr, tau) ** 2. * v2 * niu  / (niu ** 2 + dw ** 2)
    return S

def dissignal_sum_deV(w_in, deV, f, taufs, niueV, w_preV):
    w_t = eV2Ha(w_in)
    w_pr = eV2Ha(w_preV)
    niu = eV2Ha(niueV)
    tau = fs2aut(taufs)
    S = 0
    for (i, val) in enumerate(deV):
        w_ge = -1. * eV2Ha(val) #given in negative
        dw = w_t - w_ge
        if w_ge != 0:
            v2 = f[i] / 2. * 3 / w_ge
            S += envelope(w_t - w_pr, tau) ** 2. * v2 * niu / (niu ** 2 + dw ** 2)
    return S 

def dissignal_R(w_in, Ha, Ha_0, f, taufs, niueV, w_preV): # input w=eV and tau=fs
    w_t = eV2Ha(w_in)       #parameter frequency
    w_pr = eV2Ha(w_preV)    #fixed frequency
    niu = eV2Ha(niueV)      #broadening
    tau = fs2au(taufs)      #Tw
    w_ge = Ha - Ha_0        #Ueg
    dw = w_t - w_ge         #dispersion
    v2 = f / 2. * 3 / w_ge
    return envelope(w_t - w_pr, tau) ** 2. * v2 / complex(niu, dw)

def dissignal_NR(w_in, Ha, Ha_0, f, taufs, niueV, w_preV): # input w=eV and tau=fs
    w_t = eV2Ha(w_in)
    w_pr = eV2Ha(w_preV)
    niu = eV2Ha(niueV)
    tau = fs2au(taufs)
    w_ge = Ha - Ha_0
    dw = w_t - w_ge
    v2 = f / 2. * 3 / w_ge
    return envelope(w_t - w_pr, tau) ** 2. * v2 / complex(niu, -dw)

def dissignal_sum_delta(w_in, Ha, Ha_0, f, taufs, niueV, w_preV): # input w=eV and tau=fs
    w_t = eV2Ha(w_in)
    w_pr = eV2Ha(w_preV)
    niu = eV2Ha(niueV)
    tau = fs2au(taufs)
    S = 0
    for (i, val) in enumerate(Ha):
        w_ge = val - Ha_0
        dw = w_t - w_ge
        if w_ge != 0:
            v2 = f[i] / 2. * 3 / w_ge
            S += 1 * v2 / complex(niu, dw)
    return S

def dissignal_sum_R(w_in, Ha, Ha_0, f, taufs, niueV, w_preV): # input w=eV and tau=fs
    w_t = eV2Ha(w_in)
    w_pr = eV2Ha(w_preV)
    niu = eV2Ha(niueV)
    tau = fs2au(taufs)
    S = 0
    for (i, val) in enumerate(Ha):
        w_ge = val - Ha_0
        dw = w_t - w_ge
        if w_ge != 0:
            v2 = f[i] / 2. * 3 / w_ge
            S += envelope(w_t - w_pr, tau) ** 2. * v2 / complex(niu, dw)
    return S

def dissignal_sum_NR(w_in, Ha, Ha_0, f, taufs, niueV, w_preV): # input w=eV and tau=fs
    w_t = eV2Ha(w_in)
    w_pr = eV2Ha(w_preV)
    niu = eV2Ha(niueV)
    tau = fs2au(taufs)
    S = 0
    for (i, val) in enumerate(Ha):
        w_ge = val - Ha_0
        dw = w_t - w_ge
        if w_ge != 0:
            v2 = f[i] / 2. * 3 / w_ge
            S += envelope(w_t - w_pr, tau) ** 2. * v2 / complex(niu, -dw)
    return S


def dissignal_sum_NR_ESA(w_in, dEeV, f, taufs, niueV, w_preV): # input w=eV and tau=fs
    w_t = eV2Ha(w_in)
    w_pr = eV2Ha(w_preV)
    niu = eV2Ha(niueV)
    tau = fs2au(taufs)
    S = 0
    for (i, val) in enumerate(dEeV):
        w_ge = -1. * eV2Ha(val)
        dw = w_t - w_ge
        if w_ge != 0:
            v2 = f[i] / 2. * 3 / w_ge
            S += envelope(w_t - w_pr, tau) ** 2. * v2 / complex(niu, -dw)
    return S


print(str(w_pu) + 'eV as pump frequency')
print(str(w_pr) + 'eV as probe frequency')
eaEmax = 100
eaEmin = 0
print(str(tw) + ' fs as Tw')
tws = tw * 2

GSC = [600]#1, 10, 25, 50, 100, 150, 200, 300, 1000]
list_intensities = []

for gsC in GSC:
  for niu in NIU:
    for tau in TAU:
        for TRAJNO in TRAJ:
            gsCount, exCount, eaCount = [ np.zeros([TSTEP, 2], dtype=int) for _ in range(3)]
            exTCount = gsTCount = eaTCount = 0
            for itraj in range(TRAJNO):
                trajNo = itraj + 1 
                exfile = './TRAJ' + str(trajNo) + '/Results/chr_energy.dat' #excited state dynamics output
                exfile_osc = './TRAJ' + str(trajNo) + '/Results/oscill.dat' #excited state dynamics output
                gscheck = False
                excheck = False
                eacheck = False
                if exCount[tws, 1] >= gsC:
                    break
                ############################
                #  read excited dynamics   #
                ############################
                if SE_TRIGGER == 'yes':
                    if os.path.isfile(exfile):
                        #print('Bookmark')
                        #print(exfile)
                        excheck = True
                        exTCount += 1
                        exE = np.zeros([TSTEP, 7]) #t, S0, S1, S2, S3, S4, current
                        exf = np.zeros([TSTEP, 5]) #t, f1, f2, f3, f4
                        for t in range(TSTEP):
                            exCount[t, 0] = exE[t, 0] = exf[t, 0] = t * .5
                        with open(exfile,'r') as f:
                            tstep = 0
                            for line in f:
                                exCount[tstep, 1] += 1
                                exE[tstep, -1] =  line.split()[1]
                                exE[tstep, 1:6] = line.split()[-5:]
                                tstep += 1
                                if tstep == TSTEP:
                                   break
                        with open(exfile_osc,'r') as f:
                            tstep = 0
                            for line in f:
                                exf[tstep, 1:5] = line.split()
                                tstep += 1
                                if tstep == TSTEP:
                                   break
                ############################
                #  calculating DW          #
                ############################
                for (iw_t, w_t) in enumerate(W_T):
                    GSB_R, GSB_NR = [np.zeros([resolution, 3]) for _ in range(2)] #w_tau, w_t, Intensity, Mean Intensity
                    SE_R, SE_NR = [np.zeros([resolution, 3]) for _ in range(2)] #w_tau, w_t, Intensity
                    EA_NR, EA_R = [np.zeros([resolution, 3]) for _ in range(2)] #w_tau, w_t, Intensity
                    for (iw_tau, w_tau) in enumerate(W_T):
                        if excheck:
                            SE_R[iw_tau, 0] = SE_NR[iw_tau, 0] = w_tau
                            Dee_R = dissignal_R(w_tau, exE[0, int(exE[0, -1])], exE[0, 1], exf[0, int(exE[0, -1]) - 1], tau, niu, w_pu)
                            Dee_NR = dissignal_NR(w_tau, exE[0, int(exE[0, -1])], exE[0, 1], exf[0, int(exE[0, -1]) - 1], tau, niu, w_pu)
                            if int(exE[tws, -1]) != 0:
                                Wee = dissignal_NR(w_t, exE[tws, int(exE[tws, -1])], exE[tws, 1], exf[tws, int(exE[tws, -1]) - 1], tau, niu, w_pr)
                            else:
                                Wee = 0
                            SE_R[iw_tau, 2] = np.real(Dee_R * Wee)
                            SE_NR[iw_tau, 2] = np.real(Dee_NR * Wee)
                    if excheck:
                        SE_R[:, 1] = SE_NR[:, 1] = w_t
                        if iw_t == 0:
                            SE_R_full = SE_R.copy()
                            SE_NR_full = SE_NR.copy()
                        else:
                            SE_R_full = np.vstack((SE_R_full, SE_R))
                            SE_NR_full = np.vstack((SE_NR_full, SE_NR))
                #Accumulation of trajectories
                if excheck:
                    if trajNo == 1:
                        SE_R_full_sum = SE_R_full.copy()
                        SE_NR_full_sum = SE_NR_full.copy()
                    else:
                        SE_R_full_sum[:, -1] += SE_R_full.copy()[:, -1]
                        SE_NR_full_sum[:, -1] += SE_NR_full.copy()[:, -1]
            ###############
            #normalization#
            ###############
            #w_tau, w_t, SE/n, GSB/n, EA/n, S/n
            #    0,   1,    2,     3,    4,   5
            print('Trajectory Count: gs ' + str(gsTCount) + ' ; ex ' + str(exTCount) + ' ; ea ' + str(eaTCount))
            print('Valid points count: at ' + str(gsCount[tws, 0]) + ' fs, gs ' + str(gsCount[tws, 1]) + ' ; ex ' + str(exCount[tws, 1]) + ' ; ea ' + str(eaCount[tws, 1]))
            if SE_TRIGGER == 'yes':
                SE_R = SE_R_full_sum.copy()
                SE_NR = SE_NR_full_sum.copy()
                SE_R[:, -1] /= exCount[tws, 1]
                SE_NR[:, -1] /= exCount[tws, 1]
                #list_intensities += [SE_NR[:,2]/max(SE_NR[:,2])]            
                plot_signal(W_T,SE_NR[:,2]/max(SE_NR[:,2]),'2d_tr_se_tau' + str(tau) + '_' + str(tw) + 'fs' + '_traj' + str(TRAJNO) + '.png')
                S_R = SE_R.copy()
                S_NR = SE_NR.copy()
            S_R = np.hstack((S_R, S_R[:, 2:3] + S_R[:, 3:4] + S_R[:, 4:5]))
            S_NR = np.hstack((S_NR, S_NR[:, 2:3] + S_NR[:, 3:4] + S_NR[:, 4:5]))
            S = S_R.copy()
            S[:, 2:6] += S_NR[:, 2:6]
            Smax = np.maximum(abs(S_R[:, 2:6]).max(), abs(S_NR[:, 2:6]).max())
            print(Smax)
            S_R[:, 2:6] /= Smax 
            with open('new2D_adc_R_tw' + str(tw) + '_tau' + str(tau) + '_pr' + str(w_pr) + '_niu' + str(niu) + '_traj' + str(gsC) + '.dat', 'w') as output:
                np.savetxt(output, S_R)
            S_NR[:, 2:6] /= Smax
            with open('new2D_adc_NR_tw' + str(tw) + '_tau' + str(tau) + '_pr' + str(w_pr) + '_niu' + str(niu) + '_traj' + str(gsC) + '.dat', 'w') as output:
                np.savetxt(output, S_NR)
            S[:, 2:6] /= Smax
            with open('new2D_adc_S_tw' + str(tw) + '_tau' + str(tau) + '_pr' + str(w_pr) + '_niu' + str(niu) + '_traj' + str(gsC) + '.dat', 'w') as output:
                np.savetxt(output, S)



#print('list intensities:', list_intensities)
### check convegence ###
#for i in range(len(TRAJ)-1):
#    ratio = abs(list_intensities[i+1]-list_intensities[i])
#    print('num of traj:', TRAJ[i+1])
   # print('MAE:', sum(ratio)/len(ratio)) #MAE
