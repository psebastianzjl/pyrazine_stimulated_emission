import os
import numpy as np
import time
import sys
from tr_spectrum import plot_signal

#num_trajectories = sys.argv[1]
w_pu = 5.2 
resolution = 200
W_PR = np.linspace(2.5,
                   6.0,
                   resolution)
TAU = [5]
TRAJ = [50]#, 99, 199] # trajectory number
TSTEP = 401 # time step of dynamics
SE_TRIGGER = 'yes' # get stimulated emission

def fs2aut(fs): #femtosecond to atomic unit of time
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

def signal(eV, Ha, Ha_0, f, fs, delta=False):
    laser = eV2Ha(eV)
    tau = fs2aut(fs)
    if f < 0:
        #f=abs(f)
        f=0
    w_in = Ha - Ha_0
    if w_in == 0.0:
        print('Warning:', w_in)
        time.sleep(100)
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

def signal_sum_deV(eV, deV, f, fs):
    laser = eV2Ha(eV)
    tau = fs2aut(fs)
    S = 0
    for (i, val) in enumerate(deV):
        w_in = -1. * eV2Ha(val)
        dw = laser - w_in
        if w_in != 0:
            v2 = f[i] / 2. * 3 / w_in
            S += envelope(dw, tau) ** 2. * v2
    return S 


print(str(w_pu) + 'eV as pump pulse')
eaEmax = 100
eaEmin = 0
list_intensities = []
for tau in TAU:
    for TRAJNO in TRAJ:
        gsCount = np.zeros([TSTEP, 2])
        exCount = np.zeros([TSTEP, 2])
        eaCount = np.zeros([TSTEP, 2])
        exTCount = gsTCount = eaTCount = 0
        for itraj in range(TRAJNO):
            trajNo = itraj + 1 
            exfile = './TRAJ' + str(trajNo) + '/Results/chr_energy.dat' #excited state dynamics output
            exfile_osc = './TRAJ' + str(trajNo) + '/Results/oscill.dat' 
            if SE_TRIGGER == 'yes':
                exE = np.zeros([TSTEP, 7]) #t, S0, S1, S2, S3, S4, current
                exf = np.zeros([TSTEP, 5]) #t, f1, f2, f3, f4
                for t in range(TSTEP):
                    exCount[t, 0] = exE[t, 0] = exf[t, 0] = t * .5
                if os.path.isfile(exfile):
                    exTCount += 1
                    print('Excited State Dynamics:', 'tau =', tau, ',total TRAJ =', TRAJNO, ',traj =', trajNo, ',exTCount ->', exTCount)
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
                    Dee = signal(w_pu, exE[0, int(exE[0, -1])], exE[0, 1], exf[0, int(exE[0, -1])-1], tau) 
                    Dee_delta = signal(w_pu, exE[0, int(exE[0, -1])], exE[0, 1], exf[0, int(exE[0, -1])-1], tau, delta=True) 
                #window
                for (iw_pr, w_pr) in enumerate(W_PR):
                    SEi = np.zeros([TSTEP, 3]) #t, w_pr, SE
                    SEi_delta = np.zeros([TSTEP, 3]) #t, w_pr, SE
                    for t in range(TSTEP):
                        SEi[t, 0] = t * .5
                        SEi[t, 1] = w_pr
                        SEi_delta[t, 0] = t * .5
                        SEi_delta[t, 1] = w_pr
                        if t < tstep:
                            if int(exE[t, -1]) != 0:
                                Wee = signal(w_pr, exE[t, int(exE[t, -1])], exE[t, 1], exf[t, int(exE[t, -1])-1], tau) 
                            else:
                                Wee = 0
                        else:
                            Wee = 0
                        SEi[t, 2] = Dee * Wee
                        SEi_delta[t, 2] = Dee_delta * Wee
                    if iw_pr == 0:
                        SE = SEi.copy()
                        SE_delta = SEi_delta.copy()
                    else:
                        SE = np.vstack((SE, SEi))
                        SE_delta = np.vstack((SE_delta, SEi_delta))
                if trajNo == 1:
                    SEsum = SE.copy()
                    SEsum_delta = SE_delta.copy()
                else:
                    SEsum[:, 2] += SE[:, 2]
                    SEsum_delta[:, 2] += SE_delta[:, 2]
        #normalization
        #t, E, SE/n, GSB/n, EA/n, S/n
        #0, 1,    2,     3,    4,   5
        if SE_TRIGGER == 'yes':
            for i in range(len(SEsum[:,2])):
                SEsum[i, 2] /= exCount[i%TSTEP, -1]
                SEsum_delta[i, 2] /= exCount[i%TSTEP, -1]
            S = SEsum.copy()
            S_delta = SEsum_delta.copy()
            list_intensities += [SEsum[:,2]/max(SEsum[:,2])]            
            plot_signal(exCount[:,0], W_PR,SEsum[:,2]/max(SEsum[:,2]),'tr_se_tau' + str(tau) + '_traj' + str(TRAJNO) + '.png')
        S = np.hstack((S, S[:, 2:3] + S[:, 3:4] + S[:, 4:5]))
        S_delta = np.hstack((S_delta, S_delta[:, 2:3] + S_delta[:, 3:4] + S_delta[:, 4:5]))
        with open('int_tau' + str(tau) + '_traj' + str(TRAJNO) + '.dat', 'w') as output:
            np.savetxt(output, S)
        with open('int_delta_tau' + str(tau) + '_traj' + str(TRAJNO) + '.dat', 'w') as output:
            np.savetxt(output, S_delta)

#print('list intensities:', list_intensities)
### check convegence ###
#for i in range(len(TRAJ)-1):
#    ratio = abs(list_intensities[i+1]-list_intensities[i])
#    print('num of traj:', TRAJ[i+1])
#    print('MAE:', sum(ratio)/len(ratio)) #MAE
    #print('ratio min:', min(ratio))

