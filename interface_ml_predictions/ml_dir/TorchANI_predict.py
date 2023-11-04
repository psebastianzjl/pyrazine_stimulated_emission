import torch
import torchani
import os
import math
import torch.utils.tensorboard
import tqdm
import numpy as np
import h5py
from torchani.units import hartree2kcalmol
import sys
from utils import number2element, element2number
from functorch import combine_state_for_ensemble
from functorch import vmap
import copy

sys.path.append("../../")



def predict(args):    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = torch.load(args.MLmodelIn, map_location=torch.device(device))# for i in range(args.nstates)]
    energyOffset = 0

    Rcr = models[0]['args']['Rcr'] #for i in range(args.nstates)]
    Rca = models[0]['args']['Rca'] #for i in range(args.nstates)]
    EtaR = models[0]['args']['EtaR'].to(device)# for i in range(args.nstates)]
    ShfR = models[0]['args']['ShfR'].to(device)# for i in range(args.nstates)]
    Zeta = models[0]['args']['Zeta'].to(device)# for i in range(args.nstates)]
    ShfZ = models[0]['args']['ShfZ'].to(device)# for i in range(args.nstates)]
    EtaA = models[0]['args']['EtaA'].to(device)# for i in range(args.nstates)]
    ShfA = models[0]['args']['ShfA'].to(device)# for i in range(args.nstates)]

    sp_z = {'X': 0, 'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Uut': 113, 'Fl': 114, 'Uup': 115, 'Lv': 116, 'Uus': 117, 'Uuo': 118}
    z_sp={v:k for k,v in sp_z.items()}
    def switchSPandZ(d):
        if d[0] in sp_z.keys():
            out=[sp_z[i] for i in d]
        else:
            out=[z_sp[int(i)] for i in d]
        return out

    try: species_order=best_model[0]['args']['species_order']
    except: species_order = args.atype
    num_species = len(species_order)
    species_order_cap = [x.upper() for x in species_order]
    self_energies_train=[models[k]['args']['self_energies_train'] for k in range(args.nstates)]     

    aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species) ### ORIGINAL  ###
    aev_dim = aev_computer.aev_length
    networkdic = models[0]['network']

    batch_size = args.ani.batch_size
    if args.setname: args.setname='_'+args.setname
    testfile=h5py.File('ANI'+args.setname+'.h5','r')
    test=testfile.get('dataset')
    try: test = torchani.data.load('ANI'+args.setname+'.h5').species_to_indices(species_order).collate(batch_size).cache()
    except: test = torchani.data.load('ANI'+args.setname+'.h5').species_to_indices(switchSPandZ(species_order_cap)).collate(batch_size).cache()
    
    nn = torchani.ANIModel([networkdic[specie] for specie in species_order])

    models_temp = []

    for i in range(args.nstates):
        model_temp = torchani.nn.Sequential(aev_computer, nn).to(device) 
        model_temp.load_state_dict(models[i]['state_dict'])
        model_temp.eval()
        models_temp.append(copy.deepcopy(model_temp))
    fmodel, params, buffers = combine_state_for_ensemble(models_temp)
    [p.requires_grad_() for p in params]

    def getOffsets(self_energies, species):
        offsets=[]
        for sp in species:
            offset=0
            for idx in sp:
                if idx in range(len(species_order)):
                    try: offset+=self_energies[idx]
                    except: pass
            offsets.append(offset)
        return torch.Tensor(offsets).to(device)

    for properties in test:
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
        _, predicted_energies = vmap(fmodel, in_dims=(0, 0, None))(params,buffers,(species, coordinates))
        if args.current_state >= 0:
            predicted_forces = -torch.autograd.grad(predicted_energies[args.current_state].sum(), coordinates, create_graph=True, retain_graph=True)[0] 
            predicted_forces = np.array(predicted_forces.cpu().detach().numpy()).astype(np.float64) 

        predicted_energies = np.array(predicted_energies.cpu().detach().numpy()) + [getOffsets(self_energies_train[i],species).cpu().detach().numpy().astype(np.float64) for i in range(args.nstates)]
            


    import time
    time.sleep(0.0000001) # Sometimes program hangs without no reason, adding short sleep time helps to exit this module without a problem / P.O.D., 2021-02-17
    if args.current_state >= 0:
        return predicted_energies.transpose(1,0), -1*predicted_forces
    else: return predicted_energies.transpose(1,0)
        

