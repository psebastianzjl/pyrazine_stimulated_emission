#!/usr/bin/python3
'''
  !---------------------------------------------------------------------------! 
  ! Interface_TorchANI: Interface between TorchANI and MLatom 2               ! 
  ! Implementations by: Fuchun Ge and Max Pinheiro Jr                         !
  ! Edited by Sebastian V. Pios, please refer to the official MLatom homepage !
  ! for the latest stable version.                                            !
  !---------------------------------------------------------------------------! 
'''
import numpy as np
import os, sys, subprocess, time, shutil, re, math, random
import stopper
import torch
from args_class import ArgsBase

filedir = os.path.dirname(__file__)

class Args(ArgsBase):
    def __init__(self):
        super().__init__()
        self.add_default_dict_args([
            'useMLmodel'
            ],
            bool
        )
        self.add_default_dict_args([
            'xyzfile', 'yfile', 'ygradxyzfile','itrainin','itestin','isubtrainin','ivalidatein','mlmodelin','setname','nstates','current_state'
            ],
            ""
        )        
        self.add_dict_args({
            'mlmodeltype': 'ANI',
            'mlmodelout': "ANIbestmodel.pt",
            'sampling': "random",
            'yestfile': "enest.dat",
            'ygradxyzestfile': "gradest.dat",
            'hessianestfile': "",
            'lcNtrains': [],
            'atype': [],
            'nthreads': None,
            'minimizeError':'RMSE'
        })
        self.parse_input_content([
            'ani.restart=0',
            'ani.reset_optim_state=1',    
            'ani.learning_rate=0.001',
            'ani.batch_size=8',
            'ani.max_epochs=10000000',
            'ani.early_stopping_learning_rate=0.00001',
            'ani.force_coefficient=0.1',
            'ani.batch_sizes=0',
            'ani.patience=100',
            'ani.lrfactor=0.5',
            'ani.Rcr=5.2',
            'ani.Rca=3.5',
            'ani.EtaR=16',
            'ani.ShfR=0.9,1.16875,1.4375,1.70625,1.975,2.24375,2.5125,2.78125,3.05,3.31875,3.5875,3.85625,4.125,4.39375,4.6625,4.93125',
            'ani.Zeta=32',
            'ani.ShfZ=0.19634954,0.58904862,0.9817477,1.3744468,1.7671459,2.1598449,2.552544,2.9452431',
            'ani.EtaA=8',
            'ani.ShfA=0.90,1.55,2.20,2.85',
            'ani.Neurons=160.128.96',
            'ani.AFs=CELU,CELU,CELU',
            'ani.Neuron_l1=160',
            'ani.Neuron_l2=128',
            'ani.Neuron_l3=96',
            'ani.AF1=CELU',
            'ani.AF2=CELU',
            'ani.AF3=CELU',
            'ani.transfer_learning_fixed_layer=na'
            ])

    def parse(self, argsraw):
        self.parse_input_content(argsraw)
        self.argProcess()

    def argProcess(self):
        if self.ani.transfer_learning_fixed_layer=='na':
            self.ani.transfer_learning_fixed_layer=None
        else:
            self.ani.transfer_learning_fixed_layer=[str(int(l)*2) for l in str(self.ani.transfer_learning_fixed_layer).split(',')]
        for k, v in self.ani.data.items():
            if k.lower() in ['shfr','etar','zeta','shfz','shfz','etaa','shfa']:
                self.ani.data[k] = torch.tensor([float(i) for i in str(v).split(',')])
            elif k.lower() in ['neuron_l1','neuron_l2','neuron_l3']:
                self.ani.data[k] =[int(i) for i in str(v).split(',')]
            elif k.lower() in ['af1','af2','af3']:
                self.ani.data[k] = v.split(',')


        with open(self.xyzfile,'r') as f:
            for line in f:
                natom = int(line)
                f.readline()
                for i in range(natom):
                    sp=f.readline().split()[0]
                    if sp not in self.atype:
                        self.atype.append(sp)
        
        if not self.mlmodelin:
            self.mlmodelin = self.mlmodelout
        
        species_order = sorted(set(self.atype), key=lambda x: self.atype.index(x))
        if len(self.ani.Neuron_l1) == 1: self.ani.Neuron_l1 = self.ani.Neuron_l1*len(species_order)
        if len(self.ani.Neuron_l2) == 1: self.ani.Neuron_l2 = self.ani.Neuron_l2*len(species_order)
        if len(self.ani.Neuron_l3) == 1: self.ani.Neuron_l3 = self.ani.Neuron_l3*len(species_order)
        # if self.lcyonly: self.ygradxyzfile=''
        
        self.ani.Neurons=[[int(j) for j in i.split('.')] for i in self.ani.Neurons.split(',')]
        if len(self.ani.Neurons)==1: self.ani.Neurons=[self.ani.Neurons[0].copy() for _ in range(len(species_order))]
        self.ani.AFs=[[j for j in i.split('.')] for i in self.ani.AFs.split(',')]
        if len(self.ani.Afs)==1: self.ani.AFs=self.ani.AFs*len(species_order)
        
        if not self.ani.batch_size:
            if self.ntrain: self.ani.batch_size = int(math.sqrt(self.ntrain))
            else: self.ani.batch_size=8
            

class ANICls(object):
    dataConverted = False
    loaded=False
    coreset=False
    
    @classmethod
    def setCore(cls, n):
        if not cls.coreset:
            if n:
                torch.set_num_threads(2)  ###used to be n here
                torch.set_num_interop_threads(2)  ### and here
            cls.coreset=True
    @classmethod
    def load(cls):
        if not cls.loaded:
            
            available = True
            
            #try: 
            import h5py
            import torch
            import TorchANI_predict          
            #except: 
            #    available = False
            

            globals()['available'] = available
            globals()['h5py'] = h5py
            globals()['torch'] = torch
            globals()['TorchANI_predict'] = TorchANI_predict

            cls.batch_size = 8
            cls.max_epochs = 10000000
            cls.learning_rate = 0.001
            cls.early_stopping_learning_rate = 1.0E-5
            cls.force_coefficient = 0.1
            cls.batch_sizes = []
            cls.patience = 100
            cls.lrfactor = 0.5
            cls.Rcr = 5.2000e+00
            cls.Rca = 3.5000e+00
            cls.EtaR = torch.tensor([1.6000000e+01])
            cls.ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00])
            cls.Zeta = torch.tensor([3.2000000e+01])
            cls.ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00])
            cls.EtaA = torch.tensor([8.0000000e+00])
            cls.ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00])
            cls.Neuron_l1 = [160]
            cls.Neuron_l2 = [128]
            cls.Neuron_l3 = [96]
            cls.AF1 = ['CELU']
            cls.AF2 = ['CELU']
            cls.AF3 = ['CELU']

            cls.loaded=True

    def __init__(self, argsANI = sys.argv[1:]):
        print(' ___________________________________________________________\n\n%s' % __doc__)
    
    @classmethod
    def convertdata(cls, argsANI, subdatasets):
        cls.load()
        spdict={'0': 'X', '1': 'H', '2': 'He', '3': 'Li', '4': 'Be', '5': 'B', '6': 'C', '7': 'N', '8': 'O', '9': 'F', '10': 'Ne', '11': 'Na', '12': 'Mg', '13': 'Al', '14': 'Si', '15': 'P', '16': 'S', '17': 'Cl', '18': 'Ar', '19': 'K', '20': 'Ca', '21': 'Sc', '22': 'Ti', '23': 'V', '24': 'Cr', '25': 'Mn', '26': 'Fe', '27': 'Co', '28': 'Ni', '29': 'Cu', '30': 'Zn', '31': 'Ga', '32': 'Ge', '33': 'As', '34': 'Se', '35': 'Br', '36': 'Kr', '37': 'Rb', '38': 'Sr', '39': 'Y', '40': 'Zr', '41': 'Nb', '42': 'Mo', '43': 'Tc', '44': 'Ru', '45': 'Rh', '46': 'Pd', '47': 'Ag', '48': 'Cd', '49': 'In', '50': 'Sn', '51': 'Sb', '52': 'Te', '53': 'I', '54': 'Xe', '55': 'Cs', '56': 'Ba', '57': 'La', '58': 'Ce', '59': 'Pr', '60': 'Nd', '61': 'Pm', '62': 'Sm', '63': 'Eu', '64': 'Gd', '65': 'Tb', '66': 'Dy', '67': 'Ho', '68': 'Er', '69': 'Tm', '70': 'Yb', '71': 'Lu', '72': 'Hf', '73': 'Ta', '74': 'W', '75': 'Re', '76': 'Os', '77': 'Ir', '78': 'Pt', '79': 'Au', '80': 'Hg', '81': 'Tl', '82': 'Pb', '83': 'Bi', '84': 'Po', '85': 'At', '86': 'Rn', '87': 'Fr', '88': 'Ra', '89': 'Ac', '90': 'Th', '91': 'Pa', '92': 'U', '93': 'Np', '94': 'Pu', '95': 'Am', '96': 'Cm', '97': 'Bk', '98': 'Cf', '99': 'Es', '100': 'Fm', '101': 'Md', '102': 'No', '103': 'Lr', '104': 'Rf', '105': 'Db', '106': 'Sg', '107': 'Bh', '108': 'Hs', '109': 'Mt', '110': 'Ds', '111': 'Rg', '112': 'Cn', '113': 'Uut', '114': 'Fl', '115': 'Uup', '116': 'Lv', '117': 'Uus', '118': 'Uuo','X': 'X', 'H': 'H', 'He': 'He', 'Li': 'Li', 'Be': 'Be', 'B': 'B', 'C': 'C', 'N': 'N', 'O': 'O', 'F': 'F', 'Ne': 'Ne', 'Na': 'Na', 'Mg': 'Mg', 'Al': 'Al', 'Si': 'Si', 'P': 'P', 'S': 'S', 'Cl': 'Cl', 'Ar': 'Ar', 'K': 'K', 'Ca': 'Ca', 'Sc': 'Sc', 'Ti': 'Ti', 'V': 'V', 'Cr': 'Cr', 'Mn': 'Mn', 'Fe': 'Fe', 'Co': 'Co', 'Ni': 'Ni', 'Cu': 'Cu', 'Zn': 'Zn', 'Ga': 'Ga', 'Ge': 'Ge', 'As': 'As', 'Se': 'Se', 'Br': 'Br', 'Kr': 'Kr', 'Rb': 'Rb', 'Sr': 'Sr', 'Y': 'Y', 'Zr': 'Zr', 'Nb': 'Nb', 'Mo': 'Mo', 'Tc': 'Tc', 'Ru': 'Ru', 'Rh': 'Rh', 'Pd': 'Pd', 'Ag': 'Ag', 'Cd': 'Cd', 'In': 'In', 'Sn': 'Sn', 'Sb': 'Sb', 'Te': 'Te', 'I': 'I', 'Xe': 'Xe', 'Cs': 'Cs', 'Ba': 'Ba', 'La': 'La', 'Ce': 'Ce', 'Pr': 'Pr', 'Nd': 'Nd', 'Pm': 'Pm', 'Sm': 'Sm', 'Eu': 'Eu', 'Gd': 'Gd', 'Tb': 'Tb', 'Dy': 'Dy', 'Ho': 'Ho', 'Er': 'Er', 'Tm': 'Tm', 'Yb': 'Yb', 'Lu': 'Lu', 'Hf': 'Hf', 'Ta': 'Ta', 'W': 'W', 'Re': 'Re', 'Os': 'Os', 'Ir': 'Ir', 'Pt': 'Pt', 'Au': 'Au', 'Hg': 'Hg', 'Tl': 'Tl', 'Pb': 'Pb', 'Bi': 'Bi', 'Po': 'Po', 'At': 'At', 'Rn': 'Rn', 'Fr': 'Fr', 'Ra': 'Ra', 'Ac': 'Ac', 'Th': 'Th', 'Pa': 'Pa', 'U': 'U', 'Np': 'Np', 'Pu': 'Pu', 'Am': 'Am', 'Cm': 'Cm', 'Bk': 'Bk', 'Cf': 'Cf', 'Es': 'Es', 'Fm': 'Fm', 'Md': 'Md', 'No': 'No', 'Lr': 'Lr', 'Rf': 'Rf', 'Db': 'Db', 'Sg': 'Sg', 'Bh': 'Bh', 'Hs': 'Hs', 'Mt': 'Mt', 'Ds': 'Ds', 'Rg': 'Rg', 'Cn': 'Cn', 'Uut': 'Uut', 'Fl': 'Fl', 'Uup': 'Uup', 'Lv': 'Lv', 'Uus': 'Uus', 'Uuo': 'Uuo'}
        def convert(fileout, setname, yorgrad=False):
            prefix = ''
            if setname: 
                coordfile = prefix+'xyz.dat_'+setname
                yfile = prefix+'y.dat_'+setname
                gradfile = prefix+'grad.dat_'+setname
            else: 
                coordfile = args.xyzfile


            with open(coordfile ,'r') as fxyz:
                hf = h5py.File(fileout, 'w')
                grp = hf.create_group('dataset')
                fxyz.seek(0)
                idx=0
                for line in fxyz:
                    data={}
                    data['species'] = []
                    data['coordinates'] = np.array([])
                    natom = int(line)
                    fxyz.readline()

                    for i in range(natom):
                        ll=fxyz.readline().split()
                        data['species'].append(spdict[ll[0]] if ll[0] in spdict.keys() else ll[0])
                        data['coordinates'] = np.append(data['coordinates'], [float(i)/1.8897259886 for i in ll[-3:]])
                    data['species'] = np.array([i.encode('ascii') for i in data['species']])
                    data['coordinates'] = data['coordinates'].reshape(-1,natom,3)
        
                    subgrp = grp.create_group('molecule%08d'%idx)
                    idx+=1
                    for k, v in data.items():    ### ORIGINAL LOOP ###
                        subgrp[k] = v

        if not available:
            stopper.stopMLatom('Please install all Python module required for TorchANI')
            
        args = Args()
        args.parse(argsANI)
        cls.setCore(args.nthreads)
        convert('ANI.h5','')


    @classmethod
    def useMLmodel(cls, argsANI, subdatasets):
        cls.load()
        args = Args()
        args.parse(argsANI)
        cls.setCore(args.nthreads)
        if not cls.dataConverted: 
            cls.convertdata(argsANI, subdatasets)
            cls.dataConverted = True
        # This is used for ASE
        if args.useMLmodel:
            cls.dataConverted = False
        try: 
            energies, gradient = TorchANI_predict.predict(args)
            return energies, gradient
        except:
            energies = TorchANI_predict.predict(args)
            return energies

def printHelp():
    helpText = __doc__ + '''
  To use Interface_ANI, please install TorchANI and its dependencies

  Arguments with their default values:
    MLprog=TorchANI            enables this interface
    MLmodelType=ANI            requests ANI model

    ani.batch_size=8           batch size
    ani.max_epochs=10000000    max epochs
    ani.learning_rate=0.001    learning rate used in the Adam and SGD optimizers
    
    ani.early_stopping_learning_rate=0.00001
                               learning rate that triggers early-stopping
    
    ani.force_coefficient=0.1  weight for force
    ani.Rcr=5.2                radial cutoff radius
    ani.Rca=3.5                angular cutoff radius
    ani.EtaR=1.6               radial smoothness in radial part
    
    ani.ShfR=0.9,1.16875,      radial shifts in radial part
    1.4375,1.70625,1.975,
    2.24375,2.5125,2.78125,
    3.05,3.31875,3.5875,
    3.85625,4.125,4.9375,
    4.6625,4.93125
    
    ani.Zeta=32                angular smoothness
    
    ani.ShfZ=0.19634954,       angular shifts
    0.58904862,0.9817477,
    1.3744468,1.7671459,
    2.1598449,2.552544,
    2.9452431
    
    ani.EtaA=8                 radial smoothness in angular part
    ani.ShfA=0.9,1.55,2.2,2.85 radial shifts in angular part
    ani.Neurons=160,128,96     nuerons in different layers 

  Cite TorchANI:
    X. Gao, F. Ramezanghorbani, O. Isayev, J. S. Smith, A. E. Roitberg,
    J. Chem. Inf. Model. 2020, 60, 3408
    
  Cite ANI model:
    J. S. Smith, O. Isayev, A. E. Roitberg, Chem. Sci. 2017, 8, 3192
'''
    print(helpText)

if __name__ == '__main__':
    ANICls()
