""" Interface for running ML predictions interfaced with Turbomole ADC(2) calculations. """
import subprocess
import re
import os
import numpy as np
import file_utils
import interface_TorchANI
import stopper
from args_class import ArgsBase
import datetime
import utils

STDOUT = "qm.log"
STDERR = "qm.err"

class ml_prediction():
    """ Interface for turbomole calculations. """
    def __init__(self, data):
        self.data = data
        self.results = dict()

    def update_coord(self):
        """ Update coord file with self.data["geom"]. """
        file_utils.replace_cols_inplace("coord", self.data["geom"], r"\$coord")
        if self.data["qmmm"]:
            fn = file_utils.search_file("control", r"\$point_charges")[0]
            fn = fn.split("=")[1].split()[0]
            file_utils.replace_cols_inplace(fn, self.data["mm_geom"], r"\$point_charges")

    def update_coord_ml(self):
        """ Update xyz.dat file with self.data["geom"]. """
        file_utils.replace_cols_inplace("xyz.dat", self.data["geom"], r"\$bla", [1,2,3])

    def update_state(self):
        """ Update control file to request gradient of self.data["state"]. """
        raise NotImplementedError("Need to call specific interface.")

    def update_state_ml(self):
        n_sub = file_utils.replace_inplace("../dynamics.in", r"current_state=.*", r"current_state={}".format(self.data["state"]-1))
        if n_sub < 1:
            raise ValueError("Expected machine learning file not found in directory.")

    def update_input_tm(self):
        """ Update all input files with values from self.data.  """
        self.update_coord()
        self.update_state()
    
    def update_input(self):
        """ Update all input files with values from self.data for ML routine  """
        self.update_coord_ml()
        self.update_state_ml()

    def run(self):
        """ Run the calculation, check success and read results. """
        raise NotImplementedError("Need to call specific interface.")

    def read(self):
        """ Read calculation results. """
        raise NotImplementedError("Need to call specific interface.")


class interface(ml_prediction):
    """ Interface for ML predictions. """

    def __init__(self, model, data):
        self.data = data
        self.model = re.escape(model)
        self.gs_model = re.escape(model)
        if model == "adc(2)":
            self.gs_model = "mp2"
        self.results = dict()

    def update_state(self):
        """ Update control file to request gradient of self.data["state"]. """
        if self.data["state"] == 1:
            state_string = r"(x)"
        else:
            state_string = r"(a {})".format(self.data["state"]-1)
        n_sub = file_utils.replace_inplace("control",
                                   r"(geoopt +model={} +state=).*".format(self.model),
                                   r"\1" + state_string)
        if n_sub < 1:
            raise ValueError("Expected geoopt section not found in control file.")       

    def read(self):
        with open('../dynamics.in','r') as infile:
            in_args = [line.rstrip() for line in infile]   ## unparsed arguments
        args = Args()
        args.parse(in_args)
        locargs = args.args2pass
        std_limit = 0.0015936010974213599    ### Limit for STD in hartree to trigger ab initio. 0.0015936010974213599=1kcal/mol 
        en_gap_limit = 0.00367493            ### 0.1 eV in hartree
        std_en = np.std((self.results["temp_energies"][0].ravel(),self.results["temp_energies"][1].ravel()), axis=0)     ### get STD to evaluate preformance ###
        en_gap = [np.mean(self.results["temp_energies"],axis=0)[n] - np.mean(self.results["temp_energies"],axis=0)[n-1] for n in range(1,len(np.mean(self.results["temp_energies"],axis=0)))]
        std_grad = np.std((self.results["temp_grad"][0].ravel(),self.results["temp_grad"][1].ravel()), axis=0)     ### get STD to evaluate preformance ###
        try:
            if any(i > args.conf_thr*std_limit for i in std_en) or en_gap[args.current_state]<arg.min_en_gap*en_gap_limit or en_gap[args.current_state-1]<args.min_en_gap*en_gap_limit or any(j > std_limit for j in std_grad): ### check for desired accuray in hartree
                print('Invoking TM ab initio calculation')
                open('../ab_initio','a').close() 
                self.update_input_tm()     ### Update and run TM calc 
                self.run_tm()
                self.read_tm()
                if os.path.isfile('../first_iter')==True:
                    with open('../dynamics.in', 'r+') as infile:
                        content = infile.read()
                        infile.seek(0 ,0)
                        infile.write('$restart' + '\n' + content)
                    file_utils.remove('../first_iter')   #### commented for testing ###
                #open('../dynamics.stop','a').close() 
                with open ('../tracker','a') as f:
                    f.write('1\n')
            else:
                print('ML models agree')
                self.results["energy"] = np.mean(self.results["temp_energies"],axis=0)
                self.results["oscill"] = np.mean(self.results["temp_oscill"], axis=0) 
                self.results["gradient"] = np.mean(self.results["temp_grad"], axis=0)*0.529177208 #conversion from hartree/A -> hartree/bohr
                with open ('../tracker','a') as f:
                    f.write('0\n')
            file_utils.remove("ANI.h5")
        except:
            if any(i > args.conf_thr*std_limit for i in std_en) or en_gap[args.current_state-1]<args.min_en_gap*en_gap_limit:
                print('Invoking TM ab initio calculation')
                self.update_input_tm()     ### Update and run TM calc 
                self.run_tm()
                self.read_tm()
                if os.path.isfile('../first_iter')==True:
                    with open('../dynamics.in', 'r+') as infile:
                        content = infile.read()
                        infile.seek(0 ,0)
                        infile.write('$restart' + '\n' + content)
                    file_utils.remove('../first_iter')   #### commented for testing ###
                with open ('../tracker','a') as f:
                    f.write('1\n')
            else:
                print('ML models agree')
                self.results["energy"] = np.mean(self.results["temp_energies"],axis=0)
                self.results["oscill"] = np.mean(self.results["temp_oscill"], axis=0) 
                self.results["gradient"] = np.mean(self.results["temp_grad"], axis=0)*0.529177208 #conversion from hartree/A -> hartree/bohr
                with open ('../tracker','a') as f:
                    f.write('0\n')
            file_utils.remove("ANI.h5")




    def run_tm(self):
        """ Run the turbomole calculation, check success and read results. """
        with open(STDOUT, "w") as out, open(STDERR, "w") as err:
            subprocess.run("dscf", stdout=out, stderr=err)
            actual_check()
            subprocess.run("ricc2", stdout=out, stderr=err)
            actual_check()

    def read_tm(self):
        self.results["energy"] = ricc2_energy(STDOUT, self.gs_model.upper())
        self.results["gradient"] = ricc2_gradient()[self.data["state"]]
        try:
            self.results["oscill"] = ricc2_oscill(STDOUT)
        except:
            pass


    def run(self):
        """ Call ML model to predict values and read results"""
        subdatasets = [] 
        with open('../dynamics.in','r') as infile:
            in_args = [line.rstrip() for line in infile]   ## unparsed arguments
        args = Args()
        args.parse(in_args)
        locargs = args.args2pass
        self.results["temp_energies"] = np.empty((args.num_models,args.nstates))
        self.results["temp_oscill"] = np.empty((args.num_models,args.nstates-1))
        self.results["temp_grad"] = np.empty((args.num_models,args.natoms,3))
        for i in range(0,args.num_models):
            locargs = utils.addReplaceArg('MLmodelIn', 'MLmodelIn = ../ensemble%d.unf'%(i+1), locargs)
            self.results["temp_energies"][i], self.results["temp_grad"][i] = interface_TorchANI.ANICls.useMLmodel(locargs, subdatasets)
        for i in range(0,args.num_models):
            locargs = utils.addReplaceArg('nstates', 'nstates = 4', locargs)
            locargs = utils.addReplaceArg('current_state', 'current_state = -1', locargs)
            locargs = utils.addReplaceArg('MLmodelIn', 'MLmodelIn = ../mlosc%d.unf'%(i+1), locargs)
            self.results["temp_oscill"][i] = interface_TorchANI.ANICls.useMLmodel(locargs, subdatasets) 

class Args(ArgsBase):
    def __init__(self):
        super().__init__() 
        self.args2pass = []

    def parse(self, argsraw):
        if len(argsraw) == 0:
            Doc.printDoc({})
            stopper.stopMLatom('At least one option should be provided')
        elif len(argsraw) == 1:
            if os.path.exists(argsraw[0]):
                self.parse_input_file(argsraw[0])
            else:
                self.parse_input_content(argsraw[0])
            self.args2pass = self.args_string_list(['', None])
        elif len(argsraw) >= 2:
            self.parse_input_content(argsraw)
            self.args2pass = self.args_string_list(['', None])


def ricc2_energy(fname, model):
    gs_energy = file_utils.search_file(fname, "Final "+model+" energy")
    file_utils.split_columns(gs_energy, col=5, convert=np.float64)
    ex_energy = file_utils.search_file(fname, "Energy:")
    ex_energy = file_utils.split_columns(ex_energy, 1, convert=np.float64)
    energy = np.repeat(gs_energy, len(ex_energy) + 1)
    energy[1:] = energy[1:] + ex_energy
    return energy


def ricc2_gs_energy(model):
    energy = file_utils.search_file(STDOUT, "Total Energy  ")
    energy = np.float(energy[0].split()[3])
    return np.array([energy])


def ricc2_oscill(fname):
    """ Read oscillator strengths from STDOUT file. """
    oscill = file_utils.search_file(fname,
                                    r"oscillator strength \(length gauge\)")
    oscill = file_utils.split_columns(oscill, col=5)
    return np.array(oscill, dtype=np.float64)


def ricc2_gradient():
    grads = dict()
    # Try to get ground state gradient.
    try:
        cfile = file_utils.go_to_keyword(STDOUT, "GROUND STATE FIRST-ORDER PROPERTIES")[0]
        grads[1] = get_grad_from_stdout(cfile)
    except:
        pass
    # Try to get excited state gradients.
    try:
        cfile = file_utils.go_to_keyword(STDOUT, "EXCITED STATE PROPERTIES")[0]
    except:
        return grads
    while True:
        try:
            line = file_utils.search_file(cfile, 
                    "Excited state reached by transition:", 
                    max_res=1, 
                    close=False, 
                    after=3)
            cstate = int(line[0].split()[4]) + 1
        except:
            cfile.close()
            break
        try:
            file_utils.search_file(cfile, 
                    "cartesian gradient of the energy",
                    max_res=1,
                    stop_at=r"\+={73}\+",
                    close=False)
            grads[cstate] = get_grad_from_stdout(cfile)
        except:
            pass
    return grads


def get_grad_from_gradient(natom):
    grad = file_utils.search_file("gradient", r"cycle", after=2*natom)[-natom:]
    grad = file_utils.split_columns(grad, col=[0, 1, 2],  convert=file_utils.fortran_double)
    return grad

def get_grad_from_gradient_ml(num_mod):
    grad = file_utils.search_file("gradest%d.dat"%num_mod, r" ")
    grad = file_utils.split_columns(grad, col=[0, 1, 2],  convert=file_utils.fortran_double)
    return grad


def get_grad_from_stdout(cfile):
    grad = file_utils.search_file(cfile, r"ATOM", after=3, stop_at=r"resulting FORCE", close=False)
    grad = [line[5:] for line in grad]
    grad = [' '.join(grad[0::3]), ' '.join(grad[1::3]), ' '.join(grad[2::3])]
    grad = [line.split() for line in grad]
    grad = [list(map(file_utils.fortran_double, vals)) for vals in grad]
    return np.array(grad).T


def actual_check():
    """ Check that the Turbomole calculation finished without error. """
    check = subprocess.run("actual", stdout=subprocess.PIPE)
    if check.stdout.startswith(b'fine, there is no data group "$actual step"'):
        return
    raise RuntimeError("Turbomole calculation failed.")
