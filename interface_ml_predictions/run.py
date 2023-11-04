#!/usr/bin/env python3
""" Run a set of QM calculations. """
import os
from pathlib import Path
from argparse import ArgumentParser
import file_utils
import numpy as np
import ml_acc

FILES = {
    "geom": "qm_geom",
    "state": "qm_state",
    "energy": "qm_energy",
    "gradient": "qm_grad",
    "oscill": "qm_oscill",
    "mm_geom": "mm_geom",
}

INTERFACES = {
        "ml_acc": [ml_acc.interface, r"adc(2)"],
        }


def main():
    """ Run a set of QM calculations using a specified interface. """
    parser = ArgumentParser(
        description="Run an electronic structure calculation.")
    parser.add_argument("interface", type=str, help="Interface to use.")
    parser.add_argument("work_dir", type=str, help="Working directory.")
    args = parser.parse_args()
    try:
        run(args)
    except:
        cleanup()
        raise


def run(args):
    """ Read input files and run the interface. """
    in_data = dict()
    with open('geom','r') as infile:
        lines = infile.readlines()
        species_placeholder = []
        for x in lines:
            species_placeholder.append(x.split(' ')[0])

    in_data["species"] = species_placeholder 
    in_data["geom"] = np.loadtxt(FILES["geom"])
    in_data["state"] = np.loadtxt(FILES["state"], dtype=int).flat[0]
    in_data["natom"] = len(in_data["geom"])
    try:
        in_data["mm_geom"] = np.loadtxt(FILES["mm_geom"])
        in_data["qmmm"] = True
    except:
        in_data["qmmm"] = False
    # Go to work directory and run calculation.
    cwd = Path(os.getcwd())
    os.chdir(args.work_dir)
    interface = INTERFACES[args.interface]
    qm_prog = interface[0](*interface[1:], in_data)
    qm_prog.update_input()
    qm_prog.run()
    qm_prog.read()
    os.chdir(cwd)
    # Write results.
    calc_write(qm_prog.results,in_data)


def calc_write(results,data):
    """ Write output files. """
    num_format = " {:24.18f}"
    xyz_format = 3 * " {:20.14f}"
    with open(FILES["energy"], "w") as out_f:
        for energy in results["energy"]:
            out_f.write(num_format.format(energy))
        out_f.write("\n")

    with open(FILES["gradient"], "w") as out_f:
        for grad in results["gradient"]:
            out_f.write(xyz_format.format(*grad) + "\n")
    with open(FILES["oscill"], "w") as out_f:
        for oscill in results["oscill"]:
            out_f.write(num_format.format(oscill))
        out_f.write("\n")
    #if os.path.isfile('dynamics.stop')==True:  ### BACKUP  ####
    if os.path.isfile('ab_initio')==True:
        with open('../data/energy.dat', "a") as out_f:
            for energy in results["energy"]:
                out_f.write(num_format.format(energy))
            out_f.write("\n")
        with open('../data/grad.dat', "a") as out_f:
            out_f.write(str(data['natom'])+ "\n")
            out_f.write("\n")
            for atoms, grad in zip(data['species'],results["gradient"]):
                out_f.write(atoms + xyz_format.format(*grad)+ "\n")
        with open('../data/xyz.dat', "a") as out_f:
            out_f.write(str(data['natom'])+ "\n")
            out_f.write("\n")
            for atoms, geos in zip(data['species'],data["geom"]):
                out_f.write(atoms + xyz_format.format(*geos)+ "\n")
        with open('../data/current_state.dat', "a") as out_f:
            out_f.write(str(int(data['state']-1)))   ### Correct it to GS=0
            out_f.write("\n")
        with open('../data/oscillator.dat', "a") as out_f:
            for oscill in results["oscill"]:
                out_f.write(num_format.format(oscill))
            out_f.write("\n")
        file_utils.remove('ab_initio')   ### just to track ab initio calcs


def cleanup():
    for path in FILES.values():
        file_utils.remove(path)


if __name__ == "__main__":
    main()
