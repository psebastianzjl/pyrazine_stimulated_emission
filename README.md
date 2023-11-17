# pyrazine_stimulated_emission
Code used for QC-NADMD and DW spectra as well as a working example using the models employed for this publication.

This repository contains four folders: the training data used in this work to train the models and predict their accuracy (training data), the interface used to propagate the trajectories (interface_ml_predictions), the codes used for postprocessing the data to obtain the spectra (postprocessing) and one working example for a single trajectory (TRAJ1_example).

The models can be trained as described by the MLatom 2 manual resulting in sets of ML models which can be combined using the "ensemble" routine implemented in pytorch.

The interface to predict the energies, gradients and oscillator strengths for every geometry in the simulation uses parts of the MLatom 2 source code, but was changed for increased efficiency. For the latest stable version of MLatom please visit https://www.mlatom.com. 

For postprocessing, the code requires each trajectory to be propagated in its own directory ("TRAJX") as the codes provided will search for that name. The "chrono_re.py" script reorders the files in the "./TRAJX/Result" directory for improved readability. The spectra can be generated using the provided scripts.

For the working example to work properly, the path in dynamics in needs to be changed to point to the path of "interface_ml_predictions". The execution requires the Turbomole programpackage for potentially required ab initio calculations. By executing the "nad.exe", the trajectory is propagated.
