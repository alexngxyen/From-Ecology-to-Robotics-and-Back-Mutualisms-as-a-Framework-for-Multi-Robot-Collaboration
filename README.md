# From-Ecology-to-Robotics-and-Back-Mutualisms-as-a-Framework-for-Multi-Robot-Collaboration

First, please install the robotarium-matlab-simulator from https://github.com/robotarium/robotarium-matlab-simulator. The repository contains detailed instructions on how to get started.

Once installed, you should be able to run all of the "ecobotic_..." scripts successfully, meaning that these scripts are uploadable to https://www.robotarium.gatech.edu/ such that experimental videos can be acquired (Figure 1). Each of these scripts will output task fecundity, longevity, and robot fitness data as a function of the experimental parameters chosen (Table 1) for the four different landscape profiles considered in the case study. This data is then saved to "case_study_experiment_data.mat", which generates the plots of task fecundity, longevity, and robot fitness (on average) with $\pm 1 \sigma$ error bars for both robots (Figure 2) where the landscape composition varies between low-variability (i.e., landscape profiles where terrain types alternate frequently) and low-variability (i.e., profiles where terrain types alternate infrequently). Note that the landscape composition varies discretely – and not over a continuum.   

Reference: <br>
Nguyen A. A., Rodriguez Curras M., Egerstedt M., and Pauli J. N. (2025) Mutualisms as a framework for multi-robot collaboration. <i>Front. Robot. AI<i> 12:1566452. doi: 10.3389/frobt.2025.1566452
