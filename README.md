# Classification of RNA backbone conformations into rotamers using 13C' chemical shifts: datasets, raw data and classification scripts.

## This repository contains the following elements: 

### (1) The experimental and theoretical databases used for RNA backbone rotamers classification using 13C' chemical shifts (CS) contained within this two files:

    (1.1) ExperimentalDatabase.csv : a csv file with the information from the experimental database (PDB id, dinucleotide sequence, model number in the NMR ensemble, 13C' CS values, torsional angle values, rotamer and rotamer families labels).

    (1.2) TheoreticalDatabase.csv : a csv file with the information from the theoretical database (dinucleotide sequence, 13C' shielding values, torsional angle values, rotamer and rotamer families labels).​

In both of them, the column DN refers to 'dinucleotide' and contains the base identities and numbers. 
The column SEQ refers to 'sequence' and contains the dinucleotide base sequence. 
In (1.1) the term CS in the column names, refers to 'chemical shift'. The column PDBID contains the PDB id of the RNA structure. The column MODEL contains the model number from te NMR ensemble.
In (1.2) the term shielding is used in spite of chemical shift because its the isotropic 13C' shielding value obtained from DFT computations. 

### (2) The experimental and theoretical raw data, i.e. the PDB and BMRB files from the experimental database and the RES.com files used for the DFT computation of the 13C'CS from the theoretical database is contained within the RawData folder:

    (2.1) The ExperimentalData folder contains:

        (2.1.1) ChemicalShifts(BMRBfiles) : a folder with the BMRB files (i.e. experimental 13C' CS)  from the experimental database.

        (2.1.2) Coordinates(PDBfiles) : a folder with the PDB files (i.e. atom coordinates) from the experimental database.   

    (2.2) The TheoreticalData folder contains:

        (2.2.1) DFTInputFiles : a folder with the RES.com files used for the DFT computation of the 13C'CS from the theoretical database, organized in their corresponding folders for each mononucleotide (MN1 and MN2) from the theoretical dinucleotide conformations.

### (3) The machine learning classification scripts are organized inside the ClassificationScripts folder as follows:

    (3.1) files: a folder which contains all the necessary files for classification (e.g. the files with theoretical CS obtained with different references, the experimental 13C' CS dataset for the first model from the NMR ensembles, the ROSUM matrices for the different families of rotamers)

    (3.2) experimental_vs_experimental_classification.py : a python script for the experimental vs experimental classification of RNA rotamers, using a LOO-CV approach, with different machine leraning classifiers

    (3.3) experimental_vs_theoretical_classification.py : a python script for the experimental vs theoretical classification of RNA rotamers, with different machine leraning classifiers

    (3.4) theoretical_vs_theoretical_classification.py : a python script for the theoretical vs theoretical classification of RNA rotamers, using a LOO-CV approach, with different machine leraning classifiers

    (3.5) utils.py : a python module with two funtions used by the three above described classification routines
