# Data

The ADULT dataset has already been preprocessed and added to this repository. 

You can also obtain the files by following these steps:

1) Go to the IPUMS USA website (https://usa.ipums.org/) and add the following variables to your data cart:
````
ACREHOUS, AGE, AVAILBLE, CITIZEN, CLASSWKR, DIFFCARE, DIFFEYE, DIFFHEAR, DIFFMOB, 
DIFFPHYS, DIFFREM, DIFFSENS, DIVINYR, EDUC, EMPSTAT, FERTYR, FOODSTMP, GRADEATT, 
HCOVANY, HCOVPRIV, HINSCAID, HINSCARE, HINSVA, HISPAN, LABFORCE, LOOKING, MARRINYR, 
MARRNO, MARST, METRO, MIGRATE1, MIGTYPE1, MORTGAGE, MULTGEN, NCHILD, NCHLT5, 
NCOUPLES, NFATHERS, NMOTHERS, NSIBS, OWNERSHP, RACAMIND, RACASIAN, RACBLK, RACE, 
RACOTHER, RACPACIS, RACWHT, RELATE, SCHLTYPE, SCHOOL, SEX, SPEAKENG, VACANCY, 
VEHICLES, VET01LTR, VET47X50, VET55X64, VET75X90, VET90X01, VETDISAB, VETKOREA, 
VETSTAT, VETVIETN, VETWWII, WIDINYR, WORKEDYR
````
2) Submit separate extract requests for the 1-yr samples (denoted as "ACS") for the years 2010, 2014, and 2018.
3) Rename the .dat and .xml files to acs_YEAR_1yr.dat and acs_YEAR_1yr.xml (for example: acs_2010_1yr.dat)
4) Move the files (6 in total) to ./Datasets/acs/
5) Run the following command
````
python Util/process_data/process_ipums.py --fn acs_2010_1yr acs_2014_1yr acs_2018_1yr
````

# Execution

You can run PMW<sup>Pub</sup> on the 2018 ACS dataset for PA using the following command:

````
python pmw_pub.py \
--dataset <DATASET> --state <STATE>  \
--dataset_pub <DATASET_PUB> --state_pub <STATE_PUB> \
--num_runs <NUM_RUNS> --marginal <MARGINAL> \
--workload <WORKLOAD> --pub_frac <FRAC> \
--epsilon $EPSILON --T $T --permute
````

For more details, you can also refer to the example scripts found in ./scripts/ to execute the code for our experiments.

For example,
````
./scripts/acs_PA/run_pmw_pub.sh
````
Executes our method, PMW<sup>Pub</sup>, on the 2018 ACS Data for Pennsylvania (PA-18) using Ohio (OH-18) as the public dataset.

# Acknowledgements

We adapt code from

1) https://github.com/giusevtr/fem
