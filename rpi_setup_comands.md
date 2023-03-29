
# Start

Add the different RPIs IPs to the `/etc/hosts` using the following format `###.###.###.### master` or `###.###.###.### worker1` for every RPI.

Make sure that the names are consistent and that the master is the RPI4.

# Installing libraries

Update the system and install the mpi libraries
`sudo apt install`

`sudo apt -y install openmpi-bin libopenmpi-dev libblas-dev`

# Cerating user

Create an user to run the mpi experiments
`sudo adduser mpiuser` 

`sudo usermod -aG sudo mpiuser`

Change to the user

`su - mpiuser`

Create the ssh keys that will be used to connected between 
RPIs

`mkdir .ssh`

`cd .ssh`

`ssh-keygen -t rsa -f id_rsa`

`cat id_rsa.pub >> authorized_keys`

Add the local python to the PATH
`PATH=$PATH:/home/mpiuser/.local/bin`

# Python libraries

Install python libraries

`pip install tensorflow==2.10 --no-cache-dir`

`pip install scikit-learn`

`pip install mpi4py`

# SSH key sharing

Copy your ssh key to the other RPIs (only needed for the master) 

`ssh-copy-id worker1` #Do also for 2 and 3

# Files for running

Copy the dataset file of the corresponding worker node (worker1 -> subset_1), the mpi_training.py and mpi_custom_training.py to the home directory

The master file needs to have X_cv.csv and y_cv.csv files.

# Running the experiment

To run the experiment use: 

`mpirun -np 4 -hostfile host_file python mpi_training.py -d 'dataset_folder'`

or

`mpirun -np 4 -hostfile host_file python mpi_custom_training.py -d 'dataset_folder'`