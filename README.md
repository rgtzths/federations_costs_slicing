# dl_costs

## Installing

To run the scenarios locally please install the requirements

 `pip install -r requirements.txt`

If you want to run the experiments in RPIs please follow the `rpi_setup_comands.md`

If you want to run the experiments in docker please run `docker compose up`.

## Running

To run locally you only need the following commands

`mpirun -np 4 python mpi_training.py -d dataset/one_hot_encoding/`

or 

`mpirun -np 4 python mpi_custom_training.py -d dataset/one_hot_encoding/`

If you want to run the single_host setting you can run

`python model.py`

To run on docker you will need to connect to the master container

`docker exec -it --user mpiuser dl_costs-master-1 bash`

Connect one time to every worker to confirm the fingerprint of the server.

`ssh worker1` and then `exit`

After this you can run the command

`mpirun -np 4 -hostfile hostfile python mpi_training.py -d dataset`

or

`mpirun -np 4 -hostfile hostfile python mpi_custom_training.py -d dataset`

inside the `code` folder.

To change the hyperparameters please confirm the available options in the mpi_training.py and mpi_custom_training.py files


# Results

The results obtained with the different hyperparameters are presented in the `paper_results` folder
the translation for the folder names is 10_g_50_l -> (decentralized optimization with 10 global epochs and 50 local epochs) and fed_sgd_64 -> (centralized optimization with batch size 64)

## Authors

* **Rafael Teixeira** - [rgtzths](https://github.com/rgtzths)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Citation

If you use this code please site our work:
Teixeira, Rafael & Antunes, Mário & Gomes, Diogo & Aguiar, Rui. (2023). The learning costs of Federated Learning in constrained scenarios. 10.1109/FiCloud58648.2023.00011. 
