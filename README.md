# TKU-PSO
This is a repository for TKU-PSO, a Particle swarm optimization algorithm for top-k high-utility itemsets mining in quantitative transactional data.

How to run:
* Select algorithm parameters in TKU_PSO.java (k, pop_size, and iterations).
* Set the "input" string in TKU_PSO.java to the path of the database file. The file must be in SPMF format, [here are some example datasets.](https://www.philippe-fournier-viger.com/spmf/index.php?link=datasets.php)
* Set the "output" string to any .txt file path. The discovered patterns are written to this file during execution.
* Run main.java

The project also contains logic for validating algorithm accuracy in FindAcc.java.  
