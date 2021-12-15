To run this code:

1. cuda calculations on CRC:
$ qrsh -q gpu -l gpu_card=1
$ cd {the directory where the code is in}
$ module load cuda/11.0
$ module load mpich/3.3/intel/19.0 
$ make
$ ./a.out

This will generate InputA, InputB, OutputA and OutputB. The terminal will have the run time output.

2. plotting:
$ cd {the directory where the code is in}
$ python3 plot.py

This will generate Gray_Scott_Input.png and Gray_Scott_Output.png.
