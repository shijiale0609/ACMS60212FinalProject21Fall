To run this code:

1. cuda calculations on CRC:
$ qrsh -q gpu -l gpu_card=1
$ cd {the directory where the code is in}
$ module load cuda
$ nvcc main.cu
$ ./a.out

This will generate A0.txt, B0.txt, A.txt and B,txt. The terminal will have the run time output.

2. plotting:
$ cd {the directory where the code is in}
$ python3 plot.py

This will generate Initial_State.png and Final_State.png.