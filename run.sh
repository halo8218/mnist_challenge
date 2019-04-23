#!/bin/bash
#python train.py gsop > gsop_test.log
#python train.py conv '1' > conv_test1.log
#python train.py se '1' > se_test1.log
#python train.py new '1' > new_test1.log
#python train.py conv '2' > conv_test2.log
#python train.py se '2' > se_test2.log
#python train.py new '2' > new_test2.log
#python train.py conv '3' > conv_test3.log
#python train.py se '3' > se_test3.log
#python train.py new '3' > new_test3.log

python train.py new1rs '1' > new1rs_1.log
python train.py new1ts '1' > new1ts_1.log
python train.py new1ss '1' > new1ss_1.log

python train.py new1rs '2' > new1rs_2.log
python train.py new1ts '2' > new1ts_2.log
python train.py new1ss '2' > new1ss_2.log

python train.py new1rs '3' > new1rs_3.log
python train.py new1ts '3' > new1ts_3.log
python train.py new1ss '3' > new1ss_3.log
