import os

my_cmd = 'cd .. && cd DD && python3 deepdream_api.py -p mixed3b_3x3_pre_relu 1 test.jpg'

os.system(my_cmd)
