import os
import sys

current_dir = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.append(os.path.join(current_dir, 'data'))
print(sys.path)