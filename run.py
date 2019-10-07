import demos
import sys
import numpy as np
import os

task   = sys.argv[1].lower()
criterion = sys.argv[2].lower()
query_type = sys.argv[3].lower()
epsilon = float(sys.argv[4])
M = int(sys.argv[5])

assert criterion == 'information' or criterion == 'volume' or criterion == 'random', 'There is no criterion called ' + criterion

demos.nonbatch(task, criterion, query_type, epsilon, M)
