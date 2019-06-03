import demos
import sys

task   = sys.argv[1].lower()
criterion = sys.argv[2].lower()
c = float(sys.argv[3])
M = int(sys.argv[4])
N = int(sys.argv[5])

assert criterion == 'information' or criterion == 'volume' or criterion == 'random', 'There is no criterion called ' + criterion
demos.nonbatch(task, criterion, M, N)

