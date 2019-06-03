import demos
import sys

task   = sys.argv[1].lower()
criterion = sys.argv[2].lower()
query_type = sys.argv[3].lower()
c = float(sys.argv[4])
M = int(sys.argv[5])
N = int(sys.argv[6])

assert criterion == 'information' or criterion == 'volume' or criterion == 'random', 'There is no criterion called ' + criterion
demos.nonbatch(task, criterion, query_type, c, M, N)

