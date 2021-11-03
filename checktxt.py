import filecmp
import sys

f1 = sys.argv[1]
f2 = sys.argv[2]

result = filecmp.cmp(f1, f2, shallow=False)
print(result)
