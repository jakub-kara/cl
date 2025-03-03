DONKEY clustering algorithm
---------------------------

Documentation WIP

Steps to run:
1. Clone repo
2. Compile density.f90 to density.[cpython-data].so using F2PY
   (link against your LAPACK and BLAS installations, example provided in build.sh)
3. Import the algorithm and have fun!

The interface follows the scikit-learn template.

Example:\n
from clustering import Donkey
data = ...
clus = Donkey()
clus.fit(data)
labels = clus.labels_
