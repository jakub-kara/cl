# Clean previous files
rm *.pyd *.pyf *.so *.mod *.o
# Create function signatures
f2py $1.f90 -m $1 -h $1.pyf
# Generate module (otherwise next step won't work) 
gfortran -c -O3 $1.f90
# Build extension module (default flags: -Wall -g -fno-second-underscore -O3 -funroll-loops)
f2py -c $1.pyf $1.f90 -L/usr/local/lib -l:liblapack.a -l:librefblas.a -l:libtmglib.a

