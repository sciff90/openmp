#mcmc openmp Makefile
#Define Objects

OBJ = obj
SRC = src
PLOTS = plots
DATA = data
EXEC = xmcmc

OBJECTS = ${OBJ}/mcmc.o

#Define Compiler

#############g++############
COMP = g++
	
#Define Switches
#SWITCH = -O2 -fopenmp -std=c++0x
SWITCH = -g -fopenmp -std=c++0x

#Define Libraries
#LIB_DIR = 
#LIB = 
IDIR = -I.
##############################

###########icpc###############
#COMP = icpc

#Define Switches
#SWITCH = -O2 -openmp -std=gnu++11
#SWITCH = -g

#Define Libraries
#LIB_DIR = -L/opt/intel/composer_xe_2013.2.146/compiler/lib/intel64 
#LIB = -lpthread -liomp5
#IDIR = -I. -I/usr/include/x86_64-linux-gnu/c++/4.7/

##############################

#Makefile
all: $(OBJECTS)
		$(COMP) $(SWITCH) ${OBJECTS} $(IDIR) -o ${EXEC} $(LIB_DIR) $(LIB)
			
${OBJ}/mcmc.o: ${SRC}/mcmc.cpp
		$(COMP) -c $(SWITCH) $(LIB_DIR) $(LIB) $(IDIR) ${SRC}/mcmc.cpp -o ${OBJ}/mcmc.o
			
clean:
		rm ${OBJ}/*.o		
		rm ${DATA}/*.dat

#Cleaning Everything
cleanest: clean
		rm ${EXEC}
# End of the makefile
