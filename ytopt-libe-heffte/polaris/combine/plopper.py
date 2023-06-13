import os
import sys
import subprocess
import random
import psutil
import numpy as np
import warnings

class Plopper:
    def __init__(self,sourcefile,outputdir):
        # Initilizing global variables
        self.sourcefile = sourcefile
        self.outputdir = outputdir+"/tmp_files"
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

    #Creating a dictionary using parameter label and value
    def createDict(self, x, params):
        dictVal = {}
        for p, v in zip(params, x):
            if type(v) is np.ndarray and v.shape == ():
                v = v.tolist()
            dictVal[p] = v
        return(dictVal)

    #Replace the Markers in the source file with the corresponding values
    def plotValues(self, dictVal, inputfile, outputfile):
        with open(inputfile, "r") as f1:
            buf = f1.readlines()

        with open(outputfile, "w") as f2:
            for line in buf:
                modify_line = line
                for key, value in dictVal.items():
                    if key in modify_line:
                        if value and value != 'None': #For empty string options
                            modify_line = modify_line.replace('#'+key, str(value))
                if modify_line != line:
                    f2.write(modify_line)
                else:
                    #To avoid writing the Marker
                    f2.write(line)

    # Function to find the execution time of the interim file, and return the execution time as cost to the search module
    def findRuntime(self, x, params, worker):
        #print(worker, x, params)
        dictVal = self.createDict(x, params)
        #print(worker, dictVal)

        # Generate intermediate file
        counter = random.randint(1, 10001) # To reduce collision increasing the sampling intervals
        interimfile = f"{self.outputdir}/{counter}.sh"
        self.plotValues(dictVal, self.sourcefile, interimfile)

        kernel_dir = os.path.dirname(self.sourcefile)
        #cmd2 = f"{kernel_dir}/exe.pl {dictVal['P9']} {interimfile}"
        #print(worker, cmd2)

        #Find the execution time
        app_timeout = 300
        n_nodes = 8
        ppn = 4
        n_repeats = 3

        cmd = f"timeout {app_timeout} mpiexec -n {n_nodes} --ppn {ppn} --depth {dictVal['p9']} sh {interimfile}"
        print(worker,cmd)
        #exetime = float('inf')
        #exetime = sys.maxsize
        #exetime = -1
        results = []
        for attempt in range(n_repeats):
            this_log = f"{self.outputdir}/{counter}_{attempt}.log"
            execution_status = subprocess.Popen(cmd, shell=True, stdout=this_log, stderr=this_log)
            try:
                with open(this_log,"r") as logged:
                    lines = [_.rstrip() for _ in logged.readlines()]
                    for line in lines:
                        if "Performance: " in line:
                            split = [_ for _ in line.split(' ') if len(_) > 0]
                            results.append(-1 * float(split[1]))
                            break
            except Exception as e:
                warnings.warn(f"Evaluation raised {e.__class__.__name__}: {e.args}")
        return np.mean(results)

