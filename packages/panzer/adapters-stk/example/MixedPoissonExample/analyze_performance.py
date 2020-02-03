#! /usr/bin/env python3

"""
Script for analyzing Panzer kernel performance on next-generation
architectures.  Runs hierarchic parallelism and generates plots from
data.
"""

__version__ = "1.0"
__author__  = "Roger Pawlowski"
__date__    = "Dec 2018"

# Import python modules for command-line options, the operating system, regular
# expressions, and system functions
import subprocess
import argparse
import os
import re
import sys
import datetime

#############################################################################

def main():

    """Script for analyzing Phalanx performance on next-generation architectures."""

    # Initialization
    print('****************************************')
    print('* Starting Panzer Analysis')
    print('****************************************')

    parser = argparse.ArgumentParser(description='Panzer hierarchic parallelism analysis script')
    parser.add_argument('-r', '--run', action='store_true', help='Run the executable to generate data and output to files.')
    parser.add_argument('-a', '--analyze', action='store_true', help='Analyze the data from files generated with --run.')
    parser.add_argument('-p', '--prefix', help='Add a prefix string to all output filenames.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more data to screen.')
    parser.add_argument('-o', '--basis-order', type=int, required=True, help='FE basis order.')
    parser.add_argument('-ts', '--team-size', type=int, required=True, help='Team size for hierarchic parallelism.')
    parser.add_argument('-vs', '--vector-size', type=int, required=True, help='Vector size for hierarchic parallelism.')
    parser.add_argument('-s', '--use-shared-memory', action='store_true', help='Use shared memory for hierarchic parallelism.')
    args = parser.parse_args()

    nx = 100
    ny = 10
    nz = 10
    order = args.basis_order
    ts = args.team_size
    vs = args.vector_size
    shared_mem_flag = "--no-use-shared-mem-for-ad"
    if args.use_shared_memory:
        shared_mem_flag = "--use-shared-mem-for-ad"
    print("basis order = %d, team size = %d, vector size = %d\n" % (order, ts, vs))
    print("shared memory flag = %s \n" % (shared_mem_flag))

    #executable = "./PanzerAdaptersSTK_MixedPoissonExample.exe"
    executable = "jsrun -p 1 ./PanzerAdaptersSTK_MixedPoissonExample.exe"

    print("Starting Workset Analysis")

    ws_step_size = 1000
    #workset_range = range(100,2000+ws_step_size,ws_step_size)
    workset_range = []
    workset_range += list(range(100,600,100))
    workset_range += list(range(750,1250,250))
    workset_range += list(range(2000,12000,2000))
    #workset_range.append(200)
    #workset_range.append(300)
    #workset_range.append(400)
    #workset_range.append(500)
    #workset_range.append(1000)
    #workset_range.append(1200)
    #workset_range.append(1400)
    #workset_range.append(1600)
    #workset_range.append(1800)
    #workset_range.append(2000)
    print("workset range = "+str(workset_range))

    timings = {}
    if args.analyze:
        import numpy as np
        timings["panzer::AssemblyEngine::evaluate_volume(panzer::Traits::Jacobian)"] = np.zeros(len(workset_range),dtype=np.float64)
        timings["[panzer::Traits::Jacobian] GatherSolution (Tpetra): GRADPHI_FIELD (panzer::Traits::Jacobian)"] = np.zeros(len(workset_range),dtype=np.float64)
        timings["[panzer::Traits::Jacobian] DOFDiv: DIV_GRADPHI_FIELD (panzer::Traits::Jacobian)"] = np.zeros(len(workset_range),dtype=np.float64)
        timings["[panzer::Traits::Jacobian] Integrator_DivBasisTimesScalar (EVALUATES):  RESIDUAL_GRADPHI_FIELD"] = np.zeros(len(workset_range),dtype=np.float64)
        timings["[panzer::Traits::Jacobian] Sine Source"] = np.zeros(len(workset_range),dtype=np.float64)
        timings["[panzer::Traits::Jacobian] Integrator_DivBasisTimesScalar (CONTRIBUTES):  RESIDUAL_GRADPHI_FIELD"] = np.zeros(len(workset_range),dtype=np.float64)
        timings["[panzer::Traits::Jacobian] SCATTER_GRADPHI_FIELD Scatter Residual (Jacobian)"] = np.zeros(len(workset_range),dtype=np.float64)
        timings["[panzer::Traits::Jacobian] DOF: GRADPHI_FIELD accel_jac  (panzer::Traits::Jacobian)"] = np.zeros(len(workset_range),dtype=np.float64)
        timings["[panzer::Traits::Jacobian] Integrator_GradBasisDotVector (EVALUATES):  RESIDUAL_PHI_MASS_OP"] = np.zeros(len(workset_range),dtype=np.float64)
        timings["[panzer::Traits::Jacobian] GatherSolution (Tpetra): PHI (panzer::Traits::Jacobian)"] = np.zeros(len(workset_range),dtype=np.float64)
        timings["[panzer::Traits::Jacobian] DOFGradient: GRAD_PHI (panzer::Traits::Jacobian)"] = np.zeros(len(workset_range),dtype=np.float64)
        timings["[panzer::Traits::Jacobian] Integrator_GradBasisDotVector (EVALUATES):  RESIDUAL_PHI_DIFFUSION_OP"] = np.zeros(len(workset_range),dtype=np.float64)
        timings["[panzer::Traits::Jacobian] SumStatic Rank 2 Evaluator"] = np.zeros(len(workset_range),dtype=np.float64)
        timings["[panzer::Traits::Jacobian] SCATTER_PHI Scatter Residual (Jacobian)"] = np.zeros(len(workset_range),dtype=np.float64)

    #print dir(np)
    num_samples = 5;
    for ns in range(num_samples):
        print("run=%i" % (ns))

        for i in range(len(workset_range)):

            ws = workset_range[i]

            filename =   "mixed_poisson_nx_%i_ny_%i_nz_%i_order_%i_ws_%i_ts_%i_vs_%i_ns_%i.log" % (nx, ny, nz , order, ws, ts, vs, ns)
            run_output = "mixed_poisson_nx_%i_ny_%i_nz_%i_order_%i_ws_%i_ts_%i_vs_%i_ns_%i.out" % (nx, ny, nz , order, ws, ts, vs, ns)
            if args.prefix:
                filename = args.prefix+filename
                run_output = args.prefix+run_output
            command = executable+" --x-elements=%i --y-elements=%i --z-elements=%i --hgrad-basis-order=%i --hdiv-basis-order=%i --workset-size=%i %s --no-check-order --stacked-timer-filename=%s" % (nx, ny, nz, order, order, ws, shared_mem_flag, filename)  +" >& "+run_output

            if args.run:
                #print 'generating data...'
                if args.verbose:
                    print("  Running \""+command+"\" ...", end=' ')
                    sys.stdout.flush()
                os.system(command);
                if args.verbose:
                    print("completed!")
                    sys.stdout.flush()

            if args.analyze:
                f = open(filename, mode='r')
                lines = f.readlines()
                for line in lines:
                    if args.verbose:
                        print(line, end=' ')
                    for key,value in timings.items():
                        if key in line:
                            split_line = line.split()
                            timings[key][i] += float(split_line[-4])
                            if args.verbose:
                                print("  found key: "+key+" = "+str(split_line[-4]))
                            break
                f.close()

    # divide by number of samples to average timings
    for key,value in timings.items():
        for i in range(len(timings[key])):
            timings[key][i] /= num_samples

    if args.analyze:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.semilogy()
        # maroon = #990033, light blue = #00ffff
        #plt.plot(workset_range,timings["Jacobian Evaluation Time <<Host DAG>>"],label="Jac Total Time (Host DAG)",marker="o",color="#990033",markersize=8)
        #plt.plot(workset_range,timings["Jacobian Evaluation Time <<Device DAG>>"],label="Jac Total Time (Device DAG)",marker="s",color="r",markersize=8)
        #plt.plot(workset_range,timings["Residual Evaluation Time <<Host DAG>>"],label="Res Total Time (Host DAG)",marker="o",color="b",markersize=8)
        plt.plot(workset_range,timings["panzer::AssemblyEngine::evaluate_volume(panzer::Traits::Jacobian)"],label="Jacobian Volume Assembly Total Time",marker="s",color="#00ffff",markersize=8)
        plt.xlabel("Workset Size",fontsize=16)
        plt.ylabel("Time (s)",fontsize=16)
        plt.tick_params(labelsize=16)
        title = "nel=%i,order=%i" % (nx*ny*nz,order)
        plt.title(title)
        #plt.legend(bbox_to_anchor=(1,1))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.0),ncol=2,fancybox=True,shadow=True, prop={'size': 12})
        plt.grid()
        dag_timings_filename = "total_time_nx_%i_ny_%i_nz_%i_order_%i_ts_%i_vs_%i.png" % (nx, ny, nz ,order, ts, vs)
        fig.savefig(dag_timings_filename)
        #plt.show()

        fig = plt.figure(2)
        #plt.clf()
        plt.semilogy()
        plt.plot(workset_range,timings["[panzer::Traits::Jacobian] Integrator_DivBasisTimesScalar (EVALUATES):  RESIDUAL_GRADPHI_FIELD"],label="Integrate HDiv Diffusion Operator",marker='s',markersize=8,linewidth=2)
        plt.plot(workset_range,timings["[panzer::Traits::Jacobian] Integrator_DivBasisTimesScalar (CONTRIBUTES):  RESIDUAL_GRADPHI_FIELD"],label="Integrate HDiv Source Operator",marker='^',markersize=8,linewidth=2)
        plt.plot(workset_range,timings["[panzer::Traits::Jacobian] Integrator_GradBasisDotVector (EVALUATES):  RESIDUAL_PHI_MASS_OP"],label="Integrate HGrad Mass Operator",marker='*',markersize=12,linewidth=4)
        plt.plot(workset_range,timings["[panzer::Traits::Jacobian] Integrator_GradBasisDotVector (EVALUATES):  RESIDUAL_PHI_DIFFUSION_OP"],label="Integrate HGrad Diffusion Operator",marker='D',markersize=4,linewidth=2)
        plt.plot(workset_range,timings["[panzer::Traits::Jacobian] DOF: GRADPHI_FIELD accel_jac  (panzer::Traits::Jacobian)"],label="Field Evaluation: PHI",marker='+',markersize=8,linewidth=2)
        plt.plot(workset_range,timings["[panzer::Traits::Jacobian] DOFGradient: GRAD_PHI (panzer::Traits::Jacobian)"],label="Field Evaluation: Grad PHI",marker='x',markersize=8,linewidth=2)
        #plt.plot(workset_range,timings["[panzer::Traits::Jacobian] DOFDiv: DIV_GRADPHI_FIELD (panzer::Traits::Jacobian)"],label="DOF Div (GradPhi)",marker='o')
        #plt.plot(workset_range,timings[""],label="Res Scatter",marker='.',color="#ff6600")
        plt.xlabel("Workset Size",fontsize=16,fontweight='bold')
        plt.ylabel("Time (s)",fontsize=16,fontweight='bold')
        plt.tick_params(labelsize=16)
        plt.rcParams["font.weight"] = 'bold'
        plt.rcParams["axes.labelweight"] = 'bold'
        plt.ylim(1.0e-4,1.0e1)
        #title = "Flat"
        #title = "Hierarchic No Shared Memory"
        #title = "Hierarchic with Shared Memory"
        #plt.title(title,fontsize=20,fontweight='bold')
        #plt.legend(bbox_to_anchor=(1,1))
        #plt.legend(loc='lower left', bbox_to_anchor=(0.05,0.05),ncol=1,fancybox=True,shadow=True, prop={'size': 12})
        plt.legend(loc='upper center', bbox_to_anchor=(0.05,0.05),ncol=1,fancybox=True,shadow=True, prop={'size': 12})
        #plt.axis([0,2000,1.0e-4,0.1])
        plt.grid()
        res_evaluator_timings_filename = "kernel_timings_nx_%i_ny_%i_nz_%i_order_%i_ts_%i_vs_%i.png" % (nx, ny, nz, order, ts, vs)
        fig.savefig(res_evaluator_timings_filename)

        #print dir(plt)

        # Plot to assess savings
        filename_f = "raw_data_output_timer_nx_%i_ny_%i_nz_%i_order_%i_ws_%i_ts_%i_vs_%i.csv" % (nx, ny, nz, order, ws, ts, vs)
        write_file = open(filename_f,'w')

        output_lines = []
        output_lines.append("Workset Size")
        for i in workset_range:
            output_lines.append("%i" % i)

        print(output_lines)

        for key,value in timings.items():
            output_lines[0] = str(output_lines[0]+", "+key)
            print(output_lines[0])
            for i in range(len(workset_range)):
                output_lines[i+1] = output_lines[i+1]+(", %e" % value[i])

        for line in output_lines:
            write_file.write(line+"\n")

    print("Finished Workset Analysis")

    if args.verbose:
        print(timings)


        # f = open(filename, mode='r')
        # lines = f.readlines()
        # for line in lines:
        #     print line,
        #     split_line = line.split(" ")
        #     print split_line[1]
        # f.close()



    #os.chdir('/Users')

    # Write timestamp for backup
    #os.chdir('/Users/rppawlo')
    #timestamp_file = open('BACKUP_DATE', 'w')
    #today = datetime.datetime.today()
    #date = today.strftime("YYYY.MM.DD: %Y.%m.%d at HH.MM.SS: %H.%M.%S")
    #timestamp_file.write(date)
    #timestamp_file.close()

    print('****************************************')
    print('* Finished Panzer Analysis!')
    print('****************************************')


#############################################################################
# If called from the command line, call main()
#############################################################################

if __name__ == "__main__":
    main()
