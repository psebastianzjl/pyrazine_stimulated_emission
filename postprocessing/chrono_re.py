#!/usr/bin/env python3
import os, shutil
import numpy as np
import sys
import subprocess

#os.system("if [ `tail -n 1 diffus.dat | awk '{print $1}'` == '0.000000' ];then sed -i -e \"$ s/0.000000/`grep 'Starting step:' ../nad.out | tail -n 1 | cut -d '=' -f 2 | awk '{print $1}'`/\"  diffus.dat;fi")

###################################
#	control parameters
##################################
cwd = os.getcwd()
diff_color = False
if len(sys.argv) >= 3:
    plot = 0
else:
    plot = 1
try:
    int(sys.argv[1])
except:
    print("Problem by identifying atom number from trajectory.xyz")
    exit(0)
atom = int(sys.argv[1])
print('Atom number: ', atom)
TRAJ = int(sys.argv[2])
counter = 0


#pair = [[5, atom - 1], [11, atom]]
#pair = [[11, 17], [1, 18]]
#pair = [[1,7],[3,4]]
#if os.path.isfile("../../pairs"):
#    pairsfile = "../../pairs"
#elif os.path.isfile("../../../pairs"):
#    pairsfile = "../../../pairs"
#else:
#    print("pairsfile missing")
#    exit
#with open(pairsfile, "r") as f:
#    pair = np.loadtxt(f, dtype=np.int)
#pairfile = ["a" + str(pair[0][0]) + "-a" + str(pair[0][1]) + "_dist.dat",
#            "a" + str(pair[1][0]) + "-a" + str(pair[1][1]) + "_dist.dat"
#           ]
#dd = 1.3
#dE = 0.2
#endtime = 200

arrow = 0
arrows = [] #crossing time
CItime = 0
CItime_index = 0
CIdis = [0, 0]
CIen = 0
raw = ["energy.dat",
#       "dipole.dat",
       "trajectory.xyz",
       #"geometry",
       #"velocity",
       #"oscill.dat"
#       "diffus.dat"
      ]
block = np.array([[1, 1, 1], # [block size, time row, time position]
#                  [3, 3, 1],
                  [atom + 2, 2, 2],
                 # [atom, 1, 1],
                 # [atom + 1, 1, 1],
                 # [1, 1, 1],
#                  [1, 1, 1]
                 ])
Ha2eV = 27.211386245988
#activity = [2, 3] # separate activity for two pairs
####################################
#	chronological ordering
####################################
def reorder(infile, outfile, block=1, location=1, position=1):	#block: row number of a unity; location: time record row number
    suf = block - location
    pre = 0
    seudopre = 0
    x = 0
    with open(infile, "r") as inf:
        reader, times = [], []
        lines = list(line for line in (l.strip().upper() for l in inf) if line)
    i = 0
    #check starting point for time = 0
    for (ip, preline) in enumerate(lines):
        try:
            met = float(preline.split()[position - 1])
        except:
            met = -1
            continue
        if met == 0:
            #print(preline.split()[0], infile, ip, preline)
            if ip != len(lines) - 1:
                i = ip - location + 1
    #print(i)
    while i <= (len(lines) - location):
        if location != 1:
            #load prefix
            pre = location - 1
            seudopre = 0
            for _ in range(location - 1):
                if "dipole" not in infile:
                    reader.append(lines[i])
                elif "->" not in lines[i] and "->" not in lines[i - 1]: #consecutive time lines due to extra printing in dipole.dat (no more)
                    print("consecutive timelines in", infile)
                    lines.insert(i, 'blank line')
                    seudopre += 1
                    reader.append('                            ' + lines[i])
                elif "->" in lines[i] and "->" in lines[i + 1]: #consecutive dipole lines in dipole.dat (no more)
                    print("consecutive dipolelines in", infile)
                    lines.pop(i+1)
                else:
                    reader.append('          ' + lines[i])
                i += 1
        time = float(lines[i].split()[position - 1])
        #skipping rewinding part
        if len(times) > 2:
            #print(time, infile)
            for y in range(3): #maximal 2 rewind steps [-3, -2, -1] == time
                if time <= float(times [y - 3]):
                    #print('Time:', time)
                    if len(times) >  3 and time <= float(times[-4]):
                        print("input data wrong for", infile, "line", i)
                        exit(0)
                    x = -y + 3
                    re_found = True
                    break
                else:
                    re_found = False
            if re_found:
                times = times[:-x]
                reader = reader[:-x * block - pre + seudopre] + reader[-pre + seudopre:] if pre - seudopre > 0 else reader[:-x * block - pre + seudopre]
        elif len(times) == 2:
            for y in range(2): #maximal 1 rewind steps [-2, -1] == time
                if time <= float(times [y - 2]):
                    x = -y + 2
                    re_found = True
                    break
                else:
                    re_found = False
            if re_found:
                times = times[:-x]
                reader = reader[:-x * block - pre + seudopre] + reader[-pre + seudopre:] if pre - seudopre > 0 else reader[:-x * block - pre + seudopre]
    #record non-rewinding time
        times.append(time)
        reader.append(lines[i])
        if suf >= 1:
            for _ in range(suf):
                i += 1
                reader.append(lines[i])
        i += 1
    with open(outfile, 'w') as outf:
        for item in reader:
            outf.write("%s\n" % item)
    #checl chronology
    for it in range(len(times)):
        if it >= 1 and times[it] < times[it - 1]:
            print("chrono error at ", times[it])
            print(times)
            break

entzogen=["chr_" + ent for ent in raw]
#raw = ["./Results/" + r for r in raw]

for itraj in range(TRAJ):
    trajNo = itraj + 1 
    #exfile = 'ex/TRAJ' + str(trajNo) + '/dynamics.out' #excited state dynamics output  ### ORIGINAL ###
    exfile = './TRAJ' + str(trajNo) + '/Results/chr_energy.dat' #excited state dynamics output
    dynout = './TRAJ' + str(trajNo) + '/nad.out' 
    if os.path.isfile(dynout):
        if os.path.isfile(exfile):
            print('Chrono:', 'traj =', trajNo, 'already processed')
            counter += 1
            continue
        else:
            with open(dynout,'r') as infile:
                lines = infile.readlines()
                for line in lines:
                    #if line.strip().startswith("nstate "):
                    #    nstate = int(line.split()[-1])
                    #if 'time =' in line:
                    #    endtime = int(float(line.split()[-1]))
                    #    print("endtime=", endtime)
                    if str('Reached max_time, ending calculation.') in line: 
                        os.chdir('TRAJ' + str(trajNo) + '/Results/')
                        for i in range(len(raw)):
                            reorder(raw[i], entzogen[i], block[i, 0], block[i, 1], block[i, 2])
                        print('Chrono:', 'traj =', trajNo, 'now processed')
                        counter += 1
                        os.chdir(cwd)
                        continue
    else:
        print('Chrono:', 'traj =', trajNo, 'not yet finished')
    
print('Num of finished trajectories:', counter)

######################################
#        dE with MP2 energy
######################################
#def CIx():
#    global CItime, CItime_index, CIstate, dE, CIen
#    with open(entzogen[0], "r") as f: #energy file
#        lines = f.readlines()
#    for (il, line) in enumerate(lines):
#        if float(line.split()[3]) - float(line.split()[4]) <= dE / Ha2eV:
#            CItime = float(line.split()[0])
#            CItime_index = il
#            CIstate = int(line.split()[1])
#            CIen = abs(float(line.split()[3]) - float(line.split()[4]))
#            if CIstate == 1:
#                laststate = int(lines[il - 1].split()[1]) + 3
#                CIen = abs(float(line.split()[laststate]) - float(line.split()[3]))
#            CIen *= Ha2eV
#            break
#
#CIx()
#######################################
##        NH distance
#######################################
## atom1 < atom2
## distance = [time, distance, state]
#def pairdistance(atom1, atom2, atom, i):
#    global arrow, dd, CIdis, CItime, CItime_index, pair, activity
#    atom += 2
#    line1 = atom1 + 2
#    line2 = atom2 + 2
#    coord1 = np.zeros([3])
#    coord2 = np.zeros([3])
#    with open(entzogen[2], "r") as f: #chr_trajectory.xyz
#        lines = f.readlines()
#    distance = np.zeros([len(lines) // atom, 3]) #[time, dis, cstate]
#    N = -1
#    count = 0
#    for (il, line) in enumerate(lines):
#        if il % atom == 0:
#            N += 1
#        if il % atom == 1:
#            distance[N, 0] = line.split()[1] #current time
#            distance[N, 2] = line.split()[-1] #current state
#        if il % atom == line1 - 1:
#            for di in range(3):
#                coord1[di] = float(line.split()[di+1])
#        if il % atom == line2 - 1:
#            for di in range(3):
#                coord2[di] = float(line.split()[di+1])
#        if il % atom == atom - 1:
#            distance[N, 1] = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)
#            if count == 0 and distance[N, 1] > dd:
#                arrow = distance[N, 0]
#                count += 1
##found CI
#            if float(distance[N, 0]) - float(CItime) == 0 and CItime != 0: #CI line
#                CIdis[i] = distance[N, 1]
#                with open('ci.xyz', 'w') as f:
#                    for ciline in range(atom):
#                        f.write(lines[il + ciline - atom + 1])
#                if activity[i] >= 2:
#                    if CIdis[i] == np.amax(distance[:CItime_index + 1, 1]) and CIdis[i] >= dd:
#                        print('found maximum at CItime, active for', pair[i])
#                        activity[i] = 1
#                    elif CIdis[i] == np.amax(distance[:CItime_index + 1, 1]) and CIdis[i] < dd:
#                        print('found quasi maximum at CItime, active for', pair[i])
#                        activity[i] = 0
#                    elif abs(CIdis[i] - np.amax(distance[:CItime_index + 1, 1])) <= 0.01:
#                        print('found quasi maximum (<= 0.01) at CItime', end='')
#                        if CIdis[i] >= dd:
#                            print('larger than', dd,' active for', pair[i])
#                            activity[i] = 0
#                        else:
#                            print('smaller than', dd,' inactive for', pair[i])
#                            activity[i] = 2
#                    elif abs(CIdis[i] - np.amax(distance[:CItime_index + 1, 1])) > 0.01:
#                        if CIdis[i] >= dd:
#                            print('no maximum distance at CItime but larger than ', dd,', active for', pair[i])
#                            activity[i] = 0
#                        else:
#                            print('no maximum distance at CItime, inactive for', pair[i])
#                            activity[i] = 2
#                            if any(x >= dd for x in distance[:CItime_index + 1, 1]):
#                                print('back transfer!')
#                                activity[i] = -1
##no CI
#    if CItime == 0 and activity[i] >= 2:
#        if distance[-1, 1] == np.amax(distance[:, 1]):
#            activity[i] = 1
#        elif abs(distance[-1, 1] - np.amax(distance[:, 1])) <= 0.01:
#            activity[i] = 0
#        elif abs(distance[-1, 1] - np.amax(distance[:, 1])) > 0.01:
#            activity[i] = 2
#            if any(x >= dd for x in distance[:, 1]):
#                print('back transfer!')
#                activity[i] = -1
#    return distance
#
######################################
##	pairing distance
######################################
##lastdis = []
##for i in range(2):
##    distances = pairdistance(pair[i][0], pair[i][1], atom, i)
##    lastdis.append(distances[-1, 1])
##    if arrow != 0: #crossing time
##        arrows.append("set arrow from '" + str(arrow) + " + ', graph 0 to '" + str(arrow) + "', graph 2 nohead lc rgb \'red\';")
##        arrow = 0
##    else:
##        arrows.append("")
##    with open(pairfile[i], "w") as f:
##        for dist in distances:
##            [f.write(" %6.10f " % d) for d in dist]
##            f.write("\n")
##lasttime = distances[-1, 0]
##laststate = int(distances[-1, -1])
######################################
##	three points angle
######################################
#def bendangle(atom1, atom2, atom3, atom):
#    atom += 2
#    line1 = atom1 + 2
#    line2 = atom2 + 2
#    line3 = atom3 + 2
#    coord1 = np.zeros([3])
#    coord2 = np.zeros([3])
#    coord3 = np.zeros([3])
#    with open(entzogen[2], "r") as f: #chr_trajectory.xyz
#        lines = f.readlines()
#    angle = np.zeros([len(lines) // atom, 3]) #[time, dis, cstate]
#    N = -1
#    count = 0
#    for (il, line) in enumerate(lines):
#        if il % atom == 0:
#            N += 1
#        if il % atom == 1:
#            angle[N, 0] = line.split()[1] #current time
#            angle[N, 2] = line.split()[-1] #current state
#        if il % atom == line1 - 1:
#            for di in range(3):
#                coord1[di] = float(line.split()[di+1])
#        if il % atom == line2 - 1:
#            for di in range(3):
#                coord2[di] = float(line.split()[di+1])
#        if il % atom == line3 - 1:
#            for di in range(3):
#                coord3[di] = float(line.split()[di+1])
#        if il % atom == atom - 1:
#            ba = coord1 - coord2
#            bc = coord3 - coord2
#            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#            angle[N, 1] = np.degrees(np.arccos(cosine_angle))
#    return angle
#
#if os.path.isfile('../../angles'):
#    with open('../../angles', 'r'):
#        angles = np.loadtxt('../../angles', dtype=np.int)
#    bendangles = bendangle(angles[0], angles[1], angles[2], atom)
#    with open('bendangle', "w") as f:
#        for ang in bendangles:
#            [f.write(" %6.10f " % d) for d in ang]
#            f.write("\n")
######################################
##	transforming color
######################################
##if diff_color:
##    colorfile = 'chr_diffus.dat'
##else:
##    shutil.copyfile('chr_dipole.dat', 'chr_dipolecolor.dat')
##    colorfile = 'chr_dipolecolor.dat'
##    with open(colorfile, 'r') as f:
##        lines = f.readlines()
##    with open(colorfile, 'w') as f:
##        for (il, line) in enumerate(lines):
##            if il%2 == 1:
##                f.write(line + '\n')
#with open("chr_energy.dat", "r") as inf:
#    enlines = list(line for line in (l.strip() for l in inf) if line)
##with open(colorfile, "r") as inf:
##    difflines = list(line for line in (l.strip() for l in inf) if line)
##if len(enlines) != len(difflines):
##    print("mismatch between energy and diffusity files:", len(enlines), len(difflines))
##    exit(0)
##with open("chr_energy.dat", "w") as f:
##    for i,line in enumerate(enlines):
##        f.write(line)
##        for j in difflines[i].split()[2:]:
##            f.write('    ' + j)
##        f.write('\n')
######################################
##	plotting
######################################
## check completeness
#if CItime != 0:
#    CItimestr = "set arrow from '" + str(CItime) + "', graph 0 to '" + str(CItime) + "', graph 2 nohead lc rgb \'blue\';"
#    if CItime > endtime:
#        print("{name:12s}: {result:12s}".format(name='Propagation', result='completed'))
#        print('{name:12s}: {result:>11s}{laststate:1d}'.format(name='Decay', result='last state= ', laststate=laststate))
#        print('Longer time needed for relaxtion, Type: C')
#    else:
#        print("{name:12s}: {result:12s}".format(name='Propagation', result='completed'))
#        print('{name:12s}: {result:12s}'.format(name='Decay', result='ground state'))
#        if CIstate == 1:
#            print('{name:>12s}: {result:12s} with S{CIstate:2d} at {CItime:5.1f} fs for gap {CIen:3.2f} eV'.format(name='channel', result='hopping', CIstate=CIstate-1, CItime=CItime, CIen=CIen))
#        else:
#            print('{name:>12s}: {result:12s} with S{CIstate:2d} at {CItime:5.1f} fs for gap {CIen:3.2f} eV'.format(name='channel', result='CI', CIstate=CIstate-1, CItime=CItime, CIen=CIen))
#        print('{name:12s}: '.format(name='Activity'))
#        for i in range(len(pair)):
#            print('{0:14s}{result} at {CIdis}'.format('', result=pair[i], CIdis=CIdis[i]), end=': ')
#            if activity[i] == 3:
#                print('Error by evaluating!')
#                raise AssertionError 
#            elif activity[i] == 1:
#                print('ACTIVE! Type: A1')
#            elif activity[i] == 0:
#                if os.path.isfile('h' + str(i+1)):
#                    with open('h' + str(i+1), 'r') as f:
##                if os.path.isfile('human.dat'):
##                    with open('human.dat', 'r') as f:
#                        print("human assessed", end=', ')
#                        for line in f:
#                            if line.startswith('Type'):
#                                print(line)
#                else:
#                    print('Human judgement needed! Type: A0!')
#            elif activity[i] == 2:
#                print('inactive! Type: A2')
#            elif activity[i] == -1:
#                print('back transfer >>> inactive! Type: A2')
#else:
#    CItimestr = ""
#    if lasttime >= endtime:
#        print('{name:12s}: {result:12s}'.format(name='Propagation', result='completed'))
#        if laststate == 1:
#            print('{name:12s}: {result:12s}'.format(name='Decay', result='ground state'))
#            print('{name:>12s}: {result:12s} at {time:5.1f}fs'.format(name='channel', result='hopping', time=lasttime))
#            print('{name:12s}: '.format(name='Activity'))
#            print('{0:14s}{result} at {CIdis}'.format('', result=pair[i], CIdis=lastdis[i]), end=': ')
#            if activity[i] == 3:
#                print('Error by evaluating!')
#                raise
#            elif activity[i] == 1:
#                print('ACTIVE! Type: A1')
#            elif activity[i] == 0:
#                #if os.path.isfile('./human.dat'):
#                #    with open('human.dat', 'r') as f:
#                if os.path.isfile('h' + str(i+1)):
#                    with open('h' + str(i+1), 'r') as f:
#                        for line in f:
#                            if line.startswith('Type'):
#                                print(line)
#                else:
#                    print('Human judgement needed! Type: A0!')
#            elif activity[i] == 2:
#                print('inactive! Type: A2')
#            elif activity[i] == -1:
#                print('back transfer >>> inactive! Type: A2')
#        else:
#            print('{name:12s}: {result:>11s}{laststate:1d}'.format(name='Decay', result='last state= ', laststate=laststate))
#            print('Longer time needed for relaxtion, Type: C')
#    else:
#        print('{name:12s}: {result:12s}'.format(name='Propagation', result='incomplete'))
#        if os.path.isfile('human.dat'):
#            with open('human.dat', 'r') as f:
#                for line in f:
#                    if line.startswith('Type'):
#                        print(line)
#        else:
#            print('Failed to propagate to the specified time, Type: E')
#if plot:
#    trajno = os.getcwd().split('/')[-2]
#    if not os.path.isfile('bendangle'):
#        os.system('gnuplot -p -e "set terminal pdf font \'sans,6\';\
#set output \'en.pdf\';\
#set multiplot layout 3,1 margin 0.05,0.9,0.08,.99 ;\
#set autoscale xmax;\
#set bmargin at screen 0.825;\
#unset xtics;\
#set ytics 0.1;\
#set y2tics 0.1;\
#set autoscale xfixmax;\
#plot \'' + pairfile[1] + '\' u 1:2 w lp lt -1, '+str(dd)+' lc rgb \'red\';\
#unset y2tics;\
#set ytics 0.1 offset 0;\
#set bmargin at screen 0.3;\
#set tmargin at screen 0.8;\
#' + arrows[0] + '\
#' + arrows[1] + '\
#' + CItimestr + '\
#set cbtics (\'diffusity\' 100) scale 0;\
#set cbrange [20:80];\
#set palette model CMY rgbformulae 7,5,15;\
#set colorbox user origin 0.925, 0.35 size 0.01,0.4;\
#plot for [i=6:' + str(nstate + 4) + '] \'chr_energy.dat\' u 1:((column(i)==0) ? 1/0 : column(i)):(column(i+' + str(nstate - 1) + ')) w lp lt -1 notitle, \'chr_energy.dat\' u 1:5 w lp lt -1 lc rgb \'black\' notitle, \'chr_energy.dat\' u 1:4 w p ps 0.1 pt 7 lc rgb \'red\' notitle;\
#unset datafile;\
#set title \'' + trajno + '\';\
#set y2tics 0.1;\
#set tmargin at screen 0.275;\
#set xtics;\
#plot \'' + pairfile[0] + '\' u 1:2 w lp lt -1, '+str(dd)+' lc rgb \'red\';\
#unset multiplot"\
#')
##set cbrange [0:20];\
##set palette model CMY rgbformulae 7,5,15;\
#    else:
#        os.system('gnuplot -p -e "set terminal pdf font \'sans,6\';\
#set output \'en.pdf\';\
#set multiplot layout 4,1 margin 0.05,0.9,0.08,.99 ;\
#set autoscale xmax;\
#set bmargin at screen 0.825;\
#unset xtics;\
#set ytics 0.1;\
#set y2tics 0.1;\
#set autoscale xfixmax;\
#plot \'' + pairfile[1] + '\' u 1:2 w lp lt -1, '+str(dd)+' lc rgb \'red\';\
#unset y2tics;\
#set ytics 0.1 offset 0;\
#set bmargin at screen 0.4;\
#set tmargin at screen 0.8;\
#' + arrows[0] + '\
#' + arrows[1] + '\
#' + CItimestr + '\
#set cbtics (\'diffusity\' 100) scale 0;\
#set cbrange [20:80];\
#set palette model CMY rgbformulae 7,5,15;\
#set colorbox user origin 0.925, 0.35 size 0.01,0.4;\
#plot for [i=6:' + str(nstate + 4) + '] \'chr_energy.dat\' u 1:((column(i)==0) ? 1/0 : column(i)):(column(i+' + str(nstate - 1) + ')) w lp palette z lt -1 notitle, \'chr_energy.dat\' u 1:5 w lp lt -1 lc rgb \'black\' notitle, \'chr_energy.dat\' u 1:4 w p ps 0.1 pt 7 lc rgb \'red\' notitle;\
#unset datafile;\
#set title \'' + trajno + '\';\
#set y2tics 0.1;\
#set bmargin at screen 0.200;\
#set tmargin at screen 0.375;\
#plot \'' + pairfile[0] + '\' u 1:2 w lp lt -1, '+str(dd)+' lc rgb \'red\';\
#unset title;\
#set xtics;\
#set ytics auto;\
#set y2tics auto;\
#set yrange [100:180];\
#set tmargin at screen 0.175;\
#set xtics;\
#plot \'bendangle\' u 1:2 w lp lt -1, '+str(dd)+' lc rgb \'red\';\
#unset multiplot"\
#')
#    os.system('evince en.pdf')
