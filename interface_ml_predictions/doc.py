#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! doc: handling help of MLatom                                              ! 
  ! Implementations and documentation by:                                     !
  ! Fuchun Ge, Pavlo O. Dral, Bao-Xin Xue                                     ! 
  !---------------------------------------------------------------------------! 
'''
import time, sys

class Doc():
    @classmethod
    def printDoc(cls,argdict):
        mlatom_alias=['mlatomf','mlatom','kreg']
        gap_alias=[ 'gap', 'gap_fit', 'gapfit']
        sgdml_alias=['sgdml']
        deepmd_alias=['dp','deepmd','deepmd-kit']
        physnet_alias=['physnet']
        ani_alias=['torchani','ani']
        overview=True
        for key, value in argdict.items():
            for k, v in cls.help_text.items():
                if key.lower()==k.lower() and value:  
                    print(v)
                    overview=False
            if key.lower() in ['mlmodeltype','mlprog'] and value:
                if value == True: value='S'
                if value.lower() in mlatom_alias:
                    from interface_MLatomF import printHelp
                    printHelp()
                    overview=False
                elif value.lower() in ani_alias:
                    from interfaces.TorchANI.interface_TorchANI import printHelp
                    printHelp()
                    overview=False
                elif value.lower() in gap_alias:
                    from interfaces.GAP.interface_GAP import printHelp
                    printHelp()
                    overview=False
                elif value.lower() in sgdml_alias:
                    from interfaces.sGDML.interface_sGDML import printHelp
                    printHelp()
                    overview=False
                elif value.lower() in deepmd_alias:
                    from interfaces.DeePMDkit.interface_DeePMDkit import printHelp
                    printHelp()
                    overview=False
                elif value.lower() in physnet_alias:
                    from interfaces.PhysNet.interface_PhysNet import printHelp
                    printHelp()
                    overview=False
                else:
                    print(cls.Default_MLprog_type)
                    overview=False
        if overview:
            print(cls.help_text['overview'])

        print(' %s ' % ('='*78))
        print(time.strftime(" MLatom terminated on %d.%m.%Y at %H:%M:%S", time.localtime()))
        print(' %s ' % ('='*78))
        sys.stdout.flush()
        
    molDescriptorDoc = '''
  Optional arguments specifying descriptor:
    molDescriptor=S            molecular descriptor S
      RE [default]             vector {Req/R}, where R is internuclear distance
      CM                       Coulomb matrix
    molDescrType=S             type S of molecular descriptor
      sorted                   default for molDescrType=CM
                               sort by:
                                 norms of CM matrix (for molDescrType=CM)
                                 nuclear repulsions (for molDescrType=RE)
      unsorted                 default for molDescrType=RE
      permuted                 molecular descriptor with all atom permutations
  
  If molDescrType=sorted requested, additional output can be requested:
    XYZsortedFileOut=S         file S with sorted XYZ coordinates, only works
                               for molDescriptor=RE molDescrType=sorted
  
  If molDescrType=permuted requested, at least one of the arguments needed:
    permInvGroups=S            permutationally invariant groups S
                               e.g. for water dimer permInvGroups=1,2,3-4,5,6
                               permute water molecules (atoms 1,2,3 and 5,6,7)
    permInvNuclei=S            permutationally invariant nuclei S
                               e.g.permInvNuclei=2-3.5-6
                               will permute atoms 2,3 and 6,7
'''

    Default_MLprog_type='''
  Use of interfaces to ML programs
  
  Arguments:
    MLprog=S                   ML program S
      or
    MLmodelType=S              ML model type S

  Supported ML model types and default programs:

  +-------------+----------------+ 
  | MLmodelType | default MLprog | 
  +-------------+----------------+ 
  | KREG        | MLatomF        | 
  +-------------+----------------+ 
  | sGDML       | sGDML          | 
  +-------------+----------- ----+ 
  | GAP-SOAP    | GAP            | 
  +-------------+----------------+ 
  | PhysNet     | PhysNet        | 
  +-------------+----------------+ 
  | DeepPot-SE  | DeePMD-kit     | 
  +-------------+----------------+ 
  | ANI         | TorchANI       | 
  +-------------+----------------+ 
  
  Supported interfaces with default and tested ML model types:
  
  +------------+----------------------+
  | MLprog     | MLmodelType          |
  +------------+----------------------+
  | MLatomF    | KREG [default]       |
  |            | see                  |
  |            | MLatom.py KRR help   |
  +------------+----------------------+
  | sGDML      | sGDML [default]      |
  |            | GDML                 |
  +------------+----------------------+
  | GAP        | GAP-SOAP             |
  +------------+----------------------+
  | PhysNet    | PhysNet              |
  +------------+----------------------+
  | DeePMD-kit | DeepPot-SE [default] |
  |            | DPMD                 |
  +------------+----------------------+
  | TorchANI   | ANI [default]        |
  +------------+----------------------+
  
  See interface's help for more details, e.g.
    MLatom.py MLprog=TorchANI help
'''

    help_text={
    'overview':'''
  Usage:
    MLatom.py [options]

  Getting help:
    MLatom.py help             print this help and exit
    MLatom.py [option] help    print help for [option] and exit
                               e.g. MLatom.py useMLmodel help

  Options:
    ML tasks:
      geomopt                  perform geometry optimizations
      freq                     perform frequency calculations 
      TS                       perform transition state searches
      IRC                      perform reaction path searches
      useMLmodel               use existing ML model    
      createMLmodel            create and save ML model 
      estAccMLmodel            estimate accuracy of ML model
      deltaLearn               use delta-learning
      selfCorrect              use self-correcting ML
      learningCurve            generate learning curve
      crossSection             simulate absorption spectrum
      MLTPA                    simulate two-photon absorption
    
    General purpose methods:
      AIQM1                    perform AIQM1       calculations
      ANI-1ccx                 perform ANI-1ccx    calculations
      ANI-1x                   perform ANI-1x      calculations
      ANI-2x                   perform ANI-2x      calculations
      ANI-1x-D4                perform ANI-1x-D4   calculations
      ANI-2x-D4                perform ANI-2x-D4   calculations
      ODM2
      ODM2*
      GFN2-xTB
      CCSD(T)*/CBS             only single-point calculations

    Data set tasks:
      XYZ2X                    convert XYZ coordinates into descriptor X
      analyze                  analyze data sets
      sample                   sample data points from a data set
      slice                    slice data set
      sampleFromSlices         sample from each slice
      mergeSlices              merge indices from slices

    Molecular dynamics:
      MD                       3D molecular dynamics
      create4Dmodel            create and save 4D model
      use4Dmodel               use existing 4D model

    Multithreading:
      nthreads=N               set number of threads (N)
 
  Example:
    MLatom.py createMLmodel XYZfile=CH3Cl.xyz Yfile=en.dat MLmodelOut=CH3Cl.unf
''',
  'AIQM1':'''
  Run AIQM1 calculations
  Similarly, AIQM1@DFT and AIQM1@DFT* can be requested
  
  Usage: MLatom.py AIQM1 [arguments]

  Arguments:
    Input file options:
      XYZfile=S                file S with XYZ coordinates

    Other options (at least one of these arguments is required):
      YestFile=S               file S with estimated Y values
      YgradEstFile=S           file S with estimated gradients
      YgradXYZestFile=S        file S with estimated XYZ gradients
      geomopt                  perform geometry optimizations
      freq                     perform frequency calculations
      TS                       perform transition state searches
      IRC                      perform reaction path searches
      QMprog=S                 program to calculate QM part of AIQM1
                               (MNDO is default, Sparrow is optional and used when MNDO is not found)
      mndokeywords=S           file S with MNDO keywords, separating by a blank line for each molecules
                               (it is required to set iop=-22 immdp=-1
                               Often, e.g., for geomopt, jop=-2 igeom=1 nsav15=3 are also needed)
      gaussiankeywords=S       keywords used in Gaussian program, 
                               e.g. gaussiankeywords=opt(nomicro,tight)
                               (nomicro keywords is always required for geomopt)
      
  For the geomopt and freq options, see
    MLatom.py freq help
      
  Example:
    MLatom.py AIQM1 XYZfile=opt.xyz YestFile=en.dat
''',
  'geomopt':'''
  Perform geometry optimization
  
  Usage: 
         MLatom.py geomopt usemlmodel mlmodelin=... mlmodeltype=... XYZfile=... [arguments]
         or 
         MLatom.py geomopt AIQM1 XYZfile=... [arguments]

  Arguments:
    Input file options:
      XYZfile=S                file S with XYZ coordinates
    
    Optional input for choosing interfaced program:
      optprog=scipy            use scipy package [default]
      optprog=gaussian         use Gaussian program
      optprog=ASE              use ASE
      optxyz=S                 save optimized geometries in file S [default: optgeoms.xyz]
    The following options only used for ASE program:
        ase.fmax=R                    threshold of maximum force (in eV/A)
                                      [default values: 0.02]
        ase.steps=N                   maximum steps
                                      [default values: 200]
        ase.optimizer=S               optimizer
           LBFGS [default]
           BFGS
      
  Example:
    MLatom.py AIQM1 geomopt XYZfile=opt.xyz optprog=ASE
''',
  'freq':'''
  Perform geometry optimization followed by frequencies
  
  Usage: 
         MLatom.py freq usemlmodel mlmodelin=... mlmodeltype=... XYZfile=... [arguments]
         or 
         MLatom.py freq [AIQM1 or another general-purpose method] XYZfile=... [arguments]

  Arguments:
    Input file options:
      XYZfile=S                file S with XYZ coordinates
    
    Optional input for choosing interfaced program:
      optprog=gaussian         use Gaussian program [default]
      optprog=ASE              use ASE
      when do frequence analysis with ASE, the following options are also required:
      ase.linear=N,...,N            0 for nonlinear molecule, 1 for linear molecule
                                    [default vaules: 0]
      ase.symmetrynumber=N,...,N    rotational symmetry number for each molecule
                                    [default vaules: 1]
      
  Example:
    MLatom.py AIQM1 freq XYZfile=opt.xyz optprog=ASE
''',
  'useMLmodel':'''
  Use existing ML model
  
  Usage: MLatom.py useMLmodel [arguments]

  Arguments:
    Input file options:
      MLmodelIn=S              file S with ML model
      MLprog=S                 only required for third-party programs. See
                               MLatom.py MLprog help
      
      XYZfile=S                file S with XYZ coordinates
          or
      XfileIn=S                file S with input vectors X

    Output file options (at least one of these arguments is required):
      YestFile=S               file S with estimated Y values
      YgradEstFile=S           file S with estimated gradients
      YgradXYZestFile=S        file S with estimated XYZ gradients
      
  Example:
    MLatom.py useMLmodel MLmodelIn=CH3Cl.unf XYZfile=CH3Cl.xyz YestFile=MLen.dat  
''',
  'createMLmodel':'''
  Create and save ML model
  
  Usage: MLatom.py createMLmodel [arguments]

  Required arguments:
    Input files:
      XYZfile=S                file S with XYZ coordinates
          or
      XfileIn=S                file S with input vectors X
      
      Yfile=S                  file S with reference values
         and/or
      YgradXYZfile=S           file S with reference XYZ gradients

    Output files:
      MLmodelOut=S             file S with ML model
    
  Optional output files:
    XfileOut=S                 file S with X values
    XYZsortedFileOut=S         file S with sorted XYZ coordinates
                               only works for
                               molDescrType=RE molDescrType=sorted
    YestFile=S                 file S with estimated Y values
    YgradEstFile=S             file S with estimated gradients
    YgradXYZestFile=S          file S with estimated XYZ gradients
    
  KREG model with unsorted RE descriptor is trained by default
  Default hyperparameters sigma=100.0 and lambda=0.0 are used
    
  To train ML model with third-party program, see:
    MLatom.py MLprog help      to list available interfaces and 
                               default ML models
           and
    MLatom.py MLprog=S help    for specific help of interface S
    
  To train another KRR model with MLatomF, see:
    MLatom.py KRR help

  Any hyperparameters can be optimized with the hyperopt package, see:
    MLatom.py hyperopt help
      
  To optimize hyperparameters, the training set should be split into the
  sub-training and validation sets, see:
    MLatom.py sample help
  Additional optional arguments:
    sampling=user-defined      reads in indices for the sub-training and
                               validation sets from files defined by arguments
      iSubtrainIn=S            file S with indices of sub-training points
      iValidateIn=S            file S with indices of validation points
      iCVoptPrefIn=S           prefix S of files with indices for CVopt
    
  Example:
    MLatom.py createMLmodel XYZfile=CH3Cl.xyz Yfile=en.dat MLmodelOut=CH3Cl.unf
''',
  "estAccMLmodel":'''
  Estimate accuracy of ML model
  
  Usage: MLatom.py estAccMLmodel [arguments]

  estAccMLmodel task estimates accuracy of created ML models, thus it can
  take the same arguments as createMLmodel task, see:
    MLatom.py createMLmodel help
  with exception that MLmodelOut=S argument is optional
  
  To estimate accuracy, the data set should be split into the
  training and validation sets, see:
    MLatom.py sample help
  Additional optional arguments:
    sampling=user-defined      reads in indices for the training and test sets
                               from files defined by arguments
      iTrainIn=S               file S with indices of training points
      iTestIn=S                file S with indices of test points
      iCVtestPrefIn=S          prefix S of files with indices for CVtest
    MLmodelIn=S                file S with ML model
    
  Example:
    MLatom.py estAccMLmodel XYZfile=CH3Cl.xyz Yfile=en.dat
''',
  'deltaLearn':'''
  Use delta-learning
  
  Usage: MLatom.py deltaLearn [arguments]

  Arguments:
      Yb=S                     file with data obtained with baseline method
      Yt=S                     file with data obtained with target   method
      YestT=S                  file with ML estimations of  target   method
      YestFile=S               file with ML corrections to  baseline method

  delta-learning should be used with one of the following tasks, see:
    MLatom.py useMLmodel    help
    MLatom.py createMLmodel help
    MLatom.py estAccMLmodel help

  Example:
    MLatom.py estAccMLmodel deltaLearn XfileIn=x.dat Yb=UHF.dat \\
    Yt=FCI.dat YestT=D-ML.dat YestFile=corr_ML.dat
''',
  'selfCorrect':'''
  Use self-correcting ML
  
  Usage: MLatom.py selfCorrect [arguments]

  Currently works only with four layers and MLatomF
  Self-correction should be used with one of the following tasks, see:
    MLatom.py useMLmodel    help
    MLatom.py createMLmodel help
    MLatom.py estAccMLmodel help

  Example:
    MLatom.py estAccMLmodel selfcorrect XYZfile=xyz.dat Yfile=y.dat 
''',
  "learningCurve":'''
  Generate learning curve
  
  Usage: MLatom.py learningCurve [arguments]

  Required arguments:
      lcNtrains=N,N,N,...,N    training set sizes
  Optional arguments:
      lcNrepeats=N,N,N,...,N   numbers of repeats for each Ntrain
                 or
                =N             number  of repeats for all  Ntrains
                               [3 repeats default]

  Output files in directory learningCurve:
    results.json               JSON database file with all results
    lcy.csv                    CSV  database file with results for values
    lcygradxyz.csv             CSV  database file with results for XYZ gradients
    lctimetrain.csv            CSV  database file with training   timings
    lctimepredict.csv          CSV  database file with prediction timings
    
  learningCurve task also requires arguments used in estAccMLmodel task, see:
    MLatom.py estAccMLmodel help

  Example:
    MLatom.py learningCurve XYZfile=xyz.dat Yfile=en.dat \\ 
    lcNtrains=100,1000,10000 lcNrepeats=32
''',
  'crossSection':'''
  Simulate absorption cross-section using ML-NEA (nuclear ensemble approach)
      
  Usage:
    MLatom.py crossSection [arguments]

  Optional arguments:
    nExcitations=N             number of excited states [3 by default]
    nQMpoints=N                number of QM calculations
                               [determined iteratively by default]
    plotQCNEA                  plot QC-NEA cross section
    deltaQCNEA=float           set broadening parameter of QC-NEA cross section
    plotQCSPC                  plot single point convolution cross section
  
    Advanced arguments:
      nMaxPoints=N             maximum number of QC calculations
                               in the iterative procedure [10000 by default]
      nNEpoints=N              number of nuclear ensemble points
                               [50000 by default]

  Environment settings:
    $NX                        Newton-X environment
    In addition, Gaussian program environment should be setup
  
  Required input and data set files
    gaussian_optfreq.com       input file for optimization and frequency
                               calculations with Gaussian program
    gaussian_ef.com            template input file for excited-state QC
                               calculations with Gaussian program
        or
    eq.xyz                     file with optimized geometry
    nea_geoms.xyz              file with all geometries in nuclear ensemble
    
  Optional input data files:
    E[i].dat                   files with excitation energies  for excitation i
    f[i].dat                   files with oscillator strengths for excitation i
    cross-section_ref.dat      reference cross section spectrum
  
  Output files in directory cross-section:
    cross-section_ml-nea.dat   ML-NEA cross section
    cross-section_qc-nea.dat   QC-NEA cross section
    cross-section_spc.dat      single point convolution cross section
    plot.png                   cross section plots
''',
  'MLTPA':'''
  Simulate two-photon absorption cross-section with ML (ML-TPA approach)
      
  Usage:
    MLatom.py MLTPA [arguments]

  Input arguments:
    Input file options:
    SMILESfile=S               file S with SMILES
    auxfile=S                  optional file S with 
                               the information of wavelength and Et30 in the format of
                               'wavelength_lowbound,wavelength_upbound,Et30']
                               (wavelength in nm.)
                               One line per SMILES string.
                               Default value of Et30 will be 33.9 (toluene) and
                               the whole spectra between 600-1100 nm will be output.

  After the calculations finish, the predicted TPA cross section values are saved
  in a folder named 'tpa+{absolute time}'.
  The folder will contain files tpa[sequential molecular number].txt with predicted
  TPA cross section values.

  Example:
    MLatom.py MLTPA SMILESfile=Smiles.csv auxfile=_aux.txt
''',
  'XYZ2X':('''
  Convert XYZ coordinates into descriptor X
  
  Usage:
    MLatom.py XYZ2X [arguments]

  Required arguments:
    MLmodelIn=S                file S with ML model
    XYZfile=S                  file S with XYZ coordinates
    XfileOut=S                 file S with X values
''' +
    molDescriptorDoc
    +
'''
  Example:
    MLatom.py XYZ2X XYZfile=CH3Cl.xyz XfileOut=CH3Cl.x
'''),
  'analyze':'''
  Analyze data sets
  
  Usage: MLatom.py analyze [arguments]

  Arguments:
    For reference data (at least one of these arguments is required):
      Yfile=S                  file S with values
      Ygrad=S                  file S with gradients
      YgradXYZfile=S           file S with gradients in XYZ coordinates

    For estimated data:
      YestFile=S               file S with estimated Y values
      YgradEstFile=S           file S with estimated gradients
      YgradXYZestFile=S        file S with estimated XYZ gradients
    
  Example: 
    MLatom.py analyze Yfile=en.dat YestFile=enest.dat
''',
  'sample':'''
  Sample data points from a data set
  
  Usage:
    MLatom.py sample [arguments]

  Required arguments:
    Data set file
      XYZfile=S                file S with XYZ coordinates
          or
      XfileIn=S                file S with input vectors X
    
    Splitting arguments (at least one is required):
      Splitting the data set into the sub-sets:
        iTrainOut=S            file S with indices of training points
        iTestOut=S             file S with indices of test points
        iSubtrainOut=S         file S with indices of sub-training points
        iValidateOut=S         file S with indices of validation points
        
      Cross-validation for testing:
        CVtest                 CV task with optional arguments:
          NcvTestFolds=N       sets number of folds to N [5 by default]
          LOOtest              leave-one-out cross-validation
          iCVtestPrefOut=S     prefix S of files with indices for CVtest
      Cross-validation for hyperparameter optimization:
        CVopt                  CV task with optional arguments:
          NcvOptFolds=N        sets number of folds to N [5 by default]
          LOOopt               leave-one-out cross-validation
          iCVoptPrefOut=S      prefix S of files with indices for CVopt

  Optional arguments:
    sampling=S                 type S of data set sampling into splits
      random [default]         random sampling
      none                     simply split unshuffled data set into
                               the training and test sets (in this order)
                               (and sub-training and validation sets)
      structure-based          structure-based sampling
      farthest-point           farthest-point traversal iterative procedure
    Nuse=N                     N first entries of the data set file to be used
    Ntrain=R                   number of the training points
                               [0.8, i.e. 80% of the data set, by default]
    Ntest=R                    number of the test points
                               [remainder of data set, by default]
    Nsubtrain=R                number of the sub-training points
                               [0.8,    80% of the training set, by default]
    Nvalidate=R                number of the validation points
                               [remainder of the training set, by default]
  Note: Number of indices R can either be positive integer or a fraction of 1.
        R entries for integer R >= 1
        fraction of the entire data set for 0<R<1
  
  Example:
    MLatom.py sample sampling=structure-based XYZfile=CH3Cl.xyz Ntrain=1000 \\
    Ntest=10000 iTrainOut=itrain.dat iTestOut=itest.dat
''',
  'slice':'''
  Slice data set
  
  Usage:
    MLatom.py slice [arguments]
  
  Required arguments:
    XfileIn=S                  file S with input vectors X
    eqXfileIn=S                file S with input vector for equilibrium geometry
  
  Optional arguments:
    Nslices=N                  number of slices [3 by default]
  
  Example:
    MLatom.py slice Nslices=3 XfileIn=x_sorted.dat eqXfileIn=eq.x

''',
  'sampleFromSlices':'''
  Sample from each slice
  
  Usage:
    MLatom.py sampleFromSlices [arguments]
  
  Required argument:
    Ntrain=N                   total integer number N of training points
                               from all slices
  
  Optional argument:
    Nslices=N                  number of slices [3 by default]

  Example:
    MLatom.py sampleFromSlices Nslices=3 sampling=structure-based Ntrain=4480
''',
  'mergeSlices':'''
  Merge indices from slices
  
  Usage:
    MLatom.py mergeSlices [arguments]
  
  Required argument:
    Ntrain=N                   total integer number N of training points
                               from all slices
  
  Optional argument:
    Nslices=N                  number of slices [3 by default]

  Example:
    MLatom.py mlatom mergeSlices Nslices=3 Ntrain=4480
''',
  "KRR":'''
  Kernel ridge regression (KRR) calculations with MLatomF

  Optional arguments:  
    kernel=S                   kernel function type S
      Gaussian [default]
        periodKernel           periodic kernel
        decayKernel            decaying periodic kernel
      Laplacian
      exponential
      Matern
    permInvKernel              permutationally invariant kernel, sub-option:
      Nperm=N                  number of permutations N (with XfileIn argument)
      molDescrType=permuted    (with XYZfile argument, see below)
    lambda=R                   regularization hyperparameter R [0 by default]
      opt                      requests optimization of lambda on a log grid
        NlgLambda=N            N points on a logarithmic grid
                               [6 by default]
        lgLambdaL=R            lowest  value of log2(lambda) [-16.0 by default]
        lgLambdaH=R            highest value of log2(lambda) [ -6.0 by default]
    sigma=S                    length scale
                               [default values: 100 (Gaussian  & Matern)
                                                800 (Laplacian & exponential)]
      opt                      requests optimization of sigma on a log grid
        NlgSigma=N             N points on a logarithmic grid
                               [11 by default]
        lgSigmaL=R             lowest  value of log2(lambda)
                               [default values:  2.0 (Gaussian  & Matern)
                                                 5.0 (Laplacian & exponential)]
        lgSigmaH=R             highest value of log2(lambda)
                               [default values:  9.0 (Gaussian  & Matern)
                                                12.0 (Laplacian & exponential)]
    period                     period in periodic and decayed periodic kernel
                               [1.0 by default]
    sigmap=R                   length scale for a periodic part
                               in decayed periodic kernel
                               [100.0 by default]
    matDecomp=S                type of matrix decomposition
      Cholesky [default]
      LU                   
      Bunch-Kaufman   
    invMatrix                  invert matrix
    refine                     refine solution matrix
    on-the-fly                 on-the-fly calculation of kernel
                               matrix elements for validation,
                               by default it is false and those
                               elements are stored
    benchmark                  additional output for benchmarking
    debug                      additional output for debugging
    
  Additional arguments for hyperparameter optimization on a log grid:
    minimizeError=S            type S of error to minimize
      RMSE                     root-mean-square error [default]
      MAE                      mean absolute error
   lgOptDepth=N                depth of log grid optimization N [3 by default]
''' +
    molDescriptorDoc +
'''
  Example:
    MLatom.py createMLmodel XYZfile=CH3Cl.xyz Yfile=en.dat \\
              MLmodelOut=CH3Cl.unf sigma=opt kernel=Matern
''',
  "hyperopt":'''
  Hyperparameter optimization with the hyperopt package
  For now, only Tree-Structured Parzen Estimator algorithm is supported
  
  Usage: substitute numeric value(s) with hyperopt.xx()
  
  Arguments for hyperopt.xx():
    hyperopt.uniform(lb,ub)    linear search space form
    hyperopt.loguniform(lb,ub) logarithmic search space, base 2
    hyperopt.quniform(lb,ub,q) discrete linear space, rounded by q
  lb is lower bound, ub is upper bound

  Arguments:
      hyperopt.max_evals=N     max number of search attempts [8 by default]
      hyperopt.losstype=S      type of loss used in optimization
        geomean [default]      geometric mean
        weighted               weight for gradients, defined by
          hyperopt.w_grad=R    [0.1 by default]
      hyperopt.points_to_evaluate=[xx,xx,...],[xx,xx,...],...
                               specify initial guesses before auto-searching,
                               each point inside a pair of square brackets
                               should have all values to be optimized in order.
                               these evaluations are NOT counted in max_evals.

  Example:
    MLatom.py estAccMLmodel XYZfile=CH3Cl.xyz Yfile=en.dat \\
              sigma=hyperopt.loguniform(4,20)
''',
  'MD':'''
  3D molecular dynamics using user-provided ML models

  Arguments:
    Input options:
      MLmodelIn=S              file S with ML model, not necessary if AIQM1 or 
                               ANI-1ccx is used 
      MLprog=S                 ML program to use, depending on the model type
        MLatomF [default]        you can also use KREG, ID, KID, uKID, aKID,
                                 pKID or KRR-CM to call MLatomF
        GAP                      you can also use GAPSOAP or GAP-SOAP to call GAP
        sGDML                    you can also use GDML to call sGDML
        DeepMD                   you can also use DPMD, deepmd-kit or deeppot-se 
                                 to call DeepMD
        Physnet                  
        ANI                      if you want to use ANI-1ccx just 
                                 write 'ANI-1ccx' in your input file instead of 
                                 'MLprog=ANI-1ccx' (the same with ANI-1x, ANI-2x,
                                 ANI-1x-D4 and ANI-2x-D4)
        AIQM1                    
        AIQM1@DFT
        AIQM1@DFT*
      dt=R                       time step in fs [0.1 by default]
      trun=R                     maximum runtime in fs [1ps by default]
      initXYZ=S                  initial geometry (should be in Angstrom)
      initVXYZ=S                 initial velocity (should be in Angstrom/fs)
      initcond=S                 algorithm of generating initial conditions
        user-defined [by default]  user provide initial condition using 
                                   initXYZ & initVXYZ option 
        random                     generate random velocities; user still 
                                   needs to provide initXYZ; must be used 
                                   together with initTemp option
        wigner                     use wigner sampling to generate initial 
                                   geometries and velocities; must be used 
                                   together with nmfile & optXYZfile options
      initTemp=R                 initial temperature in Kelvin, necessary when generating 
                                 random initial velocity
                                 [300 by default]
      nmfile=S                   Gaussian output file containing normal modes 
      optXYZfile=S               file containing XYZ file with optimized geometry

      noAng=N                    whether to get rid of angular momentum;
                                 used with initcond=random
        0 [by default]           do not eliminate angular momentum
        1                        eliminate angular momentum
      DOF=N                      degrees of freedom [-6 by default];
                                 used with initcond=user-defined;
                                 true DOF of the system is set to be 3*Natoms+DOF
      linear=N                   if the molecule is linear
        0 [by default]           the molecule is not linear
        1                        the molecule is linear

      initVXYZout=S              output file of initial velocity
      initXYZout=S               output file of initial geometry

      MLenergyUnits=S            energy units of ML model 
        kcal/mol [by default]      
        Hartree
      MLdistanceUnits=S          distance units of ML model 
        Anstrom [by default]
        Bohr

      Thermostat=S               MD thermostat
        NVE [by default]           NVE (microcononical) ensemble
        Andersen                   Andersen NVT thermostat
        Nose-Hoover                Nose-Hoover chain NVT ensemble
      Temp=R                     environment temperature (only valid when using NVT ensemble)
                                 [300 by default]
      Gamma=R                    collision frequency in fs^-1 [0.2 by default] 
                                 *for Andersen thermostat only*
      NHClength=N                Nose-Hoover chain length [3 by default]
                                 *for Nose-Hoover thermostat only*
      Nc=N                       Multiple time step [3 by default]
                                 *for Nose-Hoover thermostat only*
      Nys=N                      number of Yoshida Suzuki steps used in NHC [7 by default]
                                 only 1,3,5,7 are available
                                 *for Nose-Hoover thermostat only* 
      NHCfreq=R                  Nose-Hoover chain frequency in fs [16 by default]
      ThermostatOn=R             turn on the thermostat after R fs
      ThermostatOff=R            turn off the thermostat after R fs
                                 
    Output options:
      trajH5MDout=S              trajectory saved in H5MD file format 
                                 [traj.h5 by default]
      trajTime=S                 file with time of each step
                                 [traj.t by default]
      trajXYZout=S               file with geometry of each step
                                 [traj.xyz by default]
      trajVXYZout=S              file with velocity of each step
                                 [traj.vxyz by default]
      trajEpot=S                 file with potential energy of each step
                                 [traj.epot by default]
      trajEkin=S                 file with kinetic energy of each step
                                 [traj.ekin by default]
      trajEtot=S                 file with total energy of each step
                                 [traj.etot by default]
      trajEgradXYZ=S             file with energy gradients of each step
                                 [traj.grad by default]
      trajDipoles=S              file with dipole moments of each step
                                 output only when MLprog=AIQM1 is used 
                                 [traj.dp by default]
      trajTemperature=S          file with instantaneous temperature of 
                                 each step 
                                 [traj.temp by default]
  Example:
    MLatom.py MD MLmodelIn=H2.unf MLprog=MLatomF initXYZ=H2.xyz initVXYZ=H2.vxyz 
    MLatom.py MD MLprog=AIQM1 initXYZ=H2.xyz initVXYZ=H2.vxyz
    MLatom.py MD ANI-1ccx initXYZ=H2.xyz initVXYZ=H2.vxyz
''',
  'create4Dmodel':'''
  Create and save 4D model

  Arguments:
    tc=R                         cutoff time in fs [10 by default]
    trajsList=S                  file with npz files names
    Nsubtrain=R                  fraction of data used in subtraining [0.95 by default]
    Nvalidate=R                  fraction of data used in validation [0.05 by default]
    MLmodelOut=S                 file of 4D model
    ICidx=S                      file with indices of internal coordinates
    FourD.batchSize3D=N          batch size in training 3D model
    FourD.batchSize4D=N          batch size in training 4D model          
    FourD.maxEpoch3D=N           maximal epoch number in training 3D model
    FourD.maxEpoch4D=N           maximal epoch number in training 4D model
''',
  'use4Dmodel':'''
  Use existing 4D model

  Arguments:
    MLmodelIn=S                  file of 4D model 
    initXYZ=S                    initial geometry (should be in Angstrom)
    initVXYZ=S                   initial velocity (should be in Angstrom/fs)
    trun=R                       maximum runtime in fs [1ps by default]
    trajXYZout=S                 file with geometry of each step
    trajTime=S                   file with time of each step
    trajH5MDout=S                trajectory saved in H5MD file format
    dt=R                         time step in fs
    tSegm=R                      segment fime in fs 
''',


  'IRSS':'''
  Infrared spectrum simulation

  Arguments:
    trajH5MDin=S                 file with trajectory in H5MD format
                                 should contain dipole moments
    trajVXYZin=S                 file containing velocities 
    trajdpin=S                   file containing dipole moments 
                                 !!!Note!!! trajH5MDin, trajVXYZin, trajdpin 
                                 cannot be used at the same time
    threshold=R                  print peaks with abosorption larger than R
                                 0.0 < R <= 1.0 [0.1 by default]
    lb=R                         lower boundary; unit: fs [0.0 by default]
    ub=R                         upper boundary; unit: fs [total time by default]
                                 use trajectory from lb to ub 
                                 [use the whole trajectory by default]
    autocorrelationDepth=N       autocorrelation depth; unit: fs [1024 by default]
    zeropadding=N                zero padding; unit: fs [1024 by default]
    title=S                      title of the plot [no title by default]
    output=S
      ir                         output infrared spectrum
      ps                         output power spectrum
                                 !!!Note!!! when this option is not specified, mlatom 
                                 will output infrared spectrum if there are dipole moments 
                                 in H5MD file, otherwise it will output power spectrum

  Examples:
    $mlatom IRSS trajH5MDin=traj.h5
'''
    }
