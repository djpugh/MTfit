import os
import sys
import traceback
import sphinx
import subprocess
import shutil
import tempfile
import glob


from examples.example_data import synthetic_event, krafla_event

# Build from tox, build everything - add man pages so that they can be installed with scripts?
# Build epub and pdf
# Link to pdf from html page

MTfit_documentation = """
*********************************
MTfit
*********************************

Bayesian Moment Tensor Inversion Code by David J Pugh
MTfit is based on the bayesian approach presented in Pugh, D J, 2015,
Bayesian Source Inversion of Microseismic Events, PhD Thesis, Department of Earth Sciences,
University of Cambridge.

The code can be called from the command line directly or from within python itself (see below)


**Restricted:  For Non-Commercial Use Only**
This code is protected intellectual property and is available solely for teaching
and non-commercially funded academic research purposes.

Applications for commercial use should be made to Schlumberger or the University of Cambridge.


Input Data
==================================

There are several different input data types, and it is also possible to add additional parsers using the ``MTfit.parsers`` entry point.


The required data structure for running MTfit is very simple, the inversion expects a python dictionary of the data in the format::

    >>> data={'PPolarity':{'Measured':numpy.matrix([[-1],[-1]...]),
                         'Error':numpy.matrix([[0.01],[0.02],...]),
                         'Stations':{'Name':['Station1','Station2',...],
                                     'Azimuth':numpy.matrix([[248.0],[122.3]...]),
                                     'TakeOffAngle':numpy.matrix([[24.5],[22.8]...]),
                                    }
                         },
              'PSHAmplitudeRatio':{...},
              ...
              'UID':'Event1'
              }

For more information on the data keywords and how to set them up, see :class:`~MTfit.inversion.Inversion` docstrings.

The data dictionary can be passed directly to the :class:`~MTfit.inversion.Inversion` object (simple if running within python), or from a binary pickled object, these can be made by simply using pickle (or cPickle)::

    >>> pickle.dump(data,open(filename,'wb'))


The coordinate system is that the Azimuth is angle from x towards y and TakeOffAngle is the angle from positive z.

For data in different formats it is necessary to write a parser to convert the data into this dictionary format.

There is a parser for csv files with format

CSV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is a CSV format parser which reads CSV files.
The CSV file format is to have events split by blank lines, a header line showing where the information is, UID and data-type information stored in the first column, e.g.::

    UID=123,,,,
    PPolarity,,,,
    Name,Azimuth,TakeOffAngle,Measured,Error
    S001,120,70,1,0.01
    S002,160,60,-1,0.02
    P/SHRMSAmplitudeRatio,,,,
    Name,Azimuth,TakeOffAngle,Measured,Error
    S003,110,10,1,0.05 0.04
    ,,,,
    PPolarity ,,,,
    Name,Azimuth,TakeOffAngle,Measured,Error
    S003,110,10,1,0.05

This is a CSV file with 2 events, one event ID of 123, and PPolarity data at station S001 and station S002 and P/SHRMSAmplitude data at station S003,
and a second event with no ID (will default to the event number, in this case 2) with PPolarity data at station S003.


hyp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is a hyp format parser which reads hyp files as defined by `NonLinLoc <http://alomax.free.fr/nlloc/soft6.00/formats.html#_location_hypphs_>`_, this allows output files from NonLinLoc to be directly read.

Output
==================================

The default output is to output a MATLAB file containing 2 structures and a cell array, although there are two other possible formats, and others can be added (see MTfit.extensions).
The ``Events`` structure has the following fieldnames: ``MTspace`` and ``Probability``.

    * ``MTspace`` - The moment tensor samples as a 6 by n vector of the form::

        Mxx
        Myy
        Mzz
        sqrt(2)*Mxy
        sqrt(2)*Mxz
        sqrt(2)*Myz

    * ``Probability`` - The corresponding probability values

The ``Other`` structure contains information about the inversion

The ``Stations`` cell array contains the station information, including, if available, the polarity:

    +-----+----------------------+---------------------------+--------------------------+
    |Name |Azimuth(angle from x) |TakeOffAngle(angle from z) |P Polarity (if available) |
    +-----+----------------------+---------------------------+--------------------------+

A log file for each event is also produced to help with debugging and understanding the results.

Pickle format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is also possible to output the data structure as a pickled file using the pickle output options, storing the output dictionary as a pickled file.

hyp format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The results can be outputted in the `NonLinLoc hyp format <http://alomax.free.fr/nlloc/soft6.00/formats.html#_location_hypphs_>`_,
with the range of solutions sampled outputted as a binary file with the following format::

    binary file version (unsigned long integer)
    total_number_samples(unsigned long integer)
    number_of_saved_samples(unsigned long integer)
    converted (bool flag)
    Ln_bayesian_evidence (double)
    Kullback-Liebeler Divergence from sampling prior (double)

Then for each moment tensor sample (up to ``number_of_saved_samples`` )::

    Probability (double)
    Ln_probability(double)
    Mnn (double)
    Mee (double)
    Mdd (double)
    Mne (double)
    Mnd (double)
    Med (double)

if Converted is true then each sample also contains::

    gamma (double)
    delta (double)
    kappa (double)
    h (double)
    sigma (double)
    u (double)
    v (double)
    strike1 (double)
    dip1 (double)
    rake1 (double)
    strike2 (double)
    dip2 (double)
    rake2 (double)

If there are multiple events saved, then the next event starts immediately after the last with the same format. The output binary file can be re-read into python using MTfit.inversion.read_binary_output.





Running in parallel
==================================

The code is written to run in parallel using multiprocessing, it will initialise as many threads as the system reports available.
A single thread mode can be forced using:

    * -l, --singlethread, --single, --single_thread flag on the command line
    * parallel=False keyword in the MTfit.inversion.Inversion object initialisation

It is also possible to run this code on a cluster using qsub [requires pyqsub]. This can be called from the commandline using a flag:

    * -q, --qsub, --pbs

This runs using a set of default parameters, however it is also possible to adjust these parameters using commandline flags (use -h flag for help and usage).

There is a bug when using mpi and very large result sizes, giving a size error (negative integer) in mpi4py. If this occurs, lower the sample size and it will be ok.


WARNING:

    If running this on a server, be aware that not setting the number of workers option ``--numberworkers``, when running in parallel, means that as many processes as processors will be spawned, slowing down the machine for any other users.


Command line flags
==================================

To obtain a list of the command line flags use the -h flag::

    $ MTfit -h

This will provide a list of the arguments and their usage.


Running from the command line
==================================

To run from the command line on  linux/*nix  it is necessary to make sure that the MTfit script installed is on the path,
or to set up a manual alias/script, e.g. for bash::

    $ python -c "import MTfit;MTfit.run.MTfit()" $*


On windows using powershell add the following commandlet to your profile (for information on customizing your powershell profile see: http://www.howtogeek.com/50236/customizing-your-powershell-profile/)::

    function MTfit{
        $script={
            python -c "import MTfit;MTfit.run.MTfit()" $args
        }
        Invoke-Command -ScriptBlock $script -ArgumentList $args
    }




Running from within python
==================================

To run from within python, (assuming the module is on your PYTHONPATH) first::

    >>> import MTfit

Then to run the inversion it is necessary to set up an MTfit.inversion.Inversion object::

    >>> myInversion=MTfit.Inversion(**kwargs)

For more information on the arguments for initialising the inversion object, see the MTfit.inversion.Inversion docstrings


Running Module Tests
==================================

There is a unittest suite for this module that can be run from the python interpreter::

    >>> import MTfit
    >>> MTfit.run_tests()

Or during setup:

    >>> python setup.py test

If there are any errors please see the documentation and if necessary contact the developer for assistance.
"""


def build_docs(html=True, manpages=True, pdf=True, epub=True, gh_pages=False, travis=None):
    if travis is None:
        travis = os.environ.get('ON_TRAVIS', False)
    if 'setup.py' not in os.listdir('.'):
        raise ValueError('Needs to be run in the top of the repository')
    print('\n\n==============================\n\nBuilding Documentation\n\n==============================\n\n')
    get_run()
    try:
        get_cli_and_man()
    except Exception:
        traceback.print_exc()
    setup_examples()
    # Make rst
    # Add to extensions rst
    setup_extensions()
    # Source code extension
    make_plot_docs()
    if html:
        pdf = True
        epub = True
    try:
        if manpages:
            build_man_pages()
        if pdf:
            build_pdf()
        if epub:
            build_epub()
        if html:
            build_html()
        print("*********************************\n\nDocumentation Build Succeeded\n\n*********************************")
    except Exception:
        traceback.print_exc()
        print("*********************************\n\nDocumentation Build Failed\n\n*********************************")
    if gh_pages:
        print("*********************************\n\nSetting up gh-pages\n\n*********************************")
        setup_gh_pages(travis)


def build_html(output_path=os.path.abspath('./docs/html/')):
            print("------------------------------\n\nHTML Build\n\n------------------------------")
            try:
                sphinx.main(['sphinx', '-b', 'html', '-a', os.path.abspath('./docs/source/'), output_path])
            except SystemExit:
                pass


def build_man_pages(output_path=os.path.abspath('./docs/man/')):
            print("------------------------------\n\nMan Build\n\n------------------------------")
            try:
                sphinx.main(['sphinx', '-b', 'man', '-a', os.path.abspath('./docs/source/'), output_path])
            except SystemExit:
                pass


def build_pdf(output_path=os.path.abspath('./docs/pdf/MTfit.pdf')):
    print("------------------------------\n\nLaTeX Build\n\n------------------------------")
    try:
        sphinx.main(['sphinx', '-b', 'latex', '-a', os.path.abspath('./docs/source/'), os.path.abspath('./docs/latex/')])
    except SystemExit:
        pass
    os.chdir('./docs/latex/')
    try:
        os.remove('MTfit.toc')
    except Exception:
        pass
    try:
        os.remove('MTfit.aux')
    except Exception:
        pass
    try:
        os.remove('MTfit.idx')
    except Exception:
        pass
    # modify table of contents location
    tex = open('MTfit.tex').readlines()
    if '\\endabstract\n' in tex:
        tex.insert(tex.index('\\endabstract\n'), tex.pop(tex.index('\\sphinxtableofcontents\n')))
    with open('MTfit.tex', 'w') as f:
        f.write(''.join(tex))
    print("------------------------------\n\nPDF Build\n\n------------------------------")
    # Two compiles to update toc
    p = subprocess.Popen(['pdflatex', '-interaction=nonstopmode', 'MTfit.tex'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (bout, berr) = p.communicate()
    p2 = subprocess.Popen(['pdflatex', '-interaction=nonstopmode', 'MTfit.tex'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (b2out, b2err) = p2.communicate()
    print(bout)
    os.chdir('../../')
    if 'fatal error occured' in bout.lower()+b2out.lower():
        raise Exception('Fatal Error in PDF generation')
    try:
        os.mkdir('./docs/pdf')
    except Exception:
        pass
    shutil.move('./docs/latex/MTfit.pdf', output_path)


def build_epub(output_path=os.path.abspath('./docs/epub/')):
    print("------------------------------\n\nepub Build\n\n------------------------------")
    try:
        sphinx.main(['sphinx', '-b', 'epub', '-a', os.path.abspath('./docs/source/'), output_path])
    except SystemExit:
        pass


def setup_gh_pages(travis=False):
    # Copy docs/html to tempfolder
    from MTfit import __version__
    print("------------------------------\n\nMaking Temporary Directory\n\n------------------------------")
    temp_dir = tempfile.mkdtemp()
    print(temp_dir)
    shutil.copytree('./docs/html', os.path.join(temp_dir, 'html'))
    # Checkout gh-pages branch
    # Need to stash any current work
    import git
    repo = git.Repo('.')
    print("------------------------------\n\nStashing Changes\n\n------------------------------")
    repo.git.stash('save')
    print("------------------------------\n\nSwitching Branch to gh-pages\n\n------------------------------")
    current_branch = repo.active_branch.name
    repo.git.checkout('gh-pages')
    # Clean folder and copy html into folder
    print("------------------------------\n\nCleaning Working Set\n\n------------------------------")
    contents = glob.glob('./*')
    for item in contents:
        if '.git' in item or '.venv' in item or '.tox' in item:
            continue
        if 'src' in item or 'docs' in item or 'examples' in item or 'wheelhouse' in item:
            # ignored in .gitignore
            continue
        if os.path.isdir(item):
            shutil.rmtree(item)
        else:
            os.remove(item)
    print("------------------------------\n\nCopying Documentation from Temporary Directory\n\n------------------------------")
    copy_recursively(os.path.join(temp_dir, 'html'), os.path.abspath('./'))
    print("------------------------------\n\nRemoving Temporary Directory\n\n------------------------------")
    shutil.rmtree(temp_dir)
    print("------------------------------\n\nCommitting Documentation to gh-pages\n\n------------------------------")
    # Commit the changes
    repo.git.add('*')
    repo.git.commit('-m', 'Documentation {}'.format(__version__))
    if not travis:
        # Checkout old branch
        print("------------------------------\n\nReturning to {} Branch\n\n------------------------------".format(current_branch))
        repo.git.checkout(current_branch)
        print("------------------------------\n\nUnstashing Changes\n\n------------------------------")
        repo.git.stash('pop')
        print("------------------------------\n\nCleaning Working Directory\n\n------------------------------")
        doctrees = glob.glob('*.doctree')
        for item in doctrees:
            os.remove(item)
        invs = glob.glob('*.inv')
        for item in invs:
            os.remove(item)
        jss = glob.glob('*.js')
        for item in jss:
            os.remove(item)


def copy_recursively(source_folder, destination_folder):
    for root, dirs, files in os.walk(source_folder):
        for item in dirs:
            if item[0] == '.':
                continue
            src_path = os.path.join(root, item)
            dst_path = os.path.join(destination_folder, os.path.relpath(src_path, source_folder))
            if destination_folder not in dst_path:
                raise ValueError('Error joining paths - {} not in {}'.format(dst_path, destination_folder))
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
        for item in files:
            if item[0] == '.':
                continue
            try:
                src_path = os.path.join(root, item)
                dst_path = os.path.join(destination_folder, os.path.relpath(src_path, source_folder))
                if destination_folder not in dst_path:
                    raise ValueError('Error joining paths - {} not in {}'.format(dst_path, destination_folder))
                shutil.copyfile(src_path, dst_path)
            except Exception:
                pass


def get_run():
    with open('./docs/source/run_base.rst') as f:
        run = f.read()
    run += '\n.. _input-data-label:\n\nInput Data'+MTfit_documentation.split('Input Data')[1].split('Command line flags')[0].replace('Output', '\n.. _MATLAB-output-label:\n\nOutput')
    run = run.replace('WARNING:', '.. warning::\n')
    run = run.replace('MTfit.inversion.read_binary_output', ':func:`MTfit.inversion.read_binary_output`')
    run = run.replace('mpi4py', ':mod:`mpi4py`')
    with open('./docs/source/run.rst', 'w') as f:
        f.write(run)


def get_cli_and_man():
    from MTfit.utilities.argparser import MTfit_parser
    old_stdout = sys.stdout
    try:
        sys.stdout = open('./docs/source/cli.rst', 'w')
        MTfit_parser(['-h'])
        sys.stdout.close()
        sys.stdout = old_stdout
    except SystemExit:
        sys.stdout.close()
        sys.stdout = old_stdout
    except Exception:
        sys.stdout.close()
        sys.stdout = old_stdout
        traceback.print_exc()
    with open('./docs/source/cli.rst') as f:
        cli = f.read()
    cli = cli.replace('usage: ', '********************************\nMTfit command line options\n********************************\nCommand line usage::\n')
    cli = cli.replace('MTfit - Moment Tensor', '#A~A~A~A')
    cli = cli.replace('positional arguments:', '#B~B~B~BPositional Arguments:\n============================\n')
    cli = cli.replace('optional arguments:', 'Optional Arguments:\n============================\n')
    cli = cli.replace('Cluster:', 'Cluster:\n============================\n')
    cli = cli.replace('scatangle:', 'Scatangle:\n============================\n')
    cli = cli.replace('Scatangle:', 'Scatangle:\n============================\n')
    cli = cli.replace('Commands for the extension scatangle', ' ')
    cli = cli.replace('form:', 'form::')
    cli = cli.replace('e.g.:', 'e.g.::')
    # Handle splitting of cli args and descrptions
    cli = cli.split('#A~A~A~A')[0]+cli.split('#B~B~B~B')[1]
    generate_man_page(cli)
    cli_lines = cli.split('\n')
    flag_indices = [i for i, u in enumerate(cli_lines) if len(u) > 3 and u[0] == ' ' and u[2] != ' ']
    same_line_indices = [i for i, u in enumerate(cli_lines) if len(u) > 25 and u[0] == ' ' and u[2] != ' ' and u[24] != ' ' and u[20:24] == '    ']
    mod_cli_lines = []
    flag = False
    code = False
    list_flag = False
    for i, u in enumerate(cli_lines):
        if code and not len(u.rstrip()):
            code = False
            mod_cli_lines.append('')
        if u == '&&':
            u = ''
        if not code and i in flag_indices:
            if not flag:
                flag = True
                mod_cli_lines.append('---------------------------')
                mod_cli_lines.append('')
                mod_cli_lines.append('::')
                mod_cli_lines.append('')
        if not code and i in same_line_indices:
            flag = False
            mod_cli_lines.append(u[0:24].rstrip())
            mod_cli_lines.append('')
            mod_cli_lines.append(u[24:])
        elif not code and i in flag_indices:
            mod_cli_lines.append(u)
        else:
            if flag:
                flag = False
                mod_cli_lines.append('')
            if code:
                mod_cli_lines.append('    '+u.lstrip())
            elif len(u.lstrip()) and u.lstrip()[0:2] == '* ':
                list_flag = True
                mod_cli_lines.append('  '+u.lstrip())
            elif list_flag and len(u.lstrip()) and not u.lstrip()[0:2] == '* ':
                mod_cli_lines[-1] += u.lstrip()
            elif list_flag and not len(u.lstrip()):
                list_flag = False
                mod_cli_lines.append(u.lstrip())
            else:
                mod_cli_lines.append(u.lstrip())
                if u[-2:] == '::':
                    code = True
                    mod_cli_lines.append('')
    cli = '\n'.join(mod_cli_lines)
    cli += '\n.. only:: not latex\n\n    :doc:`run`'
    cli = cli.replace('DO NOT USE - only for', '.. warning::\n\n\tDo not use - automatically set when')

    with open('./docs/source/cli.rst', 'w') as f:
        f.write(cli)


def generate_man_page(cli):
    man = cli.replace('********************************\nMTfit command line options\n********************************\nCommand line usage::\n', '\nCommand Line Options\n==================================\n\nUsage:\n\n\t')
    man = MTfit_documentation+'\n\n\n'+man
    with open('./docs/source/man.rst', 'w') as f:
        f.write(man)


def setup_examples():
    char_width = 70
    try:
        out = ">>> from example_data import synthetic_event\n>>> data=synthetic_event()\n>>> print data['PPolarity']\n"+",\n\t'S0".join(str(synthetic_event()['PPolarity']).split(", 'K"))
        out = out.split('\n')
        fixed_output = []
        for u in out:
            u = u.replace('\t', '    ')
            if len(u) < char_width:
                fixed_output.append(u)
            else:
                while len(u):
                    ind = u[0:char_width].rfind(' ')
                    if len(u) < char_width or ind == -1:
                        fixed_output.append('            '+u)
                        u = ''
                    else:
                        fixed_output.append('            '+u[0:ind+1])
                        u = u[ind+1:]
        with open('./docs/source/synthetic_p_polarity_data.txt', 'w') as f:
            f.write('\n'.join(fixed_output))
    except Exception:
        pass
    try:
        out = ">>> from example_data import krafla_event\n>>> data=krafla_event()\n>>> print data['PPolarity']\n"+",\n\t'S0".join(str(krafla_event()['PPolarity']).split(", 'S0"))
        out = out.split('\n')
        fixed_output = []
        for u in out:
            u = u.replace('\t', '    ')
            if len(u) < char_width:
                fixed_output.append(u)
            else:
                while len(u):
                    ind = u[0:char_width].rfind(' ')
                    if ind == -1:
                        fixed_output.append(u)
                        u = ''
                    else:
                        fixed_output.append(u[0:ind+1])
                        u = u[ind+1:]
        with open('./docs/source/krafla_p_polarity_data.txt', 'w') as f:
            f.write('\n'.join(fixed_output))
    except Exception:
        pass


def make_plot_docs():

    try:
        from MTfit.utilities.argparser import MTplot_parser
    except Exception:
        # Haven't installed the package so add src to the system path
        sys.path.insert(0, 'src')
        from MTfit.utilities.argparser import MTplot_parser
    path = './docs/source/'
    try:
        old_stdout = sys.stdout
        try:
            with open(path+'mtplotcli.rst', 'w') as f:
                sys.stdout = f
                MTplot_parser(['-h'])
                sys.stdout.close()
                sys.stdout = old_stdout
        except SystemExit:
            sys.stdout.close()
            sys.stdout = old_stdout
        except Exception:
            sys.stdout.close()
            sys.stdout = old_stdout
            traceback.print_exc()
        with open(path+'mtplotcli.rst') as f:
            cli = f.read()
        cli = cli.replace('usage: ', '********************************\nMTplot command line options\n********************************\nCommand line usage::\n')
        cli = cli.replace('MTPlot - Moment', '#A~A~A~A')
        cli = cli.replace('positional arguments:', '#B~B~B~BPositional Arguments:\n============================\n')
        cli = cli.replace('optional arguments:', 'Optional Arguments:\n============================\n')
        cli = cli.replace('Cluster:', 'Cluster:\n============================\n')
        cli = cli.replace('scatangle:', 'Scatangle:\n============================\n')
        cli = cli.replace('Scatangle:', 'Scatangle:\n============================\n')
        cli = cli.replace('Commands for the extension scatangle', ' ')
        cli = cli.replace('form:', 'form::')
        cli = cli.replace('e.g.:', 'e.g.::')
        # Handle splitting of cli args and descrptions
        cli = cli.split('#A~A~A~A')[0]+cli.split('#B~B~B~B')[1]
        man = cli.replace('********************************\nMTfit command line options\n********************************\nCommand line usage::\n', '\nCommand Line Options\n==================================\n\nUsage:\n\n\t')
        with open(path+'manMTplot.rst', 'w') as f:
            f.write(man)
        cli_lines = cli.split('\n')
        flag_indices = [i for i, u in enumerate(cli_lines) if len(u) > 3 and u[0] == ' ' and u[2] != ' ']
        same_line_indices = [i for i, u in enumerate(cli_lines) if len(u) > 25 and u[0] == ' ' and u[2] != ' ' and u[24] != ' ' and u[20:24] == '    ']
        mod_cli_lines = []
        flag = False
        code = False
        list_flag = False
        for i, u in enumerate(cli_lines):
            if code and not len(u.rstrip()):
                code = False
                mod_cli_lines.append('')
            if u == '&&':
                u = ''
            if not code and i in flag_indices:
                if not flag:
                    flag = True
                    mod_cli_lines.append('---------------------------')
                    mod_cli_lines.append('')
                    mod_cli_lines.append('::')
                    mod_cli_lines.append('')
            if not code and i in same_line_indices:
                flag = False
                mod_cli_lines.append(u[0:24].rstrip())
                mod_cli_lines.append('')
                mod_cli_lines.append(u[24:])
            elif not code and i in flag_indices:
                mod_cli_lines.append(u)
            else:
                if flag:
                    flag = False
                    mod_cli_lines.append('')
                if code:
                    mod_cli_lines.append('    '+u.lstrip())
                elif len(u.lstrip()) and u.lstrip()[0:2] == '* ':
                    list_flag = True
                    mod_cli_lines.append('  '+u.lstrip())
                elif list_flag and len(u.lstrip()) and not u.lstrip()[0:2] == '* ':
                    mod_cli_lines[-1] += u.lstrip()
                elif list_flag and not len(u.lstrip()):
                    list_flag = False
                    mod_cli_lines.append(u.lstrip())
                else:
                    mod_cli_lines.append(u.lstrip())
                    if u[-2:] == '::':
                        code = True
                        mod_cli_lines.append('')
        cli = '\n'.join(mod_cli_lines)
        cli += '\n.. only:: not latex\n\n    :doc:`run`'
        cli = cli.replace('DO NOT USE - only for', '.. warning::\n\n\tDo not use - automatically set when')
        with open(path+'mtplotcli.rst', 'w') as f:
            f.write(cli)
    except Exception:
        traceback.print_exc()


def setup_extensions():
    try:
        from MTfit.utilities.extensions import get_extensions
        from MTfit.extensions import rst_table, rst_docs, __doc1__
    except Exception:
        # Haven't installed the package so add src to the system path
        sys.path.insert(0, 'src')
        from MTfit.utilities.extensions import get_extensions
        from MTfit.extensions import rst_table, rst_docs, __doc1__
    ext_doc_names, ext_docs = get_extensions('MTfit.documentation')
    if len(ext_doc_names):
        # Have extension documentation so add extensions as toc including entry_points
        entry_points_file = 'entry_points'
        entry_points = """*********************************
MTfit Entry Points
*********************************

"""
        for ext in ext_doc_names:
            with open('./docs/source/'+ext+'.rst', 'w') as f:
                f.write(ext_docs[ext])
    else:
        entry_points_file = 'extensions'
        entry_points = """*********************************
Extending MTfit
*********************************

"""
    try:
        entry_points += """MTfit has been written with the view that it is desirable to be able to easily extend the code. This is done using `entry points <https://pythonhosted.org/setuptools/pkg_resources.html#entry-points>`_ from the `setuptools <https://pythonhosted.org/setuptools>`_ module.

The entry points are:

"""
        entry_points = entry_points.replace('The entry points are:', 'The entry points are:\n\n.. only:: latex\n\n    .. tabularcolumns:: |l|L|\n\n'+rst_table+'\n\n'+__doc1__.replace('extensions/scatangle.py', ':download:`extensions/scatangle.py <../../src/MTfit/extensions/scatangle.py>`').replace('setup.py', '``setup.py``')+'\n\n'+rst_docs)
        with open('./docs/source/'+entry_points_file+'.rst', 'w') as f:
            f.write(entry_points)
    except Exception:
        traceback.print_exc()
    setup_extensions_source_code()


def setup_extensions_source_code():
    try:
        from MTfit.utilities.extensions import get_extensions
    except Exception:
        # Haven't installed the package so add src to the system path
        sys.path.insert(0, 'src')
        from MTfit.utilities.extensions import get_extensions
    ext_source_code_names, ext_source_code = get_extensions('MTfit.source_code')
    extensions_source_code = """
    MTfit.extensions
====================

Contents
***********
.. toctree::
   :maxdepth: 1

   MTfit.extensions.scatangle<source-scatangle>"""+'\n    '.join([ext.replace('_', ' ').capitalize()+' <'+ext+'>' for ext in ext_source_code_names])+"""

For the extensions documentation see :ref:`extensions`"""
    with open('./docs/source/source-extensions.rst', 'w') as f:
        f.write(extensions_source_code)
    for ext in ext_source_code_names:
        with open('./docs/source/'+ext+'.rst', 'w') as f:
            f.write(ext_source_code[ext])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['true', 't', '1']:
        gh_pages = True
    else:
        gh_pages = False
    build_docs(gh_pages=gh_pages)
