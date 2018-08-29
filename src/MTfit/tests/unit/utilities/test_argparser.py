"""
test_argparser.py
*****************

Tests for src/utils/argparser.py
"""
import os
import glob
import sys
import tempfile
import shutil

from MTfit.extensions import default_cmd_opts
from MTfit.extensions import default_cmd_defaults
# from MTfit.extensions import default_tests
from MTfit.utilities.extensions import get_extensions
from MTfit.utilities.argparser import _ARGPARSE
from MTfit.utilities.argparser_defaults import DEFAULT_AMP_COLORMAP
from MTfit.utilities.unittest_utils import TestCase
from MTfit.utilities.argparser import MTfit_parser
from MTfit.utilities.argparser import get_MTfit_defaults
from MTfit.utilities.argparser import evaluate_extensions
from MTfit.utilities import argparser


class ParserTestCase(TestCase):

    def setUp(self):
        self.cwd = os.getcwd()
        if sys.version_info >= (3, 0):
            self.tempdir = tempfile.TemporaryDirectory()
            os.chdir(self.tempdir.name)
        else:
            self.tempdir = tempfile.mkdtemp()
            os.chdir(self.tempdir)
        out_files = glob.glob('*.out')
        log_files = glob.glob('*.log')
        mat_files = glob.glob('*.mat')
        inv_files = glob.glob('*.inv')
        mt_files = glob.glob('*.mt')
        hyp_files = glob.glob('*.hyp')
        self.old_files = {'out': out_files, 'log': log_files, 'mat': mat_files,
                          'inv': inv_files, 'mt': mt_files, 'hyp': hyp_files}
        self.keep = ['MTfit_build.log']

    def tearDown(self):
        for file_type in self.old_files.keys():
            for fname in glob.glob('*.'+file_type):
                if fname not in self.old_files[file_type] and fname not in self.keep:
                    try:
                        os.remove(fname)
                    except Exception:
                        pass
        try:
            os.remove('Test.inv')
        except Exception:
            pass
        try:
            os.remove('Test.scatangle')
        except Exception:
            pass
        try:
            os.remove('Test.p123')
        except Exception:
            pass
        try:
            os.remove('Test.p124')
        except Exception:
            pass
        try:
            os.remove('Test.i')
        except Exception:
            pass
        try:
            os.remove('Test2.i')
        except Exception:
            pass
        os.chdir(self.cwd)
        if sys.version_info >= (3, 0):
            self.tempdir.cleanup()
        else:
            try:
                shutil.rmtree(self.tempdir)
            except:
                pass

    def test_MTfit_parser(self):
        cmd_defaults = {}
        cmd_default_types = {}
        results = evaluate_extensions('MTfit.cmd_defaults', default_cmd_defaults)
        for result in results:
            cmd_defaults.update(result[0])
            cmd_default_types.update(result[1])
        defaults = get_MTfit_defaults(True, cmd_defaults, cmd_default_types)
        # extension_names, extension_test_plugins = get_extensions(
        #     'MTfit.tests', default_tests)
        extension_tests = []
        cmd_opt_names, cmd_opts = get_extensions(
            'MTfit.cmd_opts', default_cmd_opts)
        # for plugin_name in extension_names:
        #     if plugin_name in cmd_opt_names:
        #         try:
        #             extension_tests.append(
        #                 extension_test_plugins[plugin_name]()[2])
        #             test_names.append(plugin_name)
        #         except Exception:
        #             pass
        example_MTfit_script = """#!/bin/bash
        ##MTfit qsub script
        #PBS -S /bin/sh
        #PBS -N MTfit
        #PBS -l walltime=300:00:00
        #PBS -V
        #PBS -l nodes=16:ppn=8
        #PBS -l pmem=2Gb
        #PBS -q batch
        #PBS -m bae
        python -c "import MTfit; MTfit.__run__()" --number_location_samples=25 --inversionoptions="PPolarity,P/SHQRMSAmplitudeRatio,P/SVRQMSAmplitudeRatio" --compareconstrained --algorithm=time --anglescatterfilepath="""+os.path.abspath('./')+os.path.sep+"Test.scatangle --numberworkers=128 --time=36000.0 --datafile="+os.path.abspath('./')+os.path.sep+" --nstations=0\n"
        example_MTfit_script2 = """#!/bin/bash
        ##MTfit qsub script
        #PBS -S /bin/sh
        #PBS -N MTfit
        #PBS -l walltime=300:00:00
        #PBS -V
        #PBS -l nodes=16:ppn=8
        #PBS -l pmem=2Gb
        #PBS -q batch
        #PBS -m bae
        python -c "import MTfit; MTfit.__run__()" --number_location_samples=250 --inversionoptions="PPolarity,P/SHQRMSAmplitudeRatio,P/SVRQMSAmplitudeRatio" --algorithm=time --anglescatterfilepath="""+os.path.abspath('./')+os.path.sep+"Test.scatangle --numberworkers=128 --time=36000.0 --datafile="+os.path.abspath('./')+os.path.sep+" --nstations=0\n"

        try:
            os.remove('Test.i')
        except Exception:
            pass
        try:
            os.remove('Test2.i')
        except Exception:
            pass
        # data_file -d

        global _ARGPARSE
        _argparse = _ARGPARSE is True
        if _argparse:
            print('data_file -d check')
            with self.assertRaises(SystemExit):
                MTfit_parser(['test'], test=True)
            with self.assertRaises(SystemExit):
                MTfit_parser(['-d=Test'], test=True)
            with self.assertRaises(SystemExit):
                MTfit_parser(['--datafile=Test'], test=True)
            open('Test.i', 'w').write('test')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            options, options_map = MTfit_parser(["-d=Test.i"], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            options, options_map = MTfit_parser(["-dTest.i"], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            options, options_map = MTfit_parser(
                ["--datafile=Test.i"], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            options, options_map = MTfit_parser(
                ["--datafile=*.i"], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            options, options_map = MTfit_parser(["--invext=i"], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            open('Test2.i', 'w').write('test')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            options, options_map = MTfit_parser(['Test2.i'], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test2.i')])
            options, options_map = MTfit_parser(["-d=Test2.i"], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test2.i')])
            options, options_map = MTfit_parser(["-dTest.i"], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            options, options_map = MTfit_parser(
                ["--datafile=Test2.i"], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test2.i')])
            options, options_map = MTfit_parser(
                ["--datafile=*.i"], test=True)
            self.assertAlmostEqual(
                options['data_file'], [os.path.abspath('Test.i'), os.path.abspath('Test2.i')])
            options, options_map = MTfit_parser(["--invext=i"], test=True)
            self.assertAlmostEqual(
                options['data_file'], [os.path.abspath('Test.i'), os.path.abspath('Test2.i')])
            os.remove('Test2.i')
            # AngleScatterFile -s
            print('location_pdf_file -s check')
            options, options_map = MTfit_parser(
                ["-d=Test.i", "-s=*.i"], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            self.assertEqual(
                options['location_pdf_file_path'], [os.path.abspath('Test.i')])
            options, options_map = MTfit_parser(
                ["Test.i", "-s=*.i"], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            self.assertEqual(
                options['location_pdf_file_path'], [os.path.abspath('Test.i')])
            options, options_map = MTfit_parser(
                ["-s=*.i", 'Test.i'], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            self.assertEqual(
                options['location_pdf_file_path'], [os.path.abspath('Test.i')])
            # algorithm -a
            print('algorithm -a check')
            with self.assertRaises(SystemExit):
                MTfit_parser(["-a=iteration", 'Test.i'], test=True)
            options, options_map = MTfit_parser(
                ["-a=iterate", 'Test.i'], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            self.assertEqual(options['algorithm'], 'iterate')
            options, options_map = MTfit_parser(
                ["-a=time", 'Test.i'], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            self.assertEqual(options['algorithm'], 'time')
            # Parallel -l
            print('parallel -l check')
            self.assertTrue(
                options['parallel'] != defaults['single_threaded'])
            options, options_map = MTfit_parser(
                ["-l", 'Test.i'], test=True)
            self.assertFalse(options['parallel'])
            options, options_map = MTfit_parser(
                ["--single", 'Test.i'], test=True)
            self.assertFalse(options['parallel'])
            options, options_map = MTfit_parser(
                ["--singlethread", 'Test.i'], test=True)
            self.assertFalse(options['parallel'])
            self.assertFalse('singlethread' in options.keys())
            # Number of Workers -n
            print('number_workers -n check')
            self.assertEqual(options['n'], defaults['number_workers'])
            options, options_map = MTfit_parser(
                ["-n=1", 'Test.i'], test=True)
            self.assertEqual(options['n'], 1)
            with self.assertRaises(SystemExit):
                MTfit_parser(["-n=1.1", 'Test.i'], test=True)
            # PMem -m
            print('mem -m check')
            self.assertEqual(options['phy_mem'], defaults['memory'])
            options, options_map = MTfit_parser(
                ["-m=1.5", 'Test.i'], test=True)
            self.assertEqual(options['phy_mem'], 1.5)
            # DC -c
            print('dc -c check')
            self.assertTrue(options['dc'] == defaults['double-couple'])
            options, options_map = MTfit_parser(
                ["-c", 'Test.i'], test=True)
            self.assertTrue(options['dc'])
            # NStations
            print('--nstations check')
            self.assertTrue(
                options['number_stations'] == defaults['number_stations'])
            options, options_map = MTfit_parser(
                ["--nstations=20", 'Test.i'], test=True)
            self.assertEqual(options['number_stations'], 20)
            with self.assertRaises(SystemExit):
                MTfit_parser(["--nstations=20.2", 'Test.i'], test=True)
            # NAngleSamples
            print('--nanglesamples check')
            self.assertEqual(
                options['number_location_samples'], defaults['number_location_samples'])
            options, options_map = MTfit_parser(
                ["--nanglesamples=20", 'Test.i'], test=True)
            self.assertEqual(options['number_location_samples'], 20)
            options, options_map = MTfit_parser(
                ["--number_location_samples=20", 'Test.i'], test=True)
            self.assertEqual(options['number_location_samples'], 20)
            with self.assertRaises(SystemExit):
                MTfit_parser(["--nanglesamples=20.2", 'Test.i'], test=True)
            # inversion_options -i
            print('inversion_options -i check')
            if defaults['inversion_options']:
                self.assertTrue(options['inversion_options'] == defaults['inversion_options'].split(
                    ','), str(options['inversion_options'])+' '+str(defaults['inversion_options']))
            else:
                self.assertTrue(options['inversion_options'] == defaults['inversion_options'], str(
                    options['inversion_options'])+' '+str(defaults['inversion_options']))
            options, options_map = MTfit_parser(
                ["-i=PPolarity", 'Test.i'], test=True)
            self.assertEqual(options['inversion_options'], ['PPolarity'])
            options, options_map = MTfit_parser(
                ["--inversionoptions=PPolarity,PSHRMSAmplitudeRatio", 'Test.i'], test=True)
            self.assertEqual(
                options['inversion_options'], ['PPolarity', 'PSHRMSAmplitudeRatio'])
            # file_sampling check
            print('file_sampling -f check')
            self.assertTrue(
                options['file_sample'] == defaults['disk_sample'])
            options, options_map = MTfit_parser(
                ["-f", 'Test.i'], test=True)
            self.assertTrue(options['file_sample'])
            options, options_map = MTfit_parser(
                ["--file-sample", 'Test.i'], test=True)
            self.assertTrue(options['file_sample'])
            with self.assertRaises(SystemExit):
                MTfit_parser(
                    ["--file-sample", 'Test.i', '-a=mcmc'], test=True)
            # FID -o
            print('output file -o check')
            self.assertTrue(options['fid'] == defaults['output_file'])
            options, options_map = MTfit_parser(
                ["-o=XKCD", 'Test.i'], test=True)
            self.assertEqual(options['fid'], "XKCD")
            options, options_map = MTfit_parser(
                ["--fid=XKCD", 'Test.i'], test=True)
            self.assertEqual(options['fid'], "XKCD")
            # max_samples -x
            print('max_samples -x check')
            options, options_map = MTfit_parser(
                ["-a=iterate", 'Test.i'], test=True)
            options, options_map = MTfit_parser(
                ["-a=iterate", "-x=100", 'Test.i'], test=True)
            self.assertEqual(options['max_samples'], 100)
            options, options_map = MTfit_parser(
                ["-a=iterate", "--maxsamples=100", 'Test.i'], test=True)
            self.assertEqual(options['max_samples'], 100)
            with self.assertRaises(SystemExit):
                MTfit_parser(["-x=20.2", 'Test.i'], test=True)
            # max_time -t
            print('max_time -t check')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            options, options_map = MTfit_parser(
                ["-a=iterate", 'Test.i'], test=True)
            with self.assertRaises(KeyError):
                options['max_time']
            options, options_map = MTfit_parser(
                ["-a=iterate", "-t=100", 'Test.i'], test=True)
            with self.assertRaises(KeyError):
                options['max_time']
            if defaults['algorithm'] != 'time':
                options, options_map = MTfit_parser(
                    ["-a=time", "-t=100", 'Test.i'], test=True)
            else:
                options, options_map = MTfit_parser(
                    ["-t=100", 'Test.i'], test=True)
            self.assertEqual(options['max_time'], 100)
            if defaults['algorithm'] != 'time':
                options, options_map = MTfit_parser(
                    ["-a=time", "--maxtime=100.1", 'Test.i'], test=True)
            else:
                options, options_map = MTfit_parser(
                    ["--maxtime=100.1", 'Test.i'], test=True)
            self.assertEqual(options['max_time'], 100.1)
            if defaults['algorithm'] != 'time':
                with self.assertRaises(SystemExit):
                    MTfit_parser(["-a=time", "-t=a", 'Test.i'], test=True)
            else:
                with self.assertRaises(SystemExit):
                    MTfit_parser(["-t=a", 'Test.i'], test=True)
            # multiple events
            print('multiple events test')
            options, options_map = MTfit_parser(
                ["-amcmc", "-e", 'Test.i'], test=True)
            self.assertTrue(options['multiple_events'])
            # relative amplitudes
            print('relative amplitudes test')
            options, options_map = MTfit_parser(
                ["-amcmc", "-r", 'Test.i'], test=True)
            self.assertTrue(options['relative_amplitude'])
            # marginalise relative test
            print('marginalise relative test')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertFalse(options['marginalise_relative'])
            options, options_map = MTfit_parser(
                ["-amcmc", "-r", "--marginalise-relative", 'Test.i'], test=True)
            self.assertTrue(options['marginalise_relative'])
            with self.assertRaises(SystemExit):
                MTfit_parser(
                    ["-amcmc", "--marginalise-relative", 'Test.i'], test=True)
            # min number intersections
            print('minimum_number_intersections -S check')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertEqual(options['minimum_number_intersections'], 2)
            options, options_map = MTfit_parser(
                ['Test.i', '-S1'], test=True)
            self.assertEqual(options['minimum_number_intersections'], 1)
            with self.assertRaises(SystemExit):
                MTfit_parser(["-S1", 'Test.i', '-r'], test=True)
            options, options_map = MTfit_parser(
                ['Test.i', '-S10', '-r'], test=True)
            self.assertEqual(options['minimum_number_intersections'], 10)
            # McMC options
            print('McMC options check')
            options, options_map = MTfit_parser(
                ["-a=mcmc", "-x=100", 'Test.i'], test=True)
            self.assertEqual(options['chain_length'], 100)
            self.assertEqual(
                options['dimension_jump_prob'], defaults['jump_probability'])
            self.assertEqual(
                options['min_acceptance_rate'], defaults['mcmc_min_acceptance_rate'])
            self.assertEqual(
                options['max_acceptance_rate'], defaults['mcmc_max_acceptance_rate'])
            self.assertEqual(
                options['acceptance_rate_window'], defaults['acceptance_rate_window'])
            self.assertEqual(
                options['learning_length'], defaults['learning_length'])
            self.assertEqual(options['algorithm'], 'mcmc')
            with self.assertRaises(SystemExit):
                MTfit_parser(["-u=T"], test=True)
            with self.assertRaises(SystemExit):
                MTfit_parser(["-v=T"], test=True)
            with self.assertRaises(SystemExit):
                MTfit_parser(["-w=T"], test=True)
            with self.assertRaises(SystemExit):
                MTfit_parser(["-j=T"], test=True)
            with self.assertRaises(SystemExit):
                MTfit_parser(["-z=T"], test=True)
            with self.assertRaises(SystemExit):
                MTfit_parser(["-z=10.1"], test=True)
            options, options_map = MTfit_parser(
                ["-a=transdmcmc", "-u=0.1", "-v=0.8", "-w=5000", "-j=0.2", '-z=100', 'Test.i'], test=True)
            self.assertEqual(
                options['chain_length'], defaults['transdmcmc_chain_length'])
            self.assertEqual(options['algorithm'], 'transdmcmc')
            self.assertEqual(options['dimension_jump_prob'], 0.2)
            self.assertEqual(options['min_acceptance_rate'], 0.1)
            self.assertEqual(options['max_acceptance_rate'], 0.8)
            self.assertEqual(options['acceptance_rate_window'], 5000)
            self.assertEqual(options['learning_length'], 100)
            options, options_map = MTfit_parser(
                ["-a=transdmcmc", 'Test.i'], test=True)
            self.assertEqual(
                options['dimension_jump_prob'], defaults['jump_probability'])
            self.assertEqual(
                options['min_acceptance_rate'], defaults['transdmcmc_min_acceptance_rate'])
            self.assertEqual(
                options['max_acceptance_rate'], defaults['transdmcmc_max_acceptance_rate'])
            self.assertEqual(
                options['acceptance_rate_window'], defaults['acceptance_rate_window'])
            self.assertEqual(
                options['learning_length'], defaults['learning_length'])
            print('Warnings -W check')
            options, options_map = MTfit_parser(
                ["-W=d", 'Test.i'], test=True)
            self.assertEqual(options['warnings'], 'd')
            options, options_map = MTfit_parser(
                ["-W=ignore", 'Test.i'], test=True)
            self.assertEqual(options['warnings'], 'i')
            with self.assertRaises(SystemExit):
                MTfit_parser(["-WABC", 'Test.i'], test=True)
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertEqual(options['warnings'], 'd')
            # min_number_initialisation_samples -X
            print("min_number_initialisation/check_samples -X check")
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertEqual(
                options['min_number_check_samples'], defaults['min_number_check_samples'])
            options, options_map = MTfit_parser(
                ['Test.i', '-X=1000'], test=True)
            self.assertEqual(options['min_number_check_samples'], 1000)
            options, options_map = MTfit_parser(
                ['Test.i', "-amcmc"], test=True)
            self.assertEqual(options['min_number_initialisation_samples'], defaults[
                             'mcmc_min_number_initialisation_samples'])
            options, options_map = MTfit_parser(
                ['Test.i', "-amcmc", '-X=1000'], test=True)
            self.assertEqual(
                options['min_number_initialisation_samples'], 1000)
            # debug
            print('debug -D check')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertTrue(options['debug'] == defaults['debug'])
            options, options_map = MTfit_parser(
                ['Test.i', '-D'], test=True)
            self.assertTrue(options['debug'])
            with self.assertRaises(SystemExit):
                MTfit_parser(["-D", 'Test.i', '-q'], test=True)
            # benchmark check
            print('benchmark -B check')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertTrue(options['benchmark'] == defaults['benchmark'])
            options, options_map = MTfit_parser(
                ['Test.i', '-B'], test=True)
            self.assertTrue(options['benchmark'])
            # quality_check
            print('quality -Q check')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertTrue(
                options['quality_check'] == defaults['quality'])
            options, options_map = MTfit_parser(
                ['Test.i', '-Q'], test=True)
            self.assertEqual(
                options['quality_check'], defaults['quality_check_value'])
            options, options_map = MTfit_parser(
                ['Test.i', '-Q=1'], test=True)
            self.assertEqual(options['quality_check'], 1)
            options, options_map = MTfit_parser(
                ['Test.i', '-Q=1', '-o=XKCD'], test=True)
            self.assertEqual(options['fid'], "XKCD")
            self.assertEqual(options['quality_check'], 1)
            # output-format
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertTrue(
                options['output_format'] == defaults['output_format'])
            options, options_map = MTfit_parser(
                ['--output-format=matlab', 'Test.i'], test=True)
            self.assertEqual(options['output_format'], 'matlab')
            options, options_map = MTfit_parser(
                ['--output-format=pickle', 'Test.i'], test=True)
            self.assertEqual(options['output_format'], 'pickle')
            with self.assertRaises(SystemExit):
                MTfit_parser(
                    ['--output-format=goadasdafqfeqwjfoijoidfawgohgkajrghaioarhgnwae', 'Test.i'], test=True)
            # results-format
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertTrue(
                options['results_format'] == defaults['results_format'])
            options, options_map = MTfit_parser(
                ['--results-format=full_pdf', 'Test.i'], test=True)
            self.assertEqual(options['results_format'], 'full_pdf')
            with self.assertRaises(SystemExit):
                MTfit_parser(
                    ['--results-format=goadasdafqfeqwjfoijoidfawgohgkajrghaioarhgnwae', 'Test.i'], test=True)
            # dc_prior
            print('dc_prior test')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertTrue(options['dc_prior'] == defaults['dc_prior'])
            options, options_map = MTfit_parser(
                ['Test.i', '--dc_prior=0.5'], test=True)
            self.assertEqual(options['dc_prior'], 0.5)
            with self.assertRaises(SystemExit):
                MTfit_parser(
                    ['Test.i', '-a=transdmcmc', '--dc_prior=1.5'], test=True)
            # normalise
            print('normalise test')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertTrue(
                options['normalise'] != defaults['no_normalise'])
            options, options_map = MTfit_parser(
                ['Test.i', '--no_normalise'], test=True)
            self.assertEqual(options['normalise'], False)
            # convert
            print('convert test')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertTrue(options['convert'] == defaults['convert'])
            options, options_map = MTfit_parser(
                ['Test.i', '--convert'], test=True)
            self.assertEqual(options['convert'], True)
            # discard
            print('discard test')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertTrue(options['discard'] == defaults['discard'])
            options, options_map = MTfit_parser(
                ['Test.i', '--discard=23000'], test=True)
            self.assertEqual(options['discard'], 23000)
            # extensions:
            print('Testing extensions for command line parsers')
            for test in extension_tests:
                test(self, MTfit_parser, defaults, _ARGPARSE)
            # qsub
            open('Test.i', 'w').write('test')
            print('qsub -q check')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertFalse(options['qsub'])
            if 'PBS_DEFAULT' not in os.environ.keys() or 'PBS_NUM_NODES' in os.environ.keys():
                with self.assertRaises(SystemExit):
                    MTfit_parser(['Test.i', '-q'], test=True)
            open('Test.inv', 'w').write('test')
            open('Test.scatangle', 'w').write('test')
            argparser._QSUBTEST = True
            if argparser._PYQSUB:
                open('Test.p123', 'w').write(example_MTfit_script)
                options, options_map = MTfit_parser(['-R', 'Test.p123'], test=True)
                self.assertTrue(options['recover'])
                self.assertEqual(options['number_location_samples'], 25)
                self.assertTrue(options['dc_mt'])
                open('Test.p124', 'w').write(example_MTfit_script2)
                options, options_map = MTfit_parser(['-R'], test=True)
                self.assertTrue(options['recover'])
                self.assertEqual(options['number_location_samples'], 250)
                self.assertFalse(options['dc_mt'])
            argparser._QSUBTEST = False
            options, options_map = MTfit_parser(['-R'], test=True)
            self.assertTrue(options['recover'])
            try:
                os.remove('Test.inv')
            except Exception:
                pass
            try:
                os.remove('Test.scatangle')
            except Exception:
                pass
            try:
                os.remove('Test.p123')
            except Exception:
                pass
            try:
                os.remove('Test.p124')
            except Exception:
                pass
            _ARGPARSE = False
            if argparser._PYQSUB:
                argparser.pyqsub.ARGPARSE = False
        if not _ARGPARSE:
            try:
                os.remove('Test.i')
            except Exception:
                pass
            try:
                os.remove('Test2.i')
            except Exception:
                pass
            print("\n\n------------OPTPARSE----------------\n\n")
            # data_file -d
            print('data_file -d check')
            with self.assertRaises(SystemExit):
                MTfit_parser(['test'], test=True)
            with self.assertRaises(SystemExit):
                MTfit_parser(['-dTest'], test=True)
            with self.assertRaises(SystemExit):
                MTfit_parser(['--datafile=Test'], test=True)
            open('Test.i', 'w').write('test')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            options, options_map = MTfit_parser(["-dTest.i"], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            options, options_map = MTfit_parser(["-dTest.i"], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            options, options_map = MTfit_parser(
                ["--datafile=Test.i"], test=True)
            self.assertEqual(options['data_file'], [os.path.abspath('Test.i')])
            options, options_map = MTfit_parser(
                ["--datafile=*.i"], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            options, options_map = MTfit_parser(["--invext=i"], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            open('Test2.i', 'w').write('test')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            options, options_map = MTfit_parser(['Test2.i'], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test2.i')])
            options, options_map = MTfit_parser(["-dTest2.i"], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test2.i')])
            options, options_map = MTfit_parser(["-dTest.i"], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            options, options_map = MTfit_parser(
                ["--datafile=Test2.i"], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test2.i')])
            options, options_map = MTfit_parser(
                ["--datafile=*.i"], test=True)
            self.assertAlmostEqual(
                options['data_file'], [os.path.abspath('Test.i'), os.path.abspath('Test2.i')])
            options, options_map = MTfit_parser(["--invext=i"], test=True)
            self.assertAlmostEqual(
                options['data_file'], [os.path.abspath('Test.i'), os.path.abspath('Test2.i')])
            os.remove('Test2.i')
            # AngleScatterFile -s
            print('location_pdf_file -s check')
            options, options_map = MTfit_parser(
                ["-dTest.i", "-s*.i"], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            self.assertEqual(
                options['location_pdf_file_path'], [os.path.abspath('Test.i')])
            options, options_map = MTfit_parser(
                ["Test.i", "-s*.i"], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            self.assertEqual(
                options['location_pdf_file_path'], [os.path.abspath('Test.i')])
            options, options_map = MTfit_parser(
                ["-s*.i", 'Test.i'], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            self.assertEqual(
                options['location_pdf_file_path'], [os.path.abspath('Test.i')])
            # algorithm -a
            print('algorithm -a check')
            with self.assertRaises(SystemExit):
                MTfit_parser(["-a=iteration", 'Test.i'], test=True)
            options, options_map = MTfit_parser(
                ["-aiterate", 'Test.i'], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            self.assertEqual(options['algorithm'], 'iterate')
            options, options_map = MTfit_parser(
                ["-atime", 'Test.i'], test=True)
            self.assertEqual(
                options['data_file'], [os.path.abspath('Test.i')])
            self.assertEqual(options['algorithm'], 'time')
            # Parallel -l
            print('parallel -l check')
            self.assertTrue(
                options['parallel'] != defaults['single_threaded'])
            options, options_map = MTfit_parser(
                ["-l", 'Test.i'], test=True)
            self.assertFalse(options['parallel'])
            options, options_map = MTfit_parser(
                ["--single", 'Test.i'], test=True)
            self.assertFalse(options['parallel'])
            options, options_map = MTfit_parser(
                ["--singlethread", 'Test.i'], test=True)
            self.assertFalse(options['parallel'])
            self.assertFalse('singlethread' in options.keys())
            # Number of Workers -n
            print('number_workers -n check')
            self.assertEqual(options['n'], defaults['number_workers'])
            options, options_map = MTfit_parser(
                ["-n1", 'Test.i'], test=True)
            self.assertEqual(options['n'], 1)
            with self.assertRaises(SystemExit):
                MTfit_parser(["-n=1.1", 'Test.i'], test=True)
            # PMem -m
            print('mem -m check')
            self.assertEqual(options['phy_mem'], defaults['memory'])
            options, options_map = MTfit_parser(
                ["-m1.5", 'Test.i'], test=True)
            self.assertEqual(options['phy_mem'], 1.5)
            # DC -c
            print('dc -c check')
            self.assertTrue(options['dc'] == defaults['double-couple'])
            options, options_map = MTfit_parser(
                ["-c", 'Test.i'], test=True)
            self.assertTrue(options['dc'])
            # NStations
            print('--nstations check')
            self.assertTrue(
                options['number_stations'] == defaults['number_stations'])
            options, options_map = MTfit_parser(
                ["--nstations=20", 'Test.i'], test=True)
            self.assertEqual(options['number_stations'], 20)
            with self.assertRaises(SystemExit):
                MTfit_parser(["--nstations=20.2", 'Test.i'], test=True)
            # NAngleSamples
            print('--nanglesamples check')
            self.assertEqual(
                options['number_location_samples'], defaults['number_location_samples'])
            options, options_map = MTfit_parser(
                ["--nanglesamples=20", 'Test.i'], test=True)
            self.assertEqual(options['number_location_samples'], 20)
            options, options_map = MTfit_parser(
                ["--number_location_samples=20", 'Test.i'], test=True)
            self.assertEqual(options['number_location_samples'], 20)
            with self.assertRaises(SystemExit):
                MTfit_parser(["--nanglesamples=20.2", 'Test.i'], test=True)
            # inversion_options -i
            print('inversion_options -i check')
            if defaults['inversion_options']:
                self.assertTrue(options['inversion_options'] == defaults['inversion_options'].split(
                    ','), str(options['inversion_options'])+' '+str(defaults['inversion_options']))
            else:
                self.assertTrue(options['inversion_options'] == defaults['inversion_options'], str(
                    options['inversion_options'])+' '+str(defaults['inversion_options']))
            options, options_map = MTfit_parser(
                ["-i=PPolarity", 'Test.i'], test=True)
            options, options_map = MTfit_parser(
                ["-iPPolarity", 'Test.i'], test=True)
            self.assertEqual(options['inversion_options'], ['PPolarity'])
            options, options_map = MTfit_parser(
                ["--inversionoptions=PPolarity,PSHRMSAmplitudeRatio", 'Test.i'], test=True)
            self.assertEqual(
                options['inversion_options'], ['PPolarity', 'PSHRMSAmplitudeRatio'])
            # file_sampling check
            print('file_sampling -f check')
            self.assertTrue(
                options['file_sample'] == defaults['disk_sample'])
            options, options_map = MTfit_parser(
                ["-f", 'Test.i'], test=True)
            self.assertTrue(options['file_sample'])
            options, options_map = MTfit_parser(
                ["--file-sample", 'Test.i'], test=True)
            self.assertTrue(options['file_sample'])
            with self.assertRaises(SystemExit):
                MTfit_parser(
                    ["--file-sample", 'Test.i', '-a=mcmc'], test=True)
            # FID -o
            print('output file -o check')
            self.assertTrue(options['fid'] == defaults['output_file'])
            options, options_map = MTfit_parser(
                ["-oXKCD", 'Test.i'], test=True)
            self.assertEqual(options['fid'], "XKCD")
            options, options_map = MTfit_parser(
                ["--fid=XKCD", 'Test.i'], test=True)
            self.assertEqual(options['fid'], "XKCD")
            # max_samples -x
            print('max_samples -x check')
            with self.assertRaises(KeyError):
                options['max_samples']
            options, options_map = MTfit_parser(
                ["-aiterate", 'Test.i'], test=True)
            options, options_map = MTfit_parser(
                ["-aiterate", "-x100", 'Test.i'], test=True)
            self.assertEqual(options['max_samples'], 100)
            options, options_map = MTfit_parser(
                ["-aiterate", "--maxsamples=100", 'Test.i'], test=True)
            self.assertEqual(options['max_samples'], 100)
            with self.assertRaises(SystemExit):
                MTfit_parser(["-x20.2", 'Test.i'], test=True)
            # max_time -t
            print('max_time -t check')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            options, options_map = MTfit_parser(
                ["-aiterate", 'Test.i'], test=True)
            with self.assertRaises(KeyError):
                options['max_time']
            options, options_map = MTfit_parser(
                ["-aiterate", "-t100", 'Test.i'], test=True)
            with self.assertRaises(KeyError):
                options['max_time']
            if defaults['algorithm'] != 'time':
                options, options_map = MTfit_parser(
                    ["-a=time", "-t100", 'Test.i'], test=True)
            else:
                options, options_map = MTfit_parser(
                    ["-t100", 'Test.i'], test=True)
            self.assertEqual(options['max_time'], 100)
            if defaults['algorithm'] != 'time':
                options, options_map = MTfit_parser(
                    ["-a=time", "--maxtime=100.1", 'Test.i'], test=True)
            else:
                options, options_map = MTfit_parser(
                    ["--maxtime=100.1", 'Test.i'], test=True)
            self.assertEqual(options['max_time'], 100.1)
            if defaults['algorithm'] != 'time':
                with self.assertRaises(SystemExit):
                    MTfit_parser(["-a=time", "-ta", 'Test.i'], test=True)
            else:
                with self.assertRaises(SystemExit):
                    MTfit_parser(["-ta", 'Test.i'], test=True)
            # multiple events
            print('multiple events test')
            options, options_map = MTfit_parser(
                ["-amcmc", "-e", 'Test.i'], test=True)
            self.assertTrue(options['multiple_events'])
            # relative amplitudes
            print('relative amplitudes test')
            options, options_map = MTfit_parser(
                ["-amcmc", "-r", 'Test.i'], test=True)
            self.assertTrue(options['relative_amplitude'])
            # marginalise relative test
            print('marginalise relative test')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertFalse(options['marginalise_relative'])
            options, options_map = MTfit_parser(
                ["-amcmc", "-r", "--marginalise-relative", 'Test.i'], test=True)
            self.assertTrue(options['marginalise_relative'])
            with self.assertRaises(SystemExit):
                MTfit_parser(
                    ["-amcmc", "--marginalise-relative", 'Test.i'], test=True)
            # min number intersections
            print('minimum_number_intersections -S check')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertEqual(options['minimum_number_intersections'], 2)
            options, options_map = MTfit_parser(
                ['Test.i', '-S1'], test=True)
            self.assertEqual(options['minimum_number_intersections'], 1)
            with self.assertRaises(SystemExit):
                MTfit_parser(["-S1", 'Test.i', '-r'])
            options, options_map = MTfit_parser(
                ['Test.i', '-S10', '-r'], test=True)
            self.assertEqual(options['minimum_number_intersections'], 10)
            # McMC options
            print('McMC options check')
            options, options_map = MTfit_parser(
                ["-amcmc", "-x100", 'Test.i'], test=True)
            self.assertEqual(options['chain_length'], 100)
            self.assertEqual(
                options['dimension_jump_prob'], defaults['jump_probability'])
            self.assertEqual(
                options['min_acceptance_rate'], defaults['mcmc_min_acceptance_rate'])
            self.assertEqual(
                options['max_acceptance_rate'], defaults['mcmc_max_acceptance_rate'])
            self.assertEqual(
                options['acceptance_rate_window'], defaults['acceptance_rate_window'])
            self.assertEqual(
                options['learning_length'], defaults['learning_length'])
            self.assertEqual(options['algorithm'], 'mcmc')
            with self.assertRaises(SystemExit):
                MTfit_parser(["-u=T"], test=True)
            with self.assertRaises(SystemExit):
                MTfit_parser(["-v=T"], test=True)
            with self.assertRaises(SystemExit):
                MTfit_parser(["-w=T"], test=True)
            with self.assertRaises(SystemExit):
                MTfit_parser(["-j=T"], test=True)
            with self.assertRaises(SystemExit):
                MTfit_parser(["-z=T"], test=True)
            with self.assertRaises(SystemExit):
                MTfit_parser(["-z=10.1"], test=True)
            options, options_map = MTfit_parser(
                ["-atransdmcmc", "-u0.1", "-v0.8", "-w5000", "-j0.2", '-z100', 'Test.i'], test=True)
            self.assertEqual(
                options['chain_length'], defaults['transdmcmc_chain_length'])
            self.assertEqual(options['algorithm'], 'transdmcmc')
            self.assertEqual(options['dimension_jump_prob'], 0.2)
            self.assertEqual(options['min_acceptance_rate'], 0.1)
            self.assertEqual(options['max_acceptance_rate'], 0.8)
            self.assertEqual(options['acceptance_rate_window'], 5000)
            self.assertEqual(options['learning_length'], 100)
            options, options_map = MTfit_parser(
                ["-atransdmcmc", 'Test.i'], test=True)
            self.assertEqual(
                options['dimension_jump_prob'], defaults['jump_probability'])
            self.assertEqual(
                options['min_acceptance_rate'], defaults['transdmcmc_min_acceptance_rate'])
            self.assertEqual(
                options['max_acceptance_rate'], defaults['transdmcmc_max_acceptance_rate'])
            self.assertEqual(
                options['acceptance_rate_window'], defaults['acceptance_rate_window'])
            self.assertEqual(
                options['learning_length'], defaults['learning_length'])
            print('Warnings -W check')
            options, options_map = MTfit_parser(
                ["-Wd", 'Test.i'], test=True)
            self.assertEqual(options['warnings'], 'd')
            options, options_map = MTfit_parser(
                ["-Wignore", 'Test.i'], test=True)
            self.assertEqual(options['warnings'], 'i')
            with self.assertRaises(SystemExit):
                MTfit_parser(["-WABC", 'Test.i'], test=True)
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertEqual(options['warnings'], 'd')
            # min_number_initialisation_samples -X
            print("min_number_initialisation/check_samples -X check")
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertEqual(
                options['min_number_check_samples'], defaults['min_number_check_samples'])
            options, options_map = MTfit_parser(
                ['Test.i', '-X1000'], test=True)
            self.assertEqual(options['min_number_check_samples'], 1000)
            options, options_map = MTfit_parser(
                ['Test.i', "-amcmc"], test=True)
            self.assertEqual(options['min_number_initialisation_samples'], defaults[
                             'mcmc_min_number_initialisation_samples'])
            options, options_map = MTfit_parser(
                ['Test.i', "-amcmc", '-X1000'], test=True)
            self.assertEqual(
                options['min_number_initialisation_samples'], 1000)
            # debug
            print('debug -D check')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertTrue(options['debug'] == defaults['debug'])
            options, options_map = MTfit_parser(
                ['Test.i', '-D'], test=True)
            self.assertTrue(options['debug'])
            with self.assertRaises(SystemExit):
                MTfit_parser(["-D", 'Test.i', '-q'], test=True)
            # benchmark check
            print('benchmark -B check')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertTrue(options['benchmark'] == defaults['benchmark'])
            options, options_map = MTfit_parser(
                ['Test.i', '-B'], test=True)
            self.assertTrue(options['benchmark'])
            # quality_check
            print('quality -Q check')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertTrue(
                options['quality_check'] == defaults['quality'])
            options, options_map = MTfit_parser(
                ['Test.i', '-Q1'], test=True)
            self.assertEqual(options['quality_check'], 1)
            options, options_map = MTfit_parser(
                ['Test.i', '-Q1', '-oXKCD'], test=True)
            self.assertEqual(options['fid'], "XKCD")
            self.assertEqual(options['quality_check'], 1)
            # output-format
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertTrue(
                options['output_format'] == defaults['output_format'])
            options, options_map = MTfit_parser(
                ['--output-format=matlab', 'Test.i'], test=True)
            self.assertEqual(options['output_format'], 'matlab')
            options, options_map = MTfit_parser(
                ['--output-format=pickle', 'Test.i'], test=True)
            self.assertEqual(options['output_format'], 'pickle')
            # results-format
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertTrue(
                options['results_format'] == defaults['results_format'])
            options, options_map = MTfit_parser(
                ['--results-format=full_pdf', 'Test.i'], test=True)
            self.assertEqual(options['results_format'], 'full_pdf')
            with self.assertRaises(SystemExit):
                MTfit_parser(
                    ['--results-format=goadasdafqfeqwjfoijoidfawgohgkajrghaioarhgnwae', 'Test.i'], test=True)
            # dc_prior
            print('dc_prior test')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertTrue(options['dc_prior'] == defaults['dc_prior'])
            options, options_map = MTfit_parser(
                ['Test.i', '--dc_prior=0.5'], test=True)
            self.assertEqual(options['dc_prior'], 0.5)
            with self.assertRaises(SystemExit):
                MTfit_parser(
                    ['Test.i', '-atransdmcmc', '--dc_prior=1.5'], test=True)
            # normalise
            print('normalise test')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertTrue(
                options['normalise'] != defaults['no_normalise'])
            options, options_map = MTfit_parser(
                ['Test.i', '--no_normalise'], test=True)
            self.assertEqual(options['normalise'], False)
            # convert
            print('convert test')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertTrue(options['convert'] == defaults['convert'])
            options, options_map = MTfit_parser(
                ['Test.i', '--convert'], test=True)
            self.assertEqual(options['convert'], True)
            # discard
            print('discard test')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertTrue(options['discard'] == defaults['discard'])
            options, options_map = MTfit_parser(
                ['Test.i', '--discard=23000'], test=True)
            self.assertEqual(options['discard'], 23000)
            # extensions:
            print('Testing extensions for command line parsers')
            for test in extension_tests:
                test(self, MTfit_parser, defaults, _ARGPARSE)
            open('Test.i', 'w').write('test')
            # qsub
            print('qsub -q check')
            options, options_map = MTfit_parser(['Test.i'], test=True)
            self.assertFalse(options['qsub'])
            if 'PBS_DEFAULT' not in os.environ.keys() or 'PBS_NUM_NODES' in os.environ.keys():
                with self.assertRaises(SystemExit):
                    MTfit_parser(['Test.i', '-q'], test=True)
            os.remove('Test.i')
            open('Test.inv', 'w').write('test')
            open('Test.scatangle', 'w').write('test')
            argparser._QSUBTEST = True
            if argparser._PYQSUB:
                open('Test.p123', 'w').write(example_MTfit_script)
                options, options_map = MTfit_parser(
                    ['-R', 'Test.p123'], test=True)
                self.assertTrue(options['recover'])
                self.assertEqual(options['number_location_samples'], 25)
                self.assertTrue(options['dc_mt'])
                open('Test.p124', 'w').write(example_MTfit_script2)
                options, options_map = MTfit_parser(['-R'], test=True)
                self.assertTrue(options['recover'])
                self.assertEqual(options['number_location_samples'], 250)
                self.assertFalse(options['dc_mt'])
            argparser._QSUBTEST = False
            options, options_map = MTfit_parser(['-R'], test=True)
            self.assertTrue(options['recover'])
            try:
                os.remove('Test.inv')
            except Exception:
                pass
            try:
                os.remove('Test.scatangle')
            except Exception:
                pass
            try:
                os.remove('Test.p123')
            except Exception:
                pass
            try:
                os.remove('Test.p124')
            except Exception:
                pass
            _ARGPARSE = _argparse
            if argparser._PYQSUB:
                argparser.pyqsub.ARGPARSE = _argparse

    def test_MTplot_parser(self):
        from MTfit.utilities.argparser import MTplot_parser
        from MTfit.utilities.argparser import get_MTplot_defaults
        from MTfit.utilities.argparser import evaluate_extensions
        cmd_defaults = {}
        cmd_default_types = {}
        results = evaluate_extensions('MTfit.cmd_defaults', default_cmd_defaults)
        for result in results:
            cmd_defaults.update(result[0])
            cmd_default_types.update(result[1])
        defaults = get_MTplot_defaults(True, cmd_defaults, cmd_default_types)
        # data_file -d
        global _ARGPARSE
        _argparse = _ARGPARSE is True
        if _argparse:
            print('data_file -d check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['data_file'], 'test')
            self.assertEqual(
                MTplot_parser(['-d=Test'], test=True)['data_file'], 'Test')
            self.assertEqual(
                MTplot_parser(['--datafile=Test'], test=True)['data_file'], 'Test')
            print('plot-type check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['plot_type'], defaults['plot_type'])
            self.assertEqual(
                MTplot_parser(['test', '-p=hudson'], test=True)['plot_type'], 'hudson')
            self.assertEqual(MTplot_parser(
                ['test', '--plot_type=faultplane'], test=True)['plot_type'], 'faultplane')
            print('colormap check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['colormap'], DEFAULT_AMP_COLORMAP)
            self.assertEqual(
                MTplot_parser(['test', '-c=jet'], test=True)['colormap'], 'jet')
            self.assertEqual(
                MTplot_parser(['test', '--color_map=bwr'], test=True)['colormap'], 'bwr')
            print('fontsize check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['fontsize'], defaults['fontsize'])
            self.assertEqual(
                MTplot_parser(['test', '-f=12'], test=True)['fontsize'], 12)
            self.assertEqual(
                MTplot_parser(['test', '--fontsize=8'], test=True)['fontsize'], 8)
            print('linewidth check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['linewidth'], defaults['linewidth'])
            self.assertEqual(
                MTplot_parser(['test', '-l=12'], test=True)['linewidth'], 12)
            self.assertEqual(
                MTplot_parser(['test', '--linewidth=8'], test=True)['linewidth'], 8)
            print('text check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['text'], defaults['text'])
            self.assertEqual(
                MTplot_parser(['test', '-t'], test=True)['text'], True)
            self.assertEqual(
                MTplot_parser(['test', '--text'], test=True)['text'], True)
            print('resolution check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['resolution'], defaults['resolution'])
            self.assertEqual(
                MTplot_parser(['test', '-r=12'], test=True)['resolution'], 12)
            self.assertEqual(
                MTplot_parser(['test', '--resolution=8'], test=True)['resolution'], 8)
            with self.assertRaises(SystemExit):
                MTplot_parser(['test', '--resolution=8.4'], test=True)
            print('bins check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['bins'], defaults['bins'])
            self.assertEqual(
                MTplot_parser(['test', '-b=12'], test=True)['bins'], 12)
            self.assertEqual(
                MTplot_parser(['test', '--bins=8'], test=True)['bins'], 8)
            with self.assertRaises(SystemExit):
                MTplot_parser(['test', '--bins=8.4'], test=True)
            print('fault_plane check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['fault_plane'], defaults['fault_plane'])
            self.assertEqual(
                MTplot_parser(['test', '--fault_plane'], test=True)['fault_plane'], True)
            print('nodal_line check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['nodal_line'], defaults['nodal_line'])
            self.assertEqual(
                MTplot_parser(['test', '-n'], test=True)['nodal_line'], True)
            self.assertEqual(
                MTplot_parser(['test', '--nodal_line'], test=True)['nodal_line'], True)
            print('TNP check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['TNP'], defaults['TNP'])
            self.assertEqual(
                MTplot_parser(['test', '--pt'], test=True)['TNP'], True)
            self.assertEqual(
                MTplot_parser(['test', '--tnp'], test=True)['TNP'], True)
            print('markersize check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['markersize'], defaults['markersize'])
            self.assertEqual(
                MTplot_parser(['test', '--markersize=8'], test=True)['markersize'], 8)
            self.assertEqual(
                MTplot_parser(['test', '--markersize=8.4'], test=True)['markersize'], 8.4)
            print('station_markersize check')
            self.assertEqual(MTplot_parser(['test'], test=True)[
                             'station_markersize'], defaults['station_markersize'])
            self.assertEqual(MTplot_parser(
                ['test', '--station_markersize=8'], test=True)['station_markersize'], 8)
            self.assertEqual(MTplot_parser(
                ['test', '--station_markersize=8.4'], test=True)['station_markersize'], 8.4)
            print('show_max_likelihood check')
            self.assertEqual(MTplot_parser(['test'], test=True)[
                             'show_max_likelihood'], defaults['show_max_likelihood'])
            self.assertEqual(MTplot_parser(
                ['test', '--showmaxlikelihood'], test=True)['show_max_likelihood'], True)
            self.assertEqual(MTplot_parser(
                ['test', '--show_max_likelihood'], test=True)['show_max_likelihood'], True)
            print('show_mean check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['show_mean'], defaults['show_mean'])
            self.assertEqual(
                MTplot_parser(['test', '--showmean'], test=True)['show_mean'], True)
            self.assertEqual(
                MTplot_parser(['test', '--show_mean'], test=True)['show_mean'], True)
            print('grid_lines check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['grid_lines'], defaults['grid_lines'])
            self.assertEqual(
                MTplot_parser(['test', '--gridlines'], test=True)['grid_lines'], True)
            self.assertEqual(
                MTplot_parser(['test', '--grid_lines'], test=True)['grid_lines'], True)
            print('color check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['color'], defaults['color'])
            self.assertEqual(
                MTplot_parser(['test', '--color=b'], test=True)['color'], 'b')
            self.assertEqual(
                MTplot_parser(['test', '--color=purple'], test=True)['color'], 'purple')
            print('type_label check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['type_label'], defaults['type_label'])
            self.assertEqual(
                MTplot_parser(['test', '--typelabel'], test=True)['type_label'], True)
            self.assertEqual(
                MTplot_parser(['test', '--type_label'], test=True)['type_label'], True)
            print('hex_bin check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['hex_bin'], defaults['hex_bin'])
            self.assertEqual(
                MTplot_parser(['test', '--hexbin'], test=True)['hex_bin'], True)
            self.assertEqual(
                MTplot_parser(['test', '--hex_bin'], test=True)['hex_bin'], True)
            print('projection check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['projection'], defaults['projection'])
            self.assertEqual(
                MTplot_parser(['test', '--projection=uv'], test=True)['projection'], 'uv')
            self.assertEqual(MTplot_parser(
                ['test', '--projection=equalangle'], test=True)['projection'], 'equalangle')
            print('save_file check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['save_file'], '')
            self.assertEqual(MTplot_parser(
                ['test', '--save_file=test.png'], test=True)['save_file'], 'test.png')
            print('save_dpi check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['save_dpi'], 200)
            self.assertEqual(
                MTplot_parser(['test', '--save_dpi=150'], test=True)['save_dpi'], 150)
            print('hide')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['show'], True)
            self.assertEqual(
                MTplot_parser(['test', '-q'], test=True)['show'], False)
            self.assertEqual(
                MTplot_parser(['test', '--quiet'], test=True)['show'], False)
            self.assertEqual(
                MTplot_parser(['test', '--hide'], test=True)['show'], False)
            _ARGPARSE = False
        if not _ARGPARSE:
            print('data_file -d check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['data_file'], 'test')
            self.assertEqual(
                MTplot_parser(['-dTest'], test=True)['data_file'], 'Test')
            self.assertEqual(
                MTplot_parser(['--datafile=Test'], test=True)['data_file'], 'Test')
            print('plot-type check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['plot_type'], defaults['plot_type'])
            self.assertEqual(
                MTplot_parser(['test', '-phudson'], test=True)['plot_type'], 'hudson')
            self.assertEqual(MTplot_parser(
                ['test', '--plot_type=faultplane'], test=True)['plot_type'], 'faultplane')
            print('colormap check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['colormap'], DEFAULT_AMP_COLORMAP)
            self.assertEqual(
                MTplot_parser(['test', '-cjet'], test=True)['colormap'], 'jet')
            self.assertEqual(
                MTplot_parser(['test', '--color_map=bwr'], test=True)['colormap'], 'bwr')
            print('fontsize check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['fontsize'], defaults['fontsize'])
            self.assertEqual(
                MTplot_parser(['test', '-f12'], test=True)['fontsize'], 12)
            self.assertEqual(
                MTplot_parser(['test', '--fontsize=8'], test=True)['fontsize'], 8)
            print('linewidth check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['linewidth'], defaults['linewidth'])
            self.assertEqual(
                MTplot_parser(['test', '-l12'], test=True)['linewidth'], 12)
            self.assertEqual(
                MTplot_parser(['test', '--linewidth=8'], test=True)['linewidth'], 8)
            print('text check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['text'], defaults['text'])
            self.assertEqual(
                MTplot_parser(['test', '-t'], test=True)['text'], True)
            self.assertEqual(
                MTplot_parser(['test', '--text'], test=True)['text'], True)
            print('resolution check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['resolution'], defaults['resolution'])
            self.assertEqual(
                MTplot_parser(['test', '-r12'], test=True)['resolution'], 12)
            self.assertEqual(
                MTplot_parser(['test', '--resolution=8'], test=True)['resolution'], 8)
            with self.assertRaises(SystemExit):
                MTplot_parser(['test', '--resolution=8.4'], test=True)
            print('bins check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['bins'], defaults['bins'])
            self.assertEqual(
                MTplot_parser(['test', '-b12'], test=True)['bins'], 12)
            self.assertEqual(
                MTplot_parser(['test', '--bins=8'], test=True)['bins'], 8)
            with self.assertRaises(SystemExit):
                MTplot_parser(['test', '--bins=8.4'], test=True)
            print('fault_plane check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['fault_plane'], defaults['fault_plane'])
            self.assertEqual(
                MTplot_parser(['test', '--fault_plane'], test=True)['fault_plane'], True)
            print('nodal_line check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['nodal_line'], defaults['nodal_line'])
            self.assertEqual(
                MTplot_parser(['test', '-n'], test=True)['nodal_line'], True)
            self.assertEqual(
                MTplot_parser(['test', '--nodal_line'], test=True)['nodal_line'], True)
            print('TNP check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['TNP'], defaults['TNP'])
            self.assertEqual(
                MTplot_parser(['test', '--pt'], test=True)['TNP'], True)
            self.assertEqual(
                MTplot_parser(['test', '--tnp'], test=True)['TNP'], True)
            print('markersize check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['markersize'], defaults['markersize'])
            self.assertEqual(
                MTplot_parser(['test', '--markersize=8'], test=True)['markersize'], 8)
            self.assertEqual(
                MTplot_parser(['test', '--markersize=8.4'], test=True)['markersize'], 8.4)
            print('station_markersize check')
            self.assertEqual(MTplot_parser(['test'], test=True)[
                             'station_markersize'], defaults['station_markersize'])
            self.assertEqual(MTplot_parser(
                ['test', '--station_markersize=8'], test=True)['station_markersize'], 8)
            self.assertEqual(MTplot_parser(
                ['test', '--station_markersize=8.4'], test=True)['station_markersize'], 8.4)
            print('show_max_likelihood check')
            self.assertEqual(MTplot_parser(['test'], test=True)[
                             'show_max_likelihood'], defaults['show_max_likelihood'])
            self.assertEqual(MTplot_parser(
                ['test', '--showmaxlikelihood'], test=True)['show_max_likelihood'], True)
            self.assertEqual(MTplot_parser(
                ['test', '--show_max_likelihood'], test=True)['show_max_likelihood'], True)
            print('show_mean check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['show_mean'], defaults['show_mean'])
            self.assertEqual(
                MTplot_parser(['test', '--showmean'], test=True)['show_mean'], True)
            self.assertEqual(
                MTplot_parser(['test', '--show_mean'], test=True)['show_mean'], True)
            print('grid_lines check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['grid_lines'], defaults['grid_lines'])
            self.assertEqual(
                MTplot_parser(['test', '--gridlines'], test=True)['grid_lines'], True)
            self.assertEqual(
                MTplot_parser(['test', '--grid_lines'], test=True)['grid_lines'], True)
            print('color check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['color'], defaults['color'])
            self.assertEqual(
                MTplot_parser(['test', '--color=b'], test=True)['color'], 'b')
            self.assertEqual(
                MTplot_parser(['test', '--color=purple'], test=True)['color'], 'purple')
            print('type_label check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['type_label'], defaults['type_label'])
            self.assertEqual(
                MTplot_parser(['test', '--typelabel'], test=True)['type_label'], True)
            self.assertEqual(
                MTplot_parser(['test', '--type_label'], test=True)['type_label'], True)
            print('hex_bin check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['hex_bin'], defaults['hex_bin'])
            self.assertEqual(
                MTplot_parser(['test', '--hexbin'], test=True)['hex_bin'], True)
            self.assertEqual(
                MTplot_parser(['test', '--hex_bin'], test=True)['hex_bin'], True)
            print('projection check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['projection'], defaults['projection'])
            self.assertEqual(
                MTplot_parser(['test', '--projection=uv'], test=True)['projection'], 'uv')
            self.assertEqual(MTplot_parser(
                ['test', '--projection=equalangle'], test=True)['projection'], 'equalangle')
            print('save_file check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['save_file'], '')
            self.assertEqual(MTplot_parser(
                ['test', '--save_file=test.png'], test=True)['save_file'], 'test.png')
            print('save_dpi check')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['save_dpi'], 200)
            self.assertEqual(
                MTplot_parser(['test', '--save_dpi=150'], test=True)['save_dpi'], 150)
            print('hide')
            self.assertEqual(
                MTplot_parser(['test'], test=True)['show'], True)
            self.assertEqual(
                MTplot_parser(['test', '-q'], test=True)['show'], False)
            self.assertEqual(
                MTplot_parser(['test', '--quiet'], test=True)['show'], False)
            self.assertEqual(
                MTplot_parser(['test', '--hide'], test=True)['show'], False)
            _ARGPARSE = _argparse

