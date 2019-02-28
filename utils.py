# utils.py

"""
General utility functions
"""
import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import pdb
import pickle
import shutil
import subprocess
import sys
import uuid


#################################################
#
# Miscellaneous
#
#################################################

def update_moving_avg(avg_so_far, new_val, n):
    # First time, n = 1
    new_avg = avg_so_far * (n - 1) / float(n) + new_val / float(n)
    return new_avg


#################################################
#
# Simple I/O
#
#################################################

def save_file(data, path, verbose=False):
    dir = os.path.dirname(path)
    if not os.path.isdir(dir):
        os.makedirs(dir)

    if verbose:
        print('Saving: {}'.format(path))

    _, ext = os.path.splitext(path)
    if ext == '.pkl':
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=2)
    elif ext == '.json':
        with open(path, 'w') as f:
            json.dump(data, f, indent=4, separators=(',', ': '), sort_keys=True)
            f.write('\n')  # add trailing newline for POSIX compatibility


def load_file(path, append_path=None):
    _, ext = os.path.splitext(path)
    if ext == '.pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)
    elif ext == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
    return data


#################################################
#
# Hyperparams and saving experiments data
#
#################################################

def sync_run_data_to_bigstore(run_dir, exp_sub_dir='', method='rsync', tb_only=True):
    """
    Save everything but the (large) models to the Bigstore bucket periodically during training. This way
    we can rsync locally and view the tensorboard results.

    Args:
        run_dir: str (e.g. checkpoints/sum/mlstm/yelp/<name-of-experiment>)
        exp_sub_dir: str
            sub directory within <bigstore> to save to, i.e.
            checkpoints/sum/mlstm/yelp/<exp_sub_dir>/<name-of-experiment>
        method: str ('cp' or 'rsync')
        tb_only: boolean (only rsync the tensorboard directory
    """

    def execute_cmd(cmd):
        p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    dirname = os.path.dirname(run_dir)  # checkpoints/sum/mlstm/yelp/
    basename = os.path.basename(run_dir)  # batch_size_4..
    bigstore_exp_dir = os.path.join('gs://{}'.format(os.environ['BS_UNSUP_BUCKET']), dirname, exp_sub_dir)
    bigstore_full_dir = os.path.join(bigstore_exp_dir, basename)
    # <bigstore>/checkpoints/sum/mlstm/yelp/<exp_sub_dir>/batch_size_4/..

    # First time -- we have to copy to create directory on Bigstore. We need a dummy file though
    # (Technically, there's no directories)
    if method == 'cp':
        if exp_sub_dir != '':  # create <exp_sub_dir> subdirectory
            Path('/tmp/placeholder.txt').touch()
            cmd = 'gsutil cp /tmp/placeholder.txt {}/'.format(bigstore_exp_dir)
            print(cmd)
            execute_cmd(cmd)

        cmd = 'gsutil cp -r  {} {}'.format(run_dir, bigstore_full_dir)
        print(cmd)
        execute_cmd(cmd)

    elif method == 'rsync':
        src_dir = os.path.join(run_dir, 'tensorboard') if tb_only else run_dir
        dest_dir = os.path.join(bigstore_full_dir, 'tensorboard') if tb_only else bigstore_full_dir
        cmd = "gsutil rsync -r -x '.*\.pt$' {} {}".format(src_dir, dest_dir)
        print(cmd)
        execute_cmd(cmd)


def copy_tree_ignore_except(src_dir, dest_dir,
                            file_exts=['.py'],
                            ignore_dirs=['checkpoints', 'external', 'datasets', 'stable_checkpoints', 'outputs']):
    """
    Same as shutil.copytree except that we only copy files have an extension found in file_exts

    Args:
        src_dir: str (path)
        dest_dir: str (path)
        file_exts: list of strs
        ignore_dirs: list of strs
    """
    print('Copying tree from "{}" to "{}"'.format(src_dir, dest_dir))
    print('Keeping only files with the following extensions: {}'.format(', '.join(file_exts)))
    print('Ignoring the following directories completely: {}'.format(', '.join(ignore_dirs)))

    def ignore_filter(cur_dir, contents):
        # contents are from os.listdir() and could be files or directories

        # ignore this directory completely
        if os.path.basename(cur_dir) in ignore_dirs:
            return contents

        ignored = []
        for c in contents:
            if c in ignore_dirs:
                continue
            if not os.path.isdir(os.path.join(cur_dir, c)):  # isn't a directory
                # ignore files that don't have desired extension
                ignore = True
                for ext in file_exts:
                    if c.endswith(ext):
                        ignore = False
                if ignore:
                    ignored.append(c)
        return ignored

    # ignore is a callable that receives directory being visited, and list of its contents
    shutil.copytree(src_dir, dest_dir, ignore=ignore_filter)


class FlushFile(object):
    """
    Wrapper around a opened file object that flushes every time write() is called. Currently in python3, std i/o
    must be buffered -- this means if stdout is redirected to a file for a long-running program, results will not show
    up in real-time in the file.

    Other options are to a) call "sys.stdout.flush()" periodically, b) call print() with
    flush=True everytime, c) supposedly run with python -u. Options 1 and 2 are are cumbersome, and 3 didn't work
    for me -- maybe it's a python3 issue again. Plus, I don't like that the flag could be forgotten.
    """

    def __init__(self, f):
        self.f = f

    def write(self, x):
        self.f.write(x)
        self.f.flush()

    def flush(self):
        """
        If exception is thrown or Ctrl+C exits, python flushes all open files. This isn't necessary since
        the buffer will be empty, but do this so AttributeError: 'FlushFile' object has no attribute 'flush'
        isn't shown.
        """
        self.f.flush()


def save_run_data(path_to_dir, hp):
    """
    1) Save stdout to file
    2) Save files to path_to_dir:
        - code_snapshot/: Snapshot of code (.py files)
        - hp.json: dict of HParams object
        - run_details.txt: command used and start time
    """
    print('Saving run data to: {}'.format(path_to_dir))
    if os.path.isdir(path_to_dir):
        print('Data already exists in this directory (presumably from a previous run)')
        inp = input('Enter "y" if you are sure you want to remove all the old contents: ')
        if inp == 'y':
            print('Removing old contents')
            shutil.rmtree(path_to_dir)
        else:
            print('Exiting')
            raise SystemExit
    print('Creating directory and saving data')
    os.mkdir(path_to_dir)

    # Redirect stdout (print statements) to file
    # if not hp.debug:
    #     sys.stdout = FlushFile(open(os.path.join(path_to_dir, 'stdout.txt'), 'w'))

    # Save snapshot of code
    snapshot_dir = os.path.join(path_to_dir, 'code_snapshot')
    if os.path.exists(snapshot_dir):  # shutil doesn't work if dest already exists
        shutil.rmtree(snapshot_dir)
    copy_tree_ignore_except('.', snapshot_dir)

    # Save hyperparms
    save_file(vars(hp), os.path.join(path_to_dir, 'hp.json'), verbose=True)

    # Save some command used to run, start time
    with open(os.path.join(path_to_dir, 'run_details.txt'), 'w') as f:
        f.write('Command:\n')
        cmd = ' '.join(sys.argv)
        start_time = datetime.now().strftime('%B%d_%H-%M-%S')
        f.write(cmd + '\n')
        f.write('Start time: {}'.format(start_time))
        print('Command used to start program:\n', cmd)
        print('Start time: {}'.format(start_time))


def create_argparse_and_update_hp(hp):
    """
    Args:
        hp: instance of HParams object

    Returns:
        (updated) hp
        run_name: str (can be used to create directory and store training results)
        parser: argparse object (can be used to add more arguments)
    """

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Create argparse with an option for every param in hp
    parser = argparse.ArgumentParser()
    for param, default_value in vars(hp).items():
        param_type = type(default_value)
        param_type = str2bool if param_type == bool else param_type
        parser.add_argument('--{}'.format(param), dest=param, default=None, type=param_type)
    opt, unknown = parser.parse_known_args()

    # Update hp if any command line arguments passed
    # Also create description of run
    run_name = []
    for param, value in vars(opt).items():
        if value is not None:
            setattr(hp, param, value)
            if param != 'model_type':
                run_name.append('{}_{}'.format(param, value))
    run_name = '-'.join(sorted(run_name))
    run_name = ('default_' + str(uuid.uuid4())[:8]) if (run_name == '') else run_name

    return hp, run_name, parser
