# update_tensorboard.py

"""
Reinstall tensorboard from source. Also copy over a file that has a different default parameter.
This is to account for texts added from tb.add_text() not showing up.

Issue: https://github.com/lanpa/tensorboardX/issues/191

Difference in tensorboard_application.py that's being copied over:
Use 100 instead of 10.

DEFAULT_SIZE_GUIDANCE = {
    event_accumulator.TENSORS: 100,
}

Usage:
python update_tensorboard.py
"""

import subprocess
import sys

def execute_cmd(cmd):
    p = subprocess.call(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# Get path of backend/application.py that we are going to update
py_path = sys.executable  # '/usr/local/google/home/chueric/anaconda3/envs/berg/'
py_path = sys.executable.split('bin')[0]
py_version = 'python{}.{}'.format(sys.version_info.major, sys.version_info.minor)
tb_appfile_path = '{}/lib/{}/site-packages/tensorboard/backend/application.py'.format(py_path, py_version)
cmd = 'cp tensorboard_application.py {}'.format(tb_appfile_path)
print(cmd)
execute_cmd(cmd)

print('Reinstalling tensorboardx from source')
cmd = 'pip uninstall --yes tensorboardX'
execute_cmd(cmd)
cmd = 'pip install git+https://github.com/lanpa/tensorboard-pytorch'
execute_cmd(cmd)