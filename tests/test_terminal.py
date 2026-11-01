"""Real-terminal input contracts: navigation and restoring terminal settings."""
from __future__ import annotations

import os
from pathlib import Path
import select
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(os.name != 'posix', reason='Live menu uses POSIX terminal control')
@pytest.mark.parametrize(('keys', 'result'), [(b'\x1b[A\r', 'baseline'), (b'1\r', 'baseline'),
                                            (b'\x1b', 'cancelled')])
def test_live_menu_restores_terminal(keys: bytes, result: str) -> None:
    """Arrow/name selection and cancellation always restore canonical terminal input."""
    import pty
    import termios

    master, slave = pty.openpty()
    before = termios.tcgetattr(slave)
    environment = dict(os.environ, TERM='xterm-256color', COLUMNS='88', LINES='40')
    environment.pop('NO_COLOR', None)
    child = '''
from src.app.terminal import ModelSelectionCancelled, Terminal
models = {name: {'description': name} for name in ('baseline', 'diffusion')}
try:
    result = Terminal().choose_model(models, {}, set(models), 'diffusion')
except (KeyboardInterrupt, ModelSelectionCancelled):
    result = 'cancelled'
print('SELECTED=' + result, flush=True)
'''
    process = subprocess.Popen([sys.executable, '-c', child], stdin=slave, stdout=slave, stderr=slave,
                               cwd=Path(__file__).resolve().parents[1], env=environment)
    output = b''
    try:
        deadline = time.monotonic() + 10
        while b'Enter select' not in output and time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                output += os.read(master, 65536)
        assert b'Enter select' in output, output.decode(errors='replace')
        os.write(master, keys)
        while process.poll() is None and time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                output += os.read(master, 65536)
        assert process.wait(timeout=1) == 0
        while select.select([master], [], [], 0)[0]:
            output += os.read(master, 65536)
        assert f'SELECTED={result}'.encode() in output
        assert termios.tcgetattr(slave) == before
        assert b'\x1b[?25h' in output  # Cursor becomes visible after selection/cancellation.
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        os.close(master)
        os.close(slave)
