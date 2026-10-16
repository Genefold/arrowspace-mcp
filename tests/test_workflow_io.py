from __future__ import annotations

import subprocess
import sys

import pytest


def test_foreign_writer_is_rejected_before_output_changes(tmp_path):
    from filelock import Timeout

    from arrowspace_mcp.workflow_io import new_output_directory

    output = tmp_path / "bundle"
    # The child publishes its readiness only after acquiring the write scope.
    code = """
import sys
from pathlib import Path
from arrowspace_mcp.workflow_io import new_output_directory
with new_output_directory(Path(sys.argv[1])):
    print('ready', flush=True)
    sys.stdin.readline()
"""
    with subprocess.Popen(
        [sys.executable, "-c", code, str(output)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as child:
        try:
            assert child.stdout.readline().strip() == "ready"
            with pytest.raises(Timeout), new_output_directory(output):
                pytest.fail("Contending writer entered the write scope")
            assert list(output.iterdir()) == []
        finally:
            child.communicate("release\n", timeout=10)
        assert child.returncode == 0
