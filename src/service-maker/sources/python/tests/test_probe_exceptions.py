# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import sys

from pyservicemaker import BatchMetadataOperator, Buffer


BUFFER_OPERATOR_SCRIPT = r"""
import sys

from pyservicemaker import BufferOperator, Pipeline, Probe


class RaisingBufferOperator(BufferOperator):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def handle_buffer(self, buffer):
        self.calls += 1
        if self.calls == 2:
            raise ValueError("buffer operator failed")
        return True


reported = []
sys.unraisablehook = lambda args: reported.append(
    (args.exc_type.__name__, str(args.exc_value))
)

handler = RaisingBufferOperator()
pipeline = Pipeline("probe-exception")
pipeline.add("videotestsrc", "src", {"num-buffers": 5})
pipeline.add("fakesink", "sink", {"sync": False})
pipeline.link("src", "sink")
pipeline.attach("src", Probe("raise", handler))
pipeline.start()
pipeline.wait()

assert handler.calls == 5
assert reported == [("ValueError", "buffer operator failed")]
print("probe-exception-ok")
"""


class RaisingBatchMetadataOperator(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        raise ValueError("batch metadata operator failed")


def test_buffer_operator_exception_does_not_abort_pipeline():
    result = subprocess.run(
        [sys.executable, "-c", BUFFER_OPERATOR_SCRIPT],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"child process failed with {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "probe-exception-ok" in result.stdout


def test_batch_metadata_operator_exception_is_reported_and_passed(monkeypatch):
    reported = []
    monkeypatch.setattr(sys, "unraisablehook", reported.append)

    metadata = Buffer().batch_meta
    BatchMetadataOperator.handle_metadata(RaisingBatchMetadataOperator(), metadata)

    assert len(reported) == 1
    assert reported[0].exc_type is ValueError
    assert str(reported[0].exc_value) == "batch metadata operator failed"
