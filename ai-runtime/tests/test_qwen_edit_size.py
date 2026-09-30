"""The Image Studio edit "Output size" override, as the worker applies it (`edit_output_size`).

Loaded from its file so the image_studio package (which imports the model pipelines) is not.
"""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "image_studio_worker", Path(__file__).resolve().parents[1] / "aiv_runtime" / "image_studio" / "worker.py")
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)


@pytest.mark.parametrize("asked, made", [
    ((0, 0), None),
    ((1344, 0), None),
    ((1344, 768), (1344, 768)),
    ((1001, 999), (1008, 992)),
    ((3840, 2160), (2048, 1152)),
    ((4096, 4096), (2048, 2048)),
    ((100, 5000), (256, 2048)),
])
def test_the_override_is_snapped_like_the_interface_snaps_it(asked, made):
    assert worker.edit_output_size(*asked) == made


def test_the_worker_accepts_the_override_flags():
    args = worker.parser().parse_args(["edit", "--output-path", "o.png", "--out-width", "1216", "--out-height", "832"])
    assert (args.out_width, args.out_height) == (1216, 832)
    default = worker.parser().parse_args(["edit", "--output-path", "o.png"])
    assert (default.out_width, default.out_height) == (0, 0)
