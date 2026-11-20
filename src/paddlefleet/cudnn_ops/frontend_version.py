# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Version of the cudnn-frontend vendored in ``paddlefleet_ops``.

``is_cudnn_frontend_available()`` only says the package can be imported; a
``paddlefleet_ops`` wheel built from an older PaddleFleet still carries an
older cudnn-frontend. The DSA code here needs 1.32.3: 1.28 has no sparse
attention forward, and its backward returns an all-zero ``d_sink`` when
``d_qk != d_v``.
"""

import functools
import re

MIN_CUDNN_FRONTEND = (1, 32, 3)
_MIN_STR = ".".join(map(str, MIN_CUDNN_FRONTEND))


@functools.cache
def cudnn_frontend_version():
    """``(major, minor, patch)`` of ``paddlefleet_ops.cudnn``, or ``None`` if
    it cannot be imported."""
    try:
        from paddlefleet_ops.cudnn import __version__
    except (ImportError, RuntimeError, AttributeError):
        return None
    return tuple(int(x) for x in re.findall(r"\d+", __version__)[:3])


def cudnn_frontend_is_recent(version) -> bool:
    return version is not None and version >= MIN_CUDNN_FRONTEND


def require_recent_cudnn_frontend(version, what: str):
    if not cudnn_frontend_is_recent(version):
        found = (
            "not importable" if version is None else ".".join(map(str, version))
        )
        raise RuntimeError(
            f"{what} needs cudnn-frontend >= {_MIN_STR} in paddlefleet_ops, "
            f"found {found}. Rebuild paddlefleet_ops from this PaddleFleet "
            "checkout (packages/paddlefleet_ops)."
        )
