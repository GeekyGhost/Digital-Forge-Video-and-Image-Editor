# Functional Source License, Version 1.1, ALv2 Future License

## Abbreviation

FSL-1.1-ALv2

## Notice

Copyright 2026 Willie Gray

## Terms and Conditions

### Licensor ("We")

The party offering the Software under these Terms and Conditions.

### The Software

The "Software" is each version of the software that we make available under
these Terms and Conditions, as indicated by our inclusion of these Terms and
Conditions with the Software.

### License Grant

Subject to your compliance with this License Grant and the Patents,
Redistribution and Trademark clauses below, we hereby grant you the right to
use, copy, modify, create derivative works, publicly perform, publicly display
and redistribute the Software for any Permitted Purpose identified below.

### Permitted Purpose

A Permitted Purpose is any purpose other than a Competing Use. A Competing Use
means making the Software available to others in a commercial product or
service that:

1. substitutes for the Software;

2. substitutes for any other product or service we offer using the Software
   that exists as of the date we make the Software available; or

3. offers the same or substantially similar functionality as the Software.

Permitted Purposes specifically include using the Software:

1. for your internal use and access;

2. for non-commercial education;

3. for non-commercial research; and

4. in connection with professional services that you provide to a licensee
   using the Software in accordance with these Terms and Conditions.

### Patents

To the extent your use for a Permitted Purpose would necessarily infringe our
patents, the license grant above includes a license under our patents. If you
make a claim against any party that the Software infringes or contributes to
the infringement of any patent, then your patent license to the Software ends
immediately.

### Redistribution

The Terms and Conditions apply to all copies, modifications and derivatives of
the Software.

If you redistribute any copies, modifications or derivatives of the Software,
you must include a copy of or a link to these Terms and Conditions and not
remove any copyright notices provided in or with the Software.

### Disclaimer

THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING WITHOUT LIMITATION WARRANTIES OF FITNESS FOR A PARTICULAR
PURPOSE, MERCHANTABILITY, TITLE OR NON-INFRINGEMENT.

IN NO EVENT WILL WE HAVE ANY LIABILITY TO YOU ARISING OUT OF OR RELATED TO THE
SOFTWARE, INCLUDING INDIRECT, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES,
EVEN IF WE HAVE BEEN INFORMED OF THEIR POSSIBILITY IN ADVANCE.

### Trademarks

Except for displaying the License Details and identifying us as the origin of
the Software, you have no right under these Terms and Conditions to use our
trademarks, trade names, service marks or product names.

## Grant of Future License

We hereby irrevocably grant you an additional license to use the Software under
the Apache License, Version 2.0 that is effective on the second anniversary of
the date we make the Software available. On or after that date, you may use the
Software under the Apache License, Version 2.0, in which case the following
will apply:

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License.

You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

---

## Third-Party Components

The License Grant above covers **only the code in this repository that we wrote**. It does not, and
cannot, extend to anything else — we do not own those things and have no right to sublicense them.
This section is not an exception carved out of the licence; it is a description of a boundary that
already exists.

Nothing in this licence should be read as placing a third-party component under FSL-1.1-ALv2 or
under the Apache License, Version 2.0. Each component listed below, and any other third-party
software, model weights or data distributed alongside or downloaded by this Software, remains
governed exclusively by the terms its own licensor applies to it. Where those terms conflict with
this licence, **their terms govern that component**, and your rights in it come from that licensor,
not from us.

Neither FSL-1.1-ALv2 nor the Apache License, Version 2.0 is a copyleft licence. Neither contains
any reciprocity provision that would extend its own terms to a separately-licensed component
distributed with this Software.

### Model weights and packs

None of these are contained in this repository. They are downloaded on demand by
`scripts/models.py` from their upstream publishers, and installing one is an acceptance of that
publisher's terms.

| Component | Licence |
|---|---|
| LTX-2 | LTX-2 Community License Agreement |
| MiniMax-H3 (FL2VA and Ref2VA) | MiniMax Model License Agreement; adapter code Apache-2.0 |
| Z-Image-Turbo, Qwen-Image-Edit, BiRefNet | Model-specific upstream licences, individually |
| Kiwi-Edit | Repository MIT; component model licences differ |
| Real-ESRGAN | BSD 3-Clause |
| Whisper and other utilities | Mixed open-source licences |

`models/manifests/*.json` carries the canonical licence name and URL for each pack. Several are
materially stricter than this licence — some restrict commercial use, some restrict redistribution.
**Read the one you are installing.** `ACKNOWLEDGEMENTS.md` lists every upstream project and author.

### Optional NVIDIA components

The RTX / Maxine super-resolution path is **optional, is not distributed with this Software, and is
not a dependency of it.** It is obtained by the user, from NVIDIA, after accepting NVIDIA's own
terms — `scripts/setup-rtx-upscale.ps1` will not proceed without that acceptance.

NVIDIA's SDK terms prohibit using their SDK "in any manner that would cause it to become subject to
an open source software license". Nothing here does so: this licence has no copyleft reach, the
NVIDIA components are never redistributed by us, and this section makes explicit that they are not
offered under our terms. Their applicable licences are:

- [License Agreement for NVIDIA Software Development Kits](https://developer.download.nvidia.com/gameworks/dlss/NVIDIA_RTX_SDK_EULA_(1Feb2021).pdf)
- [License For Customer Use of NVIDIA Software](https://www.nvidia.com/en-us/drivers/nvidia-license/)
- The `nvidia-vfx` Python distribution is published by NVIDIA under `LicenseRef-NvidiaProprietary`.

If you fork this project and choose to redistribute any NVIDIA component yourself, that is a
distribution you are making under NVIDIA's terms, and their conditions — object-code form only,
prior notification before commercial release, and their exclusion of cloud deployment — apply to
you directly. We do not and cannot pass those rights along to you.

### Vendored source code

A small amount of third-party source code is committed in this repository. Each copy keeps its
upstream licence file beside it, and none of it is offered under FSL-1.1-ALv2.

| Component | Location | Licence |
|---|---|---|
| [mp4box.js](https://github.com/gpac/mp4box.js) 2.4.1 (GPAC, Telecom ParisTech) | `src/render/video/vendor/mp4box/`, `spikes/vendor/mp4box/` | BSD 3-Clause |
| [MOSS-SoundEffect v2](https://github.com/OpenMOSS/MOSS-TTS) (OpenMOSS), including its trimmed copy of [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) | `ai-runtime/moss_soundeffect_v2/` | Apache License 2.0; modified files are listed in its `NOTICE.md` |

### Runtime dependencies

The application depends on FFmpeg, PyTorch, Diffusers, Transformers, Tauri, React and many other
open-source projects, each under its own licence. They are declared in `package.json`,
`src-tauri/Cargo.toml` and the setup scripts, and are installed from their own distribution
channels rather than vendored here (apart from the components listed under Vendored source code). FFmpeg in particular is a separate program invoked as a
subprocess, not linked into this Software.

### Not legal advice

This section describes our understanding of how these licences interact. It is not legal advice,
and it is not a warranty that any particular use of a third-party component is permitted. If you
intend to redistribute this Software with third-party components included, or to use it
commercially in a way that depends on one of them, get the specific licences reviewed.
