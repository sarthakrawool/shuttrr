# Third-party code in this folder

`model.py` and `vae_decode.py` are adapted from **mflux** 0.24.0
(https://github.com/filipstrand/mflux, files under `src/mflux/models/z_image/model/` and
`src/mflux/models/common/vae/block_decoder.py`).

mflux is licensed under the MIT License:

    MIT License

    Copyright (c) 2026 Filip Strand

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Changes made for Shuttrr: trimmed to text to image (no image encoder, ControlNet, LoRA or float16
compute paths), imports changed, the prompt encoder builds only the 35 layers whose output the image
model reads, and the DiT's main stack can be replaced by a streamed stack. The arithmetic is unchanged.

## The model

Z-Image Turbo is by Tongyi-MAI (Alibaba), released under the Apache License 2.0
(https://huggingface.co/Tongyi-MAI/Z-Image-Turbo). The 8-bit weights Shuttrr downloads are the
mflux-community build (https://huggingface.co/mflux-community/z-image-turbo-mflux-q8, Apache 2.0).
