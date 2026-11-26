"""Pinned Hugging Face file lists for published Shuttrr builds (path, size, sha256; sha256 is None for small
non-LFS files). A download fetches exactly that revision and checks every file against its list, with no request
to Hugging Face needed for the list. `legacy` are earlier revisions whose files still count as a complete
download, so a copy fetched before a new revision keeps working without a new download."""

from __future__ import annotations

from typing import Any

# Wan 2.1 T2V 1.3B, revision 709d9c4f (2026-11-26, Shuttrr 0.2.0-0.2.5): umT5 text encoder in bf16, 13.2 GB
WAN13_BF16_TE: dict[str, Any] = {"revision": '709d9c4f88b386a38cb2044b58009912a314e464', "files": [
    {'path': '.gitattributes', 'size': 1580, 'sha256': None},
    {'path': 'LICENSE', 'size': 11357, 'sha256': None},
    {'path': 'NOTICE', 'size': 956, 'sha256': None},
    {'path': 'README.md', 'size': 5190, 'sha256': None},
    {'path': 'dit/blocks-00-03.safetensors', 'size': 197619913, 'sha256': '7b8c11c838558af450e41014a5fa2e95370a82605a277ca6e92288cc0adb6b82'},
    {'path': 'dit/blocks-04-07.safetensors', 'size': 197619913, 'sha256': 'acee165fd59dad3e75f1c2613f31e490cc7a231cccaa27339ca0ce7eb8cc2003'},
    {'path': 'dit/blocks-08-11.safetensors', 'size': 197619973, 'sha256': '91a7f4e612b4a5c56c3fdd7f13a9234919a50577cd0404424983fa13c333dd15'},
    {'path': 'dit/blocks-12-15.safetensors', 'size': 197620073, 'sha256': '017cf12a6d2d1646374e975f0d4dd4b1ea35218740e7fcac00431967f6b7b0b9'},
    {'path': 'dit/blocks-16-19.safetensors', 'size': 197620103, 'sha256': 'edc2f1e35e37ba20fcc2d05060bc4289cfbb7c172be5a36e7bb5c95983c72c4d'},
    {'path': 'dit/blocks-20-23.safetensors', 'size': 197620079, 'sha256': '994fca72655f2c86794545ab3d4bc92b2774bcf299f4a60eadafbffaf9757b64'},
    {'path': 'dit/blocks-24-27.safetensors', 'size': 197620097, 'sha256': '93f9507ff1a8fe1d0b63aac8493b47dab0881756edf6c040b2937973fd084d6a'},
    {'path': 'dit/blocks-28-29.safetensors', 'size': 98809962, 'sha256': '87ef5a801aa369c2f7f74e487abe2c664c1e45c266b14bbcfec9f0616182983f'},
    {'path': 'dit/rest.safetensors', 'size': 51558918, 'sha256': 'f64d52a4083a2ebb421bb5cb6174e6999f4c6a9da84e077121e0bab5fa83aeaa'},
    {'path': 'shuttrr.json', 'size': 3185, 'sha256': None},
    {'path': 'text_encoder/blocks-00-05.safetensors', 'size': 2315384784, 'sha256': '5994f5538afd3aededa8ba69c7318cb6db1ec5903517d680a75d03d8a99cec11'},
    {'path': 'text_encoder/blocks-06-11.safetensors', 'size': 2315384808, 'sha256': 'fc0e1e4704d7f684bf74b2b2e1ae3850d58deddd98a47092f557d450914c1ede'},
    {'path': 'text_encoder/blocks-12-17.safetensors', 'size': 2315384850, 'sha256': '49a45cc96f76aaafd8ca3871cfd3da6b8972519c4f7ca69e676785d949510c66'},
    {'path': 'text_encoder/blocks-18-23.safetensors', 'size': 2315384852, 'sha256': '7fb67c1c755a9760f25d68ec9f87457a146a69fdd9018563698e9ddbb885cb87'},
    {'path': 'text_encoder/model.safetensors.index.json', 'size': 14263, 'sha256': None},
    {'path': 'text_encoder/rest.safetensors', 'size': 2100306128, 'sha256': 'dc06d097a4e7828c362b123ca193d746c2bfb42ecfaac465068a8133cdbe62f9'},
    {'path': 'tokenizer/special_tokens_map.json', 'size': 6623, 'sha256': None},
    {'path': 'tokenizer/spiece.model', 'size': 4548313, 'sha256': 'e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458'},
    {'path': 'tokenizer/tokenizer.json', 'size': 16837417, 'sha256': '6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b'},
    {'path': 'tokenizer/tokenizer_config.json', 'size': 61728, 'sha256': None},
    {'path': 'vae/model.safetensors', 'size': 293194152, 'sha256': 'b30ffda32731cb257669265130b4485dbf2ceb23cda77d6de9b8ebbf1411cb1b'},
]}

# Wan 2.1 T2V 1.3B, revision a3029232 (2026-11-26, Shuttrr 0.2.6): umT5 text encoder in 8 bits (group size 64), 7.9 GB
WAN13_Q8_TE: dict[str, Any] = {"revision": 'a3029232d32ac0ed6ac5f7b2c7bf7b30df0adbdc', "files": [
    {'path': '.gitattributes', 'size': 1580, 'sha256': None},
    {'path': 'LICENSE', 'size': 11357, 'sha256': None},
    {'path': 'NOTICE', 'size': 1101, 'sha256': None},
    {'path': 'README.md', 'size': 6513, 'sha256': None},
    {'path': 'dit/blocks-00-03.safetensors', 'size': 197619913, 'sha256': '7b8c11c838558af450e41014a5fa2e95370a82605a277ca6e92288cc0adb6b82'},
    {'path': 'dit/blocks-04-07.safetensors', 'size': 197619913, 'sha256': 'acee165fd59dad3e75f1c2613f31e490cc7a231cccaa27339ca0ce7eb8cc2003'},
    {'path': 'dit/blocks-08-11.safetensors', 'size': 197619973, 'sha256': '91a7f4e612b4a5c56c3fdd7f13a9234919a50577cd0404424983fa13c333dd15'},
    {'path': 'dit/blocks-12-15.safetensors', 'size': 197620073, 'sha256': '017cf12a6d2d1646374e975f0d4dd4b1ea35218740e7fcac00431967f6b7b0b9'},
    {'path': 'dit/blocks-16-19.safetensors', 'size': 197620103, 'sha256': 'edc2f1e35e37ba20fcc2d05060bc4289cfbb7c172be5a36e7bb5c95983c72c4d'},
    {'path': 'dit/blocks-20-23.safetensors', 'size': 197620079, 'sha256': '994fca72655f2c86794545ab3d4bc92b2774bcf299f4a60eadafbffaf9757b64'},
    {'path': 'dit/blocks-24-27.safetensors', 'size': 197620097, 'sha256': '93f9507ff1a8fe1d0b63aac8493b47dab0881756edf6c040b2937973fd084d6a'},
    {'path': 'dit/blocks-28-29.safetensors', 'size': 98809962, 'sha256': '87ef5a801aa369c2f7f74e487abe2c664c1e45c266b14bbcfec9f0616182983f'},
    {'path': 'dit/rest.safetensors', 'size': 51558918, 'sha256': 'f64d52a4083a2ebb421bb5cb6174e6999f4c6a9da84e077121e0bab5fa83aeaa'},
    {'path': 'shuttrr.json', 'size': 3984, 'sha256': None},
    {'path': 'text_encoder_q8/blocks-00-05.safetensors', 'size': 1230116881, 'sha256': '70250813ae6959c41a10ccf201c2d0aeca58a2c88068dfae179ecac576a1d0ca'},
    {'path': 'text_encoder_q8/blocks-06-11.safetensors', 'size': 1230116943, 'sha256': '09b7e65dd8f4f058bd4ee854b391aab2496304c1cbee2d679bf985d3df4b8eeb'},
    {'path': 'text_encoder_q8/blocks-12-17.safetensors', 'size': 1230117057, 'sha256': '89e30561a61df0c82f7f9f9bc4caae6f87ab4c8ead0248ad996693645db3f20f'},
    {'path': 'text_encoder_q8/blocks-18-23.safetensors', 'size': 1230117051, 'sha256': '39cb880d34b799302539a585ac6ee483eb2248d2478bbe0d60c2012832888f17'},
    {'path': 'text_encoder_q8/model.safetensors.index.json', 'size': 33880, 'sha256': None},
    {'path': 'text_encoder_q8/rest.safetensors', 'size': 1115791820, 'sha256': '7d91f5d5c41f90d594be0139571addc1917ed2d98996ff612b70df8e40735b7a'},
    {'path': 'tokenizer/special_tokens_map.json', 'size': 6623, 'sha256': None},
    {'path': 'tokenizer/spiece.model', 'size': 4548313, 'sha256': 'e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458'},
    {'path': 'tokenizer/tokenizer.json', 'size': 16837417, 'sha256': '6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b'},
    {'path': 'tokenizer/tokenizer_config.json', 'size': 61728, 'sha256': None},
    {'path': 'vae/model.safetensors', 'size': 293194152, 'sha256': 'b30ffda32731cb257669265130b4485dbf2ceb23cda77d6de9b8ebbf1411cb1b'},
]}

