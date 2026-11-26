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


# Wan 2.2 TI2V 5B, revision 8ed5e8e3 (2026-11-26, Shuttrr 0.2.7): 8-bit DiT, umT5 text encoder in 8 bits, 14.3 GB
# (sample clips under samples/ are not part of the download)
WAN5B_Q8: dict[str, Any] = {"revision": '8ed5e8e3af147751a7f1951bd6831414ad994ae1', "files": [
    {'path': '.gitattributes', 'size': 1889, 'sha256': None},
    {'path': 'LICENSE', 'size': 11357, 'sha256': None},
    {'path': 'NOTICE', 'size': 1253, 'sha256': None},
    {'path': 'README.md', 'size': 6728, 'sha256': None},
    {'path': 'dit/blocks-00-03.safetensors', 'size': 696002963, 'sha256': '9875e25f69ec09eef2facbee68a3de3192b4fad73614b429e94e724d918928e7'},
    {'path': 'dit/blocks-04-07.safetensors', 'size': 696002943, 'sha256': '35d965f6a7f45b7a316ed9dcd8895c0f71bf0e78b72884cd3ae49cefb0588918'},
    {'path': 'dit/blocks-08-11.safetensors', 'size': 696002985, 'sha256': '92be27e155c68874411072a379d23dcaa65dfa4c63465babeedfe60418282fa0'},
    {'path': 'dit/blocks-12-15.safetensors', 'size': 696003115, 'sha256': '908f82eca3caa2ac127e4edda7162c1e8386851f859b9c7b7f5538df10e462df'},
    {'path': 'dit/blocks-16-19.safetensors', 'size': 696003169, 'sha256': 'db8554c650b5391f1156db0a4de28b630c93d6dd82bf3ceaac45aba1c55c126d'},
    {'path': 'dit/blocks-20-23.safetensors', 'size': 696003137, 'sha256': 'de972a46c6df0e6ae9c86a0a7f7ddd4f6caadfe9419b54c54c38db8db479d488'},
    {'path': 'dit/blocks-24-27.safetensors', 'size': 696003125, 'sha256': '056db9252f25f1f8183334cb0604943f8a919eaffba0919154214161beb0f098'},
    {'path': 'dit/blocks-28-29.safetensors', 'size': 348001535, 'sha256': 'bc5a90dfb667c7eb9633c56d3ccbe10761b16da3d54e8bbac6824d3fcfa75d48'},
    {'path': 'dit/rest.safetensors', 'size': 180186882, 'sha256': 'bc02bdf87a8cb454542f60a84c7947a039a6a4cb0d5c37da14c4be957a54cae3'},
    {'path': 'shuttrr.json', 'size': 6142, 'sha256': None},
    {'path': 'text_encoder_q8/blocks-00-01.safetensors', 'size': 410038988, 'sha256': '7b0bece0e55ed5302de1bde9b21ac71335ed523e08ffa8f39d608b88ef1cf493'},
    {'path': 'text_encoder_q8/blocks-02-03.safetensors', 'size': 410038984, 'sha256': 'b25e1ad3a0fc523a26dc2e41c5a2adb7004c2a72fa91bf7b9ca732eca60174ce'},
    {'path': 'text_encoder_q8/blocks-04-05.safetensors', 'size': 410038964, 'sha256': '2f26e2446c4913bd1d62f46cdec5b3c9dde621ce67ca8c44bb48eca81af0c237'},
    {'path': 'text_encoder_q8/blocks-06-07.safetensors', 'size': 410038976, 'sha256': 'eea95a310a1102753332b6033c445ab694bfd9fa49f429a78bc300297341b55b'},
    {'path': 'text_encoder_q8/blocks-08-09.safetensors', 'size': 410038980, 'sha256': '238aa262748dce6987bcdd4ddcad65b5468e5791ed996a25685e4616be2fc1aa'},
    {'path': 'text_encoder_q8/blocks-10-11.safetensors', 'size': 410039026, 'sha256': '019a7fea5c50a03b59f3a21e9730bba8e41065b2f874ee871df85bdd4e5ff831'},
    {'path': 'text_encoder_q8/blocks-12-13.safetensors', 'size': 410039030, 'sha256': 'f2f2ef5acc8b980bb5166f2f716a01c87de7f5ee6af36cf49fdd21fd4aff0d56'},
    {'path': 'text_encoder_q8/blocks-14-15.safetensors', 'size': 410039014, 'sha256': '42bc9085f3a6474217e7e05c17184b1c55cbec604858d173ec88ede58278456b'},
    {'path': 'text_encoder_q8/blocks-16-17.safetensors', 'size': 410039046, 'sha256': '3c98c06f5095bbb0dec1811bc87f9af8796cf9c3bc3f162333780e431e23744e'},
    {'path': 'text_encoder_q8/blocks-18-19.safetensors', 'size': 410039032, 'sha256': '1c45af4333e4561380d00446a5c6636be7f4211453ab8c43e5b3fcddf06611ba'},
    {'path': 'text_encoder_q8/blocks-20-21.safetensors', 'size': 410038988, 'sha256': 'f65ca3883038deca3d728675dfcab21f37fcef52fa3ac4ff648fc4be43b506fb'},
    {'path': 'text_encoder_q8/blocks-22-23.safetensors', 'size': 410039036, 'sha256': '6b496dad10ff55b509f9919ad3f2af9ab81ea951bda5f6d59fa5417e2cf20515'},
    {'path': 'text_encoder_q8/model.safetensors.index.json', 'size': 33880, 'sha256': None},
    {'path': 'text_encoder_q8/rest.safetensors', 'size': 1115791820, 'sha256': '7d91f5d5c41f90d594be0139571addc1917ed2d98996ff612b70df8e40735b7a'},
    {'path': 'tokenizer/special_tokens_map.json', 'size': 6623, 'sha256': None},
    {'path': 'tokenizer/spiece.model', 'size': 4548313, 'sha256': 'e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458'},
    {'path': 'tokenizer/tokenizer.json', 'size': 16837417, 'sha256': '6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b'},
    {'path': 'tokenizer/tokenizer_config.json', 'size': 61728, 'sha256': None},
    {'path': 'vae/model.safetensors', 'size': 2818778924, 'sha256': '20a7c9b2ee722156244b7459172a00b42cccba6327a2266e38474ca1e6a412dd'},
]}


# Fish Audio Drama 3, 8-bit (2026-11-27, Shuttrr 0.2.13): plain MLX conversion of fishaudio/drama-3, 43.2 GB.
# Pinned by file size and sha256 against the repo's main branch (no fixed revision: the upload was still running when
# 0.2.13 shipped). Shuttrr lists it as ready only once Hugging Face shows every file with these sizes / checksums.
DRAMA3_Q8: dict[str, Any] = {"revision": None, "files": [
    {'path': 'LICENSE', 'size': 19685, 'sha256': None},
    {'path': 'NOTICE', 'size': 852, 'sha256': None},
    {'path': 'README.md', 'size': 2510, 'sha256': None},
    {'path': 'backbone-00.safetensors', 'size': 3708811804, 'sha256': '7f90c4eab950ce2ba0ee62ede6737d359e236fcec9f10ad486d23eebbb92c8d0'},
    {'path': 'backbone-04.safetensors', 'size': 3708811828, 'sha256': '85182af96f37c27e2af91946e2c8221be0701bc949afa2c2fb00ef41d08371cd'},
    {'path': 'backbone-08.safetensors', 'size': 3708811859, 'sha256': 'dfa8f36e9a146960cc84344e8149a843df98bddaa35116c48bcd5e5c4531260f'},
    {'path': 'backbone-12.safetensors', 'size': 3708811925, 'sha256': 'ddbb02c364608e70a7cbc6c7f930abbcae2b89530c40a353ff8dc8c51ee807b4'},
    {'path': 'backbone-16.safetensors', 'size': 3708811901, 'sha256': '3b0e83d0d736bc6873f0d0ad675c1caeb729b2906a5e03d1c337280e72a36343'},
    {'path': 'backbone-20.safetensors', 'size': 3708811919, 'sha256': '2001aa93640c4b5fec2d2c9b5bfa320c8f3adc5820e0cb1212a5fae2382038ad'},
    {'path': 'backbone-24.safetensors', 'size': 3708811939, 'sha256': '748d2d3ec28c2a2c8d892fb28e7489e69841f834d836ef48ce2f3ae8b3588d70'},
    {'path': 'backbone-28.safetensors', 'size': 3708811901, 'sha256': '89979b0e9786c599e74e0a572ea5b97277ab34257dfa6ca14690880b3c23825a'},
    {'path': 'backbone-32.safetensors', 'size': 3708811875, 'sha256': '8ea259a72b0a1ea20b35cdccf9ce0685a9106e86a7eb0aef17b4df3bddc96de6'},
    {'path': 'backbone-36.safetensors', 'size': 3708811929, 'sha256': 'bd504efef2c15b210e79b95055c2f6de4d50b1a51c18b0d82d99f600fae6e5b0'},
    {'path': 'backbone-embed.safetensors', 'size': 2066747743, 'sha256': '8c7e9613ffb1cb87996ded07d696bc9e04033d29c8a6adee361f038b7bb976fe'},
    {'path': 'chat_template.jinja', 'size': 7756, 'sha256': None},
    {'path': 'codec.safetensors', 'size': 1565788570, 'sha256': '3e979b0db3ace20233aa869bf10fdd61ff4e0bc691a580780ae5839d463be11b'},
    {'path': 'config.json', 'size': 5642, 'sha256': None},
    {'path': 'model.safetensors.index.json', 'size': 94896, 'sha256': None},
    {'path': 'special_tokens_map.json', 'size': 102779, 'sha256': None},
    {'path': 'tokenizer.json', 'size': 20791070, 'sha256': '0e9b9db76b2f0389b04b6ef057a2ac3b89100b7e96f2dd7034d19580a09654d6'},
    {'path': 'tokenizer_config.json', 'size': 103580, 'sha256': None},
    {'path': 'tower.safetensors', 'size': 2445859253, 'sha256': '4904a6b385da75d62a1e3e881b1a87fe0ab52cd7c644b210b22759a556d409ef'},
]}
