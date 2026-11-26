"""The 8-bit umT5 text encoder (Shuttrr 0.2.6): writing it, reading it block by block, and the published Wan 1.3B
build accepting both its new 8-bit revision and the earlier bf16 one."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from shutter.core import catalog as cat  # noqa: E402
from shutter.core import text_encode as TE  # noqa: E402
from shutter.core.config import Config  # noqa: E402
from shutter.families.wan import keys as K  # noqa: E402

import wan_fixtures as F  # noqa: E402

GB = 1024**3
WAN13 = "wan2.1-t2v-1.3b-8bit"
KW = dict(vocab_size=300, dim=64, dim_attn=64, dim_ffn=128, num_heads=4, num_layers=2, num_buckets=32,
          shared_pos=False)


def _bf16_folder(folder: Path) -> dict:
    """A tiny umT5 written the way the Wan converter writes it (rest + block shards, bf16, with an index)."""
    w = {K.t5_to_mlx(k, "wan"): mx.array(v).astype(mx.bfloat16) for k, v in F.t5_weights().items()}
    folder.mkdir(parents=True)
    rest = {k: v for k, v in w.items() if not k.startswith("blocks.")}
    blocks = {k: v for k, v in w.items() if k.startswith("blocks.")}
    mx.save_safetensors(str(folder / "rest.safetensors"), rest)
    mx.save_safetensors(str(folder / "blocks-00-01.safetensors"), blocks)
    wm = {k: "rest.safetensors" for k in rest} | {k: "blocks-00-01.safetensors" for k in blocks}
    (folder / TE.INDEX).write_text(json.dumps({"weight_map": wm}))
    return w


def _encode(folder: Path):
    from shutter.families.wan.model.text_encoder import T5Encoder
    ids = mx.array([[5, 17, 42, 9, 0, 0]])
    mask = mx.array([[1, 1, 1, 1, 0, 0]])
    w = TE.lazy_weights(folder, TE.INDEX, mx)
    return TE.encode_ids(T5Encoder(**KW), w, ids, mask, mx, upcast=mx.float32)


def _sha(folder: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def test_quantized_folder_layout_and_metadata(tmp_path):
    _bf16_folder(tmp_path / "te")
    r = TE.quantize_folder(tmp_path / "te", tmp_path / "q8", min_size=0, report=lambda s: None)
    assert r["quant"] == {"bits": 8, "group_size": 64} and r["files"][0] == "rest.safetensors"
    assert TE.folder_quantization(tmp_path / "q8") == {"bits": 8, "group_size": 64}
    assert TE.folder_quantization(tmp_path / "te") is None
    raw = mx.load(str(tmp_path / "q8" / "blocks-00-01.safetensors"))
    assert raw["blocks.0.attn.q.weight"].dtype == mx.uint32
    assert raw["blocks.0.attn.q.scales"].dtype == mx.bfloat16 and "blocks.0.attn.q.biases" in raw
    assert raw["blocks.0.norm1.weight"].dtype == mx.bfloat16          # 1D tensors stay as they are
    # lazy_weights hands back plain weights under the bf16 names, nothing else
    w = TE.lazy_weights(tmp_path / "q8", TE.INDEX, mx)
    plain = TE.lazy_weights(tmp_path / "te", TE.INDEX, mx)
    assert sorted(w) == sorted(plain)
    assert w["blocks.0.attn.q.weight"].shape == plain["blocks.0.attn.q.weight"].shape
    assert w["blocks.0.attn.q.weight"].dtype == mx.bfloat16


def test_default_size_floor_leaves_small_tensors_alone(tmp_path):
    _bf16_folder(tmp_path / "te")
    TE.quantize_folder(tmp_path / "te", tmp_path / "q8", report=lambda s: None)   # real floor: all tiny here
    raw = mx.load(str(tmp_path / "q8" / "blocks-00-01.safetensors"))
    assert not [k for k in raw if k.endswith(".scales")]


def test_quantize_is_deterministic(tmp_path):
    _bf16_folder(tmp_path / "te")
    TE.quantize_folder(tmp_path / "te", tmp_path / "a", min_size=0, report=lambda s: None)
    TE.quantize_folder(tmp_path / "te", tmp_path / "b", min_size=0, report=lambda s: None)
    assert _sha(tmp_path / "a") == _sha(tmp_path / "b")


def test_8bit_encode_matches_bf16_closely_and_dequantized_exactly(tmp_path):
    _bf16_folder(tmp_path / "te")
    TE.quantize_folder(tmp_path / "te", tmp_path / "q8", min_size=0, report=lambda s: None)
    a = np.array(_encode(tmp_path / "te")[0, :4])
    b = np.array(_encode(tmp_path / "q8")[0, :4])
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    assert cos.min() > 0.999
    # the same forward as running the dequantized weights by hand
    raw = TE.lazy_weights(tmp_path / "q8", TE.INDEX, mx)
    hand = tmp_path / "hand"
    hand.mkdir()
    mx.save_safetensors(str(hand / "all.safetensors"),
                        {k: (v.dequantize() if isinstance(v, TE.QuantizedRows) else v) for k, v in raw.items()})
    assert np.array_equal(b, np.array(_encode(hand)[0, :4]))


def test_embedding_rows_equal_the_whole_table_dequantized(tmp_path):
    _bf16_folder(tmp_path / "te")
    TE.quantize_folder(tmp_path / "te", tmp_path / "q8", min_size=0, report=lambda s: None)
    w = TE.lazy_weights(tmp_path / "q8", TE.INDEX, mx)
    table = w["token_embedding.weight"]
    assert isinstance(table, TE.QuantizedRows) and table.shape == (300, 64)
    raw = mx.load(str(tmp_path / "q8" / "rest.safetensors"))
    whole = mx.dequantize(raw["token_embedding.weight"], raw["token_embedding.scales"], raw["token_embedding.biases"],
                          group_size=64, bits=8)
    ids = mx.array([[5, 17, 42, 9, 0, 0]])
    assert np.array_equal(np.array(table[ids].astype(mx.float32)), np.array(whole[ids].astype(mx.float32)))


def _wan_build(root: Path, q8: bool) -> Path:
    """A minimal Wan build folder: shuttrr.json listing the text encoder where it is."""
    te = "text_encoder_q8" if q8 else "text_encoder"
    _bf16_folder(root / "text_encoder")
    if q8:
        TE.quantize_folder(root / "text_encoder", root / te, min_size=0, report=lambda s: None)
        import shutil
        shutil.rmtree(root / "text_encoder")
    man = {"family": "wan", "parts": {"text_encoder": {"role": "text_encoder",
                                                      "files": [f"{te}/rest.safetensors",
                                                                f"{te}/blocks-00-01.safetensors"]}}}
    (root / "shuttrr.json").write_text(json.dumps(man))
    return root


@pytest.mark.parametrize("q8", [False, True])
def test_pipeline_reads_the_text_encoder_folder_from_the_manifest(tmp_path, q8):
    from shutter.families.wan.pipeline import WanPipeline
    root = _wan_build(tmp_path / "b", q8)
    p = WanPipeline.__new__(WanPipeline)
    p.build = root
    p.man = json.loads((root / "shuttrr.json").read_text())
    assert p._text_encoder_dir() == root / ("text_encoder_q8" if q8 else "text_encoder")
    p.man = {"parts": {}}                                   # very old manifests: the default folder
    assert p._text_encoder_dir() == root / "text_encoder"


def test_other_wan_builds_borrow_the_8bit_encoder_as_is(tmp_path):
    from shutter.families.wan import convert as C
    from shutter.families.wan.detect import _find_t5
    root = _wan_build(tmp_path / "b", True)
    part = _find_t5(root)
    assert part is not None and part.layout == "mlx-q" and len(part.files) == 2
    out = C.convert_t5(part.files, part.layout, tmp_path / "new" / "text_encoder", report=lambda s: None)
    assert out["files"] == ["text_encoder_q8/rest.safetensors", "text_encoder_q8/blocks-00-01.safetensors"]
    assert out["quant"] == {"bits": 8, "group_size": 64}
    assert _sha(tmp_path / "new" / "text_encoder_q8") == _sha(root / "text_encoder_q8")
    assert cat.is_complete(tmp_path / "new", {"check_files": [cat.WAN_TE]})


# ----------------------------------------------------------------------------- catalog: pinned revision + legacy


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("SHUTTER_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cat, "total_memory", lambda: 16 * GB)
    c = Config.load()
    c.ensure_dirs()
    c.save()
    return c


def _write(dest: Path, files: list[dict]) -> None:
    for f in files:
        p = dest / f["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\0" * f["size"])


def _shrunk(files: list[dict]) -> list[dict]:
    """The pinned list with tiny sizes (so a test can write it to disk), no checksums."""
    return [{"path": f["path"], "size": 3 + i, "sha256": None} for i, f in enumerate(files)]


@pytest.fixture
def lists(monkeypatch):
    e = cat.CATALOG[WAN13]
    new = _shrunk(e["files"] or [{"path": p, "size": 1} for p in
                                  ("shuttrr.json", "dit/rest.safetensors", "text_encoder_q8/rest.safetensors",
                                   "tokenizer/tokenizer.json", "vae/model.safetensors")])
    old = _shrunk(e["legacy"][0]["files"])
    monkeypatch.setitem(e, "files", new)
    monkeypatch.setitem(e, "legacy", [{"revision": "old", "files": old}])
    return e, new, old


def test_entry_is_pinned_and_lists_the_8bit_encoder():
    e = cat.CATALOG[WAN13]
    old = e["legacy"][0]
    assert old["revision"].startswith("709d9c4f") and any(f["path"].startswith("text_encoder/") for f in old["files"])
    if e["files"]:                                           # filled in once the new revision is published
        assert len(e["revision"]) == 40
        paths = {f["path"] for f in e["files"]}
        assert any(p.startswith("text_encoder_q8/") for p in paths)
        assert not any(p.startswith("text_encoder/") for p in paths)
        assert all(f["sha256"] for f in e["files"] if f["path"].endswith(".safetensors"))
        assert e["size_gb"] == round(sum(f["size"] for f in e["files"]) / GB, 1) < 8


def test_legacy_bf16_copy_counts_as_downloaded(cfg, lists):
    e, new, old = lists
    dest = cat.models_home(cfg) / e["folder"]
    _write(dest, old)
    assert cat.is_complete(dest, e) and cat.already_downloaded(dest, e, new)
    assert cat.legacy_revision(dest, e) == "old"

    def must_not_fetch(repo, d):
        raise AssertionError("fetched a model that is already downloaded")
    done = cat.download(WAN13, cfg, fetch=must_not_fetch, poll_s=0.01)
    assert done["already"] and done["legacy_revision"] == "old" and "earlier version" in done["verified_detail"]
    assert WAN13 in Config.load(cfg.path).models
    # nothing of the old copy was touched
    assert all((dest / f["path"]).stat().st_size == f["size"] for f in old)


def test_new_copy_counts_as_downloaded(cfg, lists):
    e, new, old = lists
    dest = cat.models_home(cfg) / e["folder"]
    _write(dest, new)
    done = cat.download(WAN13, cfg, fetch=lambda r, d: (_ for _ in ()).throw(AssertionError()), poll_s=0.01)
    assert done["already"] and done["legacy_revision"] is None
    assert "earlier version" not in done["verified_detail"]


def test_cut_short_legacy_copy_is_not_complete(cfg, lists):
    e, new, old = lists
    dest = cat.models_home(cfg) / e["folder"]
    _write(dest, old)
    big = next(f for f in old if f["path"].startswith("text_encoder/"))
    (dest / big["path"]).write_bytes(b"\0")
    assert not cat.already_downloaded(dest, e, new)


def test_download_fetches_the_pinned_revision_and_drops_old_only_files(cfg, lists, monkeypatch):
    e, new, old = lists
    dest = cat.models_home(cfg) / e["folder"]
    # a stopped download of the old version: some old files, one of them the bf16 encoder
    _write(dest, [f for f in old if f["path"].startswith(("text_encoder/", "dit/"))][:3])
    seen = {}

    def fake_hf(repo, folder, revision=None):
        seen.update(repo=repo, revision=revision)
        _write(Path(folder), new)
    monkeypatch.setattr(cat, "_hf_download", fake_hf)
    monkeypatch.setattr(cat, "_hf_files", lambda r: (_ for _ in ()).throw(AssertionError("list is pinned")))
    done = cat.download(WAN13, cfg, poll_s=0.01)
    assert done["state"] == "done" and not done.get("already")
    assert seen == {"repo": e["repo"], "revision": e["revision"]}
    assert not any(p.is_file() for p in (dest / "text_encoder").rglob("*"))
    assert all((dest / f["path"]).is_file() for f in new)
