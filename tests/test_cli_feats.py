"""The feats subcommand is run over a corpus synth just rendered."""

from __future__ import annotations

import json

import numpy as np

from echoscribe.cli import main


def render_two(tmp_path) -> None:
    assert main(["synth", "--workdir", str(tmp_path), "--utterances", "2", "--seed", "3"]) == 0


def test_feats_archives_one_matrix_per_utterance(tmp_path, capsys) -> None:
    render_two(tmp_path)
    code = main(["feats", "--workdir", str(tmp_path), "--num-mel-bins", "40"])
    assert code == 0
    assert "2 utterances" in capsys.readouterr().out
    with np.load(tmp_path / "features.npz") as bundle:
        names = [key for key in bundle.files if key != "meta"]
        assert len(names) == 2
        for name in names:
            assert bundle[name].ndim == 2
            assert bundle[name].shape[1] == 40
            assert bundle[name].shape[0] > 0


def test_the_archive_meta_records_the_configuration(tmp_path) -> None:
    render_two(tmp_path)
    main(["feats", "--workdir", str(tmp_path), "--num-mel-bins", "32"])
    with np.load(tmp_path / "features.npz") as bundle:
        meta = json.loads(str(bundle["meta"]))
    assert meta["num_mel_bins"] == 32
    assert meta["utterances"] == 2
    assert meta["deltas"] is False
    assert meta["width"] == 32


def test_the_frame_counts_in_the_summary_match_the_archive(tmp_path) -> None:
    render_two(tmp_path)
    main(["feats", "--workdir", str(tmp_path), "--num-mel-bins", "40"])
    with np.load(tmp_path / "features.npz") as bundle:
        meta = json.loads(str(bundle["meta"]))
        frames = sum(bundle[key].shape[0] for key in bundle.files if key != "meta")
    assert meta["frames"] == frames


def test_deltas_triple_the_width_at_the_default_order(tmp_path) -> None:
    render_two(tmp_path)
    main(["feats", "--workdir", str(tmp_path), "--num-mel-bins", "40", "--deltas"])
    with np.load(tmp_path / "features.npz") as bundle:
        meta = json.loads(str(bundle["meta"]))
        widths = {bundle[key].shape[1] for key in bundle.files if key != "meta"}
    assert widths == {120}
    assert meta["deltas"] is True
    assert meta["delta_order"] == 2


def test_one_delta_order_doubles_the_width(tmp_path) -> None:
    render_two(tmp_path)
    main(
        [
            "feats",
            "--workdir",
            str(tmp_path),
            "--num-mel-bins",
            "40",
            "--deltas",
            "--delta-order",
            "1",
        ]
    )
    with np.load(tmp_path / "features.npz") as bundle:
        widths = {bundle[key].shape[1] for key in bundle.files if key != "meta"}
    assert widths == {80}
