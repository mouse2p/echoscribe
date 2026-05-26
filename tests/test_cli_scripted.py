"""The scripted fixtures are checked before any command trusts them."""

from __future__ import annotations

import numpy as np
import pytest

from echoscribe.cli import main
from echoscribe.cli.scripted import load_scripted_corpus, scripted_path, sparse_path
from echoscribe.decode.ctc_greedy import ctc_greedy
from echoscribe.decode.posteriors import scripted_posteriors
from echoscribe.errors import ConfigError


def prepare(tmp_path, utterances: int = 2):
    """A work directory with synth and feats already run over it."""
    assert (
        main(
            [
                "synth",
                "--workdir",
                str(tmp_path),
                "--utterances",
                str(utterances),
                "--max-tokens",
                "2",
                "--seed",
                "5",
            ]
        )
        == 0
    )
    assert main(["feats", "--workdir", str(tmp_path), "--num-mel-bins", "40"]) == 0
    return tmp_path


def test_the_path_is_exactly_as_long_as_the_frame_count() -> None:
    for frames in (5, 17, 40, 200):
        assert len(scripted_path([1, 2, 3], frames)) == frames


def test_blanks_separate_the_units_when_there_is_room() -> None:
    path = scripted_path([4, 5], 10)
    assert path.count(0) == 1
    # divmod(9, 2) gives runs of 5 and 4 around the single separator
    assert path == [4, 4, 4, 4, 4, 0, 5, 5, 5, 5]


def test_tight_frames_drop_separators_rather_than_units() -> None:
    path = scripted_path([7, 8, 9], 3)
    assert path == [7, 8, 9]


def test_a_greedy_decode_replays_the_planted_units() -> None:
    ids = (3, 1, 3, 2)
    posteriors = scripted_posteriors(scripted_path(ids, 40), num_symbols=6)
    assert ctc_greedy(posteriors).symbols == ids


def test_the_argmax_of_every_frame_is_the_planted_symbol() -> None:
    path = scripted_path([2, 3], 9)
    posteriors = scripted_posteriors(path, num_symbols=5)
    assert np.argmax(posteriors, axis=1).tolist() == path


def test_a_prepared_corpus_loads_with_its_frame_counts(tmp_path) -> None:
    prepare(tmp_path)
    utterances, vocabulary = load_scripted_corpus(tmp_path)
    assert len(utterances) == 2
    for item in utterances:
        assert item.posteriors.shape == (item.num_frames, len(vocabulary))
        assert ctc_greedy(item.posteriors).symbols == item.unit_ids


def test_a_missing_manifest_is_explained(tmp_path) -> None:
    with pytest.raises(ConfigError, match="run `echoscribe synth` first"):
        load_scripted_corpus(tmp_path)


def test_a_missing_feature_archive_is_explained(tmp_path) -> None:
    main(["synth", "--workdir", str(tmp_path), "--utterances", "2", "--seed", "1"])
    with pytest.raises(ConfigError, match="run `echoscribe feats` first"):
        load_scripted_corpus(tmp_path)


def test_an_empty_unit_list_is_refused() -> None:
    with pytest.raises(ConfigError, match="at least one unit"):
        scripted_path([], 10)


def test_more_units_than_frames_is_refused() -> None:
    with pytest.raises(ConfigError, match="cannot carry"):
        scripted_path([1, 2, 3], 2)


def test_the_sparse_path_plants_one_frame_per_unit() -> None:
    path = sparse_path([3, 1, 2], 12)
    assert len(path) == 12
    assert path[0] == 3 and path[4] == 1 and path[8] == 2
    assert path.count(0) == 9


def test_tight_frames_still_fit() -> None:
    assert sparse_path([2, 3], 2) == [2, 3]


def test_a_sparse_path_also_survives_a_greedy_ctc_decode() -> None:
    posteriors = scripted_posteriors(sparse_path([3, 1, 3], 9), num_symbols=5)
    assert ctc_greedy(posteriors).symbols == (3, 1, 3)


def test_sparse_validation_matches_the_run_path() -> None:
    with pytest.raises(ConfigError, match="at least one unit"):
        sparse_path([], 4)
    with pytest.raises(ConfigError, match="cannot carry"):
        sparse_path([1, 2, 3], 2)
