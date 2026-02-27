"""Golden digests over rendered reports.

Every digest in this file was produced by actually rendering the document it
pins, on the reference interpreter, with version-free metadata so a package
version bump cannot move it.  When a digest fails, either the layout changed
by accident -- a bug -- or it changed on purpose, and then the fix is to
recompute the digest with :func:`digest_report` and say in the commit message
why the bytes moved.
"""

from __future__ import annotations

from echoscribe.eval.report import (
    alignment_section,
    digest_report,
    histogram_section,
    render_report,
    table_section,
)
from echoscribe.meta import RunMetadata

METADATA = RunMetadata(label="golden", seed=7, versions={}, configs={})
PAIRS = [("the", "the"), ("cat", "cot"), ("sat", None), (None, "down")]
SAMPLES = [1.0, 2.5, 2.5, 3.0, 7.0]
ROWS = [("u1", "WER 0.00% | S0 D0 I0 | N2"), ("u2", "CER 16.67% | S0 D1 I0 | N6")]
DESCRIPTION = "pinned by tests/test_eval_report_golden.py"


def combined_document(samples: list[float] | None = None) -> str:
    """The report every whole-document golden below is taken from."""
    return render_report(
        "golden report",
        [
            alignment_section(PAIRS),
            histogram_section(SAMPLES if samples is None else samples, bins=4),
            table_section(ROWS),
        ],
        METADATA,
        description=DESCRIPTION,
    )


def test_the_golden_metadata_digest_never_moves() -> None:
    # Version-free by construction: this digest outlives package releases.
    assert METADATA.digest() == "3d2679dd305f374f"


def test_the_alignment_report_bytes_are_pinned() -> None:
    document = render_report("golden alignment", [alignment_section(PAIRS)], METADATA)
    assert digest_report(document) == (
        "7db35ecdd69c1f3b0a150ea3187ee6e3f1cdc020ff5ecfb7353e30cb73dac7aa"
    )


def test_the_histogram_report_bytes_are_pinned() -> None:
    document = render_report("golden histogram", [histogram_section(SAMPLES, bins=4)], METADATA)
    assert digest_report(document) == (
        "26e1415a3984e96a1821f048150db6050e879a59839aa56cf37afbaac72cee98"
    )


def test_the_table_report_bytes_are_pinned() -> None:
    document = render_report("golden table", [table_section(ROWS)], METADATA)
    assert digest_report(document) == (
        "e4af2cd82b9ef95cdba92053c0ebe44182ad09717cb87e43e0167f8b54bed622"
    )


def test_the_combined_report_bytes_are_pinned() -> None:
    assert digest_report(combined_document()) == (
        "a3fcb16c197801521fcdcf56ca58dd9b0ea3824746289bec10fcafab13327039"
    )


def test_the_combined_report_stacks_its_three_sections() -> None:
    document = combined_document()
    assert document.index(">alignment<") < document.index(">latency<")
    assert document.index(">latency<") < document.index(">scores<")


def test_tiny_sample_noise_renders_the_same_document() -> None:
    # Perturbations three orders of magnitude below the rounding the length
    # formatter applies cannot reach the output, on any platform: binning is
    # integer arithmetic and coordinates round to two decimals.
    noisy = [1.0, 2.5 + 1e-12, 2.5, 3.0, 7.0 - 1e-12]
    assert digest_report(combined_document(noisy)) == digest_report(combined_document())


def test_a_real_coordinate_change_does_move_the_bytes() -> None:
    moved = [1.0, 2.5, 2.5, 3.0, 7.01]
    assert digest_report(combined_document(moved)) != digest_report(combined_document())
