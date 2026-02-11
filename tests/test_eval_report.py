"""Report output is a byte contract, so the tests pin bytes."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.eval.report import (
    CELL_HEIGHT,
    CELL_WIDTH,
    COLOR_CORRECT,
    COLOR_DELETION,
    COLOR_GAP,
    COLOR_INSERTION,
    COLOR_SUBSTITUTION,
    SvgElement,
    alignment_section,
    cell_fills,
    digest_report,
    escape_text,
    format_length,
    histogram_counts,
    histogram_section,
    render_report,
    svg_document,
    table_section,
)
from echoscribe.meta import run_metadata


def test_every_reserved_character_becomes_an_entity() -> None:
    assert escape_text("&") == "&amp;"
    assert escape_text("<") == "&lt;"
    assert escape_text(">") == "&gt;"
    assert escape_text('"') == "&quot;"
    assert escape_text("'") == "&apos;"


def test_the_ampersand_is_escaped_first() -> None:
    assert escape_text("<&>") == "&lt;&amp;&gt;"


def test_escaping_is_not_idempotent_by_design() -> None:
    assert escape_text("&amp;") == "&amp;amp;"


def test_plain_and_cjk_text_passes_through_unchanged() -> None:
    assert escape_text("hello world") == "hello world"
    assert escape_text("今天天气不错") == "今天天气不错"


def test_the_empty_string_escapes_to_the_empty_string() -> None:
    assert escape_text("") == ""


def test_a_non_string_is_refused() -> None:
    with pytest.raises(ConfigError, match="needs a string"):
        escape_text(42)


def test_whole_numbers_render_without_a_decimal_point() -> None:
    assert format_length(12.0) == "12"
    assert format_length(0.0) == "0"
    assert format_length(-3.0) == "-3"
    assert format_length(7) == "7"


def test_two_decimals_survive_and_trailing_zeros_do_not() -> None:
    assert format_length(12.5) == "12.5"
    assert format_length(12.34) == "12.34"
    assert format_length(0.1) == "0.1"


def test_values_are_rounded_to_two_decimals() -> None:
    assert format_length(12.345) == "12.35"
    assert format_length(12.344) == "12.34"


def test_float_noise_below_the_second_decimal_disappears() -> None:
    assert format_length(0.1 + 0.2) == "0.3"


def test_a_tiny_negative_rounds_to_plain_zero() -> None:
    assert format_length(-0.001) == "0"


def test_non_finite_lengths_are_refused() -> None:
    with pytest.raises(ConfigError, match="must be finite"):
        format_length(float("nan"))
    with pytest.raises(ConfigError, match="must be finite"):
        format_length(float("inf"))


def test_a_string_or_boolean_length_is_refused() -> None:
    with pytest.raises(ConfigError, match="must be a number"):
        format_length("12")
    with pytest.raises(ConfigError, match="must be a number"):
        format_length(True)


def test_a_childless_element_renders_self_closing() -> None:
    assert SvgElement("rect").render() == "<rect/>"


def test_attributes_render_in_the_order_given() -> None:
    element = SvgElement("rect", (("b", 1), ("a", 2)))
    assert element.render() == '<rect b="1" a="2"/>'


def test_text_children_render_on_the_same_line() -> None:
    element = SvgElement("text", (("x", 1), ("y", 2.5)), ("hi",))
    assert element.render() == '<text x="1" y="2.5">hi</text>'


def test_nested_elements_indent_by_two_spaces() -> None:
    element = SvgElement("g", (), (SvgElement("rect", (("x", 0),)),))
    assert element.render() == '<g>\n  <rect x="0"/>\n</g>'


def test_text_is_escaped_on_the_way_out() -> None:
    assert SvgElement("text", (), ("a<b",)).render() == "<text>a&lt;b</text>"


def test_attribute_values_are_escaped_too() -> None:
    element = SvgElement("rect", (("fill", 'a"b'),))
    assert element.render() == '<rect fill="a&quot;b"/>'


def test_the_indent_parameter_shifts_the_whole_subtree() -> None:
    element = SvgElement("g", (), (SvgElement("rect"),))
    assert element.render(indent=1) == "  <g>\n    <rect/>\n  </g>"


def test_float_attributes_use_the_length_formatter() -> None:
    element = SvgElement("rect", (("x", 0.1 + 0.2),))
    assert element.render() == '<rect x="0.3"/>'


def test_an_empty_tag_is_refused() -> None:
    with pytest.raises(ConfigError, match="usable SVG tag"):
        SvgElement("")


def test_a_tag_with_a_space_is_refused() -> None:
    with pytest.raises(ConfigError, match="usable SVG tag"):
        SvgElement("rect x")


def test_duplicate_attributes_are_refused() -> None:
    with pytest.raises(ConfigError, match="duplicate SVG attribute"):
        SvgElement("rect", (("x", 1), ("x", 2)))


def test_a_boolean_attribute_value_is_refused() -> None:
    with pytest.raises(ConfigError, match="string or a number"):
        SvgElement("rect", (("x", True),))


def test_a_bad_attribute_name_is_refused() -> None:
    with pytest.raises(ConfigError, match="usable SVG attribute"):
        SvgElement("rect", (("x y", 1),))


def test_a_non_string_non_element_child_is_refused() -> None:
    with pytest.raises(ConfigError, match="element or text"):
        SvgElement("g", (), (42,))


def test_a_small_document_is_pinned_byte_for_byte() -> None:
    document = svg_document(
        4,
        2,
        [SvgElement("rect", (("x", 0), ("y", 0), ("width", 4), ("height", 2), ("fill", "#eee")))],
        title="t",
    )
    opening = (
        '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        'width="4" height="2" viewBox="0 0 4 2">'
    )
    assert document == (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        + opening
        + "\n"
        + "  <title>t</title>\n"
        + '  <rect x="0" y="0" width="4" height="2" fill="#eee"/>\n'
        + "</svg>\n"
    )


def test_a_description_becomes_a_desc_element() -> None:
    document = svg_document(2, 2, (), description="d")
    assert "  <desc>d</desc>\n" in document


def test_title_and_description_come_before_the_content() -> None:
    document = svg_document(2, 2, [SvgElement("g")], title="t", description="d")
    assert document.index("<title>") < document.index("<desc>") < document.index("<g/>")


def test_a_document_needs_positive_integer_dimensions() -> None:
    with pytest.raises(ConfigError, match="positive integer"):
        svg_document(0, 2)
    with pytest.raises(ConfigError, match="positive integer"):
        svg_document(2, -1)
    with pytest.raises(ConfigError, match="positive integer"):
        svg_document(2.5, 2)


def test_the_same_document_always_digests_the_same() -> None:
    document = svg_document(2, 2, [SvgElement("rect")])
    assert digest_report(document) == digest_report(document)


def test_one_changed_coordinate_changes_the_digest() -> None:
    left = svg_document(2, 2, [SvgElement("rect", (("x", 1),))])
    right = svg_document(2, 2, [SvgElement("rect", (("x", 2),))])
    assert digest_report(left) != digest_report(right)


def test_the_digest_is_sixty_four_hex_characters() -> None:
    digest = digest_report("<svg/>")
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)


def test_the_digest_of_the_empty_string_is_the_standard_one() -> None:
    # SHA-256("") is fixed by the standard itself, so this pins the encoder
    # path end to end without depending on any report code.
    assert digest_report("") == ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")


def test_a_non_string_document_is_refused() -> None:
    with pytest.raises(ConfigError, match="needs a string"):
        digest_report(b"<svg/>")


def test_the_eval_subpackage_re_exports_the_report_core() -> None:
    from echoscribe import eval as evaluation

    assert evaluation.svg_document is svg_document
    assert evaluation.SvgElement is SvgElement
    for name in ("escape_text", "format_length", "digest_report"):
        assert name in evaluation.__all__


def test_a_matching_pair_renders_plain_cells() -> None:
    assert cell_fills("a", "a") == (COLOR_CORRECT, COLOR_CORRECT)


def test_a_substitution_amber_fills_both_rows() -> None:
    assert cell_fills("a", "b") == (COLOR_SUBSTITUTION, COLOR_SUBSTITUTION)


def test_a_deletion_reds_the_reference_and_gaps_the_hypothesis() -> None:
    assert cell_fills("a", None) == (COLOR_DELETION, COLOR_GAP)


def test_an_insertion_gaps_the_reference_and_reds_the_hypothesis() -> None:
    assert cell_fills(None, "a") == (COLOR_GAP, COLOR_INSERTION)


def test_the_grid_draws_two_cells_per_pair() -> None:
    section = alignment_section([("a", "a"), ("b", None)])
    rects = [child for child in section.element.children if child.tag == "rect"]
    assert len(rects) == 4
    assert section.width == 2 * CELL_WIDTH
    assert section.height == 2 * CELL_HEIGHT


def test_the_deletion_column_carries_the_documented_fills() -> None:
    section = alignment_section([("b", None)])
    rects = [child for child in section.element.children if child.tag == "rect"]
    assert dict(rects[0].attributes)["fill"] == COLOR_DELETION
    assert dict(rects[1].attributes)["fill"] == COLOR_GAP


def test_a_gap_cell_holds_no_text() -> None:
    section = alignment_section([("b", None)])
    texts = [child for child in section.element.children if child.tag == "text"]
    assert len(texts) == 1
    assert texts[0].children == ("b",)


def test_a_pair_with_neither_side_is_refused() -> None:
    with pytest.raises(ConfigError, match="at least one side"):
        alignment_section([(None, None)])


def test_an_empty_alignment_renders_an_empty_grid() -> None:
    section = alignment_section([])
    assert section.element.children == ()
    assert section.width == CELL_WIDTH


def test_equal_samples_all_land_in_the_first_bin() -> None:
    assert histogram_counts([2.0, 2.0, 2.0], bins=4) == (3, 0, 0, 0)


def test_the_maximum_lands_in_the_last_bin() -> None:
    assert histogram_counts([0.0, 1.0], bins=2) == (1, 1)


def test_hand_computed_binning() -> None:
    assert histogram_counts([0.0, 0.5, 1.0], bins=2) == (1, 2)


def test_counts_always_sum_to_the_sample_count() -> None:
    samples = [0.0, 0.3, 0.9, 1.2, 5.0]
    assert sum(histogram_counts(samples, bins=3)) == len(samples)


def test_a_single_sample_fills_one_bin() -> None:
    assert histogram_counts([3.5], bins=1) == (1,)
    assert histogram_counts([3.5], bins=5) == (1, 0, 0, 0, 0)


def test_an_empty_sample_list_is_refused() -> None:
    with pytest.raises(ConfigError, match="at least one sample"):
        histogram_counts([], bins=4)


def test_zero_bins_are_refused() -> None:
    with pytest.raises(ConfigError, match="bins must be"):
        histogram_counts([1.0], bins=0)


def test_a_non_finite_sample_is_refused() -> None:
    with pytest.raises(ConfigError, match="must be finite"):
        histogram_counts([1.0, float("nan")], bins=2)


def test_bar_heights_scale_with_the_counts() -> None:
    section = histogram_section([1.0, 2.0, 2.0], bins=3)
    rendered = section.element.render()
    # counts are (1, 0, 2): the middle bar is empty, the last is full height
    assert 'height="72"' in rendered
    assert 'height="0"' in rendered
    assert 'height="144"' in rendered


def test_the_baseline_is_drawn() -> None:
    section = histogram_section([1.0], bins=2)
    assert any(child.tag == "line" for child in section.element.children)


def test_every_bar_carries_its_count_label() -> None:
    section = histogram_section([0.0, 0.5, 1.0], bins=2)
    texts = [child for child in section.element.children if child.tag == "text"]
    assert [text.children for text in texts] == [("1",), ("2",)]


def test_the_plot_size_is_the_section_size() -> None:
    section = histogram_section([1.0, 2.0], bins=4, plot_width=400, plot_height=100)
    assert (section.width, section.height) == (400, 100)


def test_a_plot_smaller_than_the_headroom_is_refused() -> None:
    with pytest.raises(ConfigError, match="above the headroom"):
        histogram_section([1.0], plot_height=8)


def test_the_header_row_comes_first_and_is_bold() -> None:
    section = table_section([("u1", "WER 0.00%")])
    texts = [child for child in section.element.children if child.tag == "text"]
    assert texts[0].children == ("utterance",)
    assert ("font-weight", "bold") in texts[0].attributes
    assert texts[2].children == ("u1",)
    assert ("font-weight", "bold") not in texts[2].attributes


def test_cells_are_escaped_not_interpreted() -> None:
    section = table_section([("<b>", "a&b")])
    rendered = section.element.render()
    assert "&lt;b&gt;" in rendered
    assert "a&amp;b" in rendered
    assert "<b>" not in rendered


def test_cjk_cells_pass_through_as_text() -> None:
    section = table_section([("u1", "CER 16.67% | 今天")])
    assert "今天" in section.element.render()


def test_the_table_grows_one_row_per_entry() -> None:
    assert table_section([]).height == CELL_HEIGHT
    assert table_section([("a", "b"), ("c", "d")]).height == 3 * CELL_HEIGHT


def test_a_non_string_cell_is_refused() -> None:
    with pytest.raises(ConfigError, match="pair of strings"):
        table_section([("a", 3)])


def test_a_report_embeds_its_run_metadata() -> None:
    metadata = run_metadata("demo", seed=7, feature={"num_mels": 40})
    document = render_report("t", [alignment_section([("a", "a")])], metadata)
    assert f"run=demo seed=7 digest={metadata.digest()}" in document
    assert "<title>t</title>" in document


def test_the_same_report_renders_the_same_bytes() -> None:
    metadata = run_metadata("demo", seed=7)
    sections = [alignment_section([("a", "b")]), histogram_section([1.0, 2.0, 2.0])]
    first = render_report("r", sections, metadata)
    second = render_report("r", sections, metadata)
    assert first == second
    assert digest_report(first) == digest_report(second)


def test_the_description_and_the_metadata_share_the_desc() -> None:
    document = render_report(
        "t", [table_section([])], run_metadata("lab", seed=3), description="hello"
    )
    assert "<desc>hello\nrun=lab seed=3 digest=" in document


def test_the_sections_stack_top_to_bottom() -> None:
    document = render_report(
        "t",
        [alignment_section([], title="first"), table_section([], title="second")],
        run_metadata("x", seed=1),
    )
    assert document.index(">first<") < document.index(">second<")


def test_a_report_without_sections_is_refused() -> None:
    with pytest.raises(ConfigError, match="at least one section"):
        render_report("t", [], run_metadata("x", seed=1))


def test_a_report_without_a_title_is_refused() -> None:
    with pytest.raises(ConfigError, match="needs a title"):
        render_report("", [table_section([])], run_metadata("x", seed=1))


def test_a_section_needs_a_title() -> None:
    from echoscribe.eval.report import Section

    with pytest.raises(ConfigError, match="needs a title"):
        Section(title="", element=SvgElement("g"), width=1, height=1)


def test_a_section_needs_integer_sizes() -> None:
    from echoscribe.eval.report import Section

    with pytest.raises(ConfigError, match="non-negative integer"):
        Section(title="t", element=SvgElement("g"), width=1.5, height=1)


def test_a_section_needs_an_element() -> None:
    from echoscribe.eval.report import Section

    with pytest.raises(ConfigError, match="needs an SVG element"):
        Section(title="t", element="<g/>", width=1, height=1)
