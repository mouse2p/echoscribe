"""Report output is a byte contract, so the tests pin bytes."""

from __future__ import annotations

import pytest

from echoscribe.errors import ConfigError
from echoscribe.eval.report import SvgElement, escape_text, format_length, svg_document


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
