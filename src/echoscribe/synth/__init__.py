"""The synthetic acoustic laboratory.

Everything in this subpackage exists so the rest of echoscribe can be exercised
offline against *known* answers.  A seeded formant synthesizer renders waveforms
from a scripted lexicon, and the text it was rendered from is carried alongside
as planted ground truth.  There are no recordings, no pretrained weights and no
downloaded data anywhere in this package.

That constraint is the point.  A recognizer's accuracy on this audio says nothing
about its accuracy on speech; what it does say is whether the feature pipeline,
the decoder, the latency budget and the scorer behave the way their contracts
claim.  Every number produced from this laboratory is labelled synthetic.

Concretely, a result reported from this package means:

* the waveform was rendered by a formant synthesizer from an invented phone
  inventory, at a duration and pitch no speaker produced;
* the transcript is the script the renderer was handed, so it is exact by
  construction rather than transcribed by a human;
* any error rate is therefore a measure of the *pipeline* -- features, decoder,
  latency budget, scorer -- under a clean, fully specified acoustic condition.

It never means a model was evaluated on speech, and no result here should be read
as evidence about one.
"""

from echoscribe.synth.formant import (
    normalize_peak,
    render_phone,
    render_sequence,
    samples_for_duration,
)
from echoscribe.synth.lexicon import (
    ENGLISH_SPELLINGS,
    MANDARIN_SYLLABLES,
    Coverage,
    Lexicon,
    MandarinSyllable,
    PhoneSequence,
    TokenSpan,
    coverage_of,
    transcribe,
)
from echoscribe.synth.phones import ENGLISH_PHONES, FINALS, INITIALS, MANNERS, Phone
from echoscribe.synth.source import glottal_source, noise_source, phone_envelope
from echoscribe.synth.tone import TONES, ToneContour, tone_by_index

__all__ = [
    "ENGLISH_PHONES",
    "ENGLISH_SPELLINGS",
    "FINALS",
    "INITIALS",
    "MANNERS",
    "MANDARIN_SYLLABLES",
    "TONES",
    "Coverage",
    "Lexicon",
    "MandarinSyllable",
    "Phone",
    "PhoneSequence",
    "TokenSpan",
    "ToneContour",
    "coverage_of",
    "glottal_source",
    "noise_source",
    "normalize_peak",
    "phone_envelope",
    "render_phone",
    "render_sequence",
    "samples_for_duration",
    "tone_by_index",
    "transcribe",
]
