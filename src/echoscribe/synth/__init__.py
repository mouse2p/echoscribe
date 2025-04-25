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
"""

__all__: list[str] = []
