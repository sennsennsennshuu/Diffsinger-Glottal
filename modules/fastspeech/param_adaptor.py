"""Variance parameter adaptor stub for ONNX deployment compatibility."""

# VARIANCE_CHECKLIST — mapping of variance parameter names to their
# output dimensions.  The full list is:
#   pitch (1), energy (1), breathiness (1), voicing (1), tension (1)
# = 5 parameters × 1 dim each.
VARIANCE_CHECKLIST = {
    'pitch': 1,
    'energy': 1,
    'breathiness': 1,
    'voicing': 1,
    'tension': 1,
}
