# Prototype results are not the confirmation results

`smoke/` (seed301) used an earlier ReLU server suffix, without the final cost ledger.
`pilot/` (seed302, train SNR10) used the smooth suffix and revealed that the fixed
regularizer can outperform the trace correction, but preceded the final cold-start
and energy accounting fixes. These raw files are preserved as exploration history.

All headline tables use `fashion20/`, `fashion10/`, `width20/`, seeds311–313.
Their run_sfl.py code hash identifies the frozen core implementation. No result
from the pilot is pooled with these three seeds. The digital-regularizer control
was added after seeing the fashion20 comparison and is explicitly diagnostic.

There is no model checkpoint selection by final test accuracy. All reported
training runs use the final round80 model. Different final channel draws are
averaged within a training seed, not counted as independent trained models.
