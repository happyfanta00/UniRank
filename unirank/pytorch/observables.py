"""Optional hooks for an external offline research harness ("MR").

Everything here is off unless the corresponding environment variable is set, and
nothing in UniRank calls into it by default. With `MR_OBSV` and `MR_MAX_STEPS`
unset, the two call sites in `rank_model.py` / `run_expid.py` are no-ops and
training behaviour is byte-for-byte what it was before.

Why environment variables and not config keys: the harness lets an automated
agent propose config overrides, and every config key ends up in the synthesized
experiment block. A knob like `max_steps` sitting in the config would let the
agent shorten training and report the resulting AUC change as an improvement.
Environment variables are set by the launcher, which the agent cannot reach.

Two hooks:

`emit_observable(name, value, step)`
    A model modification is expected to declare, up front, an internal quantity
    that should move if its stated mechanism is real (a gate activation rate, a
    routing entropy, ...) and then emit it from the forward pass. The harness
    parses these lines back out of the log and checks the declaration against
    what actually happened, which is how it tells "the metric moved because the
    mechanism worked" apart from "the metric moved". Nothing here interprets the
    values; this is only the printing side.

`max_steps()`
    Number of optimizer steps after which training should stop, or 0 for "run
    normally". Used for a cheap A/B sanity run: build the modified model with its
    feature flag *off*, train a few hundred steps, and compare the loss series
    against the baseline. It has to be cheap to be worth running -- a full epoch
    is 45 minutes here, and a check that costs as much as the thing it guards
    does not get run.
"""

import logging
import os

import torch.distributed as dist


def _is_main_process():
    """Rank 0, or any process when not running under DDP.

    Matches `RankModel._is_main_process`, but has to stand alone: this module is
    imported from places that have no model instance.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


_OBSV_ENABLED = os.environ.get("MR_OBSV", "0") == "1"


def enabled():
    return _OBSV_ENABLED


def emit_observable(name, value, step):
    """Log one observation as `[OBSV] name=<name> value=<v> step=<i>`.

    Rank 0 only: under DDP every rank would emit its own value, and a consumer
    reading them as one series would silently mix two different numbers.

    `.8g` keeps enough digits for the value to be compared across runs while
    staying short enough to print every step for a whole epoch.
    """
    if not _OBSV_ENABLED:
        return
    if not _is_main_process():
        return
    logging.info("[OBSV] name={} value={:.8g} step={}".format(name, float(value), int(step)))


def max_steps():
    """`MR_MAX_STEPS`, or 0 when unset/unparseable.

    Deliberately falls back to 0 (= normal full training) rather than raising: a
    malformed value should not be able to turn a real training run into a
    silently truncated one.
    """
    try:
        return max(0, int(os.environ.get("MR_MAX_STEPS", "0") or 0))
    except ValueError:
        return 0
