"""
Functional undo/redo helpers for nnInteractiveInferenceSession.

Instead of subclassing the session, callers manage two :class:`collections.deque`
instances (one for undo, one for redo) and pass them to each helper together
with the active session.  This keeps the session class unchanged and makes the
history fully owned by the caller.

Typical usage::

    from collections import deque
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    from nnInteractive.inference.undo_redo import (
        capture_prediction_state,
        undo_prediction,
        redo_prediction,
    )

    session = nnInteractiveInferenceSession(...)
    undo_history: deque = deque(maxlen=7)
    redo_history: deque = deque(maxlen=7)

    session.add_point_interaction(coords, include_interaction=True, run_prediction=True)
    capture_prediction_state(session, undo_history, redo_history)

    undo_prediction(session, undo_history, redo_history)
    redo_prediction(session, undo_history, redo_history)
"""

from collections import deque
from typing import Union

import numpy as np
import torch

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession


def _clone_target_buffer(
    target_buffer: Union[np.ndarray, torch.Tensor, None],
) -> Union[np.ndarray, torch.Tensor, None]:
    """Return a copy of *target_buffer*, preserving its type."""
    if target_buffer is None:
        return None
    if isinstance(target_buffer, np.ndarray):
        return target_buffer.copy()
    if isinstance(target_buffer, torch.Tensor):
        return target_buffer.clone()
    raise RuntimeError('target_buffer must be np.ndarray or torch.Tensor')


def capture_prediction_state(
    session: nnInteractiveInferenceSession, undo_history: deque, redo_history: deque
) -> None:
    """Push the current session state onto *undo_history* and clear *redo_history*.

    Call this immediately after every prediction to make the resulting state
    available for undoing.

    Args:
        session: An active :class:`nnInteractiveInferenceSession`.
        undo_history: Deque used to store states that can be undone.  Configure
            ``maxlen`` on creation to cap memory usage.
        redo_history: Deque used to store states that can be redone.  Cleared
            by this call so that a new prediction always discards the redo stack.
    """
    if session.interactions is None or session.target_buffer is None:
        return
    undo_history.append(
        {
            'interactions': session.interactions.clone(),
            'target_buffer': _clone_target_buffer(session.target_buffer),
        }
    )
    redo_history.clear()


def restore_prediction_state(session: nnInteractiveInferenceSession, state: dict) -> None:
    """Overwrite the session's live tensors with the data stored in *state*.

    Args:
        session: An active :class:`nnInteractiveInferenceSession`.
        state: A state dict as produced by :func:`capture_prediction_state`.
    """
    if session.interactions is None or session.target_buffer is None:
        raise RuntimeError('Cannot restore state without interactions and target_buffer')

    session.interactions.copy_(state['interactions'])
    target_state = state['target_buffer']

    if isinstance(session.target_buffer, np.ndarray):
        if not isinstance(target_state, np.ndarray):
            raise RuntimeError('Stored target buffer type does not match current target buffer type')
        np.copyto(session.target_buffer, target_state)
    elif isinstance(session.target_buffer, torch.Tensor):
        if not isinstance(target_state, torch.Tensor):
            raise RuntimeError('Stored target buffer type does not match current target buffer type')
        session.target_buffer.copy_(target_state.to(session.target_buffer.device))
    else:
        raise RuntimeError('target_buffer must be np.ndarray or torch.Tensor')

    session.new_interaction_centers = []
    session.new_interaction_zoom_out_factors = []


def undo_prediction(
    session: nnInteractiveInferenceSession, undo_history: deque, redo_history: deque
) -> bool:
    """Undo the most recent prediction by restoring the previous session state.

    The current (most recent) state is moved to *redo_history* so it can be
    recovered with :func:`redo_prediction`.

    Args:
        session: An active :class:`nnInteractiveInferenceSession`.
        undo_history: Deque of saved states (managed by the caller).
        redo_history: Deque of states that can be redone (managed by the caller).

    Returns:
        ``True`` if the undo was performed, ``False`` if there was nothing to undo
        (i.e. *undo_history* holds at most one entry).
    """
    if len(undo_history) <= 1:
        return False
    current_state = undo_history.pop()
    redo_history.append(current_state)
    restore_prediction_state(session, undo_history[-1])
    return True


def redo_prediction(
    session: nnInteractiveInferenceSession, undo_history: deque, redo_history: deque
) -> bool:
    """Redo the last undone prediction by restoring the next session state.

    The restored state is moved back to *undo_history*.

    Args:
        session: An active :class:`nnInteractiveInferenceSession`.
        undo_history: Deque of saved states (managed by the caller).
        redo_history: Deque of states that can be redone (managed by the caller).

    Returns:
        ``True`` if the redo was performed, ``False`` if there was nothing to redo
        (i.e. *redo_history* is empty).
    """
    if len(redo_history) == 0:
        return False
    restored_state = redo_history.pop()
    undo_history.append(restored_state)
    restore_prediction_state(session, restored_state)
    return True
