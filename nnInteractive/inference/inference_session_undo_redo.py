from collections import deque
from typing import Union

import numpy as np
import torch

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession


class nnInteractiveInferenceSessionUndoRedo(nnInteractiveInferenceSession):
    def __init__(
        self,
        device: torch.device = torch.device('cuda'),
        use_torch_compile: bool = False,
        verbose: bool = False,
        torch_n_threads: int = 8,
        do_autozoom: bool = True,
        use_pinned_memory: bool = True,
        prediction_history_size: int = 7,
    ):
        super().__init__(
            device=device,
            use_torch_compile=use_torch_compile,
            verbose=verbose,
            torch_n_threads=torch_n_threads,
            do_autozoom=do_autozoom,
            use_pinned_memory=use_pinned_memory,
        )
        if prediction_history_size < 1:
            raise ValueError('prediction_history_size must be >= 1')
        self.prediction_history_size = prediction_history_size
        self._undo_history = deque(maxlen=self.prediction_history_size)
        self._redo_history = deque(maxlen=self.prediction_history_size)

    def _reset_session(self):
        super()._reset_session()
        self._clear_prediction_history()

    def set_target_buffer(self, target_buffer: Union[np.ndarray, torch.Tensor]):
        super().set_target_buffer(target_buffer)
        self._clear_prediction_history()

    def _predict(self):
        super()._predict()
        self._capture_prediction_state()

    def _clear_prediction_history(self):
        self._undo_history.clear()
        self._redo_history.clear()

    def _clone_target_buffer(self):
        if self.target_buffer is None:
            return None
        if isinstance(self.target_buffer, np.ndarray):
            return self.target_buffer.copy()
        if isinstance(self.target_buffer, torch.Tensor):
            return self.target_buffer.clone()
        raise RuntimeError('target_buffer must be np.ndarray or torch.Tensor')

    def _capture_prediction_state(self):
        if self.interactions is None or self.target_buffer is None:
            return
        self._undo_history.append(
            {
                'interactions': self.interactions.clone(),
                'target_buffer': self._clone_target_buffer(),
            }
        )
        self._redo_history.clear()

    def _restore_prediction_state(self, state: dict):
        if self.interactions is None or self.target_buffer is None:
            raise RuntimeError('Cannot restore state without interactions and target_buffer')

        self.interactions.copy_(state['interactions'])
        target_state = state['target_buffer']

        if isinstance(self.target_buffer, np.ndarray):
            if not isinstance(target_state, np.ndarray):
                raise RuntimeError('Stored target buffer type does not match current target buffer type')
            np.copyto(self.target_buffer, target_state)
        elif isinstance(self.target_buffer, torch.Tensor):
            if not isinstance(target_state, torch.Tensor):
                raise RuntimeError('Stored target buffer type does not match current target buffer type')
            self.target_buffer.copy_(target_state.to(self.target_buffer.device))
        else:
            raise RuntimeError('target_buffer must be np.ndarray or torch.Tensor')

        self.new_interaction_centers = []
        self.new_interaction_zoom_out_factors = []

    def undo_prediction(self) -> bool:
        if len(self._undo_history) <= 1:
            return False
        current_state = self._undo_history.pop()
        self._redo_history.append(current_state)
        self._restore_prediction_state(self._undo_history[-1])
        return True

    def redo_prediction(self) -> bool:
        if len(self._redo_history) == 0:
            return False
        restored_state = self._redo_history.pop()
        self._undo_history.append(restored_state)
        self._restore_prediction_state(restored_state)
        return True
