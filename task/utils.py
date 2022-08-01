import numpy as np
import torch

def extract_notes_wo_velocity(onsets, frames, onset_threshold=0.5, frame_threshold=0.5, rule='rule1'):
    """
    Finds the note timings based on the onsets and frames information
    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    velocities: np.ndarray of velocity values
    """
    onsets = (onsets > onset_threshold).astype(int)
    frames = (frames > frame_threshold).astype(int)
    onset_diff = np.concatenate([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], axis=0) == 1 # Make sure the activation is only 1 time-step
    if rule=='rule2':
        pass
    elif rule=='rule1':
        # Use in simple models
        onset_diff = onset_diff & (frames==1) # New condition such that both onset and frame on to get a note
    else:
        raise NameError('Please enter the correct rule name')

    pitches = []
    intervals = []

    dim1, dim2 = np.nonzero(onset_diff)    
    for frame, pitch in zip(dim1, dim2):
        # The below code is for torch
        # frame = nonzero[0].item()
        # pitch = nonzero[1].item()

        onset = frame
        offset = frame

        # This while loop is looking for where does the note ends        
        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            offset += 1
            if offset == onsets.shape[0]:
                break

        # After knowing where does the note start and end, we can return the pitch information (and velocity)        
        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])

    return np.array(pitches), np.array(intervals)