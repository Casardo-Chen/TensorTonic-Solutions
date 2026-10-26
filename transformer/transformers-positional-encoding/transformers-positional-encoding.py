import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """

    # pos
    pos = np.arange(seq_length, dtype = np.float64)[:,np.newaxis]

    # i
    i = np.arange(0, d_model, 2, dtype = np.float64)

    div_term = 10000 ** ( i / d_model)    

    pe = np.zeros((seq_length, d_model), dtype = np.float64)

    angles = pos / div_term

    pe[:,0::2] = np.sin(angles)
    pe[:,1::2] = np.cos(angles)
    
    return pe
    