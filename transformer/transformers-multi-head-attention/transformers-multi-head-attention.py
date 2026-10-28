import numpy as np
import torch

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    batch_size, query_len, d_model = Q.shape
    key_len = K.shape[1]
    d_k = d_model // num_heads
    
    # linear projection
    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v

    # multi-head
    Q = Q.reshape(batch_size, query_len, num_heads, d_k)
    K = K.reshape(batch_size, key_len, num_heads, d_k)
    V = V.reshape(batch_size, key_len, num_heads, d_k)

    # transpose
    Q = Q.transpose(0, 2, 1, 3)
    K = K.transpose(0, 2, 1, 3)
    V = V.transpose(0, 2, 1, 3)

    # scaled dot-product attention
    scores = Q @ K.transpose(0,1,3,2)
    scores = scores / np.sqrt(d_k)

    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    # weights = torch.softmax(scores, dim = -1)
    attention = weights @ V

    #concat
    output = attention.transpose(0,2,1,3)
    output = output.reshape(batch_size,query_len,d_model)

    return output @ W_o
    
    
    