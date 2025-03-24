# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

"""also using ChatGPT 
and help from Isaiah Hazelwoods code 
"""

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    
    goal here is to sample with replacement from the minority class so that both classes have the same number; picking the same item multiple times is allowed. 
    we do this because neural networks perform poorly on imbalanced data... can just predict majority class and still get high accuracy but poor real-world performance. 
    """
    #separate sequences into positive and negative classes 
    pos_indices = np.where(np.array(labels))[0] #adding [0] extracts the actual 1D array of indices, as np.where() returns solely a tuple (array([idx, idx, idx]))
    neg_indices = np.where(~np.array(labels))[0]

    if len(neg_indices) > len(pos_indices):
        sampled_indices = [i for i in neg_indices]
        sampled_indices.extend(np.random.choice(pos_indices, len(neg_indices), replace=True)) #add enough positive indices to match negatives "extending" the minor by random

    else:
        sampled_indices = [i for i in pos_indices]
        sampled_indices.extend(np.random.choice(neg_indices, len(pos_indices), replace=True))
    
    sampled_seqs = [seqs[i] for i in sampled_indices]
    #grabs the label of at the index i from the original list
    sampled_labels = [labels[i] for i in sampled_indices] 
    return sampled_seqs, sampled_labels



def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].

    hot encoding is a way to convert categorical data into a format that a machine learning model can understand - numbers. 
    so here categorical is a letter A, T, G or C
    """
    #mapping for each base
    mapping = {
        'A': [1, 0, 0, 0], 
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1]
    }

    encoded_seqs = []

    for seq in seq_arr:
        one_hot = []
        for base in seq.upper(): #make sure we are in uppercase
            one_hot.extend(mapping.get(base, [0, 0, 0, 0])) #fallback for unknowns 
        encoded_seqs.append(one_hot)

    return np.array(encoded_seqs)


    