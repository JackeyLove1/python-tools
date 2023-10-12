import torch

PAD = "<PAD>"
class BeamSearchNode:
    def __init__(self, sequence, score):
        self.sequence = sequence
        self.score = score


def simple_next_word_probs(sequence):
    if sequence[-1] == PAD:
        return {}
    return {"apple": 0.3, "like": 0.35, "peach": 0.2, "banana": 0.15}

def beam_search(initial_sequence, next_word_probs_func, num_beams, max_sequence_length):
    initial_node = BeamSearchNode(sequence=initial_sequence, score=1.0)
    candidates = [initial_node]

    final_candidates = []  # 最终的候选序列
    while candidates and len(final_candidates) < num_beams:
        candidates.sort(key=lambda x : -x.score)
        current_node = candidates.pop(0)
        if current_node.sequence[-1] == PAD or len(current_node.sequence) >= max_sequence_length:
            final_candidates.append(current_node)
        else:
            next_word_probs = next_word_probs_func(current_node.sequence)
            for next_word, next_word_prob in next_word_probs.items():
                new_sequence = current_node.sequence + [next_word]
                new_scores = current_node.score * next_word_prob
                new_node = BeamSearchNode(new_sequence, new_scores)
                candidates.append(new_node)
    return [candidate.sequence for candidate in candidates]
