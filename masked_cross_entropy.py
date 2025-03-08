from required_libraries import *
def masked_cross_entropy(syllable_output, syllable_labels, lengths):
    batch_size, max_seq_len, num_classes = syllable_output.size()
    syllable_labels = syllable_labels.view(-1)  # Flatten the syllable labels
    syllable_output = syllable_output.view(-1, num_classes)  # Flatten syllable outputs

    # Mask for ignoring padding labels
    mask = syllable_labels != -1
    syllable_labels = syllable_labels[mask]
    syllable_output = syllable_output[mask]

    return F.cross_entropy(syllable_output, syllable_labels)