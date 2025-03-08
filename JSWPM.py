from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class JSWPM(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_classes_word=2, num_classes_syllable=2, lstm_num_layers=1):
        super(JSWPM, self).__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2  # Bidirectional

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=lstm_num_layers,
                            batch_first=True, bidirectional=True)

        self.word_fc = nn.Linear(hidden_size * self.num_directions, num_classes_word)
        self.syllable_fc = nn.Linear(hidden_size * self.num_directions, num_classes_syllable)

    def forward(self, x, lengths):
        # Pack padded sequence
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)

        # Unpack to retrieve the hidden states
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Concatenate the last hidden states from both directions
        hn = hn.view(self.lstm.num_layers, self.num_directions, -1, self.hidden_size)
        last_hidden = torch.cat((hn[-1, 0], hn[-1, 1]), dim=-1)

        # Word prediction (from the concatenated last hidden states)
        word_output = self.word_fc(last_hidden)
        word_output = F.softmax(word_output, dim=-1)

        # Syllable prediction (from each timestep's output)
        syllable_output = self.syllable_fc(output)
        syllable_output = F.softmax(syllable_output, dim=-1)

        return word_output, syllable_output