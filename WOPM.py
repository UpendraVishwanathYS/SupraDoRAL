from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class WOPM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes_word, num_classes_syllable, lstm_num_layers):
        super(WOPM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=lstm_num_layers,
                            batch_first=True, bidirectional=True)
        self.word_fc = nn.Linear(hidden_size, num_classes_word)

    def forward(self, x, lengths):
        # Pack padded sequence
        # print(f'Input shape: {x.shape}')
        # print(f'Lengths shape: {lengths.shape}')
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)


        # Unpack to retrieve the hidden states
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # print(f'Output shape: {output.shape}')
        # print(f'hn shape: {hn[-1].shape}')
        # print(f'cn shape: {cn.shape}')

        # Word prediction (from the last hidden state of LSTM)
        word_output = self.word_fc(hn[-1])
        word_output = F.softmax(word_output, dim=-1)

        return word_output