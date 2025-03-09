# SupraDoRAL
Automatic Word Prominence Detection Using Suprasegmental Dependencies of Representations with Acoustic and Linguistic Context

# Installing Dependencies
To install dependencies, create a conda or virtual environment with Python 3 and then run ```pip install -r requirements.txt```

# Training:
(a) To train the Joint Syllable-Word Prominence Model (JSWPM) on w2v2-large-960h (last layer) features simply run ```python3 main.py```.
(b) To train the Word-Prominence Detection (WOPM) on w2v2-large-960h (last layer) features simply run ```python3 main.py --classification_model "WOPM" ```.

The default values are specified below:
```
    parser.add_argument('--w2v2_model_name', type=str, default="facebook/wav2vec2-large-960h")
    parser.add_argument('--layer_number', type=int, default=-1)
    parser.add_argument('--path_to_database', type=str, default='./ITA_word_syllable_phone_mapping_dataframe.csv')
    parser.add_argument('--wav_file_path', type=str, default='./wav_final')
    parser.add_argument('--feature_type', type=str, default='w2v2')
    parser.add_argument('--embedding_dir', type=str, default=None)
    parser.add_argument('--feature_extraction_level', type=str, default='syl', choices=['syl', 'word'])
    parser.add_argument('--noise_path', type=str, default=None)
    parser.add_argument('--snr_dB', type=float, default=None)
    parser.add_argument('--classification_model', type=str, default='JSWPM')
    parser.add_argument('--input_feature_dim', type=int, default=1024)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--lstm_num_layers', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_model_chkpts', type=str, default='./best_model.pth')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
```
