from required_libraries import *
from utils import create_syllable_level_database, create_word_level_database
from feature_extraction import W2V2FeatureExtraction, FastSpeechEmbeddingProcessor
from dataset import DataLoaders
from masked_cross_entropy import masked_cross_entropy
from train import train
from models import JSWPM, WOPM
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Joint Syllable-Word Prominence Model")

    # Add arguments
    parser.add_argument('--path_to_database', type=str, default='./ITA_word_syllable_phone_mapping_dataframe.csv')
    parser.add_argument('--wav_file_path', type=str, default='./wav_final')
    parser.add_argument('--feature_type', type=str, default='w2v2')
    parser.add_argument('--w2v2_model_name', type=str, default="facebook/wav2vec2-large-960h")
    parser.add_argument('--w2v2_layer_number', type=int, default=-1)
    parser.add_argument('--fastspeech_type', type=str, default='p_embedding')
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

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    word_syllable_phone_mapping_dataframe = pd.read_csv(args.path_to_database)


    if args.feature_extraction_level == 'syl':
        database = create_syllable_level_database(word_syllable_phone_mapping_dataframe)
    else:
        database = create_word_level_database(word_syllable_phone_mapping_dataframe)

    if args.feature_type == 'fast_speech':
      processor = FastSpeechEmbeddingProcessor(database, embedding_dir = args.embedding_dir)
      processor.process_all_files()
      feature_database = processor.get_dataframe()
      embedding_type = args.fastspeech_type
    else:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.w2v2_model_name, output_hidden_states=True)
        model = Wav2Vec2Model.from_pretrained(args.w2v2_model_name, output_hidden_states=True).to(device)
        audio_processor = W2V2FeatureExtraction(
            database, args.feature_extraction_level, model, feature_extractor, device,
            wav_file_path=args.wav_file_path, noise_path=args.noise_path, snr_dB=args.snr_dB
        )

        feature_database = audio_processor.process_all_files()
        feature_database['Last_layer_W2V2'] = w2v2_database['Feature_Vector'].apply(lambda x: x[args.layer_number, :, :])
        embedding_type = 'Last_layer_W2V2'

    train_loader, val_loader, test_loader, class_weights = DataLoaders(feature_database, embedding_type, batch_size=args.batch_size, device=device)

    model = eval(args.classification_model)(args.input_feature_dim, args.hidden_size, args.lstm_num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion_word = nn.CrossEntropyLoss()
    criterion_syllable = masked_cross_entropy

    train(model, train_loader, val_loader, optimizer, criterion_word,
          criterion_syllable, num_epochs=args.num_epochs, patience=args.patience,
          save_path=args.save_model_chkpts)

    # Test performance
    model.load_state_dict(torch.load(args.save_model_chkpts))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, word_labels, syllable_labels, lengths in test_loader:
            features, word_labels, lengths = features.to(device), word_labels.to(device), lengths.to(device)
            word_pred, _ = model(features, lengths)
            word_preds = torch.argmax(word_pred, dim=1).cpu().numpy()
            all_preds.extend(word_preds)
            all_labels.extend(word_labels.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"Test Accuracy: {accuracy}")
    print(f"Test F1-score: {f1}")


if __name__ == '__main__':
    main()
