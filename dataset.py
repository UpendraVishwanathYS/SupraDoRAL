from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from required_libraries import *

class LSTMDataset(Dataset):
    def __init__(self, features, word_labels, syllable_labels):
        self.features = features
        self.word_labels = word_labels
        self.syllable_labels = syllable_labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.word_labels[idx], self.syllable_labels[idx]

# Collate Function for Padding
def collate_fn(batch):
    features, word_labels, syllable_labels = zip(*batch)
    lengths = [len(f) for f in features]

    padded_features = pad_sequence(features, batch_first=True)
    padded_syllable_labels = pad_sequence(
        [torch.tensor(seq) for seq in syllable_labels], batch_first=True, padding_value=-1
    )
    return padded_features, torch.tensor(word_labels), padded_syllable_labels, torch.tensor(lengths)


def DataLoaders(df, embedding_type, batch_size=32, device='cuda'):
    """
    Processes the dataset to create DataLoaders for training, validation, and testing.

    Args:
        df (pd.DataFrame): Input dataset containing features and labels.
        embedding_type (str): The embedding type to use.

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_weights)
    """

    # Split dataset into training and testing
    final_database_train = df[df['Type'] == 'train'].copy()
    final_database_test = df[df['Type'] == 'test'].copy()

    # Vectorized encoding
    final_database_train['Label'] = final_database_train['SylStress'].map(encode)
    final_database_test['Label'] = final_database_test['SylStress'].map(encode)

    # Group by word-level attributes
    train_grouped = final_database_train.groupby(['file_name', 'Word', 'Start_word', 'End_word'])
    test_grouped = final_database_test.groupby(['file_name', 'Word', 'Start_word', 'End_word'])

    # Convert features and labels efficiently
    def process_grouped_data(grouped_df, embedding_type):
        grouped_data = grouped_df[embedding_type].apply(np.stack).values
        feature_tensors = [torch.tensor(x).type(torch.float32).squeeze(1) for x in grouped_data]

        word_labels = torch.tensor(grouped_df['Word_Label'].first().values, dtype=torch.float32)

        syllable_labels = grouped_df['Label'].apply(np.array).values
        syllable_tensors = [torch.tensor(x, dtype=torch.float32) for x in syllable_labels]

        return feature_tensors, word_labels, syllable_tensors


    # Process train and test data
    train_data, train_word_labels, train_syllable_labels = process_grouped_data(train_grouped, embedding_type)
    test_data, test_word_labels, test_syllable_labels = process_grouped_data(test_grouped, embedding_type)

    # Convert labels to lists
    word_labels = [int(x) for x in train_word_labels]
    syllable_labels = [[int(j) for j in x] for x in train_syllable_labels]

    test_word_labels = [int(x) for x in test_word_labels]
    test_syllable_labels = [[int(j) for j in x] for x in test_syllable_labels]

    # Stratified Train-Val Split
    X_train, X_val, y_train_word, y_val_word, y_train_syllable, y_val_syllable = train_test_split(
        train_data, word_labels, syllable_labels,
        test_size=0.2, stratify=word_labels, random_state=42
    )

    # Create Dataset and DataLoaders
    train_dataset = LSTMDataset(X_train, y_train_word, y_train_syllable)
    val_dataset = LSTMDataset(X_val, y_val_word, y_val_syllable)
    test_dataset = LSTMDataset(test_data, test_word_labels, test_syllable_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    # Compute class weights
    class_weights = compute_class_weight(class_weight="balanced",
                                          classes=np.unique(y_train_word),
                                          y=y_train_word)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    return train_loader, val_loader, test_loader, class_weights