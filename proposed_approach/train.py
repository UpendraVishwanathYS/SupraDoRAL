import torch
import copy
from sklearn.metrics import accuracy_score, f1_score

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion_word,
    criterion_syllable=None,
    num_epochs=10,
    lamda=0.5,
    save_path="./best_model.pth",
    patience=5,
    device = 'cuda'
):
    best_model = None
    best_f1 = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_word_loss = 0
        train_syllable_loss = 0
        train_total_loss = 0

        train_word_preds = []
        train_word_labels = []
        train_syllable_preds = []
        train_syllable_labels = []

        for features, word_labels, syllable_labels, lengths in train_loader:
            features, word_labels, lengths = features.to(device), word_labels.to(device), lengths.to(device)
            if criterion_syllable:
                syllable_labels = syllable_labels.to(device)
            
            optimizer.zero_grad()

            if criterion_syllable:
                word_pred, syllable_pred = model(features, lengths)
                word_loss = criterion_word(word_pred, word_labels)
                syllable_loss = criterion_syllable(syllable_pred, syllable_labels, lengths)
                loss = lamda * word_loss + (1 - lamda) * syllable_loss
            else:
                word_pred = model(features, lengths)
                word_loss = criterion_word(word_pred, word_labels)
                loss = word_loss
            
            loss.backward()
            optimizer.step()

            train_word_loss += word_loss.item()
            train_total_loss += loss.item()

            if criterion_syllable:
                train_syllable_loss += syllable_loss.item()
                syllable_preds = torch.argmax(syllable_pred, dim=2).cpu().numpy()
                for i, length in enumerate(lengths):
                    train_syllable_preds.extend(syllable_preds[i, :length])
                    train_syllable_labels.extend(syllable_labels[i, :length].cpu().numpy())
            
            train_word_preds.extend(torch.argmax(word_pred, dim=1).cpu().numpy())
            train_word_labels.extend(word_labels.cpu().numpy())

        train_word_loss /= len(train_loader)
        train_syllable_loss /= len(train_loader) if criterion_syllable else 0
        train_total_loss /= len(train_loader)

        train_word_acc = accuracy_score(train_word_labels, train_word_preds)
        train_word_f1 = f1_score(train_word_labels, train_word_preds, average="weighted")
        train_syllable_acc = accuracy_score(train_syllable_labels, train_syllable_preds) if criterion_syllable else 0
        train_syllable_f1 = f1_score(train_syllable_labels, train_syllable_preds, average="weighted") if criterion_syllable else 0

        model.eval()
        val_word_loss = 0
        val_syllable_loss = 0
        all_word_preds = []
        all_word_labels = []
        all_syllable_preds = []
        all_syllable_labels = []

        with torch.no_grad():
            for features, word_labels, syllable_labels, lengths in val_loader:
                features, word_labels, lengths = features.to(device), word_labels.to(device), lengths.to(device)
                if criterion_syllable:
                    syllable_labels = syllable_labels.to(device)

                if criterion_syllable:
                    word_pred, syllable_pred = model(features, lengths)
                    word_loss = criterion_word(word_pred, word_labels)
                    syllable_loss = criterion_syllable(syllable_pred, syllable_labels, lengths)
                    loss = lamda * word_loss + (1 - lamda) * syllable_loss
                else:
                    word_pred = model(features, lengths)
                    word_loss = criterion_word(word_pred, word_labels)
                    loss = word_loss

                val_word_loss += word_loss.item()
                if criterion_syllable:
                    val_syllable_loss += syllable_loss.item()

                word_preds = torch.argmax(word_pred, dim=1).cpu().numpy()
                all_word_preds.extend(word_preds)
                all_word_labels.extend(word_labels.cpu().numpy())

                if criterion_syllable:
                    syllable_preds = torch.argmax(syllable_pred, dim=2).cpu().numpy()
                    for i, length in enumerate(lengths):
                        all_syllable_preds.extend(syllable_preds[i, :length])
                        all_syllable_labels.extend(syllable_labels[i, :length].cpu().numpy())

        val_word_loss /= len(val_loader)
        val_syllable_loss /= len(val_loader) if criterion_syllable else 0

        word_acc = accuracy_score(all_word_labels, all_word_preds)
        word_f1 = f1_score(all_word_labels, all_word_preds, average="weighted")
        syllable_acc = accuracy_score(all_syllable_labels, all_syllable_preds) if criterion_syllable else 0
        syllable_f1 = f1_score(all_syllable_labels, all_syllable_preds, average="weighted") if criterion_syllable else 0

        print(f"Epoch {epoch + 1}:")
        print(f"Train Word Loss: {train_word_loss:.4f}, Word Accuracy: {train_word_acc:.4f}, Word F1: {train_word_f1:.4f}")
        if criterion_syllable:
            print(f"Syllable Accuracy: {train_syllable_acc:.4f}, Syllable F1: {train_syllable_f1:.4f}")
        print(f"Validation Word Loss: {val_word_loss:.4f}, Word Accuracy: {word_acc:.4f}, Word F1: {word_f1:.4f}")
        if criterion_syllable:
            print(f"Validation Syllable Accuracy: {syllable_acc:.4f}, Syllable F1: {syllable_f1:.4f}")

        if word_f1 > best_f1:
            best_f1 = word_f1
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, save_path)
            print(f"Best model saved at {save_path}")

        if val_word_loss < best_val_loss:
            best_val_loss = val_word_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs with no improvement.")
                break
    
    print("Training Complete. Best Word F1 Score: {:.4f}".format(best_f1))
