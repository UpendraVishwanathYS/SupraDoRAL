from add_noise import *
from required_libraries import *

class W2V2FeatureExtraction:
    def __init__(self, data, type_, model, feature_extractor, device, wav_file_path = '/content/wav_final', noise_path = None, snr_dB = None):
        self.data = data
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device
        self.type_ = type_
        self.wav_file_path = wav_file_path
        self.noise_path = noise_path
        self.snr_dB = snr_dB


    def process_file(self, file_name):
        # Filter the DataFrame for the specified file name
        sub_df = self.data[self.data['file_name'] == file_name].copy()

        # Load the audio file only once
        path_to_audio_file = glob.glob(f'{self.wav_file_path}/*/*/{file_name}.wav')[0]

        y, sr = librosa.load(path_to_audio_file, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        if self.noise_path == None:
          waveform, sample_rate = torchaudio.load(path_to_audio_file)
          waveform = waveform.to(self.device)

        else:
          waveform, sample_rate = add_noise(path_to_audio_file, self.noise_path, self.snr_dB)
          waveform = waveform.to(self.device)

        # Extract features with the model in inference mode
        inputs = self.feature_extractor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = inputs['input_values']

        with torch.no_grad():
            features = self.model(inputs.to(self.device)).hidden_states

        # Convert features to numpy array
        features_np = np.array([hs.cpu().numpy() for hs in features])

        # Calculate duration per frame
        duration_per_frame = duration / features_np.shape[-2]

        def process_row(row):

            start_time = row[f'Start_{self.type_}']
            end_time = row[f'End_{self.type_}']
            start_frame_number = int(start_time / duration_per_frame)
            end_frame_number = int(end_time / duration_per_frame)
            initial_matrix_array_average = np.mean(features_np[:, :, start_frame_number:end_frame_number, :], axis=2)
            return initial_matrix_array_average

        # Apply the processing function to each row of the DataFrame
        sub_df['Feature_Vector'] = sub_df.apply(process_row, axis=1)
        return sub_df

    def process_all_files(self):
        results_list = []
        unique_files = self.data['file_name'].unique()

        for file_name in unique_files:
            processed_df = self.process_file(file_name)
            results_list.append(processed_df)

        # Combine all processed DataFrames into a single DataFrame
        w2v2_final_results_df = pd.concat(results_list, ignore_index=True)

        w2v2_final_results_df['Last_layer_W2V2'] = w2v2_final_results_df['Feature_Vector'].apply(lambda x: x[-1, :, :])
        return w2v2_final_results_df

  
class FastSpeechEmbeddingProcessor:
    def __init__(self, mapping_dataframe, embedding_dir):
        self.mapping_dataframe = mapping_dataframe
        self.embedding_dir = embedding_dir
        self.embedding_list = []

    def process_file(self, file_name, file_df):
        file_df = file_df.copy()
        file_df['Phone_index'] = range(len(file_df))

        # Load embeddings
        p_embedding = torch.load(f'{self.embedding_dir}/p/{file_name}.pt')
        d_embedding = torch.load(f'{self.embedding_dir}/d/{file_name}.pt')
        e_embedding = torch.load(f'{self.embedding_dir}/e/{file_name}.pt')

        for keys, group in file_df.groupby(['Word', 'Start_word', 'End_word', 'Syllable', 'Start_syl', 'End_syl', 'SylStress', 'Word_Label']):
            start_idx, end_idx = group['Phone_index'].iloc[0], group['Phone_index'].iloc[-1]

            # Compute mean embeddings
            p_embed_mean = p_embedding[:, start_idx:end_idx+1, :].mean(1).tolist()
            d_embed_mean = d_embedding[:, start_idx:end_idx+1, :].mean(1).tolist()
            e_embed_mean = e_embedding[:, start_idx:end_idx+1, :].mean(1).tolist()

            result_dict = dict(zip(['Word', 'Start_word', 'End_word', 'Syllable', 'Start_syl', 'End_syl', 'SylStress', 'Word_Label'], keys))
            result_dict.update({
                'file_name': file_name,
                'Type': group['Type'].unique().tolist(),
                'p_embedding': p_embed_mean,
                'd_embedding': d_embed_mean,
                'e_embedding': e_embed_mean
            })

            self.embedding_list.append(result_dict)

    def process_all_files(self):
        for file_name, file_df in self.mapping_dataframe.groupby('file_name'):
            self.process_file(file_name, file_df)

    def get_dataframe(self):
        return pd.DataFrame(self.embedding_list)
