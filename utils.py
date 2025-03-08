from required_libraries import *

def create_word_level_database(word_syllable_phone_mapping_dataframe):
    """
    Processes the word-syllable-phone mapping DataFrame to extract poly-syllabic word-level data.

    :param word_syllable_phone_mapping_dataframe: DataFrame containing word, syllable, and phone mappings.
    :return: Processed DataFrame containing poly-syllabic words with necessary transformations.
    """
    # Grouping to get phone-level data
    word_mapping = (
        word_syllable_phone_mapping_dataframe
        .groupby(['file_name', 'Type', 'Language', 'Word', 'Start_word', 'End_word'])
        .agg({'Word_Label': 'unique', 'Syllable': 'unique'})
        .reset_index()
    )

    # Extract the first element from the 'Word_Label' list
    word_mapping['Word_Label'] = word_mapping['Word_Label'].apply(lambda x: x[0])

    # Compute the number of syllables in each word
    word_mapping['SylCount'] = word_mapping['Syllable'].apply(len)

    # Filter poly-syllabic words (words with at least one syllable)
    poly_syllabics_words_data = word_mapping[word_mapping['SylCount'] > 0].copy()

    # Remove duplicate entries based on specific columns
    poly_syllabics_words_level_data = poly_syllabics_words_data.drop_duplicates(
        subset=['file_name', 'Word', 'Start_word', 'End_word']
    )

    # Replace apostrophes in word names with underscores
    poly_syllabics_words_level_data['Word'] = poly_syllabics_words_level_data['Word'].str.replace("'", "_")

    # Sort the DataFrame for better structure
    poly_syllabics_words_level_data = poly_syllabics_words_level_data.sort_values(by=['file_name', 'Start_word'])

    return poly_syllabics_words_level_data


def create_syllable_level_database(word_syllable_phone_mapping_dataframe):
    """
    Processes the word-syllable-phone mapping DataFrame to extract poly-syllabic words
    with syllable-level and phone-level data.

    :param word_syllable_phone_mapping_dataframe: DataFrame containing word, syllable, and phone mappings.
    :return: Processed DataFrame containing syllable-level details of poly-syllabic words.
    """
    # Grouping to get phone-level data
    word_syllable_mapping = (
        word_syllable_phone_mapping_dataframe
        .groupby(['file_name', 'Type', 'Language', 'Word', 'Start_word', 'End_word',
                  'Syllable', 'Start_syl', 'End_syl'])
        .agg({'Phones': 'unique', 'Word_Label': 'unique', 'SylStress': 'unique'})
        .reset_index()
    )

    # Extract the first element from the lists
    word_syllable_mapping['Word_Label'] = word_syllable_mapping['Word_Label'].apply(lambda x: x[0])
    word_syllable_mapping['SylStress'] = word_syllable_mapping['SylStress'].apply(lambda x: x[0])

    # Grouping to get word-level syllable data
    syllabics_words_data = (
        word_syllable_phone_mapping_dataframe
        .groupby(['file_name', 'Type', 'Language', 'Word', 'Start_word', 'End_word'])
        .agg({'Syllable': 'unique'})
        .reset_index()
    )

    # Compute the number of syllables in each word
    syllabics_words_data['SylCount'] = syllabics_words_data['Syllable'].apply(len)

    # Filter poly-syllabic words (words with at least one syllable)
    poly_syllabics_words_data = syllabics_words_data[syllabics_words_data['SylCount'] > 0].copy()

    # Merging poly-syllabic words with phone-level data
    poly_syllabics_words_syllable_level_data = poly_syllabics_words_data.merge(
        word_syllable_mapping,
        on=['file_name', 'Word', 'Start_word', 'End_word', 'Type'],
        how='left'
    )

    # Dropping duplicates
    poly_syllabics_words_syllable_level_data = poly_syllabics_words_syllable_level_data.drop_duplicates(
        subset=['file_name', 'Word', 'Start_word', 'End_word', 'Syllable_y']
    )

    # Renaming columns and keeping necessary ones
    poly_syllabics_words_syllable_level_data = (
        poly_syllabics_words_syllable_level_data
        .rename(columns={'Language_x': 'Language', 'Syllable_y': 'Syllable'})
        [['file_name', 'Type', 'Language', 'Word', 'Start_word', 'End_word',
          'Syllable', 'Start_syl', 'End_syl', 'SylStress', 'Phones', 'Word_Label']]
    )

    # Replace apostrophes in words
    poly_syllabics_words_syllable_level_data['Word'] = poly_syllabics_words_syllable_level_data['Word'].str.replace("'", "_")

    # Sorting for better structure
    poly_syllabics_words_syllable_level_data = poly_syllabics_words_syllable_level_data.sort_values(by=['file_name', 'Start_word'])

    return poly_syllabics_words_syllable_level_data
