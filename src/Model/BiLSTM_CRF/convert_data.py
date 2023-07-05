import pandas as pd
import json


def convert_csv_to_text(input_file, output_file):
    df = pd.read_csv(input_file)

    with open(output_file, 'w') as txt_file:
        # Group by sentence_id and concatenate the Word and Tag columns
        grouped_data_1 = df.groupby('sentence_id').apply(lambda group: ' '.join(f"{w}" for w in group['Word']))
        grouped_data_2 = df.groupby('sentence_id').apply(lambda group: ', '.join(f'"{t}"' for t in group['Tag']))
        for sentence_1, sentence_2 in zip(grouped_data_1, grouped_data_2):
          txt_file.write(sentence_1 + "\t" + "[" + sentence_2 + "]" + '\n')


# Usage example
input_file = "" # input sequence labeling based version
output_file = 'sample_corpus/dataset.txt'
convert_csv_to_text(input_file, output_file)


data = pd.read_csv(input_file)

# Get unique words
unique_words = data['Word'].unique().tolist()

# Save to JSON file
with open('sample_corpus/vocab.json', 'w', encoding='utf-8') as f:
    json.dump(unique_words, f, ensure_ascii=False)

tags = ["B-T", "O", "I-T"]

with open('sample_corpus/tags.json', 'w') as f:
    json.dump(tags, f)
