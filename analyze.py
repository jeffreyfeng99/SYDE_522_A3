import os
import pandas as pd
import numpy as np

def compare(true_labels, predicted_labels):
    combined_df = pd.read_csv(true_labels)
    predicted_df = pd.read_csv(predicted_labels)

    combined_df['label'] = combined_df['dir'].map(predicted_df.set_index('id')['label'])

    true_labels = np.array(combined_df['label2'].to_list())
    pred_labels = np.array(combined_df['label'].to_list())

    return np.sum(true_labels == pred_labels) / len(true_labels)


dataset_root = './data'
output_root = './submission/04162022_efficientnet_b4_run6'

train_label_list = os.path.join(dataset_root, 'train_labels.csv')
test_label_list = os.path.join(dataset_root, 'dummy_test_labels.csv')

train_accs = []
train_paths = []
test_accs = []
test_paths = []

for root, dir, files in os.walk(output_root):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)

            if 'train' in file:
                acc = compare(train_label_list, file_path)

                train_accs.append(acc)
                train_paths.append(file_path)
            elif 'test' in file:
                acc = compare(test_label_list, file_path)

                test_accs.append(acc)
                test_paths.append(file_path)

train_df = pd.DataFrame({'acc': train_accs, 'file': train_paths})
sorted_train_df = train_df.sort_values(by=['acc'])
test_df = pd.DataFrame({'acc': test_accs, 'file': test_paths})
sorted_test_df = test_df.sort_values(by=['acc'])

pd.set_option('display.max_rows', train_df.shape[0]+1)
pd.set_option('display.max_colwidth', None)

print(train_df.to_string())
print(sorted_train_df.to_string())
print(test_df.to_string())
print(sorted_test_df.to_string())