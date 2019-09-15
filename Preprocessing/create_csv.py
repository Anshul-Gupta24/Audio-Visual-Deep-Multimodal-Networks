import numpy as np
import os
import pickle
import pandas as pd


image_train_path = '/home/data1/kiranp/extracted_features/imagenet_xception/data_kmeans_train.pkl'
image_val_path = '/home/data1/kiranp/extracted_features/imagenet_xception/data_kmeans_val.pkl'
# with open(image_train_path, 'rb') as file:
#     image_train_feats = pickle.load(file)
# with open(image_val_path, 'rb') as file:
#     image_val_feats = pickle.load(file)

print('loaded...')

# with open('/home/anshulg/WordNet/get_imagenet/functions/spk_ibm_train.pkl', 'rb') as fp:
#     spk_ibm_train = pickle.load(fp)

# with open('/home/anshulg/WordNet/get_imagenet/functions/spk_google_train.pkl', 'rb') as fp:
#     spk_google_train = pickle.load(fp)

# with open('/home/anshulg/WordNet/get_imagenet/functions/spk_microsoft_train.pkl', 'rb') as fp:
#     spk_ms_train = pickle.load(fp)

# with open('/home/anshulg/WordNet/get_imagenet/functions/spk_ibm_val.pkl', 'rb') as fp:
#     spk_ibm_val = pickle.load(fp)

# with open('/home/anshulg/WordNet/get_imagenet/functions/spk_google_val.pkl', 'rb') as fp:
#     spk_google_val = pickle.load(fp)

# with open('/home/anshulg/WordNet/get_imagenet/functions/spk_microsoft_val.pkl', 'rb') as fp:
#     spk_ms_val = pickle.load(fp)

# with open('/home/anshulg/WordNet/get_imagenet/functions/spk_ibm_test.pkl', 'rb') as fp:
#     spk_ibm_test = pickle.load(fp)

# with open('/home/anshulg/WordNet/get_imagenet/functions/spk_google_test.pkl', 'rb') as fp:
#     spk_google_test = pickle.load(fp)

# with open('/home/anshulg/WordNet/get_imagenet/functions/spk_microsoft_test.pkl', 'rb') as fp:
#     spk_ms_test = pickle.load(fp)


spk_ibm_train += 1
spk_ms_train += 1 + 1
spk_train = np.concatenate((spk_google_train, spk_ibm_train, spk_ms_train))
spk_train = spk_train.astype(int)
spk_ibm_val += 1
spk_ms_val += 1 + 1
spk_val = np.concatenate((spk_google_val, spk_ibm_val, spk_ms_val))
spk_val = spk_val.astype(int)
# spk_ibm_test += 14
# spk_ms_test += 14 + 3
# spk_test = np.concatenate((spk_google_test, spk_ibm_test, spk_ms_test))



# spk_train_noisy = []
# for s in spk_train:
#         s_new = s*7
#         for x in range(s_new, s_new+7):
#             spk_train_noisy.append(x)

# spk_val_noisy = []
# for s in spk_val:
#         s_new = s*7
#         for x in range(s_new, s_new+7):
#             spk_val_noisy.append(x)

# spk_test_noisy = []
# for s in spk_test:
#         s_new = s*7
#         for x in range(s_new, s_new+7):
#             spk_test_noisy.append(x)

df = pd.read_csv('/home/anshulg/WordNet/get_imagenet/functions/imagenet-katakana.csv')
katas = df['kata'].values
katakana_eng = df['eng'].values[katas=='katakana']
hiragana_eng = df['eng'].values[katas=='hiragana']
labels = np.concatenate((katakana_eng, hiragana_eng))
np.random.shuffle(labels)



# Change
spk = spk_train
start = 0
end = 25

train_set = []
for label in labels:
    image_label = label
    positive_speech_label = label

    # Image anchor
    grounding = 0
    for image_idx in range(start, end):
        for positive_speech_idx in spk:
            negative_speech_labels = [item for item in labels if item != label]
            negative_speech_label = negative_speech_labels[np.random.randint(0, len(negative_speech_labels))]
            negative_speech_idx = np.random.choice(spk)
            train_set.append([grounding, image_label, image_idx, positive_speech_label, positive_speech_idx, negative_speech_label, negative_speech_idx])

    # Audio anchor
    grounding = 1
    for positive_speech_idx in spk:    
        for image_idx in range(start, end):
            negative_image_labels = [item for item in labels if item != label]
            negative_image_label = negative_image_labels[np.random.randint(0, len(negative_image_labels))]
            negative_image_idx = np.random.randint(start, end)
            train_set.append([grounding, image_label, image_idx, positive_speech_label, positive_speech_idx, negative_image_label, negative_image_idx])


df_train = pd.DataFrame(train_set)
print(df_train)


start = 0
end = 25
spk = spk_val

# start = int(len(image_val_feats['abacus']) / 2)
# end = len(image_val_feats['abacus'])


val_set = []
for label in labels:
    image_label = label
    positive_speech_label = label   

    # Image anchor
    grounding = 0
    for image_idx in range(start, end):
        # Positive speech
        for positive_speech_idx in spk:

            negative_speech_labels = [item for item in labels if item != label]
            negative_speech_label = negative_speech_labels[np.random.randint(0, len(negative_speech_labels))]
            negative_speech_idx = np.random.choice(spk)
            for i in range(3):      # repeat thrice to make same size as train (needed because same batch size)
                val_set.append([grounding, image_label, image_idx, positive_speech_label, positive_speech_idx, negative_speech_label, negative_speech_idx])

    # Audio anchor
    grounding = 1
    for positive_speech_idx in spk:    
        for image_idx in range(start, end):

            negative_image_labels = [item for item in labels if item != label]
            negative_image_label = negative_image_labels[np.random.randint(0, len(negative_image_labels))]
            negative_image_idx = np.random.randint(start, end)
            for i in range(3):      # repeat thrice to make same size as train (needed because same batch size)
                val_set.append([grounding, image_label, image_idx, positive_speech_label, positive_speech_idx, negative_image_label, negative_image_idx])


df_val = pd.DataFrame(val_set)
print(df_val)

df_train.to_csv('/home/anshulg/WordNet/get_imagenet/train_data_jap.csv')
df_val.to_csv('/home/anshulg/WordNet/get_imagenet/val_data_jap.csv')
