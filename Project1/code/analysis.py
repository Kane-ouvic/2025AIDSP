import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 載入標註資料
annotations_file = './DEAM_Annotations/annotations/annotations_averaged_per_song/song_level/static_annotations_averaged_songs_1_2000.csv'
annotations = pd.read_csv(annotations_file)



# 檢視前幾筆資料
print(annotations.head())
print(annotations.columns.tolist())

annotations.columns = annotations.columns.str.strip()
# 提取 Valence 和 Arousal 的平均值
valence = annotations['valence_mean']
arousal = annotations['arousal_mean']


# 繪製散佈圖
plt.figure(figsize=(8, 6))
plt.scatter(valence, arousal, alpha=0.5)
plt.title('Valence vs. Arousal Scatter Plot')
plt.xlabel('Valence (正負情緒)')
plt.ylabel('Arousal (激昂度)')
plt.grid(True)
plt.show()



# 定義情緒類別
def classify_emotion(valence, arousal):
    if valence >= 5 and arousal >= 5:
        return "Happy"
    elif valence >= 5 and arousal < 5:
        return "Peaceful"
    elif valence < 5 and arousal >= 5:
        return "Tense"
    else:
        return "Sad"

# 為每首歌曲分配情緒標籤
annotations['emotion'] = annotations.apply(lambda row: classify_emotion(row['valence_mean'], row['arousal_mean']), axis=1)

# 繪製帶有情緒標籤的散佈圖
plt.figure(figsize=(8, 6))
colors = {'Happy': 'yellow', 'Peaceful': 'green', 'Tense': 'red', 'Sad': 'blue'}
for emotion, color in colors.items():
    subset = annotations[annotations['emotion'] == emotion]
    plt.scatter(subset['valence_mean'], subset['arousal_mean'], label=emotion, color=color, alpha=0.5)

plt.title('Valence vs. Arousal Scatter Plot with Emotion Labels')
plt.xlabel('Valence ')
plt.ylabel('Arousal ')
plt.legend()
plt.grid(True)
plt.savefig('scatter.png')
plt.show()