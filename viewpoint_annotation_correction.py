from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

ANNOTATIONS_CSV = Path("metadata/seals_annotations.csv")
EMBEDDINGS_NPZ  = Path("viewpoint_embeddings/seal_laplacian_embedding.npz")

K_NEIGHBORS          = 11
MAX_CV_SPLITS        = 10
SUSPICION_PERCENTILE = 80   # flag top 20% by margin among disagreements

emb_data = np.load(EMBEDDINGS_NPZ, allow_pickle=False)
embedding = emb_data['embedding']
annotation_uuids = emb_data['annotation_uuids'].astype(str)
if embedding.ndim != 2 or embedding.shape[1] < 2:
    raise ValueError(f'Expected embedding with shape [N, >=2], got {embedding.shape}')
if len(annotation_uuids) != len(embedding):
    raise ValueError('annotation_uuids and embedding must have the same length')

emb_df = pd.DataFrame({
    'annotation_uuid': annotation_uuids,
    **{f'embedding_{i}': embedding[:, i] for i in range(embedding.shape[1])}
})

annotations = pd.read_csv(ANNOTATIONS_CSV)
annotations['annotation_uuid'] = annotations['annotation_uuid'].astype(str)

df = emb_df.merge(annotations, on='annotation_uuid', how='left', validate='m:1')

missing_uuid = df['annotation_uuid'].isna().sum()
missing_label = df['viewpoint'].isna().sum()
if missing_uuid or missing_label:
    print(f'Dropping {missing_label} rows with no viewpoint label '
          f'({missing_uuid} embeddings had no matching annotation)')
df = df.dropna(subset=['viewpoint']).reset_index(drop=True)

emb_cols = [c for c in df.columns if c.startswith('embedding_')]
X = df[emb_cols].values
le = LabelEncoder()
y = le.fit_transform(df['viewpoint'])

min_class_count = df['viewpoint'].value_counts().min()
n_splits = max(2, min(MAX_CV_SPLITS, min_class_count))
if n_splits < MAX_CV_SPLITS:
    print(f'Rarest class has {min_class_count} samples -> using n_splits={n_splits}')
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

knn = KNeighborsClassifier(K_NEIGHBORS, metric='euclidean', weights='distance')

pred_proba = cross_val_predict(knn, X, y, cv=cv, method='predict_proba')
pred_labels = pred_proba.argmax(axis=1)

print(classification_report(y, pred_labels, target_names=le.classes_, zero_division=0))


row_idx = np.arange(len(y))
true_label_proba = pred_proba[row_idx, y]
pred_label_proba = pred_proba[row_idx, pred_labels]
margin = pred_label_proba - true_label_proba   # >0 means neighbors prefer the alt label

df['pred_viewpoint']  = le.inverse_transform(pred_labels)
df['suspicion_score']  = margin



disagreement = pred_labels != y
print(f'Disagreements: {disagreement.sum()} / {len(y)}')

flagged = df[disagreement].copy()
threshold = np.percentile(flagged['suspicion_score'], SUSPICION_PERCENTILE) if len(flagged) else 0.0
flagged = flagged[flagged['suspicion_score'] >= threshold].sort_values('suspicion_score', ascending=False)
print(f'High-confidence anomalies (score >= {threshold:.3f}, top {100-SUSPICION_PERCENTILE}%): {len(flagged)}')

flagged.to_csv('anomalies.csv', index=False)

with open('labels.txt', 'w') as f:
    for label in le.classes_:
        f.write(f"{label}\n")