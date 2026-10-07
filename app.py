import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path
from ast import literal_eval
import io
from datetime import datetime as dt, timezone
import json



ANNOTATIONS_CSV   = "metadata/seals_annotations.csv"
ANOMALIES_CSV     = "anomalies.csv"
LABELS_FILE       = "labels.txt"
EMBEDDINGS_NPZ    = "C:\\Users\\Quoc An\\Downloads\\RA\\AnotationCorrection\\viewpoint_embeddings\\seal_laplacian_embedding.npz"
DATASET_ROOT      = Path("miewid__norppa")
CORRECTED_CSV     = "corrected_annotations.csv"

def timestamp():
    return dt.now().astimezone().strftime("%d-%m-%Y %H:%M:%S GMT%z")

def _parse_xywh_bbox(raw_bbox):
    bbox = literal_eval(str(raw_bbox))
    if len(bbox) != 4:
        raise ValueError(f'Expected bbox [x, y, w, h], got {bbox!r}')
    x, y, w, h = [float(v) for v in bbox]   
    return x, y, x + w, y + h

def _resolve_image_path(row):
    image_path = Path(str(row['path']))
    if image_path.is_absolute():
        return image_path
    return DATASET_ROOT / image_path


@st.cache_data
def load_full_df():
    emb_data         = np.load(EMBEDDINGS_NPZ, allow_pickle=False)
    embedding        = emb_data['embedding']
    annotation_uuids = emb_data['annotation_uuids'].astype(str)

    emb_df = pd.DataFrame({
        'annotation_uuid': annotation_uuids,
        'embedding_0':     embedding[:, 0],
        'embedding_1':     embedding[:, 1],
    })

    annotations = pd.read_csv(ANNOTATIONS_CSV)
    annotations['annotation_uuid'] = annotations['annotation_uuid'].astype(str)

    return emb_df.merge(annotations, on='annotation_uuid', how='left', validate='m:1')

@st.cache_data
def load_anomalies():
    if not Path(ANOMALIES_CSV).exists():
        return pd.DataFrame()
    df = pd.read_csv(ANOMALIES_CSV)
    df['annotation_uuid'] = df['annotation_uuid'].astype(str)
    return df

@st.cache_data
def load_labels():
    if not Path(LABELS_FILE).exists():
        return []
    return Path(LABELS_FILE).read_text().splitlines()

@st.cache_data
def load_corrected():
    path = Path(CORRECTED_CSV)
    if not path.exists():
        return pd.DataFrame(columns=['annotation_uuid', 'original_viewpoint', 'pred_viewpoint', 'corrected_viewpoint', 'timestamp'])
    try:
        df = pd.read_csv(path)
        if df.empty or df.columns.tolist() == []:
            raise pd.errors.EmptyDataError
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.NaT
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=['annotation_uuid', 'original_viewpoint', 'pred_viewpoint', 'corrected_viewpoint', 'timestamp'])
    
def get_proba_columns(row, labels):
    return {label: row.get(f'proba_{label}', 0.0) for label in labels}

def top_k_suggestions(proba_dict, k=3):
    return sorted(proba_dict.items(), key=lambda kv: kv[1], reverse=True)[:k]

def plot_embedding_for_annotation(df, annotation_uuid, label_col='viewpoint'):
    row   = df[df['annotation_uuid'] == str(annotation_uuid)].iloc[0]
    x_col, y_col = 'embedding_0', 'embedding_1'

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                             gridspec_kw={'width_ratios': [1.1, 1]})

    ax = axes[0]
    if label_col in df.columns:
        labels_col = df[label_col].fillna('missing').astype(str)
        for label in sorted(labels_col.unique()):
            m = labels_col == label
            ax.scatter(df.loc[m, x_col], df.loc[m, y_col],
                       s=8, alpha=0.35, label=f'{label} ({m.sum()})')
        ax.legend(markerscale=2, fontsize=8, loc='best')
    else:
        ax.scatter(df[x_col], df[y_col], s=8, alpha=0.35)

    ax.scatter([row[x_col]], [row[y_col]], s=220, marker='*',
               c='yellow', edgecolors='black', linewidths=1.5, zorder=10)
    ax.set_xlabel('embedding dimension 0')
    ax.set_ylabel('embedding dimension 1')
    ax.set_title(f'Embedding scatter\n{annotation_uuid}')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.25)

    image_path = _resolve_image_path(row)
    image = Image.open(image_path).convert('RGB')
    draw  = ImageDraw.Draw(image)
    x1, y1, x2, y2 = _parse_xywh_bbox(row['bbox'])
    draw.rectangle([x1, y1, x2, y2], outline='yellow', width=4)

    ax = axes[1]
    ax.imshow(image)
    ax.set_title(f"{row.get('viewpoint', 'unknown')} | {Path(str(row['path'])).name}")
    ax.axis('off')

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf, row


def save_correction(annotation_uuid, original_viewpoint, pred_viewpoint, corrected_viewpoint, soft_viewpoint=None):
    corrected = load_corrected()
    corrected = corrected[corrected['annotation_uuid'] != annotation_uuid]
    new_row = pd.DataFrame([{
        'annotation_uuid':      annotation_uuid,
        'original_viewpoint':   original_viewpoint,
        'pred_viewpoint':       pred_viewpoint,
        'corrected_viewpoint':  corrected_viewpoint,
        'corrected_viewpoint_soft': json.dumps(soft_viewpoint) if soft_viewpoint else '',
        'timestamp':            timestamp(),
    }])
    corrected = pd.concat([corrected, new_row], ignore_index=True)
    corrected.to_csv(CORRECTED_CSV, index=False)
    load_corrected.clear()

def get_next_unreviewed(uuid_list: list, corrected_uuids: set, current_uuid: str) -> str:
    ## Return the next UUID after current that hasn't been corrected yet, or current if none
    try:
        start = uuid_list.index(current_uuid) + 1
    except ValueError:
        start = 0
    for uuid in uuid_list[start:] + uuid_list[:start]:
        if uuid not in corrected_uuids:
            return uuid
    return current_uuid  

st.set_page_config(layout="wide")
st.title("Anomaly Review & Annotation")

df           = load_full_df()
anomalies_df = load_anomalies()
labels       = load_labels()
corrected_df = load_corrected()

if anomalies_df.empty:
    st.warning("No anomalies found. Please run the backend script first.")
    st.stop()

uuid_list = anomalies_df['annotation_uuid'].tolist()
if 'current_uuid' not in st.session_state:
    corrected_set = set(corrected_df['annotation_uuid'].tolist())
    # Start on the first unreviewed anomaly
    st.session_state.current_uuid = next(
        (u for u in uuid_list if u not in corrected_set), uuid_list[0]
    )
# 

# Progress bar
n_total    = len(anomalies_df)
n_reviewed = corrected_df['annotation_uuid'].isin(anomalies_df['annotation_uuid']).sum()
st.progress(n_reviewed / n_total, text=f"Reviewed {n_reviewed} / {n_total} anomalies")

left_col, right_col = st.columns([2, 1])

with right_col:
    st.subheader("Controls")

    corrected_set = set(corrected_df['annotation_uuid'].tolist())

    def uuid_label(uuid: str) -> str:
        prefix = '✓ ' if uuid in corrected_set else '○ '
        return f"{prefix}{uuid}"

    selected_uuid = st.selectbox(
        "Select Anomaly by Annotation UUID",
        options=uuid_list,
        format_func=uuid_label,
        index=uuid_list.index(st.session_state.current_uuid),
    )
    if selected_uuid != st.session_state.current_uuid:
        st.session_state.current_uuid = selected_uuid
        st.rerun()
    selected_uuid = st.session_state.current_uuid

    anomaly_info  = anomalies_df[anomalies_df['annotation_uuid'] == selected_uuid].iloc[0]
    pred_label    = anomaly_info['pred_viewpoint']
    orig_label    = anomaly_info.get('viewpoint', 'unknown')

    st.info(f"**Original label:** {orig_label}")
    st.info(f"**KNN predicted:** {pred_label}")

    already = corrected_df[corrected_df['annotation_uuid'] == selected_uuid]
    if not already.empty:
        st.success(f"Already corrected → **{already.iloc[0]['corrected_viewpoint']}**")

    st.divider()
    st.subheader("Select the correct label")
    label_pairs = [labels[i:i+2] for i in range(0, len(labels), 2)]
    for pair in label_pairs:
        cols = st.columns(len(pair))
        for col, label in zip(cols, pair):
            with col:
                if st.button(label, key=f"btn_{label}", use_container_width=True):
                    save_correction(selected_uuid, orig_label, pred_label, label)
                    st.success(f"Saved '{label}' for {selected_uuid}!")
                    corrected_df   = load_corrected()          
                    corrected_set  = set(corrected_df['annotation_uuid'].tolist())
                    st.session_state.current_uuid = get_next_unreviewed(
                        uuid_list, corrected_set, selected_uuid
                    )
                    st.rerun()

    st.divider()
    st.subheader("Continuous Viewpoint (Soft Label)")
    left_right_ratio = st.slider("Left/Right Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.05, 
                                 help="0.0 is fully Left, 1.0 is fully Right")
    
    if st.button("Save Soft Label", type="secondary", use_container_width=True):
        soft_labels = {"left": round(1.0 - left_right_ratio, 2), "right": round(left_right_ratio, 2)}
        save_correction(selected_uuid, orig_label, pred_label, "continuous", soft_viewpoint=soft_labels)
        st.success(f"Saved soft labels {soft_labels}!")
        
        corrected_df = load_corrected()          
        corrected_set = set(corrected_df['annotation_uuid'].tolist())
        st.session_state.current_uuid = get_next_unreviewed(uuid_list, corrected_set, selected_uuid)
        st.rerun()

    st.divider()
    st.subheader("Export")

    corrected = load_corrected()
    st.download_button(
        label="Download corrected CSV",
        data=corrected.to_csv(index=False),
        file_name="corrected_annotations.csv",
        mime="text/csv",
        type="primary",
    )

with left_col:
    nav_left, nav_right, nav_spacer = st.columns([1, 1, 8])
    with nav_left:
        if st.button("Prev", key="btn_prev", use_container_width=True):
            cur_idx = uuid_list.index(st.session_state.current_uuid)
            st.session_state.current_uuid = uuid_list[(cur_idx - 1) % len(uuid_list)]
            st.rerun()
    with nav_right:
        if st.button("Next", key="btn_next", use_container_width=True):
            cur_idx = uuid_list.index(st.session_state.current_uuid)
            st.session_state.current_uuid = uuid_list[(cur_idx + 1) % len(uuid_list)]
            st.rerun()

    st.components.v1.html("""
        <script>
        const doc = window.parent.document;
        doc.addEventListener('keydown', function(e) {
            if (['ArrowLeft', 'ArrowRight'].includes(e.key)) {
                // Don't hijack arrow keys inside text inputs / selectboxes
                const tag = doc.activeElement.tagName.toLowerCase();
                if (['input', 'textarea', 'select'].includes(tag)) return;
                e.preventDefault();
                const btnLabel = e.key === 'ArrowRight' ? 'Next' : 'Prev';
                const btns = doc.querySelectorAll('button');
                for (const btn of btns) {
                    if (btn.innerText.trim() === btnLabel) {
                        btn.click();
                        break;
                    }
                }
            }
        });
        </script>
    """, height=0)

    st.subheader(f"Viewing: {selected_uuid}")
    try:
        buf, row = plot_embedding_for_annotation(df, selected_uuid)
        st.image(buf, use_container_width=True)

        with st.expander("Annotation details"):
            st.write(f"**Identity:** {row.get('identity', 'N/A')}")
            st.write(f"**Viewpoint:** {row.get('viewpoint', 'N/A')}")
            st.write(f"**Suspicion score:** {anomaly_info.get('suspicion_score', 'N/A'):.3f}")
            st.write(f"**Image path:** {row.get('path', 'N/A')}")
            st.write(f"**BBox:** {row.get('bbox', 'N/A')}")
            st.write(f"**embedding_0:** {row['embedding_0']:.6f}")
            st.write(f"**embedding_1:** {row['embedding_1']:.6f}")
    except FileNotFoundError:
        st.error(f"Image not found for UUID: {selected_uuid}")
    except Exception as e:
        st.error(f"Error rendering plot: {e}")