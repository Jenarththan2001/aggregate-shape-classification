"""
ROC and Precision-Recall Curves — all 4 pretrained models.

Reads the per-image prediction CSVs saved by each model's training notebook
and plots one-vs-rest ROC and PR curves for all 6 milling classes.
Run from the directory that contains the *_May11_Results/ folders.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize

# ── Config ─────────────────────────────────────────────────────────────────────
# Integer labels used during training (must match label_binarize classes order)
CLASS_LABELS = [0, 100, 500, 1000, 1500, 2000]
CLASS_NAMES  = ['0 rev', '100 rev', '500 rev', '1000 rev', '1500 rev', '2000 rev']
# Column names for softmax probabilities in the prediction CSVs
PROB_COLS    = ['Prob_0', 'Prob_100', 'Prob_500', 'Prob_1000', 'Prob_1500', 'Prob_2000']
N_CLASSES    = 6

# Prediction CSV paths — relative to the directory where this script is run
MODELS = {
    'ResNet50':       r'ResNet50_May11_Results\predictions_resnet50.csv',
    'EfficientNetB0': r'EfficientNetB0_May11_Results\predictions_efficientnetb0.csv',
    'MobileNetV2':    r'MobileNetV2_May11_Results\predictions_mobilenetv2.csv',
    'Xception':       r'Xception_May11_Results\predictions_xception.csv',
}

# Distinct colours for the 6 classes — chosen to be distinguishable in print
COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']


# ── Helper ─────────────────────────────────────────────────────────────────────
def load_and_binarize(csv_path):
    """Load prediction CSV and return (y_bin, y_score) for sklearn curve functions.

    label_binarize converts multi-class labels to one-hot so each class can be
    treated as an independent binary problem (one-vs-rest strategy).
    """
    df = pd.read_csv(csv_path)
    y_true  = df['True Label'].values
    y_score = df[PROB_COLS].values
    # classes= must match CLASS_LABELS order so column i of y_bin corresponds to
    # column i of y_score (both indexed by CLASS_LABELS position)
    y_bin   = label_binarize(y_true, classes=CLASS_LABELS)
    return y_bin, y_score


# ── Figure 1: ROC curves (one subplot per model) ──────────────────────────────
# Macro-average ROC: interpolate each class curve onto a common FPR grid then
# average — gives a single summary curve without micro-averaging bias.
fig_roc, axes_roc = plt.subplots(1, 4, figsize=(22, 5))
fig_roc.suptitle('ROC Curves – One-vs-Rest (per model)', fontsize=14, fontweight='bold')

for ax, (model_name, csv_path) in zip(axes_roc, MODELS.items()):
    y_bin, y_score = load_and_binarize(csv_path)
    macro_tpr, macro_fpr_base = [], np.linspace(0, 1, 300)

    for i, (cls, cname) in enumerate(zip(CLASS_LABELS, CLASS_NAMES)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[i], lw=1.8,
                label=f'{cname} (AUC={roc_auc:.3f})')
        # Interpolate to common FPR grid before averaging
        macro_tpr.append(np.interp(macro_fpr_base, fpr, tpr))

    macro_tpr_mean = np.mean(macro_tpr, axis=0)
    macro_auc = auc(macro_fpr_base, macro_tpr_mean)
    ax.plot(macro_fpr_base, macro_tpr_mean, 'k--', lw=2,
            label=f'Macro avg (AUC={macro_auc:.3f})')
    # Diagonal reference line: random classifier performance
    ax.plot([0, 1], [0, 1], 'gray', lw=1, linestyle=':')

    ax.set_title(model_name, fontsize=11, fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=6.5, loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('ROC_Curves_All_Models.png', dpi=300, bbox_inches='tight')
print('Saved: ROC_Curves_All_Models.png')


# ── Figure 2: PR curves (one subplot per model) ───────────────────────────────
# PR curves are preferred over ROC when classes are imbalanced; they directly
# show the tradeoff between precision and recall without inflating AUC via
# the large number of true negatives in a multi-class setting.
# Macro-average: interpolate precision onto a common recall grid then average.
fig_pr, axes_pr = plt.subplots(1, 4, figsize=(22, 5))
fig_pr.suptitle('Precision-Recall Curves – One-vs-Rest (per model)', fontsize=14, fontweight='bold')

for ax, (model_name, csv_path) in zip(axes_pr, MODELS.items()):
    y_bin, y_score = load_and_binarize(csv_path)
    macro_prec, recall_base = [], np.linspace(0, 1, 300)

    for i, (cls, cname) in enumerate(zip(CLASS_LABELS, CLASS_NAMES)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, color=COLORS[i], lw=1.8,
                label=f'{cname} (AUC={pr_auc:.3f})')
        # precision_recall_curve returns decreasing recall — reverse before interp
        macro_prec.append(np.interp(recall_base, rec[::-1], prec[::-1]))

    macro_prec_mean = np.mean(macro_prec, axis=0)
    macro_pr_auc = auc(recall_base, macro_prec_mean)
    ax.plot(recall_base, macro_prec_mean, 'k--', lw=2,
            label=f'Macro avg (AUC={macro_pr_auc:.3f})')

    ax.set_title(model_name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(fontsize=6.5, loc='lower left')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('PR_Curves_All_Models.png', dpi=300, bbox_inches='tight')
print('Saved: PR_Curves_All_Models.png')

plt.show()
