import torch
import numpy as np
import warnings

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve,\
    average_precision_score, f1_score, precision_score, recall_score,\
        mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,\
            r2_score, d2_pinball_score, top_k_accuracy_score

from .basemetric import BaseMetric

warnings.filterwarnings('ignore')



class Acc1(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach(), true.detach()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).cpu().float()
        answers = torch.cat(self.answers).cpu().float().numpy()

        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(-1).numpy()
        else:
            scores = scores.sigmoid().numpy()
            if self._use_youdenj: # binary - use Youden's J to determine a label
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores >= cutoff, 1, 0)
        return accuracy_score(answers, labels)

class Acc5(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach(), true.detach()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).cpu().float().softmax(-1).numpy()
        answers = torch.cat(self.answers).cpu().float().numpy()
        num_classes = scores.shape[-1]
        return top_k_accuracy_score(answers, scores, k=5, labels=np.arange(num_classes))

class Auroc(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach(), true.detach()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).cpu().float().softmax(-1).numpy()
        answers = torch.cat(self.answers).cpu().float().numpy()
        num_classes = scores.shape[-1]
        if num_classes == 2:
            # sklearn's binary AUROC requires 1D positive-class probabilities
            return roc_auc_score(answers, scores[:, 1])
        return roc_auc_score(answers, scores, average='weighted', multi_class='ovr', labels=np.arange(num_classes))

class Auprc(BaseMetric): # only for binary classification
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach(), true.detach()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).cpu().float().sigmoid().numpy()
        answers = torch.cat(self.answers).cpu().float().numpy()
        return average_precision_score(answers, scores, average='weighted')

class Youdenj(BaseMetric):  # only for binary classification
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach(), true.detach()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).cpu().float().sigmoid().numpy()
        answers = torch.cat(self.answers).cpu().float().numpy()
        fpr, tpr, thresholds = roc_curve(answers, scores)
        return thresholds[np.argmax(tpr - fpr)]

class F1(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach(), true.detach()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).cpu().float()
        answers = torch.cat(self.answers).cpu().float().numpy()

        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(-1).numpy()
        else:
            scores = scores.sigmoid().numpy()
            if self._use_youdenj: # binary - use Youden's J to determine a label
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores >= cutoff, 1, 0)
        return f1_score(answers, labels, average='weighted', zero_division=0)

class Precision(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach(), true.detach()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).cpu().float()
        answers = torch.cat(self.answers).cpu().float().numpy()

        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(-1).numpy()
        else:
            scores = scores.sigmoid().numpy()
            if self._use_youdenj: # binary - use Youden's J to determine a label
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores >= cutoff, 1, 0)
        return precision_score(answers, labels, average='weighted', zero_division=0)

class Recall(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach(), true.detach()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).cpu().float()
        answers = torch.cat(self.answers).cpu().float().numpy()

        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(-1).numpy()
        else:
            scores = scores.sigmoid().numpy()
            if self._use_youdenj: # binary - use Youden's J to determine a label
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores >= cutoff, 1, 0)
        return recall_score(answers, labels, average='weighted', zero_division=0)

class Seqacc(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        num_classes = pred.size(-1)
        p, t = pred.detach(), true.detach()
        self.scores.append(p.view(-1, num_classes))
        self.answers.append(t.view(-1))

    def summarize(self):
        labels = torch.cat(self.scores).cpu().float().argmax(-1).numpy()
        answers = torch.cat(self.answers).cpu().float().numpy()

        # ignore special tokens
        labels = labels[answers != -1]
        answers = answers[answers != -1]
        return np.nan_to_num(accuracy_score(answers, labels))


class Mse(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach(), true.detach()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).cpu().float().numpy()
        answers = torch.cat(self.answers).cpu().float().numpy()
        return mean_squared_error(answers, scores)

class Rmse(Mse):
    def __init__(self):
        super(Rmse, self).__init__()

    def summarize(self):
        scores = torch.cat(self.scores).cpu().float().numpy()
        answers = torch.cat(self.answers).cpu().float().numpy()
        return mean_squared_error(answers, scores, squared=False)

class Mae(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach(), true.detach()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).cpu().float().numpy()
        answers = torch.cat(self.answers).cpu().float().numpy()
        return mean_absolute_error(answers, scores)

class Mape(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach(), true.detach()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).cpu().float().numpy()
        answers = torch.cat(self.answers).cpu().float().numpy()
        return mean_absolute_percentage_error(answers, scores)

class R2(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach(), true.detach()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self, *args):
        scores = torch.cat(self.scores).cpu().float().numpy()
        answers = torch.cat(self.answers).cpu().float().numpy()
        return r2_score(answers, scores)

class D2(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach(), true.detach()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self, *args):
        scores = torch.cat(self.scores).cpu().float().numpy()
        answers = torch.cat(self.answers).cpu().float().numpy()
        return d2_pinball_score(answers, scores)


class Mlacc(BaseMetric):
    """Multi-label accuracy: threshold predictions at 0.5, compute mean per-label accuracy."""
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach(), true.detach()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).cpu().float().sigmoid().numpy()
        answers = torch.cat(self.answers).cpu().float().numpy()
        preds = (scores >= 0.5).astype(float)
        # per-label accuracy, then average
        return np.mean((preds == answers).mean(axis=0))


class Mlauroc(BaseMetric):
    """Multi-label AUROC: compute per-label AUROC then macro-average."""
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach(), true.detach()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).cpu().float().sigmoid().numpy()
        answers = torch.cat(self.answers).cpu().float().numpy()
        # per-label AUROC, skip labels with single class
        aurocs = []
        for i in range(answers.shape[1]):
            if len(np.unique(answers[:, i])) > 1:
                aurocs.append(roc_auc_score(answers[:, i], scores[:, i]))
        return np.mean(aurocs) if aurocs else 0.0
