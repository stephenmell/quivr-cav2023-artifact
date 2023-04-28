import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from typing import *
import copy
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
from tqdm import tqdm
import random

# LSTM code taken from Argoverse forecasting (https://github.com/jagjeet-singh/argoverse-forecasting), which is "Clear BSD"-licensed

def parse_arguments():
    return argparse.Namespace(
        end_epoch=5000,
        joblib_batch_size=100,
        lr=0.001,
        model_path=None,
        normalize=False,
        obs_len=20,
        pred_len=30,
        test=False,
        test_batch_size=512,
        test_features='',
        train_batch_size=512,
        train_features='',
        traj_save_path=None,
        use_delta=False,
        use_map=False,
        use_social=False,
        val_batch_size=512,
        val_features=''
    )

class ModelUtils:
    """Utils for LSTM baselines."""
    def save_checkpoint(self, save_dir: str, state: Dict[str, Any]) -> None:
        """Save checkpoint file.
        
        Args:
            save_dir: Directory where model is to be saved
            state: State of the model
        """
        filename = "{}/LSTM_rollout{}.pth.tar".format(save_dir,
                                                    state["rollout_len"])
        torch.save(state, filename)

    def load_checkpoint(
            self,
            checkpoint_file: str,
            encoder: Any,
            decoder: Any,
            encoder_optimizer: Any,
            decoder_optimizer: Any,
    ) -> Tuple[int, int, float]:
        """Load the checkpoint.
        Args:
            checkpoint_file: Path to checkpoint file
            encoder: Encoder model
            decoder: Decoder model 
        Returns:
            epoch: epoch when the model was saved.
            rollout_len: horizon used
            best_loss: loss when the checkpoint was saved
        """
        if os.path.isfile(checkpoint_file):
            print("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            epoch = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"]
            rollout_len = checkpoint["rollout_len"]
            if use_cuda:
                encoder.module.load_state_dict(
                    checkpoint["encoder_state_dict"])
                decoder.module.load_state_dict(
                    checkpoint["decoder_state_dict"])
            else:
                encoder.load_state_dict(checkpoint["encoder_state_dict"])
                decoder.load_state_dict(checkpoint["decoder_state_dict"])
            encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
            decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer"])
            print(
                f"=> loaded checkpoint {checkpoint_file} (epoch: {epoch}, loss: {best_loss})"
            )
        else:
            print(f"=> no checkpoint found at {checkpoint_file}")

        return epoch, rollout_len, best_loss

    def my_collate_fn(self, batch: List[Any]) -> List[Any]:
        """Collate function for PyTorch DataLoader.
        Args:
            batch: Batch data
        Returns: 
            input, output and helpers in the format expected by DataLoader
        """
        _input, output, helpers = [], [], []

        for item in batch:
            _input.append(item[0])
            output.append(item[1])
            helpers.append(item[2])
        _input = torch.stack(_input)
        output = torch.stack(output)
        return [_input, output, helpers]

    def init_hidden(self, batch_size: int,
                    hidden_size: int) -> Tuple[Any, Any]:
        """Get initial hidden state for LSTM.
        Args:
            batch_size: Batch size
            hidden_size: Hidden size of LSTM
        Returns:
            Initial hidden states
        """
        return (
            torch.zeros(batch_size, hidden_size).to(device),
            torch.zeros(batch_size, hidden_size).to(device),
        )

class DecoderRNN(nn.Module):
    """Decoder Network."""
    def __init__(self, embedding_size=8, hidden_size=16, output_size=2):
        """Initialize the decoder network.
        Args:
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM
            output_size: number of features in the output
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(output_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """Run forward propagation.
        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            output: output from lstm
            hidden: final hidden state
        """
        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
        output = self.linear2(hidden[0])
        return output, hidden

args = parse_arguments()

def bb_center_point(bb):
    assert bb.shape[-1] == 4, bb.shape
    return torch.stack([
        (bb[..., 0] + bb[..., 2])/2,
        (bb[..., 1] + bb[..., 3])/2,
    ], -1)

model_utils = ModelUtils()

def pad_sequences(seq_list):
    padded = nn.utils.rnn.pad_sequence(seq_list, batch_first = True)
    device = padded.device

    lens = torch.tensor([len(t) for t in seq_list], dtype = torch.long, device = device)
    mask = torch.arange(padded.shape[1], device = device)[None, :] < lens[:, None]

    return padded, lens, mask

def trackset_to_tensor(trackset, w, h, max_length):
    n_obj = len(trackset.track_intervals)
    l = trackset.times.shape[0]

    ret = torch.zeros(l, 1+n_obj*3, device = device)
    ret[:, 0] = trackset.times / max_length
    for i in range(n_obj):
        offset = 1 + 3*i
        s, e = trackset.track_intervals[i]
        bb = trackset.get_overlap(i)
        pos = bb_center_point(bb[:, 0, :])
        pos[:, 0] /= w
        pos[:, 1] /= h

        ret[s:e, offset] = 1.0 # mask
        ret[s:e, (offset + 1):(offset + 3)] = pos
    
    return ret


def compute_tfpn(actual, pred):
    tp = actual * pred
    tn = (1 - actual) * (1 - pred)
    fp = (1 - actual) * pred
    fn = actual * (1 - pred)

    return tp.sum(), tn.sum(), fp.sum(), fn.sum()

def torch_load(path):
    return torch.load(path, map_location = device)






##### eval_lstm #####
seed = 0
np.random.seed(seed)
torch.manual_seed(seed + 10000)
random.seed(seed + 20000)

import sys

if "get_ipython" in globals():
    dataset_name = "maritime_surveillance"
    task_name = "a"
    device_name = "cuda:3"
else:
    assert len(sys.argv) == 4
    dataset_name = sys.argv[1]
    task_name = sys.argv[2]
    device_name = sys.argv[3]

device = torch.device(device_name)

dataset = torch_load(f"datasets/{dataset_name}/data.torch")
raw_tracks = torch_load(f"datasets_raw/{dataset_name}.torch")

task = torch_load(f"datasets/{dataset_name}/task_{task_name}/task.torch")

batches = [pad_sequences(raw_tracks)]    

latent_cache = {}
def compute_latent(tid):
    if tid in latent_cache:
        return latent_cache[tid]

    t = raw_tracks[tid]
    # Initialize encoder hidden state
    decoder_hidden = model_utils.init_hidden(
        1,
        decoder.hidden_size
    )

    t = t.unsqueeze(0)

    for di in range(t.shape[1]):
        decoder_output, decoder_hidden = decoder(t[:, di, :], decoder_hidden)
    
    # decoder_hidden is actually (hidden, cell)
    ret = decoder_hidden[0].squeeze(0).detach()
    latent_cache[tid] = ret
    return ret


def run_baseline_on_task(tids_pos, tids_unlabel, tids_label, tids_test):
    def logistify_dataset(tids, tids_pos):
        X = []
        y = []

        for tid in tids:
            X.append(latent_cache[tid])
            y.append(1 if tid in tids_pos else 0)

        return torch.stack(X, 0), torch.tensor(y, dtype = torch.long, device = device)

    def train_logistic(X, y):
        X = X.cpu().numpy()
        y = y.cpu().numpy()
        
        print("=== training logistic ===")
        # np.save("/tmp/dsfjsdf_X.npy", X)
        # np.save("/tmp/dsfjsdf_y.npy", y)
        clf = LogisticRegression(random_state=0, penalty="none").fit(X, y)
        # clf = LogisticRegression(random_state=0).fit(X, y)
        train_loss = sklearn.metrics.log_loss(y, clf.predict_proba(X))
        print(X.shape, train_loss)
        if not (train_loss.item() < 0.001):
            print("!!!!! WARNING train loss is high !!!!!")

        return clf

    tids_unlabel_list = list(tids_unlabel) # this is so that we have a stable order for later

    X_labeled, y_labeled = logistify_dataset(tids_label, tids_pos)
    X_unlabeled, y_unlabeled = logistify_dataset(tids_unlabel_list, tids_pos)
    #X_val, y_val = logistify_dataset(tids_val, tids_pos)
    X_test, y_test = logistify_dataset(tids_test, tids_pos)

    clf = train_logistic(X_labeled, y_labeled)

    #X_f1, y_f1 = X_labeled, y_labeled
    # cheat in favour of the baseline
    X_f1, y_f1 = X_test, y_test

    proba = clf.predict_proba(X_f1.cpu().numpy())[:, 1]
    thresh = np.linspace(0.0, 1.0, 100)
    f1s = np.empty(thresh.shape)
    for i, t in enumerate(thresh):
        f1s[i] = sklearn.metrics.f1_score(
            y_f1.cpu(),
            proba >= t,
        )

    opt_thresh = thresh[f1s.argmax()]

    test_f1 = sklearn.metrics.f1_score(
        y_test.cpu(),
        clf.predict_proba(X_test.cpu().numpy())[:, 1] >= opt_thresh,
    )
    test_tp_count, test_tn_count, test_fp_count, test_fn_count = compute_tfpn(
        y_test.cpu(),
        clf.predict_proba(X_test.cpu().numpy())[:, 1] >= opt_thresh,
    )
    precision = test_tp_count / (test_tp_count + test_fp_count)
    recall = test_tp_count / (test_tp_count + test_fn_count)
    test_f1_check = 2/(1/precision + 1/recall)
    assert abs(test_f1 - test_f1_check) < 0.001, (test_f1 - test_f1_check, test_f1, test_f1_check)

    # select most uncertain
    # unlabeled_delta_to_thresh = np.abs(
    #     opt_thresh - 
    #     clf.predict_proba(X_unlabeled.cpu().numpy())[:, 1]
    # )
    # opt_unlabeled_index = unlabeled_delta_to_thresh.argmin()
    # opt_value = unlabeled_delta_to_thresh.min()

    # select most positive
    unlabeled_scores = clf.predict_proba(X_unlabeled.cpu().numpy())[:, 1]
    opt_unlabeled_index = unlabeled_scores.argmax()
    opt_value = unlabeled_scores.max()

    opt_unlabeled = tids_unlabel_list[opt_unlabeled_index]

    return (test_tp_count, test_tn_count, test_fp_count, test_fn_count), opt_unlabeled, clf, opt_thresh, opt_value

def add_label_to_task(tids_label, tids_unlabel, tid):
    return (
        tids_label | frozenset({tid}),
        tids_unlabel - frozenset({tid}),
    )

decoder = torch_load(f"lstm_models/{dataset_name}.torch")

#with open(f"/tmp/foo/lstm_al_{dataset_name}_{task_name}.csv", "a") as output_file:
tids_train = frozenset(x.item() for x in dataset["train_indices"])
tids_test = frozenset(x.item() for x in dataset["test_indices"])
tids_pos = frozenset(x.item() for x in task["labels"].nonzero().squeeze())
tids_label_initial = frozenset(x.item() for x in task["labeled_initial_indices"])
tids_unlabel_initial = tids_train - tids_label_initial

tids_label = tids_label_initial
tids_unlabel = tids_unlabel_initial

with torch.no_grad():
    print("computing trajectory embeddings")
    for tid in tqdm(tuple(tids_train) + tuple(tids_test)):
        compute_latent(tid)

    res = []
    for i in range(26):
        print("======== " + str(i) + " ========")
        (tp, tn, fp, fn), tid_to_label, cls, thresh, dist = run_baseline_on_task(
            tids_pos,
            tids_unlabel,
            tids_label,
            tids_test,
        )
        tids_label, tids_unlabel = add_label_to_task(tids_label, tids_unlabel, tid_to_label)
        res.append((tp, tn, fp, fn))
        # torch.save(res, f"/home/sm1/tmp/vq3_{dataset_name}_lstm_{task_name}_al.torch.inprog")
        #ret[qid, i] = f1, thresh, tid_to_label, dist
        #print("res", qid, i, f1, "dbg", thresh, tid_to_label, dist)
        # output_file.write("\t".join([
        #     #f"{seed}",                              # "seed",
        #     "0", # seed has different implications for the Quivr version
        #     "lstm_semi",                            # "label_choice_method",
        #     dataset_name,                # "scenario",
        #     f"{task_name}",                               # "query_id",
        #     f"{i}",                                 # "al_step",
        #     f"{tp}",                                # "tp",
        #     f"{tn}",                                # "tn",
        #     f"{fp}",                                # "fp",
        #     f"{fn}",                                # "fn",
        #     #"None",                                 # "phquery",
        #     f"{seed}", # seed here is more analagous to phquery for quivr
        #     f"{tid_to_label}",                      # "tid",
        #     str(tid_to_label in tids_pos),   # "tid_is_pos",
        # ]) + "\n")
        # output_file.flush()

    torch.save(res, f"lstm_results/{dataset_name}_{task_name}.torch")