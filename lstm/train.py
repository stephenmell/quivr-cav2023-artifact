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
import math
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

model_utils = ModelUtils()

def bb_center_point(bb):
    assert bb.shape[-1] == 4, bb.shape
    return torch.stack([
        (bb[..., 0] + bb[..., 2])/2,
        (bb[..., 1] + bb[..., 3])/2,
    ], -1)

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


##### train_lstm #####
seed = 0
np.random.seed(seed)
torch.manual_seed(seed + 10000)
random.seed(seed + 20000)

import sys

if "get_ipython" in globals():
    dataset_name = "maritime_surveillance"
    device_name = "cuda:3"
else:
    assert len(sys.argv) == 3
    dataset_name = sys.argv[1]
    device_name = sys.argv[2]

device = torch.device(device_name)

raw_tracks = torch_load(f"datasets_raw/{dataset_name}.torch")

batch_size = 4096
n_chunks = int(math.ceil(len(raw_tracks) / batch_size))
batches = [pad_sequences(raw_tracks[i*batch_size:(i+1)*batch_size]) for i in range(n_chunks)]

input_size = batches[0][0].shape[-1]
embedding_size = 96

print("batch with shape ", batches[0][0].shape)
print("Input size:", input_size)
# crude hack to sanity check the time axis vs the keypoint axis
assert batches[0][0].shape[1] == 60 or batches[0][0].shape[1] == 61 or batches[0][0].shape[1] == 80

# default is embedding_size=8, hidden_size=16, output_size=2
print("embedding_size = ", embedding_size, "hidden_size =", embedding_size * 2, "output_size =", input_size)
print("raw_tracks len", len(raw_tracks))

decoder = DecoderRNN(embedding_size = embedding_size, hidden_size = embedding_size * 2, output_size = input_size).to(device)
# Set to train mode
#encoder.train()
decoder.train()

#encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

# Zero the gradients
#encoder_optimizer.zero_grad()
decoder_optimizer.zero_grad()

predict_from_own_predictions = False
criterion = lambda t1, t2: F.mse_loss(t1, t2, reduction = 'none')

batch_loss = 0
batch_count = 0

snapshots = [copy.deepcopy(decoder)]
do_print = False
for j in range(10):
    for i in range(100):
        for batch_padded, batch_lens, batch_mask in batches:
            #print(i)

            input = batch_padded[:, 0, :]
            target = batch_padded[:, 1:, :]
            batch_size = input.shape[0]
            output_length = target.shape[1]
            rollout_len = output_length
            
            # Initialize encoder hidden state
            encoder_hidden = model_utils.init_hidden(
                batch_size,
                decoder.hidden_size
            )

            # Initialize losses
            loss = 0

            decoder_input = input

            # Initialize decoder hidden state as encoder hidden state
            decoder_hidden = encoder_hidden

            decoder_outputs = torch.zeros(target.shape).to(device)
            # Decode hidden state in future trajectory
            for di in range(rollout_len):
                decoder_output, decoder_hidden = decoder(decoder_input,
                                                            decoder_hidden)
                decoder_outputs[:, di, :] = decoder_output
                # Update loss
                #print(di, decoder_output[:, :2], target[:, di, :2])
                if do_print:
                    print("------ predicting ------")
                    print(decoder_input)
                    print(decoder_output)
                    print(target[:, di, :])
                    new_loss = criterion(decoder_output[:, :], target[:, di, :])
                    print(new_loss)
                #loss += new_loss

                if predict_from_own_predictions:
                    decoder_input = decoder_output
                else:
                    if di != rollout_len - 1:
                        decoder_input = target[:, di, :]
                    else:
                        decoder_input = None
            #print("------ predicting ------")
            #print(decoder_outputs)
            #print(target)

            normalizer = batch_lens.unsqueeze(-1).unsqueeze(-1)
            losses = criterion(decoder_outputs, target)
            mask = batch_mask[:, 1:].unsqueeze(-1)
            losses_normalized_masked = torch.where(mask, losses / normalizer, torch.tensor(0.0, device = device))

            loss = torch.sum(losses_normalized_masked)

            if do_print:
                print(decoder_outputs.shape)
                print(target.shape)
                print(losses.shape)
                print(normalizer.shape)
                print(mask.shape)

            batch_loss += loss

            batch_count += 1
            if batch_count == 1:
                print(batch_loss/batch_count)
                # Backpropagate
                batch_loss.backward()
                #encoder_optimizer.step()
                decoder_optimizer.step()
                #encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                batch_loss = 0
                batch_count = 0
    snapshots.append(copy.deepcopy(decoder))

for snapshot in snapshots:
    snapshot.eval()

torch.save(snapshots[-1], f"lstm_models/{dataset_name}.torch")