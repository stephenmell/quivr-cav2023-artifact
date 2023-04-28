import torch
import individual_TF
import torch.nn.functional as F
import torch.nn as nn
from transformer.noam_opt import NoamOpt
import math
import sklearn.metrics
import sys
import random
import numpy as np

assert len(sys.argv) == 5
dataset_name = sys.argv[1]
task_name = sys.argv[2]
train_or_eval = sys.argv[3]
assert train_or_eval in {"train", "eval"}
device = sys.argv[4]

seed = 0
np.random.seed(seed)
torch.manual_seed(seed + 10000)
random.seed(seed + 20000)

# device = "cuda:0"
# train_or_eval = "train"
# inp = torch.load("/tmp/gjgjjg_inp.torch").to(device)
# dec_inp = torch.load("/tmp/gjgjjg_dec_inp.torch").to(device)
# 
# src_att = torch.load("/tmp/gjgjjg_src_att.torch").to(device)
# trg_att = torch.load("/tmp/gjgjjg_trg_att.torch").to(device)
# params = torch.load("/tmp/gjgjjg_params.torch")

# dataset_name = "maritime_surveillance"
# task_name = "a"

def do_eval(model, X_chunk):
    this_batch_size = X_chunk.shape[0]
    inp = X_chunk
    src_att = torch.ones((this_batch_size, 1, inp_len)).to(X_chunk.device)
    dec_inp = torch.zeros((this_batch_size, outp_len, dim_dec_in)).to(X_chunk.device)
    trg_att = torch.tril(torch.ones((this_batch_size, outp_len, outp_len), dtype=torch.bool, device=X_chunk.device), 0)

    return model(inp, dec_inp, src_att, trg_att)[:,0,:]

def binclass(model, train_X):
    return proba(model, train_X) > 0.5

def binclass_chunked(model, train_X, batch_size):
    n_chunks = int(math.ceil(train_X.shape[0] / batch_size))

    X_chunks = torch.chunk(train_X, n_chunks, dim = 0)
    y_chunks = [binclass(model, X_chunk) for X_chunk in X_chunks]

    return torch.concat(y_chunks, dim = 0)

def proba(model, train_X):
    return F.softmax(do_eval(model, train_X), dim=1)[:,1]

def proba_chunked(model, train_X, batch_size):
    n_chunks = int(math.ceil(train_X.shape[0] / batch_size))
    
    X_chunks = torch.chunk(train_X, n_chunks, dim = 0)
    y_chunks = [proba(model, X_chunk) for X_chunk in X_chunks]

    return torch.concat(y_chunks, dim = 0)

training_data = "labeled"

raw_tracks_list = torch.load(f"datasets_raw/{dataset_name}.torch", map_location = device)
raw_tracks = nn.utils.rnn.pad_sequence(raw_tracks_list, batch_first = True)

dataset = torch.load(f"datasets/{dataset_name}/data.torch", map_location = device)
task = torch.load(f"datasets/{dataset_name}/task_{task_name}/task.torch", map_location = device)

tuple(len(r) for r in raw_tracks if r.shape[0] != 80)

def normalize_mean_stddev(nonnorm):
    nonnorm_flat = nonnorm.reshape((nonnorm.shape[0] * nonnorm.shape[1], nonnorm.shape[2]))
    means = nonnorm_flat.mean(dim = 0)
    stddevs = nonnorm_flat.std(dim = 0)
    return lambda t: (t - means)/stddevs

nonnorm_train_X = raw_tracks[dataset['train_indices'], :]
normalizer = normalize_mean_stddev(nonnorm_train_X)
normalizer_cheat = normalize_mean_stddev(raw_tracks[dataset['test_indices']])

# test_X = normalizer(raw_tracks[dataset['test_indices']])
test_X = normalizer(raw_tracks[dataset['test_indices']])
test_y = task["labels"][dataset['test_indices']].long()

def shuffle_and_chunk(X, y, batch_size):
    assert X.shape[0] == y.shape[0]
    p = torch.randperm(X.shape[0])
    n_chunks = int(math.ceil(X.shape[0] / batch_size))

    X_chunks = torch.chunk(X[p, ...], n_chunks, dim = 0)
    y_chunks = torch.chunk(y[p, ...], n_chunks, dim = 0)

    return tuple(zip(X_chunks, y_chunks))

train_indices = set(tid.item() for tid in dataset['train_indices'])
labeled_indices = set(tid.item() for tid in task['labeled_initial_indices'])
unlabeled_indices = set(frozenset(train_indices) - frozenset(labeled_indices))

model = None
res = []
for i in range(27):
    model_path = f"transformer_models/{dataset_name}_{task_name}_{training_data}_{i}.torch"
    print(f"===== AL STEP {i} =====")
    # quivr_al_state = torch.load(f"quivr_results/{dataset_name}/task_{task_name}/al_state_smart_{i}.torch", map_location = device)
    # quivr_al_state = torch.load(f"quivr_results/{dataset_name}/task_{task_name}/al_state_smart_26.torch", map_location = device)
    # labeltrain_X = normalizer(raw_tracks[quivr_al_state['labeled_indices']])
    # labeltrain_y = task["labels"][quivr_al_state['labeled_indices']].long()

    ordered_labeled_indices = list(labeled_indices)
    labeltrain_X = normalizer(raw_tracks[ordered_labeled_indices])
    labeltrain_y = task["labels"][ordered_labeled_indices].long()

    match training_data:
        case "labeled":
            train_X = labeltrain_X
            train_y = labeltrain_y
        case "all":
            train_X = alltrain_X
            train_y = alltrain_y
        case _:
            assert False
    dataset_size = test_X.shape[0]

    # batch_size = 100
    batch_size = 20

    factor = 1
    d_model = 512
    warmup=10

    dim_enc_in = test_X.shape[2]
    dim_dec_in = 1
    dim_dec_out = 2
    inp_len = test_X.shape[1]
    outp_len = 1

    homerolled_dataloader = lambda: shuffle_and_chunk(train_X, train_y, batch_size)

    if train_or_eval == "train":
        if model is None:
            model = individual_TF.IndividualTF(
                dim_enc_in,
                dim_dec_in,
                dim_dec_out,
                # N = params["N"],
                N = 6,
                # d_model = params["d_model"],
                d_model = d_model,
                d_ff=2048,
                # h=params["h"],
                h=8,
                # dropout=params["dropout"],
                dropout=0.1,
                mean=None,
                std=None,
            ).to(device)

            optim = NoamOpt(d_model, factor, dataset_size*warmup,
                                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        for j in range(10000):
            do_break = False
            for X_chunk, y_chunk in homerolled_dataloader():
                optim.optimizer.zero_grad()
                pred = do_eval(model, X_chunk)
                loss = F.cross_entropy(pred, y_chunk)
                loss.backward()

                loss_norm = loss.item() / X_chunk.shape[0]
                print(loss_norm)

                optim.step()

                if loss_norm <= 0.001:
                    do_break = True
                    break
            if do_break:
                break

        torch.save(model, model_path)

        # do active labeling
        with torch.no_grad():
            model_eval = torch.load(model_path, map_location=device)
            model_eval.eval()
            
            ordered_unlabeled_indices = list(unlabeled_indices)
            unlabeltrain_X = normalizer(raw_tracks[ordered_unlabeled_indices])
            unlabeltrain_pred = proba_chunked(model_eval, unlabeltrain_X, batch_size)
            assert len(unlabeltrain_pred.shape) == 1

            opt_unlabeled_index = unlabeltrain_pred.argmax()
            opt_value = unlabeltrain_pred.max()

            opt_unlabeled = ordered_unlabeled_indices[opt_unlabeled_index]
            print("labeling", opt_unlabeled, "with prob", opt_value, "gt label", task["labels"][opt_unlabeled])

            labeled_indices.add(opt_unlabeled)
            assert opt_unlabeled in unlabeled_indices
            unlabeled_indices.remove(opt_unlabeled)
            print(len(labeled_indices), len(unlabeled_indices))

    elif train_or_eval == "eval":
        with torch.no_grad():
            model = torch.load(model_path, map_location=device)
            model.eval()

            # train_f1 = sklearn.metrics.f1_score(
            #     train_y.cpu() == 1,
            #     binclass_chunked(model, train_X).cpu(),
            # )
            #
            # test_f1 = sklearn.metrics.f1_score(
            #     test_y.cpu() == 1,
            #     binclass_chunked(model, test_X).cpu(),
            # )
            #
            # print(dataset_name, task_name, training_data, train_f1, test_f1)
            
            test_gt = test_y.cpu() == 1
            test_pred = binclass_chunked(model, test_X, batch_size).cpu() == 1

            tp = test_pred[test_gt].sum()
            fp = test_pred[~test_gt].sum()
            fn = (~test_pred)[test_gt].sum()
            tn = (~test_pred)[~test_gt].sum()

            # print("DBG", test_f1, 2*tp/(2*tp + fp + fn))
            res.append((tp, tn, fp, fn))

if train_or_eval == "eval":
    torch.save(res, f"transformer_results/{dataset_name}_{task_name}.torch")

if False:
    labeltrain_f1 = sklearn.metrics.f1_score(
        labeltrain_y.cpu() == 1,
        binclass(model, labeltrain_X).cpu(),
    )

    alltrain_f1 = sklearn.metrics.f1_score(
        alltrain_y.cpu() == 1,
        binclass(model, alltrain_X).cpu(),
    )

    test_f1 = sklearn.metrics.f1_score(
        test_y.cpu() == 1,
        binclass(model, test_X).cpu(),
    )

    print(alltrain_f1, test_f1)

if False:
    lstm0 = [0.99884, 0.46547, 0.38302, 0.51851, 0.48137, 0.78017, 0.65410, 0.11946, 0.12667, 0.01284, 0.14800, 0.08163, 0.09865, 0.03253, 0.01469, 0.01837, 0.01340]
    transformer0 = [0.65962, 0.21157, 0.13661, 0.25250, 0.31717, 0.31715, 0.69530, 0.93749, 0.775, 1.0, 0.8, 0.37288, 0.45255, 0.12598, 0.08571, 0.08571, 0.06603]
    smart0 =[0.68545, 0.98913, 0.95578, 0.76697, 1.0, 0.88343, 0.67803, 0.30392, 0.37096, 0.06896, 0.27868, 0.66666, 0.91666, 0.16161, 0.07407, 0.6, 0.10884]
    lstm25 = lstm0
    transformer25 = [0.9100480659189746, 0.5167410714285714, 0.3293029871977241, 0.5031185031185031, 0.6277636824936572, 0.5591900311526481, 1.0, 0.9393939393939393, 0.9696969696969697, 1.0, 0.9600000000000001, 0.32432432432432434, 0.16216216216216217, 0.10975609756097561, 0.09790209790209789, 0.03773584905660377, 0.07142857142857144]
    smart25 = [0.99968, 0.98913, 0.98612, 0.99056, 1.0	, 0.99951, 1.0	, 0.91891, 1.0	, 0.8	, 0.99367, 0.95652, 0.98507, 1.0	, 1.0	, 0.18181	, 1.0]
    transformer25train=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8571428571428571, 0.888888888888889, 1.0, 1.0, 1.0, 1.0]
    transformer25all=[0.99884, 0.63362, 0.52526, 0.70924, 0.83610, 0.84322, 1.0, 0.96875, 0.94285, 0.8, 0.96000, 0.0, 0.49056, 0.28571, 0.0, 0.75000, 0.0]

    print("LSTM0:", sum(lstm0)/len(lstm0))
    print("Transformer0:", sum(transformer0)/len(transformer0))
    print("Synthesis0:", sum(smart0)/len(smart0))
    print("LSTM25:", sum(lstm25)/len(lstm25))
    print("Transformer25:", sum(transformer25)/len(transformer25))
    print("Synthesis25:", sum(smart25)/len(smart25))
    print("Transformer25train:", sum(transformer25train)/len(transformer25train))
    print("Transformer25all:", sum(transformer25all)/len(transformer25all))