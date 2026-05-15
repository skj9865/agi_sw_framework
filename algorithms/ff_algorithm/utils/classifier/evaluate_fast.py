import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils.orig_scff.create_layers import build_networks
from utils.custom_func.optimizer_custom import adamW_step, sgd_step
from utils.custom_func.stdnorm_custom import stdnorm_lut
import user_variables as uv

class CachedTensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self): return self.X.size(0)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class CustomStepLR(StepLR):
    """
    Custom Learning Rate schedule with step functions for supervised training of linear readout (classifier)
    """

    def __init__(self, optimizer, nb_epochs):
        threshold_ratios = [0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.step_thresold = [int(nb_epochs * r) for r in threshold_ratios]
        super().__init__(optimizer, -1, False)

    def get_lr(self):
        if self.last_epoch in self.step_thresold:
            return [group['lr'] * 0.5
                    for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]
    
@torch.no_grad()
def extract_split_features(loader, nets, pool, extra_pool, config):
    """
    loader: DataLoader that provices {images, labels}
    nets, pool, extra_pool: layers in SCFF
    config: EvaluationConfig (dims_in/out, stdnorm_out, Layer_out, all_neurons, device)

    return:
        feats: (N, D) torch.Tensor
        labels: (N,) torch.Tensor
    """
    # Fix evaluation mode
    for net in nets:
        net.eval()

    all_feats = []
    all_labels = []

    for x, labels in loader:
        x = x.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)

        outputs = []
        x_curr = x

        for j, net in enumerate(nets):
            if net.concat:
                x_curr = stdnorm_lut(x_curr, dims = config.dims_in, lut_ideal=uv.sfu_lut_ideal, lut_size=uv.sfu_lut_size)
                # x_curr = torch.cat((x_curr, x_curr), dim=1)

            # Conv → act → pool
            x_curr = pool[j]( net.act( net(x_curr) ) )

            # extra_pool is used when all_neurons is zero
            out = x_curr if config.all_neurons else extra_pool[j](x_curr)

            if config.stdnorm_out:
                out = stdnorm_lut(out, dims = config.dims_in, lut_ideal=uv.sfu_lut_ideal, lut_size=uv.sfu_lut_size)

            # Flatten
            out = out.flatten(start_dim=1)

            # Collecting layers of interest
            if j in config.Layer_out:
                outputs.append(out)

        feats = torch.cat(outputs, dim=1) if len(outputs)>0 else out
        all_feats.append(feats)
        all_labels.append(labels)

    feats = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return feats, labels


def extract_all_feature_splits(loaders, nets, pool, extra_pool, config, use_test_as_val=False):
    """
    loaders = (trainloader, valloader, testloader, suptrloader)
    use_test_as_val=test classifier using validation or test datasets
    return: dict with tensors
    """
    _, valloader, testloader, suptrloader = loaders
    # Classifier is trained using suptrloader
    trX, trY = extract_split_features(suptrloader, nets, pool, extra_pool, config)

    if use_test_as_val:
        vaX, vaY = extract_split_features(testloader, nets, pool, extra_pool, config)
    else:
        vaX, vaY = extract_split_features(valloader, nets, pool, extra_pool, config)

    teX, teY = extract_split_features(testloader, nets, pool, extra_pool, config)
    return {"train": (trX, trY), "val": (vaX, vaY), "test": (teX, teY)}



def build_classifier_from_cached(feat_dim, num_classes=10, dropout=0.2, device="cuda"):
    clf = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(feat_dim, num_classes)
    ).to(device)
    if uv.dtype == torch.float16:
        clf.half()

    return clf

def evaluate(loader, classifier, criterion):
    classifier.eval()
    correct=0; total=0; loss_sum=0.0
    with torch.no_grad():
        for Xb, yb in loader:
            logits = classifier(Xb)
            loss = criterion(logits, yb)
            loss_sum += loss.item() * yb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred==yb).sum().item()
            total += yb.size(0)
    return correct/total, loss_sum/total

def train_classifier(
    splits, # dict: {"train":(trX,trY), "val":(vaX,vaY), "test":(teX,teY)}
    config, 
    epochs=50,
    batch_size=512,
    base_lr=0.025
):
    trX, trY = splits["train"]
    vaX, vaY = splits["val"]
    teX, teY = splits["test"]

    device = config.device

    feat_dim = trX.size(1)
    classifier = build_classifier_from_cached(
        feat_dim, num_classes=10, dropout=config.out_dropout, device=device
    )

    train_ds = CachedTensorDataset(trX, trY)
    val_ds   = CachedTensorDataset(vaX, vaY)
    test_ds  = CachedTensorDataset(teX, teY)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # optimizer = optim.Adam(classifier.parameters(), lr=0.001, eps=uv.eps)
    optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3, nesterov=False)
    scheduler = CustomStepLR(optimizer, nb_epochs=60)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    for ep in range(epochs):
        classifier.train()
        for Xb, yb in train_loader:
            Xb = Xb.to(device, non_blocking=True) 
            if uv.dtype == torch.float16:
                Xb = Xb.half()
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            # optimizer.step()
            # adamW_step(optimizer, mode=uv.opm_type, lut_ideal=uv.sfu_lut_ideal, lut_size=uv.sfu_lut_size)
            sgd_step(optimizer)
        scheduler.step()

        if ep % 20 == 0 or ep == epochs-1:
            acc_tr, _ = evaluate(train_loader, classifier, criterion)
            acc_va, _ = evaluate(val_loader, classifier, criterion)
            print(f"[INFO] epoch {ep:03d} | train {acc_tr*100:.2f}% | val {acc_va*100:.2f}%")
            best_val = max(best_val, acc_va)

    acc_test, _ = evaluate(test_loader, classifier, criterion)
    return classifier, best_val, acc_test

def evaluate_model(layer_model, config, loaders, search, save_path, concats, device, act):
    print("Train & Test classifier (Revised version)")
    nets, pools, extra_pools = build_networks(layer_model, concats, device, act)

    saved_states = torch.load(save_path)
    for net, state in zip(nets, saved_states):
        net.load_state_dict(state)

    splits = extract_all_feature_splits(loaders, nets, pools, extra_pools, config, use_test_as_val=(not search))

    _, best_val, acc_test = train_classifier(
        splits,
        config=config,
        epochs=160,
        batch_size=32,
        base_lr=0.001
    )
    print(f"[Classifier] best val={best_val*100:.2f}%, test={acc_test*100:.2f}%")
