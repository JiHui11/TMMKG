from __future__ import print_function
import argparse
import torch
import numpy as np
from utils import *
import torch.optim as optim
import random
from Optim import ScheduledOptim
from emb_transe import TransE, get_device, create_mapping
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='TMMKG_KGC')

parser.add_argument('--base_dir', type=str, default='../data/', help='dataset dir')
parser.add_argument('--nb_epoch', type=int, default=30, help='number of epoch')
parser.add_argument('--batch_size', type=int, default=1, help='number of batch size')
parser.add_argument('--save_epoch', type=int, default=1, help='number of epoch for saving models')

parser.add_argument('--tva_train_triplets_dir', type=str, default='../data/tva_triples_train.json')
parser.add_argument('--tva_test_triplets_dir', type=str, default='../data/tva_triples_test.json')
parser.add_argument('--gt_train_triplets_dir', type=str, default='../data/tva_triples_train.json')
parser.add_argument('--gt_test_triplets_dir', type=str, default='../data/tva_triples_test.json')

parser.add_argument('--train', action='store_true', default=False, help='train a new model')
parser.add_argument('--output_path', action='store_true', default='./kgc_models/', help='the dir for saving the best model')
parser.add_argument('--model_path', type=str, default=None, help="Path to the model file")

FixSeed = 123
random.seed(FixSeed)
np.random.seed(FixSeed)
torch.manual_seed(FixSeed)
torch.cuda.manual_seed(FixSeed)
torch.cuda.manual_seed_all(FixSeed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train(args, model, optimizer):
    tva_triplets = read_json(args.tva_train_triplets_dir)
    gt_triplets = read_json(args.gt_train_triplets_dir)
    nb_batch = len(tva_triplets) // args.batch_size  
    print('nb_batch:', nb_batch)
    best_accuracy = 0.0
    for epoch in range(args.nb_epoch):
        epoch_loss = 0
        epoch_acc = 0
        epoch_micro_f1 = 0
        epoch_macro_f1 = 0
        model.train()
        tva_triplets = dict(random.sample(tva_triplets.items(), len(tva_triplets)))
        tva_items = list(tva_triplets.items())
#         model.zero_grad()
        for i in range(0, len(tva_items), args.batch_size):
            loss_kgc = 0
            batch = tva_items[i:i + args.batch_size]
            model.zero_grad()
            for key, tva_kg in batch:
                gt_kg = gt_triplets[key]
                result = model(tva_kg)
                loss_kgc += result
                train_metrics = model.predict(test_triplets=tva_kg, gt_triplets=gt_kg)
                epoch_acc += train_metrics[0]
                epoch_micro_f1 += train_metrics[1]
                epoch_macro_f1 += train_metrics[2]
            loss = loss_kgc
            epoch_loss += loss.cpu().data.numpy()
            loss.backward()
            optimizer.step_lr()

        epoch_acc /= len(tva_triplets)
        epoch_micro_f1 /= len(tva_triplets)
        epoch_macro_f1 /= len(tva_triplets)
        print("=== Epoch {%s} | lr: {%.6f} | Loss: [{%.4f}] | acc: [{%.4f}] | micro_f1: [{%.4f}] | macro_f1: {%.4f}" \
              % (str(epoch), optimizer._optimizer.param_groups[0]['lr'], epoch_loss, epoch_acc, epoch_micro_f1, epoch_macro_f1))

        if (epoch+1) % args.save_epoch == 0:
            test_metrics = test(args, model)
            accuracy = test_metrics[0]
            print(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch

                print('test accuracy:', accuracy, 'epoch=', epoch)
                print('best test accuracy: {} ***************************************'.format(best_accuracy))
                torch.save({"state_dict": model.state_dict()}, args.output_path + f"best_model_acc_{best_accuracy:.4f}_epoch_{best_epoch}.tar")


def test(args, model, model_path=None):
    if model_path is not None:
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(">>> [Testing] Load pretrained model from " + model_path)
    model.eval()
    tva_triplets = read_json(args.tva_test_triplets_dir)
    gt_triplets = read_json(args.gt_test_triplets_dir)
    acc = 0
    micro_f1 = 0
    macro_f1 = 0
    for key, tva_kg in tva_triplets.items():
        gt_kg = gt_triplets[key]
        results = model.predict(test_triplets=tva_kg, gt_triplets=gt_kg)
        acc += results[0]
        micro_f1 += results[1]
        macro_f1 += results[2]
    acc /= len(tva_triplets)
    micro_f1 /= len(tva_triplets)
    macro_f1 /= len(tva_triplets)
    return acc, micro_f1, macro_f1


if __name__ == "__main__":
    args = parser.parse_args()
    print("args: ", args)
    device = get_device()
    entity2id, rel2id = create_mapping(args.base_dir)
    ent_num = len(entity2id)
    rel_num = len(rel2id)
    model = TransE(ent_num, rel_num, device,
                   norm=2,
                   embed_dim=100,
                   margin=2.0,
                   entity2id=entity2id,
                   rel2id=rel2id
                   )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = ScheduledOptim(optimizer)
    if args.train:
        train(args, model, optimizer)
    else:
        results = test(args, model, model_path=args.model_path)
        print(results[0])
        print(results[1])
        print(results[2])



