import argparse

from dig.threedgraph.dataset import QM93D
from dig.threedgraph.method import SphereNet
from dig.threedgraph.evaluation import ThreeDEvaluator
import run
import torch
import sys
import eval
import pandas as pd


#task1 homo,lumo,gap,r2
#task2 zpve,U0,U,H,G,Cv
#task3 mu,alpha
from spherenet import MultiSphereNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='task1')
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--results_dir', type=str, default='results')

    parser.add_argument('--cutoff', type=float, default=5.0)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--out_channels', type=int, default=1)
    parser.add_argument('--int_emb_size', type=int, default=64)
    parser.add_argument('--basis_emb_size_dist', type=int, default=8)
    parser.add_argument('--basis_emb_size_angle', type=int, default=8)
    parser.add_argument('--basis_emb_size_torsion', type=int, default=8)
    parser.add_argument('--out_emb_channels', type=int, default=256)
    parser.add_argument('--num_spherical', type=int, default=3)
    parser.add_argument('--num_radial', type=int, default=6)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--vt_batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=100)
    parser.add_argument('--checkpoint', type=str, default=None)


    args = parser.parse_args()


    device = args.device

    # Load the dataset and split
    dataset = QM93D(root='dataset/')
    target = args.target
    if target == 'task1':
        targets = ['homo', 'lumo', 'gap']
    elif target == 'task2':
        targets = ['zpve', 'U0', 'U', 'H', 'G', 'Cv']
    else:
        targets = ['mu', 'alpha']
    dataset.data.y = torch.stack([dataset.data[target] for target in targets], 1)
    data = dataset.data.y
    split_idx = dataset.get_idx_split(len(data), train_size=100000, valid_size=10000, seed=42)
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[
        split_idx['test']]

    # Define model, loss, and evaluation
    model = MultiSphereNet(energy_and_force=False, cutoff=args.cutoff,
                           num_layers=args.num_layers,
                           hidden_channels=args.hidden_channels,
                           out_channels=args.out_channels,
                           int_emb_size=args.int_emb_size,
                           basis_emb_size_dist=args.basis_emb_size_dist,
                           basis_emb_size_angle=args.basis_emb_size_angle,
                           basis_emb_size_torsion=args.basis_emb_size_torsion,
                           out_emb_channels=args.out_emb_channels,
                           num_spherical=args.num_spherical,
                           num_radial=args.num_radial,
                           envelope_exponent=5, num_before_skip=1,
                           num_after_skip=2, num_output_layers=3)
    # if args.checkpoint is not None:
    #     m = torch.load(args.checkpoint)

    loss_func = torch.nn.L1Loss()
    evaluation = eval.ThreeDEvaluator()

    # Train and evaluate
    run3d = run.run()
    run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation,
              epochs=args.num_epochs, batch_size=64, vt_batch_size=32, lr=args.lr,lr_decay_factor=0.5, lr_decay_step_size=15,args=args)

