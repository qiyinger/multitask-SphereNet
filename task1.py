import argparse

from dig.threedgraph.dataset import QM93D
from dig.threedgraph.method import SphereNet
from dig.threedgraph.evaluation import ThreeDEvaluator
import run
import torch
import sys


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--target', type=str, default='U0')
    p.add_argument('--lr', type=float, default=0.0005)
    p.add_argument('--num_epochs', type=int, default=19)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--results_dir', type=str, default='results')
    args = p.parse_args()


    device = args.device

    # Load the dataset and split
    dataset = QM93D(root='dataset/')
    target = args.target
    dataset.data.y = dataset.data[target]
    data = dataset.data.y[0:300]
    split_idx = dataset.get_idx_split(len(data), train_size=100, valid_size=100, seed=42)
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[
        split_idx['test']]

    # Define model, loss, and evaluation
    model = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4,
                      hidden_channels=128, out_channels=1, int_emb_size=64,
                      basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
                      num_spherical=3, num_radial=6, envelope_exponent=5,
                      num_before_skip=1, num_after_skip=2, num_output_layers=3)
    loss_func = torch.nn.L1Loss()
    evaluation = ThreeDEvaluator()

    # Train and evaluate
    run3d = run.run()
    run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation,
              epochs=args.num_epochs, batch_size=64, vt_batch_size=128, lr=args.lr,lr_decay_factor=0.5, lr_decay_step_size=15,args=args)

