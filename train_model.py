# Standard library
import os
from argparse import ArgumentParser

# Third-party
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed

# First-party
from neural_lam import config, utils
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.models.hi_lam import HiLAM
from neural_lam.models.hi_lam_parallel import HiLAMParallel
from neural_lam.weather_dataset import WeatherDataModule

MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "base_graph": BaseGraphModel,
    "hi_lam_parallel": HiLAMParallel,
}


def main():
    # pylint: disable=too-many-branches
    """
    Main function for training and evaluating models
    """
    parser = ArgumentParser(
        description="Train or evaluate NeurWP models for LAM"
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="neural_lam/data_config.yaml",
        help="Path to data config file (default: neural_lam/data_config.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graph_lam",
        help="Model architecture to train/evaluate/predict"
        "(default: graph_lam)",
    )
    parser.add_argument(
        "--subset_ds",
        type=int,
        default=0,
        help="Use only a small subset of the dataset, for debugging"
        "(default: 0=false)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="upper epoch limit (default: 200)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size (default: 4)"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to load model parameters from (default: None)",
    )
    parser.add_argument(
        "--resume_run", type=str, help="Run ID to resume (default: None)"
    )
    parser.add_argument(
        "--restore_opt",
        type=int,
        default=0,
        help="If optimizer state should be restored with model "
        "(default: 0 (false))",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=32,
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
    )

    # Model architecture
    parser.add_argument(
        "--graph",
        type=str,
        default="multiscale",
        help="Graph to load and use in graph-based model "
        "(default: multiscale)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimensionality of all hidden representations (default: 64)",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers in all MLPs (default: 1)",
    )
    parser.add_argument(
        "--processor_layers",
        type=int,
        default=4,
        help="Number of GNN layers in processor GNN (default: 4)",
    )
    parser.add_argument(
        "--mesh_aggr",
        type=str,
        default="sum",
        help="Aggregation to use for m2m processor GNN layers (sum/mean) "
        "(default: sum)",
    )
    parser.add_argument(
        "--output_std",
        type=int,
        default=0,
        help="If models should additionally output std.-dev. per "
        "output dimensions "
        "(default: 0 (no))",
    )

    # Training options
    parser.add_argument(
        "--ar_steps",
        type=int,
        default=1,
        help="Number of steps to unroll prediction for in loss (1-19) "
        "(default: 1)",
    )
    parser.add_argument(
        "--control_only",
        type=int,
        default=0,
        help="Train only on control member of ensemble data "
        "(default: 0 (False))",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="wmse",
        help="Loss function to use, see metric.py (default: wmse)",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=1,
        help="Step length in hours to consider single time step 1-3 "
        "(default: 1)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run "
        "(default: 1)",
    )

    # Evaluation options
    parser.add_argument(
        "--eval",
        type=str,
        help="Eval model on given data split (val/test/predict) "
        "(default: None (train model))",
    )
    parser.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during evaluation "
        "(default: 1)",
    )

    # Logger Settings
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="neural_lam",
        help="Wandb project name (default: neural_lam)",
    )
    parser.add_argument(
        "--val_steps_to_log",
        type=list,
        default=[1, 2, 3, 5, 10, 15, 19],
        help="Steps to log val loss for (default: [1, 2, 3, 5, 10, 15, 19])",
    )
    parser.add_argument(
        "--metrics_watch",
        type=list,
        default=[],
        help="List of metrics to watch, including any prefix (e.g. val_rmse)",
    )
    parser.add_argument(
        "--var_leads_metrics_watch",
        type=dict,
        default={},
        help="Dict with variables and lead times to log watched metrics for",
    )
    args = parser.parse_args()

    config_loader = config.Config.from_file(args.data_config)

    # Asserts for arguments
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.step_length <= 3, "Too high step length"
    assert args.eval in (
        None,
        "val",
        "test",
        "predict",
    ), f"Unknown eval setting: {args.eval}"

    # Set seed
    seed.seed_everything(args.seed)

    # Load data
    train_loader = torch.utils.data.DataLoader(
        WeatherDataset(
            config_loader.dataset.name,
            pred_length=args.ar_steps,
            split="train",
            subsample_step=args.step_length,
            subset=bool(args.subset_ds),
            control_only=args.control_only,
        ),
        args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
    )
    max_pred_length = (65 // args.step_length) - 2  # 19
    val_loader = torch.utils.data.DataLoader(
        WeatherDataset(
            config_loader.dataset.name,
            pred_length=max_pred_length,
            split="val",
            subsample_step=args.step_length,
            subset=bool(args.subset_ds),
            control_only=args.control_only,
        ),
        args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
    )

    # Instantiate model + trainer
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s

    # Load model parameters Use new args for model
    model_class = MODELS[args.model]
    model = model_class(args)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"saved_models/{run_name}",
        filename="min_val_loss",
        monitor="val_mean_loss",
        mode="min",
        save_last=True,
    )
    logger = pl.loggers.WandbLogger(
        project=args.wandb_project, name=run_name, config=args
    )
    if args.eval:
        use_distributed_sampler = False
    else:
        use_distributed_sampler = True
    utils.rank_zero_print("Arguments:")
    for arg in vars(args):
        utils.rank_zero_print(f"{arg}: {getattr(args, arg)}")

    if torch.cuda.is_available():
        accelerator = "cuda"
        devices = int(
            os.environ.get("SLURM_GPUS_PER_NODE", torch.cuda.device_count())
        )
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    else:
        accelerator = "cpu"
        devices = 1
        num_nodes = 1

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        log_every_n_steps=1,
        callbacks=(
            [checkpoint_callback] if checkpoint_callback is not None else []
        ),
        check_val_every_n_epoch=args.val_interval,
        precision=args.precision,
        use_distributed_sampler=use_distributed_sampler,
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        profiler="simple",
        deterministic=True,
        limit_predict_batches=1,
        # num_sanity_val_steps=0
        # strategy="ddp",
        # limit_val_batches=0
        # fast_dev_run=True
    )
    # Only init once, on rank 0 only
    if trainer.global_rank == 0:
        utils.init_wandb_metrics(
            logger, args.val_steps_to_log
        )  # Do after wandb.init

    if args.eval:
        if args.eval == "val":
            eval_loader = val_loader
        else:  # Test
            eval_loader = torch.utils.data.DataLoader(
                WeatherDataset(
                    config_loader.dataset.name,
                    pred_length=max_pred_length,
                    split="test",
                    subsample_step=args.step_length,
                    subset=bool(args.subset_ds),
                ),
                args.batch_size,
                shuffle=False,
                num_workers=args.n_workers,
            )

        print(f"Running evaluation on {args.eval}")
        trainer.test(model=model, dataloaders=eval_loader, ckpt_path=args.load)
    else:
        # Train model
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.load,
        )
    # Default mode is training
    else:
        data_module.split = "train"
        if args.load:
            trainer.fit(
                model=model, datamodule=data_module, ckpt_path=args.load
            )
        else:
            trainer.fit(model=model, datamodule=data_module)

    # Print profiler
    print(trainer.profiler)  # pylint: disable=no-member


if __name__ == "__main__":
    main()
