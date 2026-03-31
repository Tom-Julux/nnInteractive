import argparse
import multiprocessing
import os
from os.path import join
from typing import Optional, Union

import nnInteractive
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.run_training import maybe_load_checkpoint
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from torch.backends import cudnn
from torch.multiprocessing import spawn


def run_ddp(
    rank,
    dataset_name_or_id,
    configuration,
    fold,
    trainer_name,
    plans_identifier,
    disable_checkpointing,
    continue_training,
    only_run_validation,
    pretrained_weights,
    export_validation_probabilities,
    val_with_best,
    world_size,
):
    from nnunetv2.run.run_training import cleanup_ddp, setup_ddp

    setup_ddp(rank, world_size)
    torch.cuda.set_device(torch.device("cuda", torch.distributed.get_rank()))

    trainer = get_nninteractive_trainer_from_args(
        dataset_name_or_id=dataset_name_or_id,
        configuration=configuration,
        fold=fold,
        trainer_name=trainer_name,
        plans_identifier=plans_identifier,
    )

    if disable_checkpointing:
        trainer.disable_checkpointing = True

    assert not (
        continue_training and only_run_validation
    ), "Cannot set --c and --val at the same time."

    maybe_load_checkpoint(
        trainer, continue_training, only_run_validation, pretrained_weights
    )

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not only_run_validation:
        trainer.run_training()

    if val_with_best:
        trainer.load_checkpoint(join(trainer.output_folder, "checkpoint_best.pth"))
    trainer.perform_actual_validation(export_validation_probabilities)
    cleanup_ddp()


def get_nninteractive_trainer_from_args(
    dataset_name_or_id: Union[int, str],
    configuration: str,
    fold: int,
    trainer_name: str = "nnInteractiveTrainer",
    plans_identifier: str = "nnUNetPlans",
    device: torch.device = torch.device("cuda"),
) -> nnUNetTrainer:
    trainer_class = recursive_find_python_class(
        join(nnInteractive.__path__[0], "trainer"),
        trainer_name,
        "nnInteractive.trainer",
    )
    if trainer_class is None:
        raise RuntimeError(
            f"Could not find trainer {trainer_name} in nnInteractive.trainer "
            f"({join(nnInteractive.__path__[0], 'trainer')})."
        )
    assert issubclass(
        trainer_class, nnUNetTrainer
    ), "Requested trainer class must inherit nnUNetTrainer"

    if not str(dataset_name_or_id).startswith("Dataset"):
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError as exc:
            raise ValueError(
                "dataset_name_or_id must either be an integer or "
                "a valid dataset name with the pattern DatasetXXX_YYY."
            ) from exc

    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, dataset_name)
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + ".json")
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, "dataset.json"))
    return trainer_class(
        plans=plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        device=device,
    )


def run_training(
    dataset_name_or_id: Union[str, int],
    configuration: str,
    fold: Union[int, str],
    trainer_class_name: str = "nnInteractiveTrainer",
    plans_identifier: str = "nnUNetPlans",
    pretrained_weights: Optional[str] = None,
    num_gpus: int = 1,
    export_validation_probabilities: bool = False,
    continue_training: bool = False,
    only_run_validation: bool = False,
    disable_checkpointing: bool = False,
    val_with_best: bool = False,
    device: torch.device = torch.device("cuda"),
):
    if isinstance(fold, str):
        if fold != "all":
            fold = int(fold)

    if val_with_best:
        assert (
            not disable_checkpointing
        ), "--val_best is not compatible with --disable_checkpointing"

    if num_gpus > 1:
        assert (
            device.type == "cuda"
        ), f"DDP training requires cuda devices. Got: {device}."
        os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            from nnunetv2.utilities.networking import find_free_network_port

            os.environ["MASTER_PORT"] = str(find_free_network_port())
        spawn(
            run_ddp,
            args=(
                dataset_name_or_id,
                configuration,
                fold,
                trainer_class_name,
                plans_identifier,
                disable_checkpointing,
                continue_training,
                only_run_validation,
                pretrained_weights,
                export_validation_probabilities,
                val_with_best,
                num_gpus,
            ),
            nprocs=num_gpus,
            join=True,
        )
        return

    trainer = get_nninteractive_trainer_from_args(
        dataset_name_or_id,
        configuration,
        fold,
        trainer_class_name,
        plans_identifier,
        device=device,
    )

    if disable_checkpointing:
        trainer.disable_checkpointing = True

    assert not (
        continue_training and only_run_validation
    ), "Cannot set --c and --val at the same time."

    maybe_load_checkpoint(
        trainer,
        continue_training,
        only_run_validation,
        pretrained_weights,
    )

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not only_run_validation:
        trainer.run_training()

    if val_with_best:
        trainer.load_checkpoint(join(trainer.output_folder, "checkpoint_best.pth"))
    trainer.perform_actual_validation(export_validation_probabilities)


def run_training_entry():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name_or_id", type=str)
    parser.add_argument("configuration", type=str)
    parser.add_argument("fold", type=str)
    parser.add_argument(
        "-tr",
        type=str,
        required=False,
        default="nnInteractiveTrainer",
        help="Custom trainer in nnInteractive.trainer (default: nnInteractiveTrainer).",
    )
    parser.add_argument(
        "-p",
        type=str,
        required=False,
        default="nnUNetPlans",
        help="Custom plans identifier.",
    )
    parser.add_argument("-pretrained_weights", type=str, required=False, default=None)
    parser.add_argument("-num_gpus", type=int, default=1, required=False)
    parser.add_argument("--npz", action="store_true", required=False)
    parser.add_argument("--c", action="store_true", required=False)
    parser.add_argument("--val", action="store_true", required=False)
    parser.add_argument("--val_best", action="store_true", required=False)
    parser.add_argument("--disable_checkpointing", action="store_true", required=False)
    parser.add_argument(
        "-device",
        type=str,
        default="cuda",
        required=False,
        help="Device to use: cuda, cpu or mps.",
    )
    args = parser.parse_args()

    assert args.device in ["cpu", "cuda", "mps"]
    if args.device == "cpu":
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device("cpu")
    elif args.device == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
    else:
        device = torch.device("mps")

    run_training(
        dataset_name_or_id=args.dataset_name_or_id,
        configuration=args.configuration,
        fold=args.fold,
        trainer_class_name=args.tr,
        plans_identifier=args.p,
        pretrained_weights=args.pretrained_weights,
        num_gpus=args.num_gpus,
        export_validation_probabilities=args.npz,
        continue_training=args.c,
        only_run_validation=args.val,
        disable_checkpointing=args.disable_checkpointing,
        val_with_best=args.val_best,
        device=device,
    )


if __name__ == "__main__":
    run_training_entry()
