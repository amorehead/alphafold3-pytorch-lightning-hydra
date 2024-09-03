import os
import time
from typing import Any, Dict, List, Literal, Tuple

import rootutils
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.memory import garbage_collection_cuda
from torch import Tensor
from torchmetrics import MaxMetric, MeanMetric

from alphafold3_pytorch.data import mmcif_writing
from alphafold3_pytorch.models.components.alphafold3 import (
    ComputeModelSelectionScore,
    LossBreakdown,
    Sample,
)
from alphafold3_pytorch.models.components.inputs import BatchedAtomInput
from alphafold3_pytorch.utils import RankedLogger
from alphafold3_pytorch.utils.model_utils import default_lambda_lr_fn
from alphafold3_pytorch.utils.tensor_typing import Bool, Float, typecheck
from alphafold3_pytorch.utils.utils import exists

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

AVAILABLE_LR_SCHEDULERS = ["wcd", "plateau"]
PHASES = Literal["train", "val", "test", "predict"]

log = RankedLogger(__name__, rank_zero_only=False)

# lightning module


class Alphafold3LitModule(LightningModule):
    """A `LightningModule` for AlphaFold 3. Implements details from Section 5.4 of the paper.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt

        self.net = None
        self.save_hyperparameters(logger=False)

        # for averaging loss across batches

        self.train_loss = MeanMetric()

        # for tracking best so far validation metrics as well as test metrics

        self.compute_model_selection_score = ComputeModelSelectionScore(
            is_fine_tuning=self.hparams.is_fine_tuning
        )

        self.val_model_selection_score_best = MaxMetric()

        self.val_model_selection_score = MeanMetric()
        self.test_model_selection_score = MeanMetric()

        self.val_top_ranked_lddt = MeanMetric()
        self.test_top_ranked_lddt = MeanMetric()

    @property
    def is_main(self) -> bool:
        """Check if the current process is the main process."""
        return self.trainer.global_rank == 0

    @typecheck
    def prepare_batch_dict(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the input batch dictionary for the model.

        :param batch_dict: The input batch dictionary.
        :return: The prepared batch dictionary.
        """
        if not self.net.has_molecule_mod_embeds:
            batch_dict["is_molecule_mod"] = None
        return batch_dict

    @typecheck
    def forward(self, batch: BatchedAtomInput) -> Tuple[Float[""], LossBreakdown]:  # type: ignore
        """Perform a forward pass through the model `self.net`.

        :param x: A batch of `AtomInput` data.
        :return: A tensor of losses as well as a breakdown of the component losses.
        """
        batch_dict = self.prepare_batch_dict(batch.model_forward_dict())
        return self.net(
            **batch_dict,
            return_loss_breakdown=True,
            diffusion_add_bond_loss=self.hparams.diffusion_add_bond_loss,
            diffusion_add_smooth_lddt_loss=self.hparams.diffusion_add_smooth_lddt_loss,
        )

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""

        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks

        self.val_model_selection_score.reset()
        self.val_model_selection_score_best.reset()

    @typecheck
    def model_step(self, batch: BatchedAtomInput) -> Tuple[Float[""], LossBreakdown]:  # type: ignore
        """Perform a single model step on a batch of data.

        :param batch: A batch of `AtomInput` data.
        :return: A tensor of losses as well as a breakdown of the component losses.
        """
        loss, loss_breakdown = self.forward(batch)
        return loss, loss_breakdown

    @typecheck
    def training_step(self, batch: BatchedAtomInput | None, batch_idx: int) -> Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of `AtomInput` data.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses.
        """
        if not exists(batch):
            return torch.tensor(0.0)

        loss, loss_breakdown = self.model_step(batch)

        # update and log metrics

        self.train_loss(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch.atom_inputs),
        )
        self.log_dict(
            loss_breakdown._asdict(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=len(batch.atom_inputs),
        )

        # visualize samples

        if self.hparams.visualize_train_samples_every_n_steps > 0:
            if batch_idx % self.hparams.visualize_train_samples_every_n_steps == 0:
                self.sample_and_visualize(batch, batch_idx, phase="train")

        # return loss or backpropagation will fail

        return loss

    @typecheck
    def validation_step(self, batch: BatchedAtomInput | None, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of `AtomInput` data.
        :param batch_idx: The index of the current batch.
        """
        if not exists(batch):
            return

        batch_dict = batch.dict()
        prepared_model_batch_dict = self.prepare_batch_dict(batch.model_forward_dict())

        # generate multiple samples per example in each batch

        samples: List[Sample] = []

        for _ in range(self.hparams.num_samples_per_example):
            batch_sampled_atom_pos, logits = self.net(
                **prepared_model_batch_dict,
                return_loss=False,
                return_confidence_head_logits=True,
                return_distogram_head_logits=True,
            )
            plddt = self.compute_model_selection_score.compute_confidence_score.compute_plddt(
                logits.plddt
            )
            samples.append((batch_sampled_atom_pos, logits.pde, plddt, logits.distance))

        score_details = self.compute_model_selection_score.compute_model_selection_score(
            batch,
            samples,
            is_fine_tuning=self.hparams.is_fine_tuning,
            return_details=True,
            # NOTE: The AF3 supplement (Section 5.7) suggests that DM did not compute validation RASA for unresolved regions
            compute_rasa=False,
        )

        top_sample = score_details.scored_samples[score_details.best_gpde_index]
        (
            top_sample_idx,
            top_batch_sampled_atom_pos,
            top_sample_plddt,
            top_model_selection_score,
        ) = (
            top_sample[0],
            top_sample[1],
            top_sample[2],
            top_sample[3],
        )

        # compute the unweighted lDDT score

        unweighted_score_details = (
            self.compute_model_selection_score.compute_model_selection_score(
                batch,
                samples,
                is_fine_tuning=self.hparams.is_fine_tuning,
                return_details=True,
                return_unweighted_scores=True,
                compute_rasa=False,
            )
        )

        unweighted_top_sample = unweighted_score_details.scored_samples[
            unweighted_score_details.best_gpde_index
        ]
        top_ranked_lddt = unweighted_top_sample[3]

        # update and log metrics

        self.val_model_selection_score(score_details.score)
        self.log(
            "val/model_selection_score",
            self.val_model_selection_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch.atom_inputs),
        )

        self.val_top_ranked_lddt(top_ranked_lddt)
        self.log(
            "val/top_ranked_lddt",
            self.val_top_ranked_lddt,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch.atom_inputs),
        )

        # visualize (top) samples

        if self.hparams.visualize_val_samples_every_n_steps > 0:
            if batch_idx % self.hparams.visualize_val_samples_every_n_steps == 0:
                assert exists(
                    top_batch_sampled_atom_pos
                ), "The top sampled atom positions must be provided to visualize them."
                filename_suffixes = [
                    f"-score-{score:.4f}" for score in top_model_selection_score.tolist()
                ]
                self.visualize(
                    sampled_atom_pos=top_batch_sampled_atom_pos,
                    atom_mask=~batch_dict["missing_atom_mask"],
                    filepaths=list(batch_dict["filepath"]),
                    batch_idx=batch_idx,
                    phase="val",
                    sample_idx=top_sample_idx,
                    filename_suffixes=filename_suffixes,
                    b_factors=top_sample_plddt,
                )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        val_model_selection_score = (
            self.val_model_selection_score.compute()
        )  # get current val model selection score
        self.val_model_selection_score_best(
            val_model_selection_score
        )  # update best so far val model selection score

        # log `val_model_selection_score_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        self.log(
            "val/val_model_selection_score_best",
            self.val_model_selection_score_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

        # free up GPU memory
        garbage_collection_cuda()

    @typecheck
    def test_step(self, batch: BatchedAtomInput | None, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of `AtomInput` data.
        :param batch_idx: The index of the current batch.
        """
        if not exists(batch):
            return

        batch_dict = batch.dict()
        prepared_model_batch_dict = self.prepare_batch_dict(batch.model_forward_dict())

        # generate multiple samples per example in each batch

        samples: List[Sample] = []

        for _ in range(self.hparams.num_samples_per_example):
            batch_sampled_atom_pos, logits = self.net(
                **prepared_model_batch_dict,
                return_loss=False,
                return_confidence_head_logits=True,
                return_distogram_head_logits=True,
            )
            plddt = self.compute_model_selection_score.compute_confidence_score.compute_plddt(
                logits.plddt
            )
            samples.append((batch_sampled_atom_pos, logits.pde, plddt, logits.distance))

        score_details = self.compute_model_selection_score.compute_model_selection_score(
            batch,
            samples,
            is_fine_tuning=self.hparams.is_fine_tuning,
            return_details=True,
            return_unweighted_scores=False,
            # NOTE: The AF3 supplement (Section 5.7) suggests that DM computed RASA only for the test set's unresolved regions
            # TODO: Find where to get the unresolved chain IDs and residue masks from to statically set `compute_rasa=True` to match the AF3 supplement
            compute_rasa=self.compute_model_selection_score.can_calculate_unresolved_protein_rasa,
            unresolved_cid=None,
            unresolved_residue_mask=None,
        )

        top_sample = score_details.scored_samples[score_details.best_gpde_index]
        (
            top_sample_idx,
            top_batch_sampled_atom_pos,
            top_sample_plddt,
            top_model_selection_score,
        ) = (
            top_sample[0],
            top_sample[1],
            top_sample[2],
            top_sample[3],
        )

        # compute the unweighted lDDT score

        unweighted_score_details = (
            self.compute_model_selection_score.compute_model_selection_score(
                batch,
                samples,
                is_fine_tuning=self.hparams.is_fine_tuning,
                return_details=True,
                return_unweighted_scores=True,
                compute_rasa=False,
            )
        )

        unweighted_top_sample = unweighted_score_details.scored_samples[
            unweighted_score_details.best_gpde_index
        ]
        top_ranked_lddt = unweighted_top_sample[3]

        # update and log metrics

        self.test_model_selection_score(score_details.score)
        self.log(
            "test/model_selection_score",
            self.test_model_selection_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch.atom_inputs),
        )

        self.test_top_ranked_lddt(top_ranked_lddt)
        self.log(
            "test/top_ranked_lddt",
            self.test_top_ranked_lddt,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch.atom_inputs),
        )

        # visualize (top) samples

        if self.hparams.visualize_test_samples_every_n_steps > 0:
            if batch_idx % self.hparams.visualize_test_samples_every_n_steps == 0:
                assert exists(
                    top_batch_sampled_atom_pos
                ), "The top sampled atom positions must be provided to visualize them."
                filename_suffixes = [
                    f"-score-{score:.4f}" for score in top_model_selection_score.tolist()
                ]
                self.visualize(
                    sampled_atom_pos=top_batch_sampled_atom_pos,
                    atom_mask=~batch_dict["missing_atom_mask"],
                    filepaths=list(batch_dict["filepath"]),
                    batch_idx=batch_idx,
                    phase="test",
                    sample_idx=top_sample_idx,
                    filename_suffixes=filename_suffixes,
                    b_factors=top_sample_plddt,
                )

    @typecheck
    @torch.inference_mode()
    def visualize(
        self,
        sampled_atom_pos: Float["b m 3"],  # type: ignore
        atom_mask: Bool["b m"],  # type: ignore
        filepaths: List[str],
        batch_idx: int,
        phase: PHASES,
        sample_idx: int = 1,
        filename_suffixes: List[str] | None = None,
        b_factors: Float["b m"] | None = None,  # type: ignore
    ) -> None:
        """Visualize samples pre-generated for the examples in a batch.

        :param sampled_atom_pos: The sampled atom positions for the batch.
        :param atom_mask: The atom mask for the batch.
        :param batch_idx: The index of the current batch.
        :param phase: The phase of the current step.
        :param sample_idx: The index of the sample to visualize.
        :param filename_suffixes: The suffixes to append to the filenames.
        :param b_factors: The B-factors or equivalent mmCIF field values to list for each atom.
        """
        samples_output_dir = os.path.join(self.trainer.default_root_dir, f"{phase}_samples")
        os.makedirs(samples_output_dir, exist_ok=True)

        batch_size = len(atom_mask)

        for b in range(batch_size):
            input_filepath = filepaths[b]
            file_id = os.path.splitext(os.path.basename(input_filepath))[0]
            filename_suffix = filename_suffixes[b] if exists(filename_suffixes) else ""

            output_filepath = os.path.join(
                samples_output_dir,
                os.path.basename(input_filepath).replace(
                    ".cif",
                    f"-sampled-epoch-{self.current_epoch}-step-{self.global_step}-batch-{batch_idx}-example-{b}-sample-{sample_idx}{filename_suffix}.cif",
                ),
            )

            example_atom_mask = atom_mask[b]
            sampled_atom_positions = sampled_atom_pos[b][example_atom_mask].float().cpu().numpy()
            example_b_factors = (
                b_factors[b][example_atom_mask].float().cpu().numpy()
                if exists(b_factors)
                else None
            )

            mmcif_writing.write_mmcif_from_filepath_and_id(
                input_filepath=input_filepath,
                output_filepath=output_filepath,
                file_id=file_id,
                gapless_poly_seq=True,
                insert_orig_atom_names=True,
                insert_alphafold_mmcif_metadata=True,
                sampled_atom_positions=sampled_atom_positions,
                b_factors=example_b_factors,
            )

    @typecheck
    @torch.inference_mode()
    def sample_and_visualize(
        self,
        batch: BatchedAtomInput,
        batch_idx: int,
        phase: PHASES,
        sample_idx: int = 1,
        filename_suffixes: List[str] | None = None,
    ) -> None:
        """Visualize samples generated for the examples in the input batch.

        :param batch: A batch of `AtomInput` data.
        :param batch_idx: The index of the current batch.
        :param phase: The phase of the current step.
        :param sample_idx: The index of the sample to visualize.
        :param filename_suffixes: The suffixes to append to the filenames.
        """
        batch_dict = batch.dict()
        prepared_model_batch_dict = self.prepare_batch_dict(batch.model_forward_dict())

        batch_sampled_atom_pos = self.net(
            **prepared_model_batch_dict,
            return_loss=False,
        )

        samples_output_dir = os.path.join(self.trainer.default_root_dir, f"{phase}_samples")
        os.makedirs(samples_output_dir, exist_ok=True)

        for example_idx, sampled_atom_pos in enumerate(batch_sampled_atom_pos):
            input_filepath = batch_dict["filepath"][example_idx]
            file_id = os.path.splitext(os.path.basename(input_filepath))[0]
            filename_suffix = filename_suffixes[example_idx] if exists(filename_suffixes) else ""

            output_filepath = os.path.join(
                samples_output_dir,
                os.path.basename(input_filepath).replace(
                    ".cif",
                    f"-sampled-epoch-{self.current_epoch}-step-{self.global_step}-batch-{batch_idx}-example-{example_idx}-sample-{sample_idx}{filename_suffix}.cif",
                ),
            )

            atom_mask = ~batch_dict["missing_atom_mask"][example_idx]
            sampled_atom_positions = sampled_atom_pos[atom_mask].cpu().numpy()

            mmcif_writing.write_mmcif_from_filepath_and_id(
                input_filepath=input_filepath,
                output_filepath=output_filepath,
                file_id=file_id,
                gapless_poly_seq=True,
                insert_orig_atom_names=True,
                insert_alphafold_mmcif_metadata=True,
                sampled_atom_positions=sampled_atom_positions,
            )

    def on_after_backward(self):
        """Skip updates in case of unstable gradients.

        Reference: https://github.com/Lightning-AI/lightning/issues/4956
        """
        if self.hparams.skip_invalid_gradient_updates:
            valid_gradients = True
            for _, param in self.named_parameters():
                if param.grad is not None:
                    valid_gradients = not (
                        torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                    )
                    if not valid_gradients:
                        break
            if not valid_gradients:
                log.warning(
                    f"Detected `inf` or `nan` values in gradients at global step {self.trainer.global_step}. Not updating model parameters."
                )
                self.zero_grad()

    @typecheck
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_model(self):
        """Configure the model to be used for training, validation, testing, or prediction.

        :return: The configured model.
        """
        if exists(self.net):
            return

        if exists(self.trainer):
            sleep = self.trainer.global_rank * 4
            log.info(f"Rank {self.trainer.global_rank}: Sleeping for {sleep}s to avoid CPU OOMs.")
            time.sleep(sleep)

        net_config = {k: v for k, v in self.hparams.net.items() if k != "target"}
        self.net = self.hparams.net["target"](**net_config)

    def configure_optimizers(self):
        """Choose what optimizers and optional learning-rate schedulers to use during model
        optimization.

        :return: Configured optimizer(s) and optional learning-rate scheduler(s) to be used for
            training.
        """
        try:
            optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        except TypeError:
            # NOTE: Trainer strategies such as DeepSpeed require `params` to instead be specified as `model_params`
            optimizer = self.hparams.optimizer(model_params=self.trainer.model.parameters())
        if self.hparams.scheduler is None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=default_lambda_lr_fn, verbose=True
            )
        else:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/model_selection_score",
                "interval": "step",
                "frequency": 1,
                "name": "lambda_lr",
            },
        }


if __name__ == "__main__":
    _ = Alphafold3LitModule(None, None, None, None)
