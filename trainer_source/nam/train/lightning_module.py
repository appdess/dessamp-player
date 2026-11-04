# File: base.py
# Created Date: Saturday February 5th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Implements the base PyTorch Lightning module.
This is meant to combine an actual model (subclassed from `..models.base.BaseNet`)
along with loss function boilerplate.

For the base *PyTorch* model containing the actual architecture, see `..models.base`.
"""

from dataclasses import dataclass as _dataclass
from enum import Enum as _Enum
from typing import (
    Any as _Any,
    Dict as _Dict,
    NamedTuple as _NamedTuple,
    Optional as _Optional,
    Tuple as _Tuple,
)

import auraloss as _auraloss
import logging as _logging
import pytorch_lightning as _pl
import torch as _torch
import torch.nn as _nn
import torchaudio as _ta

from .._core import InitializableFromConfig as _InitializableFromConfig
from ..models.conv_net import ConvNet as _ConvNet
from ..models.linear import Linear as _Linear
from ..models.losses import (
    apply_pre_emphasis_filter as _apply_pre_emphasis_filter,
    esr as _esr,
    multi_resolution_stft_loss as _multi_resolution_stft_loss,
    mse_fft as _mse_fft,
)
from ..models.recurrent import LSTM as _LSTM
from ..models.wavenet import WaveNet as _WaveNet
from ..models.vanilla_wavenet import VanillaWaveNet as _VanillaWaveNet
from .checkpoint import load_weights_only_checkpoint as _load_weights_only_checkpoint

logger = _logging.getLogger(__name__)


class ValidationLoss(_Enum):
    """
    mse: mean squared error
    esr: error signal ratio (Eq. (10) from
        https://www.mdpi.com/2076-3417/10/3/766/htm
        NOTE: Be careful when computing ESR on minibatches! The average ESR over
        a minibatch of data not the same as the ESR of all of the same data in
        the minibatch calculated over at once (because of the denominator).
        (Hint: think about what happens if one item in the minibatch is all
        zeroes...)
    """

    MSE = "mse"
    ESR = "esr"


@_dataclass
class LossConfig(_InitializableFromConfig):
    """
    :param mrstft_weight: Multi-resolution short-time Fourier transform loss
        coefficient. None means to skip; 2e-4 works pretty well if one wants to use it.
    :param mask_first: How many of the first samples to ignore when comptuing the loss.
    :param dc_weight: Weight for the DC loss term. If 0, ignored.
    :params val_loss: Which loss to track for the best model checkpoint.
    :param pre_emph_coef: Coefficient of 1st-order pre-emphasis filter from
        https://www.mdpi.com/2076-3417/10/3/766. Paper value: 0.95.
    :param pre_
    """

    mrstft_weight: _Optional[float] = None
    fourier: bool = False
    mask_first: int = 0
    dc_weight: float = None
    val_loss: ValidationLoss = ValidationLoss.MSE
    pre_emph_weight: _Optional[float] = None
    pre_emph_coef: _Optional[float] = None
    pre_emph_mrstft_weight: _Optional[float] = None
    pre_emph_mrstft_coef: _Optional[float] = None
    mel_weight: _Optional[float] = None

    @classmethod
    def parse_config(cls, config):
        config = super().parse_config(config)
        return {
            "fourier": config.get("fourier", False),
            "mask_first": config.get("mask_first", 0),
            "dc_weight": config.get("dc_weight"),
            "val_loss": ValidationLoss(config.get("val_loss", "mse")),
            "pre_emph_coef": config.get("pre_emph_coef"),
            "pre_emph_weight": config.get("pre_emph_weight"),
            "mrstft_weight": cls._get_mrstft_weight(config),
            "pre_emph_mrstft_weight": config.get("pre_emph_mrstft_weight"),
            "pre_emph_mrstft_coef": config.get("pre_emph_mrstft_coef"),
            "mel_weight": config.get("mel_weight")
        }

    def apply_mask(self, *args):
        """
        :param args: (L,) or (B,)
        :return: (L-M,) or (B, L-M)
        """
        return tuple(a[..., self.mask_first :] for a in args)

    @classmethod
    def _get_mrstft_weight(cls, config) -> _Optional[float]:
        key = "mrstft_weight"
        wrong_key = "mstft_key"  # Backward compatibility
        if key in config:
            if "mstft_weight" in config:
                raise ValueError(
                    f"Received loss configuration with both '{key}' and "
                    f"'{wrong_key}'. Provide only '{key}'."
                )
            return config[key]
        elif wrong_key in config:
            logger.warning(
                f"Use of '{wrong_key}' is deprecated and will be removed in a future "
                f"version. Use '{key}' instead."
            )
            return config[wrong_key]
        else:
            return None


class _LossItem(_NamedTuple):
    weight: _Optional[float]
    value: _Optional[_torch.Tensor]


_model_net_init_registry = {
    "ConvNet": _ConvNet.init_from_config,
    "Linear": _Linear.init_from_config,
    "LSTM": _LSTM.init_from_config,
    "WaveNet": _WaveNet.init_from_config,
    "VanillaWaveNet": _VanillaWaveNet.init_from_config
}


class LightningModule(_pl.LightningModule, _InitializableFromConfig):
    """
    The PyTorch Lightning Module that unites the model with its loss and
    optimization recipe.
    """

    def __init__(
        self,
        net,
        optimizer_config: _Optional[dict] = None,
        scheduler_config: _Optional[dict] = None,
        loss_config: _Optional[LossConfig] = None,
    ):
        """
        :param scheduler_config: contains
            Required:
            * "class"
            * "kwargs"
            Optional (defaults to Lightning defaults):
            * "interval" ("epoch" of "step")
            * "frequency" (int)
            * "monitor" (str)
        """
        super().__init__()
        self._net = net
        self._optimizer_config = {} if optimizer_config is None else optimizer_config
        self._scheduler_config = scheduler_config
        self._loss_config = LossConfig() if loss_config is None else loss_config
        self._mrstft = None  # Multi-resolution short-time Fourier transform loss
        self._mel_mrstft = None
        # Where to compute the MRSTFT.
        # Keeping it on-device is preferable, but if that fails, then remember to drop
        # it to cpu from then on.
        self._mrstft_device: _Optional[_torch.device] = None

    @classmethod
    def init_from_config(cls, config):
        checkpoint_path = config.get("checkpoint_path")
        config = cls.parse_config(config)
        return (
            cls(**config)
            if checkpoint_path is None
            else cls.load_from_safe_checkpoint(checkpoint_path, **config)
        )

    @classmethod
    def load_from_safe_checkpoint(
        cls,
        checkpoint_path,
        *,
        map_location="cpu",
        strict=True,
        **kwargs,
    ):
        """Build a module from tensor-only checkpoint data and explicit config."""
        checkpoint = _load_weights_only_checkpoint(
            checkpoint_path,
            map_location=map_location,
        )
        model = cls(**kwargs)
        model.on_load_checkpoint(checkpoint)
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
        return model

    @classmethod
    def parse_config(cls, config):
        """
        e.g.

        {
            "net": {
                "name": "ConvNet",
                "config": {...}
            },
            "loss": {
                "dc_weight": 0.1
            },
            "optimizer": {
                "lr": 0.0003
            },
            "lr_scheduler": {
                "class": "ReduceLROnPlateau",
                "kwargs": {
                    "factor": 0.8,
                    "patience": 10,
                    "cooldown": 15,
                    "min_lr": 1e-06,
                    "verbose": true
                },
                "monitor": "val_loss"
            }
        }
        """
        config = super().parse_config(config)
        net_config = config["net"]
        net = _model_net_init_registry[net_config["name"]](net_config["config"])
        loss_config = LossConfig.init_from_config(config.get("loss", {}))
        return {
            "net": net,
            "optimizer_config": config["optimizer"],
            "scheduler_config": config["lr_scheduler"],
            "loss_config": loss_config,
        }

    @classmethod
    def register_net_initializer(cls, name, constructor, overwrite: bool = False):
        if name in _model_net_init_registry and not overwrite:
            raise KeyError(
                f"A constructor for net name '{name}' is already registered!"
            )
        _model_net_init_registry[name] = constructor

    @property
    def net(self) -> _nn.Module:
        return self._net

    def configure_optimizers(self):
        optimizer = _torch.optim.Adam(self.parameters(), **self._optimizer_config)
        if self._scheduler_config is None:
            return optimizer
        else:
            lr_scheduler = getattr(
                _torch.optim.lr_scheduler, self._scheduler_config["class"]
            )(optimizer, **self._scheduler_config["kwargs"])
            lr_scheduler_config = {"scheduler": lr_scheduler}
            for key in ("interval", "frequency", "monitor"):
                if key in self._scheduler_config:
                    lr_scheduler_config[key] = self._scheduler_config[key]
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def forward(self, x: _torch.Tensor, g: _torch.Tensor=None, **kwargs):
        if g is None:
            return self.net(x, **kwargs)
        else:
            return self.net(x, g, **kwargs)  

    def on_load_checkpoint(self, checkpoint: _Dict[str, _Any]) -> None:
        # Resolves https://github.com/sdatkinson/neural-amp-modeler/issues/351
        self.net.sample_rate = checkpoint["sample_rate"]

    def on_save_checkpoint(self, checkpoint: _Dict[str, _Any]) -> None:
        # Resolves https://github.com/sdatkinson/neural-amp-modeler/issues/351
        checkpoint["sample_rate"] = self.net.sample_rate

    def _shared_step(
        self, batch
    ) -> _Tuple[_torch.Tensor, _torch.Tensor, _Dict[str, _LossItem]]:
        """
        B: Batch size
        L: Sequence length

        :return: (B,L), (B,L)

        Always assumes batch is (x, g, y) where x is the input signal, g is the global
        condition, and y is the target signal.
        """

        if len(batch) != 3:
             # Add basic check for expected batch structure
             raise ValueError(f"Expected batch to have 3 elements (x, g, y), but got {len(batch)}")

        x, g, targets = batch
        if (g < 0).any():
            preds = self(x, pad_start=False)
        else:
            preds = self(x, g, pad_start=False)

        # Compute all relevant losses.
        loss_dict = {}  # Mind keys versus validation loss requested...
        # Prediction aka MSE loss
        if self._loss_config.fourier:
            loss_dict["MSE_FFT"] = _LossItem(1.0, _mse_fft(preds, targets))
        else:
            loss_dict["MSE"] = _LossItem(1.0, self._mse_loss(preds, targets))
        # Pre-emphasized MSE
        if self._loss_config.pre_emph_weight is not None:
            if (self._loss_config.pre_emph_coef is None) != (
                self._loss_config.pre_emph_weight is None
            ):
                raise ValueError("Invalid pre-emph")
            loss_dict["Pre-emphasized MSE"] = _LossItem(
                self._loss_config.pre_emph_weight,
                self._mse_loss(
                    preds, targets, pre_emph_coef=self._loss_config.pre_emph_coef
                ),
            )
        # Multi-resolution short-time Fourier transform loss
        if self._loss_config.mrstft_weight is not None:
            loss_dict["MRSTFT"] = _LossItem(
                self._loss_config.mrstft_weight, self._mrstft_loss(preds, targets)
            )
        # Pre-emphasized MRSTFT
        if self._loss_config.pre_emph_mrstft_weight is not None:
            loss_dict["Pre-emphasized MRSTFT"] = _LossItem(
                self._loss_config.pre_emph_mrstft_weight,
                self._mrstft_loss(
                    preds, targets, pre_emph_coef=self._loss_config.pre_emph_mrstft_coef
                ),
            )
        # DC loss
        dc_weight = self._loss_config.dc_weight
        if dc_weight is not None and dc_weight > 0.0:
            # Denominator could be a bad idea. I'm going to omit it esp since I'm
            # using mini batches
            mean_dims = _torch.arange(1, preds.ndim).tolist()
            dc_loss = _nn.MSELoss()(
                preds.mean(dim=mean_dims), targets.mean(dim=mean_dims)
            )
            loss_dict["DC MSE"] = _LossItem(dc_weight, dc_loss)

        # Mel MRSTFT loss
        mel_weight = self._loss_config.mel_weight
        if mel_weight is not None and mel_weight > 0.0:
            loss_dict["mel"] = _LossItem(
                mel_weight, self._mel_mrstft_loss(preds, targets)
            )

        return preds, targets, loss_dict

    def training_step(self, batch, batch_idx):
        _, _, loss_dict = self._shared_step(batch)

        loss = 0.0
        for v in loss_dict.values():
            if v.weight is not None and v.weight > 0.0:
                loss = loss + v.weight * v.value
            # check for nan
            if _torch.isnan(v.value).any():
                raise RuntimeError(
                    f"NaN detected in loss value {v.value} for batch {batch_idx}"
                )
        return loss

    def validation_step(self, batch, batch_idx):
        preds, targets, loss_dict = self._shared_step(batch)

        def get_val_loss():
            # "esr" -> "ESR"
            # "mse" -> "MSE"
            # Others unsupported...
            # TODO better mapping from Enum to dict keys
            val_loss_type = self._loss_config.val_loss
            val_loss_key_for_loss_dict = val_loss_type.value.upper()
            if val_loss_key_for_loss_dict in loss_dict:
                return loss_dict[val_loss_key_for_loss_dict].value
            else:
                raise RuntimeError(
                    f"Undefined validation loss routine for {val_loss_type}"
                )

        loss_dict["ESR"] = _LossItem(None, self._esr_loss(preds, targets))
        val_loss = get_val_loss()
        self.log_dict(
            {
                "val_loss": val_loss,
                **{key: value.value for key, value in loss_dict.items()},
            }
        )
        return val_loss

    def _esr_loss(self, preds: _torch.Tensor, targets: _torch.Tensor) -> _torch.Tensor:
        """
        Error signal ratio aka ESR loss.

        Eq. (10), from
        https://www.mdpi.com/2076-3417/10/3/766/htm

        B: Batch size
        L: Sequence length

        :param preds: (B,L)
        :param targets: (B,L)
        :return: ()
        """
        return _esr(preds, targets)

    def _mse_loss(self, preds, targets, pre_emph_coef: _Optional[float] = None):
        if pre_emph_coef is not None:
            preds, targets = [
                _apply_pre_emphasis_filter(z, pre_emph_coef) for z in (preds, targets)
            ]
        return _nn.MSELoss()(preds, targets)

    def _mrstft_loss(
        self,
        preds: _torch.Tensor,
        targets: _torch.Tensor,
        pre_emph_coef: _Optional[float] = None,
    ) -> _torch.Tensor:
        """
        Experimental Multi Resolution Short Time Fourier Transform Loss using auraloss implementation.
        B: Batch size
        L: Sequence length

        :param preds: (B,L)
        :param targets: (B,L)
        :return: ()
        """
        if self._mrstft is None:
            self._mrstft = _auraloss.freq.MultiResolutionSTFTLoss()
        backup_device = "cpu"

        if pre_emph_coef is not None:
            preds, targets = [
                _apply_pre_emphasis_filter(z, pre_emph_coef) for z in (preds, targets)
            ]

        try:
            return _multi_resolution_stft_loss(
                preds, targets, self._mrstft, device=self._mrstft_device
            )
        except Exception as e:
            if self._mrstft_device == backup_device:
                raise e
            logger.warning("MRSTFT failed on device; falling back to CPU")
            self._mrstft_device = backup_device
            return _multi_resolution_stft_loss(
                preds, targets, self._mrstft, device=self._mrstft_device
            )

    def _mel_mrstft_loss(
        self, preds: _torch.Tensor, targets: _torch.Tensor
    ) -> _torch.Tensor:
        """
        Mel loss.

        :param preds: (B,L)
        :param targets: (B,L)
        :return: ()
        """
        if self._mel_mrstft is None: # Using setup from https://arxiv.org/pdf/2306.06546
            self._mel_mrstft = []
            window_lengths = [32, 64, 128, 256, 512, 1024, 2048]
            mel_bins = [5, 10, 20, 40, 80, 160, 320]
            sample_rate = self.net.sample_rate
            for win_len, n_mels in zip(window_lengths, mel_bins):
                hop_length = win_len // 4
                mel = _ta.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=win_len,
                    hop_length=hop_length,
                    win_length=win_len,
                    n_mels=n_mels,
                    power=1.0,
                    normalized=False
                ).to(preds.device)
                self._mel_mrstft.append(mel)

        try:
            preds_mel = [_torch.log10(_mel(preds) + 1e-5) for _mel in self._mel_mrstft]
            targets_mel = [_torch.log10(_mel(targets) + 1e-5) for _mel in self._mel_mrstft]
            loss = sum(_torch.nn.functional.l1_loss(pm, tm) for pm, tm in zip(preds_mel, targets_mel))

            return loss
        
        except Exception as e:
            raise RuntimeError(
                "Mel MRSTFT loss failed. Make sure you have a GPU."
            ) from e

