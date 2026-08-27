"""AMP 梯度溢出处理与精度选择的回归测试。

背景：screen 队列中 4 个模型均因 fp16 反向梯度瞬时溢出（梯度范数 inf）
触发 FloatingPointError 而中止。正确语义是：损失与输出有限时按 AMP 标准跳过
该步参数更新，仅连续溢出超过容忍上限才判定真实发散。
"""
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.optim as optim

from train import OceanModelTrainer


def _make_minimal_trainer(skip_limit=30, batches=2):
    """构造只具备 train_epoch 所需属性的最小 trainer（CPU，AMP 关闭）。"""
    torch.manual_seed(0)
    trainer = object.__new__(OceanModelTrainer)
    model = nn.Conv3d(2, 2, kernel_size=1)
    trainer.model = model
    trainer.forward_model = model
    trainer.compile_active = False
    trainer.device = torch.device('cpu')
    trainer.amp_enabled = False
    trainer.amp_dtype = torch.bfloat16
    trainer.grad_scaler = torch.amp.GradScaler('cuda', enabled=False)
    trainer.config = {
        'grad_clip_norm': 1.0,
        'use_gradient_loss': False,
    }
    trainer.criterion = nn.MSELoss()
    trainer.target_variables = ['TEMP']
    trainer.target_channel_slices = {'TEMP': slice(0, 2)}
    trainer.target_loss_weights = {'TEMP': 1.0}
    trainer.nonfinite_grad_skip_limit = skip_limit
    trainer.grad_overflow_skips_total = 0
    trainer._consecutive_nonfinite_grads = 0
    trainer.optimizer = optim.SGD(model.parameters(), lr=1e-2)
    trainer.epoch = 0
    trainer.writer = mock.Mock()

    inputs = torch.randn(batches, 2, 2, 4, 4)
    targets = torch.randn(batches, 2, 2, 4, 4)
    trainer.train_loader = [(inputs, targets)] * batches
    return trainer, model


class AmpDtypeResolutionTests(unittest.TestCase):
    def _trainer_on_device(self, device):
        trainer = object.__new__(OceanModelTrainer)
        trainer.device = torch.device(device)
        return trainer

    def test_bfloat16_alias_maps_to_bf16(self):
        for alias in ('bfloat16', 'bf16', 'BF16'):
            self.assertIs(
                self._trainer_on_device('cpu')._resolve_amp_dtype(alias),
                torch.bfloat16,
            )

    def test_float16_alias_maps_to_fp16(self):
        for alias in ('float16', 'fp16', 'half'):
            self.assertIs(
                self._trainer_on_device('cpu')._resolve_amp_dtype(alias),
                torch.float16,
            )

    def test_auto_resolves_to_a_valid_dtype(self):
        dtype = self._trainer_on_device('cpu')._resolve_amp_dtype('auto')
        self.assertIn(dtype, (torch.bfloat16, torch.float16))

    def test_unknown_dtype_is_rejected(self):
        with self.assertRaises(ValueError):
            self._trainer_on_device('cpu')._resolve_amp_dtype('fp8')


class GradOverflowSkipTests(unittest.TestCase):
    def test_transient_overflow_skips_step_without_failing(self):
        trainer, model = _make_minimal_trainer(skip_limit=30, batches=2)
        params_before = [p.detach().clone() for p in model.parameters()]

        real_clip = torch.nn.utils.clip_grad_norm_
        calls = {'n': 0}

        def clip_sometimes(parameters, max_norm):
            calls['n'] += 1
            if calls['n'] == 1:
                return torch.tensor(float('inf'))
            return real_clip(parameters, max_norm=max_norm)

        with mock.patch('torch.nn.utils.clip_grad_norm_', side_effect=clip_sometimes):
            avg_loss = trainer.train_epoch()

        self.assertTrue(torch.isfinite(torch.tensor(avg_loss)))
        self.assertEqual(trainer.grad_overflow_skips_total, 1)
        self.assertEqual(trainer._consecutive_nonfinite_grads, 0)
        # 溢出 batch 被跳过（参数保持不变），随后正常 batch 完成更新。
        changed = [
            not torch.equal(before, p.detach())
            for before, p in zip(params_before, model.parameters())
        ]
        self.assertTrue(any(changed))

    def test_consecutive_overflow_beyond_limit_raises(self):
        trainer, model = _make_minimal_trainer(skip_limit=1, batches=3)

        with mock.patch(
            'torch.nn.utils.clip_grad_norm_',
            return_value=torch.tensor(float('inf')),
        ):
            with self.assertRaises(FloatingPointError):
                trainer.train_epoch()

        self.assertEqual(trainer.grad_overflow_skips_total, 2)

    def test_persistent_divergence_still_fails_via_nonfinite_loss(self):
        trainer, model = _make_minimal_trainer(skip_limit=30, batches=1)
        # 损失本身非有限必须立即失败，与梯度溢出跳步语义无关。
        trainer.train_loader = [(
            torch.randn(1, 2, 2, 4, 4),
            torch.full((1, 2, 2, 4, 4), float('nan')),
        )]
        with self.assertRaises(FloatingPointError):
            trainer.train_epoch()


if __name__ == '__main__':
    unittest.main()
