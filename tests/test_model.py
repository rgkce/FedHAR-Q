"""
Adım 3 Doğrulama Testi: 1D-CNN Mimarisi, Opacus (DP-SGD) ve Kuantizasyon Uyumluluk Testleri.
"""

import torch
from opacus.validators import ModuleValidator
from opacus.grad_sample import GradSampleModule

from configs.config_loader import load_config
from models.cnn1d import FedHAR1DCNN, build_model_from_config, get_model_summary


def test_forward_pass_shape():
    """Modelin (Batch, 9, 128) girdisiyle (Batch, 6) çıktısı ürettiğini doğrular."""
    cfg = load_config()
    model = build_model_from_config(cfg)
    model.eval()

    x = torch.randn(16, 9, 128)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (16, 6), f"Beklenen çıktı boyutu (16, 6), gelen: {logits.shape}"


def test_opacus_dp_compatibility():
    """
    Modelin hiçbir katman değişikliğine gerek kalmadan Opacus ModuleValidator
    testinden 0 hata ile geçtiğini ve örnek-bazlı gradyan (per-sample gradient)
    hesaplayabildiğini doğrular.
    """
    model = FedHAR1DCNN()
    errors = ModuleValidator.validate(model, strict=False)
    assert len(errors) == 0, f"Opacus uyumsuz katman hatası: {errors}"

    # Gerçek bir ileri-geri yayılımda (backward) her örnek için ayrı gradyan üretildiğini test et
    gsm_model = GradSampleModule(model)
    gsm_model.train()
    x = torch.randn(8, 9, 128)
    y = torch.randint(0, 6, (8,))
    criterion = torch.nn.CrossEntropyLoss()

    out = gsm_model(x)
    loss = criterion(out, y)
    loss.backward()

    for name, param in gsm_model.named_parameters():
        if param.requires_grad:
            assert hasattr(param, "grad_sample"), f"{name} için grad_sample üretilmedi!"
            assert param.grad_sample is not None
            assert param.grad_sample.shape[0] == 8, f"{name} grad_sample batch boyutu 8 olmalı!"


def test_model_lightweight_footprint():
    """Modelin hafif (lightweight) sınırlar içinde olduğunu doğrular."""
    model = FedHAR1DCNN()
    summary = get_model_summary(model)
    assert 10_000 <= summary["total_params"] <= 100_000
    assert summary["fp32_param_size_kb"] < 400.0
