from pathlib import Path
import base64
import struct
import threading
import zipfile
import pytest

from asr2rpp.model_conversion import (
    checkpoint_to_safetensors, parse_big_beta7_yaml, ConversionError,
)


PICKLE_B64 = "gAJ9cQAoWCAAAABiYW5kX3NwbGl0LnRvX2ZlYXR1cmVzLjAuMC5nYW1tYXEBY3RvcmNoLl91dGlscwpfcmVidWlsZF90ZW5zb3JfdjIKcQIoKFgHAAAAc3RvcmFnZXEDY3RvcmNoCkZsb2F0U3RvcmFnZQpxBFgBAAAAMHEFWAMAAABjcHVxBksCdHEHUUsASwKFcQhLAYVxCYljY29sbGVjdGlvbnMKT3JkZXJlZERpY3QKcQopUnELdHEMUnENWCcAAABtYXNrX2VzdGltYXRvcnMuMC50b19mcmVxcy4wLjAuMC53ZWlnaHRxDmgCKChoA2gEWAEAAAAxcQ9oBksEdHEQUUsASwJLAoZxEUsCSwGGcRKJaAopUnETdHEUUnEVWCMAAABsYXllcnMuMC4wLmxheWVycy4wLjAudG9fcWt2LndlaWdodHEWaAIoKGgDaARYAQAAADJxF2gGSxJ0cRhRSwBLBksDhnEZSwNLAYZxGoloCilScRt0cRxScR11Lg=="


def make_torch_zip(path):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as zf:
        zf.writestr('tiny/data.pkl', base64.b64decode(PICKLE_B64))
        zf.writestr('tiny/data/0', struct.pack('<2f', 1.0, 2.0))
        zf.writestr('tiny/data/1', struct.pack('<4f', 0.0, 1.0, 2.0, 3.0))
        zf.writestr('tiny/data/2', struct.pack('<18f', *map(float, range(18))))

YAML = """audio:
  chunk_size: 676935
  n_fft: 2048
  hop_length: 441
  num_channels: 2
  sample_rate: 44100
model:
  dim: 384
  depth: 8
  stereo: true
  num_stems: 1
  time_transformer_depth: 1
  freq_transformer_depth: 1
  num_bands: 60
  dim_head: 64
  heads: 8
  dim_freqs_in: 1025
  sample_rate: 44100
  stft_n_fft: 2048
  stft_hop_length: 441
  stft_win_length: 2048
  stft_normalized: false
  mask_estimator_depth: 2
  multi_stft_resolutions_window_sizes: !!python/tuple
  - 4096
inference:
  batch_size: 2
  dim_t: 1201
  num_overlap: 2
"""


def test_big_beta7_yaml_mapping(tmp_path):
    config = tmp_path / 'big_beta7.yaml'
    config.write_text(YAML, encoding='utf-8')
    data = parse_big_beta7_yaml(config)
    assert data['chunk_size'] == 676935
    assert data['sample_rate'] == 44100
    assert data['num_overlap'] == 2
    assert data['n_fft'] == 2048
    assert data['stereo'] is True



def test_torch_free_checkpoint_reader_and_qkv_split(tmp_path):
    st = pytest.importorskip('safetensors.numpy')
    checkpoint = tmp_path / 'tiny.ckpt'
    make_torch_zip(checkpoint)
    config = tmp_path / 'big_beta7.yaml'
    config.write_text(YAML, encoding='utf-8')
    output = tmp_path / 'out'
    info = checkpoint_to_safetensors(checkpoint, config, output, threading.Event(), lambda _: None)
    tensors = st.load_file(str(output / 'model.safetensors'))
    assert info['qkv_split_count'] == 1
    assert 'layers.0.0.layers.0.0.to_qkv.weight' not in tensors
    assert tensors['layers.0.0.layers.0.0.to_q.weight'].shape == (2, 3)
    assert tensors['layers.0.0.layers.0.0.to_k.weight'].shape == (2, 3)
    assert tensors['layers.0.0.layers.0.0.to_v.weight'].shape == (2, 3)


def test_invalid_checkpoint_is_rejected_without_pickle_execution(tmp_path):
    checkpoint = tmp_path / 'bad.ckpt'
    checkpoint.write_bytes(b'not a zip')
    config = tmp_path / 'big_beta7.yaml'
    config.write_text(YAML, encoding='utf-8')
    with pytest.raises(ConversionError, match='ZIP'):
        checkpoint_to_safetensors(checkpoint, config, tmp_path / 'out', threading.Event(), lambda _: None)
