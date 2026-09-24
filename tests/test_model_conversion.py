from pathlib import Path
import threading
import pytest

from asr2rpp.model_conversion import (
    checkpoint_to_safetensors, parse_big_beta7_yaml, ConversionError,
)

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


def test_invalid_checkpoint_is_rejected_without_pickle_execution(tmp_path):
    checkpoint = tmp_path / 'bad.ckpt'
    checkpoint.write_bytes(b'not a zip')
    config = tmp_path / 'big_beta7.yaml'
    config.write_text(YAML, encoding='utf-8')
    with pytest.raises(ConversionError, match='ZIP'):
        checkpoint_to_safetensors(checkpoint, config, tmp_path / 'out', threading.Event(), lambda _: None)
