import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from safetensors.numpy import save_file
import convert_anime_whisper as c
from inspect_ggml import inspect


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = dict(model_type='whisper', vocab_size=3, d_model=4,
                        encoder_layers=1, decoder_layers=1, max_source_positions=6,
                        max_target_positions=7, encoder_attention_heads=2,
                        decoder_attention_heads=2, num_mel_bins=80)
        (self.root / 'config.json').write_text(json.dumps(self.cfg))
        (self.root / 'vocab.json').write_text(json.dumps({'a': 0, 'b': 1, 'c': 2}))
        np.savez(self.root / 'mel_filters.npz', mel_80=np.zeros((80, 201), dtype=np.float32))
        rng = np.random.default_rng(123)
        self.weights = {key: rng.normal(size=shape).astype(np.float32)
                        for key, (_, shape) in c.specification(self.cfg).items()}
        self.save()

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        save_file(self.weights, self.root / 'model.safetensors')

    def test_f32_round_trip_is_exact(self):
        report = c.convert(self.root, self.root/'f32.bin', 'f32')
        parsed = inspect(self.root/'f32.bin')
        self.assertEqual(parsed['tensors'], report['tensors'])
        self.assertEqual(parsed['header'][6], 7)
        import hashlib
        for key, (name, shape) in c.specification(self.cfg).items():
            self.assertEqual(parsed['tensors'][name]['sha256'], hashlib.sha256(self.weights[key].tobytes()).hexdigest())
        self.assertNotEqual(np.float32(0.1234567), np.float32(np.float16(0.1234567)))

    def test_f16_keeps_auxiliary_f32_without_round_trip(self):
        report = c.convert(self.root, self.root/'f16.bin')
        parsed = inspect(self.root/'f16.bin')
        self.assertEqual(parsed['tensors'], report['tensors'])
        self.assertEqual(parsed['tensors']['encoder.conv1.weight']['dtype'], 'float16')
        self.assertEqual(parsed['tensors']['encoder.conv1.bias']['dtype'], 'float32')
        self.assertEqual(parsed['tensors']['decoder.positional_embedding']['dtype'], 'float32')
        a = np.array([0.1234567], dtype=np.float32)
        self.assertEqual(c.prepare_tensor(a, 'decoder.ln.bias', 'f16')[0], a[0])

    def test_unknown_tensor_is_rejected_and_no_output_remains(self):
        self.weights['unexpected'] = np.zeros(1, np.float32)
        self.save()
        with self.assertRaisesRegex(ValueError, 'tensor set'):
            c.convert(self.root, self.root/'bad.bin')
        self.assertFalse((self.root/'bad.bin').exists())
        self.assertFalse(list(self.root.glob('*.part')))

    def test_missing_tensor_is_rejected(self):
        del self.weights['model.encoder.conv1.bias']
        self.save()
        with self.assertRaisesRegex(ValueError, 'missing'):
            c.convert(self.root, self.root/'bad.bin')

    def test_untied_projection_is_not_silently_discarded(self):
        self.weights['proj_out.weight'] = np.zeros((3,4), np.float32)
        self.save()
        with self.assertRaisesRegex(ValueError, 'differs'):
            c.convert(self.root, self.root/'bad.bin')

    def test_projection_alias_is_supported(self):
        self.weights['proj_out.weight'] = self.weights.pop('model.decoder.embed_tokens.weight')
        self.save()
        c.convert(self.root, self.root/'alias.bin')
        self.assertIn('decoder.token_embedding.weight', inspect(self.root/'alias.bin')['tensors'])

    def test_nonfinite_is_rejected(self):
        self.weights['model.encoder.conv1.bias'][0] = np.nan
        self.save()
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            c.convert(self.root, self.root/'bad.bin')

    def test_shape_mismatch_is_rejected(self):
        self.weights['model.encoder.conv1.bias'] = np.zeros(8, np.float32)
        self.save()
        with self.assertRaisesRegex(ValueError, 'expected'):
            c.convert(self.root, self.root/'bad.bin')

    def test_vocab_byte_alphabet(self):
        self.assertEqual(c.vocab_bytes({'Ġ': 0, 'a': 1, 'Ċ': 2}), [b' ', b'a', b'\n'])
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            c.vocab_bytes({'x': 1})

    def test_does_not_overwrite_output(self):
        output = self.root/'exists.bin'
        output.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            c.convert(self.root, output)
        self.assertEqual(output.read_bytes(), b'keep')

if __name__ == '__main__':
    unittest.main()
