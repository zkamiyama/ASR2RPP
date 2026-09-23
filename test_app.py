import sys
import tempfile
import unittest
import wave
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch
import rpp_writer as rpp
from common import read_json, write_json, digest, run_process
from registry import Registry, merge
from engines import parse_whisper, parse_audio, native_command
from pipeline import assign, run_job


class WriterTests(unittest.TestCase):
    def test_quotes(self):
        self.assertEqual(rpp.quote('こんにちは "確認"'), '\'こんにちは "確認"\'')
    def test_quote_reject_newline(self):
        with self.assertRaises(ValueError): rpp.quote('a\nb')
    def test_quote_reject_all_delimiters(self):
        with self.assertRaises(ValueError): rpp.quote('"\'`')
    def test_fraction_required(self):
        with self.assertRaises(TypeError): rpp.seconds(0.1)
    def test_one_sample(self):
        self.assertAlmostEqual(float(rpp.seconds(Fraction(1,48000))),1/48000,places=11)
    def test_invalid_source(self):
        with self.assertRaises(ValueError): rpp.Source('x', 'NOT_RPP')
    def test_independent_offsets(self):
        item=rpp.Item('声',rpp.Source('原音.wav'),Fraction(1),Fraction(3),Fraction(2))
        text=rpp.dumps(rpp.Project((rpp.Track('話者',(item,)),)))
        self.assertIn('POSITION 1',text); self.assertIn('SOFFS 3',text)
    def test_invalid_length(self):
        with self.assertRaises(ValueError): rpp.Item('a',rpp.Source('a.wav'),Fraction(0),Fraction(0),Fraction(0))


class ParseTests(unittest.TestCase):
    def test_whisper_milliseconds(self):
        rows=parse_whisper({'transcription':[{'offsets':{'from':1234,'to':2000},'text':'テスト'}]})
        self.assertEqual(rows[0]['start'],1.234)
    def test_whisper_no_fake_times(self):
        with self.assertRaises(ValueError): parse_whisper({'transcription':[{'text':'a'}]})
    def test_audio_samples(self):
        rows=parse_audio([{'start_sample':1600,'end_sample':3200,'speaker_id':'spk0','text':'a'}],16000,'turns')
        self.assertEqual(rows[0]['start'],0.1); self.assertEqual(rows[0]['speaker'],'spk0')
    def test_audio_reject_unknown_time_units(self):
        with self.assertRaises(ValueError): parse_audio([{'start':1,'end':2}],16000,'turns')
    def test_no_speech(self):
        self.assertEqual(assign([],[],1)[0],[])
    def test_bounds_audited(self):
        seg=assign([dict(start=-0.1,end=1.2,text='a')],None,1)[0][0]
        self.assertEqual((seg['start'],seg['end']),(0,1)); self.assertIn('clipped_to_audio_bounds',seg['review'])
    def test_unknown_speaker(self):
        seg=assign([dict(start=0,end=1,text='a')],[],1)[0][0]
        self.assertEqual(seg['speaker'],'UNRESOLVED')
    def test_cross_speaker_not_split_by_chars(self):
        seg=assign([dict(start=0,end=1,text='会話')],[dict(start=0,end=0.5,speaker='A'),dict(start=0.5,end=1,speaker='B')],1)[0]
        self.assertEqual(len(seg),1); self.assertEqual(seg[0]['speaker'],'UNRESOLVED')
    def test_bad_turn(self):
        with self.assertRaises(ValueError): assign([], [dict(start=1,end=float('nan'))],1)
    def test_bad_asr(self):
        with self.assertRaises(ValueError): assign([dict(start=1,end=0,text='a')],None,2)
    def test_tokens_grouped(self):
        rows=[dict(start=0,end=.2,text='こ',granularity='token'),dict(start=.2,end=.5,text='んにちは',granularity='token')]
        seg=assign(rows,None,1)[0]
        self.assertEqual(len(seg),1); self.assertEqual(seg[0]['text'],'こんにちは')


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.cfg=read_json(Path(__file__).parent/'models.json')
        self.cfg['models']=self.cfg['models'][:1]
        self.cfg['models'][0]['path']='model.bin'
        self.cfg['engines']['whisper.cpp']['cpu']=sys.executable
        (self.root/'model.bin').write_bytes(b'model fixture, not actual weights')
        self.save()
    def tearDown(self): self.temp.cleanup()
    def save(self): write_json(self.root/'models.json', self.cfg)
    def test_registry(self): self.assertEqual(len(Registry(self.root/'models.json').models),1)
    def test_type_reject(self):
        self.cfg['models'][0]['type']='command'; self.save()
        with self.assertRaises(ValueError): Registry(self.root/'models.json')
    def test_engine_reject(self):
        self.cfg['models'][0]['engine']='shell'; self.save()
        with self.assertRaises(ValueError): Registry(self.root/'models.json')
    def test_duplicate(self):
        self.cfg['models']*=2; self.save()
        with self.assertRaises(ValueError): Registry(self.root/'models.json')
    def test_user_override(self):
        write_json(self.root/'models.user.json',dict(schema_version=1,models=[],tools={'ffmpeg':'other'}))
        self.assertEqual(Registry(self.root/'models.json').data['tools']['ffmpeg'],'other')
    def test_json_not_shell(self):
        reg=Registry(self.root/'models.json'); m=reg.model('whisper-tiny')
        cmd,_,_=native_command(reg,m,'a.wav',self.root,'cpu',{'prompt':'hello; touch x'})
        self.assertIn('hello; touch x',cmd); self.assertIn('-ng',cmd)
    def test_no_unknown_parameters(self):
        reg=Registry(self.root/'models.json')
        with self.assertRaises(ValueError): native_command(reg,reg.model('whisper-tiny'),'a.wav',self.root,'cpu',{'shell':True})
    def test_anime_prompt_block(self):
        reg=Registry(self.root/'models.json'); m=reg.model('whisper-tiny'); m['allow_prompt']=False
        with self.assertRaises(ValueError): native_command(reg,m,'a.wav',self.root,'cpu',{'prompt':'foo'})
    def test_missing_device_no_fallback(self):
        self.cfg['engines']['whisper.cpp']={'cpu':sys.executable}; self.save()
        with self.assertRaises(ValueError): Registry(self.root/'models.json').executable('whisper.cpp','vulkan')
    def test_deep_merge(self): self.assertEqual(merge({'a':{'x':1,'y':2}},{'a':{'x':3}}),{'a':{'x':3,'y':2}})
    def test_local_pipeline(self):
        wav=self.root/'original.wav'
        with wave.open(str(wav),'wb') as w:
            w.setparams((1,2,16000,0,'NONE','not compressed')); w.writeframes(b'\0\0'*32000)
        before=digest(wav)
        mock_units=[dict(start=.1,end=.5,text='こんにちは',granularity='segment')]
        with patch('pipeline.transcribe',return_value=(mock_units,{'process':{'wall_seconds':0},'test_fixture':True})):
            data=run_job(Registry(self.root/'models.json'),wav,self.root/'out','whisper-tiny',start=.5,duration=1,log=lambda x:None)
        self.assertEqual(digest(wav),before)
        text=(self.root/'out/project.rpp').read_text(encoding='utf-8')
        self.assertIn('SOFFS 0.6',text); self.assertIn('../original.wav',text)
        self.assertNotIn('inference_16000.wav',text)
        self.assertEqual(data['clip_source_start'],.5)
        with self.assertRaises(FileExistsError): run_job(Registry(self.root/'models.json'),wav,self.root/'out','whisper-tiny')
    def test_process_error(self):
        with self.assertRaises(RuntimeError): run_process([sys.executable,'-c','raise SystemExit(3)'],self.root/'err',5)
    def test_cancellation(self):
        cancel=self.root/'cancel'; cancel.touch()
        with self.assertRaises(InterruptedError): run_process([sys.executable,'-c','import time; time.sleep(10)'],self.root/'cancel_log',5,cancel)


if __name__=='__main__': unittest.main()
