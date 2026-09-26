"""Apply a narrow, idempotent PCM-plan extension to the pinned upstream CLI.

Inference and result serialization remain upstream code. No network/decoder fork.
"""
from pathlib import Path
import hashlib
import shutil
import json

ROOT = Path(__file__).resolve().parents[1]


def patch_digest():
    return hashlib.sha256(Path(__file__).read_bytes() + (ROOT/'native/pcm_plan.h').read_bytes()).hexdigest()


def apply(source):
    target = Path(source)/'examples/cli/cli.cpp'
    original = target.read_bytes()
    backup = target.with_suffix('.asr2rpp-upstream')
    stamp = target.with_suffix('.asr2rpp-patch.json')
    if b'ASR2RPP_PCM_PLAN_V1' in original:
        if not stamp.is_file() or json.loads(stamp.read_text())['cli_sha256'] != hashlib.sha256(original).hexdigest():
            raise ValueError('Native CLI has unrecorded local edits; refusing to overwrite it')
        original = backup.read_bytes()
    original = original.replace(b'\r\n', b'\n')  # Windows checkout may use CRLF.
    blob = hashlib.sha1(b'blob '+str(len(original)).encode()+b'\0'+original).hexdigest()
    if blob != '55d5d4336d2f60b33b059a019d1659a8bf8b4bf3':
        raise ValueError('PCM patch requires the exact app-pinned upstream CLI')
    text = original.decode('utf-8')
    def change(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError('Unexpected pinned whisper CLI patch anchor: ' + old[:80])
        text = text.replace(old, new, 1)
    change('#include "common.h"', '#include "pcm_plan.h"\n#include "common.h"')
    change('    std::vector<std::string> fname_inp = {};',
           '    std::string pcm_results;\n    std::vector<std::string> pcm_plans;\n    std::vector<asr2rpp::region> pcm_regions;\n    std::vector<std::string> fname_inp = {};')
    change('        else if (arg == "-t"',
           '        else if (arg == "--asr2rpp-pcm-plan") { params.pcm_plans.emplace_back(ARGV_NEXT); }\n        else if (arg == "--asr2rpp-pcm-results") { params.pcm_results = ARGV_NEXT; }\n        else if (arg == "-t"')
    start = text.index('static void output_json(')
    end = text.index('    const bool full', start)
    signature = text[start:end]
    change(signature, signature.replace('std::ofstream', 'std::ostream'))
    change('    // remove non-existent files', '''    // ASR2RPP_PCM_PLAN_V1: bounded PCM inputs, one model, independent decoder history.
    if (!params.pcm_results.empty() && params.pcm_plans.empty()) return 2;
    if (!params.pcm_plans.empty()) {
        if (!params.fname_inp.empty() || !params.fname_out.empty() || params.vad ||
            !params.no_timestamps || params.max_context != 0 || params.n_processors != 1 ||
            params.translate || params.diarize || params.offset_t_ms || params.duration_ms) {
            fprintf(stderr, "error: PCM plans require independent -nt -mc 0 input\\n");
            return 2;
        }
        try {
            for (const auto &p : params.pcm_plans) asr2rpp::plan(p, params.pcm_regions);
            for (const auto &r : params.pcm_regions) {
                if (r.source == params.pcm_results || params.model == params.pcm_results) throw std::runtime_error("PCM result collision");
                params.fname_inp.push_back(r.source); params.fname_out.push_back(r.destination);
            }
        } catch (const std::exception &e) {
            fprintf(stderr, "error: PCM plan: %s\\n", e.what()); return 2;
        }
    }
    // remove non-existent files''')
    change('if (*it != "-" && !is_file_exist(fname_inp))',
           'if (params.pcm_plans.empty() && *it != "-" && !is_file_exist(fname_inp))')
    change('    for (int f = 0; f < (int) params.fname_inp.size(); ++f) {',
           '''    asr2rpp::region_reader pcm_reader;
    std::ofstream aggregate;
    if (!params.pcm_results.empty()) {
        aggregate = asr2rpp::output(params.pcm_results);
        if (!aggregate) return 12;
        aggregate << "{\\"schema\\":1,\\"results\\":[\\n";
    }
    for (int f = 0; f < (int) params.fname_inp.size(); ++f) {''')
    change('        if (!::read_audio_data(fname_inp, pcmf32, pcmf32s, params.diarize)) {',
           '''        if (!params.pcm_regions.empty()) {
            try { pcm_reader.read(params.pcm_regions.at(f), pcmf32); }
            catch (const std::exception &e) {
                fprintf(stderr, "error: PCM region: %s\\n", e.what()); return 11;
            }
        } else if (!::read_audio_data(fname_inp, pcmf32, pcmf32s, params.diarize)) {''')
    change('            output_func(output_json, ".json", params.output_jsn, pcmf32s);',
           '''            if (aggregate.is_open()) {
                if (f) aggregate << ",\\n";
                output_json(ctx, aggregate, params, pcmf32s);
            } else {
                output_func(output_json, ".json", params.output_jsn, pcmf32s);
            }''')
    change('    whisper_free(ctx);',
           '''    if (aggregate.is_open()) {
        aggregate << "]}\\n";
        aggregate.close();
        if (!aggregate) return 12;
    }
    whisper_free(ctx);''')
    change('                fout = std::ofstream{fname_out};',
           '                fout = asr2rpp::output(fname_out);')
    change('    fprintf(stderr, "options:\\n");',
           '    fprintf(stderr, "options:\\n");\n    fprintf(stderr, "  --asr2rpp-pcm-plan FILE   ASR2RPP_PCM_PLAN_V1 (16k mono PCM16 ranges)\\n");')
    # Fixed upstream @response-file parser accepts LF-delimited argv, without shell quoting.
    change('    fprintf(stderr, "supported audio formats: flac, mp3, ogg, wav\\n");',
           '    fprintf(stderr, "supported audio formats: flac, mp3, ogg, wav\\n");\n    fprintf(stderr, "  @response-file           ASR2RPP_RESPONSE_V1 ASR2RPP_PCM_RESULT_V1\\n");')
    backup.write_bytes(original)
    target.write_text(text, encoding='utf-8', newline='\n')
    shutil.copy2(ROOT/'native/pcm_plan.h', target.with_name('pcm_plan.h'))
    stamp.write_text(json.dumps({'cli_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                                 'patch_sha256':patch_digest()}),encoding='utf-8')


if __name__ == '__main__':
    import sys
    apply(Path(sys.argv[1]))
