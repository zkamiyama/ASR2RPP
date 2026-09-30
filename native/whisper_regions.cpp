// MIT. Bounded text-only ASR using ONLY whisper.cpp's public C API.
// This executable does not include or patch examples/cli/cli.cpp.
#include "pcm_plan.h"
#include "whisper.h"
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <functional>

namespace fs = std::filesystem;
static std::string quoted(const std::string &s) {
    std::ostringstream out; out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << char(c);
        else if (c < 32) out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << int(c) << std::dec;
        else out << char(c);
    }
    out << '"'; return out.str();
}
static double numeric(const std::string &s) {
    size_t end = 0; const auto value = std::stod(s, &end);
    if (end != s.size() || !std::isfinite(value)) throw std::runtime_error("Invalid numeric argument");
    return value;
}
static int integer(const std::string &s, int lo, int hi) {
    auto v = numeric(s);
    if (v < lo || v > hi || v != std::floor(v)) throw std::runtime_error("Integer argument out of range");
    return int(v);
}
static void expand(const std::string &arg, std::vector<std::string> &args) {
    if (arg.empty() || arg[0] != '@') { args.push_back(arg); return; }
    auto in = asr2rpp::input(arg.substr(1));
    in.seekg(0,std::ios::end);
    if (in.tellg()<0 || in.tellg()>16*1024*1024) throw std::runtime_error("Invalid response file size");
    in.seekg(0); std::string line;
    while (std::getline(in,line)) {
        if (line.find('\0')!=std::string::npos || line.find('\r')!=std::string::npos || (!line.empty()&&line[0]=='@'))
            throw std::runtime_error("Invalid/nested response argument");
        args.push_back(line);
        if (args.size()>50000) throw std::runtime_error("Too many arguments");
    }
}
static std::string identity(const std::string &p) {
    return fs::weakly_canonical(fs::u8path(p)).u8string();
}
static int execute(const std::vector<std::string> &args) {
    auto cp = whisper_context_default_params(); cp.flash_attn = true;
    auto p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    int beam = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
    bool nt=false, no_fallback=false; int context=-1;
    std::string model, bundle, language="ja", prompt, suppress;
    std::vector<std::string> plans;
    for (size_t i=0;i<args.size();++i) {
        const auto &a=args[i];
        auto next=[&]() -> std::string { if(++i>=args.size()) throw std::runtime_error("Missing argument: "+a); return args[i]; };
        if(a=="--help" || a=="-h") {
            std::cout << "ASR2RPP public-C-API text-only region runner\n"
                         "ASR2RPP_PCM_PLAN_V1 ASR2RPP_PCM_RESULT_V1 ASR2RPP_RESPONSE_V1\n"
                         "--asr2rpp-pcm-plan FILE --asr2rpp-pcm-results FILE -m MODEL -nt -mc 0\n";
            return 0;
        }
        else if(a=="--asr2rpp-pcm-plan") plans.push_back(next());
        else if(a=="--asr2rpp-pcm-results") bundle=next();
        else if(a=="-m") model=next();
        else if(a=="-l") language=next();
        else if(a=="-t") p.n_threads=integer(next(),1,128);
        else if(a=="-bs") beam=integer(next(),1,128);
        else if(a=="-bo") p.greedy.best_of=integer(next(),1,128);
        else if(a=="-mc") context=integer(next(),0,0);
        else if(a=="-p") integer(next(),1,1);
        else if(a=="-dev") cp.gpu_device=integer(next(),0,128);
        else if(a=="-ac") p.audio_ctx=integer(next(),0,1500);
        else if(a=="-ml") p.max_len=integer(next(),0,100000);
        else if(a=="-wt") p.thold_pt=float(numeric(next()));
        else if(a=="-et") p.entropy_thold=float(numeric(next()));
        else if(a=="-lpt") p.logprob_thold=float(numeric(next()));
        else if(a=="-nth") p.no_speech_thold=float(numeric(next()));
        else if(a=="-tp") p.temperature=float(numeric(next()));
        else if(a=="-tpi") p.temperature_inc=float(numeric(next()));
        else if(a=="--prompt") prompt=next();
        else if(a=="--suppress-regex") suppress=next();
        else if(a=="--carry-initial-prompt") p.carry_initial_prompt=true;
        else if(a=="-sns") p.suppress_nst=true;
        else if(a=="-sow") p.split_on_word=true;
        else if(a=="-nf") no_fallback=true;
        else if(a=="-nt") nt=true;
        else if(a=="-ng") cp.use_gpu=false;
        else if(a=="-fa") cp.flash_attn=true;
        else if(a=="-nfa") cp.flash_attn=false;
        else if(a=="-oj" || a=="-np") {} // Compatibility with host's text-only request.
        else throw std::runtime_error("Unsupported region-runner argument: "+a);
    }
    if(model.empty() || plans.empty() || !nt || context!=0)
        throw std::runtime_error("Region inference requires model, PCM plan, -nt and -mc 0");
    std::vector<asr2rpp::region> regions;
    for(const auto &name:plans) asr2rpp::plan(name,regions);
    std::set<std::string> inputs{identity(model)}, outputs;
    for(const auto &r:regions) inputs.insert(identity(r.source));
    for(const auto &r:regions) {
        auto dest=identity(r.destination+".json");
        if(inputs.count(dest) || !outputs.insert(dest).second) throw std::runtime_error("Output collision");
    }
    if(!bundle.empty() && (inputs.count(identity(bundle)) || outputs.count(identity(bundle))))
        throw std::runtime_error("Aggregate output collision");
    // The public loader callback also handles UTF-8 model paths on Windows.
    auto file=asr2rpp::input(model);
    if(!file) throw std::runtime_error("Model is not readable");
    whisper_model_loader loader{}; loader.context=&file;
    loader.read=[](void *c,void *out,size_t n)->size_t { auto &f=*static_cast<std::ifstream*>(c); f.read(static_cast<char*>(out),n); return size_t(f.gcount()); };
    loader.eof=[](void *c)->bool { return static_cast<std::ifstream*>(c)->eof(); };
    loader.close=[](void *c) { static_cast<std::ifstream*>(c)->close(); };
    std::unique_ptr<whisper_context,decltype(&whisper_free)> ctx(whisper_init_with_params(&loader,cp),&whisper_free);
    if(!ctx) throw std::runtime_error("Whisper model initialization failed");
    p.strategy=beam>1?WHISPER_SAMPLING_BEAM_SEARCH:WHISPER_SAMPLING_GREEDY;
    p.beam_search.beam_size=beam;
    if (!whisper_is_multilingual(ctx.get())) language="en";
    if (no_fallback) p.temperature_inc=0;
    p.language=language.c_str(); p.initial_prompt=prompt.c_str();
    p.suppress_regex=suppress.empty()?nullptr:suppress.c_str();
    p.n_max_text_ctx=0; p.no_timestamps=true; p.token_timestamps=p.max_len>0;
    p.print_realtime=false; p.print_progress=false; p.print_timestamps=false; p.print_special=false;
    std::ofstream aggregate;
    if(!bundle.empty()) { aggregate=asr2rpp::output(bundle); if(!aggregate) throw std::runtime_error("Cannot open aggregate output"); aggregate << "{\"schema\":1,\"results\":[\n"; }
    asr2rpp::region_reader reader; std::vector<float> samples;
    for(size_t i=0;i<regions.size();++i) {
        std::string result;
        try {
            reader.read(regions[i],samples);
            if(whisper_full(ctx.get(),p,samples.data(),int(samples.size()))!=0) throw std::runtime_error("Whisper inference failed");
            std::ostringstream out; out << "{\"transcription\":[";
            for(int j=0;j<whisper_full_n_segments(ctx.get());++j) {
                if(j) out << ',';
                out << "{\"text\":" << quoted(whisper_full_get_segment_text(ctx.get(),j)) << '}';
            }
            out << "]}"; result=out.str();
        } catch(const std::exception &e) { result="{\"error\":"+quoted(e.what())+"}"; }
        if(aggregate.is_open()) { if(i) aggregate << ",\n"; aggregate << result; aggregate.flush(); }
        else { auto out=asr2rpp::output(regions[i].destination+".json"); out << result; out.flush(); if(!out) throw std::runtime_error("Cannot write region output"); }
        std::cerr << "region_completed=" << i+1 << '/' << regions.size() << '\n';
    }
    if(aggregate.is_open()) { aggregate << "]}\n"; aggregate.flush(); if(!aggregate) throw std::runtime_error("Cannot finish aggregate output"); }
    whisper_print_timings(ctx.get());
    return 0;
}
#ifdef _WIN32
int wmain(int argc,wchar_t **argv) {
    try {
        std::vector<std::string> args;
        for(int i=1;i<argc;++i) {
            int n=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,argv[i],-1,nullptr,0,nullptr,nullptr);
            if(n<=0) throw std::runtime_error("Invalid Windows argument encoding");
            std::string value(size_t(n),'\0');
            WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,argv[i],-1,&value[0],n,nullptr,nullptr);
            value.pop_back(); expand(value,args);
        }
        return execute(args);
    } catch(const std::exception &e) { std::cerr << "region_runner_error: " << e.what() << '\n'; return 2; }
}
#else
int main(int argc,char **argv) {
    try {
        std::vector<std::string> args;
        for(int i=1;i<argc;++i) expand(argv[i],args);
        return execute(args);
    } catch(const std::exception &e) { std::cerr << "region_runner_error: " << e.what() << '\n'; return 2; }
}
#endif
