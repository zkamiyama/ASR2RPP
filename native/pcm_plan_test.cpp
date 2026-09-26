// Standalone native protocol/numeric regression check, no model or GPU needed.
#include "pcm_plan.h"
#include <iostream>
#include <functional>
void require(bool value) { if (!value) throw std::runtime_error("Native PCM regression"); }
void rejects(const std::function<void()> &f) {
    bool rejected=false; try { f(); } catch (const std::exception &) { rejected=true; }
    require(rejected);
}
void u32(std::ostream &s,uint32_t v) { for (int i=0;i<4;i++) s.put(char((v>>(i*8))&255)); }
void u16(std::ostream &s,uint16_t v) { s.put(char(v&255));s.put(char(v>>8)); }
int main(int argc,char **argv) {
    try {
        if (argc!=2) throw std::runtime_error("Supply an existing test directory");
        const std::string root=argv[1], audio=root+"/pcm.wav", plan=root+"/plan.txt";
        {
            auto f=asr2rpp::output(audio);f.write("RIFF",4);u32(f,36+2*65536);f.write("WAVEfmt ",8);
            u32(f,16);u16(f,1);u16(f,1);u32(f,16000);u32(f,32000);u16(f,2);u16(f,16);
            f.write("data",4);u32(f,2*65536);
            for (uint32_t i=0;i<65536;i++) u16(f,uint16_t(i));
        }
        asr2rpp::pcm_reader reader(audio);std::vector<float> values;
        reader.region_data(0,65536,values);require(values.size()==65536);
        for (int i=0;i<65536;i++) require(values[i]==float(i>=32768?i-65536:i)/32768.0f);
        reader.region_data(19,400,values);require(values[0]==19/32768.0f);
        rejects([&]{reader.region_data(65530,10,values);});
        rejects([&]{reader.region_data(0,28*16000+1,values);});
        rejects([&]{reader.region_data(0,0,values);});
        auto write_plan=[&](const std::string &body) { auto f=asr2rpp::output(plan);f<<body; };
        std::vector<asr2rpp::region> regions;
        write_plan("ASR2RPP_PCM_PLAN_V1\n1\n"+audio+"\n"+root+"/result\n19\n400\n");
        asr2rpp::plan(plan,regions);require(regions.size()==1 && regions[0].first==19);
        rejects([&]{asr2rpp::plan(plan,regions);}); // duplicate output
        for (const auto &bad : std::vector<std::string>{"0","-1","9999999999999999999999"}) {
            regions.clear();write_plan("ASR2RPP_PCM_PLAN_V1\n"+bad+"\n");
            rejects([&]{asr2rpp::plan(plan,regions);});
        }
        write_plan("bad-version\n1\n");regions.clear();rejects([&]{asr2rpp::plan(plan,regions);});
        // Chunk lengths, format, and truncation are independently validated.
        { auto f=asr2rpp::output(root+"/bad.wav");f<<"RIFFbadfile"; }
        rejects([&]{asr2rpp::pcm_reader bad(root+"/bad.wav");});
        std::cout<<"PCM plan: all 65536 PCM16 values, bounds, plan schema and malformed input passed\n";
        return 0;
    } catch (const std::exception &e) { std::cerr<<e.what()<<"\n";return 1; }
}
