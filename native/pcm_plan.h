// ASR2RPP optional bounded PCM plan adapter for the app-pinned whisper.cpp CLI.
// MIT, same license as ASR2RPP. Decoder/options/output remain upstream code.
#pragma once
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace asr2rpp {
#ifdef _WIN32
inline std::wstring wide(const std::string &s) {
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), int(s.size()), nullptr, 0);
    if (!n) throw std::runtime_error("Invalid UTF-8 path in PCM plan");
    std::wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), int(s.size()), &w[0], n);
    return w;
}
#endif
inline std::ifstream input(const std::string &path) {
#ifdef _WIN32
    return std::ifstream(wide(path).c_str(), std::ios::binary);
#else
    return std::ifstream(path, std::ios::binary);
#endif
}
inline std::ofstream output(const std::string &path) {
#ifdef _WIN32
    return std::ofstream(wide(path).c_str(), std::ios::binary);
#else
    return std::ofstream(path, std::ios::binary);
#endif
}
inline std::string line(std::istream &in) {
    std::string s;
    if (!std::getline(in, s) || s.size() > 32768 || s.empty())
        throw std::runtime_error("Missing/oversized PCM plan field");
    if (s.find_first_of("\r\n\t") != std::string::npos || s.find('\0') != std::string::npos)
        throw std::runtime_error("Control character in PCM plan");
    return s;
}
inline uint64_t number(const std::string &s) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Invalid PCM frame number");
    size_t end = 0;
    auto n = std::stoull(s, &end);
    if (end != s.size()) throw std::runtime_error("Invalid PCM frame number");
    return n;
}
struct region { std::string source, destination; uint64_t first, count; };
inline void plan(const std::string &path, std::vector<region> &regions) {
    auto in = input(path);
    in.seekg(0, std::ios::end);
    if (in.tellg() < 0 || in.tellg() > 64*1024*1024) throw std::runtime_error("PCM plan too large");
    in.seekg(0);
    if (line(in) != "ASR2RPP_PCM_PLAN_V1") throw std::runtime_error("Unsupported PCM plan");
    const auto n = number(line(in));
    if (!n || n > 100000 || regions.size() + n > 100000)
        throw std::runtime_error("Invalid PCM plan size");
    std::set<std::string> outputs;
    for (const auto &r : regions) outputs.insert(r.destination);
    for (uint64_t i = 0; i < n; ++i) {
        region r; r.source = line(in); r.destination = line(in);
        r.first = number(line(in)); r.count = number(line(in));
        if (!r.count || r.count > 28*16000 || r.first > uint64_t(INT64_MAX)/2-r.count)
            throw std::runtime_error("PCM interval exceeds bounded sample limits");
        if (r.source == r.destination + ".json" || !outputs.insert(r.destination).second)
            throw std::runtime_error("PCM plan output collision");
        regions.push_back(r);
    }
    if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Trailing PCM plan data");
}
inline uint32_t le32(const unsigned char *p) {
    return uint32_t(p[0]) | uint32_t(p[1])<<8 | uint32_t(p[2])<<16 | uint32_t(p[3])<<24;
}
inline uint16_t le16(const unsigned char *p) { return uint16_t(p[0]) | uint16_t(p[1])<<8; }
class pcm_reader {
    std::ifstream file;
    uint64_t data = 0, frames = 0;
    void read(unsigned char *p, size_t n) {
        if (!file.read(reinterpret_cast<char *>(p), n)) throw std::runtime_error("Truncated PCM cache");
    }
public:
    explicit pcm_reader(const std::string &path) : file(input(path)) {
        file.seekg(0, std::ios::end);
        const auto end = file.tellg();
        if (end < 12) throw std::runtime_error("Invalid PCM cache size");
        const uint64_t size = uint64_t(end);
        file.seekg(0); unsigned char header[16]; read(header,12);
        if (std::string(reinterpret_cast<char *>(header),4) != "RIFF" ||
            std::string(reinterpret_cast<char *>(header+8),4) != "WAVE" || le32(header+4)+8ULL > size)
            throw std::runtime_error("PCM plan requires RIFF WAVE");
        bool format = false; uint64_t position = 12;
        const uint64_t riff_end = le32(header+4)+8ULL;
        while (position+8 <= riff_end) {
            file.seekg(position); read(header,8);
            const std::string tag(reinterpret_cast<char *>(header),4);
            const uint64_t length = le32(header+4); position += 8;
            if (length > riff_end-position) throw std::runtime_error("Truncated WAV chunk");
            if (tag == "fmt ") {
                if (length < 16) throw std::runtime_error("Invalid WAV format");
                read(header,16);
                if (le16(header)!=1 || le16(header+2)!=1 || le32(header+4)!=16000 ||
                    le16(header+12)!=2 || le16(header+14)!=16)
                    throw std::runtime_error("PCM plan requires 16 kHz mono PCM16");
                format = true;
            } else if (tag == "data") {
                if (length % 2 || data) throw std::runtime_error("Invalid WAV data chunk");
                data=position; frames=length/2;
            }
            position += length + (length & 1);
        }
        if (!format || !data || !frames) throw std::runtime_error("Missing PCM data");
    }
    void region_data(uint64_t first, uint64_t count, std::vector<float> &out) {
        if (!count || count>28*16000 || first>frames || count>frames-first)
            throw std::runtime_error("PCM region outside source data");
        file.clear(); file.seekg(data+first*2);
        std::vector<unsigned char> bytes(size_t(count*2)); read(bytes.data(),bytes.size());
        out.resize(size_t(count));
        for (size_t i=0;i<out.size();++i) {
            const uint16_t value=le16(bytes.data()+i*2);
            const int32_t signed_value = value>=32768 ? int32_t(value)-65536 : value;
            out[i]=float(signed_value)*(1.0f/32768.0f);
        }
    }
};
// Keep one file handle and at most 28 seconds of samples, regardless of source duration.
class region_reader {
    std::string source;
    std::unique_ptr<pcm_reader> reader;
public:
    void read(const region &r, std::vector<float> &out) {
        if (!reader || source!=r.source) {
            reader.reset(new pcm_reader(r.source)); source=r.source;
        }
        reader->region_data(r.first,r.count,out);
    }
};
} // namespace asr2rpp
