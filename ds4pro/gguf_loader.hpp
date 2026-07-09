/*
 * gguf_loader.hpp — GGUF 格式解析器 + 量化反量化
 *
 * 支持格式：F32, F16, Q4_0, Q4_1, Q8_0
 * 架构：LLaMA（兼容 Mistral, DeepSeek-LLM 等）
 */

#ifndef GGUF_LOADER_HPP
#define GGUF_LOADER_HPP

#include "ds4pro.hpp"
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace ds4pro {

// ============================================================
//  扩展类型别名（依赖 ds4pro.hpp 中的基础类型）
// ============================================================
using u8  = uint8_t;
using i8  = int8_t;
using u16 = uint16_t;
using i16 = int16_t;
using u64 = uint64_t;
using i64 = int64_t;
using f64 = double;
using f16 = u16;   // fp16 存储为 uint16_t

// ============================================================
//  GGUF 常量
// ============================================================
constexpr u32 GGUF_MAGIC   = 0x46554747u;  // "GGUF"
constexpr u32 GGUF_VERSION = 3;
constexpr i32 GGUF_ALIGN   = 32;

enum GGMLType : u32 {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
};

// Q4_0 量化块：32 个值 → 16 字节权重 + 1 个 fp16 scale
struct block_q4_0 {
    f16 d;         // scale (fp16)
    u8  qs[16];    // 4-bit quants, 2 per byte
};

// Q8_0 量化块：32 个值 → 32 字节权重 + 1 个 fp16 scale
struct block_q8_0 {
    f16 d;         // scale (fp16)
    i8  qs[32];    // 8-bit quants
};

inline f32 fp16_to_f32(f16 x) {
    u32 sign     = (x >> 15) & 1;
    u32 exponent = (x >> 10) & 0x1F;
    u32 mantissa = x & 0x3FF;
    
    if (exponent == 0) {
        if (mantissa == 0) {
            u32 bits = sign << 31;
            f32 val; std::memcpy(&val, &bits, sizeof(f32)); return val;
        }
        // Subnormal
        exponent = -14;
        while ((mantissa & 0x400) == 0) {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= 0x3FF;
        f32 val = (f32)mantissa / 1024.0f * std::pow(2.0f, (f32)exponent);
        return sign ? -val : val;
    }
    if (exponent == 0x1F) {
        u32 bits = (sign << 31) | 0x7F800000 | (mantissa << 13);
        f32 val; std::memcpy(&val, &bits, sizeof(f32)); return val;
    }
    f32 val = (1.0f + (f32)mantissa / 1024.0f) * std::pow(2.0f, (f32)((i32)exponent - 15));
    return sign ? -val : val;
}

// ============================================================
//  GGUF 读取器
// ============================================================
struct GGUFReader {
    std::ifstream file;
    u64 tensor_count;
    u64 kv_count;
    std::map<std::string, std::string> metadata_str;
    std::map<std::string, i32>      metadata_i32;
    std::map<std::string, f32>      metadata_f32;
    std::map<std::string, bool>      metadata_bool;
    std::vector<std::string>         metadata_arrays;
    
    struct TensorInfo {
        std::string name;
        i32 n_dims;
        u64 shape[4];
        GGMLType type;
        u64 offset;
    };
    std::vector<TensorInfo> tensors;
    u64 data_offset;  // 张量数据起始
    
    // ------ 辅助读取 ------
    u32 read_u32() {
        u32 v; file.read((char*)&v, 4); return v;
    }
    u64 read_u64() {
        u64 v; file.read((char*)&v, 8); return v;
    }
    i32 read_i32() {
        i32 v; file.read((char*)&v, 4); return v;
    }
    f32 read_f32() {
        f32 v; file.read((char*)&v, 4); return v;
    }
    f64 read_f64() {
        f64 v; file.read((char*)&v, 8); return v;
    }
    bool read_bool() {
        u8 v; file.read((char*)&v, 1); return v != 0;
    }
    std::string read_string() {
        u64 len = read_u64();
        std::string s(len, '\0');
        file.read(&s[0], len);
        return s;
    }
    void align(i32 boundary = GGUF_ALIGN) {
        u64 pos = file.tellg();
        u64 offset = (boundary - (pos % boundary)) % boundary;
        file.seekg(offset, std::ios::cur);
    }
    
    // ------ 打开文件 ------
    bool open(const std::string& path) {
        file.open(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "GGUF: 无法打开 " << path << std::endl;
            return false;
        }
        
        // Magic
        u32 magic = read_u32();
        if (magic != GGUF_MAGIC) {
            std::cerr << "GGUF: 无效的 magic number: 0x" << std::hex << magic << std::dec << std::endl;
            return false;
        }
        
        // Version
        u32 version = read_u32();
        if (version < 2 || version > 3) {
            std::cerr << "GGUF: 不支持的版本 " << version << std::endl;
            return false;
        }
        
        tensor_count = read_u64();
        kv_count     = read_u64();
        
        // 读取元数据
        for (u64 i = 0; i < kv_count; i++) {
            std::string key = read_string();
            u32 value_type = read_u32();
            
            switch (value_type) {
                case 0: read_u8(); break;  // u8
                case 1: read_i8(); break;  // i8
                case 2: read_u16(); break; // u16
                case 3: read_i16(); break; // i16
                case 4: metadata_i32[key] = read_i32(); break;
                case 5: read_i64(); break; // i64
                case 6: metadata_f32[key] = read_f32(); break;
                case 7: read_f64(); break; // f64
                case 8: metadata_bool[key] = read_bool(); break;
                case 9: metadata_str[key] = read_string(); break;
                case 10: {  // array
                    u32 arr_type = read_u32();
                    u64 arr_len  = read_u64();
                    // 跳过数组（我们不解析数组元数据）
                    i32 elem_size = 0;
                    switch (arr_type) {
                        case 0: case 1: elem_size = 1; break;
                        case 2: case 3: elem_size = 2; break;
                        case 4: case 5: elem_size = 4; break;
                        case 6: case 7: elem_size = 8; break;
                        case 8: elem_size = 1; break;
                        case 9: {
                            // String array — read length-prefixed strings
                            for (u64 j = 0; j < arr_len; j++) {
                                std::string s = read_string();
                                metadata_arrays.push_back(s);
                            }
                            elem_size = -1;  // special: already read
                            break;
                        }
                    }
                    if (elem_size > 0) {
                        file.seekg(arr_len * elem_size, std::ios::cur);
                    }
                    break;
                }
                default:
                    std::cerr << "GGUF: 未知元数据类型 " << value_type << std::endl;
                    return false;
            }
        }
        
        // 对齐到32字节（GGUF规范：元数据与张量信息之间需要对齐）
        align(GGUF_ALIGN);
        
        // 读取张量信息
        tensors.resize(tensor_count);
        for (u64 i = 0; i < tensor_count; i++) {
            tensors[i].name = read_string();
            tensors[i].n_dims = read_i32();
            for (i32 d = 0; d < tensors[i].n_dims; d++)
                tensors[i].shape[d] = read_u64();
            // 填充剩余维度
            for (i32 d = tensors[i].n_dims; d < 4; d++)
                tensors[i].shape[d] = 1;
            tensors[i].type   = (GGMLType)read_u32();
            tensors[i].offset = read_u64();
        }
        
        // 对齐
        align(GGUF_ALIGN);
        data_offset = file.tellg();
        
        return true;
    }
    
    // ------ 反量化张量 ------
    bool load_tensor(const TensorInfo& info, Tensor& dst) {
        // 计算张量大小
        u64 n_elements = 1;
        for (i32 d = 0; d < info.n_dims; d++)
            n_elements *= info.shape[d];
        
        i32 rows, cols;
        if (info.n_dims == 1) {
            rows = 1;
            cols = (i32)info.shape[0];
        } else if (info.n_dims == 2) {
            rows = (i32)info.shape[0];
            cols = (i32)info.shape[1];
        } else {
            // 多维展平为 1×N
            rows = 1;
            cols = (i32)n_elements;
        }
        
        dst = Tensor(rows, cols);
        
        file.seekg(data_offset + info.offset);
        
        switch (info.type) {
            case GGML_TYPE_F32: {
                file.read((char*)dst.data.data(), n_elements * sizeof(f32));
                break;
            }
            case GGML_TYPE_F16: {
                for (u64 i = 0; i < n_elements; i++) {
                    u16 v; file.read((char*)&v, 2);
                    dst.data[i] = fp16_to_f32(v);
                }
                break;
            }
            case GGML_TYPE_Q4_0: {
                u64 n_blocks = n_elements / 32;
                for (u64 b = 0; b < n_blocks; b++) {
                    u16 d_raw; file.read((char*)&d_raw, 2);
                    f32 d = fp16_to_f32(d_raw);
                    u8 qs[16]; file.read((char*)qs, 16);
                    for (i32 j = 0; j < 32; j++) {
                        i32 byte_idx = j / 2;
                        i32 nibble   = j % 2;
                        i8  q = (qs[byte_idx] >> (4 * nibble)) & 0x0F;
                        f32 val = (q - 8) * d;
                        dst.data[b * 32 + j] = val;
                    }
                }
                break;
            }
            case GGML_TYPE_Q8_0: {
                u64 n_blocks = n_elements / 32;
                for (u64 b = 0; b < n_blocks; b++) {
                    u16 d_raw; file.read((char*)&d_raw, 2);
                    f32 d = fp16_to_f32(d_raw);
                    i8 qs[32]; file.read((char*)qs, 32);
                    for (i32 j = 0; j < 32; j++) {
                        dst.data[b * 32 + j] = qs[j] * d;
                    }
                }
                break;
            }
            default:
                std::cerr << "GGUF: 不支持的量化类型 " << info.type << std::endl;
                return false;
        }
        return true;
    }
    
    u8 read_u8()  { u8 v; file.read((char*)&v, 1); return v; }
    i8 read_i8()  { i8 v; file.read((char*)&v, 1); return v; }
    u16 read_u16() { u16 v; file.read((char*)&v, 2); return v; }
    i16 read_i16() { i16 v; file.read((char*)&v, 2); return v; }
    i64 read_i64() { i64 v; file.read((char*)&v, 8); return v; }
};

// ============================================================
//  LLaMA 架构权重映射器
// ============================================================
struct LLaMAWeights {
    Tensor token_embd;       // [vocab, dim]
    std::vector<Tensor> attn_norm;   // [dim] per layer
    std::vector<Tensor> attn_q;      // [dim, n_heads*head_dim]
    std::vector<Tensor> attn_k;      // [dim, n_kv_heads*head_dim]
    std::vector<Tensor> attn_v;      // [dim, n_kv_heads*head_dim]
    std::vector<Tensor> attn_o;      // [n_heads*head_dim, dim]
    std::vector<Tensor> ffn_norm;    // [dim] per layer
    std::vector<Tensor> ffn_gate;    // [dim, ff_dim]
    std::vector<Tensor> ffn_up;      // [dim, ff_dim]
    std::vector<Tensor> ffn_down;    // [ff_dim, dim]
    Tensor output_norm;      // [dim]
    Tensor output;           // [dim, vocab]
    
    i32 dim, n_layers, n_heads, n_kv_heads, ff_dim, vocab_size;
    f32 rms_norm_eps;
    
    bool load(GGUFReader& reader, i32 /*n_blocks*/) {
        // 获取元数据
        auto get_i32 = [&](const std::string& key, i32 def) -> i32 {
            auto it = reader.metadata_i32.find(key);
            return it != reader.metadata_i32.end() ? it->second : def;
        };
        auto get_f32 = [&](const std::string& key, f32 def) -> f32 {
            auto it = reader.metadata_f32.find(key);
            return it != reader.metadata_f32.end() ? it->second : def;
        };
        
        std::string arch = reader.metadata_str.count("general.architecture")
            ? reader.metadata_str["general.architecture"] : "llama";
        
        std::string prefix = arch + ".";
        
        dim        = get_i32(prefix + "embedding_length", 4096);
        n_layers   = get_i32(prefix + "block_count", 32);
        n_heads    = get_i32(prefix + "attention.head_count", 32);
        n_kv_heads = get_i32(prefix + "attention.head_count_kv", n_heads);
        ff_dim     = get_i32(prefix + "feed_forward_length", 11008);
        rms_norm_eps = get_f32(prefix + "attention.layer_norm_rms_epsilon", 1e-6f);
        
        // 从 token_embd 获取 vocab
        vocab_size = 0;
        
        // 创建张量映射
        std::map<std::string, Tensor*> tensor_map;
        
        // 预分配
        attn_norm.resize(n_layers);
        attn_q.resize(n_layers);
        attn_k.resize(n_layers);
        attn_v.resize(n_layers);
        attn_o.resize(n_layers);
        ffn_norm.resize(n_layers);
        ffn_gate.resize(n_layers);
        ffn_up.resize(n_layers);
        ffn_down.resize(n_layers);
        
        // 构建名称 → 目标映射
        for (i32 i = 0; i < n_layers; i++) {
            std::string blk = "blk." + std::to_string(i);
            tensor_map[blk + ".attn_norm.weight"]  = &attn_norm[i];
            tensor_map[blk + ".attn_q.weight"]     = &attn_q[i];
            tensor_map[blk + ".attn_k.weight"]     = &attn_k[i];
            tensor_map[blk + ".attn_v.weight"]     = &attn_v[i];
            tensor_map[blk + ".attn_output.weight"]= &attn_o[i];
            tensor_map[blk + ".ffn_norm.weight"]   = &ffn_norm[i];
            tensor_map[blk + ".ffn_gate.weight"]   = &ffn_gate[i];
            tensor_map[blk + ".ffn_up.weight"]     = &ffn_up[i];
            tensor_map[blk + ".ffn_down.weight"]   = &ffn_down[i];
        }
        tensor_map["token_embd.weight"] = &token_embd;
        tensor_map["output_norm.weight"]= &output_norm;
        tensor_map["output.weight"]     = &output;
        
        // 逐一加载张量
        u32 loaded = 0;
        for (auto& info : reader.tensors) {
            auto it = tensor_map.find(info.name);
            if (it == tensor_map.end()) {
                // 跳过未知张量
                continue;
            }
            
            Tensor& dst = *it->second;
            if (!reader.load_tensor(info, dst)) {
                std::cerr << "GGUF: 加载张量失败: " << info.name << std::endl;
                return false;
            }
            
            // 从 token_embd 推断 vocab_size
            if (info.name == "token_embd.weight") {
                vocab_size = dst.rows;
            }
            
            loaded++;
        }
        
        if (vocab_size == 0) vocab_size = 32000;
        
        std::cout << "GGUF: 加载了 " << loaded << " 个张量\n";
        std::cout << "  arch=" << arch << " dim=" << dim << " layers=" << n_layers
                  << " heads=" << n_heads << " kv_heads=" << n_kv_heads
                  << " ff=" << ff_dim << " vocab=" << vocab_size << std::endl;
        
        return loaded >= (u32)(3 + 9 * n_layers);  // token_embd + output_norm + output + 9*layers
    }
    
    // 将权重注入到 ds4pro Transformer
    void inject(Transformer& model) {
        // Token embedding
        std::cout << "  注入 token_embd: " << token_embd.rows << "x" << token_embd.cols << "\n";
        model.token_embedding = std::move(token_embd);
        
        // 逐层注入
        for (i32 l = 0; l < n_layers && l < model.n_layers; l++) {
            auto& layer = *model.layers[l];
            auto& attn  = layer.attention;
            
            // Attention weights
            attn.q_proj.weight = std::move(attn_q[l]);
            attn.k_proj.weight = std::move(attn_k[l]);
            attn.v_proj.weight = std::move(attn_v[l]);
            attn.o_proj.weight = std::move(attn_o[l]);
            
            // RMS Norm weights — need to reshape [dim] → [1, dim]
            layer.attn_norm.weight = Tensor(1, dim);
            std::copy(attn_norm[l].data.begin(), attn_norm[l].data.end(),
                      layer.attn_norm.weight.data.begin());
            
            // FFN weights
            if (layer.ffn) {
                layer.ffn->gate_proj.weight = std::move(ffn_gate[l]);
                layer.ffn->up_proj.weight   = std::move(ffn_up[l]);
                layer.ffn->down_proj.weight = std::move(ffn_down[l]);
            }
            
            layer.ffn_norm.weight = Tensor(1, dim);
            std::copy(ffn_norm[l].data.begin(), ffn_norm[l].data.end(),
                      layer.ffn_norm.weight.data.begin());
        }
        
        // Final norm
        model.final_norm.weight = Tensor(1, dim);
        std::copy(output_norm.data.begin(), output_norm.data.end(),
                  model.final_norm.weight.data.begin());
        
        // LM head
        model.lm_head.weight = std::move(output);
        
        std::cout << "  权重注入完成\n";
    }
};

// ============================================================
//  GGUF Tokenizer 构建器
// ============================================================
inline BPETokenizer build_tokenizer_from_gguf(GGUFReader& reader) {
    BPETokenizer tok;
    
    // 读取 tokenizer 元数据
    auto& arr = reader.metadata_arrays;
    
    if (arr.empty()) {
        // 回退到语料驱动 tokenizer
        return BPETokenizer::build_corpus();
    }
    
    i32 n_tokens = (i32)arr.size();
    
    // 添加所有 token
    for (i32 i = 0; i < n_tokens; i++) {
        tok.add_token(arr[i]);
    }
    
    // 设置特殊 token
    auto get_i32 = [&](const std::string& key, i32 def) -> i32 {
        auto it = reader.metadata_i32.find(key);
        return it != reader.metadata_i32.end() ? it->second : def;
    };
    
    tok.bos_id = get_i32("tokenizer.ggml.bos_token_id", 1);
    tok.eos_id = get_i32("tokenizer.ggml.eos_token_id", 2);
    tok.unk_id = get_i32("tokenizer.ggml.unknown_token_id", 0);
    tok.pad_id = get_i32("tokenizer.ggml.padding_token_id", 0);
    
    return tok;
}

} // namespace ds4pro

#endif // GGUF_LOADER_HPP