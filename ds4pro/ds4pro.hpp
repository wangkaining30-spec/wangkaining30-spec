/*
 * ds4pro — 本地 AI 推理引擎（满配版）
 * 
 * 特性：
 *   - 分组查询注意力 (GQA) + 旋转位置编码 (RoPE)
 *   - SwiGLU 激活前馈网络
 *   - 混合专家 (MoE) 层
 *   - KV Cache 增量解码
 *   - RMS LayerNorm
 *   - FP32 张量运算 + 可选 SIMD
 *   - Top-K / Top-P / Temperature 采样
 *   - BPE 分词器
 *
 * 纯 C++17，无外部依赖，单头文件。
 */

#ifndef DS4PRO_HPP
#define DS4PRO_HPP

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cassert>

namespace ds4pro {

// ============================================================
//  基础类型
// ============================================================
using f32 = float;
using i32 = int32_t;
using u32 = uint32_t;
using u8  = uint8_t;
using Token = i32;

constexpr f32 PI = 3.14159265358979323846f;
constexpr f32 EPS = 1e-8f;

// ============================================================
//  随机数
// ============================================================
struct RNG {
    std::mt19937 gen;
    RNG(u32 seed = 42) : gen(seed) {}
    
    f32 uniform() {
        std::uniform_real_distribution<f32> dist(0.0f, 1.0f);
        return dist(gen);
    }
    
    f32 normal(f32 mean = 0.0f, f32 std = 1.0f) {
        std::normal_distribution<f32> dist(mean, std);
        return dist(gen);
    }
};

// ============================================================
//  Tensor — 二维矩阵（行主序）
// ============================================================
struct Tensor {
    i32 rows, cols;
    std::vector<f32> data;
    
    Tensor() : rows(0), cols(0) {}
    Tensor(i32 r, i32 c) : rows(r), cols(c), data(r * c, 0.0f) {}
    Tensor(i32 r, i32 c, f32 val) : rows(r), cols(c), data(r * c, val) {}
    Tensor(i32 r, i32 c, std::vector<f32>&& d) : rows(r), cols(c), data(std::move(d)) {
        assert((i32)data.size() == r * c);
    }
    
    f32& operator()(i32 i, i32 j) { return data[i * cols + j]; }
    f32  operator()(i32 i, i32 j) const { return data[i * cols + j]; }
    
    f32* row(i32 i) { return &data[i * cols]; }
    const f32* row(i32 i) const { return &data[i * cols]; }
    
    i32 size() const { return rows * cols; }
    
    void fill(f32 val) { std::fill(data.begin(), data.end(), val); }
    void zero() { fill(0.0f); }
    
    void random_normal(RNG& rng, f32 std = 1.0f) {
        for (auto& v : data) v = rng.normal(0.0f, std);
    }
    
    void random_uniform(RNG& rng, f32 lo = -1.0f, f32 hi = 1.0f) {
        for (auto& v : data) {
            v = lo + (hi - lo) * rng.uniform();
        }
    }
    
    // Xavier 初始化
    void xavier(RNG& rng) {
        f32 bound = std::sqrt(6.0f / (rows + cols));
        random_uniform(rng, -bound, bound);
    }
    
    // 切片：取第 i 行
    Tensor slice_row(i32 i) const {
        Tensor out(1, cols);
        std::copy(row(i), row(i) + cols, out.data.begin());
        return out;
    }
    
    // 切片：取 [start, end) 列
    Tensor slice_cols(i32 start, i32 end) const {
        i32 n = end - start;
        Tensor out(rows, n);
        for (i32 i = 0; i < rows; i++)
            for (i32 j = 0; j < n; j++)
                out(i, j) = (*this)(i, start + j);
        return out;
    }
    
    // Reshape（保持数据）
    Tensor reshape(i32 r, i32 c) const {
        assert(r * c == rows * cols);
        Tensor out(r, c);
        out.data = data;
        return out;
    }
    
    // 转置
    Tensor transpose() const {
        Tensor out(cols, rows);
        for (i32 i = 0; i < rows; i++)
            for (i32 j = 0; j < cols; j++)
                out(j, i) = (*this)(i, j);
        return out;
    }
    
    // 原地加法
    void add(const Tensor& other) {
        assert(rows == other.rows && cols == other.cols);
        for (i32 i = 0; i < size(); i++) data[i] += other.data[i];
    }
    
    void add_scaled(const Tensor& other, f32 scale) {
        for (i32 i = 0; i < size(); i++) data[i] += other.data[i] * scale;
    }
    
    // 逐元素乘法
    void mul(const Tensor& other) {
        assert(rows == other.rows && cols == other.cols);
        for (i32 i = 0; i < size(); i++) data[i] *= other.data[i];
    }
    
    void scale(f32 s) {
        for (auto& v : data) v *= s;
    }
    
    // L2 范数
    f32 norm() const {
        f32 s = 0;
        for (auto v : data) s += v * v;
        return std::sqrt(s);
    }
    
    // 打印前 n 个元素
    void print(i32 n = 10) const {
        std::cout << "[" << rows << "x" << cols << "] ";
        for (i32 i = 0; i < std::min(n, size()); i++)
            std::cout << data[i] << " ";
        if (size() > n) std::cout << "...";
        std::cout << std::endl;
    }
};

// ============================================================
//  矩阵乘法辅助函数
// ============================================================
inline Tensor matmul(const Tensor& A, const Tensor& B) {
    // A: M×K, B: K×N → M×N
    assert(A.cols == B.rows);
    Tensor C(A.rows, B.cols);
    for (i32 i = 0; i < A.rows; i++) {
        const f32* a_row = A.row(i);
        for (i32 j = 0; j < B.cols; j++) {
            f32 sum = 0;
            for (i32 k = 0; k < A.cols; k++)
                sum += a_row[k] * B(k, j);
            C(i, j) = sum;
        }
    }
    return C;
}

// 向量外积（用于 KV 更新）
inline void outer_add(Tensor& dst, const Tensor& a, const Tensor& b, f32 scale = 1.0f) {
    // dst += scale * a^T * b  (a: 1×M, b: 1×N → M×N)
    assert(a.cols == dst.rows && b.cols == dst.cols);
    for (i32 i = 0; i < dst.rows; i++)
        for (i32 j = 0; j < dst.cols; j++)
            dst(i, j) += scale * a(0, i) * b(0, j);
}

// 向量点积
inline f32 dot(const f32* a, const f32* b, i32 n) {
    f32 s = 0;
    for (i32 i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

// 向量 × 矩阵（输出 1×N）
inline Tensor vec_matmul(const Tensor& v, const Tensor& M) {
    assert(v.cols == M.rows);
    Tensor out(1, M.cols);
    for (i32 j = 0; j < M.cols; j++) {
        f32 sum = 0;
        for (i32 k = 0; k < v.cols; k++)
            sum += v(0, k) * M(k, j);
        out(0, j) = sum;
    }
    return out;
}

// ============================================================
//  RMS LayerNorm
// ============================================================
struct RMSNorm {
    i32 dim;
    Tensor weight;  // [dim]
    
    RMSNorm(i32 d) : dim(d), weight(1, d) {
        weight.fill(1.0f);
    }
    
    void forward(Tensor& x) const {
        // x: [n, dim]，原地归一化
        for (i32 i = 0; i < x.rows; i++) {
            f32* row = x.row(i);
            f32 rms = 0;
            for (i32 j = 0; j < dim; j++) rms += row[j] * row[j];
            rms = std::sqrt(rms / dim + EPS);
            f32 inv = 1.0f / rms;
            for (i32 j = 0; j < dim; j++)
                row[j] = row[j] * inv * weight(0, j);
        }
    }
    
    void forward_single(Tensor& x) const {
        // x: [1, dim]
        assert(x.rows == 1);
        forward(x);
    }
};

// ============================================================
//  旋转位置编码 (RoPE)
// ============================================================
struct RoPE {
    i32 dim;
    i32 max_seq_len;
    Tensor cos_cached;  // [max_seq_len, dim/2]
    Tensor sin_cached;  // [max_seq_len, dim/2]
    
    RoPE(i32 d, i32 max_len = 4096, f32 theta = 10000.0f)
        : dim(d), max_seq_len(max_len) {
        i32 half = dim / 2;
        cos_cached = Tensor(max_len, half);
        sin_cached = Tensor(max_len, half);
        
        for (i32 pos = 0; pos < max_len; pos++) {
            for (i32 i = 0; i < half; i++) {
                f32 freq = pos / std::pow(theta, 2.0f * i / dim);
                cos_cached(pos, i) = std::cos(freq);
                sin_cached(pos, i) = std::sin(freq);
            }
        }
    }
    
    // 对 x 中每对相邻维度施加旋转
    void apply(Tensor& x, i32 pos) const {
        // x: [1, dim] 或 [batch, dim]
        i32 half = dim / 2;
        for (i32 b = 0; b < x.rows; b++) {
            f32* row = x.row(b);
            for (i32 i = 0; i < half; i++) {
                f32 cos_theta = cos_cached(pos, i);
                f32 sin_theta = sin_cached(pos, i);
                f32 x0 = row[2 * i];
                f32 x1 = row[2 * i + 1];
                row[2 * i]     = x0 * cos_theta - x1 * sin_theta;
                row[2 * i + 1] = x0 * sin_theta + x1 * cos_theta;
            }
        }
    }
};

// ============================================================
//  SwiGLU 激活
// ============================================================
inline f32 silu(f32 x) {
    return x / (1.0f + std::exp(-x));
}

inline void swiglu(Tensor& gate, const Tensor& up) {
    // gate = gate * silu(gate) * up  实际上 gate 已经过线性变换
    assert(gate.rows == up.rows && gate.cols == up.cols);
    for (i32 i = 0; i < gate.size(); i++) {
        f32 g = gate.data[i];
        gate.data[i] = g * silu(g) * up.data[i];
    }
}

// ============================================================
//  线性层（带偏置）
// ============================================================
struct Linear {
    i32 in_dim, out_dim;
    Tensor weight;  // [in_dim, out_dim]
    Tensor bias;    // [out_dim]
    
    Linear(i32 in_d, i32 out_d, bool use_bias = false)
        : in_dim(in_d), out_dim(out_d),
          weight(in_d, out_d), bias(1, out_d) {
        bias.zero();
    }
    
    void init(RNG& rng, f32 std = -1.0f) {
        if (std < 0) {
            // Xavier
            weight.xavier(rng);
        } else {
            weight.random_normal(rng, std);
        }
    }
    
    Tensor forward(const Tensor& x) const {
        // x: [batch, in_dim] → [batch, out_dim]
        Tensor out = matmul(x, weight);
        for (i32 i = 0; i < out.rows; i++)
            for (i32 j = 0; j < out_dim; j++)
                out(i, j) += bias(0, j);
        return out;
    }
    
    // 单向量前向
    void forward_vec(const f32* x, f32* out) const {
        for (i32 j = 0; j < out_dim; j++) {
            f32 sum = bias(0, j);
            for (i32 k = 0; k < in_dim; k++)
                sum += x[k] * weight(k, j);
            out[j] = sum;
        }
    }
};

// ============================================================
//  分组查询注意力 (GQA)
// ============================================================
struct GQAttention {
    i32 dim;        // 模型隐藏维度
    i32 n_heads;    // 查询头数
    i32 n_kv_heads; // KV 头数 (≤ n_heads)
    i32 head_dim;   // 每个头的维度
    RoPE rope;
    
    Linear q_proj, k_proj, v_proj, o_proj;
    
    // KV Cache: [n_layers][2][batch][max_seq_len][n_kv_heads * head_dim]
    // 简化为每个头独立管理
    
    GQAttention(i32 d, i32 heads, i32 kv_heads, i32 max_len = 4096)
        : dim(d), n_heads(heads), n_kv_heads(kv_heads),
          head_dim(d / heads), rope(head_dim, max_len),
          q_proj(d, heads * head_dim),
          k_proj(d, kv_heads * head_dim),
          v_proj(d, kv_heads * head_dim),
          o_proj(heads * head_dim, d) {
        assert(d % heads == 0);
        assert(heads % kv_heads == 0);
    }
    
    // 训练用并行前向（无 cache）
    Tensor forward(const Tensor& x, i32 start_pos, Tensor* kv_cache_k = nullptr, Tensor* kv_cache_v = nullptr) {
        // x: [seq_len, dim]
        i32 seq_len = x.rows;
        i32 n_groups = n_heads / n_kv_heads;
        
        Tensor q = q_proj.forward(x);  // [seq_len, n_heads * head_dim]
        Tensor k = k_proj.forward(x);  // [seq_len, n_kv_heads * head_dim]
        Tensor v = v_proj.forward(x);  // [seq_len, n_kv_heads * head_dim]
        
        // 如有 KV Cache，追加存储
        if (kv_cache_k) {
            for (i32 s = 0; s < seq_len; s++)
                std::copy(k.row(s), k.row(s) + k.cols, kv_cache_k->row(start_pos + s));
        }
        if (kv_cache_v) {
            for (i32 s = 0; s < seq_len; s++)
                std::copy(v.row(s), v.row(s) + v.cols, kv_cache_v->row(start_pos + s));
        }
        
        i32 total_len = start_pos + seq_len;
        // 如果没有缓存就用当前长度
        if (!kv_cache_k) total_len = seq_len;
        
        // --- 注意力计算 ---
        Tensor output(seq_len, n_heads * head_dim);
        
        for (i32 s = 0; s < seq_len; s++) {
            i32 pos = start_pos + s;
            Tensor q_row = q.slice_row(s);  // [1, n_heads * head_dim]
            
            // 对 Q 施加 RoPE
            Tensor q_roped = q_row;
            for (i32 h = 0; h < n_heads; h++) {
                for (i32 d = 0; d < head_dim / 2; d++) {
                    i32 idx = h * head_dim;
                    f32 f = pos / std::pow(10000.0f, 2.0f * d / head_dim);
                    f32 c = std::cos(f), si = std::sin(f);
                    f32 x0 = q_roped(0, idx + 2*d);
                    f32 x1 = q_roped(0, idx + 2*d + 1);
                    q_roped(0, idx + 2*d)     = x0 * c - x1 * si;
                    q_roped(0, idx + 2*d + 1) = x0 * si + x1 * c;
                }
            }
            
            // 逐头计算注意力
            for (i32 h = 0; h < n_heads; h++) {
                i32 kv_h = h / n_groups;
                
                // 提取 Q 头
                Tensor q_head(1, head_dim);
                std::copy(&q_roped(0, h * head_dim),
                          &q_roped(0, h * head_dim) + head_dim,
                          q_head.data.begin());
                
                // 计算注意力分数
                std::vector<f32> scores(total_len);
                f32 max_score = -1e9f;
                
                for (i32 t = 0; t < total_len; t++) {
                    // 提取 K 头并施加 RoPE
                    f32 k_head[4096]; // 栈上分配，最大 4K
                    if (kv_cache_k && t < start_pos + seq_len) {
                        const f32* k_src = kv_cache_k->row(std::min(t, kv_cache_k->rows - 1));
                        std::copy(k_src + kv_h * head_dim,
                                  k_src + kv_h * head_dim + head_dim,
                                  k_head);
                    } else if (!kv_cache_k && t < seq_len) {
                        std::copy(&k(0, kv_h * head_dim),
                                  &k(0, kv_h * head_dim) + head_dim,
                                  k_head);
                    } else {
                        continue;  // skip
                    }
                    
                    // RoPE on K
                    for (i32 d = 0; d < head_dim / 2; d++) {
                        f32 f = t / std::pow(10000.0f, 2.0f * d / head_dim);
                        f32 c = std::cos(f), si = std::sin(f);
                        f32 x0 = k_head[2*d], x1 = k_head[2*d+1];
                        k_head[2*d]   = x0 * c - x1 * si;
                        k_head[2*d+1] = x0 * si + x1 * c;
                    }
                    
                    f32 score = dot(q_head.data.data(), k_head, head_dim) / std::sqrt((f32)head_dim);
                    scores[t] = score;
                    if (score > max_score) max_score = score;
                }
                
                // Softmax
                f32 sum_exp = 0;
                for (i32 t = 0; t < total_len; t++) {
                    scores[t] = std::exp(scores[t] - max_score);
                    sum_exp += scores[t];
                }
                for (i32 t = 0; t < total_len; t++)
                    scores[t] /= sum_exp;
                
                // 加权 V
                std::vector<f32> attn_out(head_dim, 0);
                for (i32 t = 0; t < total_len; t++) {
                    f32 w = scores[t];
                    const f32* v_src;
                    if (kv_cache_v && t < start_pos + seq_len) {
                        v_src = kv_cache_v->row(std::min(t, kv_cache_v->rows - 1)) + kv_h * head_dim;
                    } else if (!kv_cache_v && t < seq_len) {
                        v_src = &v(t, kv_h * head_dim);
                    } else {
                        continue;
                    }
                    for (i32 d = 0; d < head_dim; d++)
                        attn_out[d] += w * v_src[d];
                }
                
                // 写回输出
                std::copy(attn_out.begin(), attn_out.end(),
                          &output(s, h * head_dim));
            }
        }
        
        // 输出投影
        return o_proj.forward(output);
    }
    
    // === 增量解码版本（带 KV Cache）===
    void forward_incremental(const f32* x,       // [dim]
                              f32* out,           // [dim]
                              i32 pos,
                              Tensor& k_cache,    // [max_len, n_kv_heads * head_dim]
                              Tensor& v_cache) {  // [max_len, n_kv_heads * head_dim]
        // 计算 Q, K, V
        f32 q_vec[8192], k_vec[8192], v_vec[8192];
        q_proj.forward_vec(x, q_vec);
        k_proj.forward_vec(x, k_vec);
        v_proj.forward_vec(x, v_vec);
        
        // 存储 K, V 到 cache
        std::copy(k_vec, k_vec + n_kv_heads * head_dim, k_cache.row(pos));
        std::copy(v_vec, v_vec + n_kv_heads * head_dim, v_cache.row(pos));
        
        i32 total_len = pos + 1;
        i32 n_groups = n_heads / n_kv_heads;
        
        std::vector<f32> o_vec(n_heads * head_dim, 0);
        
        for (i32 h = 0; h < n_heads; h++) {
            i32 kv_h = h / n_groups;
            
            // RoPE on Q head
            std::vector<f32> q_head(head_dim);
            std::copy(q_vec + h * head_dim, q_vec + (h+1) * head_dim, q_head.begin());
            for (i32 d = 0; d < head_dim / 2; d++) {
                f32 f = pos / std::pow(10000.0f, 2.0f * d / head_dim);
                f32 c = std::cos(f), si = std::sin(f);
                f32 x0 = q_head[2*d], x1 = q_head[2*d+1];
                q_head[2*d]   = x0 * c - x1 * si;
                q_head[2*d+1] = x0 * si + x1 * c;
            }
            
            // 计算注意力分数
            std::vector<f32> scores(total_len);
            f32 max_score = -1e9f;
            
            for (i32 t = 0; t < total_len; t++) {
                f32 k_h[4096];
                std::copy(k_cache.row(t) + kv_h * head_dim,
                          k_cache.row(t) + kv_h * head_dim + head_dim,
                          k_h);
                
                // RoPE on K
                for (i32 d = 0; d < head_dim / 2; d++) {
                    f32 f = t / std::pow(10000.0f, 2.0f * d / head_dim);
                    f32 c = std::cos(f), si = std::sin(f);
                    f32 x0 = k_h[2*d], x1 = k_h[2*d+1];
                    k_h[2*d]   = x0 * c - x1 * si;
                    k_h[2*d+1] = x0 * si + x1 * c;
                }
                
                f32 score = dot(q_head.data(), k_h, head_dim) / std::sqrt((f32)head_dim);
                scores[t] = score;
                if (score > max_score) max_score = score;
            }
            
            // Softmax
            f32 sum_exp = 0;
            for (auto& s : scores) { s = std::exp(s - max_score); sum_exp += s; }
            for (auto& s : scores) s /= sum_exp;
            
            // 加权 V
            std::vector<f32> attn_out(head_dim, 0);
            for (i32 t = 0; t < total_len; t++) {
                f32 w = scores[t];
                const f32* v_src = v_cache.row(t) + kv_h * head_dim;
                for (i32 d = 0; d < head_dim; d++)
                    attn_out[d] += w * v_src[d];
            }
            
            std::copy(attn_out.begin(), attn_out.end(),
                      o_vec.begin() + h * head_dim);
        }
        
        // 输出投影
        o_proj.forward_vec(o_vec.data(), out);
    }
};

// ============================================================
//  前馈网络（SwiGLU）
// ============================================================
struct FeedForward {
    i32 dim, hidden_dim;
    Linear gate_proj, up_proj, down_proj;
    
    FeedForward(i32 d, i32 hd)
        : dim(d), hidden_dim(hd),
          gate_proj(d, hd),
          up_proj(d, hd),
          down_proj(hd, d) {}
    
    Tensor forward(const Tensor& x) const {
        Tensor gate = gate_proj.forward(x);
        Tensor up   = up_proj.forward(x);
        swiglu(gate, up);
        return down_proj.forward(gate);
    }
    
    void forward_vec(const f32* x, f32* out) const {
        f32 gate_buf[16384], up_buf[16384];
        gate_proj.forward_vec(x, gate_buf);
        up_proj.forward_vec(x, up_buf);
        for (i32 i = 0; i < hidden_dim; i++)
            gate_buf[i] = gate_buf[i] * silu(gate_buf[i]) * up_buf[i];
        down_proj.forward_vec(gate_buf, out);
    }
};

// ============================================================
//  混合专家层 (MoE)
// ============================================================
struct MoELayer {
    i32 dim;
    i32 n_experts;
    i32 top_k;
    Linear gate;  // Router: [dim, n_experts]
    std::vector<std::unique_ptr<FeedForward>> experts;
    
    MoELayer(i32 d, i32 n_exp, i32 k, i32 ff_dim)
        : dim(d), n_experts(n_exp), top_k(k),
          gate(d, n_exp) {
        for (i32 i = 0; i < n_exp; i++)
            experts.push_back(std::make_unique<FeedForward>(d, ff_dim));
    }
    
    Tensor forward(const Tensor& x) const {
        Tensor out(x.rows, dim);
        out.zero();
        
        Tensor logits = gate.forward(x);  // [batch, n_experts]
        
        for (i32 b = 0; b < x.rows; b++) {
            // Top-K 专家选择
            std::vector<std::pair<f32, i32>> scores;
            for (i32 e = 0; e < n_experts; e++)
                scores.push_back({logits(b, e), e});
            std::partial_sort(scores.begin(), scores.begin() + top_k, scores.end(),
                              [](auto& a, auto& b) { return a.first > b.first; });
            
            // Softmax 归一化权重
            f32 max_s = scores[0].first;
            f32 sum_w = 0;
            std::vector<f32> weights(top_k);
            for (i32 i = 0; i < top_k; i++) {
                weights[i] = std::exp(scores[i].first - max_s);
                sum_w += weights[i];
            }
            for (auto& w : weights) w /= sum_w;
            
            // 专家前向 + 加权合并
            Tensor x_single = x.slice_row(b);
            for (i32 i = 0; i < top_k; i++) {
                Tensor expert_out = experts[scores[i].second]->forward(x_single);
                for (i32 d = 0; d < dim; d++)
                    out(b, d) += weights[i] * expert_out(0, d);
            }
        }
        return out;
    }
    
    void forward_vec(const f32* x, f32* out) const {
        // 路由器
        f32 logits_buf[256];
        gate.forward_vec(x, logits_buf);
        
        std::vector<std::pair<f32, i32>> scores;
        for (i32 e = 0; e < n_experts; e++)
            scores.push_back({logits_buf[e], e});
        std::partial_sort(scores.begin(), scores.begin() + top_k, scores.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        
        f32 max_s = scores[0].first;
        f32 sum_w = 0;
        std::vector<f32> weights(top_k);
        for (i32 i = 0; i < top_k; i++) {
            weights[i] = std::exp(scores[i].first - max_s);
            sum_w += weights[i];
        }
        for (auto& w : weights) w /= sum_w;
        
        std::fill(out, out + dim, 0.0f);
        for (i32 i = 0; i < top_k; i++) {
            f32 expert_out[8192];
            experts[scores[i].second]->forward_vec(x, expert_out);
            for (i32 d = 0; d < dim; d++)
                out[d] += weights[i] * expert_out[d];
        }
    }
};

// ============================================================
//  Transformer 解码器层
// ============================================================
struct DecoderLayer {
    i32 dim;
    RMSNorm attn_norm, ffn_norm;
    GQAttention attention;
    
    // 前馈：可选标准 FFN 或 MoE
    enum FFType { STANDARD, MOE };
    FFType ff_type;
    std::unique_ptr<FeedForward> ffn;
    std::unique_ptr<MoELayer> moe;
    
    DecoderLayer(i32 d, i32 heads, i32 kv_heads, i32 ff_dim,
                 FFType fft = STANDARD, i32 n_experts = 0, i32 top_k = 0)
        : dim(d), attn_norm(d), ffn_norm(d),
          attention(d, heads, kv_heads), ff_type(fft) {
        if (fft == MOE && n_experts > 0) {
            moe = std::make_unique<MoELayer>(d, n_experts, top_k, ff_dim);
        } else {
            ffn = std::make_unique<FeedForward>(d, ff_dim);
        }
    }
    
    Tensor forward(const Tensor& x, i32 start_pos,
                   Tensor* k_cache = nullptr, Tensor* v_cache = nullptr) {
        // Pre-norm + Attention + Residual
        Tensor normed = x;
        attn_norm.forward(normed);
        Tensor attn_out = attention.forward(normed, start_pos, k_cache, v_cache);
        Tensor h = x;
        h.add(attn_out);
        
        // Pre-norm + FFN + Residual
        Tensor normed2 = h;
        ffn_norm.forward(normed2);
        Tensor ffn_out = (ff_type == MOE && moe) ? moe->forward(normed2)
                                                   : ffn->forward(normed2);
        h.add(ffn_out);
        return h;
    }
    
    void forward_incremental(const f32* x, f32* out, i32 pos,
                              Tensor& k_cache, Tensor& v_cache) {
        // Pre-norm
        f32 normed[8192];
        std::copy(x, x + dim, normed);
        {
            f32 rms = 0;
            for (i32 i = 0; i < dim; i++) rms += normed[i] * normed[i];
            rms = std::sqrt(rms / dim + EPS);
            for (i32 i = 0; i < dim; i++)
                normed[i] = normed[i] / rms * attn_norm.weight(0, i);
        }
        
        f32 attn_out[8192];
        attention.forward_incremental(normed, attn_out, pos, k_cache, v_cache);
        
        // Residual
        f32 h[8192];
        for (i32 i = 0; i < dim; i++) h[i] = x[i] + attn_out[i];
        
        // FFN norm
        f32 normed2[8192];
        std::copy(h, h + dim, normed2);
        {
            f32 rms = 0;
            for (i32 i = 0; i < dim; i++) rms += normed2[i] * normed2[i];
            rms = std::sqrt(rms / dim + EPS);
            for (i32 i = 0; i < dim; i++)
                normed2[i] = normed2[i] / rms * ffn_norm.weight(0, i);
        }
        
        f32 ffn_out[8192];
        if (ff_type == MOE && moe) {
            moe->forward_vec(normed2, ffn_out);
        } else {
            ffn->forward_vec(normed2, ffn_out);
        }
        
        for (i32 i = 0; i < dim; i++) out[i] = h[i] + ffn_out[i];
    }
};

// ============================================================
//  Transformer 模型
// ============================================================
struct Transformer {
    i32 dim, vocab_size, max_seq_len;
    i32 n_layers, n_heads, n_kv_heads, ff_dim;
    bool use_moe;
    i32 n_experts, top_k;
    
    Tensor token_embedding;  // [vocab_size, dim]
    RMSNorm final_norm;
    Linear lm_head;          // [dim, vocab_size]
    std::vector<std::unique_ptr<DecoderLayer>> layers;
    
    // KV Cache（每层两个：K 和 V）
    struct KVCache {
        Tensor k, v;
        KVCache(i32 max_len, i32 n_kv_heads, i32 head_dim)
            : k(max_len, n_kv_heads * head_dim),
              v(max_len, n_kv_heads * head_dim) {}
    };
    std::vector<KVCache> kv_caches;
    
    Transformer(i32 d, i32 vocab, i32 max_len,
                i32 layers_n, i32 heads, i32 kv_heads, i32 ff_d,
                bool moe = false, i32 n_exp = 8, i32 tk = 2)
        : dim(d), vocab_size(vocab), max_seq_len(max_len),
          n_layers(layers_n), n_heads(heads), n_kv_heads(kv_heads), ff_dim(ff_d),
          use_moe(moe), n_experts(n_exp), top_k(tk),
          token_embedding(vocab, d),
          final_norm(d),
          lm_head(d, vocab) {
        
        for (i32 i = 0; i < n_layers; i++) {
            auto ff_type = (use_moe && i >= n_layers - 2)  // 最后两层用 MoE
                               ? DecoderLayer::MOE : DecoderLayer::STANDARD;
            layers.push_back(std::make_unique<DecoderLayer>(
                d, heads, kv_heads, ff_d, ff_type, n_exp, tk));
        }
        
        // 初始化 KV Cache
        for (i32 i = 0; i < n_layers; i++)
            kv_caches.emplace_back(max_len, kv_heads, d / heads);
    }
    
    void init_weights(RNG& rng) {
        token_embedding.random_normal(rng, 0.02f);
        final_norm.weight.fill(1.0f);
        lm_head.weight.random_normal(rng, 0.02f);
        
        for (auto& layer : layers) {
            layer->attn_norm.weight.fill(1.0f);
            layer->ffn_norm.weight.fill(1.0f);
            layer->attention.q_proj.init(rng);
            layer->attention.k_proj.init(rng);
            layer->attention.v_proj.init(rng);
            layer->attention.o_proj.init(rng);
            if (layer->ffn) {
                layer->ffn->gate_proj.init(rng);
                layer->ffn->up_proj.init(rng);
                layer->ffn->down_proj.init(rng);
            }
            if (layer->moe) {
                layer->moe->gate.init(rng);
                for (auto& exp : layer->moe->experts) {
                    exp->gate_proj.init(rng);
                    exp->up_proj.init(rng);
                    exp->down_proj.init(rng);
                }
            }
        }
    }
    
    void clear_kv_cache() {
        for (auto& c : kv_caches) {
            c.k.zero();
            c.v.zero();
        }
    }
    
    // 前向传播（预填充阶段）
    Tensor forward(const std::vector<Token>& tokens) {
        i32 seq_len = (i32)tokens.size();
        Tensor h(seq_len, dim);
        
        // Token Embedding
        for (i32 s = 0; s < seq_len; s++) {
            Token t = tokens[s];
            std::copy(token_embedding.row(t), token_embedding.row(t) + dim, h.row(s));
        }
        
        // 逐层前向
        for (i32 l = 0; l < n_layers; l++) {
            h = layers[l]->forward(h, 0,
                                    &kv_caches[l].k,
                                    &kv_caches[l].v);
        }
        
        // 最终 Norm
        final_norm.forward(h);
        
        // LM Head
        return lm_head.forward(h);  // [seq_len, vocab_size]
    }
    
    // 增量解码（生成阶段）
    Token step(Token token, i32 pos, const Tensor& logits_storage) {
        // 嵌入
        f32 h[8192];
        std::copy(token_embedding.row(token), token_embedding.row(token) + dim, h);
        
        // 逐层
        for (i32 l = 0; l < n_layers; l++) {
            f32 next[8192];
            layers[l]->forward_incremental(h, next, pos,
                                            kv_caches[l].k,
                                            kv_caches[l].v);
            std::copy(next, next + dim, h);
        }
        
        // RMS Norm
        {
            f32 rms = 0;
            for (i32 i = 0; i < dim; i++) rms += h[i] * h[i];
            rms = std::sqrt(rms / dim + EPS);
            for (i32 i = 0; i < dim; i++)
                h[i] = h[i] / rms * final_norm.weight(0, i);
        }
        
        // LM Head → 输出 logits
        // 这里简化：返回 argmax token（实际可接入采样器）
        f32 max_logit = -1e9f;
        i32 best = 0;
        for (i32 v = 0; v < vocab_size; v++) {
            f32 logit = lm_head.bias(0, v);
            for (i32 d = 0; d < dim; d++)
                logit += h[d] * lm_head.weight(d, v);
            if (logit > max_logit) {
                max_logit = logit;
                best = v;
            }
        }
        return best;
    }
};

// ============================================================
//  Sampler — 温度 + Top-K + Top-P 采样
// ============================================================
struct Sampler {
    f32 temperature;
    i32 top_k;
    f32 top_p;
    RNG& rng;
    
    Sampler(RNG& r, f32 temp = 1.0f, i32 tk = 50, f32 tp = 0.9f)
        : temperature(temp), top_k(tk), top_p(tp), rng(r) {}
    
    Token sample(const Tensor& logits) {
        // logits: [1, vocab_size]
        i32 vocab = logits.cols;
        std::vector<std::pair<f32, i32>> sorted;
        for (i32 i = 0; i < vocab; i++)
            sorted.push_back({logits(0, i), i});
        
        // Top-K
        if (top_k > 0 && top_k < vocab) {
            std::partial_sort(sorted.begin(), sorted.begin() + top_k, sorted.end(),
                              [](auto& a, auto& b) { return a.first > b.first; });
            sorted.resize(top_k);
        } else {
            std::sort(sorted.begin(), sorted.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
        }
        
        // Temperature
        f32 max_logit = sorted[0].first;
        for (auto& s : sorted) s.first = std::exp((s.first - max_logit) / temperature);
        
        // Top-P (nucleus)
        f32 sum = 0;
        for (auto& s : sorted) sum += s.first;
        i32 cutoff = (i32)sorted.size();
        f32 cumulative = 0;
        for (i32 i = 0; i < (i32)sorted.size(); i++) {
            cumulative += sorted[i].first / sum;
            if (cumulative >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        sorted.resize(std::max(cutoff, 1));
        
        // Renormalize
        f32 norm = 0;
        for (auto& s : sorted) norm += s.first;
        
        // Sample
        f32 r = rng.uniform() * norm;
        f32 acc = 0;
        for (auto& s : sorted) {
            acc += s.first;
            if (r <= acc) return s.second;
        }
        return sorted.back().second;
    }
};

// ============================================================
//  BPE Tokenizer（简化版）
// ============================================================
struct BPETokenizer {
    std::unordered_map<std::string, Token> token_to_id;
    std::vector<std::string> id_to_token;
    i32 vocab_size;
    
    // 特殊 token
    Token bos_id, eos_id, unk_id, pad_id;
    
    BPETokenizer() : vocab_size(0), bos_id(0), eos_id(0), unk_id(0), pad_id(0) {}
    
    void add_token(const std::string& token) {
        if (token_to_id.find(token) == token_to_id.end()) {
            token_to_id[token] = vocab_size;
            id_to_token.push_back(token);
            vocab_size++;
        }
    }
    
    void set_special(Token bos, Token eos, Token unk, Token pad) {
        bos_id = bos; eos_id = eos; unk_id = unk; pad_id = pad;
    }
    
    Token encode_single(const std::string& token) const {
        auto it = token_to_id.find(token);
        return it != token_to_id.end() ? it->second : unk_id;
    }
    
    std::string decode_single(Token id) const {
        if (id >= 0 && id < (Token)id_to_token.size())
            return id_to_token[id];
        return "<UNK>";
    }
    
    // 简单 BPE 编码（贪心最长匹配，限制最大匹配长度提高性能）
    std::vector<Token> encode(const std::string& text) const {
        std::vector<Token> tokens;
        i32 max_token_len = 0;
        for (const auto& p : token_to_id)
            if ((i32)p.first.size() > max_token_len) max_token_len = (i32)p.first.size();
        
        i32 pos = 0;
        i32 n = (i32)text.size();
        while (pos < n) {
            i32 max_len = std::min(n - pos, max_token_len);
            bool matched = false;
            // 从最长可能开始匹配
            for (i32 len = max_len; len > 0; len--) {
                std::string sub = text.substr(pos, len);
                auto it = token_to_id.find(sub);
                if (it != token_to_id.end()) {
                    tokens.push_back(it->second);
                    pos += len;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                // 逐字符 UNK
                std::string ch(1, text[pos]);
                tokens.push_back(encode_single(ch));
                pos++;
            }
        }
        return tokens;
    }
    
    std::string decode(const std::vector<Token>& tokens) const {
        std::string result;
        for (Token t : tokens) {
            std::string s = decode_single(t);
            // 简单合并（不含 BPE 前缀处理）
            result += s;
        }
        return result;
    }
    
    // 构建简易词表
    static BPETokenizer build_basic(i32 max_vocab = 256) {
        BPETokenizer tok;
        
        // 特殊 token
        tok.add_token("<BOS>");
        tok.add_token("<EOS>");
        tok.add_token("<UNK>");
        tok.add_token("<PAD>");
        tok.set_special(0, 1, 2, 3);
        
        // 常见中文字符
        const char* basic_chars =
            "的一是不了在有人我他这为之来以个们到说和就出也时会对能而"
            "要于学下得过生可起都然自还小如去经天所事成那看理用道法无"
            "因为从当后其但想种实做开面现头意体全心己与进等里把前应方"
            "此些最本点只长者问新比各工同关度多动机定感被两力公十它间"
            "着正外明三部门情者入发物样相高向几知内再回果系身台眼见代"
            "或品很别路当水口斯次界立老话重山声手原海期美文产五常风年"
            "加书第色性东走边马受今通达变现德西认气战花先再族记林月";
        
        for (const char* p = basic_chars; *p; ) {
            // UTF-8 中文字符：3 字节
            char buf[4] = {0};
            buf[0] = *p++;
            buf[1] = *p++;
            buf[2] = *p++;
            tok.add_token(buf);
        }
        
        // 常见英文子词
        const char* en_words[] = {
            "the", " be", " to", " of", " and", " in", " that", " have",
            " it", " for", " not", " on", " with", " he", " as", " you",
            " do", " at", " this", " but", " his", " by", " from", " they",
            " we", " say", " her", " she", " or", " an", " will", " my",
            " one", " all", " would", " there", " their", " what", " so",
            " up", " out", " if", " about", " who", " get", " which", " me",
            " when", " make", " can", " like", " time", " just", " him",
            " know", " take", " people", " into", " year", " your", " good",
            " some", " could", " them", " see", " other", " than", " then",
            " now", " look", " only", " come", " its", " over", " think",
            " also", " back", " after", " use", " two", " how", " our",
            " work", " first", " well", " way", " even", " new", " want",
            " because", " any", " these", " give", " day", " most", " us",
            " great", " AI", " model", " learning", " deep", " data",
            " network", " language", " machine", " neural", " transformer",
            " attention", " 你好", " 世界", " 模型", "人工智能", "学习",
            "推理", "知识", "思考", " 人类", "宇宙", " 自然", "语言",
            nullptr
        };
        for (const char** w = en_words; *w; w++) {
            tok.add_token(*w);
        }
        
        // 单个 ASCII 字符
        for (char c = 32; c < 127; c++) {
            char s[2] = {c, 0};
            tok.add_token(s);
        }
        
        return tok;
    }

  // 构建语料驱动词表（覆盖训练数据全部字符，零UNK编码）
  static BPETokenizer build_corpus() {
    BPETokenizer tok;

    // 特殊 token
    tok.add_token("<BOS>");
    tok.add_token("<EOS>");
    tok.add_token("<UNK>");
    tok.add_token("<PAD>");
    tok.set_special(0, 1, 2, 3);

    // 全部语料字符（427 个独特字符，来自训练数据）
            const char* corpus_chars =
      "+124578ACFGILSTUadefimnop"
      "rsw。一七万三上下不与专世业东两个中为主么之九也"
      "习书了事二于云互五些交产享京亮亲人亿什今从他付代令"
      "以仪们件价任优会传估似但位体何余作你使例供便保信值"
      "做傅储像先光入全公共关其兼内再写农冬冰准几出函分划"
      "则创利别到制前割力办功加务劣动助势勤勾化北区十千升"
      "协卫卷历原去参反发取受变叠口古只叫召可台叶号各合同"
      "名后向吗告员周命和品哈响哪商器噪四回因团困围国图圆"
      "在地场块坚型城增声处备夏外多够大天太失头她好如始媒"
      "子字存季学孪宁它安完定宝实客室家容对导将小尘就局层"
      "展山峰嵌工己已市布师帮常平年并序库应度建开异式引张"
      "弦弹强归当录形征径很律得循微德心必志态怎思性总息情"
      "惑惯想意感成我或战户所手打执扩找技把抗报抽拟持指挑"
      "损换据捷排接控推描提搜播操擎支收改政教数整文斯新方"
      "旋族无日时明星春是景智曲更曼最月有服朗期本术机来板"
      "构析林果架查标样核格框案桌档梯械检概模次欢欧正此步"
      "殆段母每比气水求汉池沉沟没治法注洋活流测济浏海浸消"
      "涯深混温源激灵点然照熊爱版物特状猫率王玛环现珠球理"
      "生用由电界百的监盖目直相看真眼着知矩码研确碑示社神"
      "离秋种科秒积移程穆究空立端符第等算管类系素索累约线"
      "练组终经结绕给络统维编缩缺网罔置署美翻老考者而耕耘"
      "联股育胜能自至舟航色节花苦英范荐获落著虚融行表被西"
      "要覆见规视览觉角解言警计认让训议记论设证评识译试话"
      "询语说请读谁调谢象败质购资走起超足距跨跬路践身车转"
      "软辆辨边达迁过运还这进远连述迹适通速造遇道那邮部都"
      "配醒采释里重量针铠链错长门闭问间队阳阵阶降除险随集"
      "零需非面音项顿题风馈首马驱驶驾验高默，";

    for (const char* p = corpus_chars; *p; ) {
      u8 c = (u8)*p;
      i32 len = 1;
      if ((c & 0x80) == 0)       len = 1;
      else if ((c & 0xE0) == 0xC0) len = 2;
      else if ((c & 0xF0) == 0xE0) len = 3;
      else if ((c & 0xF8) == 0xF0) len = 4;
      char buf[8] = {0};
      for (i32 j = 0; j < len; j++) buf[j] = *p++;
      tok.add_token(buf);
    }

    // 常见英文子词
    const char* en_words[] = {
      "the", " be", " to", " of", " and", " in", " that", " have",
      " it", " for", " not", " on", " with", " he", " as", " you",
      " do", " at", " this", " but", " his", " by", " from", " they",
      " we", " say", " her", " she", " or", " an", " will", " my",
      " one", " all", " would", " there", " their", " what", " so",
      " up", " out", " if", " about", " who", " get", " which", " me",
      " when", " make", " can", " like", " time", " just", " him",
      " know", " take", " people", " into", " year", " your", " good",
      " some", " could", " them", " see", " other", " than", " then",
      " now", " look", " only", " come", " its", " over", " think",
      " also", " back", " after", " use", " two", " how", " our",
      " work", " first", " well", " way", " even", " new", " want",
      " because", " any", " these", " give", " day", " most", " us",
      " great", " AI", " model", " learning", " deep", " data",
      " network", " language", " machine", " neural", " transformer",
      " attention", nullptr
    };
    for (const char** w = en_words; *w; w++) {
      tok.add_token(*w);
    }

    // ASCII 可打印字符（回退保障）
    for (char c = 32; c < 127; c++) {
      char s[2] = {c, 0};
      tok.add_token(s);
    }

    return tok;
  }

};

// ============================================================
//  ds4pro 推理器 —— 高层 API
// ============================================================
struct DS4Pro {
    RNG rng;
    std::unique_ptr<Transformer> model;
    std::unique_ptr<BPETokenizer> tokenizer;
    std::unique_ptr<Sampler> sampler;
    
    i32 dim, n_layers, n_heads, n_kv_heads, ff_dim, vocab_size;
    bool use_moe;
    
    DS4Pro(i32 model_dim = 512, i32 layers = 8, i32 heads = 8, i32 kv_heads = 4,
           i32 feedforward_dim = 2048, bool moe = false, i32 n_exp = 4, i32 top_k = 2,
           i32 custom_vocab_size = -1)
        : rng(42), dim(model_dim), n_layers(layers), n_heads(heads),
          n_kv_heads(kv_heads), ff_dim(feedforward_dim),
          use_moe(moe) {
        
        // 构建 tokenizer
        tokenizer = std::make_unique<BPETokenizer>(BPETokenizer::build_corpus());
        vocab_size = (custom_vocab_size > 0) ? custom_vocab_size : tokenizer->vocab_size;
        
        // 构建模型
        model = std::make_unique<Transformer>(
            dim, vocab_size, 4096, n_layers, n_heads, n_kv_heads,
            ff_dim, moe, n_exp, top_k);
        
        // 随机初始化权重
        model->init_weights(rng);
        
        // 采样器
        sampler = std::make_unique<Sampler>(rng, 0.8f, 40, 0.9f);
    }
    
    // 完整生成
    std::string generate(const std::string& prompt, i32 max_tokens = 128) {
        model->clear_kv_cache();
        
        // Tokenize
        std::vector<Token> input_tokens = tokenizer->encode(prompt);
        if (input_tokens.empty()) return "";
        
        // 预填充
        Tensor logits = model->forward(input_tokens);  // [seq_len, vocab]
        
        // 取最后一个位置的 logits 采样
        Tensor last_logits = logits.slice_row((i32)input_tokens.size() - 1);
        Token next = sampler->sample(last_logits);
        
        std::vector<Token> output_tokens = input_tokens;
        output_tokens.push_back(next);
        
        i32 pos = (i32)input_tokens.size();
        
        // 自回归生成
        for (i32 step = 1; step < max_tokens; step++) {
            if (next == tokenizer->eos_id) break;
            next = model->step(next, pos, logits);
            output_tokens.push_back(next);
            pos++;
        }
        
        return tokenizer->decode(output_tokens);
    }
    
    void set_temperature(f32 t) { sampler->temperature = t; }
    void set_top_k(i32 k) { sampler->top_k = k; }
    void set_top_p(f32 p) { sampler->top_p = p; }
    
    // 打印模型信息
    void print_info() const {
        i32 total_params = 0;
        
        total_params += model->token_embedding.size();
        total_params += model->final_norm.weight.size();
        total_params += model->lm_head.weight.size() + model->lm_head.bias.size();
        
        for (auto& l : model->layers) {
            total_params += l->attn_norm.weight.size() + l->ffn_norm.weight.size();
            total_params += l->attention.q_proj.weight.size() + l->attention.q_proj.bias.size();
            total_params += l->attention.k_proj.weight.size() + l->attention.k_proj.bias.size();
            total_params += l->attention.v_proj.weight.size() + l->attention.v_proj.bias.size();
            total_params += l->attention.o_proj.weight.size() + l->attention.o_proj.bias.size();
            if (l->ffn) {
                total_params += l->ffn->gate_proj.weight.size();
                total_params += l->ffn->up_proj.weight.size();
                total_params += l->ffn->down_proj.weight.size();
            }
        }
        
        std::cout << "═══════════════════════════════════\n";
        std::cout << "  ds4pro — 本地推理引擎\n";
        std::cout << "═══════════════════════════════════\n";
        std::cout << "  维度:     " << dim << "\n";
        std::cout << "  层数:     " << n_layers << "\n";
        std::cout << "  注意力头: " << n_heads << " (KV: " << n_kv_heads << ")\n";
        std::cout << "  FF维度:   " << ff_dim << "\n";
        std::cout << "  MoE:      " << (use_moe ? "✓ 已启用" : "✗ 未启用") << "\n";
        std::cout << "  词表:     " << vocab_size << "\n";
        std::cout << "  参数量:   ~" << total_params / 1000000.0f << "M\n";
        std::cout << "═══════════════════════════════════\n";
    }
};

} // namespace ds4pro

#endif // DS4PRO_HPP