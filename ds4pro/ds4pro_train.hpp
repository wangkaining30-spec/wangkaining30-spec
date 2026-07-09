/*
 * ds4pro_train.hpp — 训练模块（反向传播 + 优化器 + 损失函数）
 *
 * 包含：
 *   - 梯度张量操作（matmul 反向等）
 *   - AdamW 优化器
 *   - 各层反向传播方法
 *   - 交叉熵损失
 *   - 训练循环 + Checkpoint
 *   - GGUF 模型导出
 */

#ifndef DS4PRO_TRAIN_HPP
#define DS4PRO_TRAIN_HPP

#include "ds4pro.hpp"
#include "gguf_loader.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <chrono>
#include <algorithm>

namespace ds4pro {

// ============================================================
//  激活缓存（保存前向中间值，供反向使用）
// ============================================================
struct FFNCache {
    Tensor gate_pre;   // gate 线性层输出（SwiGLU 之前）
    Tensor up_pre;     // up 线性层输出
    Tensor gate_post;  // SwiGLU 之后（gate * silu(gate) * up）
};

struct AttentionCache {
    Tensor q_pre;      // Q 投影后、RoPE 前  [seq, n_heads*head_dim]
    Tensor k_pre;      // K 投影后、RoPE 前  [seq, n_kv_heads*head_dim]
    Tensor v_pre;      // V 投影后           [seq, n_kv_heads*head_dim]
    Tensor q_roped;    // Q 投影后、RoPE 后  [seq, n_heads*head_dim]
    Tensor k_roped;    // K 投影后、RoPE 后  [seq, n_kv_heads*head_dim]
    std::vector<f32> scores; // [n_heads * seq * total_len] softmax 后注意力权重
    i32 n_heads, n_kv_heads, head_dim, seq_len, total_len;
    Tensor attn_out;   // 注意力输出（o_proj 之前）[seq, n_heads*head_dim]
};

struct LayerCache {
    Tensor x;               // 输入
    Tensor attn_normed;     // attn_norm 之后
    AttentionCache attn;
    Tensor attn_out;        // attention 输出
    Tensor h;               // attn 残差后
    Tensor ffn_normed;      // ffn_norm 之后
    FFNCache ffn;
    Tensor ffn_out;         // FFN 输出
};

struct ForwardCache {
    Tensor h_after_embed;   // embedding 后 [seq, dim]
    std::vector<LayerCache> layers;
    Tensor h_before_final;  // 最终 norm 前
    Tensor h_after_final;   // 最终 norm 后
    Tensor logits;          // [seq, vocab]
};

// ============================================================
//  梯度 Tensor 操作
// ============================================================

// C = A @ B, A[M,K], B[K,N], C[M,N]
// dA[M,K] = dC[M,N] @ B^T[N,K]
// dB[K,N] = A^T[K,M] @ dC[M,N]
inline void matmul_backward(const Tensor& A, const Tensor& B,
                             const Tensor& dC, Tensor& dA, Tensor& dB) {
    i32 M = A.rows, K = A.cols, N = B.cols;
    
    // dA = dC @ B^T
    dA = Tensor(M, K); dA.zero();
    for (i32 i = 0; i < M; i++) {
        for (i32 k = 0; k < K; k++) {
            f32 sum = 0;
            for (i32 j = 0; j < N; j++)
                sum += dC(i, j) * B(k, j);
            dA(i, k) = sum;
        }
    }
    
    // dB = A^T @ dC
    dB = Tensor(K, N); dB.zero();
    for (i32 k = 0; k < K; k++) {
        for (i32 j = 0; j < N; j++) {
            f32 sum = 0;
            for (i32 i = 0; i < M; i++)
                sum += A(i, k) * dC(i, j);
            dB(k, j) = sum;
        }
    }
}

// ============================================================
//  AdamW 优化器
// ============================================================
struct AdamW {
    struct ParamState {
        Tensor m;       // 一阶矩
        Tensor v;       // 二阶矩
        Tensor w_save;  // 原始权重副本（做 weight decay 用）
        Tensor* weight;
        bool has_grad;
        
        ParamState() : weight(nullptr), has_grad(false) {}
        void init(Tensor* w) {
            weight = w;
            m = Tensor(w->rows, w->cols); m.zero();
            v = Tensor(w->rows, w->cols); v.zero();
            w_save = Tensor(w->rows, w->cols);
            std::copy(w->data.begin(), w->data.end(), w_save.data.begin());
            has_grad = true;
        }
    };
    
    f32 lr;
    f32 beta1, beta2, eps, weight_decay;
    i32 step_count;
    std::vector<ParamState> params;
    
    AdamW(f32 learning_rate = 3e-4f, f32 b1 = 0.9f, f32 b2 = 0.999f,
          f32 e = 1e-8f, f32 wd = 0.01f)
        : lr(learning_rate), beta1(b1), beta2(b2), eps(e),
          weight_decay(wd), step_count(0) {}
    
    void register_param(Tensor* w) {
        ParamState ps;
        ps.init(w);
        params.push_back(ps);
    }
    
    // 注册模型所有参数
    void register_model(Transformer& model) {
        register_param(&model.token_embedding);
        register_param(&model.final_norm.weight);
        register_param(&model.lm_head.weight);
        register_param(&model.lm_head.bias);
        
        for (auto& layer : model.layers) {
            register_param(&layer->attn_norm.weight);
            register_param(&layer->ffn_norm.weight);
            register_param(&layer->attention.q_proj.weight);
            register_param(&layer->attention.q_proj.bias);
            register_param(&layer->attention.k_proj.weight);
            register_param(&layer->attention.k_proj.bias);
            register_param(&layer->attention.v_proj.weight);
            register_param(&layer->attention.v_proj.bias);
            register_param(&layer->attention.o_proj.weight);
            register_param(&layer->attention.o_proj.bias);
            if (layer->ffn) {
                register_param(&layer->ffn->gate_proj.weight);
                register_param(&layer->ffn->gate_proj.bias);
                register_param(&layer->ffn->up_proj.weight);
                register_param(&layer->ffn->up_proj.bias);
                register_param(&layer->ffn->down_proj.weight);
                register_param(&layer->ffn->down_proj.bias);
            }
        }
    }
    
    void step() {
        step_count++;
        f32 bias_corr1 = 1.0f - std::pow(beta1, step_count);
        f32 bias_corr2 = 1.0f - std::pow(beta2, step_count);
        
        for (auto& p : params) {
            if (!p.has_grad) continue;
            Tensor& w = *p.weight;       // 此时 w.data 已被 apply_gradients 替换为梯度
            Tensor& w_orig = p.w_save;   // 原始权重副本
            
            for (i32 i = 0; i < w.size(); i++) {
                f32 g = w.data[i];       // 梯度值
                f32 w_val = w_orig.data[i]; // 原始权重
                
                // Adam 动量
                p.m.data[i] = beta1 * p.m.data[i] + (1 - beta1) * g;
                p.v.data[i] = beta2 * p.v.data[i] + (1 - beta2) * g * g;
                
                f32 m_hat = p.m.data[i] / bias_corr1;
                f32 v_hat = p.v.data[i] / bias_corr2;
                
                f32 update = lr * m_hat / (std::sqrt(v_hat) + eps);
                
                // Weight decay（使用原始权重）
                f32 new_w = w_val - update - lr * weight_decay * w_val;
                
                w.data[i] = new_w;          // 更新权重
                w_orig.data[i] = new_w;     // 同步副本
            }
        }
    }
    
    void zero_grad() {}
};

// ============================================================
//  反向传播：SwiGLU
// ============================================================
// 前向：y = x * silu(x) * up
// silu(x) = x * sigmoid(x)
// silu'(x) = silu(x) + sigmoid(x) * (1 - silu(x))
inline f32 silu_grad(f32 x) {
    f32 s = 1.0f / (1.0f + std::exp(-x));  // sigmoid
    f32 silu_x = x * s;
    return silu_x + s * (1.0f - silu_x);
}

// y = gate * silu(gate) * up
// dy/dgate = silu(gate)*up + gate*silu'(gate)*up
// dy/dup = gate * silu(gate)
inline void swiglu_backward(const Tensor& gate_pre, const Tensor& up_pre,
                             const Tensor& dout, Tensor& dgate, Tensor& dup) {
    i32 N = gate_pre.size();
    dgate = Tensor(gate_pre.rows, gate_pre.cols);
    dup = Tensor(up_pre.rows, up_pre.cols);
    
    for (i32 i = 0; i < N; i++) {
        f32 g = gate_pre.data[i];
        f32 u = up_pre.data[i];
        f32 s = 1.0f / (1.0f + std::exp(-g));  // sigmoid(g)
        f32 silu_g = g * s;
        f32 silu_prime = silu_g + s * (1.0f - silu_g);
        
        // y = g * silu(g) * u
        dgate.data[i] = dout.data[i] * (silu_g * u + g * silu_prime * u);
        dup.data[i]    = dout.data[i] * g * silu_g;
    }
}

// ============================================================
//  反向传播：RMSNorm
// ============================================================
// y = x / rms(x) * w, where rms(x) = sqrt(mean(x^2) + eps)
inline void rmsnorm_backward(const Tensor& x, const Tensor& w,
                              const Tensor& dout, Tensor& dx, Tensor& dw) {
    i32 rows = x.rows, dim = x.cols;
    dx = Tensor(rows, dim);
    dw = Tensor(1, dim); dw.zero();
    
    for (i32 r = 0; r < rows; r++) {
        const f32* xr = x.row(r);
        const f32* wr = w.row(0);
        const f32* dor = dout.row(r);
        f32* dxr = dx.row(r);
        f32* dwr = dw.row(0);
        
        // rms
        f32 sum_sq = 0;
        for (i32 d = 0; d < dim; d++) sum_sq += xr[d] * xr[d];
        f32 rms = std::sqrt(sum_sq / dim + EPS);
        f32 inv_rms = 1.0f / rms;
        
        // 预计算
        f32 sum_dout = 0, sum_dout_x = 0;
        for (i32 d = 0; d < dim; d++) {
            f32 y = xr[d] * inv_rms * wr[d];
            sum_dout += dor[d] * wr[d];
            sum_dout_x += dor[d] * wr[d] * xr[d];
        }
        
        for (i32 d = 0; d < dim; d++) {
            f32 x_normed = xr[d] * inv_rms;
            
            // dx
            dxr[d] = dor[d] * wr[d] * inv_rms
                     - x_normed / (dim * rms * rms) * sum_dout_x;
            
            // dw
            dwr[d] += dor[d] * x_normed;
        }
    }
}

// ============================================================
//  反向传播：Linear
// ============================================================
inline void linear_backward(const Tensor& x, const Tensor& weight,
                             const Tensor& dout, Tensor& dx, Tensor& dw, Tensor& db) {
    // x: [batch, in_dim], weight: [in_dim, out_dim], dout: [batch, out_dim]
    i32 batch = x.rows, in_dim = x.cols, out_dim = weight.cols;
    
    // dx = dout @ weight^T
    dx = Tensor(batch, in_dim);
    for (i32 b = 0; b < batch; b++) {
        for (i32 i = 0; i < in_dim; i++) {
            f32 sum = 0;
            for (i32 o = 0; o < out_dim; o++)
                sum += dout(b, o) * weight(i, o);
            dx(b, i) = sum;
        }
    }
    
    // dw = x^T @ dout
    dw = Tensor(in_dim, out_dim); dw.zero();
    for (i32 i = 0; i < in_dim; i++) {
        for (i32 o = 0; o < out_dim; o++) {
            f32 sum = 0;
            for (i32 b = 0; b < batch; b++)
                sum += x(b, i) * dout(b, o);
            dw(i, o) = sum;
        }
    }
    
    // db = sum over batch
    db = Tensor(1, out_dim); db.zero();
    for (i32 b = 0; b < batch; b++)
        for (i32 o = 0; o < out_dim; o++)
            db(0, o) += dout(b, o);
}

// ============================================================
//  交叉熵损失
// ============================================================
inline f32 cross_entropy_loss(const Tensor& logits, const std::vector<Token>& targets) {
    // logits: [seq, vocab], targets: [seq]
    i32 seq = logits.rows, vocab = logits.cols;
    f32 loss = 0;
    
    for (i32 s = 0; s < seq; s++) {
        Token t = targets[s];
        if (t < 0 || t >= vocab) continue;
        
        // Softmax + NLL
        f32 max_logit = -1e9f;
        for (i32 v = 0; v < vocab; v++)
            if (logits(s, v) > max_logit) max_logit = logits(s, v);
        
        f32 sum_exp = 0;
        for (i32 v = 0; v < vocab; v++)
            sum_exp += std::exp(logits(s, v) - max_logit);
        
        f32 log_prob = logits(s, t) - max_logit - std::log(sum_exp);
        loss -= log_prob;
    }
    return loss / seq;
}

inline void cross_entropy_backward(const Tensor& logits, const std::vector<Token>& targets,
                                    Tensor& dlogits) {
    i32 seq = logits.rows, vocab = logits.cols;
    dlogits = Tensor(seq, vocab);
    
    for (i32 s = 0; s < seq; s++) {
        // Softmax
        f32 max_logit = -1e9f;
        for (i32 v = 0; v < vocab; v++)
            if (logits(s, v) > max_logit) max_logit = logits(s, v);
        
        f32 sum_exp = 0;
        for (i32 v = 0; v < vocab; v++)
            sum_exp += std::exp(logits(s, v) - max_logit);
        
        Token t = targets[s];
        for (i32 v = 0; v < vocab; v++) {
            f32 prob = std::exp(logits(s, v) - max_logit) / sum_exp;
            dlogits(s, v) = prob - (v == t ? 1.0f : 0.0f);
        }
    }
    dlogits.scale(1.0f / seq);
}

// ============================================================
//  训练用前向传播（保存中间激活）
// ============================================================

// 带缓存的 Embedding 前向
inline Tensor embed_forward(const Tensor& embedding, const std::vector<Token>& tokens) {
    i32 seq = (i32)tokens.size();
    i32 dim = embedding.cols;
    Tensor h(seq, dim);
    for (i32 s = 0; s < seq; s++)
        std::copy(embedding.row(tokens[s]), embedding.row(tokens[s]) + dim, h.row(s));
    return h;
}

// 带缓存的 Linear 前向
inline Tensor linear_forward_cached(const Linear& layer, const Tensor& x) {
    return layer.forward(x);
}

// 带缓存的 Attention 前向（训练模式，无 KV cache，完整缓存供反向使用）
inline void attention_forward_train(GQAttention& attn, const Tensor& x,
                                     i32 start_pos, AttentionCache& cache,
                                     Tensor& output) {
    i32 seq_len = x.rows;
    i32 n_heads = attn.n_heads;
    i32 n_kv_heads = attn.n_kv_heads;
    i32 head_dim = attn.head_dim;
    i32 n_groups = n_heads / n_kv_heads;
    i32 total_len = (start_pos == 0) ? seq_len : (start_pos + seq_len);
    
    // 缓存维度信息
    cache.n_heads = n_heads;
    cache.n_kv_heads = n_kv_heads;
    cache.head_dim = head_dim;
    cache.seq_len = seq_len;
    cache.total_len = total_len;
    
    // Q/K/V 投影
    Tensor q = attn.q_proj.forward(x);
    Tensor k = attn.k_proj.forward(x);
    Tensor v = attn.v_proj.forward(x);
    
    cache.q_pre = q;
    cache.k_pre = k;
    cache.v_pre = v;
    
    // 预计算 post-RoPE Q 和 K
    cache.q_roped = Tensor(seq_len, n_heads * head_dim);
    cache.k_roped = Tensor(total_len, n_kv_heads * head_dim);
    
    // 对 Q 施加 RoPE（所有 head 同时处理）
    for (i32 s = 0; s < seq_len; s++) {
        i32 pos = start_pos + s;
        for (i32 h = 0; h < n_heads; h++) {
            for (i32 d = 0; d < head_dim / 2; d++) {
                i32 idx = h * head_dim;
                f32 theta = pos / std::pow(10000.0f, 2.0f * d / head_dim);
                f32 c = std::cos(theta), si = std::sin(theta);
                f32 x0 = q(s, idx + 2*d);
                f32 x1 = q(s, idx + 2*d + 1);
                cache.q_roped(s, idx + 2*d)     = x0 * c - x1 * si;
                cache.q_roped(s, idx + 2*d + 1) = x0 * si + x1 * c;
            }
        }
    }
    
    // 对 K 施加 RoPE
    for (i32 t = 0; t < total_len; t++) {
        for (i32 kv_h = 0; kv_h < n_kv_heads; kv_h++) {
            for (i32 d = 0; d < head_dim / 2; d++) {
                i32 idx = kv_h * head_dim;
                f32 theta = t / std::pow(10000.0f, 2.0f * d / head_dim);
                f32 c = std::cos(theta), si = std::sin(theta);
                f32 x0 = k(t, idx + 2*d);
                f32 x1 = k(t, idx + 2*d + 1);
                cache.k_roped(t, idx + 2*d)     = x0 * c - x1 * si;
                cache.k_roped(t, idx + 2*d + 1) = x0 * si + x1 * c;
            }
        }
    }
    
    // 分配 scores: [n_heads * seq_len * total_len]
    cache.scores.resize(n_heads * seq_len * total_len);
    
    output = Tensor(seq_len, n_heads * head_dim);
    f32 inv_sqrt_d = 1.0f / std::sqrt((f32)head_dim);
    
    // 逐 head 计算注意力
    for (i32 h = 0; h < n_heads; h++) {
        i32 kv_h = h / n_groups;
        
        for (i32 s = 0; s < seq_len; s++) {
            // 计算 scores
            f32 max_score = -1e9f;
            i32 score_offset = h * seq_len * total_len + s * total_len;
            
            for (i32 t = 0; t < total_len; t++) {
                // 因果 mask
                if (t > s && start_pos == 0) {
                    cache.scores[score_offset + t] = -1e9f;
                    continue;
                }
                
                f32 score = 0;
                for (i32 d = 0; d < head_dim; d++) {
                    score += cache.q_roped(s, h * head_dim + d)
                           * cache.k_roped(t, kv_h * head_dim + d);
                }
                score *= inv_sqrt_d;
                cache.scores[score_offset + t] = score;
                if (score > max_score) max_score = score;
            }
            
            // Softmax
            f32 sum_exp = 0;
            for (i32 t = 0; t < total_len; t++) {
                f32 sc = cache.scores[score_offset + t];
                if (sc < -1e8f) { cache.scores[score_offset + t] = 0; continue; }
                f32 exp_sc = std::exp(sc - max_score);
                cache.scores[score_offset + t] = exp_sc;
                sum_exp += exp_sc;
            }
            f32 inv_sum = 1.0f / (sum_exp + 1e-12f);
            for (i32 t = 0; t < total_len; t++)
                cache.scores[score_offset + t] *= inv_sum;
            
            // 加权 V
            std::vector<f32> attn_out(head_dim, 0);
            for (i32 t = 0; t < total_len; t++) {
                f32 w = cache.scores[score_offset + t];
                for (i32 d = 0; d < head_dim; d++)
                    attn_out[d] += w * v(t, kv_h * head_dim + d);
            }
            
            for (i32 d = 0; d < head_dim; d++)
                output(s, h * head_dim + d) = attn_out[d];
        }
    }
    
    cache.attn_out = output;
    output = attn.o_proj.forward(output);
}

// 带缓存的 FFN 前向
inline void ffn_forward_train(FeedForward& ffn, const Tensor& x, FFNCache& cache,
                               Tensor& output) {
    Tensor gate = ffn.gate_proj.forward(x);
    Tensor up   = ffn.up_proj.forward(x);
    
    cache.gate_pre = gate;
    cache.up_pre   = up;
    
    // SwiGLU
    for (i32 i = 0; i < gate.size(); i++)
        gate.data[i] = gate.data[i] * silu(gate.data[i]) * up.data[i];
    
    cache.gate_post = gate;
    output = ffn.down_proj.forward(gate);
}

// ============================================================
//  Transformer 训练前向（完整前向 + 缓存）
// ============================================================
inline void transformer_forward_train(Transformer& model,
                                       const std::vector<Token>& tokens,
                                       ForwardCache& cache) {
    i32 seq_len = (i32)tokens.size();
    i32 dim = model.dim;
    
    // Embedding
    cache.h_after_embed = embed_forward(model.token_embedding, tokens);
    Tensor h = cache.h_after_embed;
    
    cache.layers.resize(model.n_layers);
    
    // 逐层
    for (i32 l = 0; l < model.n_layers; l++) {
        auto& layer = *model.layers[l];
        auto& lc = cache.layers[l];
        
        lc.x = h;
        
        // Pre-norm + Attention
        Tensor normed = h;
        layer.attn_norm.forward(normed);
        lc.attn_normed = normed;
        
        attention_forward_train(layer.attention, normed, 0, lc.attn, lc.attn_out);
        
        // Residual
        h = Tensor(seq_len, dim);
        for (i32 s = 0; s < seq_len; s++)
            for (i32 d = 0; d < dim; d++)
                h(s, d) = lc.x(s, d) + lc.attn_out(s, d);
        lc.h = h;
        
        // Pre-norm + FFN
        Tensor normed2 = h;
        layer.ffn_norm.forward(normed2);
        lc.ffn_normed = normed2;
        
        if (layer.ff_type == DecoderLayer::MOE && layer.moe) {
            // MoE forward — simplified without cache
            lc.ffn_out = layer.moe->forward(normed2);
        } else {
            ffn_forward_train(*layer.ffn, normed2, lc.ffn, lc.ffn_out);
        }
        
        // Residual
        h = Tensor(seq_len, dim);
        for (i32 s = 0; s < seq_len; s++)
            for (i32 d = 0; d < dim; d++)
                h(s, d) = lc.h(s, d) + lc.ffn_out(s, d);
    }
    
    cache.h_before_final = h;
    
    // Final norm
    Tensor h_normed = h;
    model.final_norm.forward(h_normed);
    cache.h_after_final = h_normed;
    
    // LM Head
    cache.logits = model.lm_head.forward(h_normed);
}

// ============================================================
//  反向传播：RoPE
// ============================================================
// RoPE 前向：对于每对 (x[2d], x[2d+1]) @ 位置 p，旋转 theta = p / 10000^(2d/D)
//   y[2d]   = x[2d]*cos - x[2d+1]*sin
//   y[2d+1] = x[2d]*sin + x[2d+1]*cos
// 反向：d_x = R(theta)^T @ d_y
//   dx[2d]   = dy[2d]*cos + dy[2d+1]*sin
//   dx[2d+1] = -dy[2d]*sin + dy[2d+1]*cos
inline void rope_backward(const Tensor& dy, i32 pos, i32 head_dim,
                           i32 n_heads, Tensor& dx) {
    // dy: [n_heads * head_dim], dx: preallocated [n_heads * head_dim]
    for (i32 h = 0; h < n_heads; h++) {
        for (i32 d = 0; d < head_dim / 2; d++) {
            f32 theta = pos / std::pow(10000.0f, 2.0f * d / head_dim);
            f32 c = std::cos(theta), si = std::sin(theta);
            i32 idx = h * head_dim;
            f32 dy0 = dy.data[idx + 2*d];
            f32 dy1 = dy.data[idx + 2*d + 1];
            dx.data[idx + 2*d]     = dy0 * c + dy1 * si;
            dx.data[idx + 2*d + 1] = -dy0 * si + dy1 * c;
        }
    }
}

// ============================================================
//  注意力反向传播（完整版）
// ============================================================
// 从 dO（attention 输出梯度，形状 [seq, n_heads*head_dim]）反向传播：
//   → dV, dK_roped, dQ_roped
//   → RoPE 反向 → dK_pre, dQ_pre
//   → K/Q/V projection 反向 → d_normed
inline void attention_backward_full(GQAttention& attn,
                                     const AttentionCache& cache,
                                     const Tensor& dO,        // [seq, n_heads*head_dim]
                                     const Tensor& attn_normed_x, // attn_norm 的输入 [seq, dim]
                                     Tensor& d_attn_normed,   // 输出：对 normed_x 的梯度
                                     Tensor& dq_w, Tensor& dq_b,
                                     Tensor& dk_w, Tensor& dk_b,
                                     Tensor& dv_w, Tensor& dv_b) {
    i32 n_heads = cache.n_heads;
    i32 n_kv_heads = cache.n_kv_heads;
    i32 head_dim = cache.head_dim;
    i32 seq_len = cache.seq_len;
    i32 total_len = cache.total_len;
    i32 n_groups = n_heads / n_kv_heads;
    i32 dim = attn_normed_x.cols;
    f32 inv_sqrt_d = 1.0f / std::sqrt((f32)head_dim);
    
    // 初始化梯度
    Tensor dQ_roped(seq_len, n_heads * head_dim); dQ_roped.zero();
    Tensor dK_roped(total_len, n_kv_heads * head_dim); dK_roped.zero();
    Tensor dV(total_len, n_kv_heads * head_dim); dV.zero();
    
    const Tensor& V = cache.v_pre;      // [total_len, n_kv_heads * head_dim]
    const Tensor& Q_roped = cache.q_roped;
    const Tensor& K_roped = cache.k_roped;
    
    // 逐 head 逐位置计算梯度
    for (i32 h = 0; h < n_heads; h++) {
        i32 kv_h = h / n_groups;
        
        for (i32 s = 0; s < seq_len; s++) {
            i32 score_offset = h * seq_len * total_len + s * total_len;
            
            // dO_h[s]：head h 在位置 s 的输出梯度
            // 计算 dP（softmax 梯度）
            // dP[t] = sum_d dO_h[s,d] * V[t, kv_h*head_dim + d]
            std::vector<f32> dP(total_len, 0);
            for (i32 t = 0; t < total_len; t++) {
                f32 dot = 0;
                for (i32 d = 0; d < head_dim; d++)
                    dot += dO(s, h * head_dim + d) * V(t, kv_h * head_dim + d);
                dP[t] = dot;
            }
            
            // dS = softmax_backward(P, dP)
            // dS[t] = P[t] * (dP[t] - sum_j P[j] * dP[j])
            f32 sum_P_dP = 0;
            for (i32 t = 0; t < total_len; t++)
                sum_P_dP += cache.scores[score_offset + t] * dP[t];
            
            std::vector<f32> dS(total_len, 0);
            for (i32 t = 0; t < total_len; t++) {
                f32 P_t = cache.scores[score_offset + t];
                if (P_t < 1e-12f) continue;  // 因果 mask 或数值零
                dS[t] = P_t * (dP[t] - sum_P_dP);
            }
            
            // dV += sum over query positions of P^T @ dO_h
            // dV[t, kv_h*head_dim + d] += P_s[t] * dO_h[s, d]
            for (i32 t = 0; t < total_len; t++) {
                f32 w = cache.scores[score_offset + t];
                if (w < 1e-12f) continue;
                for (i32 d = 0; d < head_dim; d++)
                    dV(t, kv_h * head_dim + d) += w * dO(s, h * head_dim + d);
            }
            
            // dQ_roped[s, h*head_dim + d] = sum_t dS[t] * K_roped[t, kv_h*head_dim + d] / sqrt(d)
            for (i32 t = 0; t < total_len; t++) {
                f32 ds = dS[t];
                if (std::abs(ds) < 1e-15f) continue;
                for (i32 d = 0; d < head_dim; d++)
                    dQ_roped(s, h * head_dim + d) += ds * K_roped(t, kv_h * head_dim + d) * inv_sqrt_d;
            }
            
            // dK_roped[t, kv_h*head_dim + d] += dS[t] * Q_roped[s, h*head_dim + d] / sqrt(d)
            for (i32 t = 0; t < total_len; t++) {
                f32 ds = dS[t];
                if (std::abs(ds) < 1e-15f) continue;
                for (i32 d = 0; d < head_dim; d++)
                    dK_roped(t, kv_h * head_dim + d) += ds * Q_roped(s, h * head_dim + d) * inv_sqrt_d;
            }
        }
    }
    
    // RoPE 反向：dQ_roped → dQ_pre（对每个位置）
    Tensor dQ_pre(seq_len, n_heads * head_dim);
    for (i32 s = 0; s < seq_len; s++) {
        // 提取 dQ_roped[s] 作为单行
        Tensor dq_s(1, n_heads * head_dim);
        std::copy(&dQ_roped(s, 0), &dQ_roped(s, n_heads * head_dim), dq_s.data.begin());
        Tensor dx_s(1, n_heads * head_dim);
        rope_backward(dq_s, s, head_dim, n_heads, dx_s);
        std::copy(dx_s.data.begin(), dx_s.data.end(), &dQ_pre(s, 0));
    }
    
    // RoPE 反向：dK_roped → dK_pre
    Tensor dK_pre(total_len, n_kv_heads * head_dim);
    for (i32 t = 0; t < total_len; t++) {
        Tensor dk_t(1, n_kv_heads * head_dim);
        std::copy(&dK_roped(t, 0), &dK_roped(t, n_kv_heads * head_dim), dk_t.data.begin());
        Tensor dx_t(1, n_kv_heads * head_dim);
        rope_backward(dk_t, t, head_dim, n_kv_heads, dx_t);
        std::copy(dx_t.data.begin(), dx_t.data.end(), &dK_pre(t, 0));
    }
    
    // Q/K/V projection backward
    // dQ_pre: [seq, n_heads*head_dim], Q = attn_normed_x @ q_proj.weight^T
    // q_proj: [dim, n_heads*head_dim]
    {
        Tensor dx_q, dw_q, db_q;
        linear_backward(attn_normed_x, attn.q_proj.weight, dQ_pre, dx_q, dw_q, db_q);
        dq_w = dw_q; dq_b = db_q;
        d_attn_normed = dx_q;  // 来自 Q 的梯度
    }
    
    // K projection backward
    {
        Tensor dx_k, dw_k, db_k;
        linear_backward(attn_normed_x, attn.k_proj.weight, dK_pre, dx_k, dw_k, db_k);
        dk_w = dw_k; dk_b = db_k;
        // 累加来自 K 的梯度
        for (i32 i = 0; i < d_attn_normed.size(); i++)
            d_attn_normed.data[i] += dx_k.data[i];
    }
    
    // V projection backward
    {
        Tensor dx_v, dw_v, db_v;
        linear_backward(attn_normed_x, attn.v_proj.weight, dV, dx_v, dw_v, db_v);
        dv_w = dw_v; dv_b = db_v;
        // 累加来自 V 的梯度
        for (i32 i = 0; i < d_attn_normed.size(); i++)
            d_attn_normed.data[i] += dx_v.data[i];
    }
}
struct Gradients {
    // Token embedding
    Tensor dembed;
    
    // Final norm
    Tensor dfinal_norm_w;
    
    // LM head
    Tensor dlm_head_w, dlm_head_b;
    
    // Per layer
    struct LayerGrad {
        Tensor dattn_norm_w;
        Tensor dffn_norm_w;
        Tensor dq_w, dq_b;
        Tensor dk_w, dk_b;
        Tensor dv_w, dv_b;
        Tensor do_w, do_b;
        Tensor dgate_w, dgate_b;
        Tensor dup_w, dup_b;
        Tensor ddown_w, ddown_b;
    };
    std::vector<LayerGrad> layers;
    
    void init(Transformer& model) {
        i32 dim = model.dim, vocab = model.vocab_size, ff = model.ff_dim;
        
        dembed = Tensor(vocab, dim); dembed.zero();
        dfinal_norm_w = Tensor(1, dim); dfinal_norm_w.zero();
        dlm_head_w = Tensor(dim, vocab); dlm_head_w.zero();
        dlm_head_b = Tensor(1, vocab); dlm_head_b.zero();
        
        layers.resize(model.n_layers);
        for (i32 l = 0; l < model.n_layers; l++) {
            auto& g = layers[l];
            g.dattn_norm_w = Tensor(1, dim); g.dattn_norm_w.zero();
            g.dffn_norm_w  = Tensor(1, dim); g.dffn_norm_w.zero();
            g.dq_w = Tensor(dim, dim); g.dq_w.zero();
            g.dq_b = Tensor(1, dim); g.dq_b.zero();
            g.dk_w = Tensor(dim, dim/2); g.dk_w.zero();
            g.dk_b = Tensor(1, dim/2); g.dk_b.zero();
            g.dv_w = Tensor(dim, dim/2); g.dv_w.zero();
            g.dv_b = Tensor(1, dim/2); g.dv_b.zero();
            g.do_w = Tensor(dim, dim); g.do_w.zero();
            g.do_b = Tensor(1, dim); g.do_b.zero();
            g.dgate_w = Tensor(dim, ff); g.dgate_w.zero();
            g.dgate_b = Tensor(1, ff); g.dgate_b.zero();
            g.dup_w = Tensor(dim, ff); g.dup_w.zero();
            g.dup_b = Tensor(1, ff); g.dup_b.zero();
            g.ddown_w = Tensor(ff, dim); g.ddown_w.zero();
            g.ddown_b = Tensor(1, dim); g.ddown_b.zero();
        }
    }
};

// 主反向传播函数
inline void transformer_backward(Transformer& model,
                                  const ForwardCache& cache,
                                  const std::vector<Token>& targets,
                                  Gradients& grads,
                                  f32& loss) {
    // 1. 计算损失和 logits 梯度
    loss = cross_entropy_loss(cache.logits, targets);
    Tensor dlogits;
    cross_entropy_backward(cache.logits, targets, dlogits);
    
    // 2. LM head backward
    Tensor d_h_final;
    {
        Tensor dx, dw, db;
        linear_backward(cache.h_after_final, model.lm_head.weight,
                         dlogits, dx, dw, db);
        d_h_final = dx;
        grads.dlm_head_w = dw;
        grads.dlm_head_b = db;
    }
    
    // 3. Final norm backward
    Tensor d_h_before_final;
    {
        Tensor dx, dw;
        rmsnorm_backward(cache.h_before_final, model.final_norm.weight,
                          d_h_final, dx, dw);
        d_h_before_final = dx;
        grads.dfinal_norm_w = dw;
    }
    
    Tensor dh = d_h_before_final;
    i32 seq = (i32)targets.size(), dim = model.dim;
    
    // 4. 逐层反向（从最后一层到第一层）
    for (i32 l = model.n_layers - 1; l >= 0; l--) {
        auto& layer = *model.layers[l];
        auto& lc = cache.layers[l];
        auto& g = grads.layers[l];
        
        // dh 同时流向 FFN 和残差
        Tensor dffn_out = dh;  // 残差连接：直接传递
        Tensor dresidual2 = dh;
        
        // 4a. FFN backward
        Tensor dffn_normed_input;  // 合并 gate/up 的梯度，用于 ffn_norm 反向
        if (layer.ff_type != DecoderLayer::MOE && layer.ffn) {
            // down_proj backward
            Tensor dgate_post, dd_w, dd_b;
            linear_backward(lc.ffn.gate_post, layer.ffn->down_proj.weight,
                             dffn_out, dgate_post, dd_w, dd_b);
            g.ddown_w = dd_w;
            g.ddown_b = dd_b;
            
            // SwiGLU backward
            Tensor dgate_pre, dup_pre;
            swiglu_backward(lc.ffn.gate_pre, lc.ffn.up_pre,
                             dgate_post, dgate_pre, dup_pre);
            
            // gate_proj backward
            Tensor dx_gate;
            {
                Tensor dw, db;
                linear_backward(lc.ffn_normed, layer.ffn->gate_proj.weight,
                                 dgate_pre, dx_gate, dw, db);
                g.dgate_w = dw;
                g.dgate_b = db;
            }
            
            // up_proj backward
            {
                Tensor dx_up, dw, db;
                linear_backward(lc.ffn_normed, layer.ffn->up_proj.weight,
                                 dup_pre, dx_up, dw, db);
                g.dup_w = dw;
                g.dup_b = db;
                // 合并 gate 和 up 的梯度
                dffn_normed_input = Tensor(dx_gate.rows, dx_gate.cols);
                for (i32 i = 0; i < dx_gate.size(); i++)
                    dffn_normed_input.data[i] = dx_gate.data[i] + dx_up.data[i];
            }
        }
        
        // ffn_norm backward（使用合并后的 gate/up 梯度）
        Tensor dffn_normed;
        {
            Tensor dx, dw;
            rmsnorm_backward(lc.h, layer.ffn_norm.weight,
                              dffn_normed_input, dx, dw);
            dffn_normed = dx;
            g.dffn_norm_w = dw;
        }
        
        // 合并 FFN 和残差的梯度
        Tensor dattn_residual = dresidual2;
        for (i32 s = 0; s < seq; s++)
            for (i32 d = 0; d < dim; d++)
                dattn_residual(s, d) += dffn_normed(s, d);
        
        dh = dattn_residual;
        
        // 4b. Attention backward（完整版：通过 softmax + Q/K/V 投影反向）
        Tensor dattn_out, d_attn_normed_full;
        {
            // o_proj backward: dh → dattn_out
            Tensor dx_o, dw_o, db_o;
            linear_backward(lc.attn.attn_out, layer.attention.o_proj.weight,
                             dh, dattn_out, dw_o, db_o);
            g.do_w = dw_o;
            g.do_b = db_o;
            
            // 完整注意力反向（包括 Q/K/V/RoPE）
            attention_backward_full(layer.attention, lc.attn,
                                     dattn_out, lc.attn_normed,
                                     d_attn_normed_full,
                                     g.dq_w, g.dq_b,
                                     g.dk_w, g.dk_b,
                                     g.dv_w, g.dv_b);
        }
        
        // attn_norm backward
        {
            Tensor dx, dw;
            rmsnorm_backward(lc.x, layer.attn_norm.weight,
                               d_attn_normed_full, dx, dw);
            g.dattn_norm_w = dw;
            // dx 是 norm 输入的梯度，再加上残差连接
            Tensor d_embed_grad = dx;
            for (i32 s = 0; s < seq; s++)
                for (i32 d = 0; d < dim; d++)
                    d_embed_grad(s, d) += dh(s, d);  // 残差连接
            dh = d_embed_grad;
        }
    }
    
    // 5. Embedding backward
    for (i32 s = 0; s < seq; s++) {
        Token t = targets[s];
        if (t >= 0 && t < model.vocab_size) {
            for (i32 d = 0; d < dim; d++)
                grads.dembed(t, d) += dh(s, d);
        }
    }
}

// ============================================================
//  将梯度应用到优化器
// ============================================================
inline void apply_gradients(Transformer& model, Gradients& grads) {
    // Embedding
    model.token_embedding.data = grads.dembed.data;  // 复用 data 为梯度
    
    // Final norm
    model.final_norm.weight.data = grads.dfinal_norm_w.data;
    
    // LM head
    model.lm_head.weight.data = grads.dlm_head_w.data;
    model.lm_head.bias.data = grads.dlm_head_b.data;
    
    for (i32 l = 0; l < model.n_layers; l++) {
        auto& layer = *model.layers[l];
        auto& g = grads.layers[l];
        
        layer.attn_norm.weight.data = g.dattn_norm_w.data;
        layer.ffn_norm.weight.data = g.dffn_norm_w.data;
        layer.attention.q_proj.weight.data = g.dq_w.data;
        layer.attention.q_proj.bias.data = g.dq_b.data;
        layer.attention.k_proj.weight.data = g.dk_w.data;
        layer.attention.k_proj.bias.data = g.dk_b.data;
        layer.attention.v_proj.weight.data = g.dv_w.data;
        layer.attention.v_proj.bias.data = g.dv_b.data;
        layer.attention.o_proj.weight.data = g.do_w.data;
        layer.attention.o_proj.bias.data = g.do_b.data;
        
        if (layer.ffn) {
            layer.ffn->gate_proj.weight.data = g.dgate_w.data;
            layer.ffn->gate_proj.bias.data = g.dgate_b.data;
            layer.ffn->up_proj.weight.data = g.dup_w.data;
            layer.ffn->up_proj.bias.data = g.dup_b.data;
            layer.ffn->down_proj.weight.data = g.ddown_w.data;
            layer.ffn->down_proj.bias.data = g.ddown_b.data;
        }
    }
}

// ============================================================
//  简易文本数据加载器
// ============================================================
struct TextDataLoader {
    std::vector<Token> all_tokens;
    i32 seq_len;
    i32 current_pos;
    BPETokenizer* tokenizer;
    
    TextDataLoader(BPETokenizer* tok, i32 sl = 64)
        : seq_len(sl), current_pos(0), tokenizer(tok) {}
    
    bool load_file(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
        all_tokens = tokenizer->encode(content);
        std::cout << "  加载 " << path << ": " << all_tokens.size() << " tokens\n";
        return !all_tokens.empty();
    }
    
    bool load_text(const std::string& text) {
        all_tokens = tokenizer->encode(text);
        std::cout << "  加载文本: " << all_tokens.size() << " tokens\n";
        return !all_tokens.empty();
    }
    
    // 获取下一个 batch
    bool next_batch(std::vector<Token>& input, std::vector<Token>& target) {
        if (current_pos + seq_len + 1 >= (i32)all_tokens.size()) {
            current_pos = 0;  // 循环
            if (all_tokens.size() < seq_len + 1) return false;
        }
        
        input.resize(seq_len);
        target.resize(seq_len);
        for (i32 i = 0; i < seq_len; i++) {
            input[i]  = all_tokens[current_pos + i];
            target[i] = all_tokens[current_pos + i + 1];
        }
        current_pos += seq_len;
        return true;
    }
    
    i32 total_batches() const {
        return (i32)all_tokens.size() / seq_len;
    }
    
    void reset() { current_pos = 0; }
};

// ============================================================
//  Trainer — 训练循环
// ============================================================
struct Trainer {
    Transformer& model;
    AdamW optimizer;
    BPETokenizer* tokenizer;
    TextDataLoader loader;
    
    i32 total_steps;
    f32 current_loss;
    i32 log_interval;
    i32 checkpoint_interval;
    std::string checkpoint_dir;
    
    Trainer(Transformer& m, BPETokenizer* tok, i32 seq_len = 64,
            f32 lr = 3e-4f)
        : model(m), tokenizer(tok),
          loader(tok, seq_len),
          total_steps(0), current_loss(0),
          log_interval(10), checkpoint_interval(500),
          checkpoint_dir("./checkpoints") {
        optimizer = AdamW(lr);
        optimizer.register_model(model);
    }
    
    // 单步训练
    f32 train_step(const std::vector<Token>& input, const std::vector<Token>& target) {
        // 前向
        ForwardCache cache;
        transformer_forward_train(model, input, cache);
        
        // 反向
        Gradients grads;
        grads.init(model);
        f32 loss;
        transformer_backward(model, cache, target, grads, loss);
        
        // 应用梯度到模型权重（累加到权重 data 中供 AdamW 使用）
        // 注意：这里将梯度存入权重 data 中（原地替换），AdamW 会读取并更新
        apply_gradients(model, grads);
        
        // 优化器步骤（需要同时知道原始权重和梯度）
        // 修复：AdamW 内部从 ParamState.m/v 和当前梯度值更新
        optimizer.step();
        
        total_steps++;
        current_loss = loss;
        return loss;
    }
    
    // 训练完整数据集
    void train(i32 max_steps, bool verbose = true) {
        std::vector<Token> input, target;
        
        auto t0 = std::chrono::steady_clock::now();
        
        for (i32 step = 0; step < max_steps; step++) {
            if (!loader.next_batch(input, target)) {
                std::cout << "  数据集用完，重新开始\n";
                loader.reset();
                if (!loader.next_batch(input, target)) break;
            }
            
            f32 loss = train_step(input, target);
            
            if (verbose && step % log_interval == 0) {
                auto t1 = std::chrono::steady_clock::now();
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                f32 tokens_per_sec = (f32)(step + 1) * loader.seq_len / (ms / 1000.0f + 1e-6f);
                std::cout << "  step " << step << "/" << max_steps
                          << " | loss=" << loss
                          << " | " << (i32)tokens_per_sec << " tok/s"
                          << " | " << ms/1000.0f << "s" << std::endl;
            }
            
            if (checkpoint_interval > 0 && step > 0 && step % checkpoint_interval == 0) {
                save_checkpoint(step);
            }
        }
        
        auto t1 = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "\n  训练完成: " << max_steps << " steps, "
                  << ms/1000.0f << "s, final loss=" << current_loss << "\n";
    }
    
    // 保存 checkpoint
    void save_checkpoint(i32 step) {
        std::string path = checkpoint_dir + "/ds4_step" + std::to_string(step) + ".gguf";
        export_gguf(path);
        std::cout << "  💾 Checkpoint: " << path << "\n";
    }
    
    // 导出 GGUF 模型
    void export_gguf(const std::string& path) {
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "  无法写入 " << path << "\n";
            return;
        }
        
        auto w32 = [&](u32 v) { f.write((char*)&v, 4); };
        auto w64 = [&](u64 v) { f.write((char*)&v, 8); };
        auto wstr = [&](const std::string& s) {
            u64 len = s.size();
            f.write((char*)&len, 8);
            f.write(s.data(), len);
        };
        auto wf32 = [&](f32 v) { f.write((char*)&v, 4); };
        auto wi32 = [&](i32 v) { f.write((char*)&v, 4); };
        auto wbool = [&](bool v) { u8 b = v ? 1 : 0; f.write((char*)&b, 1); };
        
        // GGUF Header
        w32(GGUF_MAGIC);
        w32(GGUF_VERSION);
        
        // 计算张量数量
        u64 n_tensors = 3 + 9 * model.n_layers;
        u64 n_kv = 23;
        
        w64(n_tensors);
        w64(n_kv);
        
        // 元数据
        auto write_meta_str = [&](const std::string& key, const std::string& val) {
            wstr(key); w32(9); wstr(val);
        };
        auto write_meta_i32 = [&](const std::string& key, i32 val) {
            wstr(key); w32(4); wi32(val);
        };
        auto write_meta_f32 = [&](const std::string& key, f32 val) {
            wstr(key); w32(6); wf32(val);
        };
        auto write_meta_bool = [&](const std::string& key, bool val) {
            wstr(key); w32(8); wbool(val);
        };
        
        write_meta_str("general.architecture", "ds4");
        write_meta_str("general.name", "ds4");
        write_meta_i32("ds4.embedding_length", model.dim);
        write_meta_i32("ds4.block_count", model.n_layers);
        write_meta_i32("ds4.attention.head_count", model.n_heads);
        write_meta_i32("ds4.attention.head_count_kv", model.n_kv_heads);
        write_meta_i32("ds4.feed_forward_length", model.ff_dim);
        write_meta_f32("ds4.attention.layer_norm_rms_epsilon", 1e-6f);
        write_meta_i32("ds4.vocab_size", model.vocab_size);
        write_meta_i32("ds4.context_length", model.max_seq_len);
        write_meta_str("tokenizer.ggml.model", "bpe");
        write_meta_str("tokenizer.ggml.bos_token", "<s>");
        write_meta_str("tokenizer.ggml.eos_token", "</s>");
        write_meta_i32("tokenizer.ggml.bos_token_id", 1);
        write_meta_i32("tokenizer.ggml.eos_token_id", 2);
        write_meta_i32("tokenizer.ggml.unknown_token_id", 0);
        write_meta_i32("tokenizer.ggml.padding_token_id", 0);
        write_meta_bool("ds4.use_moe", model.use_moe);
        write_meta_str("general.file_type", "F32");
        write_meta_str("general.quantization_version", "1");
        write_meta_str("ds4.attention.causal", "true");
        write_meta_str("ds4.attention.rope_theta", "10000.0");
        
        // 写入 tokenizer tokens（作为 string array）
        if (tokenizer) {
            wstr("tokenizer.ggml.tokens");
            w32(10); w32(9);
            i32 n_tokens = (i32)tokenizer->id_to_token.size();
            w64((u64)n_tokens);
            for (i32 i = 0; i < n_tokens; i++) {
                wstr(tokenizer->id_to_token[i]);
            }
        }
        
        // 对齐
        u64 pos = f.tellp();
        u64 pad = (32 - (pos % 32)) % 32;
        for (u64 i = 0; i < pad; i++) f.put(0);
        
        // 收集张量
        struct TensorEntry {
            std::string name;
            const Tensor* t;
            i32 n_dims;
            u64 d0, d1;
        };
        std::vector<TensorEntry> entries;
        
        entries.push_back({"token_embd.weight", &model.token_embedding, 2,
                           (u64)model.token_embedding.rows, (u64)model.token_embedding.cols});
        
        for (i32 l = 0; l < model.n_layers; l++) {
            auto& layer = *model.layers[l];
            std::string blk = "blk." + std::to_string(l);
            
            entries.push_back({blk + ".attn_norm.weight", &layer.attn_norm.weight, 1,
                               (u64)layer.attn_norm.weight.cols, 0});
            entries.push_back({blk + ".attn_q.weight", &layer.attention.q_proj.weight, 2,
                               (u64)layer.attention.q_proj.weight.rows,
                               (u64)layer.attention.q_proj.weight.cols});
            entries.push_back({blk + ".attn_k.weight", &layer.attention.k_proj.weight, 2,
                               (u64)layer.attention.k_proj.weight.rows,
                               (u64)layer.attention.k_proj.weight.cols});
            entries.push_back({blk + ".attn_v.weight", &layer.attention.v_proj.weight, 2,
                               (u64)layer.attention.v_proj.weight.rows,
                               (u64)layer.attention.v_proj.weight.cols});
            entries.push_back({blk + ".attn_output.weight", &layer.attention.o_proj.weight, 2,
                               (u64)layer.attention.o_proj.weight.rows,
                               (u64)layer.attention.o_proj.weight.cols});
            entries.push_back({blk + ".ffn_norm.weight", &layer.ffn_norm.weight, 1,
                               (u64)layer.ffn_norm.weight.cols, 0});
            
            if (layer.ffn) {
                entries.push_back({blk + ".ffn_gate.weight", &layer.ffn->gate_proj.weight, 2,
                                   (u64)layer.ffn->gate_proj.weight.rows,
                                   (u64)layer.ffn->gate_proj.weight.cols});
                entries.push_back({blk + ".ffn_up.weight", &layer.ffn->up_proj.weight, 2,
                                   (u64)layer.ffn->up_proj.weight.rows,
                                   (u64)layer.ffn->up_proj.weight.cols});
                entries.push_back({blk + ".ffn_down.weight", &layer.ffn->down_proj.weight, 2,
                                   (u64)layer.ffn->down_proj.weight.rows,
                                   (u64)layer.ffn->down_proj.weight.cols});
            }
        }
        
        entries.push_back({"output_norm.weight", &model.final_norm.weight, 1,
                           (u64)model.final_norm.weight.cols, 0});
        entries.push_back({"output.weight", &model.lm_head.weight, 2,
                           (u64)model.lm_head.weight.rows, (u64)model.lm_head.weight.cols});
        
        // 计算偏移量
        u64 offset = 0;
        for (auto& e : entries) {
            u64 data_size = e.t->size() * sizeof(f32);
            
            // 写入张量信息头（严格按照 GGUF 规范：n_dims 个 shape 值）
            wstr(e.name);
            wi32(e.n_dims);
            if (e.n_dims >= 1) w64(e.d0);
            if (e.n_dims >= 2) w64(e.d1);
            // 注意：不写多余的 padding 维度，GGUF reader 根据 n_dims 读取
            w32(GGML_TYPE_F32);
            w64(offset);
            
            offset += data_size;
        }
        
        // 对齐
        pos = f.tellp();
        pad = (32 - (pos % 32)) % 32;
        for (u64 i = 0; i < pad; i++) f.put(0);
        
        // 写张量数据
        for (auto& e : entries) {
            u64 size = e.t->size() * sizeof(f32);
            f.write((char*)e.t->data.data(), size);
        }
        
        f.close();
        std::cout << "  📦 导出 GGUF: " << path << " (" << (offset / 1024 / 1024) << " MB)\n";
    }
};

} // namespace ds4pro

#endif // DS4PRO_TRAIN_HPP