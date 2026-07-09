/*
 * train_ds4 — ds4 模型训练程序
 *
 * 用法：
 *   ./train_ds4                         交互式训练
 *   ./train_ds4 train data.txt          从文件训练
 *   ./train_ds4 gen model.gguf "prompt" 用训练好的模型推理
 */

#include "ds4pro.hpp"
#include "ds4pro_train.hpp"
#include <iostream>
#include <string>
#include <chrono>

using namespace ds4pro;

// 内置训练语料 —— ds4 的身份认知数据
const char* BUILTIN_CORPUS = R"(
ds4是一个由王铠宁创建的本地AI模型。
ds4运行在ds4pro推理引擎上。
ds4pro是一个用C++17编写的深度学习推理引擎。
ds4pro支持Transformer架构，包括分组查询注意力、旋转位置编码、SwiGLU激活函数。
ds4pro支持混合专家层。
ds4pro的默认配置是512维8层8头注意力。
ds4使用GGUF格式保存和加载模型。
ds4可以在手机上本地运行，不需要联网。
ds4的创造者是王铠宁，他是一名学生。
ds4是客尘的个人AI助手。
你好，我是ds4，很高兴认识你。
人工智能可以帮助人类解决很多问题。
深度学习是机器学习的一个分支。
Transformer架构由Vaswani等人在2017年提出。
语言模型可以生成文本、翻译语言、回答问题。
知识就是力量。
学习是一生的事业。
天空是蓝色的。
地球是圆的。
水在零度结冰。
太阳从东方升起。
一年有四季：春夏秋冬。
月亮绕着地球转。
光速是每秒三十万公里。
世界上最高的山是珠穆朗玛峰。
中国有五千年的文明历史。
编程是一种创造性的活动。
C++是一门功能强大的编程语言。
数学是科学的基础。
音乐是人类共通的语言。
阅读可以开阔视野。
健康是最重要的财富。
善良是一种选择。
诚实是最好的策略。
耐心是一种美德。
友谊是人生中最珍贵的礼物之一。
家人是我们最坚强的后盾。
时间是最公平的，每个人一天都有二十四小时。
成功需要努力和坚持。
失败是成功之母。
知识改变命运。
想象力比知识更重要。
逻辑可以带你从A到B，想象力可以带你去任何地方。
简单是终极的复杂。
保持饥饿，保持愚蠢。
活着就是为了改变世界。
)";

int main(int argc, char* argv[]) {
    std::cout << "\n"
              << "  ╔══════════════════════════════╗\n"
              << "  ║   ds4 — 从头训练你的AI模型   ║\n"
              << "  ╚══════════════════════════════╝\n\n";

    // 训练配置
    i32 model_dim = 256;       // 256维
    i32 n_layers = 4;          // 4层
    i32 n_heads = 4;
    i32 n_kv_heads = 2;
    i32 ff_dim = 1024;
    i32 seq_len = 64;
    i32 train_steps = 5000;    // 增加到5000步
    f32 learning_rate = 5e-4f;
    
    // 命令行参数解析
    std::string mode = "train";
    std::string data_path;
    std::string model_path;
    std::string prompt;
    
    if (argc > 1) {
        mode = argv[1];
        if (argc > 2) data_path = argv[2];
        if (argc > 3) prompt = argv[3];
    }
    
    // ========== 推理模式 ==========
    if (mode == "gen" || mode == "generate" || mode == "once") {
        if (data_path.empty()) {
            std::cerr << "用法: ./train_ds4 gen model.gguf \"prompt\"\n";
            return 1;
        }
        model_path = data_path;
        if (prompt.empty()) prompt = "你好";
        
        std::cout << "📂 加载: " << model_path << "\n";
        
        GGUFReader reader;
        if (!reader.open(model_path)) return 1;
        
        auto get_i32 = [&](const std::string& key, i32 def) -> i32 {
            auto it = reader.metadata_i32.find(key);
            return it != reader.metadata_i32.end() ? it->second : def;
        };
        std::string arch = reader.metadata_str.count("general.architecture")
            ? reader.metadata_str["general.architecture"] : "ds4";
        std::string pfx = arch + ".";
        
        i32 dim  = get_i32(pfx + "embedding_length", 256);
        i32 n_l  = get_i32(pfx + "block_count", 4);
        i32 n_h  = get_i32(pfx + "attention.head_count", 4);
        i32 n_kv = get_i32(pfx + "attention.head_count_kv", 2);
        i32 ff_d = get_i32(pfx + "feed_forward_length", 1024);
        
        auto tokenizer = build_tokenizer_from_gguf(reader);
        i32 vocab = tokenizer.vocab_size > 0 ? tokenizer.vocab_size : 512;
        
        DS4Pro engine(dim, n_l, n_h, n_kv, ff_d, false, 4, 2, vocab);
        engine.tokenizer = std::make_unique<BPETokenizer>(std::move(tokenizer));
        engine.vocab_size = vocab;
        
        LLaMAWeights weights;
        if (!weights.load(reader, n_l)) {
            std::cerr << "❌ 权重加载失败\n";
            return 1;
        }
        weights.inject(*engine.model);
        
        engine.print_info();
        engine.set_temperature(0.7f);
        
        // 单次推理模式（适合脚本/API调用）
        if (mode == "once") {
            engine.set_temperature(0.7f);
            std::string out = engine.generate(prompt, 64);
            std::cout << out << std::endl;
            return 0;
        }
        
        // 交互
        std::cout << "\n━━━ ds4 对话（/quit 退出）━━━\n";
        std::string in = prompt;
        while (true) {
            std::cout << "👤 > " << in << "\n";
            auto ts = std::chrono::steady_clock::now();
            std::string out = engine.generate(in, 64);
            auto te = std::chrono::steady_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
            std::cout << "🤖   " << out << "  [" << ms << "ms]\n\n";
            
            if (prompt != "" && in == prompt) break;  // 非交互模式
            
            std::cout << "👤 > ";
            std::getline(std::cin, in);
            if (in == "/quit" || in == "/exit") break;
        }
        return 0;
    }
    
    // ========== 训练模式 ==========
    std::cout << "⚙ 模型配置:\n"
              << "  dim=" << model_dim << " layers=" << n_layers
              << " heads=" << n_heads << " kv_heads=" << n_kv_heads
              << " ff=" << ff_dim << "\n"
              << "  seq_len=" << seq_len << " steps=" << train_steps
              << " lr=" << learning_rate << "\n\n";

    // 构建模型
    std::cout << "🔧 初始化模型...\n";
    auto t0 = std::chrono::steady_clock::now();
    
    DS4Pro engine(model_dim, n_layers, n_heads, n_kv_heads, ff_dim, false);
    
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    engine.print_info();
    std::cout << "  初始化: " << ms << " ms\n\n";
    
    // 创建 Trainer
    Trainer trainer(*engine.model, engine.tokenizer.get(), seq_len, learning_rate);
    trainer.log_interval = 50;           // 每50步打印
    trainer.checkpoint_interval = 500;   // 每500步保存
    
    // 加载训练数据（默认使用大数据文件）
    std::cout << "📚 加载训练数据...\n";
    if (!data_path.empty()) {
        if (!trainer.loader.load_file(data_path)) {
            std::cerr << "  无法加载文件，使用内置语料\n";
            trainer.loader.load_text(BUILTIN_CORPUS);
        }
    } else {
        // 默认尝试加载大数据文件
        if (!trainer.loader.load_file("training_data.txt")) {
            std::cout << "  未找到training_data.txt，使用内置语料\n";
            trainer.loader.load_text(BUILTIN_CORPUS);
        }
    }
    
    i32 total_batches = trainer.loader.total_batches();
    std::cout << "  共计 " << trainer.loader.all_tokens.size() << " tokens, "
              << total_batches << " batches (seq=" << seq_len << ")\n";
    if (train_steps > total_batches) {
        std::cout << "  ⚠ 数据只有 " << total_batches << " batches，将循环使用\n";
    }
    
    std::cout << "\n🚀 开始训练...\n";
    std::cout << "  (在手机上训练较慢，请耐心等待)\n\n";
    
    trainer.train(train_steps, true);
    
    // 保存最终模型
    std::cout << "\n💾 保存最终模型...\n";
    trainer.export_gguf("ds4_final.gguf");
    
    // 快速测试
    std::cout << "\n🧪 快速测试:\n";
    engine.set_temperature(0.7f);
    for (const char* p : {"你好", "ds4是什么", "知识"}) {
        std::cout << "  👤 " << p << "\n";
        std::string out = engine.generate(p, 32);
        std::cout << "  🤖 " << out << "\n\n";
    }
    
    std::cout << "✅ 训练完成！模型已保存为 ds4_final.gguf\n";
    std::cout << "   推理: ./train_ds4 gen ds4_final.gguf \"你好\"\n";
    
    return 0;
}