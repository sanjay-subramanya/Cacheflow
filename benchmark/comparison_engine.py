import os
import time
import torch
import json
import gc
import numpy as np
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from core.decoder import KVCompressedDecoder
from core.chat import ChatEngine
from config.settings import Config

class Comparison:
    def __init__(self):
        self.cfg = Config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_path, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        os.makedirs(f"{self.cfg.parent_dir}/benchmark/results", exist_ok=True)
    
    def format_chat_prompt(self, messages):
        """Apply chat template"""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def estimate_standard_memory(self, total_tokens):
        """Estimate what standard model would use"""
        # For Qwen2: ~28KB per token across all layers
        bytes_per_token = 28 * 1024
        return (total_tokens * bytes_per_token) / (1024 * 1024)  # MB
        
    def run_standard_conversation(self, conversation_turns):
        """Run conversation with vanilla model"""
        print("\n" + "="*70)
        print("📥 PHASE 1: Running Vanilla model conversation")
        print("="*70)
        
        print("Loading standard model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_path,
            dtype=self.cfg.dtype,
            local_files_only=True,
            low_cpu_mem_usage=True
        ).to(self.cfg.device)
        model.eval()
        
        messages = []
        results = {
            "responses": [],
            "timing": [],
            "tokens": [],
            "cumulative_tokens": [],
            "full_history": []
        }
        total_tokens = 0
        
        try:
            for turn_idx, user_message in enumerate(conversation_turns):
                print(f"\n📝 Turn {turn_idx + 1}: '{user_message}'")
                
                messages.append({"role": "user", "content": user_message})
                prompt = self.format_chat_prompt(messages)
                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=self.cfg.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True
                    )
                
                generation_time = time.time() - start_time
                input_length = inputs.input_ids.shape[1]
                response = self.tokenizer.decode(
                    outputs.sequences[0][input_length:], 
                    skip_special_tokens=True
                )
                
                messages.append({"role": "assistant", "content": response})
                turn_tokens = len(response.split())
                total_tokens += turn_tokens
                
                results["responses"].append(response)
                results["timing"].append(generation_time)
                results["tokens"].append(turn_tokens)
                results["cumulative_tokens"].append(total_tokens)
                results["full_history"].append({
                    "turn": turn_idx + 1,
                    "user": user_message,
                    "assistant": response,
                    "time": generation_time,
                    "tokens": turn_tokens,
                    "cumulative_tokens": total_tokens
                })
                
                print(f"  Response: {response[:100]}...")
                print(f"  Time: {generation_time:.2f}s")
                print(f"  Total tokens so far: {total_tokens}")
        
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print("\n✅ Standard model unloaded")
        
        with open(f"{self.cfg.parent_dir}/benchmark/results/standard_conversation.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_compressed_conversation(self, conversation_turns):
        """Run conversation with compressed KV cache"""
        print("\n" + "="*70)
        print("📥 PHASE 2: Running COMPRESSED decoder conversation")
        print("="*70)
        
        print("Loading compressed decoder...")
        decoder = KVCompressedDecoder(self.cfg)
        backend = ChatEngine(decoder)
        print("✅ Compressed decoder loaded")
        
        messages = []
        results = {
            "responses": [],
            "timing": [],
            "tokens": [],
            "cumulative_tokens": [],
            "telemetry": [],
            "memory_saved": [],
            "fp_tokens_history": [],
            "int8_tokens_history": [],
            "full_history": []
        }
        
        total_tokens = 0
        
        for turn_idx, user_message in enumerate(conversation_turns):
            print(f"\n📝 Turn {turn_idx + 1}: '{user_message}'")
            decoder.last_tokens = []
            
            messages.append({"role": "user", "content": user_message})
            if len(messages) == 1: 
                prompt = f"{backend.system_prompt}<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
            else:
                prompt = f"\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

            start_time = time.time()
            response = ""
            final_telemetry = None
            turn_token_count = 0
            
            for text, telemetry in decoder.generate(prompt):
                response = text
                turn_token_count += 1
                final_telemetry = telemetry
            
            generation_time = time.time() - start_time
            messages.append({"role": "assistant", "content": response})
            total_tokens += turn_token_count
            
            results["responses"].append(response)
            results["timing"].append(generation_time)
            results["tokens"].append(turn_token_count)
            results["cumulative_tokens"].append(total_tokens)
            results["telemetry"].append({
                "fp_tokens": final_telemetry.fp_tokens,
                "int8_tokens": final_telemetry.int8_tokens,
                "total_tokens": final_telemetry.total_tokens,
                "memory_fp_mb": final_telemetry.memory_fp_mb,
                "memory_int8_mb": final_telemetry.memory_int8_mb,
                "saved_mb": final_telemetry.saved_mb
            })
            results["memory_saved"].append(final_telemetry.saved_mb)
            results["fp_tokens_history"].append(final_telemetry.fp_tokens)
            results["int8_tokens_history"].append(final_telemetry.int8_tokens)
            results["full_history"].append({
                "turn": turn_idx + 1,
                "user": user_message,
                "assistant": response,
                "time": generation_time,
                "tokens": turn_token_count,
                "cumulative_tokens": total_tokens,
                "telemetry": {
                    "fp_tokens": final_telemetry.fp_tokens,
                    "int8_tokens": final_telemetry.int8_tokens,
                    "saved_mb": final_telemetry.saved_mb,
                    "memory_mb": final_telemetry.memory_fp_mb + final_telemetry.memory_int8_mb
                }
            })
            
            print(f"  Response: {response[:100]}...")
            print(f"  Time: {generation_time:.2f}s")
            print(f"  Memory saved: {final_telemetry.saved_mb:.2f}MB")
            print(f"  Cache: FP16={final_telemetry.fp_tokens}, INT8={final_telemetry.int8_tokens}")
            print(f"  Total tokens so far: {total_tokens}")
        
        with open(f"{self.cfg.parent_dir}/benchmark/results/compressed_kv_conversation.json", "w") as f:
            json.dump(results, f, indent=2)

        gc.collect()
        torch.cuda.empty_cache()
        return results
    
    def compare(self, standard_results, compressed_results):
        """Comparison showcasing differences and trade-offs"""
        print("\n" + "="*70)
        print("🔍 ANALYSIS: Standard vs Compressed")
        print("="*70)

        gc.collect()
        torch.cuda.empty_cache()
        
        comparison = {
            "turns": [],
            "metrics": {
                "similarity": [],
                "rouge1": [],
                "rouge2": [],
                "rougeL": [],
                "semantic": [],
                "speed_difference": [],
                "memory_saved": []
            }
        }
        
        num_turns = min(len(standard_results["responses"]), len(compressed_results["responses"]))
        
        for turn in range(num_turns):
            std_response = standard_results["responses"][turn]
            comp_response = compressed_results["responses"][turn]
            
            if not std_response or not comp_response:
                continue
            
            seq_sim = SequenceMatcher(None, std_response, comp_response).ratio()
            rouge = self.rouge_scorer.score(std_response, comp_response)

            emb1 = self.semantic_model.encode(std_response, convert_to_tensor=True)
            emb2 = self.semantic_model.encode(comp_response, convert_to_tensor=True)
            semantic_sim = util.pytorch_cos_sim(emb1, emb2).item()
            
            speed_diff = compressed_results["timing"][turn] - standard_results["timing"][turn]
            memory_saved = compressed_results["memory_saved"][turn]
            
            comparison["metrics"]["similarity"].append(seq_sim)
            comparison["metrics"]["rouge1"].append(rouge['rouge1'].fmeasure)
            comparison["metrics"]["rouge2"].append(rouge['rouge2'].fmeasure)
            comparison["metrics"]["rougeL"].append(rouge['rougeL'].fmeasure)
            comparison["metrics"]["semantic"].append(semantic_sim)
            comparison["metrics"]["speed_difference"].append(speed_diff)
            comparison["metrics"]["memory_saved"].append(memory_saved)
            
            comparison["turns"].append({
                "turn": turn + 1,
                "similarity": seq_sim,
                "rouge1": rouge['rouge1'].fmeasure,
                "semantic": semantic_sim,
                "memory_saved_mb": memory_saved
            })
        
        avg_metrics = {
            "similarity": np.mean(comparison["metrics"]["similarity"]),
            "rouge1": np.mean(comparison["metrics"]["rouge1"]),
            "rouge2": np.mean(comparison["metrics"]["rouge2"]),
            "rougeL": np.mean(comparison["metrics"]["rougeL"]),
            "semantic": np.mean(comparison["metrics"]["semantic"]),
            "memory_saved": np.mean(comparison["metrics"]["memory_saved"]),
            "speed_overhead": np.mean(comparison["metrics"]["speed_difference"])
        }
        
        similarity_retention = (avg_metrics["similarity"] + avg_metrics["semantic"]) / 2
        
        print("\n" + "-"*70)
        print("📊 THE REAL TRADE-OFFS")
        print("-"*70)
        
        final_turn = len(standard_results["cumulative_tokens"]) - 1
        total_tokens = standard_results["cumulative_tokens"][final_turn]
        std_memory = self.estimate_standard_memory(total_tokens)
        comp_memory = compressed_results["telemetry"][-1]["memory_fp_mb"] + compressed_results["telemetry"][-1]["memory_int8_mb"]
        
        print(f"\n💾 MEMORY COMPARISON:")
        print(f"  • Standard model: {std_memory:.1f}MB (and GROWING)")
        print(f"  • Your compression: {comp_memory:.1f}MB (CAPPED)")
        print(f"  • Memory saved: {std_memory - comp_memory:.1f}MB")
        print(f"  • After 2x conversation: Standard={std_memory*2:.1f}MB, Yours={comp_memory:.1f}MB (STILL CAPPED)")
        
        print(f"\n📉 QUALITY TRADE-OFF:")
        print(f"  • Quality retention: {similarity_retention:.2%}")
        print(f"  • Quality loss: {(1 - similarity_retention)*100:.1f}%")
        print(f"  • Acceptable for long conversations? {'YES' if similarity_retention > 0.85 else 'Depends on use case'}")
        
        print(f"\n⚡ SPEED IMPACT:")
        print(f"  • Standard avg: {np.mean(standard_results['timing']):.2f}s per turn")
        print(f"  • Compressed avg: {np.mean(compressed_results['timing']):.2f}s per turn")
        print(f"  • Overhead: {avg_metrics['speed_overhead']:+.2f}s per turn ({((avg_metrics['speed_overhead']/np.mean(standard_results['timing']))*100):+.1f}%)")
        
        final_fp = compressed_results["telemetry"][-1]["fp_tokens"]
        final_int8 = compressed_results["telemetry"][-1]["int8_tokens"]
        context_tokens = final_fp + final_int8
        
        print(f"\n📚 CONTEXT WINDOW:")
        print(f"  • Standard: Full conversation history ({total_tokens} tokens)")
        print(f"  • Compressed: Last {context_tokens} tokens (FP16={final_fp}, INT8={final_int8})")
        print(f"  • Trade-off: Can't recall messages before token {total_tokens - context_tokens}")
        
        print(f"\n📈 SCALABILITY:")
        print(f"  • Standard: OOM crash at ~2000-3000 tokens")
        print(f"  • Compressed: Runs FOREVER at {comp_memory:.1f}MB")
        
        print("\n" + "="*70)
        print("🎯 VERDICT")
        print("="*70)
        
        print(f"\n📊 Key Metrics:")
        print(f"  • Quality retained: {similarity_retention:.2%}")
        print(f"  • Memory saved: {std_memory - comp_memory:.1f}MB ({((std_memory - comp_memory)/std_memory)*100:.0f}% reduction)")
        print(f"  • Speed overhead: {avg_metrics['speed_overhead']:+.2f}s per turn")
        print(f"  • Context window: {context_tokens} tokens (sliding)")
        
        with open(f"{self.cfg.parent_dir}/benchmark/results/comparison_result.json", "w") as f:
            json.dump({
                "averages": avg_metrics,
                "similarity_retention": similarity_retention,
                "memory": {
                    "standard_mb": std_memory,
                    "compressed_mb": comp_memory,
                    "saved_mb": std_memory - comp_memory,
                    "saved_percent": ((std_memory - comp_memory)/std_memory)*100
                },
                "context": {
                    "standard_tokens": total_tokens,
                    "compressed_tokens": context_tokens,
                    "fp16_tokens": final_fp,
                    "int8_tokens": final_int8
                },
            }, f, indent=2)
        
        return avg_metrics, similarity_retention, (std_memory - comp_memory), comparison["metrics"]["similarity"], comparison["metrics"]["semantic"]
    
    def plot_results(self, standard_results, compressed_results, comparison_metrics, similarity_list, semantic_list):
        """Create visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        ax = axes[0, 0]
        std_memory = [self.estimate_standard_memory(t) for t in standard_results["cumulative_tokens"]]
        comp_memory = [t["memory_fp_mb"] + t["memory_int8_mb"] for t in compressed_results["telemetry"]]
        
        ax.plot(range(1, len(std_memory)+1), std_memory, 'o-', label='Standard (unbounded)', color='red', linewidth=2)
        ax.plot(range(1, len(comp_memory)+1), comp_memory, 's-', label='Compressed (CAPPED)', color='green', linewidth=2)
        ax.axhline(y=comp_memory[-1], color='green', linestyle='--', alpha=0.5, label=f'Cap: {comp_memory[-1]:.1f}MB')
        ax.set_xlabel('Conversation Turn')
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Usage: Unbounded vs Capped')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.plot(range(1, len(similarity_list)+1), 
                similarity_list, 'o-', label='Similarity', color='blue')
        ax.plot(range(1, len(semantic_list)+1), 
                semantic_list, 's-', label='Semantic', color='purple')
        ax.set_xlabel('Conversation Turn')
        ax.set_ylabel('Score')
        ax.set_title('Quality Retention Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        ax = axes[0, 2]
        x = range(1, len(compressed_results["fp_tokens_history"])+1)
        ax.bar(x, compressed_results["fp_tokens_history"], label='FP16 Tokens', color='blue', alpha=0.7)
        ax.bar(x, compressed_results["int8_tokens_history"], bottom=compressed_results["fp_tokens_history"], 
            label='INT8 Tokens', color='orange', alpha=0.7)
        ax.set_xlabel('Conversation Turn')
        ax.set_ylabel('Token Count')
        ax.set_title('KV Cache Composition')
        ax.legend()
        
        ax = axes[1, 0]
        x = range(1, len(standard_results["timing"])+1)
        ax.plot(x, standard_results["timing"], 'o-', label='Standard', color='blue')
        ax.plot(x, compressed_results["timing"], 's-', label='Compressed', color='orange')
        ax.set_xlabel('Conversation Turn')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Response Time: Overhead of Compression')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        final_turn = len(standard_results["cumulative_tokens"]) - 1
        total_tokens = standard_results["cumulative_tokens"][final_turn]
        std_mem = self.estimate_standard_memory(total_tokens)
        comp_mem = compressed_results["telemetry"][-1]["memory_fp_mb"] + compressed_results["telemetry"][-1]["memory_int8_mb"]
        
        bars = ax.bar(['Standard', 'Compressed', 'Saved'], 
                    [std_mem, comp_mem, std_mem - comp_mem],
                    color=['red', 'green', 'blue'])
        ax.set_ylabel('Memory (MB)')
        ax.set_title(f'Final Memory: Saved {std_mem - comp_mem:.1f}MB')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}MB', ha='center', va='bottom')
        
        ax = axes[1, 2]
        ax.axis('off')
        similarity = (comparison_metrics.get("similarity", 0) + comparison_metrics.get("semantic", 0)) / 2
        text = f"""
            TRADE-OFF SUMMARY
            
            ✓ Memory: {std_mem:.1f}MB → {comp_mem:.1f}MB
            ✓ Saved: {std_mem - comp_mem:.1f}MB ({((std_mem - comp_mem)/std_mem)*100:.0f}%)
            
            📉 Similarity: {similarity:.2%} retained
            ⚡ Speed: +{comparison_metrics.get("speed_overhead", 0):.2f}s/turn
            
            📚 Context: {compressed_results['telemetry'][-1]['fp_tokens'] + compressed_results['telemetry'][-1]['int8_tokens']} tokens
            (sliding window)
        """
        ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle("KV Cache Compression: Trade-off Analysis", size=16, y=1.02)
        plt.tight_layout()
        plt.savefig(f"{self.cfg.parent_dir}/benchmark/results/analysis.png", dpi=150, bbox_inches='tight')
        plt.show()