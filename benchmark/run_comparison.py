from .comparison_engine import Comparison
from .test_prompts import PROMPTS
from config.settings import Config

def run_benchmarks():
    """Run the complete honest comparison"""
    comparator = Comparison()
    standard_results = comparator.run_standard_conversation(PROMPTS)
    compressed_results = comparator.run_compressed_conversation(PROMPTS)
    
    avg_metrics, similarity_metrics, memory_saved, similarity_list, semantic_list = comparator.compare(standard_results, compressed_results)
    
    comparator.plot_results(
                standard_results, 
                compressed_results, 
                comparison_metrics=avg_metrics,
                similarity_list=similarity_list,
                semantic_list=semantic_list
            )
    
    print("\n📁 All results saved to 'benchmark/results/' directory")
