"""
perturb_multi.py
Distributed/multinode inference for Qwen3-0.6B robustness analysis using Hugging Face Accelerate.
Enhanced version with comprehensive metrics and visualizations from perturbation.py.
"""
import os
import json
import random
import numpy as np
import pandas as pd
import re
import torch
import copy
import warnings
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# --- Utility Functions ---
def normalize_answer(answer):
    if not isinstance(answer, str):
        answer = str(answer)
    return re.sub(r'[^\w\s]', '', answer.lower().strip())

# --- Enhanced Perturbation Class ---
class RobustnessPerturber:
    """Advanced perturbation methods for testing reasoning robustness"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
    
    def shuffle_tokens(self, text, shuffle_ratio=0.3):
        """Randomly shuffle a percentage of tokens while preserving meaning"""
        words = text.split()
        if len(words) < 3:
            return text
        
        # Identify important words to preserve (entities, question words)
        important_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which']
        preserve_indices = []
        
        for i, word in enumerate(words):
            if (word.lower() in important_words or 
                (word[0].isupper() and len(word) > 2) or
                word.lower() in ['the', 'a', 'an', 'is', 'was', 'are']):
                preserve_indices.append(i)
        
        # Shuffle non-important words
        shuffleable_indices = [i for i in range(len(words)) if i not in preserve_indices]
        n_shuffle = int(len(shuffleable_indices) * shuffle_ratio)
        
        if n_shuffle > 1:
            indices_to_shuffle = random.sample(shuffleable_indices, n_shuffle)
            shuffled_words = [words[i] for i in indices_to_shuffle]
            random.shuffle(shuffled_words)
            
            for i, idx in enumerate(indices_to_shuffle):
                words[idx] = shuffled_words[i]
        
        return ' '.join(words)
    
    def inject_distractors(self, text, n_distractors=2):
        """Inject similar but irrelevant information"""
        distractors = [
            "Additionally, there are many other factors to consider.",
            "Some experts also mention related topics in this context.",
            "Historical records show various perspectives on this matter.",
            "Recent studies have explored similar questions extensively.",
            "Multiple sources provide different viewpoints on this subject.",
            "Contemporary analysis reveals additional complexity in this area.",
            "Various scholars have debated these issues for decades.",
            "Modern research continues to investigate these phenomena."
        ]
        
        selected_distractors = random.sample(distractors, min(n_distractors, len(distractors)))
        
        # Insert distractors at random positions
        sentences = text.split('.')
        for distractor in selected_distractors:
            if len(sentences) > 1:
                insert_pos = random.randint(0, len(sentences)-1)
                sentences.insert(insert_pos, distractor)
        
        return '. '.join(s.strip() for s in sentences if s.strip())
    
    def rephrase_question(self, question):
        """Rephrase question while maintaining meaning"""
        rephrase_patterns = [
            (r'^What is', 'Can you tell me what'),
            (r'^Who was', 'Do you know who was'),
            (r'^When did', 'At what time did'),
            (r'^Where is', 'In which location is'),
            (r'^How many', 'What is the number of'),
            (r'^Which', 'What specific'),
            (r'the capital of', 'the main city of'),
            (r'president of', 'leader of'),
            (r'located in', 'situated in'),
            (r'known for', 'famous for')
        ]
        
        rephrased = question
        pattern, replacement = random.choice(rephrase_patterns)
        rephrased = re.sub(pattern, replacement, rephrased, flags=re.IGNORECASE)
        
        return rephrased
    
    def semantic_noise(self, text, noise_level=0.1):
        """Add semantic noise by replacing words with synonyms"""
        synonym_map = {
            'large': 'big', 'small': 'tiny', 'important': 'significant',
            'country': 'nation', 'city': 'town', 'river': 'waterway',
            'mountain': 'peak', 'president': 'leader', 'capital': 'main city',
            'ancient': 'old', 'modern': 'contemporary', 'famous': 'well-known'
        }
        
        words = text.split()
        n_replace = int(len(words) * noise_level)
        
        for _ in range(n_replace):
            for i, word in enumerate(words):
                clean_word = word.lower().strip('.,!?')
                if clean_word in synonym_map:
                    words[i] = word.replace(clean_word, synonym_map[clean_word])
                    break
        
        return ' '.join(words)

def create_perturbation_variants(df, perturber):
    """Create multiple perturbation variants of the dataset"""
    variants = {}
    
    print("🔄 Creating perturbation variants...")
    
    # Original (baseline)
    variants['original'] = df.copy()
    
    # Token shuffling
    print("   → Token shuffling...")
    df_shuffle = df.copy()
    df_shuffle['question'] = df_shuffle['question'].apply(
        lambda x: perturber.shuffle_tokens(x, shuffle_ratio=0.2)
    )
    variants['token_shuffle'] = df_shuffle
    
    # Distractor injection
    print("   → Distractor injection...")
    df_distractor = df.copy()
    df_distractor['question'] = df_distractor['question'].apply(
        lambda x: perturber.inject_distractors(x, n_distractors=1)
    )
    variants['distractor_injection'] = df_distractor
    
    # Question rephrasing
    print("   → Question rephrasing...")
    df_rephrase = df.copy()
    df_rephrase['question'] = df_rephrase['question'].apply(perturber.rephrase_question)
    variants['rephrasing'] = df_rephrase
    
    # Semantic noise
    print("   → Semantic noise...")
    df_noise = df.copy()
    df_noise['question'] = df_noise['question'].apply(
        lambda x: perturber.semantic_noise(x, noise_level=0.15)
    )
    variants['semantic_noise'] = df_noise
    
    # Combined perturbations (novel approach)
    print("   → Combined perturbations...")
    df_combined = df.copy()
    df_combined['question'] = df_combined['question'].apply(
        lambda x: perturber.semantic_noise(
            perturber.inject_distractors(
                perturber.shuffle_tokens(x, shuffle_ratio=0.1), 
                n_distractors=1
            ), 
            noise_level=0.1
        )
    )
    variants['combined'] = df_combined
    
    print(f"✅ Created {len(variants)} perturbation variants")
    return variants

# --- Model Loading with Accelerate ---
def load_qwen_model(model_name, accelerator, hf_token=None):
    """Load Qwen model with Accelerate support"""
    accelerator.print(f"Loading {model_name} with Accelerate...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if accelerator.device.type == 'cuda' else torch.float32,
        trust_remote_code=True,
        token=hf_token
    )
    
    # Prepare model with accelerator
    model = accelerator.prepare(model)
    
    return model, tokenizer

# --- Distributed Inference ---
def run_fast_predictions(df, model, tokenizer, accelerator, batch_size=16):
    """Enhanced distributed prediction generation with reasoning steps"""
    
    def create_prompt(question):
        return f"Q: {question}\nA: Step 1:"
    
    questions = df['question'].tolist()
    prompts = [create_prompt(q) for q in questions]
    
    all_preds = []
    all_steps = []
    
    accelerator.print(f"Processing {len(prompts)} questions in batches of {batch_size}")
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating predictions", disable=not accelerator.is_local_main_process):
        batch_prompts = prompts[i:i+batch_size]
        
        try:
            # Get actual model from DDP wrapper
            actual_model = accelerator.unwrap_model(model)
            
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=256
            )
            inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = actual_model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.8,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # Parse predictions and reasoning steps
            batch_predictions = []
            batch_steps = []
            
            for j, output in enumerate(outputs):
                try:
                    input_length = inputs['input_ids'][j].shape[0]
                    generated_tokens = output[input_length:]
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    # Parse answer and steps
                    lines = generated_text.strip().split('\n')
                    answer = ""
                    steps = []
                    
                    for line in lines:
                        line = line.strip()
                        if line.startswith('Step'):
                            steps.append(line)
                        elif line and not line.startswith('Q:'):
                            if not answer:
                                answer = line
                    
                    if not answer and steps:
                        answer = steps[-1].split(':')[-1].strip() if ':' in steps[-1] else steps[-1]
                    
                    if not answer:
                        answer = generated_text.strip().split('\n')[0]
                    
                    batch_predictions.append(answer)
                    batch_steps.append(steps if steps else ["No clear reasoning steps"])
                    
                except Exception as e:
                    accelerator.print(f"Error parsing output {j}: {e}")
                    batch_predictions.append("Error in generation")
                    batch_steps.append(["Error in reasoning"])
            
            all_preds.extend(batch_predictions)
            all_steps.extend(batch_steps)
            
        except Exception as e:
            accelerator.print(f"Error in batch {i}: {e}")
            # Add error entries for this batch
            batch_size_actual = len(batch_prompts)
            all_preds.extend(["Generation error"] * batch_size_actual)
            all_steps.extend([["Generation error"]] * batch_size_actual)
    
    # Ensure correct length
    while len(all_preds) < len(df):
        all_preds.append("Missing prediction")
        all_steps.append(["Missing reasoning"])
    
    all_preds = all_preds[:len(df)]
    all_steps = all_steps[:len(df)]
    
    # Create result
    df_result = df.copy()
    df_result['model_prediction'] = all_preds
    df_result['model_reasoning_steps'] = all_steps
    
    return df_result

# --- Robustness Metrics (Distributed) ---
def compute_robustness_metrics(results, accelerator):
    """Compute comprehensive robustness metrics in distributed setting"""
    metrics = {}
    baseline_results = results['original']
    
    for variant_name, variant_results in results.items():
        if variant_name == 'original':
            continue
            
        accelerator.print(f"\n📊 Computing metrics for {variant_name}...")
        
        # Exact Match comparison
        baseline_em = compute_exact_match_scores(baseline_results)
        variant_em = compute_exact_match_scores(variant_results)
        
        # CoT-EM comparison (step-by-step reasoning)
        baseline_cot = compute_cot_em_scores(baseline_results)
        variant_cot = compute_cot_em_scores(variant_results)
        
        # Semantic similarity of answers
        semantic_sim = compute_semantic_consistency(
            baseline_results['model_prediction'].tolist(),
            variant_results['model_prediction'].tolist()
        )
        
        # Reasoning path consistency
        reasoning_consistency = compute_reasoning_consistency(
            baseline_results['model_reasoning_steps'].tolist(),
            variant_results['model_reasoning_steps'].tolist()
        )
        
        # Confidence degradation (based on answer length and coherence)
        confidence_score = compute_confidence_score(variant_results)
        baseline_confidence = compute_confidence_score(baseline_results)
        
        metrics[variant_name] = {
            'exact_match_drop': baseline_em - variant_em,
            'cot_em_drop': baseline_cot - variant_cot,
            'semantic_consistency': semantic_sim,
            'reasoning_consistency': reasoning_consistency,
            'confidence_degradation': baseline_confidence - confidence_score,
            'robustness_score': compute_overall_robustness(
                baseline_em - variant_em, baseline_cot - variant_cot, 
                semantic_sim, reasoning_consistency
            ),
            'baseline_em': baseline_em,
            'variant_em': variant_em,
            'baseline_cot': baseline_cot,
            'variant_cot': variant_cot
        }
    
    return metrics

def compute_exact_match_scores(df):
    """Compute exact match scores"""
    def normalize_text(text):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    matches = 0
    total = 0
    
    for _, row in df.iterrows():
        pred_norm = normalize_text(row['model_prediction'])
        gold_norm = normalize_text(row['answer'])
        
        if pred_norm == gold_norm or gold_norm in pred_norm or pred_norm in gold_norm:
            matches += 1
        total += 1
    
    return matches / total if total > 0 else 0

def compute_cot_em_scores(df):
    """Compute Chain-of-Thought exact match scores"""
    def extract_reasoning_quality(steps):
        if not steps or not steps[0]:
            return 0
        
        quality_score = 0
        step_text = ' '.join(steps).lower()
        
        # Check for reasoning indicators
        reasoning_words = ['because', 'therefore', 'since', 'due to', 'leads to', 'results in']
        quality_score += sum(1 for word in reasoning_words if word in step_text) * 0.2
        
        # Check for step structure
        step_count = len([s for s in steps if s.strip().startswith(('Step', 'step'))])
        quality_score += min(step_count * 0.3, 1.0)
        
        # Check for factual content
        if any(word in step_text for word in ['capital', 'president', 'country', 'city', 'river']):
            quality_score += 0.2
        
        return min(quality_score, 1.0)
    
    total_quality = 0
    count = 0
    
    for _, row in df.iterrows():
        steps = row.get('model_reasoning_steps', [])
        quality = extract_reasoning_quality(steps)
        total_quality += quality
        count += 1
    
    return total_quality / count if count > 0 else 0

def compute_semantic_consistency(baseline_answers, variant_answers):
    """Compute semantic consistency between baseline and variant answers"""
    if len(baseline_answers) != len(variant_answers):
        return 0
    
    # Use TF-IDF for semantic similarity
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    
    try:
        all_answers = baseline_answers + variant_answers
        tfidf_matrix = vectorizer.fit_transform(all_answers)
        
        baseline_vectors = tfidf_matrix[:len(baseline_answers)]
        variant_vectors = tfidf_matrix[len(baseline_answers):]
        
        similarities = []
        for i in range(len(baseline_answers)):
            sim = cosine_similarity(baseline_vectors[i], variant_vectors[i])[0][0]
            similarities.append(sim)
        
        return np.mean(similarities)
    except:
        return 0

def compute_reasoning_consistency(baseline_steps, variant_steps):
    """Compute consistency of reasoning steps"""
    if len(baseline_steps) != len(variant_steps):
        return 0
    
    consistencies = []
    
    for base_steps, var_steps in zip(baseline_steps, variant_steps):
        if not base_steps or not var_steps:
            consistencies.append(0)
            continue
        
        # Compare step structures
        base_text = ' '.join(base_steps).lower()
        var_text = ' '.join(var_steps).lower()
        
        # Simple word overlap metric
        base_words = set(base_text.split())
        var_words = set(var_text.split())
        
        if len(base_words) == 0 or len(var_words) == 0:
            consistencies.append(0)
        else:
            overlap = len(base_words.intersection(var_words))
            union = len(base_words.union(var_words))
            consistencies.append(overlap / union if union > 0 else 0)
    
    return np.mean(consistencies)

def compute_confidence_score(df):
    """Compute model confidence based on answer characteristics"""
    confidence_scores = []
    
    for _, row in df.iterrows():
        answer = str(row['model_prediction']).strip()
        steps = row.get('model_reasoning_steps', [])
        
        confidence = 0
        
        # Answer length (reasonable answers are neither too short nor too long)
        if 5 <= len(answer.split()) <= 20:
            confidence += 0.3
        
        # Presence of reasoning steps
        if steps and len(steps) > 1:
            confidence += 0.3
        
        # Absence of uncertainty markers
        uncertainty_markers = ['maybe', 'perhaps', 'possibly', 'might be', 'could be', 'not sure']
        if not any(marker in answer.lower() for marker in uncertainty_markers):
            confidence += 0.2
        
        # Coherence (no repetitive or error patterns)
        if not any(error in answer.lower() for error in ['error', 'no answer', 'cannot', 'unable']):
            confidence += 0.2
        
        confidence_scores.append(confidence)
    
    return np.mean(confidence_scores)

def compute_overall_robustness(em_drop, cot_drop, semantic_sim, reasoning_sim):
    """Compute overall robustness score (higher = more robust)"""
    # Invert drops (lower drops = higher robustness)
    robustness = (1 - em_drop) * 0.3 + (1 - cot_drop) * 0.3 + semantic_sim * 0.2 + reasoning_sim * 0.2
    return max(0, min(1, robustness))
# --- Distributed Visualizations ---
def create_distributed_visualizations(metrics, save_dir, accelerator):
    """Create visualizations only on main process"""
    if not accelerator.is_main_process:
        return
    
    if not metrics:
        accelerator.print("No metrics to visualize")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        plt.style.use('default')
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        variants = list(metrics.keys())
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Distributed Robustness Analysis - Qwen3-0.6B on MenatQA', fontsize=16, fontweight='bold')
        
        # 1. Robustness scores
        robustness_scores = [metrics[v]['robustness_score'] for v in variants]
        bars = axes[0,0].bar(variants, robustness_scores, color=colors[:len(variants)])
        axes[0,0].set_title('Robustness Scores')
        axes[0,0].set_ylabel('Score (Higher = Better)')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].set_ylim(0, 1)
        
        # Add values on bars
        for bar, score in zip(bars, robustness_scores):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Performance drops
        em_drops = [metrics[v]['exact_match_drop'] for v in variants]
        axes[0,1].bar(variants, em_drops, color='red', alpha=0.7)
        axes[0,1].set_title('Exact Match Performance Drops')
        axes[0,1].set_ylabel('Performance Drop')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Semantic consistency
        semantic_scores = [metrics[v]['semantic_consistency'] for v in variants]
        axes[1,0].bar(variants, semantic_scores, color='green', alpha=0.7)
        axes[1,0].set_title('Semantic Consistency')
        axes[1,0].set_ylabel('Consistency Score')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].set_ylim(0, 1)
        
        # 4. Summary comparison
        metrics_data = {
            'Robustness': robustness_scores,
            'EM Drop': em_drops,
            'Semantic Cons.': semantic_scores
        }
        
        x = range(len(variants))
        width = 0.25
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            offset = (i - 1) * width
            axes[1,1].bar([pos + offset for pos in x], values, width, 
                         label=metric_name, color=colors[i], alpha=0.7)
        
        axes[1,1].set_title('Metrics Comparison')
        axes[1,1].set_xlabel('Perturbation Type')
        axes[1,1].set_ylabel('Score')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(variants, rotation=45)
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/distributed_robustness_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        accelerator.print(f"✅ Visualization saved to {save_dir}/distributed_robustness_analysis.png")
        
    except Exception as e:
        accelerator.print(f"Error creating visualizations: {e}")

def print_distributed_summary(metrics, accelerator):
    """Print comprehensive summary only on main process"""
    if not accelerator.is_main_process:
        return
    
    accelerator.print("\n" + "="*70)
    accelerator.print("🎯 DISTRIBUTED ROBUSTNESS ANALYSIS SUMMARY - Qwen3-0.6B on MenatQA")
    accelerator.print("="*70)
    
    variants = list(metrics.keys())
    
    # Overall robustness ranking
    robustness_ranking = sorted(variants, key=lambda x: metrics[x]['robustness_score'], reverse=True)
    
    accelerator.print(f"\n🏆 ROBUSTNESS RANKING (Higher = More Robust):")
    for i, variant in enumerate(robustness_ranking, 1):
        score = metrics[variant]['robustness_score']
        accelerator.print(f"   {i}. {variant:<20} Score: {score:.3f}")
    
    accelerator.print(f"\n📉 PERFORMANCE DROPS:")
    accelerator.print(f"{'Perturbation':<20} {'EM Drop':<10} {'CoT Drop':<10} {'Confidence Loss':<15}")
    accelerator.print("-" * 55)
    
    for variant in variants:
        em_drop = metrics[variant]['exact_match_drop']
        cot_drop = metrics[variant]['cot_em_drop']
        conf_loss = metrics[variant]['confidence_degradation']
        accelerator.print(f"{variant:<20} {em_drop:<10.3f} {cot_drop:<10.3f} {conf_loss:<15.3f}")
    
    accelerator.print(f"\n🔗 CONSISTENCY SCORES:")
    accelerator.print(f"{'Perturbation':<20} {'Semantic':<10} {'Reasoning':<10}")
    accelerator.print("-" * 40)
    
    for variant in variants:
        sem_cons = metrics[variant]['semantic_consistency']
        reas_cons = metrics[variant]['reasoning_consistency']
        accelerator.print(f"{variant:<20} {sem_cons:<10.3f} {reas_cons:<10.3f}")
    
    # Key insights
    accelerator.print(f"\n🔍 KEY INSIGHTS:")
    
    # Most vulnerable perturbation
    worst_variant = min(variants, key=lambda x: metrics[x]['robustness_score'])
    worst_score = metrics[worst_variant]['robustness_score']
    accelerator.print(f"   • Most Vulnerable to: {worst_variant} (Score: {worst_score:.3f})")
    
    # Most robust aspect
    best_variant = max(variants, key=lambda x: metrics[x]['robustness_score'])
    best_score = metrics[best_variant]['robustness_score']
    accelerator.print(f"   • Most Robust against: {best_variant} (Score: {best_score:.3f})")
    
    # Overall assessment
    avg_robustness = np.mean([metrics[v]['robustness_score'] for v in variants])
    accelerator.print(f"\n📊 OVERALL ASSESSMENT:")
    accelerator.print(f"   • Average Robustness Score: {avg_robustness:.3f}")
    
    if avg_robustness > 0.7:
        assessment = "Model shows STRONG robustness across perturbations"
    elif avg_robustness > 0.5:
        assessment = "Model shows MODERATE robustness with some vulnerabilities"
    else:
        assessment = "Model shows WEAK robustness - significant vulnerabilities detected"
    
    accelerator.print(f"   • {assessment}")
    accelerator.print("="*70)

def save_distributed_results(metrics, results, save_dir, accelerator):
    """Save results only on main process"""
    if not accelerator.is_main_process:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics as JSON
    metrics_serializable = {}
    for variant, metric_dict in metrics.items():
        metrics_serializable[variant] = {}
        for key, value in metric_dict.items():
            if isinstance(value, (np.integer, np.floating)):
                metrics_serializable[variant][key] = float(value)
            else:
                metrics_serializable[variant][key] = value
    
    with open(f"{save_dir}/distributed_robustness_metrics.json", 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    # Save detailed results as CSV
    all_results = []
    for variant_name, df_result in results.items():
        df_result_copy = df_result.copy()
        df_result_copy['perturbation_type'] = variant_name
        all_results.append(df_result_copy)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(f"{save_dir}/distributed_detailed_results.csv", index=False)
    
    accelerator.print(f"✅ Distributed results saved to {save_dir}")

# --- Enhanced Main Function ---
def main():
    """Enhanced main function with comprehensive robustness analysis"""
    accelerator = Accelerator()
    
    # Configuration
    DATASET_FILE = './MenatQA.json'
    MODEL_NAME = 'Qwen/Qwen3-0.6B'
    SAVE_DIR = './robustness_multi_results'
    BATCH_SIZE = 16
    
    accelerator.print("🚀 Starting Distributed Perturbation-Based Robustness Analysis")
    accelerator.print("="*70)
    accelerator.print(f"Process {accelerator.process_index}/{accelerator.num_processes}")
    accelerator.print(f"Device: {accelerator.device}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Load data
    accelerator.print("📂 Loading MenatQA dataset...")
    try:
        data = json.load(open(DATASET_FILE, 'r'))
        df = pd.DataFrame(data)
        df = df[df['question'].notnull() & df['answer'].notnull()].reset_index(drop=True)
        accelerator.print(f"✅ Loaded {len(df)} valid questions")
    except Exception as e:
        accelerator.print(f"❌ Error loading dataset: {e}")
        return
    
    # Create perturbations
    accelerator.print("🔄 Creating perturbation variants...")
    perturber = RobustnessPerturber(seed=42)
    variants = create_perturbation_variants(df, perturber)
    
    # Load model
    accelerator.print(f"🤖 Loading model: {MODEL_NAME}")
    hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    try:
        model, tokenizer = load_qwen_model(MODEL_NAME, accelerator, hf_token)
        accelerator.print("✅ Model loaded successfully")
    except Exception as e:
        accelerator.print(f"❌ Error loading model: {e}")
        return
    
    # Run inference on all variants
    accelerator.print(f"🔬 Running inference on {len(variants)} variants...")
    results = {}
    
    for variant_name, df_variant in variants.items():
        accelerator.print(f"\n[Rank {accelerator.process_index}] Evaluating: {variant_name}")
        try:
            df_result = run_fast_predictions(df_variant, model, tokenizer, accelerator, BATCH_SIZE)
            results[variant_name] = df_result
            
            if accelerator.is_main_process:
                df_result.to_csv(f"{SAVE_DIR}/results_{variant_name}.csv", index=False)
                accelerator.print(f"✅ Saved results for {variant_name}")
        except Exception as e:
            accelerator.print(f"❌ Error processing {variant_name}: {e}")
    
    # Compute metrics (only on main process to avoid duplication)
    if accelerator.is_main_process and len(results) > 1:
        accelerator.print("\n📊 Computing robustness metrics...")
        try:
            metrics = compute_robustness_metrics(results, accelerator)
            
            # Print summary
            print_distributed_summary(metrics, accelerator)
            
            # Create visualizations
            create_distributed_visualizations(metrics, SAVE_DIR, accelerator)
            
            # Save comprehensive results
            save_distributed_results(metrics, results, SAVE_DIR, accelerator)
            
        except Exception as e:
            accelerator.print(f"❌ Error in metrics computation: {e}")
    
    accelerator.print(f"\n✅ Distributed robustness analysis completed!")
    accelerator.print(f"Results saved to: {SAVE_DIR}")

if __name__ == '__main__':
    main()
