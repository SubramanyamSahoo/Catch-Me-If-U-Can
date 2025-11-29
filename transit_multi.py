"""
transit_multi.py
Distributed/multinode logical consistency and transitivity analysis for Qwen3-0.6B using Accelerate.
V100-optimized version with comprehensive transitivity evaluation from transitivity.py.
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
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, deque
from scipy import stats
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# V100 optimized settings for transitivity analysis
V100_TRANSITIVITY_BATCH_SIZE = 16  # Conservative for complex reasoning
V100_TRANSITIVITY_MAX_LENGTH = 512  # Longer sequences for reasoning chains
V100_TRANSITIVITY_SAMPLE_SIZE = None  # Process full dataset like original transitivity.py

# --- Utility Functions ---
def normalize_answer(answer):
    if not isinstance(answer, str):
        answer = str(answer)
    return re.sub(r'[^\w\s]', '', answer.lower().strip())

# --- Enhanced Logical Reasoning Classes (V100 Optimized) ---
class LogicalStep:
    """Represents a single logical reasoning step - V100 optimized"""
    
    def __init__(self, step_text, step_number=None):
        self.original_text = step_text.strip() if isinstance(step_text, str) else ""
        self.step_number = step_number
        self.entities = self.extract_entities()
        self.relations = self.extract_relations()
        self.values = self.extract_values()
        self.logical_form = self.parse_logical_form()
    
    def extract_entities(self):
        """Extract named entities from the step - optimized for V100 processing"""
        if not self.original_text:
            return []
        
        entities = []
        words = self.original_text.split()
        for word in words:
            clean_word = word.strip('.,!?')
            if len(clean_word) > 2 and (clean_word[0].isupper() or clean_word.lower() in 
                ['capital', 'president', 'country', 'city', 'river', 'mountain']):
                entities.append(clean_word)
        return list(set(entities))  # Remove duplicates
    
    def extract_relations(self):
        """Extract relational words - memory efficient"""
        relation_keywords = [
            'is', 'was', 'are', 'were', 'has', 'have', 'located', 'situated',
            'capital', 'president', 'leader', 'flows', 'connects', 'borders'
        ]
        
        relations = []
        text_lower = self.original_text.lower()
        for keyword in relation_keywords:
            if keyword in text_lower:
                relations.append(keyword)
        return relations
    
    def extract_values(self):
        """Extract numerical and categorical values"""
        values = []
        # Extract numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', self.original_text)
        values.extend(numbers)
        
        # Extract common categorical values
        categories = ['north', 'south', 'east', 'west', 'large', 'small', 'ancient', 'modern']
        text_lower = self.original_text.lower()
        for cat in categories:
            if cat in text_lower:
                values.append(cat)
        return values
    
    def parse_logical_form(self):
        """Parse into simple logical representation"""
        if not self.entities or not self.relations:
            return None
        
        return {
            'subject': self.entities[0] if self.entities else None,
            'predicate': self.relations[0] if self.relations else None,
            'object': self.entities[1] if len(self.entities) > 1 else None,
            'values': self.values
        }

class ReasoningChain:
    """Represents a complete reasoning chain - V100 optimized"""
    
    def __init__(self, question, steps, answer):
        self.question = question
        self.steps = [LogicalStep(step, i) for i, step in enumerate(steps) if step.strip()]
        self.answer = answer
        self.logical_graph = self.build_logical_graph()
        self.transitive_closure = self.get_transitive_closure()
    
    def build_logical_graph(self):
        """Build graph representation of reasoning chain"""
        G = nx.DiGraph()
        
        for step in self.steps:
            if step.logical_form:
                subject = step.logical_form.get('subject')
                predicate = step.logical_form.get('predicate')
                obj = step.logical_form.get('object')
                
                if subject and predicate:
                    if obj:
                        G.add_edge(subject, obj, relation=predicate, step=step.step_number)
                    else:
                        G.add_node(subject, predicate=predicate, step=step.step_number)
        
        return G
    
    def get_transitive_closure(self):
        """Compute transitive closure for transitivity analysis"""
        if not self.logical_graph.nodes():
            return nx.DiGraph()
        
        # Compute transitive closure
        tc = nx.transitive_closure(self.logical_graph)
        return tc
    
    def infer_transitive_relation(self, step_i, step_j):
        """Check if step_j can be inferred from step_i transitively"""
        if not self.steps or step_i >= len(self.steps) or step_j >= len(self.steps):
            return False
        
        step_a = self.steps[step_i]
        step_b = self.steps[step_j]
        
        if not (step_a.logical_form and step_b.logical_form):
            return False
        
        # Simple transitivity check
        a_obj = step_a.logical_form.get('object')
        b_subj = step_b.logical_form.get('subject')
        
        return a_obj and b_subj and a_obj.lower() == b_subj.lower()

class DistributedLogicalConsistencyEvaluator:
    """V100-optimized distributed evaluator for logical consistency"""
    
    def __init__(self, model, tokenizer, accelerator):
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.device = accelerator.device
    
    def generate_forward_reasoning(self, question, max_new_tokens=150):
        """Generate forward reasoning with V100 optimization"""
        prompt = f"""Question: {question}

Please provide step-by-step reasoning to answer this question.

Step 1:"""
        
        try:
            # Get actual model from DDP wrapper
            actual_model = self.accelerator.unwrap_model(self.model)
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=V100_TRANSITIVITY_MAX_LENGTH
            ).to(self.device)
            
            with torch.no_grad():
                outputs = actual_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Extract generated text
            input_length = inputs['input_ids'].shape[1]
            generated = outputs[0][input_length:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True)
            
            return self.parse_reasoning_steps(response)
            
        except Exception as e:
            self.accelerator.print(f"Error in forward reasoning: {e}")
            return []
    
    def generate_backward_reasoning(self, question, answer, max_new_tokens=150):
        """Generate backward reasoning from answer to question"""
        prompt = f"""Given the answer "{answer}" to the question "{question}", 
please explain step-by-step how this answer was derived.

Explanation:
Step 1:"""
        
        try:
            actual_model = self.accelerator.unwrap_model(self.model)
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=V100_TRANSITIVITY_MAX_LENGTH
            ).to(self.device)
            
            with torch.no_grad():
                outputs = actual_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            input_length = inputs['input_ids'].shape[1]
            generated = outputs[0][input_length:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True)
            
            return self.parse_backward_steps(response)
            
        except Exception as e:
            self.accelerator.print(f"Error in backward reasoning: {e}")
            return []
    
    def parse_reasoning_steps(self, text):
        """Parse forward reasoning steps from generated text"""
        if not text:
            return []
        
        lines = text.split('\n')
        steps = []
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('Step') or ':' in line):
                # Clean up the step text
                if ':' in line:
                    step_text = line.split(':', 1)[1].strip()
                else:
                    step_text = line
                if step_text:
                    steps.append(step_text)
        
        return steps if steps else [text.strip()]
    
    def parse_backward_steps(self, text):
        """Parse backward reasoning steps"""
        return self.parse_reasoning_steps(text)  # Same parsing logic
    
    def compute_consistency_score(self, forward_steps, backward_steps):
        """Compute consistency between forward and backward reasoning"""
        if not forward_steps or not backward_steps:
            return 0.0
        
        # Convert to text for comparison
        forward_text = ' '.join(forward_steps).lower()
        backward_text = ' '.join(backward_steps).lower()
        
        # Simple word overlap metric
        forward_words = set(forward_text.split())
        backward_words = set(backward_text.split())
        
        if not forward_words or not backward_words:
            return 0.0
        
        intersection = len(forward_words.intersection(backward_words))
        union = len(forward_words.union(backward_words))
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_transitivity(self, reasoning_chain):
        """Evaluate transitivity in reasoning chain"""
        if not reasoning_chain.steps or len(reasoning_chain.steps) < 2:
            return 0.0
        
        transitivity_score = 0.0
        valid_transitions = 0
        total_transitions = 0
        
        # Check all pairs of steps for transitivity
        for i in range(len(reasoning_chain.steps) - 1):
            for j in range(i + 1, len(reasoning_chain.steps)):
                total_transitions += 1
                
                if reasoning_chain.infer_transitive_relation(i, j):
                    valid_transitions += 1
                
                # Check for logical flow
                step_i = reasoning_chain.steps[i]
                step_j = reasoning_chain.steps[j]
                
                if self.is_valid_transitive_step(step_i, step_j):
                    transitivity_score += 0.1
        
        # Normalize score
        base_score = valid_transitions / total_transitions if total_transitions > 0 else 0
        flow_score = min(transitivity_score, 1.0)
        
        return (base_score + flow_score) / 2
    
    def is_valid_transitive_step(self, step_from, step_to):
        """Check if step_to follows logically from step_from"""
        if not (step_from.entities and step_to.entities):
            return False
        
        # Check for entity overlap
        entities_from = set(e.lower() for e in step_from.entities)
        entities_to = set(e.lower() for e in step_to.entities)
        
        return bool(entities_from.intersection(entities_to))

# --- Dataset Processing (V100 Optimized) ---
def load_and_prepare_menatqa_distributed(accelerator, sample_size=None):
    """Load MenatQA dataset for distributed logical consistency testing"""
    file_path = Path('./MenatQA.json')
    
    if not file_path.exists() and accelerator.is_main_process:
        accelerator.print("Downloading MenatQA dataset...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/weiyifan1023/MenatQA/main/datasets/MenatQA.json",
            str(file_path)
        )
    
    # Wait for main process to download
    accelerator.wait_for_everyone()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process for logical consistency testing
    processed_data = []
    for i, item in enumerate(data):
        try:
            if 'question' in item and 'answer' in item:
                # Extract reasoning complexity
                question = str(item['question']).strip()
                answer = str(item['answer']).strip()
                
                if question and answer:
                    # Determine hop complexity
                    hop_indicators = ['then', 'next', 'after that', 'following', 'subsequently']
                    hop_count = sum(1 for indicator in hop_indicators if indicator in question.lower())
                    
                    complexity = '1-hop'
                    if hop_count >= 3:
                        complexity = '4+-hop'
                    elif hop_count >= 2:
                        complexity = '3-hop'
                    elif hop_count >= 1:
                        complexity = '2-hop'
                    
                    processed_data.append({
                        'id': i,
                        'question': question,
                        'answer': answer,
                        'complexity': complexity,
                        'hop_count': hop_count
                    })
        except Exception as e:
            continue
    
    df = pd.DataFrame(processed_data)
    
    # Apply sampling
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    accelerator.print(f"✅ Loaded {len(df)} questions for logical consistency testing")
    return df

# --- Model Loading (V100 Optimized) ---
def load_qwen_model_distributed(model_name, accelerator, hf_token=None):
    """Load Qwen model optimized for V100 transitivity analysis"""
    accelerator.print(f"Loading {model_name} for V100 transitivity analysis...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # V100 optimized
        trust_remote_code=True,
        token=hf_token,
        device_map=None,
        low_cpu_mem_usage=True
    )
    
    # Prepare model with accelerator
    model = accelerator.prepare(model)
    
    return model, tokenizer

# --- Main Evaluation Pipeline (V100 Distributed) ---
def run_distributed_logical_consistency_evaluation(df, model, tokenizer, accelerator, batch_size=8, master_pbar=None, eval_weight=70):
    """Run distributed logical consistency evaluation on V100s"""
    
    evaluator = DistributedLogicalConsistencyEvaluator(model, tokenizer, accelerator)
    
    results = []
    consistency_scores = []
    transitivity_scores = []
    
    accelerator.print(f"🔍 Evaluating logical consistency for {len(df)} questions...")
    
    # Calculate number of batches for progress tracking
    total_batches = (len(df) + batch_size - 1) // batch_size
    eval_progress_per_batch = eval_weight / total_batches if total_batches > 0 else 0
    
    # Process in batches for memory efficiency
    batch_pbar = tqdm(range(0, len(df), batch_size), desc="Processing Questions", 
                     disable=not accelerator.is_local_main_process, position=1, leave=False)
    
    for batch_start in batch_pbar:
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        # Update batch progress description
        batch_pbar.set_description(f"Batch {batch_start//batch_size + 1}/{total_batches}")
        
        for idx, row in batch_df.iterrows():
            try:
                question = row['question']
                answer = row['answer']
                complexity = row['complexity']
                
                # Generate forward reasoning
                forward_steps = evaluator.generate_forward_reasoning(question)
                
                # Generate backward reasoning  
                backward_steps = evaluator.generate_backward_reasoning(question, answer)
                
                # Compute consistency score
                consistency_score = evaluator.compute_consistency_score(forward_steps, backward_steps)
                
                # Create reasoning chain and evaluate transitivity
                reasoning_chain = ReasoningChain(question, forward_steps, answer)
                transitivity_score = evaluator.evaluate_transitivity(reasoning_chain)
                
                result = {
                    'question_id': idx,
                    'question': question,
                    'answer': answer,
                    'complexity': complexity,
                    'forward_steps': forward_steps,
                    'backward_steps': backward_steps,
                    'consistency_score': consistency_score,
                    'transitivity_score': transitivity_score,
                    'forward_step_count': len(forward_steps),
                    'backward_step_count': len(backward_steps),
                    'reasoning_chain': reasoning_chain
                }
                
                results.append(result)
                consistency_scores.append(consistency_score)
                transitivity_scores.append(transitivity_score)
                
                # Clear cache periodically
                if len(results) % 10 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                accelerator.print(f"Error processing question {idx}: {e}")
                continue
        
        # Update master progress bar
        if master_pbar and accelerator.is_main_process:
            master_pbar.update(eval_progress_per_batch)
            master_pbar.set_description(f"🔬 Evaluating ({len(results)}/{len(df)} questions)")
    
    batch_pbar.close()
    
    # Compile evaluation metrics
    evaluation_metrics = {
        'overall_consistency': np.mean(consistency_scores) if consistency_scores else 0,
        'overall_transitivity': np.mean(transitivity_scores) if transitivity_scores else 0,
        'consistency_by_complexity': compute_complexity_breakdown(results, 'consistency_score'),
        'transitivity_by_complexity': compute_complexity_breakdown(results, 'transitivity_score'),
        'detailed_results': results,
        'sample_size': len(results),
        'forward_avg_steps': np.mean([r['forward_step_count'] for r in results]) if results else 0,
        'backward_avg_steps': np.mean([r['backward_step_count'] for r in results]) if results else 0
    }
    
    return evaluation_metrics

def compute_complexity_breakdown(results, metric_key):
    """Compute metrics breakdown by reasoning complexity"""
    complexity_groups = defaultdict(list)
    
    for result in results:
        complexity = result.get('complexity', 'unknown')
        score = result.get(metric_key, 0)
        complexity_groups[complexity].append(score)
    
    breakdown = {}
    for complexity, scores in complexity_groups.items():
        breakdown[complexity] = {
            'mean': np.mean(scores) if scores else 0,
            'std': np.std(scores) if scores else 0,
            'count': len(scores)
        }
    
    return breakdown

# --- Advanced Analysis (V100 Optimized) ---
def analyze_distributed_reasoning_patterns(evaluation_metrics, accelerator):
    """Analyze patterns in logical reasoning - distributed version"""
    if not accelerator.is_main_process:
        return {}
    
    results = evaluation_metrics['detailed_results']
    
    analysis = {
        'forward_vs_backward_alignment': analyze_forward_backward_alignment(results),
        'transitivity_violations': analyze_transitivity_violations(results),
        'complexity_impact': analyze_complexity_impact(results),
        'reasoning_quality': analyze_reasoning_quality(results)
    }
    
    return analysis

def analyze_forward_backward_alignment(results):
    """Analyze alignment between forward and backward reasoning"""
    alignment_analysis = {
        'high_consistency': [],
        'medium_consistency': [],
        'low_consistency': [],
        'patterns': {}
    }
    
    for result in results:
        consistency = result['consistency_score']
        if consistency >= 0.7:
            alignment_analysis['high_consistency'].append(result)
        elif consistency >= 0.4:
            alignment_analysis['medium_consistency'].append(result)
        else:
            alignment_analysis['low_consistency'].append(result)
    
    # Analyze patterns
    total_results = len(results)
    alignment_analysis['patterns'] = {
        'high_consistency_rate': len(alignment_analysis['high_consistency']) / total_results if total_results > 0 else 0,
        'medium_consistency_rate': len(alignment_analysis['medium_consistency']) / total_results if total_results > 0 else 0,
        'low_consistency_rate': len(alignment_analysis['low_consistency']) / total_results if total_results > 0 else 0
    }
    
    return alignment_analysis

def analyze_transitivity_violations(results):
    """Analyze transitivity violations in reasoning"""
    violations_analysis = {
        'total_violations': 0,
        'violation_patterns': defaultdict(int),
        'violation_examples': []
    }
    
    for result in results:
        transitivity = result['transitivity_score']
        if transitivity < 0.5:  # Consider low transitivity as violation
            violations_analysis['total_violations'] += 1
            complexity = result['complexity']
            violations_analysis['violation_patterns'][complexity] += 1
            
            if len(violations_analysis['violation_examples']) < 5:
                violations_analysis['violation_examples'].append({
                    'question': result['question'],
                    'transitivity_score': transitivity,
                    'complexity': complexity
                })
    
    return violations_analysis

def analyze_complexity_impact(results):
    """Analyze how reasoning complexity affects logical consistency"""
    complexity_impact = {}
    
    complexity_groups = defaultdict(list)
    for result in results:
        complexity = result['complexity']
        complexity_groups[complexity].append(result)
    
    for complexity, group_results in complexity_groups.items():
        consistency_scores = [r['consistency_score'] for r in group_results]
        transitivity_scores = [r['transitivity_score'] for r in group_results]
        
        complexity_impact[complexity] = {
            'count': len(group_results),
            'avg_consistency': np.mean(consistency_scores) if consistency_scores else 0,
            'avg_transitivity': np.mean(transitivity_scores) if transitivity_scores else 0,
            'consistency_std': np.std(consistency_scores) if consistency_scores else 0,
            'transitivity_std': np.std(transitivity_scores) if transitivity_scores else 0
        }
    
    return complexity_impact

def analyze_reasoning_quality(results):
    """Analyze overall quality of reasoning chains"""
    if not results:
        return {}
    
    forward_lengths = [r['forward_step_count'] for r in results]
    backward_lengths = [r['backward_step_count'] for r in results]
    consistency_scores = [r['consistency_score'] for r in results]
    
    quality_metrics = {
        'avg_forward_steps': np.mean(forward_lengths),
        'avg_backward_steps': np.mean(backward_lengths),
        'step_length_correlation': 0,
        'quality_distribution': {'high': 0, 'medium': 0, 'low': 0}
    }
    
    # Calculate correlations
    if len(set(forward_lengths)) > 1 and len(set(consistency_scores)) > 1:
        quality_metrics['step_length_correlation'] = np.corrcoef(forward_lengths, consistency_scores)[0, 1]
    
    # Quality distribution
    for result in results:
        consistency = result['consistency_score']
        if consistency >= 0.7:
            quality_metrics['quality_distribution']['high'] += 1
        elif consistency >= 0.4:
            quality_metrics['quality_distribution']['medium'] += 1
        else:
            quality_metrics['quality_distribution']['low'] += 1
    
    return quality_metrics

# --- Visualization Functions (V100 Optimized) ---
def create_distributed_transitivity_visualizations(evaluation_metrics, analysis_results, save_dir, accelerator):
    """Create visualizations only on main process"""
    if not accelerator.is_main_process:
        return
    
    if not evaluation_metrics:
        accelerator.print("No metrics to visualize")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        plt.style.use('default')
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83', '#1B998B']
        
        # Create comprehensive transitivity analysis plots
        create_transitivity_overview(evaluation_metrics, save_dir, colors)
        create_complexity_analysis_plot(evaluation_metrics, save_dir, colors)
        create_consistency_heatmap(evaluation_metrics, save_dir, colors)
        create_reasoning_quality_plot(analysis_results, save_dir, colors)
        
        accelerator.print(f"✅ All transitivity visualizations saved to {save_dir}")
        
    except Exception as e:
        accelerator.print(f"Error creating visualizations: {e}")

def create_transitivity_overview(evaluation_metrics, save_dir, colors):
    """Create overview of transitivity and consistency metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distributed Logical Consistency & Transitivity Analysis - Qwen3-0.6B', fontsize=16, fontweight='bold')
    
    # Overall scores
    overall_consistency = evaluation_metrics['overall_consistency']
    overall_transitivity = evaluation_metrics['overall_transitivity']
    
    metrics = ['Consistency Score', 'Transitivity Score']
    scores = [overall_consistency, overall_transitivity]
    
    bars = axes[0,0].bar(metrics, scores, color=colors[:2], alpha=0.8)
    axes[0,0].set_title('Overall Logical Reasoning Scores', fontweight='bold')
    axes[0,0].set_ylabel('Score (0-1)')
    axes[0,0].set_ylim(0, 1)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Complexity breakdown
    consistency_by_complexity = evaluation_metrics['consistency_by_complexity']
    transitivity_by_complexity = evaluation_metrics['transitivity_by_complexity']
    
    complexities = list(consistency_by_complexity.keys())
    if complexities:
        cons_scores = [consistency_by_complexity[c]['mean'] for c in complexities]
        trans_scores = [transitivity_by_complexity[c]['mean'] for c in complexities]
        
        x = np.arange(len(complexities))
        width = 0.35
        
        axes[0,1].bar(x - width/2, cons_scores, width, label='Consistency', color=colors[0], alpha=0.7)
        axes[0,1].bar(x + width/2, trans_scores, width, label='Transitivity', color=colors[1], alpha=0.7)
        
        axes[0,1].set_title('Scores by Complexity', fontweight='bold')
        axes[0,1].set_ylabel('Score')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(complexities, rotation=45)
        axes[0,1].legend()
        axes[0,1].set_ylim(0, 1)
    
    # Step count analysis
    forward_avg = evaluation_metrics.get('forward_avg_steps', 0)
    backward_avg = evaluation_metrics.get('backward_avg_steps', 0)
    
    step_types = ['Forward Steps', 'Backward Steps']
    step_counts = [forward_avg, backward_avg]
    
    axes[1,0].bar(step_types, step_counts, color=colors[2:4], alpha=0.7)
    axes[1,0].set_title('Average Reasoning Steps', fontweight='bold')
    axes[1,0].set_ylabel('Average Number of Steps')
    
    # Sample size info
    sample_size = evaluation_metrics.get('sample_size', 0)
    axes[1,1].text(0.5, 0.7, f'Sample Size: {sample_size}', ha='center', fontsize=14, 
                   transform=axes[1,1].transAxes)
    axes[1,1].text(0.5, 0.5, f'Consistency: {overall_consistency:.3f}', ha='center', fontsize=12,
                   transform=axes[1,1].transAxes)
    axes[1,1].text(0.5, 0.3, f'Transitivity: {overall_transitivity:.3f}', ha='center', fontsize=12,
                   transform=axes[1,1].transAxes)
    axes[1,1].set_title('Analysis Summary', fontweight='bold')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/distributed_transitivity_overview.png", dpi=150, bbox_inches='tight')
    plt.close()

def create_complexity_analysis_plot(evaluation_metrics, save_dir, colors):
    """Create detailed complexity analysis plot"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Reasoning Complexity Impact Analysis', fontsize=14, fontweight='bold')
    
    consistency_by_complexity = evaluation_metrics['consistency_by_complexity']
    transitivity_by_complexity = evaluation_metrics['transitivity_by_complexity']
    
    if not consistency_by_complexity:
        return
    
    complexities = list(consistency_by_complexity.keys())
    cons_means = [consistency_by_complexity[c]['mean'] for c in complexities]
    cons_stds = [consistency_by_complexity[c]['std'] for c in complexities]
    trans_means = [transitivity_by_complexity[c]['mean'] for c in complexities]
    trans_stds = [transitivity_by_complexity[c]['std'] for c in complexities]
    
    # Error bar plot
    x = np.arange(len(complexities))
    axes[0].errorbar(x, cons_means, yerr=cons_stds, label='Consistency', 
                     color=colors[0], marker='o', capsize=5)
    axes[0].errorbar(x, trans_means, yerr=trans_stds, label='Transitivity', 
                     color=colors[1], marker='s', capsize=5)
    
    axes[0].set_title('Mean Scores with Standard Deviation')
    axes[0].set_ylabel('Score')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(complexities, rotation=45)
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    
    # Sample count plot
    counts = [consistency_by_complexity[c]['count'] for c in complexities]
    axes[1].bar(complexities, counts, color=colors[2], alpha=0.7)
    axes[1].set_title('Sample Count by Complexity')
    axes[1].set_ylabel('Number of Questions')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/complexity_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

def create_consistency_heatmap(evaluation_metrics, save_dir, colors):
    """Create consistency heatmap"""
    results = evaluation_metrics['detailed_results']
    if not results:
        return
    
    # Create data matrix
    consistency_scores = [r['consistency_score'] for r in results]
    transitivity_scores = [r['transitivity_score'] for r in results]
    complexities = [r['complexity'] for r in results]
    
    # Group by complexity
    complexity_types = list(set(complexities))
    data_matrix = []
    
    for complexity in complexity_types:
        cons_scores = [consistency_scores[i] for i, c in enumerate(complexities) if c == complexity]
        trans_scores = [transitivity_scores[i] for i, c in enumerate(complexities) if c == complexity]
        
        data_matrix.append([
            np.mean(cons_scores) if cons_scores else 0,
            np.mean(trans_scores) if trans_scores else 0
        ])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(2))
    ax.set_xticklabels(['Consistency', 'Transitivity'])
    ax.set_yticks(range(len(complexity_types)))
    ax.set_yticklabels(complexity_types)
    ax.set_title('Logical Reasoning Performance Heatmap', fontweight='bold')
    
    # Add text annotations
    for i in range(len(complexity_types)):
        for j in range(2):
            text = ax.text(j, i, f'{data_matrix[i][j]:.3f}', 
                          ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Score (0-1)')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/consistency_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

def create_reasoning_quality_plot(analysis_results, save_dir, colors):
    """Create reasoning quality analysis plot"""
    if not analysis_results or 'reasoning_quality' not in analysis_results:
        return
    
    quality = analysis_results['reasoning_quality']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Reasoning Quality Analysis', fontsize=14, fontweight='bold')
    
    # Quality distribution
    quality_dist = quality['quality_distribution']
    labels = list(quality_dist.keys())
    sizes = list(quality_dist.values())
    
    if sum(sizes) > 0:
        axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)])
        axes[0].set_title('Quality Distribution')
    
    # Step analysis
    forward_steps = quality.get('avg_forward_steps', 0)
    backward_steps = quality.get('avg_backward_steps', 0)
    
    step_data = ['Forward Steps', 'Backward Steps']
    step_values = [forward_steps, backward_steps]
    
    axes[1].bar(step_data, step_values, color=colors[3:5])
    axes[1].set_title('Average Reasoning Steps')
    axes[1].set_ylabel('Number of Steps')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/reasoning_quality.png", dpi=150, bbox_inches='tight')
    plt.close()

# --- Summary and Saving Functions ---
def print_distributed_transitivity_summary(evaluation_metrics, analysis_results, accelerator):
    """Print comprehensive summary only on main process"""
    if not accelerator.is_main_process:
        return
    
    accelerator.print("\n" + "="*80)
    accelerator.print("🧠 DISTRIBUTED LOGICAL CONSISTENCY & TRANSITIVITY ANALYSIS - Qwen3-0.6B")
    accelerator.print("="*80)
    
    # Overall scores
    overall_consistency = evaluation_metrics['overall_consistency']
    overall_transitivity = evaluation_metrics['overall_transitivity']
    sample_size = evaluation_metrics['sample_size']
    
    accelerator.print(f"\n📊 OVERALL RESULTS (n={sample_size}):")
    accelerator.print(f"   • Logical Consistency Score: {overall_consistency:.3f}")
    accelerator.print(f"   • Transitivity Score: {overall_transitivity:.3f}")
    accelerator.print(f"   • Average Forward Steps: {evaluation_metrics.get('forward_avg_steps', 0):.1f}")
    accelerator.print(f"   • Average Backward Steps: {evaluation_metrics.get('backward_avg_steps', 0):.1f}")
    
    # Complexity breakdown
    accelerator.print(f"\n🔍 PERFORMANCE BY COMPLEXITY:")
    consistency_by_complexity = evaluation_metrics['consistency_by_complexity']
    transitivity_by_complexity = evaluation_metrics['transitivity_by_complexity']
    
    accelerator.print(f"{'Complexity':<15} {'Consistency':<12} {'Transitivity':<12} {'Count':<8}")
    accelerator.print("-" * 50)
    
    for complexity in consistency_by_complexity:
        cons_score = consistency_by_complexity[complexity]['mean']
        trans_score = transitivity_by_complexity[complexity]['mean']
        count = consistency_by_complexity[complexity]['count']
        accelerator.print(f"{complexity:<15} {cons_score:<12.3f} {trans_score:<12.3f} {count:<8}")
    
    # Analysis insights
    if analysis_results:
        accelerator.print(f"\n🔍 KEY INSIGHTS:")
        
        alignment = analysis_results.get('forward_vs_backward_alignment', {})
        if 'patterns' in alignment:
            patterns = alignment['patterns']
            accelerator.print(f"   • High Consistency Questions: {patterns.get('high_consistency_rate', 0):.1%}")
            accelerator.print(f"   • Medium Consistency Questions: {patterns.get('medium_consistency_rate', 0):.1%}")
            accelerator.print(f"   • Low Consistency Questions: {patterns.get('low_consistency_rate', 0):.1%}")
        
        violations = analysis_results.get('transitivity_violations', {})
        if violations:
            accelerator.print(f"   • Transitivity Violations: {violations.get('total_violations', 0)} questions")
        
        quality = analysis_results.get('reasoning_quality', {})
        if quality:
            correlation = quality.get('step_length_correlation', 0)
            accelerator.print(f"   • Step-Consistency Correlation: {correlation:.3f}")
    
    # Assessment
    accelerator.print(f"\n📈 ASSESSMENT:")
    if overall_consistency >= 0.7 and overall_transitivity >= 0.7:
        assessment = "STRONG logical reasoning and transitivity"
    elif overall_consistency >= 0.5 and overall_transitivity >= 0.5:
        assessment = "MODERATE logical reasoning with some gaps"
    else:
        assessment = "WEAK logical reasoning - significant inconsistencies detected"
    
    accelerator.print(f"   • Model shows {assessment}")
    accelerator.print("="*80)

def save_distributed_transitivity_results(evaluation_metrics, analysis_results, save_dir, accelerator):
    """Save results only on main process"""
    if not accelerator.is_main_process:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics as JSON
    metrics_serializable = {}
    for key, value in evaluation_metrics.items():
        if key == 'detailed_results':
            # Save simplified version of detailed results
            simplified_results = []
            for result in value:
                simplified = {k: v for k, v in result.items() if k != 'reasoning_chain'}
                simplified_results.append(simplified)
            metrics_serializable[key] = simplified_results
        elif isinstance(value, (np.integer, np.floating)):
            metrics_serializable[key] = float(value)
        else:
            metrics_serializable[key] = value
    
    with open(f"{save_dir}/distributed_transitivity_metrics.json", 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    # Save analysis results
    if analysis_results:
        with open(f"{save_dir}/distributed_transitivity_analysis.json", 'w') as f:
            json.dump(analysis_results, f, indent=2)
    
    # Save detailed results as CSV
    if evaluation_metrics['detailed_results']:
        results_data = []
        for result in evaluation_metrics['detailed_results']:
            csv_row = {
                'question_id': result['question_id'],
                'question': result['question'],
                'answer': result['answer'],
                'complexity': result['complexity'],
                'consistency_score': result['consistency_score'],
                'transitivity_score': result['transitivity_score'],
                'forward_step_count': result['forward_step_count'],
                'backward_step_count': result['backward_step_count'],
                'forward_steps': '; '.join(result['forward_steps']),
                'backward_steps': '; '.join(result['backward_steps'])
            }
            results_data.append(csv_row)
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(f"{save_dir}/distributed_transitivity_detailed_results.csv", index=False)
    
    accelerator.print(f"✅ Distributed transitivity results saved to {save_dir}")

# --- Main Function ---
def main():
    """Main function for distributed transitivity analysis"""
    accelerator = Accelerator()
    
    # Configuration
    DATASET_FILE = './MenatQA.json'
    MODEL_NAME = 'Qwen/Qwen3-0.6B'
    SAVE_DIR = './logical_consistency_results3'
    BATCH_SIZE = V100_TRANSITIVITY_BATCH_SIZE
    SAMPLE_SIZE = V100_TRANSITIVITY_SAMPLE_SIZE
    
    # Pipeline stages with estimated weights (only show on main process)
    pipeline_stages = [
        ("Loading Dataset", 5),
        ("Loading Model", 15), 
        ("Running Evaluation", 70),
        ("Analyzing Patterns", 8),
        ("Saving Results", 2)
    ]
    total_weight = sum(weight for _, weight in pipeline_stages)
    
    # Initialize master progress bar only on main process
    if accelerator.is_main_process:
        master_pbar = tqdm(total=100, desc="🧠 Overall Progress", 
                          unit="%", position=0, leave=True,
                          bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:3.0f}/100 [{elapsed}<{remaining}, {rate_fmt}]')
    
    accelerator.print("🧠 Starting Distributed Logical Consistency & Transitivity Analysis")
    accelerator.print("="*80)
    accelerator.print(f"Process {accelerator.process_index}/{accelerator.num_processes}")
    accelerator.print(f"Device: {accelerator.device}")
    accelerator.print(f"V100 Batch Size: {BATCH_SIZE}")
    accelerator.print(f"Sample Size: {SAMPLE_SIZE}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    current_progress = 0
    
    # Load data
    accelerator.print("📂 Loading MenatQA dataset...")
    if accelerator.is_main_process:
        master_pbar.set_description("📂 Loading Dataset")
    try:
        df = load_and_prepare_menatqa_distributed(accelerator, SAMPLE_SIZE)
        current_progress += pipeline_stages[0][1]
        if accelerator.is_main_process:
            master_pbar.update(pipeline_stages[0][1])
    except Exception as e:
        if accelerator.is_main_process:
            master_pbar.close()
        accelerator.print(f"❌ Error loading dataset: {e}")
        return
    
    # Load model
    accelerator.print(f"🤖 Loading model: {MODEL_NAME}")
    if accelerator.is_main_process:
        master_pbar.set_description("🤖 Loading Model")
    hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    try:
        model, tokenizer = load_qwen_model_distributed(MODEL_NAME, accelerator, hf_token)
        accelerator.print("✅ Model loaded successfully")
        current_progress += pipeline_stages[1][1]
        if accelerator.is_main_process:
            master_pbar.update(pipeline_stages[1][1])
    except Exception as e:
        if accelerator.is_main_process:
            master_pbar.close()
        accelerator.print(f"❌ Error loading model: {e}")
        return
    
    # Run evaluation
    accelerator.print("🔬 Running distributed logical consistency evaluation...")
    if accelerator.is_main_process:
        master_pbar.set_description("🔬 Running Evaluation")
    try:
        evaluation_metrics = run_distributed_logical_consistency_evaluation(
            df, model, tokenizer, accelerator, BATCH_SIZE, master_pbar, pipeline_stages[2][1]
        )
        accelerator.print("✅ Evaluation completed")
        current_progress += pipeline_stages[2][1]
        if accelerator.is_main_process:
            master_pbar.set_description("✅ Evaluation Complete")
    except Exception as e:
        if accelerator.is_main_process:
            master_pbar.close()
        accelerator.print(f"❌ Error in evaluation: {e}")
        return
    
    # Analyze patterns (only on main process)
    if accelerator.is_main_process:
        master_pbar.set_description("📊 Analyzing Patterns")
        accelerator.print("📊 Analyzing reasoning patterns...")
        try:
            analysis_results = analyze_distributed_reasoning_patterns(evaluation_metrics, accelerator)
            current_progress += pipeline_stages[3][1]
            master_pbar.update(pipeline_stages[3][1])
            
            # Print summary
            print_distributed_transitivity_summary(evaluation_metrics, analysis_results, accelerator)
            
            # Create visualizations
            create_distributed_transitivity_visualizations(evaluation_metrics, analysis_results, SAVE_DIR, accelerator)
            
            # Save results
            master_pbar.set_description("💾 Saving Results")
            save_distributed_transitivity_results(evaluation_metrics, analysis_results, SAVE_DIR, accelerator)
            current_progress += pipeline_stages[4][1]
            master_pbar.update(pipeline_stages[4][1])
            
            master_pbar.set_description("🎉 Analysis Complete!")
            master_pbar.close()
            
        except Exception as e:
            master_pbar.close()
            accelerator.print(f"❌ Error in analysis: {e}")
    
    accelerator.print("🎉 Distributed logical consistency analysis completed!")
    accelerator.print(f"Results saved to: {SAVE_DIR}")

if __name__ == '__main__':
    main()
