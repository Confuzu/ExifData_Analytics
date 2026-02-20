#!/usr/bin/env python3
"""
Advanced Analytics for EXIF Data

Provides correlation analysis, prompt effectiveness, recommendations,
comparison tools, and trend detection for AI image generation parameters.
"""

import sys
import sqlite3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
import logging
import configparser

# Check Python version
if sys.version_info < (3, 7):
    print("This script requires Python 3.7 or higher.")
    sys.exit(1)

# Get script directory
script_dir = Path(__file__).resolve().parent

# Load configuration
config = configparser.ConfigParser()
config_path = script_dir / 'config.ini'
try:
    config.read(config_path)
    DB_NAME = config['Database']['db_name']
except Exception as e:
    print(f"Error reading config: {e}")
    print("Using default database name: statistics_image_metadata.db")
    DB_NAME = "statistics_image_metadata.db"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('advanced_analytics')

# Output directory for analytics
OUTPUT_DIR = script_dir / 'advanced_analytics'
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def connect_db() -> sqlite3.Connection:
    """Connect to the database."""
    db_path = script_dir / DB_NAME
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    return sqlite3.connect(str(db_path))


def parse_exif_data(exif_text: str) -> Dict[str, str]:
    """
    Parse EXIF text into dictionary of parameters.

    Extracts common Stable Diffusion parameters from EXIF data.
    """
    params = {}

    # Common SD parameters patterns (case-insensitive)
    patterns = {
        'sampler': r'Sampler:\s*([^\n,]+?)(?:\s*,|\s*$)',
        'cfg_scale': r'CFG scale:\s*([\d.]+)',
        'size': r'Size:\s*(\d+x\d+)',
        'model': r'Model:\s*([^\n,]+?)(?:\s*,|\s*Denoising|\s*,)',
        'vae': r'VAE:\s*([^\n,]+?)(?:\s*,|\s*$)',
        'denoising_strength': r'Denoising strength:\s*([\d.]+)',
        'prompt': r'(?:parameters|template):\s*([^\n]+?)(?:\.|$)',
        'negative_prompt': r'(?:Negative Template|negative prompt):\s*([^\n]+?)(?:\.|$)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, exif_text, re.IGNORECASE | re.MULTILINE)
        if match:
            params[key] = match.group(1).strip()

    return params


def get_all_parameters() -> List[Dict[str, Any]]:
    """
    Retrieve all image parameters from database.

    Returns list of dicts with file_name, file_path, and parsed parameters.
    """
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT file_name, file_path, metadata, metadata_after_prompt FROM file_metadata")
    rows = cursor.fetchall()
    conn.close()

    results = []
    for file_name, file_path, metadata, metadata_after_prompt in rows:
        # Combine both metadata fields for parsing
        combined_metadata = f"{metadata}\n{metadata_after_prompt if metadata_after_prompt else ''}"
        params = parse_exif_data(combined_metadata)
        params['file_name'] = file_name
        params['file_path'] = file_path
        results.append(params)

    logger.info(f"Loaded {len(results)} images from database")
    return results


# ============================================================================
# PHASE 1: CORRELATION ANALYSIS
# ============================================================================

def analyze_correlations(data: List[Dict[str, Any]]) -> None:
    """
    Analyze parameter correlations and co-occurrences.

    Outputs:
    - Parameter combination frequencies
    - Co-occurrence statistics
    - Top combinations report
    """
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70 + "\n")

    # Count individual parameters
    param_counts = defaultdict(Counter)

    for img in data:
        for key in ['sampler', 'cfg_scale', 'size', 'model', 'denoising_strength']:
            if key in img and img[key]:
                param_counts[key][img[key]] += 1

    # Display individual parameter distributions
    print("Individual Parameter Distributions:")
    print("-" * 70)

    for param_name, counts in sorted(param_counts.items()):
        print(f"\n{param_name.replace('_', ' ').title()}:")
        total = sum(counts.values())
        for value, count in counts.most_common(10):
            percentage = (count / total) * 100
            print(f"  {value:30s} {count:5d} ({percentage:5.1f}%)")

    # Analyze parameter combinations
    print("\n" + "="*70)
    print("PARAMETER COMBINATIONS")
    print("="*70 + "\n")

    # Sampler + CFG + Denoising combinations
    combo_counts = Counter()

    for img in data:
        sampler = img.get('sampler', 'Unknown')
        cfg = img.get('cfg_scale', 'Unknown')
        denoise = img.get('denoising_strength', 'Unknown')

        combo = f"{sampler} | CFG:{cfg} | Denoise:{denoise}"
        combo_counts[combo] += 1

    print("Top 10 Parameter Combinations (Sampler + CFG + Denoising):")
    print("-" * 70)
    total_images = len(data)

    for i, (combo, count) in enumerate(combo_counts.most_common(10), 1):
        percentage = (count / total_images) * 100
        print(f"{i:2d}. {combo:50s} {count:5d} ({percentage:5.1f}%)")

    # Model + Sampler combinations
    model_sampler_combos = Counter()

    for img in data:
        model = img.get('model', 'Unknown')
        sampler = img.get('sampler', 'Unknown')
        combo = f"{model} + {sampler}"
        model_sampler_combos[combo] += 1

    print("\n\nTop 10 Model + Sampler Combinations:")
    print("-" * 70)

    for i, (combo, count) in enumerate(model_sampler_combos.most_common(10), 1):
        percentage = (count / total_images) * 100
        print(f"{i:2d}. {combo:60s} {count:5d} ({percentage:5.1f}%)")

    # Save to file
    output_file = OUTPUT_DIR / 'correlation_analysis.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("CORRELATION ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total images analyzed: {len(data)}\n\n")

        f.write("Individual Parameter Distributions:\n")
        f.write("-" * 70 + "\n")

        for param_name, counts in sorted(param_counts.items()):
            f.write(f"\n{param_name.replace('_', ' ').title()}:\n")
            total = sum(counts.values())
            for value, count in counts.most_common():
                percentage = (count / total) * 100
                f.write(f"  {value:30s} {count:5d} ({percentage:5.1f}%)\n")

        f.write("\n" + "="*70 + "\n")
        f.write("PARAMETER COMBINATIONS\n")
        f.write("="*70 + "\n\n")

        f.write("All Sampler + CFG + Denoising Combinations:\n")
        f.write("-" * 70 + "\n")

        for combo, count in combo_counts.most_common():
            percentage = (count / total_images) * 100
            f.write(f"{combo:50s} {count:5d} ({percentage:5.1f}%)\n")

        f.write("\n\nAll Model + Sampler Combinations:\n")
        f.write("-" * 70 + "\n")

        for combo, count in model_sampler_combos.most_common():
            percentage = (count / total_images) * 100
            f.write(f"{combo:60s} {count:5d} ({percentage:5.1f}%)\n")

    print(f"\nSUCCESS: Full report saved to: {output_file}")
    logger.info(f"Correlation analysis complete: {output_file}")


# ============================================================================
# PHASE 1: PROMPT EFFECTIVENESS ANALYSIS
# ============================================================================

def analyze_prompt_effectiveness(data: List[Dict[str, Any]]) -> None:
    """
    Analyze prompt patterns and effectiveness.

    Outputs:
    - Essential tags (appear in >90% of prompts)
    - Variable tags (experimentation)
    - Tag co-occurrence patterns
    - Prompt template extraction
    """
    print("\n" + "="*70)
    print("PROMPT EFFECTIVENESS ANALYSIS")
    print("="*70 + "\n")

    # Extract all prompt words/tags
    tag_counts = Counter()
    total_prompts = 0

    for img in data:
        prompt = img.get('prompt', '')
        if prompt:
            total_prompts += 1
            # Split by common delimiters
            tags = re.split(r'[,\n]+', prompt)
            for tag in tags:
                tag = tag.strip()
                if tag:
                    tag_counts[tag] += 1

    print(f"Total prompts analyzed: {total_prompts}")
    print(f"Unique tags found: {len(tag_counts)}\n")

    # Categorize tags by frequency
    essential_tags = []  # >90%
    common_tags = []     # 50-90%
    variable_tags = []   # <50%

    for tag, count in tag_counts.items():
        percentage = (count / total_prompts) * 100

        if percentage > 90:
            essential_tags.append((tag, count, percentage))
        elif percentage > 50:
            common_tags.append((tag, count, percentage))
        else:
            variable_tags.append((tag, count, percentage))

    # Sort by frequency
    essential_tags.sort(key=lambda x: x[1], reverse=True)
    common_tags.sort(key=lambda x: x[1], reverse=True)
    variable_tags.sort(key=lambda x: x[1], reverse=True)

    # Display results
    print("ESSENTIAL TAGS (appear in >90% of images):")
    print("-" * 70)
    if essential_tags:
        for tag, count, pct in essential_tags:
            print(f"  {tag:50s} {count:5d} ({pct:5.1f}%)")
    else:
        print("  None found")

    print("\n\nCOMMON TAGS (appear in 50-90% of images):")
    print("-" * 70)
    if common_tags:
        for tag, count, pct in common_tags[:20]:  # Top 20
            print(f"  {tag:50s} {count:5d} ({pct:5.1f}%)")
        if len(common_tags) > 20:
            print(f"  ... and {len(common_tags) - 20} more")
    else:
        print("  None found")

    print("\n\nVARIABLE TAGS (appear in <50% of images - experimentation):")
    print("-" * 70)
    print(f"Total variable tags: {len(variable_tags)}")
    print("\nTop 20 most common variable tags:")
    for tag, count, pct in variable_tags[:20]:
        print(f"  {tag:50s} {count:5d} ({pct:5.1f}%)")

    # Extract prompt template
    print("\n\n" + "="*70)
    print("PROMPT TEMPLATE EXTRACTION")
    print("="*70 + "\n")

    print("Based on essential tags, your core prompt template appears to be:")
    print("-" * 70)

    if essential_tags:
        core_template = ", ".join([tag for tag, _, _ in essential_tags])
        print(f"{core_template}")
        print(f"\n+ [VARIABLE LOCATION/SCENE]")
    else:
        print("Unable to extract template (no essential tags)")

    # Save to file
    output_file = OUTPUT_DIR / 'prompt_effectiveness.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("PROMPT EFFECTIVENESS ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total prompts analyzed: {total_prompts}\n")
        f.write(f"Unique tags found: {len(tag_counts)}\n\n")

        f.write("ESSENTIAL TAGS (>90% frequency):\n")
        f.write("-" * 70 + "\n")
        for tag, count, pct in essential_tags:
            f.write(f"{tag:50s} {count:5d} ({pct:5.1f}%)\n")

        f.write("\n\nCOMMON TAGS (50-90% frequency):\n")
        f.write("-" * 70 + "\n")
        for tag, count, pct in common_tags:
            f.write(f"{tag:50s} {count:5d} ({pct:5.1f}%)\n")

        f.write("\n\nVARIABLE TAGS (<50% frequency):\n")
        f.write("-" * 70 + "\n")
        for tag, count, pct in variable_tags:
            f.write(f"{tag:50s} {count:5d} ({pct:5.1f}%)\n")

        f.write("\n\n" + "="*70 + "\n")
        f.write("PROMPT TEMPLATE\n")
        f.write("="*70 + "\n\n")

        if essential_tags:
            core_template = ", ".join([tag for tag, _, _ in essential_tags])
            f.write(f"{core_template}\n")
            f.write(f"\n+ [VARIABLE LOCATION/SCENE]\n")

    print(f"\nSUCCESS: Full report saved to: {output_file}")
    logger.info(f"Prompt effectiveness analysis complete: {output_file}")


# ============================================================================
# PHASE 2: PARAMETER RECOMMENDATIONS
# ============================================================================

def recommend_parameters(data: List[Dict[str, Any]], current_settings: Dict[str, str]) -> None:
    """
    Recommend parameter combinations based on historical data.

    Strategies:
    1. Most frequent combinations (what you use most = likely works)
    2. Similar to current settings (small variations)
    3. Unexplored combinations (suggest experiments)
    """
    print("\n" + "="*70)
    print("PARAMETER RECOMMENDATIONS")
    print("="*70 + "\n")

    print("Current settings:")
    for key, value in current_settings.items():
        print(f"  {key}: {value}")
    print()

    # Strategy 1: Most frequent combinations
    print("Strategy 1: Most Frequently Used Combinations")
    print("-" * 70)

    combo_counts = Counter()
    for img in data:
        sampler = img.get('sampler', '')
        cfg = img.get('cfg_scale', '')
        denoise = img.get('denoising_strength', '')
        model = img.get('model', '')

        if sampler and cfg and denoise:
            combo = (sampler, cfg, denoise, model)
            combo_counts[combo] += 1

    print("\nTop 5 most used combinations:")
    for i, (combo, count) in enumerate(combo_counts.most_common(5), 1):
        sampler, cfg, denoise, model = combo
        percentage = (count / len(data)) * 100
        print(f"\n{i}. Used {count} times ({percentage:.1f}%)")
        print(f"   Sampler: {sampler}")
        print(f"   CFG Scale: {cfg}")
        print(f"   Denoising: {denoise}")
        print(f"   Model: {model}")

    # Strategy 2: Similar to current settings
    print("\n\nStrategy 2: Similar to Your Current Settings")
    print("-" * 70)

    similar_combos = []
    current_sampler = current_settings.get('sampler', '')
    current_model = current_settings.get('model', '')

    for (sampler, cfg, denoise, model), count in combo_counts.items():
        similarity_score = 0

        # Same sampler = +2 points
        if sampler == current_sampler:
            similarity_score += 2

        # Same model = +2 points
        if model == current_model:
            similarity_score += 2

        # Different but exists = +1 point per param
        if sampler and sampler != current_sampler:
            similarity_score += 1

        if similarity_score > 0:
            similar_combos.append((similarity_score, count, sampler, cfg, denoise, model))

    similar_combos.sort(reverse=True, key=lambda x: (x[0], x[1]))

    print("\nTop 5 similar combinations:")
    for i, (score, count, sampler, cfg, denoise, model) in enumerate(similar_combos[:5], 1):
        percentage = (count / len(data)) * 100
        print(f"\n{i}. Similarity score: {score}/4, Used {count} times ({percentage:.1f}%)")
        print(f"   Sampler: {sampler}")
        print(f"   CFG Scale: {cfg}")
        print(f"   Denoising: {denoise}")
        print(f"   Model: {model}")

    # Strategy 3: Parameter variations to try
    print("\n\nStrategy 3: Suggested Experiments")
    print("-" * 70)

    # Get all unique values for each parameter
    all_samplers = set(img.get('sampler', '') for img in data if img.get('sampler'))
    all_cfg = set(img.get('cfg_scale', '') for img in data if img.get('cfg_scale'))
    all_denoise = set(img.get('denoising_strength', '') for img in data if img.get('denoising_strength'))

    print("\nParameters you could experiment with:")
    print(f"\nSamplers available: {', '.join(sorted(all_samplers))}")
    print(f"CFG scales used: {', '.join(sorted(all_cfg))}")
    print(f"Denoising strengths used: {', '.join(sorted(all_denoise))}")

    # Save to file
    output_file = OUTPUT_DIR / 'recommendations.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("PARAMETER RECOMMENDATIONS REPORT\n")
        f.write("="*70 + "\n\n")

        f.write("Current settings:\n")
        for key, value in current_settings.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("Strategy 1: Most Frequently Used Combinations\n")
        f.write("-" * 70 + "\n")
        for i, (combo, count) in enumerate(combo_counts.most_common(10), 1):
            sampler, cfg, denoise, model = combo
            percentage = (count / len(data)) * 100
            f.write(f"\n{i}. Used {count} times ({percentage:.1f}%)\n")
            f.write(f"   Sampler: {sampler}\n")
            f.write(f"   CFG Scale: {cfg}\n")
            f.write(f"   Denoising: {denoise}\n")
            f.write(f"   Model: {model}\n")

        f.write("\n\nStrategy 2: Similar to Current Settings\n")
        f.write("-" * 70 + "\n")
        for i, (score, count, sampler, cfg, denoise, model) in enumerate(similar_combos[:10], 1):
            percentage = (count / len(data)) * 100
            f.write(f"\n{i}. Similarity: {score}/4, Used {count} times ({percentage:.1f}%)\n")
            f.write(f"   Sampler: {sampler}\n")
            f.write(f"   CFG Scale: {cfg}\n")
            f.write(f"   Denoising: {denoise}\n")
            f.write(f"   Model: {model}\n")

    print(f"\nSUCCESS: Full recommendations saved to: {output_file}")
    logger.info(f"Parameter recommendations complete: {output_file}")


# ============================================================================
# PHASE 2: COMPARISON TOOL
# ============================================================================

def compare_datasets(data: List[Dict[str, Any]], filter1_str: str, filter2_str: str) -> None:
    """
    Compare two filtered subsets of the dataset.

    Args:
        filter1_str: Filter like "model=ANIME_abyssorangemix2_Hard"
        filter2_str: Filter like "model=ANIME_abyssorangemix2NSFW_Nsfw"
    """
    print("\n" + "="*70)
    print("DATASET COMPARISON")
    print("="*70 + "\n")

    # Parse filters
    def parse_filter(filter_str):
        parts = filter_str.split('=', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid filter format: {filter_str}. Expected key=value")
        return parts[0].strip(), parts[1].strip()

    key1, value1 = parse_filter(filter1_str)
    key2, value2 = parse_filter(filter2_str)

    print(f"Comparing: {key1}={value1} vs {key2}={value2}\n")

    # Filter datasets
    dataset1 = [img for img in data if img.get(key1) == value1]
    dataset2 = [img for img in data if img.get(key2) == value2]

    print(f"Dataset 1 ({key1}={value1}): {len(dataset1)} images")
    print(f"Dataset 2 ({key2}={value2}): {len(dataset2)} images")

    if not dataset1:
        print(f"\nERROR: No images found matching {key1}={value1}")
        return

    if not dataset2:
        print(f"\nERROR: No images found matching {key2}={value2}")
        return

    print("\n" + "-" * 70)

    # Compare parameter distributions
    params_to_compare = ['sampler', 'cfg_scale', 'size', 'denoising_strength']

    for param in params_to_compare:
        print(f"\n{param.replace('_', ' ').title()} Distribution:")
        print("-" * 70)

        # Count for dataset 1
        counts1 = Counter()
        for img in dataset1:
            if param in img and img[param]:
                counts1[img[param]] += 1

        # Count for dataset 2
        counts2 = Counter()
        for img in dataset2:
            if param in img and img[param]:
                counts2[img[param]] += 1

        # Get all unique values
        all_values = set(counts1.keys()) | set(counts2.keys())

        # Display comparison
        print(f"{'Value':<30} {'Dataset 1':<20} {'Dataset 2':<20}")
        print(f"{'-'*30} {'-'*20} {'-'*20}")

        for value in sorted(all_values):
            count1 = counts1.get(value, 0)
            count2 = counts2.get(value, 0)
            pct1 = (count1 / len(dataset1)) * 100 if dataset1 else 0
            pct2 = (count2 / len(dataset2)) * 100 if dataset2 else 0

            print(f"{value:<30} {count1:5d} ({pct1:5.1f}%)    {count2:5d} ({pct2:5.1f}%)")

    # Save to file
    output_file = OUTPUT_DIR / f'comparison_{value1}_vs_{value2}.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("DATASET COMPARISON REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Comparing: {key1}={value1} vs {key2}={value2}\n\n")
        f.write(f"Dataset 1: {len(dataset1)} images\n")
        f.write(f"Dataset 2: {len(dataset2)} images\n\n")

        for param in params_to_compare:
            f.write(f"\n{param.replace('_', ' ').title()} Distribution:\n")
            f.write("-" * 70 + "\n")

            counts1 = Counter()
            for img in dataset1:
                if param in img and img[param]:
                    counts1[img[param]] += 1

            counts2 = Counter()
            for img in dataset2:
                if param in img and img[param]:
                    counts2[img[param]] += 1

            all_values = set(counts1.keys()) | set(counts2.keys())

            f.write(f"{'Value':<30} {'Dataset 1':<20} {'Dataset 2':<20}\n")
            f.write(f"{'-'*30} {'-'*20} {'-'*20}\n")

            for value in sorted(all_values):
                count1 = counts1.get(value, 0)
                count2 = counts2.get(value, 0)
                pct1 = (count1 / len(dataset1)) * 100 if dataset1 else 0
                pct2 = (count2 / len(dataset2)) * 100 if dataset2 else 0

                f.write(f"{value:<30} {count1:5d} ({pct1:5.1f}%)    {count2:5d} ({pct2:5.1f}%)\n")

    print(f"\nSUCCESS: Full comparison saved to: {output_file}")
    logger.info(f"Dataset comparison complete: {output_file}")


# ============================================================================
# PHASE 3: TREND DETECTION
# ============================================================================

def analyze_trends(data: List[Dict[str, Any]]) -> None:
    """
    Analyze temporal trends in parameter usage.

    Requires file timestamps or creation dates.
    """
    print("\n" + "="*70)
    print("TREND DETECTION")
    print("="*70 + "\n")

    # Try to get file modification times
    images_with_time = []

    for img in data:
        file_path = img.get('file_path', '')
        if file_path and Path(file_path).exists():
            try:
                mtime = Path(file_path).stat().st_mtime
                img_copy = img.copy()
                img_copy['timestamp'] = mtime
                images_with_time.append(img_copy)
            except Exception as e:
                logger.debug(f"Could not get timestamp for {file_path}: {e}")

    if not images_with_time:
        print("WARNING: No timestamps available for trend analysis.")
        print("   Files may have been moved/copied, losing original timestamps.")
        print("\n   Alternative: Check EXIF creation dates in the metadata.")
        return

    # Sort by timestamp
    images_with_time.sort(key=lambda x: x['timestamp'])

    print(f"Analyzing trends for {len(images_with_time)} images with timestamps\n")

    import datetime

    earliest = datetime.datetime.fromtimestamp(images_with_time[0]['timestamp'])
    latest = datetime.datetime.fromtimestamp(images_with_time[-1]['timestamp'])

    print(f"Date range: {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")
    print()

    # Group by time periods (if enough data)
    # For now, just show parameter evolution

    # Divide into quarters
    quarter_size = len(images_with_time) // 4

    if quarter_size < 10:
        print("WARNING: Not enough images for meaningful trend analysis (need at least 40)")
        return

    quarters = [
        images_with_time[:quarter_size],
        images_with_time[quarter_size:quarter_size*2],
        images_with_time[quarter_size*2:quarter_size*3],
        images_with_time[quarter_size*3:],
    ]

    quarter_names = ["Q1 (Earliest)", "Q2", "Q3", "Q4 (Latest)"]

    # Analyze parameter changes across quarters
    print("Parameter Evolution Across Time:")
    print("-" * 70)

    for param in ['sampler', 'cfg_scale', 'model', 'denoising_strength']:
        print(f"\n{param.replace('_', ' ').title()}:")

        for i, (quarter, name) in enumerate(zip(quarters, quarter_names)):
            counts = Counter()
            for img in quarter:
                if param in img and img[param]:
                    counts[img[param]] += 1

            print(f"\n  {name}:")
            for value, count in counts.most_common(3):
                pct = (count / len(quarter)) * 100
                print(f"    {value:40s} {count:4d} ({pct:5.1f}%)")

    # Save to file
    output_file = OUTPUT_DIR / 'trend_analysis.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("TREND ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Images analyzed: {len(images_with_time)}\n")
        f.write(f"Date range: {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}\n\n")

        f.write("Parameter Evolution Across Time Quarters:\n")
        f.write("-" * 70 + "\n")

        for param in ['sampler', 'cfg_scale', 'model', 'denoising_strength']:
            f.write(f"\n{param.replace('_', ' ').title()}:\n")

            for i, (quarter, name) in enumerate(zip(quarters, quarter_names)):
                counts = Counter()
                for img in quarter:
                    if param in img and img[param]:
                        counts[img[param]] += 1

                f.write(f"\n  {name}:\n")
                for value, count in counts.most_common():
                    pct = (count / len(quarter)) * 100
                    f.write(f"    {value:40s} {count:4d} ({pct:5.1f}%)\n")

    print(f"\nSUCCESS: Full trend analysis saved to: {output_file}")
    logger.info(f"Trend analysis complete: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Advanced analytics for AI image generation parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
    )

    parser.add_argument('--correlations', action='store_true',
                        help='Analyze parameter correlations')
    parser.add_argument('--prompt-analysis', action='store_true',
                        help='Analyze prompt effectiveness')
    parser.add_argument('--recommend', type=str, metavar='SETTINGS',
                        help='Get parameter recommendations (e.g., "sampler=Heun,model=X")')
    parser.add_argument('--compare', nargs=2, metavar=('FILTER1', 'FILTER2'),
                        help='Compare two datasets (e.g., "model=X" "model=Y")')
    parser.add_argument('--trends', action='store_true',
                        help='Analyze temporal trends')
    parser.add_argument('--all', action='store_true',
                        help='Run all analyses')

    args = parser.parse_args()

    # If no arguments, show help
    if not any([args.correlations, args.prompt_analysis, args.recommend,
                args.compare, args.trends, args.all]):
        parser.print_help()
        sys.exit(0)

    # Load data
    try:
        print("Loading data from database...")
        data = get_all_parameters()

        if not data:
            print("ERROR: No data found in database. Run DB_filler.py first.")
            sys.exit(1)

        print(f"SUCCESS: Loaded {len(data)} images\n")

    except Exception as e:
        print(f"ERROR: Error loading data: {e}")
        logger.exception("Failed to load data")
        sys.exit(1)

    # Run requested analyses
    try:
        if args.all or args.correlations:
            analyze_correlations(data)

        if args.all or args.prompt_analysis:
            analyze_prompt_effectiveness(data)

        if args.recommend:
            # Parse current settings
            settings = {}
            for pair in args.recommend.split(','):
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    settings[key.strip()] = value.strip()

            recommend_parameters(data, settings)

        if args.compare:
            compare_datasets(data, args.compare[0], args.compare[1])

        if args.all or args.trends:
            analyze_trends(data)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nAll reports saved to: {OUTPUT_DIR}/")
        print("\nFiles generated:")
        for file in sorted(OUTPUT_DIR.glob('*.txt')):
            print(f"  - {file.name}")

    except Exception as e:
        print(f"\nERROR: Error during analysis: {e}")
        logger.exception("Analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
