"""
Visualization Module
Generate publication-quality figures from model results

Creates:
- Market failure vs fund stability curves
- Model comparison capability matrices
- Regional scalability maps
- Implementation timeline Gantt charts

Author: Research Team
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set publication style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ResultsVisualizer:
    """Visualize model results and framework outputs"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize visualizer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.output_dir = Path(self.config['paths']['figures'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("ResultsVisualizer initialized")

    def plot_insurer_retreat_vs_fund_stability(
        self,
        mrep_retreat_prob: np.ndarray,
        sfm_solvency: np.ndarray,
        years: np.ndarray,
        output_name: str = "market_failure_vs_fund_stability"
    ) -> str:
        """
        Dual-axis plot comparing MREP uninsurability forecast with SFM stability

        Args:
            mrep_retreat_prob: Insurer retreat probability (0-1) over time
            sfm_solvency: Fund solvency confidence (0-1) over time
            years: Year array
            output_name: Output filename

        Returns:
            Path to saved figure
        """
        fig, ax1 = plt.subplots(figsize=(12, 7))

        # Left axis: Insurer retreat
        color1 = '#d62728'
        ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Probability of Insurer Retreat (%)', color=color1, fontsize=12, fontweight='bold')
        line1 = ax1.plot(years, mrep_retreat_prob * 100, color=color1, linewidth=2.5, label='MREP Uninsurability Forecast', marker='o', markersize=4)
        ax1.axhline(y=75, color=color1, linestyle='--', alpha=0.5, linewidth=1.5, label='Critical Threshold (75%)')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim([0, 100])
        ax1.grid(True, alpha=0.3)

        # Right axis: Fund solvency
        ax2 = ax1.twinx()
        color2 = '#2ca02c'
        ax2.set_ylabel('Fund Solvency Confidence (%)', color=color2, fontsize=12, fontweight='bold')
        line2 = ax2.plot(years, sfm_solvency * 100, color=color2, linewidth=2.5, label='SFM Fund Stability', marker='s', markersize=4)
        ax2.axhline(y=99.5, color=color2, linestyle='--', alpha=0.5, linewidth=1.5, label='Regulatory Minimum (99.5%)')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim([98, 100.5])

        # Title and legend
        plt.title('Market Failure vs. Mutualized Fund Stability', fontsize=14, fontweight='bold', pad=20)
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=10)

        # Formatting
        fig.tight_layout()

        # Save
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
        plt.close()

        return str(output_path)

    def plot_model_capabilities_heatmap(
        self,
        models: List[str],
        capabilities: List[str],
        capability_matrix: np.ndarray,
        output_name: str = "model_capabilities_comparison"
    ) -> str:
        """
        Heatmap comparing models across capability dimensions

        Args:
            models: List of model names
            capabilities: List of capability names
            capability_matrix: Matrix of capability scores (0-1)
            output_name: Output filename

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create heatmap
        sns.heatmap(
            capability_matrix,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            xticklabels=capabilities,
            yticklabels=models,
            cbar_kws={'label': 'Capability Score'},
            linewidths=0.5,
            ax=ax
        )

        plt.title('Integrated Framework vs. Existing Models: Capability Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Capability Dimensions', fontsize=12, fontweight='bold')
        plt.ylabel('Models', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')

        fig.tight_layout()

        # Save
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
        plt.close()

        return str(output_path)

    def plot_regional_scalability(
        self,
        regions: Dict[str, Dict],
        output_name: str = "global_scalability_map"
    ) -> str:
        """
        Bar chart showing global scalability by region

        Args:
            regions: Dict with region data (farmers, qualification %)
            output_name: Output filename

        Returns:
            Path to saved figure
        """
        region_names = list(regions.keys())
        fully_qualified = [regions[r].get('fully_qualified', 0) for r in region_names]
        moderate_adaptation = [regions[r].get('moderate_adaptation', 0) for r in region_names]

        fig, ax = plt.subplots(figsize=(12, 7))

        x = np.arange(len(region_names))
        width = 0.6

        bars1 = ax.bar(x, fully_qualified, width, label='Fully Qualified', color='#2ca02c')
        bars2 = ax.bar(x, moderate_adaptation, width, bottom=fully_qualified, 
                       label='Requires Adaptation', color='#ff7f0e')

        # Add value labels
        for i, (val1, val2) in enumerate(zip(fully_qualified, moderate_adaptation)):
            total = val1 + val2
            ax.text(i, total + 2, f'{total}M', ha='center', fontweight='bold')

        ax.set_ylabel('Farmer Population (Millions)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Region', fontsize=12, fontweight='bold')
        ax.set_title('Global Scalability: Framework Replication Potential by Region', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(region_names, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        fig.tight_layout()

        # Save
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
        plt.close()

        return str(output_path)

    def plot_implementation_timeline(
        self,
        phases: List[Dict],
        output_name: str = "implementation_timeline"
    ) -> str:
        """
        Gantt-style chart for implementation phases

        Args:
            phases: List of phase dicts with start, end, label
            output_name: Output filename

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for idx, phase in enumerate(phases):
            start = phase['start']
            end = phase['end']
            label = phase['label']
            duration = end - start

            ax.barh(idx, duration, left=start, height=0.6, 
                   color=colors[idx % len(colors)], edgecolor='black', linewidth=1.5)

            # Add label
            ax.text(start + duration/2, idx, label, 
                   ha='center', va='center', fontweight='bold', fontsize=10, color='white')

            # Add details below
            details = phase.get('details', '')
            ax.text(start + duration/2, idx - 0.35, details,
                   ha='center', va='top', fontsize=8, style='italic')

        ax.set_ylim(-0.5, len(phases) - 0.5)
        ax.set_xlim(2025, 2041)
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_title('Global Implementation Timeline: 2026-2040', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_yticks([])
        ax.grid(axis='x', alpha=0.3)

        # Add milestone markers
        milestones = [2026, 2030, 2035, 2040]
        for milestone in milestones:
            ax.axvline(milestone, color='red', linestyle='--', alpha=0.3, linewidth=1)
            ax.text(milestone, len(phases) - 0.1, str(milestone), 
                   ha='center', fontsize=9, color='red', fontweight='bold')

        fig.tight_layout()

        # Save
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
        plt.close()

        return str(output_path)

    def plot_solvency_trajectory(
        self,
        trajectory_df: pd.DataFrame,
        output_name: str = "fund_solvency_trajectory"
    ) -> str:
        """
        Plot fund solvency trajectory over forecast horizon

        Args:
            trajectory_df: DataFrame with solvency trajectory
            output_name: Output filename

        Returns:
            Path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Solvency buffer
        ax1.plot(trajectory_df['year'], trajectory_df['solvency_buffer'], 
                linewidth=2.5, marker='o', color='#2ca02c', label='Solvency Buffer')
        ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Critical Threshold')
        ax1.fill_between(trajectory_df['year'], 0, trajectory_df['solvency_buffer'], 
                        where=(trajectory_df['solvency_buffer'] >= 1), alpha=0.3, color='green', label='Solvent Region')
        ax1.fill_between(trajectory_df['year'], 0, trajectory_df['solvency_buffer'],
                        where=(trajectory_df['solvency_buffer'] < 1), alpha=0.3, color='red', label='Insolvent Risk')
        ax1.set_ylabel('Solvency Buffer (ratio)', fontsize=11, fontweight='bold')
        ax1.set_title('Fund Solvency Trajectory: Long-Term Viability Forecast', 
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Fund capital
        ax2.plot(trajectory_df['year'], trajectory_df['fund_capital'] / 1e9, 
                linewidth=2.5, marker='s', color='#1f77b4', label='Fund Capital')
        ax2.fill_between(trajectory_df['year'], 0, trajectory_df['fund_capital'] / 1e9, 
                        alpha=0.3, color='blue')
        ax2.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Fund Capital ($ Billions)', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()

        # Save
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
        plt.close()

        return str(output_path)

    def plot_damage_detection_results(
        self,
        satellite_image: np.ndarray,
        damage_mask: np.ndarray,
        flood_pct: float,
        drought_pct: float,
        output_name: str = "damage_detection_example"
    ) -> str:
        """
        Show satellite imagery and detected damage

        Args:
            satellite_image: RGB satellite image
            damage_mask: Damage segmentation mask
            flood_pct: Flood damage percentage
            drought_pct: Drought damage percentage
            output_name: Output filename

        Returns:
            Path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Original image
        ax1.imshow(satellite_image)
        ax1.set_title('Sentinel-2 RGB Composite', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Damage mask
        damage_display = np.where(damage_mask > 0, damage_mask, np.nan)
        im = ax2.imshow(satellite_image, alpha=0.7)
        im_mask = ax2.imshow(damage_display, cmap='Reds', alpha=0.5)
        ax2.set_title(f'Damage Detection\nFlood: {flood_pct:.1f}% | Drought: {drought_pct:.1f}%', 
                     fontsize=12, fontweight='bold')
        ax2.axis('off')

        fig.tight_layout()

        # Save
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
        plt.close()

        return str(output_path)


if __name__ == "__main__":
    viz = ResultsVisualizer()

    # Example 1: Market failure vs stability plot
    years = np.arange(2025, 2046)
    retreat_prob = 0.12 + (0.7 * (1 / (1 + np.exp(-0.3 * (years - 2037)))))
    solvency = 0.996 + 0.002 * np.sin((years - 2025) * 0.5)

    path1 = viz.plot_insurer_retreat_vs_fund_stability(
        retreat_prob, solvency, years
    )
    print(f"Plot 1 saved: {path1}")

    # Example 2: Implementation timeline
    phases = [
        {'start': 2026, 'end': 2028, 'label': 'Phase 1: Proof-of-Concept', 'details': '7-10M farmers'},
        {'start': 2029, 'end': 2032, 'label': 'Phase 2: Regional Expansion', 'details': '35-50M farmers'},
        {'start': 2033, 'end': 2036, 'label': 'Phase 3: Global Mainstreaming', 'details': '80-120M farmers'},
        {'start': 2037, 'end': 2040, 'label': 'Phase 4: Market Maturity', 'details': '150-200M farmers'}
    ]

    path2 = viz.plot_implementation_timeline(phases)
    print(f"Timeline plot saved: {path2}")
