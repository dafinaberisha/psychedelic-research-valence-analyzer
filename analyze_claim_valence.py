import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from db_connector import Neo4jConnector
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ValenceAnalyzer")

try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False
    logger.warning("Pingouin library not available. Install with: pip install pingouin")


class ValenceAnalyzer:
    def __init__(self, neo4j_uri=None, neo4j_user=None, neo4j_password=None):
        neo4j_uri = neo4j_uri or config.NEOJ4_URI
        neo4j_user = neo4j_user or config.NEOJ4_USER
        neo4j_password = neo4j_password or config.NEOJ4_PASSWORD
        
        self.db_connector = Neo4jConnector(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
        
        self.output_dir = "valence_analysis"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def analyze_claim_valences(self):
        logger.info("Starting valence comparison analysis...")
        
        claims = self._get_claims_with_both_valences()
        logger.info(f"Found {len(claims)} claims with both Zero-Shot Classification and OpenAI valence values")
        
        if not claims:
            logger.warning("No claims found with both valence values")
            return
        
        df = pd.DataFrame(claims)
        
        df['valence_diff'] = df['zeroshot_valence'] - df['openai_valence']
        df['abs_diff'] = np.abs(df['valence_diff'])
        
        self._print_basic_stats(df)
        self._generate_visualizations(df)
        self._identify_outliers(df)
        
        logger.info("Analysis complete.")
        
    def _get_claims_with_both_valences(self):
        with self.db_connector.driver.session() as session:
            query = """
            MATCH (c:Claim)
            WHERE c.zero_shot_valence IS NOT NULL AND c.openai_valence IS NOT NULL
            RETURN 
                elementId(c) as id, 
                c.text as text, 
                c.zero_shot_valence as zeroshot_valence, 
                c.openai_valence as openai_valence
            """
            
            result = session.run(query)
            claims = [dict(record) for record in result]
            
            for claim in claims:
                claim['zeroshot_valence'] = float(claim['zeroshot_valence'])
                claim['openai_valence'] = float(claim['openai_valence'])
                
            return claims
    
    def _print_basic_stats(self, df):
        zeroshot_mean = df['zeroshot_valence'].mean()
        openai_mean = df['openai_valence'].mean()
        zeroshot_std = df['zeroshot_valence'].std()
        openai_std = df['openai_valence'].std()
        
        mae = mean_absolute_error(df['openai_valence'], df['zeroshot_valence'])
        rmse = np.sqrt(mean_squared_error(df['openai_valence'], df['zeroshot_valence']))
        
        pearson_corr, p_value = stats.pearsonr(df['zeroshot_valence'], df['openai_valence'])
        
        logger.info("\n=== VALENCE COMPARISON STATISTICS ===")
        logger.info(f"Number of claims analyzed: {len(df)}")
        logger.info(f"Zero-Shot Classification valence: mean={zeroshot_mean:.3f}, std={zeroshot_std:.3f}, min={df['zeroshot_valence'].min():.3f}, max={df['zeroshot_valence'].max():.3f}")
        logger.info(f"OpenAI valence: mean={openai_mean:.3f}, std={openai_std:.3f}, min={df['openai_valence'].min():.3f}, max={df['openai_valence'].max():.3f}")
        logger.info(f"Mean absolute difference: {mae:.3f}")
        logger.info(f"Root mean squared error: {rmse:.3f}")
        logger.info(f"Pearson correlation: {pearson_corr:.3f} (p-value: {p_value:.4f})")
        
        logger.info("\nAverage Absolute Difference by OpenAI Valence Range:")
        ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for low, high in ranges:
            in_range = df[(df['openai_valence'] >= low) & (df['openai_valence'] < high)]
            if len(in_range) > 0:
                mean_diff = in_range['abs_diff'].mean()
                logger.info(f"  {low}-{high}: {mean_diff:.3f} (n={len(in_range)})")
        
        stats_file = os.path.join(self.output_dir, "valence_stats.txt")
        with open(stats_file, 'w') as f:
            f.write("=== VALENCE COMPARISON STATISTICS ===\n")
            f.write(f"Number of claims analyzed: {len(df)}\n")
            f.write(f"Zero-Shot Classification valence: mean={zeroshot_mean:.3f}, std={zeroshot_std:.3f}, min={df['zeroshot_valence'].min():.3f}, max={df['zeroshot_valence'].max():.3f}\n")
            f.write(f"OpenAI valence: mean={openai_mean:.3f}, std={openai_std:.3f}, min={df['openai_valence'].min():.3f}, max={df['openai_valence'].max():.3f}\n")
            f.write(f"Mean absolute difference: {mae:.3f}\n")
            f.write(f"Root mean squared error: {rmse:.3f}\n")
            f.write(f"Pearson correlation: {pearson_corr:.3f} (p-value: {p_value:.4f})\n")
            
            f.write("\nAverage Absolute Difference by OpenAI Valence Range:\n")
            for low, high in ranges:
                in_range = df[(df['openai_valence'] >= low) & (df['openai_valence'] < high)]
                if len(in_range) > 0:
                    mean_diff = in_range['abs_diff'].mean()
                    f.write(f"  {low}-{high}: {mean_diff:.3f} (n={len(in_range)})\n")
        
        # Calculate ICC (Intraclass Correlation Coefficient)
        icc_results = self._calculate_icc(df)
        if icc_results:
            logger.info("\nIntraclass Correlation Coefficient (ICC):")
            logger.info(f"ICC: {icc_results['ICC']:.3f}")
            
            # Handle F-value - check if it exists and is not NaN
            if 'F' in icc_results and not pd.isna(icc_results['F']):
                logger.info(f"F-value: {icc_results['F']:.3f}")
            else:
                logger.info("F-value: Not available")
            
            # Handle p-value - check if it exists and is not NaN
            if 'p' in icc_results and not pd.isna(icc_results['p']):
                logger.info(f"p-value: {icc_results['p']:.4f}")
            else:
                logger.info("p-value: Not available")
            
            # Handle confidence interval - check if it exists and values are not NaN
            if 'CI95%' in icc_results and isinstance(icc_results['CI95%'], (list, tuple)):
                ci_lower = icc_results['CI95%'][0]
                ci_upper = icc_results['CI95%'][1]
                
                if not pd.isna(ci_lower) and not pd.isna(ci_upper):
                    logger.info(f"95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
                else:
                    logger.info("95% Confidence Interval: Not available")
            else:
                logger.info("95% Confidence Interval: Not available")
            
            # Add ICC to the statistics file
            with open(stats_file, 'a') as f:
                f.write("\nIntraclass Correlation Coefficient (ICC):\n")
                f.write(f"ICC: {icc_results['ICC']:.3f}\n")
                
                if 'F' in icc_results and not pd.isna(icc_results['F']):
                    f.write(f"F-value: {icc_results['F']:.3f}\n")
                else:
                    f.write("F-value: Not available\n")
                
                if 'p' in icc_results and not pd.isna(icc_results['p']):
                    f.write(f"p-value: {icc_results['p']:.4f}\n")
                else:
                    f.write("p-value: Not available\n")
                
                if 'CI95%' in icc_results and isinstance(icc_results['CI95%'], (list, tuple)):
                    ci_lower = icc_results['CI95%'][0]
                    ci_upper = icc_results['CI95%'][1]
                    
                    if not pd.isna(ci_lower) and not pd.isna(ci_upper):
                        f.write(f"95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]\n")
                    else:
                        f.write("95% Confidence Interval: Not available\n")
                else:
                    f.write("95% Confidence Interval: Not available\n")
        
        logger.info(f"Statistics saved to {stats_file}")
    
    def _generate_visualizations(self, df):
        """Generate visualizations for the valence comparisons."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Set larger figure size
            plt.figure(figsize=(12, 8))
            
            # 1. Histogram of both valence distributions
            plt.hist(df['zeroshot_valence'], bins=20, alpha=0.5, label='Zero-Shot Classification Valence')
            plt.hist(df['openai_valence'], bins=20, alpha=0.5, label='OpenAI Valence')
            plt.xlabel('Valence Value')
            plt.ylabel('Frequency')
            plt.title('Distribution of Valence Values')
            plt.legend()
            plt.grid(True, alpha=0.3)
            hist_file = os.path.join(self.output_dir, "valence_histogram.png")
            plt.savefig(hist_file, dpi=300)
            plt.close()
            logger.info(f"Saved histogram to {hist_file}")
            
            # 2. Scatter plot with correlation
            plt.figure(figsize=(10, 10))
            plt.scatter(df['openai_valence'], df['zeroshot_valence'], alpha=0.5)
            
            # Add correlation line
            m, b = np.polyfit(df['openai_valence'], df['zeroshot_valence'], 1)
            plt.plot(df['openai_valence'], m*df['openai_valence'] + b, color='red')
            
            # Add diagonal line (perfect agreement)
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            
            plt.xlabel('OpenAI Valence')
            plt.ylabel('Zero-Shot Classification Valence')
            plt.title(f'Correlation between OpenAI and Zero-Shot Classification Valence\nPearson r = {stats.pearsonr(df["openai_valence"], df["zeroshot_valence"])[0]:.3f}')
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            scatter_file = os.path.join(self.output_dir, "valence_correlation.png")
            plt.savefig(scatter_file, dpi=300)
            plt.close()
            logger.info(f"Saved correlation plot to {scatter_file}")
            
            # 3. Histogram of differences
            plt.figure(figsize=(10, 6))
            plt.hist(df['valence_diff'], bins=20, alpha=0.7)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.xlabel('Zero-Shot Classification - OpenAI Valence')
            plt.ylabel('Frequency')
            plt.title('Distribution of Valence Differences')
            plt.grid(True, alpha=0.3)
            diff_file = os.path.join(self.output_dir, "valence_differences.png")
            plt.savefig(diff_file, dpi=300)
            plt.close()
            logger.info(f"Saved differences histogram to {diff_file}")
            
            # 4. Box plot of valence by method
            plt.figure(figsize=(8, 6))
            boxplot_data = [df['zeroshot_valence'], df['openai_valence']]
            plt.boxplot(boxplot_data, labels=['Zero-Shot Classification', 'OpenAI'])
            plt.ylabel('Valence Value')
            plt.title('Comparison of Valence Distributions')
            plt.grid(True, alpha=0.3)
            box_file = os.path.join(self.output_dir, "valence_boxplot.png")
            plt.savefig(box_file, dpi=300)
            plt.close()
            logger.info(f"Saved box plot to {box_file}")
            
            # 5. Heatmap of values in different ranges
            plt.figure(figsize=(10, 8))
            
            # Create bins
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
            
            # Digitize the values into bins
            df['openai_bin'] = pd.cut(df['openai_valence'], bins=bins, labels=labels, include_lowest=True)
            df['zeroshot_bin'] = pd.cut(df['zeroshot_valence'], bins=bins, labels=labels, include_lowest=True)
            
            # Create a cross-tabulation
            heatmap_data = pd.crosstab(df['zeroshot_bin'], df['openai_bin'])
            
            # Plot heatmap
            plt.imshow(heatmap_data, cmap='YlGnBu')
            plt.colorbar(label='Count')
            plt.xticks(np.arange(len(labels)), labels, rotation=45)
            plt.yticks(np.arange(len(labels)), labels)
            plt.xlabel('OpenAI Valence')
            plt.ylabel('Zero-Shot Classification Valence')
            plt.title('Heatmap of Valence Value Ranges')
            
            # Add text annotations
            for i in range(len(heatmap_data.index)):
                for j in range(len(heatmap_data.columns)):
                    plt.text(j, i, heatmap_data.iloc[i, j], 
                             ha="center", va="center", color="black")
            
            heat_file = os.path.join(self.output_dir, "valence_heatmap.png")
            plt.savefig(heat_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved heatmap to {heat_file}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    def _identify_outliers(self, df):
        """Identify claims with large differences between valence values."""
        # Sort by absolute difference
        large_diff = df.sort_values('abs_diff', ascending=False).head(20)
        
        # Save to file
        outliers_file = os.path.join(self.output_dir, "valence_outliers.txt")
        with open(outliers_file, 'w') as f:
            f.write("=== CLAIMS WITH LARGEST VALENCE DIFFERENCES ===\n\n")
            
            for _, row in large_diff.iterrows():
                f.write(f"Difference: {row['valence_diff']:.3f} (abs: {row['abs_diff']:.3f})\n")
                f.write(f"Zero-Shot Classification: {row['zeroshot_valence']:.3f}, OpenAI: {row['openai_valence']:.3f}\n")
                f.write(f"Claim: {row['text']}\n")
                f.write("-" * 80 + "\n\n")
        
        logger.info(f"Saved outliers to {outliers_file}")
        
        # Print top 5 outliers
        logger.info("\n=== TOP 5 CLAIMS WITH LARGEST VALENCE DIFFERENCES ===")
        for _, row in large_diff.head(5).iterrows():
            logger.info(f"Difference: {row['valence_diff']:.3f} (abs: {row['abs_diff']:.3f})")
            logger.info(f"Zero-Shot Classification: {row['zeroshot_valence']:.3f}, OpenAI: {row['openai_valence']:.3f}")
            logger.info(f"Claim: {row['text']}")
            logger.info("-" * 80)
            
        # Cases where OpenAI thinks it's therapeutic but Zero-Shot Classification doesn't
        openai_high = df[(df['openai_valence'] > 0.7) & (df['zeroshot_valence'] < 0.3)].sort_values('abs_diff', ascending=False)
        
        if len(openai_high) > 0:
            logger.info("\n=== CLAIMS WHERE OPENAI IS HIGH BUT ZERO-SHOT CLASSIFICATION IS LOW ===")
            openai_file = os.path.join(self.output_dir, "openai_high_zeroshot_low.txt")
            with open(openai_file, 'w') as f:
                f.write("=== CLAIMS WHERE OPENAI IS HIGH BUT ZERO-SHOT CLASSIFICATION IS LOW ===\n\n")
                
                for _, row in openai_high.iterrows():
                    logger.info(f"Difference: {row['valence_diff']:.3f}")
                    logger.info(f"Zero-Shot Classification: {row['zeroshot_valence']:.3f}, OpenAI: {row['openai_valence']:.3f}")
                    logger.info(f"Claim: {row['text']}")
                    logger.info("-" * 80)
                    
                    f.write(f"Difference: {row['valence_diff']:.3f}\n")
                    f.write(f"Zero-Shot Classification: {row['zeroshot_valence']:.3f}, OpenAI: {row['openai_valence']:.3f}\n")
                    f.write(f"Claim: {row['text']}\n")
                    f.write("-" * 80 + "\n\n")
            
            logger.info(f"Saved OpenAI high / Zero-Shot Classification low cases to {openai_file}")
        
        # Cases where Zero-Shot Classification thinks it's therapeutic but OpenAI doesn't
        zeroshot_high = df[(df['zeroshot_valence'] > 0.7) & (df['openai_valence'] < 0.3)].sort_values('abs_diff', ascending=False)
        
        if len(zeroshot_high) > 0:
            logger.info("\n=== CLAIMS WHERE ZERO-SHOT CLASSIFICATION IS HIGH BUT OPENAI IS LOW ===")
            zeroshot_file = os.path.join(self.output_dir, "zeroshot_high_openai_low.txt")
            with open(zeroshot_file, 'w') as f:
                f.write("=== CLAIMS WHERE ZERO-SHOT CLASSIFICATION IS HIGH BUT OPENAI IS LOW ===\n\n")
                
                for _, row in zeroshot_high.iterrows():
                    logger.info(f"Difference: {row['valence_diff']:.3f}")
                    logger.info(f"Zero-Shot Classification: {row['zeroshot_valence']:.3f}, OpenAI: {row['openai_valence']:.3f}")
                    logger.info(f"Claim: {row['text']}")
                    logger.info("-" * 80)
                    
                    f.write(f"Difference: {row['valence_diff']:.3f}\n")
                    f.write(f"Zero-Shot Classification: {row['zeroshot_valence']:.3f}, OpenAI: {row['openai_valence']:.3f}\n")
                    f.write(f"Claim: {row['text']}\n")
                    f.write("-" * 80 + "\n\n")
            
            logger.info(f"Saved Zero-Shot Classification high / OpenAI low cases to {zeroshot_file}")
    
    def _calculate_icc(self, df):
        if not PINGOUIN_AVAILABLE:
            logger.warning("pingouin not available")
            return None

        try:
            # 1) build long DataFrame correctly
            n = len(df)
            data_long = pd.DataFrame({
                'target': np.repeat(np.arange(n), 2),
                'rater':  np.tile(['openai','zeroshot'], n),
                'rating': np.vstack([df['openai_valence'], df['zeroshot_valence']]).T.flatten()
            })

            # 2) run pingouin
            icc_df = pg.intraclass_corr(
                data=data_long,
                targets='target',
                raters='rater',
                ratings='rating',
                nan_policy='omit'
            )
            icc_row = icc_df.loc[icc_df['Type']=='ICC3', :].iloc[0].to_dict()
            logger.info(f"ICC(3,1) = {icc_row['ICC']:.3f}")
            return icc_row

        except Exception as e:
            logger.error("pingouin ICC failed, falling back to manual", exc_info=e)

            # Manual one-way random-effects ICC(1,1)
            openai_vals = df['openai_valence'].to_numpy()
            zeroshot_vals = df['zeroshot_valence'].to_numpy()
            k = 2
            n = len(df)
            grand_mean = np.mean(np.concatenate([openai_vals, zeroshot_vals]))
            row_means  = (openai_vals + zeroshot_vals) / 2

            MSB = (k * ((row_means - grand_mean)**2).sum()) / (n - 1)
            MSE = ((openai_vals - row_means)**2 + (zeroshot_vals - row_means)**2).sum() / (n * (k - 1))

            icc_val = (MSB - MSE) / (MSB + (k - 1) * MSE)
            logger.info(f"Manual ICC(1,1) = {icc_val:.3f}")
            return {'ICC': icc_val, 'Type': 'ICC1,1'}

    
    def close(self):
        """Close database connection."""
        self.db_connector.close()

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ValenceAnalyzer()
    
    try:
        # Run analysis
        analyzer.analyze_claim_valences()
        
        print(f"\nAnalysis complete. Results saved to {analyzer.output_dir}/")
        
    finally:
        # Ensure resources are closed
        analyzer.close()