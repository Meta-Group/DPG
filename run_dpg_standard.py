import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from collections import defaultdict
import re
import pandas as pd
import yaml
import argparse
import random
import dpg.sklearn_dpg as test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="iris", help="Basic dataset to be analyzed")
    parser.add_argument("--l", type=int, default=5, help="Number of learners for the Random Forest")
    parser.add_argument("--model_name", type=str, default="RandomForestClassifier", help="Chosen tree-based ensemble model")
    parser.add_argument("--dir", type=str, default="examples/", help="Directory to save results")
    parser.add_argument("--plot", action='store_true', help="Plot the DPG, add the argument to use it as True")
    parser.add_argument("--save_plot_dir", type=str, default="examples/", help="Directory to save the plot image")
    parser.add_argument("--attribute", type=str, default=None, help="A specific node attribute to visualize")
    parser.add_argument("--communities", action='store_true', help="Boolean indicating whether to visualize communities, add the argument to use it as True")
    parser.add_argument("--clusters", action='store_true', help="Boolean indicating whether to visualize clusters, add the argument to use it as True")
    parser.add_argument("--threshold_clusters", type=float, default=None, help="Threshold for detecting ambiguous nodes in clusters")
    parser.add_argument("--class_flag", action='store_true', help="Boolean indicating whether to highlight class nodes, add the argument to use it as True")
    parser.add_argument("--seed", type=int, help="Randomicity control")
    args = parser.parse_args()

    config_path="config.yaml"
    try:
        with open(config_path) as f:
                config = yaml.safe_load(f)

    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file: {str(e)}")
    
    pv = config['dpg']['default']['perc_var']
    t = config['dpg']['default']['decimal_threshold']
    j = config['dpg']['default']['n_jobs']
        
    df, df_edges, df_dpg_metrics, clusters, node_prob, confidence = test.test_dpg(datasets = args.ds,
                                        n_learners = args.l, 
                                        perc_var = pv, 
                                        decimal_threshold = t,
                                        n_jobs = j,
                                        model_name = args.model_name,
                                        file_name = os.path.join(args.dir, f'{args.ds}_l{args.l}_seed{args.seed}_stats.txt'), 
                                        plot = args.plot, 
                                        save_plot_dir = args.save_plot_dir, 
                                        attribute = args.attribute, 
                                        communities = args.communities,
                                        clusters_flag = args.clusters,
                                        threshold_clusters = args.threshold_clusters,
                                        class_flag = args.class_flag,
                                        seed = args.seed)
        
    df.sort_values(['Degree'])

    df.to_csv(os.path.join(args.dir, f'{args.ds}_l{args.l}_seed{args.seed}_node_metrics.csv'),
                encoding='utf-8')

    with open(os.path.join(args.dir, f'{args.ds}_l{args.l}_seed{args.seed}_dpg_metrics.txt'), 'w') as f:
        for key, value in df_dpg_metrics.items():
            f.write(f"{key}: {value}\n")
    


    
    if args.clusters:
        
        def extract_feature_intervals(decisions):
            feature_count = defaultdict(int)
            feature_intervals = defaultdict(lambda: {"min": float('-inf'), "max": float('inf')})
            
            regex = r'([a-zA-Z0-9_]+)\s*([<=|>]+)\s*([-+]?[\d.]+)'
            
            for decision in decisions:
                match = re.search(regex, decision)
                if match:
                    feature, operator, value = match.groups()
                    value = float(value)
                    feature_count[feature] += 1
                    
                    if '>' in operator:
                        feature_intervals[feature]["min"] = max(feature_intervals[feature]["min"], value)
                    elif '<=' in operator:
                        feature_intervals[feature]["max"] = min(feature_intervals[feature]["max"], value)
                        
            return feature_count, feature_intervals

        def create_dataframes(data):
            all_found_features = set()
            temp_counts = {}
            temp_intervals = {}

            for class_name, decisions in data.items():
                counts, intervals = extract_feature_intervals(decisions)
                temp_counts[class_name] = counts
                temp_intervals[class_name] = intervals
                all_found_features.update(counts.keys())

            sorted_features = sorted(list(all_found_features))
            
            feature_count_df = pd.DataFrame(index=sorted_features, columns=data.keys())
            
            interval_index = []
            for f in sorted_features:
                interval_index.extend([f"{f}_min", f"{f}_max"])
            feature_intervals_df = pd.DataFrame(index=interval_index, columns=data.keys())

            for class_name in data.keys():
                for feat in sorted_features:
                    feature_count_df.loc[feat, class_name] = temp_counts[class_name].get(feat, 0)
                    
                    inter = temp_intervals[class_name].get(feat, {"min": float('-inf'), "max": float('inf')})
                    feature_intervals_df.loc[f"{feat}_min", class_name] = inter["min"]
                    feature_intervals_df.loc[f"{feat}_max", class_name] = inter["max"]

            return feature_count_df, feature_intervals_df
        


        node_to_label = df.set_index('Node')['Label'].to_dict()
        clusters_labels = {k: [node_to_label.get(n, n) for n in v] for k, v in clusters.items()}
        node_probs_labels = {node_to_label.get(str(k), str(k)): v for k, v in node_prob.items()}
        confidence_labels = {node_to_label.get(str(k), str(k)): v for k, v in confidence.items()}
        feature_count_df, feature_intervals_df = create_dataframes(clusters_labels)
        feature_count_df.to_csv(os.path.join(args.dir, f'{args.ds}_l{args.l}_seed{args.seed}_t{args.threshold_clusters}_dpg_clusters_count.csv'))
        feature_intervals_df.to_csv(os.path.join(args.dir, f'{args.ds}_l{args.l}_seed{args.seed}_t{args.threshold_clusters}_dpg_clusters_intervals.csv'))
        with open(os.path.join(args.dir, f'{args.ds}_l{args.l}_seed{args.seed}_t{args.threshold_clusters}_dpg_clusters.txt'), 'w') as f:
            f.write("Clusters:\n")
            for key, value in clusters_labels.items():
                f.write(f"{key}: {value}\n")
            f.write("\n\nProbability:\n")
            for key, value in node_probs_labels.items():
                f.write(f"{key}: {value}\n")
            f.write("\n\nConfidence Interval:\n")
            for key, value in confidence_labels.items():
                f.write(f"{key}: {value}\n")


# python run_dpg_standard.py --ds datasets\tpot_clustered_files\thy.csv --l 5 --dir examples\tpot --plot --save_plot_dir examples\tpot --clusters --threshold_clusters 0.2 --seed 160898S