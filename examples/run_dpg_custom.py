import sys
import os

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

import yaml
import argparse
import dpg.sklearn_dpg as test



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom DPG pipeline runner.")
    parser.add_argument("--dataset", type=str, required=True, help="Basic dataset to be analyzed")
    parser.add_argument("--target_column", type=str, help="Name of the column to be used as the target variable")
    parser.add_argument("--n_learners", type=int, default=5, help="Number of learners for the Ensemble model")
    parser.add_argument("--model_name", type=str, default="RandomForestClassifier", help="Chosen tree-based ensemble model")
    parser.add_argument("--dir", type=str, default=os.path.join(SCRIPT_DIR, "results"), help="Directory to save results")
    parser.add_argument("--no-plot", dest='plot', action='store_false', help="Disable exporting the DPG plot image (exported by default)")
    parser.set_defaults(plot=True)
    parser.add_argument("--save_plot_dir", type=str, default=os.path.join(SCRIPT_DIR, "results"), help="Directory to save the plot image")
    parser.add_argument("--attribute", type=str, default=None, help="A specific node attribute to visualize")
    parser.add_argument("--communities", action='store_true', help="Boolean indicating whether to visualize communities, add the argument to use it as True")
    parser.add_argument("--clusters", action='store_true', help="Boolean indicating whether to visualize clusters, add the argument to use it as True")
    parser.add_argument("--threshold_clusters", type=float, default=None, help="Threshold for detecting ambiguous nodes in clusters")
    parser.add_argument("--class_flag", action='store_true', help="Boolean indicating whether to highlight class nodes, add the argument to use it as True")
    parser.add_argument("--seed", type=int, default=160898, help="Randomicity control")
    parser.add_argument("--pv", type=float, default=None, help="Override perc_var from config")
    parser.add_argument("--t", type=int, default=None, help="Override decimal_threshold from config")
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
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
    if args.pv is not None:
        pv = args.pv
    if args.t is not None:
        t = args.t

    os.makedirs(args.dir, exist_ok=True)

    df, df_edges, df_dpg_metrics, clusters, node_prob, confidence = test.test_dpg(
                                        datasets = args.dataset,
                                        target_column = args.target_column,
                                        n_learners = args.n_learners, 
                                        perc_var = pv, 
                                        decimal_threshold = t,
                                        n_jobs = j,
                                        model_name = args.model_name,
                                        file_name = os.path.join(args.dir, f'custom_l{args.n_learners}_pv{pv}_t{t}_stats.txt'), 
                                        plot = args.plot, 
                                        save_plot_dir = args.save_plot_dir, 
                                        attribute = args.attribute, 
                                        communities = args.communities,
                                        clusters_flag = args.clusters,
                                        threshold_clusters = args.threshold_clusters,
                                        class_flag = args.class_flag,
                                        seed = args.seed
                                        )

    df.sort_values(['Degree'])

    df.to_csv(os.path.join(args.dir, f'custom_l{args.n_learners}_pv{pv}_t{t}_node_metrics.csv'),
                encoding='utf-8')

    with open(os.path.join(args.dir, f'custom_l{args.n_learners}_pv{pv}_t{t}_dpg_metrics.txt'), 'w') as f:
        for key, value in df_dpg_metrics.items():
            f.write(f"{key}: {value}\n")
        
