"""
Main script to execute the drug classification ML pipeline using Random Forest.
This script provides a CLI interface to run different stages of the pipeline.

Usage examples:
    python main.py --action prepare --data drug200.csv
    python main.py --action train --n_estimators 100
    python main.py --action evaluate
    python main.py --action optimize
    python main.py --action full_pipeline --data drug200.csv
"""

import argparse
import sys
import os
from model_pipeline import (
    load_data, explore_data, prepare_data, train_model,
    evaluate_model, save_model, load_model,
    visualize_results, optimize_hyperparameters
)


def main():
    """Main function to handle CLI arguments and execute pipeline stages."""

    parser = argparse.ArgumentParser(
        description='Drug Classification ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py --action full_pipeline --data drug200.csv

  # Prepare data only
  python main.py --action prepare --data drug200.csv

  # Train Random Forest model
  python main.py --action train --n_estimators 100

  # Evaluate the model
  python main.py --action evaluate

  # Optimize hyperparameters
  python main.py --action optimize
        """
    )

    parser.add_argument(
        '--action',
        type=str,
        required=True,
        choices=['explore', 'prepare', 'train', 'evaluate', 'save', 'load',
                 'full_pipeline', 'optimize'],
        help='Action to perform'
    )

    parser.add_argument(
        '--data',
        type=str,
        default='drug200.csv',
        help='Path to the dataset CSV file (default: drug200.csv)'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='models/random_forest_model.pkl',
        help='Path to save/load the model (default: models/random_forest_model.pkl)')

    parser.add_argument(
        '--test_size',
        type=float,
        default=0.3,
        help='Test set size proportion (default: 0.3)'
    )

    parser.add_argument(
        '--no_smote',
        action='store_true',
        help='Disable SMOTE oversampling'
    )

    parser.add_argument(
        '--random_state',
        type=int,
        default=0,
        help='Random state for reproducibility (default: 0)'
    )

    # Random Forest parameters
    parser.add_argument(
        '--n_estimators',
        type=int,
        default=100,
        help='Number of estimators for Random Forest (default: 100)')
    parser.add_argument('--max_leaf_nodes', type=int, default=30,
                        help='Max leaf nodes for Random Forest (default: 30)')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" " * 15 + "DRUG CLASSIFICATION ML PIPELINE")
    print("=" * 70)

    # Global variables to store data
    global X_train, X_test, y_train, y_test, trained_model

    try:
        if args.action == 'explore':
            # Explore data
            df = load_data(args.data)
            if df is not None:
                explore_data(df)

        elif args.action == 'prepare':
            # Prepare data
            df = load_data(args.data)
            if df is not None:
                X_train, X_test, y_train, y_test = prepare_data(
                    df,
                    test_size=args.test_size,
                    random_state=args.random_state,
                    apply_smote=not args.no_smote
                )
                # Save prepared data
                import joblib
                os.makedirs('data', exist_ok=True)
                joblib.dump((X_train, X_test, y_train, y_test),
                            'data/prepared_data.pkl')
                print("\n[OK] Prepared data saved to 'data/prepared_data.pkl'")

        elif args.action == 'train':
            # Load prepared data
            import joblib
            if os.path.exists('data/prepared_data.pkl'):
                X_train, X_test, y_train, y_test = joblib.load(
                    'data/prepared_data.pkl')
                print("[OK] Loaded prepared data from 'data/prepared_data.pkl'")
            else:
                print("[ERROR] Prepared data not found. Run 'prepare' action first.")
                sys.exit(1)

            # Train Random Forest model
            trained_model = train_model(
                X_train, y_train,
                n_estimators=args.n_estimators,
                max_leaf_nodes=args.max_leaf_nodes,
                random_state=args.random_state
            )

            # Save model automatically
            os.makedirs('models', exist_ok=True)
            model_filename = 'models/random_forest_model.pkl'
            save_model(trained_model, model_filename)

        elif args.action == 'evaluate':
            # Load prepared data and model
            import joblib
            if os.path.exists('data/prepared_data.pkl'):
                X_train, X_test, y_train, y_test = joblib.load(
                    'data/prepared_data.pkl')
            else:
                print("[ERROR] Prepared data not found. Run 'prepare' action first.")
                sys.exit(1)

            model_filename = 'models/random_forest_model.pkl'
            trained_model = load_model(model_filename)

            if trained_model is not None:
                results = evaluate_model(
                    trained_model, X_test, y_test, 'Random Forest')
                visualize_results(
                    y_test, results['predictions'], 'Random Forest')

        elif args.action == 'optimize':
            # Load prepared data
            import joblib
            if os.path.exists('data/prepared_data.pkl'):
                X_train, X_test, y_train, y_test = joblib.load(
                    'data/prepared_data.pkl')
            else:
                print("[ERROR] Prepared data not found. Run 'prepare' action first.")
                sys.exit(1)

            # Optimize Random Forest hyperparameters (max_leaf_nodes)
            print("\n[OK] Optimizing Random Forest hyperparameters...")
            results = optimize_hyperparameters(
                X_train, X_test, y_train, y_test,
                param_name='max_leaf_nodes'
            )

            # Save optimization results
            import joblib
            os.makedirs('results', exist_ok=True)
            joblib.dump(results, 'results/random_forest_optimization.pkl')
            print(
                f"\n[OK] Optimization results saved to 'results/random_forest_optimization.pkl'")

        elif args.action == 'full_pipeline':
            # Execute complete pipeline
            print("\n>>> STEP 1: Loading Data")
            df = load_data(args.data)

            if df is None:
                sys.exit(1)

            print("\n>>> STEP 2: Exploring Data")
            explore_data(df)

            print("\n>>> STEP 3: Preparing Data")
            X_train, X_test, y_train, y_test = prepare_data(
                df,
                test_size=args.test_size,
                random_state=args.random_state,
                apply_smote=not args.no_smote
            )

            print("\n>>> STEP 4: Training Random Forest Model")
            trained_model = train_model(
                X_train, y_train,
                n_estimators=args.n_estimators,
                max_leaf_nodes=args.max_leaf_nodes,
                random_state=args.random_state
            )

            print("\n>>> STEP 5: Evaluating Model")
            results = evaluate_model(
                trained_model, X_test, y_test, 'Random Forest')

            print("\n>>> STEP 6: Saving Model")
            os.makedirs('models', exist_ok=True)
            model_filename = 'models/random_forest_model.pkl'
            save_model(trained_model, model_filename)

            print("\n>>> STEP 7: Visualizing Results")
            visualize_results(y_test, results['predictions'], 'Random Forest')

            print("\n" + "=" * 70)
            print(" " * 20 + "PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 70)

        else:
            print(f"[ERROR] Unknown action: {args.action}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n[ERROR] Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
