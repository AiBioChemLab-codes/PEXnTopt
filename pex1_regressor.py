import argparse
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
import xgboost as xgb
from sklearn.preprocessing import StandardScaler  # replace cuml
from sklearn.metrics import r2_score  # replace cuml
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import joblib
import warnings
warnings.filterwarnings('ignore')

class XGBoostRegressor:  # remove GPU identifier from class name
    def __init__(self, project_name, output_prefix, k_fold=5):
        self.project_name = project_name
        self.output_prefix = output_prefix
        self.k_fold = k_fold
        self.scaler = None
        self.best_model = None
        self.best_params = None
        self.setup_directories()
        self.setup_logging()
        
    def setup_directories(self):
        """Create project directory structure"""
        self.dirs = {
            'root': self.project_name,
            'models': os.path.join(self.project_name, 'models'),
            'results': os.path.join(self.project_name, 'results'),
            'logs': os.path.join(self.project_name, 'logs'),
            'scalers': os.path.join(self.project_name, 'scalers'),
            'cv_results': os.path.join(self.project_name, 'cv_results'),
            'hyperopt': os.path.join(self.project_name, 'hyperopt_results')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging"""
        log_file = os.path.join(
            self.dirs['logs'], 
            f"{self.output_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        self.logger.info(f"Project initialized: {self.project_name}")
        self.logger.info(f"XGBoost version: {xgb.__version__}")
    
    def load_data(self, train_file, test_file):
        """Load training and test data"""
        self.logger.info("Loading data...")
        
        try:
            # Read training set
            train_df = pd.read_csv(train_file)
            self.logger.info(f"Train set size: {train_df.shape}")
            
            # Read test set
            test_df = pd.read_csv(test_file)
            self.logger.info(f"Test set size: {test_df.shape}")
            
            # Extract features and target
            self.X_train = train_df.iloc[:, 2:].values.astype(np.float32)
            self.y_train = train_df.iloc[:, 1].values.astype(np.float32)
            self.X_test = test_df.iloc[:, 2:].values.astype(np.float32)
            self.y_test = test_df.iloc[:, 1].values.astype(np.float32)
            
            # Record basic data info
            self.n_samples_train = len(self.y_train)
            self.n_features = self.X_train.shape[1]
            
            # Check data quality
            self._check_data_quality()
            
            self.logger.info("Data loading complete")
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise
    
    def _check_data_quality(self):
        """Check data quality"""
        self.logger.info("Checking data quality...")
        
        # Check NaN values
        if np.isnan(self.X_train).any() or np.isnan(self.X_test).any():
            self.logger.warning("NaN values detected, filling")
            self.X_train = np.nan_to_num(self.X_train)
            self.X_test = np.nan_to_num(self.X_test)
        
        # Check infinite values
        if np.isinf(self.X_train).any() or np.isinf(self.X_test).any():
            self.logger.warning("Infinite values detected, replacing")
            self.X_train = np.where(np.isinf(self.X_train), 0, self.X_train)
            self.X_test = np.where(np.isinf(self.X_test), 0, self.X_test)
        
        # Check target range
        self.logger.info(f"Train target range: [{self.y_train.min():.6f}, {self.y_train.max():.6f}]")
        self.logger.info(f"Test target range: [{self.y_test.min():.6f}, {self.y_test.max():.6f}]")
        
        # Check feature scale differences
        feature_ranges = np.ptp(self.X_train, axis=0)
        if np.max(feature_ranges) / np.min(feature_ranges[feature_ranges > 0]) > 1000:
            self.logger.warning("Feature scales vary greatly, standardization is important")
    
    def standardize_data(self):
        """Standardize data"""
        self.logger.info("Starting data standardization...")
        
        try:
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            # Save scaler
            scaler_file = os.path.join(
                self.dirs['scalers'], 
                f"{self.output_prefix}_scaler.joblib"
            )
            joblib.dump(self.scaler, scaler_file)
            self.logger.info(f"Scaler saved: {scaler_file}")
        except Exception as e:
            self.logger.error(f"Data standardization failed: {str(e)}")
            raise
    
    def calculate_adjusted_r2(self, r2, n_samples, n_features):
        """Calculate adjusted R-squared"""
        if n_samples <= n_features + 1:
            return 0.0
        return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    
    def calculate_metrics(self, y_true, y_pred, dataset_type='test'):
        """Calculate all regression metrics"""
        try:
            # Basic regression metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Calculate sample size and feature count for adjusted R2
            n_samples = len(y_true)
            n_features = self.n_features
            
            # Adjusted R-squared
            adj_r2 = self.calculate_adjusted_r2(r2, n_samples, n_features)
            
            # Correlation coefficients
            pearson_corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
            spearman_corr, _ = spearmanr(y_true.flatten(), y_pred.flatten())
            
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'adjusted_r2': float(adj_r2),
                'pearson_corr': float(pearson_corr),
                'spearman_corr': float(spearman_corr)
            }
            
            # Format values to 6 significant figures
            formatted_metrics = {}
            for key, value in metrics.items():
                if np.isfinite(value):
                    formatted_metrics[key] = float(f"{value:.6g}")
                else:
                    formatted_metrics[key] = 0.0
            
            return formatted_metrics
        except Exception as e:
            self.logger.error(f"Error calculating {dataset_type} metrics: {str(e)}")
            return {key: 0.0 for key in [
                'mse', 'rmse', 'mae', 'r2', 'adjusted_r2', 'pearson_corr', 'spearman_corr'
            ]}
    
    def safe_xgb_fit(self, model, X, y, eval_set=None):
        """Safe XGBoost training method"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if eval_set:
                    model.fit(X, y, eval_set=eval_set, verbose=False)
                else:
                    model.fit(X, y, verbose=False)
                return model
            except Exception as e:
                self.logger.warning(f"XGBoost training attempt {attempt + 1}/{max_attempts} failed: {str(e)}")
                if attempt < max_attempts - 1:
                    time.sleep(1)
                else:
                    raise
    
    def objective_function(self, params):
        """Hyperopt objective function"""
        try:
            start_time = time.time()
            
            # Set XGBoost parameters (using CPU hist algorithm)
            xgb_params = {
                'n_estimators': int(params['n_estimators']),
                'max_depth': int(params['max_depth']),
                'learning_rate': float(params['learning_rate']),
                'subsample': float(params['subsample']),
                'colsample_bytree': float(params['colsample_bytree']),
                'reg_alpha': float(params['reg_alpha']),
                'reg_lambda': float(params['reg_lambda']),
                'min_child_weight': float(params['min_child_weight']),
                'gamma': float(params['gamma']),
                'tree_method': 'hist',  # key change: CPU histogram algorithm
                'random_state': 42
            }
            
            self.logger.info(f"Training model, params: n_estimators={xgb_params['n_estimators']}, max_depth={xgb_params['max_depth']}, learning_rate={xgb_params['learning_rate']:.6f}")
            
            # Train XGBoost model
            xgb_model = xgb.XGBRegressor(**xgb_params)
            xgb_model = self.safe_xgb_fit(xgb_model, self.X_train_scaled, self.y_train)
            
            # Predict on train and test sets
            y_train_pred = xgb_model.predict(self.X_train_scaled)
            y_test_pred = xgb_model.predict(self.X_test_scaled)
            
                # Calculate metrics
            train_metrics = self.calculate_metrics(self.y_train, y_train_pred, 'train')
            test_metrics = self.calculate_metrics(self.y_test, y_test_pred, 'test')
            
            # Primary optimization target: RMSE (smaller is better), also consider R2 and adjusted R2 (larger is better)
            # Composite score: reciprocal of RMSE + R2 + adjusted R2
            rmse_weight = 1.0 / (test_metrics['rmse'] + 1e-8)  # avoid division by zero
            r2_weight = test_metrics['r2']
            adj_r2_weight = test_metrics['adjusted_r2']
            
            composite_score = rmse_weight + max(0, r2_weight) + max(0, adj_r2_weight)
            
            # Record results
            result = {
                'params': params,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'composite_score': composite_score,
                'training_time': time.time() - start_time
            }
            
            self.hyperopt_results.append(result)
            
            self.logger.info(f"Training complete, RMSE: {test_metrics['rmse']:.6f}, R²: {test_metrics['r2']:.6f}, time: {result['training_time']:.2f}s")
            
            # Hyperopt minimizes, so return negative composite score
            return {'loss': -composite_score, 'status': STATUS_OK}
            
        except Exception as e:
            self.logger.error(f"Objective function error: {str(e)}")
            return {'loss': 1.0, 'status': STATUS_FAIL}
    
    def hyperparameter_optimization(self, max_evals=100):
        """Run hyperparameter optimization"""
        self.logger.info("Starting hyperparameter optimization...")
        self.hyperopt_results = []
        
        # Define XGBoost search space
        space = {
            'n_estimators': hp.quniform('n_estimators', 50, 1000, 50),
            'max_depth': hp.quniform('max_depth', 3, 15, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(0.001), np.log(10)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(0.001), np.log(10)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'gamma': hp.loguniform('gamma', np.log(0.001), np.log(1))
        }
        
        try:
            trials = Trials()
            best = fmin(
                fn=self.objective_function,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials
            )
            
            # Save hyperparameter optimization results
            self.save_hyperopt_results()
            self.logger.info("Hyperparameter optimization complete")
            
            return best
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {str(e)}")
            self.logger.info("Continuing with default parameters")
            self.hyperopt_results = [{
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'min_child_weight': 1,
                    'gamma': 0
                },
                'train_metrics': {key: 0.0 for key in ['mse', 'rmse', 'mae', 'r2', 'adjusted_r2', 'pearson_corr', 'spearman_corr']},
                'test_metrics': {key: 0.0 for key in ['mse', 'rmse', 'mae', 'r2', 'adjusted_r2', 'pearson_corr', 'spearman_corr']},
                'composite_score': 0.0,
                'training_time': 0.0
            }]
            return {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'min_child_weight': 1,
                'gamma': 0
            }
    
    def save_hyperopt_results(self):
        """Save hyperparameter optimization results"""
        if not self.hyperopt_results:
            self.logger.warning("No hyperparameter optimization results to save")
            return
            
        results_df = pd.DataFrame([{
            **r['params'],
            **{f'train_{k}': v for k, v in r['train_metrics'].items()},
            **{f'test_{k}': v for k, v in r['test_metrics'].items()},
            'composite_score': r['composite_score'],
            'training_time': r['training_time']
        } for r in self.hyperopt_results])
        
        results_file = os.path.join(
            self.dirs['hyperopt'], 
            f"{self.output_prefix}_hyperopt_results.csv"
        )
        results_df.to_csv(results_file, index=False)
        self.logger.info(f"Hyperparameter results saved: {results_file}")
    
    def find_best_model(self):
        """Find the best model"""
        if not self.hyperopt_results:
            self.logger.error("No hyperparameter optimization results available")
            raise ValueError("No hyperparameter optimization results available")
            
        # Select model with smallest RMSE and largest R2 & adjusted R2
        best_result = min(self.hyperopt_results, 
                         key=lambda x: (x['test_metrics']['rmse'], 
                                      -x['test_metrics']['r2'], 
                                      -x['test_metrics']['adjusted_r2']))
        
        self.best_params = best_result['params']
        self.best_test_metrics = best_result['test_metrics']
        
        self.logger.info(f"Best params: n_estimators={self.best_params['n_estimators']}, max_depth={self.best_params['max_depth']}, learning_rate={self.best_params['learning_rate']:.6f}")
        self.logger.info(f"Best metrics - RMSE: {self.best_test_metrics['rmse']:.6f}, R²: {self.best_test_metrics['r2']:.6f}, Adjusted R²: {self.best_test_metrics['adjusted_r2']:.6f}")
        
        # Train final model
        xgb_params = {
            'n_estimators': int(self.best_params['n_estimators']),
            'max_depth': int(self.best_params['max_depth']),
            'learning_rate': float(self.best_params['learning_rate']),
            'subsample': float(self.best_params['subsample']),
            'colsample_bytree': float(self.best_params['colsample_bytree']),
            'reg_alpha': float(self.best_params['reg_alpha']),
            'reg_lambda': float(self.best_params['reg_lambda']),
            'min_child_weight': float(self.best_params['min_child_weight']),
            'gamma': float(self.best_params['gamma']),
            'tree_method': 'hist',  # key change: CPU histogram algorithm
            'random_state': 42
        }
        
        self.best_model = xgb.XGBRegressor(**xgb_params)
        
        try:
            self.best_model = self.safe_xgb_fit(self.best_model, self.X_train_scaled, self.y_train)
            
            # Save best model
            model_file = os.path.join(
                self.dirs['models'], 
                f"{self.output_prefix}_best_model.joblib"
            )
            joblib.dump(self.best_model, model_file)
            
            # Save best metrics
            metrics_df = pd.DataFrame([self.best_test_metrics])
            metrics_file = os.path.join(
                self.dirs['results'], 
                f"{self.output_prefix}_best_metrics.csv"
            )
            metrics_df.to_csv(metrics_file, index=False, float_format='%.6f')
            
            self.logger.info(f"Best model saved: {model_file}")
        except Exception as e:
            self.logger.error(f"Final model training failed: {str(e)}")
            self.best_model = None
    
    def cross_validation(self):
        """Run K-fold cross-validation"""
        self.logger.info("Starting K-fold cross-validation...")
        
        if self.best_params is None:
            self.logger.error("No best parameters found, cannot perform cross-validation")
            return None
            
        cv_results = []
        kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train)):
            fold_start_time = time.time()
            
            try:
                # Split data
                X_train_fold, X_val_fold = self.X_train[train_idx], self.X_train[val_idx]
                y_train_fold, y_val_fold = self.y_train[train_idx], self.y_train[val_idx]
                
                # Standardize (fit on training fold)
                scaler_fold = StandardScaler()
                X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
                X_val_fold_scaled = scaler_fold.transform(X_val_fold)
                X_test_fold_scaled = scaler_fold.transform(self.X_test)
                
                # Train model
                xgb_params = {
                    'n_estimators': int(self.best_params['n_estimators']),
                    'max_depth': int(self.best_params['max_depth']),
                    'learning_rate': float(self.best_params['learning_rate']),
                    'subsample': float(self.best_params['subsample']),
                    'colsample_bytree': float(self.best_params['colsample_bytree']),
                    'reg_alpha': float(self.best_params['reg_alpha']),
                    'reg_lambda': float(self.best_params['reg_lambda']),
                    'min_child_weight': float(self.best_params['min_child_weight']),
                    'gamma': float(self.best_params['gamma']),
                    'tree_method': 'hist',  # key change: CPU histogram algorithm
                    'random_state': 42
                }
                
                model = xgb.XGBRegressor(**xgb_params)
                model = self.safe_xgb_fit(model, X_train_fold_scaled, y_train_fold)
                
                # Validation set prediction
                y_val_pred = model.predict(X_val_fold_scaled)
                
                # Test set prediction
                y_test_pred = model.predict(X_test_fold_scaled)
                
            # Calculate metrics
                val_metrics = self.calculate_metrics(y_val_fold, y_val_pred, 'validation')
                test_metrics = self.calculate_metrics(self.y_test, y_test_pred, 'test')
                
                cv_results.append({
                    'fold': fold + 1,
                    **{f'val_{k}': v for k, v in val_metrics.items()},
                    **{f'test_{k}': v for k, v in test_metrics.items()},
                    'training_time': time.time() - fold_start_time
                })
                
                self.logger.info(f"Fold {fold + 1} done, val RMSE: {val_metrics['rmse']:.6f}, test RMSE: {test_metrics['rmse']:.6f}")
                
            except Exception as e:
                self.logger.error(f"Fold {fold + 1} cross-validation failed: {str(e)}")
                continue
        
        if not cv_results:
            self.logger.error("All cross-validation folds failed")
            return None
            
        # Save cross-validation results
        cv_df = pd.DataFrame(cv_results)
        cv_file = os.path.join(
            self.dirs['cv_results'], 
            f"{self.output_prefix}_cv_results.csv"
        )
        cv_df.to_csv(cv_file, index=False, float_format='%.6f')
        
        # Calculate average metrics
        mean_metrics = {}
        for col in cv_df.columns:
            if col != 'fold':
                mean_metrics[col] = cv_df[col].mean()
        
        mean_metrics_file = os.path.join(
            self.dirs['results'], 
            f"{self.output_prefix}_cv_mean_metrics.csv"
        )
        pd.DataFrame([mean_metrics]).to_csv(mean_metrics_file, index=False, float_format='%.6f')
        
        self.logger.info("K-fold cross-validation complete")
        return cv_df
    
    def run_pipeline(self, train_file, test_file, max_evals=100):
        """Run full pipeline"""
        try:
            self.logger.info("Starting full pipeline...")
            
            # Execute each step
            self.load_data(train_file, test_file)
            self.standardize_data()
            best_params = self.hyperparameter_optimization(max_evals)
            self.find_best_model()
            cv_results = self.cross_validation()
            
            # Calculate total time
            total_time = time.time() - self.start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            
            self.logger.info(f"Pipeline complete! Total time: {hours}h {minutes}m {seconds}s")
            
            # Output final result summary
            if hasattr(self, 'best_test_metrics'):
                self.logger.info("Final result summary:")
                self.logger.info(f"Best params: n_estimators={self.best_params['n_estimators']}, max_depth={self.best_params['max_depth']}, learning_rate={self.best_params['learning_rate']:.6f}")
                self.logger.info(f"Independent test set results:")
                for metric, value in self.best_test_metrics.items():
                    self.logger.info(f"  {metric}: {value:.6f}")
            
            return {
                'best_params': self.best_params,
                'best_metrics': self.best_test_metrics if hasattr(self, 'best_test_metrics') else None,
                'cv_results': cv_results,
                'total_time': total_time
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline execution error: {str(e)}")
            if not hasattr(self, 'best_params'):
                self.logger.info("Creating model with default parameters")
                self.best_params = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'min_child_weight': 1,
                    'gamma': 0
                }
                self.find_best_model()
            
            return {
                'best_params': self.best_params if hasattr(self, 'best_params') else None,
                'best_metrics': self.best_test_metrics if hasattr(self, 'best_test_metrics') else None,
                'cv_results': None,
                'total_time': time.time() - self.start_time
            }

def predict_with_model(model_path, scaler_path, features):
    """Load a trained XGBoost model + scaler and predict"""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    X = scaler.transform(features)
    return model.predict(X)


def main():
    parser = argparse.ArgumentParser(description='XGBoost regression project (CPU)')
    parser.add_argument('--project_name', type=str, required=True,
                       help='Project name')
    parser.add_argument('--model_type', type=str, default='XGBoost',
                       help='Model type (currently only XGBoost supported)')
    parser.add_argument('--output_prefix', type=str, required=True,
                       help='Prefix for saved filenames')
    parser.add_argument('--k_fold', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--train_file', type=str, required=True,
                       help='Training set file path')
    parser.add_argument('--test_file', type=str, required=True,
                       help='Test set file path')
    parser.add_argument('--max_evals', type=int, default=100,
                       help='Number of hyperparameter optimization iterations')
    
    args = parser.parse_args()
    
    try:
        # Create and run regressor
        regressor = XGBoostRegressor(
            project_name=args.project_name,
            output_prefix=args.output_prefix,
            k_fold=args.k_fold
        )
        
        results = regressor.run_pipeline(
            train_file=args.train_file,
            test_file=args.test_file,
            max_evals=args.max_evals
        )
        
        if results['best_params']:
            total_time = results['total_time']
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            
            print(f"\nPipeline complete!")
            print(f"Total time: {hours}h {minutes}m {seconds}s")
            print(f"Best params: n_estimators={results['best_params']['n_estimators']}, max_depth={results['best_params']['max_depth']}, learning_rate={results['best_params']['learning_rate']:.6f}")
            
            if results['best_metrics']:
                print("Independent test set results:")
                for metric, value in results['best_metrics'].items():
                    print(f"  {metric}: {value:.6f}")
            
            if results['cv_results'] is not None:
                cv_means = results['cv_results'].mean(numeric_only=True)
                print("Cross-validation average results:")
                for col in results['cv_results'].columns:
                    if col.startswith('test_') and col != 'fold':
                        metric_name = col.replace('test_', '')
                        print(f"  {metric_name}: {cv_means[col]:.6f}")
        else:
            print("Pipeline complete, but there were issues. Check logs.")
        
    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    main()