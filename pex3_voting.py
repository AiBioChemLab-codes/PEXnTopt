import os
import sys
import json
import time
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from hyperopt.pyll import scope
from datetime import datetime
import joblib
import multiprocessing
import psutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('regression_optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RegressionMetrics:
    """Regression metrics calculator"""
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, n_features=None, n_samples=None):
        """Calculate all regression metrics"""
        metrics = {}
        
        # Mean squared error
        mse = mean_squared_error(y_true, y_pred)
        metrics['mse'] = mse
        
        # Root mean squared error
        rmse = np.sqrt(mse)
        metrics['rmse'] = rmse
        
        # Mean absolute error
        mae = mean_absolute_error(y_true, y_pred)
        metrics['mae'] = mae
        
        # R²
        r2 = r2_score(y_true, y_pred)
        metrics['r2'] = r2
        
        # Adjusted R²
        if n_features is not None and n_samples is not None and n_samples > n_features + 1:
            adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
            metrics['adjusted_r2'] = adjusted_r2
        else:
            metrics['adjusted_r2'] = None
        
        # Pearson correlation coefficient
        try:
            pearson_corr, pearson_p = pearsonr(y_true, y_pred)
            metrics['pearson'] = pearson_corr
            metrics['pearson_p'] = pearson_p
        except:
            metrics['pearson'] = None
            metrics['pearson_p'] = None
        
        # Spearman correlation coefficient
        try:
            spearman_corr, spearman_p = spearmanr(y_true, y_pred)
            metrics['spearman'] = spearman_corr
            metrics['spearman_p'] = spearman_p
        except:
            metrics['spearman'] = None
            metrics['spearman_p'] = None
        
        return metrics
    
    @staticmethod
    def metrics_to_dataframe(metrics_dict):
        """Convert metrics dict to DataFrame"""
        df = pd.DataFrame([metrics_dict])
        return df

class VotingXGBRegressor(BaseEstimator, RegressorMixin):
    """
    Voting regressor based on multiple XGBoost models
    Each XGBoost model can have different parameters
    """
    
    def __init__(self, 
                 n_models=3,
                 n_estimators_list=None,
                 max_depth_list=None,
                 learning_rate_list=None,
                 subsample_list=None,
                 colsample_bytree_list=None,
                 reg_alpha_list=None,
                 reg_lambda_list=None,
                 min_child_weight_list=None,
                 gamma_list=None,
                 voting_method='average',
                 weights=None,
                 random_state=42,
                 n_jobs=-1):
        
        self.n_models = n_models
        self.n_estimators_list = n_estimators_list or [100] * n_models
        self.max_depth_list = max_depth_list or [6] * n_models
        self.learning_rate_list = learning_rate_list or [0.1] * n_models
        self.subsample_list = subsample_list or [0.8] * n_models
        self.colsample_bytree_list = colsample_bytree_list or [0.8] * n_models
        self.reg_alpha_list = reg_alpha_list or [0] * n_models
        self.reg_lambda_list = reg_lambda_list or [1] * n_models
        self.min_child_weight_list = min_child_weight_list or [1] * n_models
        self.gamma_list = gamma_list or [0] * n_models
        self.voting_method = voting_method
        self.weights = weights
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Model components
        self.models = []
        self.feature_names = None
        
    def fit(self, X, y):
        """Train multiple XGBoost models"""
        # Save feature names
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
            self.feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Initialize model list
        self.models = []
        
        # Train each model
        for i in range(self.n_models):
            logger.info(f"Training XGBoost model {i+1}/{self.n_models}...")
            
            # Create XGBoost model
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators_list[i],
                max_depth=self.max_depth_list[i],
                learning_rate=self.learning_rate_list[i],
                subsample=self.subsample_list[i],
                colsample_bytree=self.colsample_bytree_list[i],
                reg_alpha=self.reg_alpha_list[i],
                reg_lambda=self.reg_lambda_list[i],
                min_child_weight=self.min_child_weight_list[i],
                gamma=self.gamma_list[i],
                random_state=self.random_state + i + 1,
                verbosity=0,
                n_jobs=self.n_jobs
            )
            
            # Train model
            model.fit(X, y)
            self.models.append(model)
        
        return self
    
    def predict(self, X):
        """Predict using voting strategy"""
        if not self.models:
            raise ValueError("Model not trained")
        
        # Get predictions from each model
        predictions = []
        for i, model in enumerate(self.models):
            pred = model.predict(X)
            predictions.append(pred)
        
        # Convert to numpy array
        pred_array = np.array(predictions)
        
        # Combine predictions based on voting strategy
        if self.voting_method == 'average':
            # Average voting
            if self.weights is not None and len(self.weights) == len(self.models):
                # Weighted average
                weighted_sum = np.zeros_like(predictions[0])
                for i, weight in enumerate(self.weights):
                    weighted_sum += predictions[i] * weight
                return weighted_sum
            else:
                # Simple average
                return np.mean(pred_array, axis=0)
                
        elif self.voting_method == 'median':
            # Median voting
            return np.median(pred_array, axis=0)
            
        elif self.voting_method == 'maximum':
            # Maximum voting
            return np.max(pred_array, axis=0)
            
        elif self.voting_method == 'minimum':
            # Minimum voting
            return np.min(pred_array, axis=0)
            
        else:
            # Default to average voting
            return np.mean(pred_array, axis=0)
    
    def get_params(self, deep=True):
        """Get model parameters"""
        params = {
            'n_models': self.n_models,
            'n_estimators_list': self.n_estimators_list,
            'max_depth_list': self.max_depth_list,
            'learning_rate_list': self.learning_rate_list,
            'subsample_list': self.subsample_list,
            'colsample_bytree_list': self.colsample_bytree_list,
            'reg_alpha_list': self.reg_alpha_list,
            'reg_lambda_list': self.reg_lambda_list,
            'min_child_weight_list': self.min_child_weight_list,
            'gamma_list': self.gamma_list,
            'voting_method': self.voting_method,
            'weights': self.weights,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }
        return params
    
    def set_params(self, **params):
        """Set model parameters"""
        for key, value in params.items():
            setattr(self, key, value)
        return self

class RegressionOptimizer:
    """
    Regression model optimizer
    Includes Hyperopt hyperparameter optimization, cross-validation, etc.
    """
    
    def __init__(self, project_name, train_file, test_file, 
                 model_name='voting_xgb', n_iter=100, cv_folds=5, 
                 random_state=42, max_threads_percent=80, n_models=3):
        
        self.project_name = project_name
        self.train_file = train_file
        self.test_file = test_file
        self.model_name = model_name
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.max_threads_percent = max_threads_percent
        self.n_models = min(max(2, n_models), 9)  # limit to 2-9
        
        # Calculate available CPU threads, limited to 80%
        self.total_cpus = os.cpu_count()
        self.available_cpus = max(1, int(self.total_cpus * self.max_threads_percent / 100))
        logger.info(f"System CPU cores: {self.total_cpus}")
        logger.info(f"Thread limit: {self.max_threads_percent}% = {self.available_cpus} cores")
        logger.info(f"XGBoost models: {self.n_models}")
        
        # Execution time tracking
        self.start_time = None
        self.end_time = None
        self.total_time = None
        
        # Create directory structure
        self._create_directories()
        
        # Data
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_ids = None
        self.test_ids = None
        self.feature_scaler = None
        self.label_scaler = None
        
        # Model and results
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.trials = None
        
        # Log file
        self._setup_logging()
    
    def _create_directories(self):
        """Create directory structure"""
        self.base_dir = self.project_name
        self.dirs = {
            'base': self.base_dir,
            'cv': os.path.join(self.base_dir, 'cv'),
            'results': os.path.join(self.base_dir, 'results'),
            'best': os.path.join(self.base_dir, 'best'),
            'scaler': os.path.join(self.base_dir, 'scaler'),
            'logs': os.path.join(self.base_dir, 'logs'),
            'hyperopt': os.path.join(self.base_dir, 'hyperopt')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Creating directory: {dir_path}")
    
    def _setup_logging(self):
        """Setup logging"""
        log_file = os.path.join(self.dirs['logs'], f"{self.model_name}_optimization.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Clear previous handlers to avoid duplicates
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(logging.StreamHandler(sys.stdout))
    
    def load_data(self):
        """Load data"""
        logger.info("Loading data...")
        
        # Load training data
        train_data = pd.read_csv(self.train_file)
        logger.info(f"Train set shape: {train_data.shape}")
        
        # Check data format
        if train_data.shape[1] < 3:
            raise ValueError("Training set needs at least 3 columns: ID, label, features")
        
        # Separate ID, label and features
        self.train_ids = train_data.iloc[:, 0].values
        self.y_train = train_data.iloc[:, 1].values
        self.X_train = train_data.iloc[:, 2:].values
        
        # Load test data
        test_data = pd.read_csv(self.test_file)
        logger.info(f"Test set shape: {test_data.shape}")
        
        if test_data.shape[1] < 3:
            raise ValueError("Test set needs at least 3 columns: ID, label, features")
        
        # Separate ID, label and features
        self.test_ids = test_data.iloc[:, 0].values
        self.y_test = test_data.iloc[:, 1].values
        self.X_test = test_data.iloc[:, 2:].values
        
        logger.info(f"Train features: {self.X_train.shape[1]}, samples: {self.X_train.shape[0]}")
        logger.info(f"Test features: {self.X_test.shape[1]}, samples: {self.X_test.shape[0]}")
        
        return self
    
    def preprocess_data(self, scale_features=True, scale_labels=False):
        """Preprocess and standardize data"""
        logger.info("Starting data preprocessing...")
        
        if scale_features:
            # Feature standardization
            self.feature_scaler = StandardScaler()
            self.X_train = self.feature_scaler.fit_transform(self.X_train)
            self.X_test = self.feature_scaler.transform(self.X_test)
            
            # Save feature scaler
            scaler_path = os.path.join(self.dirs['scaler'], 'feature_scaler.joblib')
            joblib.dump(self.feature_scaler, scaler_path)
            logger.info(f"Feature scaler saved to: {scaler_path}")
        
        if scale_labels:
            # Label standardization (optional)
            self.label_scaler = StandardScaler()
            self.y_train = self.label_scaler.fit_transform(self.y_train.reshape(-1, 1)).ravel()
            self.y_test = self.label_scaler.transform(self.y_test.reshape(-1, 1)).ravel()
            
            # Save label scaler
            scaler_path = os.path.join(self.dirs['scaler'], 'label_scaler.joblib')
            joblib.dump(self.label_scaler, scaler_path)
            logger.info(f"Label scaler saved to: {scaler_path}")
        
        logger.info("Preprocessing complete")
        return self
    
    def _define_search_space(self):
        """Define Hyperopt search space - parameters for multiple XGBoost models"""
        space = {
            # Voting strategy parameters
            'voting_method': hp.choice('voting_method', ['average', 'median']),
        }
        
        # Add parameters for each model
        for i in range(self.n_models):
            # XGBoost model parameters
            space[f'n_estimators_{i}'] = scope.int(hp.quniform(f'n_estimators_{i}', 50, 2000, 10))
            space[f'max_depth_{i}'] = scope.int(hp.quniform(f'max_depth_{i}', 3, 12, 1))
            space[f'learning_rate_{i}'] = hp.loguniform(f'learning_rate_{i}', np.log(0.01), np.log(0.3))
            space[f'subsample_{i}'] = hp.uniform(f'subsample_{i}', 0.6, 1.0)
            space[f'colsample_bytree_{i}'] = hp.uniform(f'colsample_bytree_{i}', 0.6, 1.0)
            space[f'reg_alpha_{i}'] = hp.loguniform(f'reg_alpha_{i}', np.log(1e-5), np.log(10))
            space[f'reg_lambda_{i}'] = hp.loguniform(f'reg_lambda_{i}', np.log(1e-5), np.log(10))
            space[f'min_child_weight_{i}'] = hp.quniform(f'min_child_weight_{i}', 1, 10, 1)
            space[f'gamma_{i}'] = hp.uniform(f'gamma_{i}', 0, 5)
        
        # Add weight parameters (only first n-1 weights, last weight is calculated)
        for i in range(self.n_models - 1):
            space[f'weight_{i}'] = hp.uniform(f'weight_{i}', 0, 1)
        
        return space
    
    def _objective(self, params):
        """Hyperopt objective function"""
        try:
            start_time = time.time()
            
            # Clean integer parameters
            int_params = []
            for i in range(self.n_models):
                int_params.extend([
                    f'n_estimators_{i}', f'max_depth_{i}', f'min_child_weight_{i}'
                ])
            
            for param in int_params:
                if param in params:
                    params[param] = int(params[param])
            
            # Extract model parameters
            n_estimators_list = [params.get(f'n_estimators_{i}', 100) for i in range(self.n_models)]
            max_depth_list = [params.get(f'max_depth_{i}', 6) for i in range(self.n_models)]
            learning_rate_list = [params.get(f'learning_rate_{i}', 0.1) for i in range(self.n_models)]
            subsample_list = [params.get(f'subsample_{i}', 0.8) for i in range(self.n_models)]
            colsample_bytree_list = [params.get(f'colsample_bytree_{i}', 0.8) for i in range(self.n_models)]
            reg_alpha_list = [params.get(f'reg_alpha_{i}', 0) for i in range(self.n_models)]
            reg_lambda_list = [params.get(f'reg_lambda_{i}', 1) for i in range(self.n_models)]
            min_child_weight_list = [params.get(f'min_child_weight_{i}', 1) for i in range(self.n_models)]
            gamma_list = [params.get(f'gamma_{i}', 0) for i in range(self.n_models)]
            
            # Calculate weights
            weights = []
            if params.get('voting_method') == 'average':
                # Extract first n-1 weights
                weights = [params.get(f'weight_{i}', 1.0/self.n_models) for i in range(self.n_models - 1)]
                
                # Calculate last weight
                weight_sum = sum(weights)
                if weight_sum > 1.0:
                    # If weight sum > 1, normalize
                    weights = [w / weight_sum for w in weights]
                    last_weight = 0
                else:
                    last_weight = 1.0 - weight_sum
                
                weights.append(last_weight)
            
            # Create voting model
            model = VotingXGBRegressor(
                n_models=self.n_models,
                n_estimators_list=n_estimators_list,
                max_depth_list=max_depth_list,
                learning_rate_list=learning_rate_list,
                subsample_list=subsample_list,
                colsample_bytree_list=colsample_bytree_list,
                reg_alpha_list=reg_alpha_list,
                reg_lambda_list=reg_lambda_list,
                min_child_weight_list=min_child_weight_list,
                gamma_list=gamma_list,
                voting_method=params.get('voting_method', 'average'),
                weights=weights if params.get('voting_method') == 'average' else None,
                random_state=self.random_state,
                n_jobs=self.available_cpus
            )
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Predict on test set
            y_pred = model.predict(self.X_test)
            
            # Calculate all evaluation metrics
            n_samples = len(self.y_test)
            n_features = self.X_train.shape[1]
            metrics = RegressionMetrics.calculate_all_metrics(
                self.y_test, y_pred, n_features, n_samples
            )
            
            # Hyperopt minimizes the objective, so return negative R² (maximizing R² = minimizing -R²)
            loss = -metrics['r2']
            
            # Record time
            train_time = time.time() - start_time
            
            # Save result to CSV
            result = {
                'iteration': len(self.trials.trials) + 1,
                'loss': loss,
                'train_time': train_time,
                'params': params.copy(),
                'metrics': metrics
            }
            
            # Save to file
            self._save_hyperopt_result(result)
            
            logger.info(f"Iteration {result['iteration']}: R² = {metrics['r2']:.4f}, "
                       f"MSE = {metrics['mse']:.4f}, Time = {train_time:.2f}s")
            
            return {
                'loss': loss,
                'status': STATUS_OK,
                'metrics': metrics,
                'params': params,
                'train_time': train_time
            }
            
        except Exception as e:
            logger.error(f"Objective function error: {e}")
            return {
                'loss': float('inf'),
                'status': 'fail',
                'error': str(e)
            }
    
    def _save_hyperopt_result(self, result):
        """Save Hyperopt single iteration result"""
        # Prepare data row
        row = {
            'iteration': result['iteration'],
            'loss': result['loss'],
            'train_time': result['train_time']
        }
        
        # Add hyperparameters
        for key, value in result['params'].items():
            if isinstance(value, (int, float, str, bool)):
                row[f'param_{key}'] = value
            else:
                row[f'param_{key}'] = str(value)
        
        # Add evaluation metrics
        metrics = result['metrics']
        row.update({
            'test_mse': metrics.get('mse', np.nan),
            'test_rmse': metrics.get('rmse', np.nan),
            'test_mae': metrics.get('mae', np.nan),
            'test_r2': metrics.get('r2', np.nan),
            'test_adjusted_r2': metrics.get('adjusted_r2', np.nan),
            'test_pearson': metrics.get('pearson', np.nan),
            'test_spearman': metrics.get('spearman', np.nan)
        })
        
        # Save to CSV
        file_path = os.path.join(self.dirs['hyperopt'], 'hyperopt_results.csv')
        
            # If first time, create file and write header
        if not os.path.exists(file_path):
            df = pd.DataFrame([row])
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
        else:
            # Append data
            df = pd.read_csv(file_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
    
    def hyperopt_optimization(self, max_evals=100):
        """Run Hyperopt hyperparameter optimization"""
        logger.info(f"Starting Hyperopt optimization, max iterations: {max_evals}")
        
        # Define search space
        search_space = self._define_search_space()
        
        # Initialize Trials object
        self.trials = Trials()
        
        # Create custom random state object
        class CompatibleRandomState:
            def __init__(self, seed=None):
                self.rng = np.random.default_rng(seed)
                self.seed = seed
            
            def randint(self, *args, **kwargs):
                if len(args) == 1:
                    return self.rng.integers(args[0], size=kwargs.get('size', None))
                elif len(args) == 2:
                    return self.rng.integers(args[0], args[1], size=kwargs.get('size', None))
                else:
                    return self.rng.integers(*args, **kwargs)
            
            def integers(self, *args, **kwargs):
                return self.rng.integers(*args, **kwargs)
        
        # Create compatible random state
        rstate = CompatibleRandomState(self.random_state)
        
        # Run Hyperopt optimization
        best = fmin(
            fn=self._objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=self.trials,
            rstate=rstate,
            verbose=0
        )
        
        # Get best parameters
        self.best_params = space_eval(search_space, best)
        
        # Get best score
        successful_trials = [trial for trial in self.trials.trials if trial['result']['status'] == STATUS_OK]
        if not successful_trials:
            raise ValueError("All trials failed, check logs")
        
        best_trial_idx = np.argmin([trial['result']['loss'] for trial in successful_trials])
        best_trial = successful_trials[best_trial_idx]
        self.best_score = -best_trial['result']['loss']  # negative loss is R²
        
        logger.info(f"Hyperopt complete, best R²: {self.best_score:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        # Save optimization summary
        self._save_hyperopt_summary()
        
        return self.best_params, self.best_score
    
    def _save_hyperopt_summary(self):
        """Save Hyperopt optimization summary"""
        summary = {
            'project_name': self.project_name,
            'model_name': self.model_name,
            'n_models': self.n_models,
            'n_iterations': self.n_iter,
            'cv_folds': self.cv_folds,
            'random_state': self.random_state,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'optimization_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save as JSON
        summary_file = os.path.join(self.dirs['hyperopt'], 'hyperopt_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Hyperopt summary saved to: {summary_file}")
        
        # Save best parameters to CSV
        best_params_dict = {}
        for key, value in self.best_params.items():
            if isinstance(value, (int, float, str, bool)):
                best_params_dict[key] = value
            else:
                best_params_dict[key] = str(value)
        
        best_params_df = pd.DataFrame([best_params_dict])
        best_params_file = os.path.join(self.dirs['hyperopt'], 'best_params.csv')
        best_params_df.to_csv(best_params_file, index=False, encoding='utf-8-sig')
        logger.info(f"Best params saved to: {best_params_file}")
    
    def train_best_model(self):
        """Train final model with best parameters"""
        logger.info("Training final model with best parameters...")
        
        if self.best_params is None:
            raise ValueError("Run hyperopt_optimization first to get best parameters")
        
        # Extract model parameters
        n_estimators_list = []
        max_depth_list = []
        learning_rate_list = []
        subsample_list = []
        colsample_bytree_list = []
        reg_alpha_list = []
        reg_lambda_list = []
        min_child_weight_list = []
        gamma_list = []
        
        for i in range(self.n_models):
            n_estimators_list.append(int(self.best_params.get(f'n_estimators_{i}', 100)))
            max_depth_list.append(int(self.best_params.get(f'max_depth_{i}', 6)))
            learning_rate_list.append(self.best_params.get(f'learning_rate_{i}', 0.1))
            subsample_list.append(self.best_params.get(f'subsample_{i}', 0.8))
            colsample_bytree_list.append(self.best_params.get(f'colsample_bytree_{i}', 0.8))
            reg_alpha_list.append(self.best_params.get(f'reg_alpha_{i}', 0))
            reg_lambda_list.append(self.best_params.get(f'reg_lambda_{i}', 1))
            min_child_weight_list.append(int(self.best_params.get(f'min_child_weight_{i}', 1)))
            gamma_list.append(self.best_params.get(f'gamma_{i}', 0))
        
        # Calculate weights
        weights = None
        if self.best_params.get('voting_method') == 'average':
            weights = []
            for i in range(self.n_models - 1):
                weights.append(self.best_params.get(f'weight_{i}', 1.0/self.n_models))
            
            # Calculate last weight
            weight_sum = sum(weights)
            if weight_sum > 1.0:
                weights = [w / weight_sum for w in weights]
                last_weight = 0
            else:
                last_weight = 1.0 - weight_sum
            
            weights.append(last_weight)
        
        # Create best model
        self.best_model = VotingXGBRegressor(
            n_models=self.n_models,
            n_estimators_list=n_estimators_list,
            max_depth_list=max_depth_list,
            learning_rate_list=learning_rate_list,
            subsample_list=subsample_list,
            colsample_bytree_list=colsample_bytree_list,
            reg_alpha_list=reg_alpha_list,
            reg_lambda_list=reg_lambda_list,
            min_child_weight_list=min_child_weight_list,
            gamma_list=gamma_list,
            voting_method=self.best_params.get('voting_method', 'average'),
            weights=weights,
            random_state=self.random_state,
            n_jobs=self.available_cpus
        )
        
        # Train model
        self.best_model.fit(self.X_train, self.y_train)
        
        # Evaluate on test set
        y_pred = self.best_model.predict(self.X_test)
        n_samples = len(self.y_test)
        n_features = self.X_train.shape[1]
        metrics = RegressionMetrics.calculate_all_metrics(
            self.y_test, y_pred, n_features, n_samples
        )
        
        # Save model
        model_file = os.path.join(self.dirs['best'], 'best_model.joblib')
        joblib.dump(self.best_model, model_file)
        logger.info(f"Best model saved to: {model_file}")
        
        # Save test results
        self._save_test_results(metrics)
        
        return self.best_model, metrics
    
    def _save_test_results(self, metrics):
        """Save test set results"""
        # Create results DataFrame
        results_df = pd.DataFrame([{
            'test_mse': metrics['mse'],
            'test_rmse': metrics['rmse'],
            'test_mae': metrics['mae'],
            'test_r2': metrics['r2'],
            'test_adjusted_r2': metrics['adjusted_r2'],
            'test_pearson': metrics['pearson'],
            'test_spearman': metrics['spearman']
        }])
        
        # Save to CSV
        results_file = os.path.join(self.dirs['best'], 'best_model_test_results.csv')
        results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
        logger.info(f"Best model test results saved to: {results_file}")
        
        # Also save to results directory
        results_file2 = os.path.join(self.dirs['results'], 'best_model_test_results.csv')
        results_df.to_csv(results_file2, index=False, encoding='utf-8-sig')
    
    def cross_validation(self):
        """K-fold cross-validation on the best model"""
        logger.info(f"Starting {self.cv_folds}-fold cross-validation...")
        
        if self.best_params is None:
            raise ValueError("Run train_best_model first")
        
        # Extract model parameters
        n_estimators_list = []
        max_depth_list = []
        learning_rate_list = []
        subsample_list = []
        colsample_bytree_list = []
        reg_alpha_list = []
        reg_lambda_list = []
        min_child_weight_list = []
        gamma_list = []
        
        for i in range(self.n_models):
            n_estimators_list.append(int(self.best_params.get(f'n_estimators_{i}', 100)))
            max_depth_list.append(int(self.best_params.get(f'max_depth_{i}', 6)))
            learning_rate_list.append(self.best_params.get(f'learning_rate_{i}', 0.1))
            subsample_list.append(self.best_params.get(f'subsample_{i}', 0.8))
            colsample_bytree_list.append(self.best_params.get(f'colsample_bytree_{i}', 0.8))
            reg_alpha_list.append(self.best_params.get(f'reg_alpha_{i}', 0))
            reg_lambda_list.append(self.best_params.get(f'reg_lambda_{i}', 1))
            min_child_weight_list.append(int(self.best_params.get(f'min_child_weight_{i}', 1)))
            gamma_list.append(self.best_params.get(f'gamma_{i}', 0))
        
        # Calculate weights
        weights = None
        if self.best_params.get('voting_method') == 'average':
            weights = []
            for i in range(self.n_models - 1):
                weights.append(self.best_params.get(f'weight_{i}', 1.0/self.n_models))
            
            weight_sum = sum(weights)
            if weight_sum > 1.0:
                weights = [w / weight_sum for w in weights]
                last_weight = 0
            else:
                last_weight = 1.0 - weight_sum
            
            weights.append(last_weight)
        
        # Prepare cross-validation
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Store results for each fold
        fold_results = []
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train)):
            logger.info(f"Processing fold {fold+1}/{self.cv_folds}")
            
            # Split into training and validation sets
            X_train_fold = self.X_train[train_idx]
            y_train_fold = self.y_train[train_idx]
            X_val_fold = self.X_train[val_idx]
            y_val_fold = self.y_train[val_idx]
            
            # Create model (with best params)
            model = VotingXGBRegressor(
                n_models=self.n_models,
                n_estimators_list=n_estimators_list,
                max_depth_list=max_depth_list,
                learning_rate_list=learning_rate_list,
                subsample_list=subsample_list,
                colsample_bytree_list=colsample_bytree_list,
                reg_alpha_list=reg_alpha_list,
                reg_lambda_list=reg_lambda_list,
                min_child_weight_list=min_child_weight_list,
                gamma_list=gamma_list,
                voting_method=self.best_params.get('voting_method', 'average'),
                weights=weights,
                random_state=self.random_state + fold,
                n_jobs=self.available_cpus
            )
            
            # Train model
            model.fit(X_train_fold, y_train_fold)
            
            # Predict on validation set
            y_val_pred = model.predict(X_val_fold)
            
            # Calculate evaluation metrics
            n_samples = len(y_val_fold)
            n_features = X_train_fold.shape[1]
            metrics = RegressionMetrics.calculate_all_metrics(
                y_val_fold, y_val_pred, n_features, n_samples
            )
            
            # Save per-fold results
            fold_result = {
                'fold': fold + 1,
                'train_samples': len(train_idx),
                'val_samples': len(val_idx),
                **metrics
            }
            fold_results.append(fold_result)
            fold_metrics.append(metrics)
            
            # Save detailed per-fold results
            self._save_fold_results(fold, fold_result)
        
        # Calculate average metrics
        avg_metrics = self._calculate_average_metrics(fold_metrics)
        
        # Save cross-validation results
        self._save_cv_results(fold_results, avg_metrics)
        
        return fold_results, avg_metrics
    
    def _save_fold_results(self, fold, fold_result):
        """Save single fold cross-validation result"""
        # Create results DataFrame
        results_df = pd.DataFrame([{
            'fold': fold_result['fold'],
            'train_samples': fold_result['train_samples'],
            'val_samples': fold_result['val_samples'],
            'mse': fold_result['mse'],
            'rmse': fold_result['rmse'],
            'mae': fold_result['mae'],
            'r2': fold_result['r2'],
            'adjusted_r2': fold_result['adjusted_r2'],
            'pearson': fold_result['pearson'],
            'spearman': fold_result['spearman']
        }])
        
        # Save to CSV
        fold_file = os.path.join(self.dirs['cv'], f'fold_{fold+1}_results.csv')
        results_df.to_csv(fold_file, index=False, encoding='utf-8-sig')
    
    def _calculate_average_metrics(self, fold_metrics):
        """Calculate average metrics"""
        avg_metrics = {}
        
        # All metrics to average
        metric_keys = ['mse', 'rmse', 'mae', 'r2', 'adjusted_r2', 'pearson', 'spearman']
        
        for key in metric_keys:
            values = [metrics[key] for metrics in fold_metrics if metrics[key] is not None]
            if values:
                avg_metrics[f'avg_{key}'] = np.mean(values)
                avg_metrics[f'std_{key}'] = np.std(values)
            else:
                avg_metrics[f'avg_{key}'] = None
                avg_metrics[f'std_{key}'] = None
        
        return avg_metrics
    
    def _save_cv_results(self, fold_results, avg_metrics):
        """Save cross-validation results"""
        # Save detailed results for all folds
        all_folds_df = pd.DataFrame(fold_results)
        all_folds_file = os.path.join(self.dirs['cv'], 'all_folds_results.csv')
        all_folds_df.to_csv(all_folds_file, index=False, encoding='utf-8-sig')
        logger.info(f"All fold results saved to: {all_folds_file}")
        
        # Save average results
        avg_results_df = pd.DataFrame([avg_metrics])
        avg_results_file = os.path.join(self.dirs['cv'], 'cv_average_results.csv')
        avg_results_df.to_csv(avg_results_file, index=False, encoding='utf-8-sig')
        logger.info(f"CV average results saved to: {avg_results_file}")
        
        # Also save to results directory
        results_file = os.path.join(self.dirs['results'], 'cv_average_results.csv')
        avg_results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    
    def run_pipeline(self):
        """Run the full optimization pipeline"""
        # Record start time
        self.start_time = time.time()
        logger.info("=" * 60)
        logger.info("Starting regression model optimization pipeline")
        logger.info("=" * 60)
        
        try:
            # 1. Load data
            self.load_data()
            
            # 2. Preprocess data
            self.preprocess_data(scale_features=True, scale_labels=False)
            
            # 3. Hyperopt optimization
            logger.info("\n" + "=" * 60)
            logger.info("Step 1: Hyperopt optimization")
            logger.info("=" * 60)
            best_params, best_score = self.hyperopt_optimization(max_evals=self.n_iter)
            
            # 4. Train best model
            logger.info("\n" + "=" * 60)
            logger.info("Step 2: Training best model")
            logger.info("=" * 60)
            best_model, test_metrics = self.train_best_model()
            
            # 5. Cross-validation
            logger.info("\n" + "=" * 60)
            logger.info("Step 3: Cross-validation")
            logger.info("=" * 60)
            fold_results, avg_metrics = self.cross_validation()
            
            # 6. Generate final report
            self._generate_final_report(test_metrics, avg_metrics)
            
            # Record end time
            self.end_time = time.time()
            self.total_time = self.end_time - self.start_time
            
            logger.info("\n" + "=" * 60)
            logger.info("Optimization pipeline complete!")
            logger.info("=" * 60)
            
            # Output summary
            self._print_summary(best_params, test_metrics)
            
            return best_model, best_params, test_metrics, avg_metrics
            
        except Exception as e:
            # Record end time (even on error)
            self.end_time = time.time()
            self.total_time = self.end_time - self.start_time
            logger.error(f"Optimization pipeline error: {e}")
            raise
    
    def _generate_final_report(self, test_metrics, avg_metrics):
        """Generate final report"""
        report = {
            'project_name': self.project_name,
            'model_name': self.model_name,
            'n_models': self.n_models,
            'optimization_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_hyperparameters': self.best_params,
            'test_set_performance': test_metrics,
            'cross_validation_performance': avg_metrics
        }
        
        # Save as JSON
        report_file = os.path.join(self.dirs['results'], 'final_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Final report saved to: {report_file}")
        
        # Create summary CSV
        summary_data = {
            'Metric': ['MSE', 'RMSE', 'MAE', 'R²', 'Adjusted R²', 'Pearson', 'Spearman'],
            'Test_Set': [
                test_metrics['mse'],
                test_metrics['rmse'],
                test_metrics['mae'],
                test_metrics['r2'],
                test_metrics['adjusted_r2'],
                test_metrics['pearson'],
                test_metrics['spearman']
            ],
            'CV_Mean': [
                avg_metrics.get('avg_mse', np.nan),
                avg_metrics.get('avg_rmse', np.nan),
                avg_metrics.get('avg_mae', np.nan),
                avg_metrics.get('avg_r2', np.nan),
                avg_metrics.get('avg_adjusted_r2', np.nan),
                avg_metrics.get('avg_pearson', np.nan),
                avg_metrics.get('avg_spearman', np.nan)
            ],
            'CV_Std': [
                avg_metrics.get('std_mse', np.nan),
                avg_metrics.get('std_rmse', np.nan),
                avg_metrics.get('std_mae', np.nan),
                avg_metrics.get('std_r2', np.nan),
                avg_metrics.get('std_adjusted_r2', np.nan),
                avg_metrics.get('std_pearson', np.nan),
                avg_metrics.get('std_spearman', np.nan)
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.dirs['results'], 'performance_summary.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        logger.info(f"Performance summary saved to: {summary_file}")
    
    def _print_summary(self, best_params, test_metrics):
        """Output summary information"""
        print("\n" + "="*80)
        print("Regression Model Optimization Summary")
        print("="*80)
        
        # 1. Execution time
        print(f"\n1. Execution Time:")
        print(f"   Total: {self.total_time:.2f} s ({self.total_time/60:.2f} min)")
        if self.start_time and self.end_time:
            start_str = datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')
            end_str = datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')
            print(f"   Start: {start_str}")
            print(f"   End: {end_str}")
        
        # 2. Hardware info
        print(f"\n2. Hardware:")
        print(f"   System CPU cores: {self.total_cpus}")
        print(f"   Threads used: {self.available_cpus} (limit: {self.max_threads_percent}%)")
        print(f"   XGBoost models: {self.n_models}")
        
        # 3. Project info
        print(f"\n3. Project Info:")
        print(f"   Project name: {self.project_name}")
        print(f"   Model type: {self.model_name}")
        print(f"   Hyperopt iterations: {self.n_iter}")
        print(f"   CV folds: {self.cv_folds}")
        
        # 4. Best hyperparameters
        print(f"\n4. Best Hyperparameters:")
        
        # Calculate weights
        weights = []
        if best_params.get('voting_method') == 'average':
            for i in range(self.n_models - 1):
                weights.append(best_params.get(f'weight_{i}', 1.0/self.n_models))
            
            weight_sum = sum(weights)
            if weight_sum > 1.0:
                weights = [w / weight_sum for w in weights]
                last_weight = 0
            else:
                last_weight = 1.0 - weight_sum
            
            weights.append(last_weight)
        
        for key, value in best_params.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.6f}")
            else:
                print(f"   {key}: {value}")
        
        if weights:
            print(f"\n   Model weights:")
            for i, weight in enumerate(weights):
                print(f"   Model {i+1} weight: {weight:.4f}")
        
        # 5. Independent test results
        print(f"\n5. Best Model Test Results:")
        print(f"   R²: {test_metrics['r2']:.6f}")
        print(f"   Adjusted R²: {test_metrics.get('adjusted_r2', 'N/A')}")
        print(f"   MSE: {test_metrics['mse']:.6f}")
        print(f"   RMSE: {test_metrics['rmse']:.6f}")
        print(f"   MAE: {test_metrics['mae']:.6f}")
        print(f"   Pearson: {test_metrics.get('pearson', 'N/A'):.6f}")
        print(f"   Spearman: {test_metrics.get('spearman', 'N/A'):.6f}")
        
        # 6. File save locations
        print(f"\n6. Saved Files:")
        print(f"   Best model: {self.dirs['best']}/best_model.joblib")
        print(f"   Feature scaler: {self.dirs['scaler']}/feature_scaler.joblib")
        print(f"   Test results: {self.dirs['best']}/best_model_test_results.csv")
        print(f"   Hyperopt results: {self.dirs['hyperopt']}/hyperopt_results.csv")
        print(f"   CV results: {self.dirs['cv']}/all_folds_results.csv")
        print(f"   Final report: {self.dirs['results']}/final_report.json")
        
        print("\n" + "="*80)
        
        # Also log to file
        logger.info("\n" + "="*80)
        logger.info("Regression Model Optimization Summary")
        logger.info("="*80)
        logger.info(f"Total time: {self.total_time:.2f} s ({self.total_time/60:.2f} min)")
        logger.info(f"Best R²: {test_metrics['r2']:.6f}")
        logger.info(f"Models used: {self.n_models}")
        logger.info(f"Best model saved to: {self.dirs['best']}/best_model.joblib")
        logger.info("="*80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Regression model optimization pipeline')
    
    # Required arguments
    parser.add_argument('--project', type=str, required=True,
                       help='Project name for directory structure')
    parser.add_argument('--train', type=str, required=True,
                       help='Training set CSV file path')
    parser.add_argument('--test', type=str, required=True,
                       help='Test set CSV file path')
    parser.add_argument('--prefix', type=str, required=True,
                       help='Prefix for output filenames')
    
    # Optional arguments
    parser.add_argument('--model', type=str, default='voting_xgb',
                       help='Model type (default: voting_xgb)')
    parser.add_argument('--kfold', type=int, default=5,
                       help='K-fold cross-validation folds (default: 5)')
    parser.add_argument('--n_iter', type=int, default=100,
                       help='Hyperopt optimization iterations (default: 100)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--scale_labels', action='store_true',
                       help='Standardize labels (default: False)')
    parser.add_argument('--max_threads_percent', type=int, default=80,
                       help='Max thread usage percentage (default: 80%%)')
    parser.add_argument('--n_models', type=int, default=3,
                       help='Number of XGBoost models (default: 3, range: 2-9)')
    
    args = parser.parse_args()
    
        # Validate number of models
    if args.n_models < 2 or args.n_models > 9:
        print(f"Warning: model count {args.n_models} out of range (2-9), using default 3")
        args.n_models = 3
    
    # Create optimizer instance
    optimizer = RegressionOptimizer(
        project_name=args.project,
        train_file=args.train,
        test_file=args.test,
        model_name=args.model,
        n_iter=args.n_iter,
        cv_folds=args.kfold,
        random_state=args.random_state,
        max_threads_percent=args.max_threads_percent,
        n_models=args.n_models
    )
    
        # Run full pipeline
    try:
        best_model, best_params, test_metrics, avg_metrics = optimizer.run_pipeline()
        
    except Exception as e:
        logger.error(f"Optimization pipeline error: {e}")
        raise


if __name__ == "__main__":
    # Example run command:
    # python regression_optimizer.py --project my_regression_project --train train.csv --test test.csv --prefix my_model --model voting_xgb --kfold 5 --n_iter 50 --random_state 42 --max_threads_percent 80 --n_models 5
    
    main()
