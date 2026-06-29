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

# 设置日志
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
    """回归模型评价指标计算器"""
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, n_features=None, n_samples=None):
        """计算所有回归指标"""
        metrics = {}
        
        # 均方误差
        mse = mean_squared_error(y_true, y_pred)
        metrics['mse'] = mse
        
        # 均方根误差
        rmse = np.sqrt(mse)
        metrics['rmse'] = rmse
        
        # 平均绝对误差
        mae = mean_absolute_error(y_true, y_pred)
        metrics['mae'] = mae
        
        # R²
        r2 = r2_score(y_true, y_pred)
        metrics['r2'] = r2
        
        # 调整后的R²
        if n_features is not None and n_samples is not None and n_samples > n_features + 1:
            adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
            metrics['adjusted_r2'] = adjusted_r2
        else:
            metrics['adjusted_r2'] = None
        
        # 皮尔逊相关系数
        try:
            pearson_corr, pearson_p = pearsonr(y_true, y_pred)
            metrics['pearson'] = pearson_corr
            metrics['pearson_p'] = pearson_p
        except:
            metrics['pearson'] = None
            metrics['pearson_p'] = None
        
        # 斯皮尔曼相关系数
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
        """将指标字典转换为DataFrame"""
        df = pd.DataFrame([metrics_dict])
        return df

class VotingXGBRegressor(BaseEstimator, RegressorMixin):
    """
    基于多个XGBoost模型的投票回归器
    每个XGBoost模型可以有不同的参数
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
        
        # 模型组件
        self.models = []
        self.feature_names = None
        
    def fit(self, X, y):
        """训练多个XGBoost模型"""
        # 保存特征名称
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
            self.feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # 初始化模型列表
        self.models = []
        
        # 训练每个模型
        for i in range(self.n_models):
            logger.info(f"训练第 {i+1}/{self.n_models} 个XGBoost模型...")
            
            # 创建XGBoost模型
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
            
            # 训练模型
            model.fit(X, y)
            self.models.append(model)
        
        return self
    
    def predict(self, X):
        """使用投票策略进行预测"""
        if not self.models:
            raise ValueError("模型未训练")
        
        # 获取每个模型的预测
        predictions = []
        for i, model in enumerate(self.models):
            pred = model.predict(X)
            predictions.append(pred)
        
        # 转换为numpy数组
        pred_array = np.array(predictions)
        
        # 根据投票策略组合预测结果
        if self.voting_method == 'average':
            # 平均投票
            if self.weights is not None and len(self.weights) == len(self.models):
                # 加权平均
                weighted_sum = np.zeros_like(predictions[0])
                for i, weight in enumerate(self.weights):
                    weighted_sum += predictions[i] * weight
                return weighted_sum
            else:
                # 简单平均
                return np.mean(pred_array, axis=0)
                
        elif self.voting_method == 'median':
            # 中位数投票
            return np.median(pred_array, axis=0)
            
        elif self.voting_method == 'maximum':
            # 最大值投票
            return np.max(pred_array, axis=0)
            
        elif self.voting_method == 'minimum':
            # 最小值投票
            return np.min(pred_array, axis=0)
            
        else:
            # 默认使用平均投票
            return np.mean(pred_array, axis=0)
    
    def get_params(self, deep=True):
        """获取模型参数"""
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
        """设置模型参数"""
        for key, value in params.items():
            setattr(self, key, value)
        return self

class RegressionOptimizer:
    """
    回归模型优化器
    包含Hyperopt超参数优化、交叉验证等功能
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
        self.n_models = min(max(2, n_models), 9)  # 限制在2-9之间
        
        # 计算可用的CPU线程数，不超过80%
        self.total_cpus = os.cpu_count()
        self.available_cpus = max(1, int(self.total_cpus * self.max_threads_percent / 100))
        logger.info(f"系统CPU核心数: {self.total_cpus}")
        logger.info(f"使用线程数限制: {self.max_threads_percent}% = {self.available_cpus} 个核心")
        logger.info(f"使用XGBoost模型数量: {self.n_models} 个")
        
        # 执行时间记录
        self.start_time = None
        self.end_time = None
        self.total_time = None
        
        # 创建目录结构
        self._create_directories()
        
        # 数据
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_ids = None
        self.test_ids = None
        self.feature_scaler = None
        self.label_scaler = None
        
        # 模型和结果
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.trials = None
        
        # 日志文件
        self._setup_logging()
    
    def _create_directories(self):
        """创建目录结构"""
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
            logger.info(f"创建目录: {dir_path}")
    
    def _setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.dirs['logs'], f"{self.model_name}_optimization.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # 清除之前的处理器，避免重复
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(logging.StreamHandler(sys.stdout))
    
    def load_data(self):
        """加载数据"""
        logger.info("开始加载数据...")
        
        # 加载训练数据
        train_data = pd.read_csv(self.train_file)
        logger.info(f"训练集形状: {train_data.shape}")
        
        # 检查数据格式
        if train_data.shape[1] < 3:
            raise ValueError("训练集至少需要3列: ID, 标签, 特征")
        
        # 分离ID、标签和特征
        self.train_ids = train_data.iloc[:, 0].values
        self.y_train = train_data.iloc[:, 1].values
        self.X_train = train_data.iloc[:, 2:].values
        
        # 加载测试数据
        test_data = pd.read_csv(self.test_file)
        logger.info(f"测试集形状: {test_data.shape}")
        
        if test_data.shape[1] < 3:
            raise ValueError("测试集至少需要3列: ID, 标签, 特征")
        
        # 分离ID、标签和特征
        self.test_ids = test_data.iloc[:, 0].values
        self.y_test = test_data.iloc[:, 1].values
        self.X_test = test_data.iloc[:, 2:].values
        
        logger.info(f"训练集特征: {self.X_train.shape[1]} 个, 样本: {self.X_train.shape[0]} 个")
        logger.info(f"测试集特征: {self.X_test.shape[1]} 个, 样本: {self.X_test.shape[0]} 个")
        
        return self
    
    def preprocess_data(self, scale_features=True, scale_labels=False):
        """数据预处理和标准化"""
        logger.info("开始数据预处理...")
        
        if scale_features:
            # 特征标准化
            self.feature_scaler = StandardScaler()
            self.X_train = self.feature_scaler.fit_transform(self.X_train)
            self.X_test = self.feature_scaler.transform(self.X_test)
            
            # 保存特征标准化模型
            scaler_path = os.path.join(self.dirs['scaler'], 'feature_scaler.joblib')
            joblib.dump(self.feature_scaler, scaler_path)
            logger.info(f"特征标准化模型已保存到: {scaler_path}")
        
        if scale_labels:
            # 标签标准化（可选）
            self.label_scaler = StandardScaler()
            self.y_train = self.label_scaler.fit_transform(self.y_train.reshape(-1, 1)).ravel()
            self.y_test = self.label_scaler.transform(self.y_test.reshape(-1, 1)).ravel()
            
            # 保存标签标准化模型
            scaler_path = os.path.join(self.dirs['scaler'], 'label_scaler.joblib')
            joblib.dump(self.label_scaler, scaler_path)
            logger.info(f"标签标准化模型已保存到: {scaler_path}")
        
        logger.info("数据预处理完成")
        return self
    
    def _define_search_space(self):
        """定义Hyperopt搜索空间 - 包含多个XGBoost模型的参数"""
        space = {
            # 投票策略参数
            'voting_method': hp.choice('voting_method', ['average', 'median']),
        }
        
        # 为每个模型添加参数
        for i in range(self.n_models):
            # XGBoost模型参数
            space[f'n_estimators_{i}'] = scope.int(hp.quniform(f'n_estimators_{i}', 50, 2000, 10))
            space[f'max_depth_{i}'] = scope.int(hp.quniform(f'max_depth_{i}', 3, 12, 1))
            space[f'learning_rate_{i}'] = hp.loguniform(f'learning_rate_{i}', np.log(0.01), np.log(0.3))
            space[f'subsample_{i}'] = hp.uniform(f'subsample_{i}', 0.6, 1.0)
            space[f'colsample_bytree_{i}'] = hp.uniform(f'colsample_bytree_{i}', 0.6, 1.0)
            space[f'reg_alpha_{i}'] = hp.loguniform(f'reg_alpha_{i}', np.log(1e-5), np.log(10))
            space[f'reg_lambda_{i}'] = hp.loguniform(f'reg_lambda_{i}', np.log(1e-5), np.log(10))
            space[f'min_child_weight_{i}'] = hp.quniform(f'min_child_weight_{i}', 1, 10, 1)
            space[f'gamma_{i}'] = hp.uniform(f'gamma_{i}', 0, 5)
        
        # 添加权重参数（只添加前n-1个权重，最后一个权重通过计算得到）
        for i in range(self.n_models - 1):
            space[f'weight_{i}'] = hp.uniform(f'weight_{i}', 0, 1)
        
        return space
    
    def _objective(self, params):
        """Hyperopt目标函数"""
        try:
            start_time = time.time()
            
            # 清理整数参数
            int_params = []
            for i in range(self.n_models):
                int_params.extend([
                    f'n_estimators_{i}', f'max_depth_{i}', f'min_child_weight_{i}'
                ])
            
            for param in int_params:
                if param in params:
                    params[param] = int(params[param])
            
            # 提取模型参数
            n_estimators_list = [params.get(f'n_estimators_{i}', 100) for i in range(self.n_models)]
            max_depth_list = [params.get(f'max_depth_{i}', 6) for i in range(self.n_models)]
            learning_rate_list = [params.get(f'learning_rate_{i}', 0.1) for i in range(self.n_models)]
            subsample_list = [params.get(f'subsample_{i}', 0.8) for i in range(self.n_models)]
            colsample_bytree_list = [params.get(f'colsample_bytree_{i}', 0.8) for i in range(self.n_models)]
            reg_alpha_list = [params.get(f'reg_alpha_{i}', 0) for i in range(self.n_models)]
            reg_lambda_list = [params.get(f'reg_lambda_{i}', 1) for i in range(self.n_models)]
            min_child_weight_list = [params.get(f'min_child_weight_{i}', 1) for i in range(self.n_models)]
            gamma_list = [params.get(f'gamma_{i}', 0) for i in range(self.n_models)]
            
            # 计算权重
            weights = []
            if params.get('voting_method') == 'average':
                # 提取前n-1个权重
                weights = [params.get(f'weight_{i}', 1.0/self.n_models) for i in range(self.n_models - 1)]
                
                # 计算最后一个权重
                weight_sum = sum(weights)
                if weight_sum > 1.0:
                    # 如果权重和大于1，则进行归一化
                    weights = [w / weight_sum for w in weights]
                    last_weight = 0
                else:
                    last_weight = 1.0 - weight_sum
                
                weights.append(last_weight)
            
            # 创建投票模型
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
            
            # 训练模型
            model.fit(self.X_train, self.y_train)
            
            # 在测试集上预测
            y_pred = model.predict(self.X_test)
            
            # 计算所有评价指标
            n_samples = len(self.y_test)
            n_features = self.X_train.shape[1]
            metrics = RegressionMetrics.calculate_all_metrics(
                self.y_test, y_pred, n_features, n_samples
            )
            
            # Hyperopt需要最小化目标函数，所以我们返回负的R²（最大化R²相当于最小化负R²）
            loss = -metrics['r2']
            
            # 记录时间
            train_time = time.time() - start_time
            
            # 保存结果到CSV
            result = {
                'iteration': len(self.trials.trials) + 1,
                'loss': loss,
                'train_time': train_time,
                'params': params.copy(),
                'metrics': metrics
            }
            
            # 保存到文件
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
            logger.error(f"目标函数错误: {e}")
            return {
                'loss': float('inf'),
                'status': 'fail',
                'error': str(e)
            }
    
    def _save_hyperopt_result(self, result):
        """保存Hyperopt单次迭代结果"""
        # 准备数据行
        row = {
            'iteration': result['iteration'],
            'loss': result['loss'],
            'train_time': result['train_time']
        }
        
        # 添加超参数
        for key, value in result['params'].items():
            if isinstance(value, (int, float, str, bool)):
                row[f'param_{key}'] = value
            else:
                row[f'param_{key}'] = str(value)
        
        # 添加评价指标
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
        
        # 保存到CSV
        file_path = os.path.join(self.dirs['hyperopt'], 'hyperopt_results.csv')
        
        # 如果是第一次，创建文件并写入表头
        if not os.path.exists(file_path):
            df = pd.DataFrame([row])
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
        else:
            # 追加数据
            df = pd.read_csv(file_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
    
    def hyperopt_optimization(self, max_evals=100):
        """执行Hyperopt超参数优化"""
        logger.info(f"开始Hyperopt超参数优化，最大迭代次数: {max_evals}")
        
        # 定义搜索空间
        search_space = self._define_search_space()
        
        # 初始化Trials对象
        self.trials = Trials()
        
        # 创建自定义的随机状态对象
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
        
        # 创建兼容的随机状态
        rstate = CompatibleRandomState(self.random_state)
        
        # 运行Hyperopt优化
        best = fmin(
            fn=self._objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=self.trials,
            rstate=rstate,
            verbose=0
        )
        
        # 获取最佳参数
        self.best_params = space_eval(search_space, best)
        
        # 获取最佳得分
        successful_trials = [trial for trial in self.trials.trials if trial['result']['status'] == STATUS_OK]
        if not successful_trials:
            raise ValueError("所有试验都失败了，请检查日志")
        
        best_trial_idx = np.argmin([trial['result']['loss'] for trial in successful_trials])
        best_trial = successful_trials[best_trial_idx]
        self.best_score = -best_trial['result']['loss']  # 负的loss就是R²
        
        logger.info(f"Hyperopt优化完成，最佳R²: {self.best_score:.4f}")
        logger.info(f"最佳参数: {self.best_params}")
        
        # 保存优化结果摘要
        self._save_hyperopt_summary()
        
        return self.best_params, self.best_score
    
    def _save_hyperopt_summary(self):
        """保存Hyperopt优化结果摘要"""
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
        
        # 保存为JSON
        summary_file = os.path.join(self.dirs['hyperopt'], 'hyperopt_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Hyperopt摘要已保存到: {summary_file}")
        
        # 保存最佳参数到CSV
        best_params_dict = {}
        for key, value in self.best_params.items():
            if isinstance(value, (int, float, str, bool)):
                best_params_dict[key] = value
            else:
                best_params_dict[key] = str(value)
        
        best_params_df = pd.DataFrame([best_params_dict])
        best_params_file = os.path.join(self.dirs['hyperopt'], 'best_params.csv')
        best_params_df.to_csv(best_params_file, index=False, encoding='utf-8-sig')
        logger.info(f"最佳参数已保存到: {best_params_file}")
    
    def train_best_model(self):
        """使用最佳参数训练最终模型"""
        logger.info("使用最佳参数训练最终模型...")
        
        if self.best_params is None:
            raise ValueError("请先运行hyperopt_optimization获取最佳参数")
        
        # 提取模型参数
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
        
        # 计算权重
        weights = None
        if self.best_params.get('voting_method') == 'average':
            weights = []
            for i in range(self.n_models - 1):
                weights.append(self.best_params.get(f'weight_{i}', 1.0/self.n_models))
            
            # 计算最后一个权重
            weight_sum = sum(weights)
            if weight_sum > 1.0:
                weights = [w / weight_sum for w in weights]
                last_weight = 0
            else:
                last_weight = 1.0 - weight_sum
            
            weights.append(last_weight)
        
        # 创建最佳模型
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
        
        # 训练模型
        self.best_model.fit(self.X_train, self.y_train)
        
        # 在测试集上评估
        y_pred = self.best_model.predict(self.X_test)
        n_samples = len(self.y_test)
        n_features = self.X_train.shape[1]
        metrics = RegressionMetrics.calculate_all_metrics(
            self.y_test, y_pred, n_features, n_samples
        )
        
        # 保存模型
        model_file = os.path.join(self.dirs['best'], 'best_model.joblib')
        joblib.dump(self.best_model, model_file)
        logger.info(f"最佳模型已保存到: {model_file}")
        
        # 保存测试结果
        self._save_test_results(metrics)
        
        return self.best_model, metrics
    
    def _save_test_results(self, metrics):
        """保存测试集结果"""
        # 创建结果DataFrame
        results_df = pd.DataFrame([{
            'test_mse': metrics['mse'],
            'test_rmse': metrics['rmse'],
            'test_mae': metrics['mae'],
            'test_r2': metrics['r2'],
            'test_adjusted_r2': metrics['adjusted_r2'],
            'test_pearson': metrics['pearson'],
            'test_spearman': metrics['spearman']
        }])
        
        # 保存到CSV
        results_file = os.path.join(self.dirs['best'], 'best_model_test_results.csv')
        results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
        logger.info(f"最佳模型测试结果已保存到: {results_file}")
        
        # 同时保存到results目录
        results_file2 = os.path.join(self.dirs['results'], 'best_model_test_results.csv')
        results_df.to_csv(results_file2, index=False, encoding='utf-8-sig')
    
    def cross_validation(self):
        """对最佳模型进行K折交叉验证"""
        logger.info(f"开始{self.cv_folds}折交叉验证...")
        
        if self.best_params is None:
            raise ValueError("请先运行train_best_model训练最佳模型")
        
        # 提取模型参数
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
        
        # 计算权重
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
        
        # 准备交叉验证
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # 存储每折的结果
        fold_results = []
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train)):
            logger.info(f"处理第 {fold+1}/{self.cv_folds} 折")
            
            # 划分训练集和验证集
            X_train_fold = self.X_train[train_idx]
            y_train_fold = self.y_train[train_idx]
            X_val_fold = self.X_train[val_idx]
            y_val_fold = self.y_train[val_idx]
            
            # 创建模型（使用最佳参数）
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
            
            # 训练模型
            model.fit(X_train_fold, y_train_fold)
            
            # 在验证集上预测
            y_val_pred = model.predict(X_val_fold)
            
            # 计算评价指标
            n_samples = len(y_val_fold)
            n_features = X_train_fold.shape[1]
            metrics = RegressionMetrics.calculate_all_metrics(
                y_val_fold, y_val_pred, n_features, n_samples
            )
            
            # 保存每折结果
            fold_result = {
                'fold': fold + 1,
                'train_samples': len(train_idx),
                'val_samples': len(val_idx),
                **metrics
            }
            fold_results.append(fold_result)
            fold_metrics.append(metrics)
            
            # 保存每折的详细结果
            self._save_fold_results(fold, fold_result)
        
        # 计算平均指标
        avg_metrics = self._calculate_average_metrics(fold_metrics)
        
        # 保存交叉验证结果
        self._save_cv_results(fold_results, avg_metrics)
        
        return fold_results, avg_metrics
    
    def _save_fold_results(self, fold, fold_result):
        """保存单折交叉验证结果"""
        # 创建结果DataFrame
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
        
        # 保存到CSV
        fold_file = os.path.join(self.dirs['cv'], f'fold_{fold+1}_results.csv')
        results_df.to_csv(fold_file, index=False, encoding='utf-8-sig')
    
    def _calculate_average_metrics(self, fold_metrics):
        """计算平均指标"""
        avg_metrics = {}
        
        # 所有需要平均的指标
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
        """保存交叉验证结果"""
        # 保存所有折的详细结果
        all_folds_df = pd.DataFrame(fold_results)
        all_folds_file = os.path.join(self.dirs['cv'], 'all_folds_results.csv')
        all_folds_df.to_csv(all_folds_file, index=False, encoding='utf-8-sig')
        logger.info(f"所有折的详细结果已保存到: {all_folds_file}")
        
        # 保存平均结果
        avg_results_df = pd.DataFrame([avg_metrics])
        avg_results_file = os.path.join(self.dirs['cv'], 'cv_average_results.csv')
        avg_results_df.to_csv(avg_results_file, index=False, encoding='utf-8-sig')
        logger.info(f"交叉验证平均结果已保存到: {avg_results_file}")
        
        # 同时保存到results目录
        results_file = os.path.join(self.dirs['results'], 'cv_average_results.csv')
        avg_results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    
    def run_pipeline(self):
        """运行完整的优化流程"""
        # 记录开始时间
        self.start_time = time.time()
        logger.info("=" * 60)
        logger.info("开始回归模型优化流程")
        logger.info("=" * 60)
        
        try:
            # 1. 加载数据
            self.load_data()
            
            # 2. 数据预处理
            self.preprocess_data(scale_features=True, scale_labels=False)
            
            # 3. Hyperopt超参数优化
            logger.info("\n" + "=" * 60)
            logger.info("步骤1: Hyperopt超参数优化")
            logger.info("=" * 60)
            best_params, best_score = self.hyperopt_optimization(max_evals=self.n_iter)
            
            # 4. 训练最佳模型
            logger.info("\n" + "=" * 60)
            logger.info("步骤2: 训练最佳模型")
            logger.info("=" * 60)
            best_model, test_metrics = self.train_best_model()
            
            # 5. 交叉验证
            logger.info("\n" + "=" * 60)
            logger.info("步骤3: 交叉验证")
            logger.info("=" * 60)
            fold_results, avg_metrics = self.cross_validation()
            
            # 6. 生成最终报告
            self._generate_final_report(test_metrics, avg_metrics)
            
            # 记录结束时间
            self.end_time = time.time()
            self.total_time = self.end_time - self.start_time
            
            logger.info("\n" + "=" * 60)
            logger.info("优化流程完成!")
            logger.info("=" * 60)
            
            # 输出总结信息
            self._print_summary(best_params, test_metrics)
            
            return best_model, best_params, test_metrics, avg_metrics
            
        except Exception as e:
            # 记录结束时间（即使出错）
            self.end_time = time.time()
            self.total_time = self.end_time - self.start_time
            logger.error(f"优化流程出错: {e}")
            raise
    
    def _generate_final_report(self, test_metrics, avg_metrics):
        """生成最终报告"""
        report = {
            'project_name': self.project_name,
            'model_name': self.model_name,
            'n_models': self.n_models,
            'optimization_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_hyperparameters': self.best_params,
            'test_set_performance': test_metrics,
            'cross_validation_performance': avg_metrics
        }
        
        # 保存为JSON
        report_file = os.path.join(self.dirs['results'], 'final_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        logger.info(f"最终报告已保存到: {report_file}")
        
        # 创建摘要CSV
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
        logger.info(f"性能摘要已保存到: {summary_file}")
    
    def _print_summary(self, best_params, test_metrics):
        """输出总结信息"""
        print("\n" + "="*80)
        print("回归模型优化总结")
        print("="*80)
        
        # 1. 执行时间
        print(f"\n1. 执行时间统计:")
        print(f"   总耗时: {self.total_time:.2f} 秒 ({self.total_time/60:.2f} 分钟)")
        if self.start_time and self.end_time:
            start_str = datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')
            end_str = datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')
            print(f"   开始时间: {start_str}")
            print(f"   结束时间: {end_str}")
        
        # 2. 硬件信息
        print(f"\n2. 硬件使用情况:")
        print(f"   系统CPU核心数: {self.total_cpus}")
        print(f"   使用线程数: {self.available_cpus} (限制: {self.max_threads_percent}%)")
        print(f"   使用XGBoost模型数量: {self.n_models} 个")
        
        # 3. 项目信息
        print(f"\n3. 项目信息:")
        print(f"   项目名称: {self.project_name}")
        print(f"   模型类型: {self.model_name}")
        print(f"   Hyperopt迭代次数: {self.n_iter}")
        print(f"   交叉验证折数: {self.cv_folds}")
        
        # 4. 最佳超参数
        print(f"\n4. 最佳模型超参数:")
        
        # 计算权重
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
            print(f"\n   模型权重:")
            for i, weight in enumerate(weights):
                print(f"   模型{i+1}权重: {weight:.4f}")
        
        # 5. 独立测试结果
        print(f"\n5. 最佳模型独立测试结果:")
        print(f"   R²: {test_metrics['r2']:.6f}")
        print(f"   调整后R²: {test_metrics.get('adjusted_r2', 'N/A')}")
        print(f"   MSE: {test_metrics['mse']:.6f}")
        print(f"   RMSE: {test_metrics['rmse']:.6f}")
        print(f"   MAE: {test_metrics['mae']:.6f}")
        print(f"   皮尔逊相关系数: {test_metrics.get('pearson', 'N/A'):.6f}")
        print(f"   斯皮尔曼相关系数: {test_metrics.get('spearman', 'N/A'):.6f}")
        
        # 6. 文件保存位置
        print(f"\n6. 结果文件保存位置:")
        print(f"   最佳模型: {self.dirs['best']}/best_model.joblib")
        print(f"   特征标准化模型: {self.dirs['scaler']}/feature_scaler.joblib")
        print(f"   最佳模型测试结果: {self.dirs['best']}/best_model_test_results.csv")
        print(f"   Hyperopt优化结果: {self.dirs['hyperopt']}/hyperopt_results.csv")
        print(f"   交叉验证结果: {self.dirs['cv']}/all_folds_results.csv")
        print(f"   最终报告: {self.dirs['results']}/final_report.json")
        
        print("\n" + "="*80)
        
        # 同时记录到日志
        logger.info("\n" + "="*80)
        logger.info("回归模型优化总结")
        logger.info("="*80)
        logger.info(f"总耗时: {self.total_time:.2f} 秒 ({self.total_time/60:.2f} 分钟)")
        logger.info(f"最佳R²: {test_metrics['r2']:.6f}")
        logger.info(f"使用模型数量: {self.n_models} 个")
        logger.info(f"最佳模型已保存到: {self.dirs['best']}/best_model.joblib")
        logger.info("="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='回归模型优化管道')
    
    # 必需参数
    parser.add_argument('--project', type=str, required=True,
                       help='项目名称，用于创建目录')
    parser.add_argument('--train', type=str, required=True,
                       help='训练集CSV文件路径')
    parser.add_argument('--test', type=str, required=True,
                       help='测试集CSV文件路径')
    parser.add_argument('--prefix', type=str, required=True,
                       help='保存文件的前缀名称')
    
    # 可选参数
    parser.add_argument('--model', type=str, default='voting_xgb',
                       help='模型类型 (默认: voting_xgb)')
    parser.add_argument('--kfold', type=int, default=5,
                       help='K折交叉验证的折数 (默认: 5)')
    parser.add_argument('--n_iter', type=int, default=100,
                       help='Hyperopt优化迭代次数 (默认: 100)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--scale_labels', action='store_true',
                       help='是否对标签进行标准化 (默认: False)')
    parser.add_argument('--max_threads_percent', type=int, default=80,
                       help='最大线程使用百分比 (默认: 80%%)')
    parser.add_argument('--n_models', type=int, default=3,
                       help='XGBoost模型数量 (默认: 3, 范围: 2-9)')
    
    args = parser.parse_args()
    
    # 验证模型数量
    if args.n_models < 2 or args.n_models > 9:
        print(f"警告: 模型数量 {args.n_models} 不在有效范围内 (2-9)，将使用默认值 3")
        args.n_models = 3
    
    # 创建优化器实例
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
    
    # 运行完整流程
    try:
        best_model, best_params, test_metrics, avg_metrics = optimizer.run_pipeline()
        
    except Exception as e:
        logger.error(f"优化流程出错: {e}")
        raise


if __name__ == "__main__":
    # 示例运行命令:
    # python regression_optimizer.py --project my_regression_project --train train.csv --test test.csv --prefix my_model --model voting_xgb --kfold 5 --n_iter 50 --random_state 42 --max_threads_percent 80 --n_models 5
    
    main()
