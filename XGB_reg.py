import argparse
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
import xgboost as xgb
from sklearn.preprocessing import StandardScaler  # 替换cuml
from sklearn.metrics import r2_score  # 替换cuml
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import joblib
import warnings
warnings.filterwarnings('ignore')

class XGBoostRegressor:  # 类名移除GPU标识，更贴合实际
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
        """创建项目目录结构"""
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
        """设置日志记录"""
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
        self.logger.info(f"项目初始化完成: {self.project_name}")
        self.logger.info(f"XGBoost版本: {xgb.__version__}")
    
    def load_data(self, train_file, test_file):
        """加载训练集和测试集数据"""
        self.logger.info("开始加载数据...")
        
        try:
            # 读取训练集
            train_df = pd.read_csv(train_file)
            self.logger.info(f"训练集大小: {train_df.shape}")
            
            # 读取测试集
            test_df = pd.read_csv(test_file)
            self.logger.info(f"测试集大小: {test_df.shape}")
            
            # 提取特征和回归值
            self.X_train = train_df.iloc[:, 2:].values.astype(np.float32)
            self.y_train = train_df.iloc[:, 1].values.astype(np.float32)
            self.X_test = test_df.iloc[:, 2:].values.astype(np.float32)
            self.y_test = test_df.iloc[:, 1].values.astype(np.float32)
            
            # 记录数据基本信息
            self.n_samples_train = len(self.y_train)
            self.n_features = self.X_train.shape[1]
            
            # 检查数据质量
            self._check_data_quality()
            
            self.logger.info("数据加载完成")
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            raise
    
    def _check_data_quality(self):
        """检查数据质量"""
        self.logger.info("检查数据质量...")
        
        # 检查NaN值
        if np.isnan(self.X_train).any() or np.isnan(self.X_test).any():
            self.logger.warning("数据中存在NaN值，进行填充处理")
            self.X_train = np.nan_to_num(self.X_train)
            self.X_test = np.nan_to_num(self.X_test)
        
        # 检查无穷值
        if np.isinf(self.X_train).any() or np.isinf(self.X_test).any():
            self.logger.warning("数据中存在无穷值，进行替换处理")
            self.X_train = np.where(np.isinf(self.X_train), 0, self.X_train)
            self.X_test = np.where(np.isinf(self.X_test), 0, self.X_test)
        
        # 检查回归值范围
        self.logger.info(f"训练集回归值范围: [{self.y_train.min():.6f}, {self.y_train.max():.6f}]")
        self.logger.info(f"测试集回归值范围: [{self.y_test.min():.6f}, {self.y_test.max():.6f}]")
        
        # 检查特征尺度差异
        feature_ranges = np.ptp(self.X_train, axis=0)
        if np.max(feature_ranges) / np.min(feature_ranges[feature_ranges > 0]) > 1000:
            self.logger.warning("特征尺度差异较大，标准化处理很重要")
    
    def standardize_data(self):
        """数据标准化处理"""
        self.logger.info("开始数据标准化...")
        
        try:
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            # 保存标准化模型
            scaler_file = os.path.join(
                self.dirs['scalers'], 
                f"{self.output_prefix}_scaler.joblib"
            )
            joblib.dump(self.scaler, scaler_file)
            self.logger.info(f"标准化模型已保存: {scaler_file}")
        except Exception as e:
            self.logger.error(f"数据标准化失败: {str(e)}")
            raise
    
    def calculate_adjusted_r2(self, r2, n_samples, n_features):
        """计算校正决定系数"""
        if n_samples <= n_features + 1:
            return 0.0
        return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    
    def calculate_metrics(self, y_true, y_pred, dataset_type='test'):
        """计算所有回归评估指标"""
        try:
            # 基础回归指标
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # 计算样本数和特征数用于adjusted R2
            n_samples = len(y_true)
            n_features = self.n_features
            
            # 校正决定系数
            adj_r2 = self.calculate_adjusted_r2(r2, n_samples, n_features)
            
            # 相关系数
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
            
            # 格式化数值为6位有效数字
            formatted_metrics = {}
            for key, value in metrics.items():
                if np.isfinite(value):
                    formatted_metrics[key] = float(f"{value:.6g}")
                else:
                    formatted_metrics[key] = 0.0
            
            return formatted_metrics
        except Exception as e:
            self.logger.error(f"计算{dataset_type}指标时出错: {str(e)}")
            return {key: 0.0 for key in [
                'mse', 'rmse', 'mae', 'r2', 'adjusted_r2', 'pearson_corr', 'spearman_corr'
            ]}
    
    def safe_xgb_fit(self, model, X, y, eval_set=None):
        """安全的XGBoost训练方法"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if eval_set:
                    model.fit(X, y, eval_set=eval_set, verbose=False)
                else:
                    model.fit(X, y, verbose=False)
                return model
            except Exception as e:
                self.logger.warning(f"XGBoost训练尝试 {attempt + 1}/{max_attempts} 失败: {str(e)}")
                if attempt < max_attempts - 1:
                    time.sleep(1)
                else:
                    raise
    
    def objective_function(self, params):
        """Hyperopt目标函数"""
        try:
            start_time = time.time()
            
            # 设置XGBoost参数（改用CPU版hist算法）
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
                'tree_method': 'hist',  # 关键修改：CPU版直方图算法
                'random_state': 42
            }
            
            self.logger.info(f"训练模型，参数: n_estimators={xgb_params['n_estimators']}, max_depth={xgb_params['max_depth']}, learning_rate={xgb_params['learning_rate']:.6f}")
            
            # 训练XGBoost模型
            xgb_model = xgb.XGBRegressor(**xgb_params)
            xgb_model = self.safe_xgb_fit(xgb_model, self.X_train_scaled, self.y_train)
            
            # 预测训练集和测试集
            y_train_pred = xgb_model.predict(self.X_train_scaled)
            y_test_pred = xgb_model.predict(self.X_test_scaled)
            
            # 计算指标
            train_metrics = self.calculate_metrics(self.y_train, y_train_pred, 'train')
            test_metrics = self.calculate_metrics(self.y_test, y_test_pred, 'test')
            
            # 主要优化目标：RMSE（越小越好），同时考虑R2和adjusted R2（越大越好）
            # 使用复合评分：RMSE的倒数 + R2 + adjusted R2
            rmse_weight = 1.0 / (test_metrics['rmse'] + 1e-8)  # 避免除零
            r2_weight = test_metrics['r2']
            adj_r2_weight = test_metrics['adjusted_r2']
            
            composite_score = rmse_weight + max(0, r2_weight) + max(0, adj_r2_weight)
            
            # 记录结果
            result = {
                'params': params,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'composite_score': composite_score,
                'training_time': time.time() - start_time
            }
            
            self.hyperopt_results.append(result)
            
            self.logger.info(f"模型训练完成，RMSE: {test_metrics['rmse']:.6f}, R²: {test_metrics['r2']:.6f}, 耗时: {result['training_time']:.2f}秒")
            
            # Hyperopt需要最小化，所以返回负的综合评分
            return {'loss': -composite_score, 'status': STATUS_OK}
            
        except Exception as e:
            self.logger.error(f"目标函数执行出错: {str(e)}")
            return {'loss': 1.0, 'status': STATUS_FAIL}
    
    def hyperparameter_optimization(self, max_evals=100):
        """执行超参数优化"""
        self.logger.info("开始超参数优化...")
        self.hyperopt_results = []
        
        # 定义XGBoost搜索空间
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
            
            # 保存超参数优化结果
            self.save_hyperopt_results()
            self.logger.info("超参数优化完成")
            
            return best
        except Exception as e:
            self.logger.error(f"超参数优化失败: {str(e)}")
            self.logger.info("使用默认参数继续流程")
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
        """保存超参数优化结果"""
        if not self.hyperopt_results:
            self.logger.warning("没有超参数优化结果可保存")
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
        self.logger.info(f"超参数优化结果已保存: {results_file}")
    
    def find_best_model(self):
        """找到最优模型"""
        if not self.hyperopt_results:
            self.logger.error("没有可用的超参数优化结果")
            raise ValueError("没有可用的超参数优化结果")
            
        # 选择RMSE最小且R2和adjusted R2较大的模型
        best_result = min(self.hyperopt_results, 
                         key=lambda x: (x['test_metrics']['rmse'], 
                                      -x['test_metrics']['r2'], 
                                      -x['test_metrics']['adjusted_r2']))
        
        self.best_params = best_result['params']
        self.best_test_metrics = best_result['test_metrics']
        
        self.logger.info(f"最优参数: n_estimators={self.best_params['n_estimators']}, max_depth={self.best_params['max_depth']}, learning_rate={self.best_params['learning_rate']:.6f}")
        self.logger.info(f"最优指标 - RMSE: {self.best_test_metrics['rmse']:.6f}, R²: {self.best_test_metrics['r2']:.6f}, Adjusted R²: {self.best_test_metrics['adjusted_r2']:.6f}")
        
        # 训练最终模型
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
            'tree_method': 'hist',  # 关键修改：CPU版直方图算法
            'random_state': 42
        }
        
        self.best_model = xgb.XGBRegressor(**xgb_params)
        
        try:
            self.best_model = self.safe_xgb_fit(self.best_model, self.X_train_scaled, self.y_train)
            
            # 保存最优模型
            model_file = os.path.join(
                self.dirs['models'], 
                f"{self.output_prefix}_best_model.joblib"
            )
            joblib.dump(self.best_model, model_file)
            
            # 保存最优指标
            metrics_df = pd.DataFrame([self.best_test_metrics])
            metrics_file = os.path.join(
                self.dirs['results'], 
                f"{self.output_prefix}_best_metrics.csv"
            )
            metrics_df.to_csv(metrics_file, index=False, float_format='%.6f')
            
            self.logger.info(f"最优模型已保存: {model_file}")
        except Exception as e:
            self.logger.error(f"最终模型训练失败: {str(e)}")
            self.best_model = None
    
    def cross_validation(self):
        """执行K折交叉验证"""
        self.logger.info("开始K折交叉验证...")
        
        if self.best_params is None:
            self.logger.error("没有找到最优参数，无法进行交叉验证")
            return None
            
        cv_results = []
        kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train)):
            fold_start_time = time.time()
            
            try:
                # 分割数据
                X_train_fold, X_val_fold = self.X_train[train_idx], self.X_train[val_idx]
                y_train_fold, y_val_fold = self.y_train[train_idx], self.y_train[val_idx]
                
                # 标准化（使用训练折的数据拟合）
                scaler_fold = StandardScaler()
                X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
                X_val_fold_scaled = scaler_fold.transform(X_val_fold)
                X_test_fold_scaled = scaler_fold.transform(self.X_test)
                
                # 训练模型
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
                    'tree_method': 'hist',  # 关键修改：CPU版直方图算法
                    'random_state': 42
                }
                
                model = xgb.XGBRegressor(**xgb_params)
                model = self.safe_xgb_fit(model, X_train_fold_scaled, y_train_fold)
                
                # 验证集预测
                y_val_pred = model.predict(X_val_fold_scaled)
                
                # 测试集预测
                y_test_pred = model.predict(X_test_fold_scaled)
                
                # 计算指标
                val_metrics = self.calculate_metrics(y_val_fold, y_val_pred, 'validation')
                test_metrics = self.calculate_metrics(self.y_test, y_test_pred, 'test')
                
                cv_results.append({
                    'fold': fold + 1,
                    **{f'val_{k}': v for k, v in val_metrics.items()},
                    **{f'test_{k}': v for k, v in test_metrics.items()},
                    'training_time': time.time() - fold_start_time
                })
                
                self.logger.info(f"第{fold + 1}折完成, 验证集RMSE: {val_metrics['rmse']:.6f}, 测试集RMSE: {test_metrics['rmse']:.6f}")
                
            except Exception as e:
                self.logger.error(f"第{fold + 1}折交叉验证失败: {str(e)}")
                continue
        
        if not cv_results:
            self.logger.error("所有交叉验证折都失败了")
            return None
            
        # 保存交叉验证结果
        cv_df = pd.DataFrame(cv_results)
        cv_file = os.path.join(
            self.dirs['cv_results'], 
            f"{self.output_prefix}_cv_results.csv"
        )
        cv_df.to_csv(cv_file, index=False, float_format='%.6f')
        
        # 计算平均指标
        mean_metrics = {}
        for col in cv_df.columns:
            if col != 'fold':
                mean_metrics[col] = cv_df[col].mean()
        
        mean_metrics_file = os.path.join(
            self.dirs['results'], 
            f"{self.output_prefix}_cv_mean_metrics.csv"
        )
        pd.DataFrame([mean_metrics]).to_csv(mean_metrics_file, index=False, float_format='%.6f')
        
        self.logger.info("K折交叉验证完成")
        return cv_df
    
    def run_pipeline(self, train_file, test_file, max_evals=100):
        """运行完整流程"""
        try:
            self.logger.info("开始执行完整流程...")
            
            # 执行各个步骤
            self.load_data(train_file, test_file)
            self.standardize_data()
            best_params = self.hyperparameter_optimization(max_evals)
            self.find_best_model()
            cv_results = self.cross_validation()
            
            # 计算总耗时
            total_time = time.time() - self.start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            
            self.logger.info(f"项目执行完成！总耗时: {hours}小时{minutes}分钟{seconds}秒")
            
            # 输出最终结果摘要
            if hasattr(self, 'best_test_metrics'):
                self.logger.info("最终结果摘要:")
                self.logger.info(f"最优参数: n_estimators={self.best_params['n_estimators']}, max_depth={self.best_params['max_depth']}, learning_rate={self.best_params['learning_rate']:.6f}")
                self.logger.info(f"独立测试集结果:")
                for metric, value in self.best_test_metrics.items():
                    self.logger.info(f"  {metric}: {value:.6f}")
            
            return {
                'best_params': self.best_params,
                'best_metrics': self.best_test_metrics if hasattr(self, 'best_test_metrics') else None,
                'cv_results': cv_results,
                'total_time': total_time
            }
            
        except Exception as e:
            self.logger.error(f"流程执行出错: {str(e)}")
            if not hasattr(self, 'best_params'):
                self.logger.info("使用默认参数创建模型")
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
    """加载已训练的单 XGBoost 模型 + scaler 并预测"""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    X = scaler.transform(features)
    return model.predict(X)


def main():
    parser = argparse.ArgumentParser(description='XGBoost回归项目（CPU版）')
    parser.add_argument('--project_name', type=str, required=True, 
                       help='项目名称')
    parser.add_argument('--model_type', type=str, default='XGBoost',
                       help='模型类型（当前仅支持XGBoost）')
    parser.add_argument('--output_prefix', type=str, required=True,
                       help='保存文件的名称前缀')
    parser.add_argument('--k_fold', type=int, default=5,
                       help='交叉验证的折数')
    parser.add_argument('--train_file', type=str, required=True,
                       help='训练集文件路径')
    parser.add_argument('--test_file', type=str, required=True,
                       help='测试集文件路径')
    parser.add_argument('--max_evals', type=int, default=100,
                       help='超参数优化迭代次数')
    
    args = parser.parse_args()
    
    try:
        # 创建并运行回归器
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
            
            print(f"\n项目完成！")
            print(f"总耗时: {hours}小时{minutes}分钟{seconds}秒")
            print(f"最优参数: n_estimators={results['best_params']['n_estimators']}, max_depth={results['best_params']['max_depth']}, learning_rate={results['best_params']['learning_rate']:.6f}")
            
            if results['best_metrics']:
                print("独立测试集结果:")
                for metric, value in results['best_metrics'].items():
                    print(f"  {metric}: {value:.6f}")
            
            if results['cv_results'] is not None:
                cv_means = results['cv_results'].mean(numeric_only=True)
                print("交叉验证平均结果:")
                for col in results['cv_results'].columns:
                    if col.startswith('test_') and col != 'fold':
                        metric_name = col.replace('test_', '')
                        print(f"  {metric_name}: {cv_means[col]:.6f}")
        else:
            print("项目完成，但存在一些问题，请检查日志")
        
    except Exception as e:
        print(f"项目执行失败: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    main()