import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def preprocess_questionnaire_data(file_path):
    """
    处理问卷数据的完整流程
    """
    
    # 1. 读取数据
    df = pd.read_excel(file_path)
    print(f"原始数据形状: {df.shape}")
    
    # 2. 数据清洗
    # 删除完全空白的行和列
    df_clean = df.dropna(how='all').dropna(axis=1, how='all')
    
    # 处理时间列（如果需要）
    if '提交答卷时间' in df_clean.columns:
        try:
            df_clean['提交答卷时间'] = pd.to_datetime(df_clean['提交答卷时间'], unit='d', origin='1899-12-30')
        except:
            print("时间列转换失败，跳过时间处理")
    
    # 3. 识别并处理无效数据
    # 假设某些列有特定的无效值
    invalid_patterns = {
        '身高': lambda x: (x < 100) | (x > 250),  # 身高异常范围
        '体重': lambda x: (x < 20) | (x > 200),   # 体重异常范围
        '总分': lambda x: x < 0,                   # 负分
    }
    
    # 创建无效数据标识
    invalid_mask = pd.Series([False] * len(df_clean))
    
    for col, pattern in invalid_patterns.items():
        if col in df_clean.columns:
            col_name = next((c for c in df_clean.columns if col in str(c)), None)
            if col_name:
                invalid_mask = invalid_mask | pattern(df_clean[col_name])
    
    # 删除无效数据
    df_clean = df_clean[~invalid_mask].copy()
    print(f"删除无效数据后形状: {df_clean.shape}")
    print(f"删除无效数据数量: {invalid_mask.sum()}")
    
    # 4. 分离特征类型
    # 识别数值型列（排除序号、ID等）
    numeric_cols = []
    categorical_cols = []
    binary_cols = []
    
    for col in df_clean.columns:
        # 跳过非数值列
        if col in ['序号', '提交答卷时间', '所用时间', '来源', '来源详情', '来自IP']:
            continue
        
        # 尝试转换为数值
        try:
            pd.to_numeric(df_clean[col])
            unique_vals = df_clean[col].dropna().unique()
            
            if len(unique_vals) <= 2:  # 二值特征
                binary_cols.append(col)
            else:  # 多值数值特征
                numeric_cols.append(col)
        except:
            # 分类特征
            categorical_cols.append(col)
    
    print(f"\n特征类型统计:")
    print(f"数值型特征: {len(numeric_cols)} 个")
    print(f"二值特征: {len(binary_cols)} 个")
    print(f"分类特征: {len(categorical_cols)} 个")
    
    # 5. 处理缺失值
    # 数值列用中位数填充
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
    
    # 二值列用众数填充
    for col in binary_cols:
        if df_clean[col].isnull().any():
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
            df_clean[col] = df_clean[col].fillna(mode_val)
    
    # 6. 数据归一化（只对数值型特征）
    scaler = MinMaxScaler()
    
    if numeric_cols:
        # 创建归一化后的数据框
        normalized_data = scaler.fit_transform(df_clean[numeric_cols])
        df_normalized = pd.DataFrame(normalized_data, 
                                     columns=[f'{col}_norm' for col in numeric_cols],
                                     index=df_clean.index)
        
        # 合并回原数据框
        df_clean = pd.concat([df_clean, df_normalized], axis=1)
        print(f"\n已对 {len(numeric_cols)} 个数值特征进行归一化")
    
    # 7. 编码分类变量（如果需要）
    # 这里可以选择性地对重要的分类变量进行编码
    important_categorical = ['1、你所在的年级是：', '2、你的性别是：', '3、你的月生活费：']
    
    label_encoders = {}
    for col in important_categorical:
        if col in df_clean.columns:
            le = LabelEncoder()
            df_clean[f'{col}_encoded'] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le
    
    # 8. 特征工程（根据问卷特点）
    # 创建BMI指数特征
    height_col = next((c for c in df_clean.columns if '身高' in str(c)), None)
    weight_col = next((c for c in df_clean.columns if '体重' in str(c)), None)
    
    if height_col and weight_col:
        # 确保是数值
        df_clean[height_col] = pd.to_numeric(df_clean[height_col], errors='coerce')
        df_clean[weight_col] = pd.to_numeric(df_clean[weight_col], errors='coerce')
        
        # 计算BMI（体重kg / 身高m^2）
        df_clean['BMI'] = df_clean[weight_col] / ((df_clean[height_col] / 100) ** 2)
        
        # BMI分类
        def categorize_bmi(bmi):
            if pd.isna(bmi):
                return np.nan
            elif bmi < 18.5:
                return 0  # 偏瘦
            elif bmi < 24:
                return 1  # 正常
            elif bmi < 28:
                return 2  # 偏胖
            else:
                return 3  # 肥胖
        
        df_clean['BMI_category'] = df_clean['BMI'].apply(categorize_bmi)
    
    # 9. 保存处理后的数据
    output_path = file_path.replace('.xlsx', '_processed.xlsx')
    df_clean.to_excel(output_path, index=False)
    
    print(f"\n数据处理完成!")
    print(f"处理后的数据形状: {df_clean.shape}")
    print(f"处理后的数据已保存到: {output_path}")
    
    # 10. 返回处理后的数据和关键信息
    return {
        'data': df_clean,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'binary_cols': binary_cols,
        'scaler': scaler if numeric_cols else None,
        'label_encoders': label_encoders
    }

# 使用示例
if __name__ == "__main__":
    # 替换为你的文件路径
    file_path = "问卷数据.xlsx"
    
    try:
        result = preprocess_questionnaire_data(file_path)
        
        # 查看处理后的数据基本信息
        print("\n处理后的数据前5行:")
        print(result['data'].head())
        
        # 查看数据统计信息
        print("\n数值特征描述统计:")
        print(result['data'].describe())
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")