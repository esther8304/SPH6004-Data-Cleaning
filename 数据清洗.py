import pandas as pd
import numpy as np

print("开始数据清洗...")

# 读取数据
static = pd.read_csv("MIMIC-IV-static(Group Assignment).csv")
text = pd.read_csv("MIMIC-IV-text(Group Assignment).csv")
ts = pd.read_csv("MIMIC-IV-time_series(Group Assignment).csv")

print(f"原始数据: 静态{static.shape}, 文本{text.shape}, 时序{ts.shape}")

# 1. 静态数据清洗
# 移除异常ICU住院时长
static = static[(static['icu_los_hours'] > 0) & (static['icu_los_hours'] <= 720)]

# 填充分类变量
cat_cols = ['first_careunit', 'last_careunit', 'insurance', 'language', 'race', 'marital_status', 'gender']
for col in cat_cols:
    if col in static.columns:
        static[col] = static[col].fillna('Unknown')

# 填充数值变量（用中位数）
num_cols = static.select_dtypes(include=[np.number]).columns.tolist()
id_cols = ['subject_id', 'hadm_id', 'stay_id']
num_cols = [c for c in num_cols if c not in id_cols + ['icu_los_hours']]
for col in num_cols:
    if col in static.columns and static[col].isnull().sum() > 0:
        static[col] = static[col].fillna(static[col].median())

# 2. 文本数据处理
# 先填充NaN值
text['radiology_note_text'] = text['radiology_note_text'].fillna('')

# 合并同一患者的文本
text_agg = text.groupby('stay_id').agg({
    'radiology_note_text': lambda x: ' '.join(x),
    'subject_id': 'first'
}).reset_index()
text_agg['text_length'] = text_agg['radiology_note_text'].str.len()

# 3. 时序数据处理
# 提取统计特征
ts_features = ts.groupby('stay_id').agg({
    'hr': ['mean', 'std', 'min', 'max'],
    'map': ['mean', 'std', 'min', 'max'],
    'rr': ['mean', 'std'],
    'spo2': ['mean', 'min'],
    'temp': ['mean'],
    'gcs': ['mean', 'min'],
    'ventilation_flag': ['mean'],
    'sofa_resp': ['max'],
    'sofa_cardio': ['max'],
    'sofa_renal': ['max'],
    'sofa_liver': ['max'],
    'sofa_coag': ['max'],
    'sofa_cns': ['max']
}).reset_index()
ts_features.columns = ['stay_id'] + [f'{col[0]}_{col[1]}' for col in ts_features.columns[1:]]

# 4. 合并数据
merged = static.merge(ts_features, on='stay_id', how='inner')
merged = merged.merge(text_agg, on='stay_id', how='left')

# 填充文本缺失值
merged['text_length'] = merged['text_length'].fillna(0)
merged['radiology_note_text'] = merged['radiology_note_text'].fillna('')

# 移除不需要的列
remove_cols = ['subject_id', 'hadm_id', 'intime', 'outtime', 'deathtime']
merged = merged.drop(columns=[c for c in remove_cols if c in merged.columns])

# 保存
merged.to_csv("cleaned_data.csv", index=False)
print(f"\n清洗完成! 最终数据: {merged.shape}")
print(f"目标变量: icu_los_hours")
print(f"统计: {merged['icu_los_hours'].describe()}")