from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\clg\Desktop\降采样\SC_predictions.csv")
# 确保 label_data 和 final_prediction 长度一致

y_true = data['real label'].values
y_pred = data['Predicted'].values

# 计算并绘制混淆矩阵
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['W', 'N1', 'N2', 'N3', 'REM'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# 打印分类报告（precision, recall, f1-score, support）
print("分类报告：")
print(classification_report(y_true, y_pred, target_names=['W', 'N1', 'N2', 'N3', 'REM']))
