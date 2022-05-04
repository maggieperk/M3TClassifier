import numpy as np
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix

def compare_results_of_bert_models(model_list, x_bert_test_list, y_test_list):

    prec_list = []
    recall_list = []
    accuracy_list = []
    all_predictions = []
    
    for i in range(5):
        model = model_list[i]
        x_test = x_bert_test_list[i].reshape(-1, 1, 300)
        
        y_test = y_test_list[i]
        test_labels = np.argmax(convert_da_labels_to_categorical(y_test), axis=-1)
        
        print(f"Calculating accuracy for split{i}")
        model_preds = model_list[i].predict(x_test)
        unilabel_model_preds = np.argmax(model_preds, axis=-1)
        
        all_predictions.extend(unilabel_model_preds)
        
        lstm_prec = precision_score(test_labels, unilabel_model_preds, average='macro')
        lstm_recall = recall_score(test_labels, unilabel_model_preds, average='macro')        
        accuracy = accuracy_score(test_labels, unilabel_model_preds)
        
        prec_list.append(lstm_prec)
        recall_list.append(lstm_recall)
        accuracy_list.append(accuracy)
        
    
    return prec_list, recall_list, accuracy_list, all_predictions


def calculate_accuracy_metrics_and_print_cm(actual, model_preds):
  cr = classification_report(actual, model_preds)
  cm = confusion_matrix(actual, model_preds)
  df_cm = pd.DataFrame(cm, index=da_labels, columns=da_labels)
  ax = sn.heatmap(df_cm, cmap='Blues', annot=True)
  print(ax)
  return cr
