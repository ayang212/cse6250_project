import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	plt.plot(train_losses, label = "Training Loss")
	plt.plot(valid_losses, label = "Validation Loss")
	plt.xlabel('Epoch') 
	plt.ylabel('Loss')
	plt.title('Loss Curve')
	plt.legend()
	plt.show()

	plt.plot(train_accuracies, label = "Training Accuracy")
	plt.plot(valid_accuracies, label = "Validation Accuracy")
	plt.xlabel('Epoch') 
	plt.ylabel('Accuracy')
	plt.title('Accuracy Curve')
	plt.legend()
	plt.show()
	


def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	# results = (zip(y_true, y_pred))
	# results_unzipped = [y_true, y_pred]
	y_true, y_pred = list(zip(*results))
	cm = confusion_matrix(y_true, y_pred)
	cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	fig, ax = plt.subplots(figsize=(10,10))
	sn.heatmap(cmn, annot=True, fmt='.2f', xticklabels=class_names, yticklabels=class_names)
	plt.ylabel('Actual')
	plt.xlabel('Predicted')
	plt.show()
