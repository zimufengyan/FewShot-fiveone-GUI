import sys
import os
import logging
from PyQt6.QtCore import *
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6.QtGui import QPixmap, QIcon
import torch
from torchvision.transforms import ToPILImage
import PIL
from PIL import ImageQt, Image

from ui.Ui_main import Ui_MainWindow
from factory import ModelFactor
from dataloader import DatasetLoader


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        # init log
        logging.basicConfig(
            level = logging.DEBUG,
            format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.getLogger('PIL').setLevel(logging.WARNING) # 设置PIL模块的日志等级为WARNING
        self.log = logging.getLogger(__name__)
        
        # init ui
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # signals and slots
        self.ui.sampleBtn.clicked.connect(self.on_click_sample_btn)
        self.ui.predBtn.clicked.connect(self.on_click_pred_btn)
        self.ui.loadBtn.clicked.connect(self.on_click_load_btn)        
        # dir
        self.ref_dir = '.' + os.sep + 'ref'
        
        # for model
        self.n_ways = 5
        self.k_shot = 1
        self.model_factor = ModelFactor()
        self.dataloader = DatasetLoader()
        self.model = None
        self.dataset_name = None
        
        self.initUi()        
        
    def initUi(self):
        self.setWindowTitle("Five-Way-One-Shot Prediction")
        self.setWindowIcon(QIcon(os.path.join(self.ref_dir, 'icon.png')))
        self.ui.modelCombo.addItems(self.model_factor.model_names)
        self.ui.datasetCombo.addItems(['Omniglot', 'mini-ImageNet'])
        self.ui.accLabel.setText("")
        
    @staticmethod
    def _to_qpixmap(sample, size=(128, 128)):
        if isinstance(sample, torch.Tensor):
            trans = ToPILImage()
            sample = trans(sample)
        if not isinstance(sample, PIL.Image.Image):
            raise TypeError("The type of param 'sample' should be torch.Tensor or PIL.Image.Image")
        image = sample.resize(size, Image.Resampling.BICUBIC)
        pixmap = ImageQt.toqpixmap(image)
        return pixmap
    
    def on_click_load_btn(self):
        sender = self.sender()
        self.log.debug(f'The button "{sender.objectName()}" has been clicked')
        self.load_dataset()
        self.load_model()
        
    def load_dataset(self):
        self.dataset_name = self.ui.datasetCombo.currentText()
        self.dataloader.init_dataset(self.dataset_name)
        
    def load_model(self):
        # load corresponding model
        model_name = self.ui.modelCombo.currentText()
        self.log.debug("Loading model")
        self.model = self.model_factor.get_model(model_name, self.dataset_name, self.n_ways)
        self.log.debug("Model has been loaded")
        accuracy = self.model_factor.get_accuracy(model_name, self.dataset_name)
        self.log.debug(f"Model accuracy: {accuracy}")
        self.ui.accLabel.setText(str(accuracy))
        
    def on_click_sample_btn(self):
        sender = self.sender()
        self.log.debug(f'The button "{sender.objectName()}" has been clicked')
        
        # hide arrow labels
        self.hide_arrows()
        
        self.log.debug(f'Sample a batch of samples to show')
        if not self.dataloader.is_available(): 
            self.log.error('The dataloader is not available, return')
            warning_txt = '请先选择模型和数据源，并点击加载按钮加载响应组件'
            QMessageBox.warning(
                self, "警告", warning_txt, QMessageBox.StandardButton.Ok, QMessageBox.StandardButton.Ok
            )
            return
        self.support_set, self.query, self.query_label = self.dataloader.get_batch(self.n_ways)
       
        self.ui.targetClassLabel.setText(str(self.query_label + 1))
        target_sample = self.query.copy() 
        target_pixmap = self._to_qpixmap(target_sample)
        self.ui.targetImageLabel.setPixmap(target_pixmap)
        for i in range(0, self.n_ways):
            sample = self.support_set[i].copy()
            pixmap = self._to_qpixmap(sample)
            label = getattr(self.ui, f'imgLabel_{i+1}')
            label.setPixmap(pixmap)  
        self.log.debug(f'End showing')  
        
        self.show_true_arrow()
    
    def on_click_pred_btn(self):
        sender = self.sender()
        self.log.debug(f'The button "{sender.objectName()}" has been clicked')
        if self.model == None:
            self.log.error("The model, dataset, sampler are NontType, load them")
            warning_txt = '请先选择模型和数据源，并点击加载按钮加载响应组件'
            QMessageBox.warning(
                self, "警告", warning_txt, QMessageBox.StandardButton.Ok, QMessageBox.StandardButton.Ok
            )
            return
        # transform samples
        self.log.debug('Predicting samples')
        Xs = self.dataloader.transformer(self.support_set)
        Xq = self.dataloader.transformer(self.query)
        pred = self.model_factor.pred(self.model, Xs=Xs, Xq=Xq, n_way=self.n_ways)
        self.log.debug(f"End prediction, pred value: {pred}")
        self.show_pred_arrow(pred + 1)
            
    def show_true_arrow(self):
        target_label = self.ui.targetClassLabel.text()
        arrow_path = os.path.join(self.ref_dir, f'right_arrow_{target_label}.png')
        if not os.path.exists(arrow_path):
            self.log.error(f"No such file or directory: {arrow_path}")
            return
        pixmap = QPixmap(arrow_path)
        self.ui.trueLabel.setVisible(True)
        self.ui.trueLabel.setPixmap(pixmap)
        
    def show_pred_arrow(self, pred_class):
        arrow_path = os.path.join(self.ref_dir, f'blue_arrow_{pred_class}.png')
        if not os.path.exists(arrow_path):
            self.log.error(f"No such file or directory: {arrow_path}")
            return
        pixmap = QPixmap(arrow_path)
        self.ui.predLabel.setVisible(True)
        self.ui.predLabel.setPixmap(pixmap)
        
    def hide_arrows(self):
        self.ui.trueLabel.setVisible(False)
        self.ui.predLabel.setVisible(False)
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec())    # exec() for PyQt6, exec_() for PyQt5