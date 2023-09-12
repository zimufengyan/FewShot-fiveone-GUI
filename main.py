import sys
import os
import logging
from PyQt6.QtCore import *
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6.QtGui import QPixmap, QIcon
import torch
from torchvision import datasets
from torchvision.transforms import transforms, InterpolationMode
import PIL
from PIL import ImageQt, Image

from ui.Ui_main import Ui_MainWindow
from metric_models.matching import MatchingModel
from metric_models.proto import ProtoModel
from sampler import OneShotSampler

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
        self.model = None
        self.dataset = None
        self.sampler = None
        self.samples = None
        self.dataset_name = None
        self.transform = None
        self.load_dir = '.' + os.sep + 'checkpoints'
        self.accuracy_dict = {
            'MatchingNet_on_Omniglot': 0.8180, 'ProtoNet_on_Omniglot': 0.9840,
        }
        
        self.initUi()        
        
    def initUi(self):
        self.setWindowTitle("Five-Ways-One-Shot Prediction")
        self.setWindowIcon(QIcon(os.path.join(self.ref_dir, 'icon.png')))
        self.ui.modelCombo.addItems(['MatchingNet', 'ProtoNet'])
        self.ui.datasetCombo.addItems(['Omniglot', 'mini-ImageNet'])
        self.ui.accLabel.setText("")
        
    @staticmethod
    def _to_qpixmap(sample, size=(128, 128)):
        if isinstance(sample, torch.Tensor):
            trans = transforms.ToPILImage()
            sample = trans(sample)
        if isinstance(sample, PIL.PngImagePlugin.PngImageFile):
            raise TypeError("The type of param 'sample' should be torch.Tensor or PIL.PngImagePlugin.PngImageFile")
        image = sample.resize(size, Image.Resampling.BICUBIC)
        pixmap = ImageQt.toqpixmap(image)
        return pixmap
    
    def on_click_load_btn(self):
        sender = self.sender()
        self.log.debug(f'The button "{sender.objectName()}" has been clicked')
        self.load_dataset()
        self.load_model()
        accuracy = self.accuracy_dict.get(self.model_name, 0.0000)
        self.ui.accLabel.setText(str(accuracy))
        
    def load_dataset(self):
        self.dataset_name = self.ui.datasetCombo.currentText()
        if self.dataset is not None and self.dataset.__class__.__name__ == self.dataset_name:
            self.log.debug("The dataset had been loaded, skipping this operation")
            return
        if self.dataset_name == 'Omniglot':
            self.log.debug("Loading Omniglot dataset")
            self.transform = transforms.Compose([
                transforms.Resize((28, 28), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.87, std=0.33)
            ])
            # use transform before prediction, but after showing
            self.dataset = datasets.Omniglot(root='./data', background=False, download=False)
            self.sampler = OneShotSampler(self.dataset)
        elif self.dataset_name == 'mini_ImageNet':
            return
        else:
            return
        self.log.debug("Dataset has been loaded")
        
    def load_model(self):
        # load corresponding model
        model_name = self.ui.modelCombo.currentText()
        self.model_name = model_name + '_on_' + self.dataset_name
        folder = os.path.join(self.load_dir, model_name)
        if self.dataset_name == 'Omniglot':
            in_channels, out_channels, num_hiddens = 1, 64, 64
            input_shape = [28, 28]
            epoch = 100
        else:
            return
        if model_name == 'MatchingNet':
            self.model = MatchingModel(in_channels, out_channels, input_shape, num_hiddens=num_hiddens, is_train=False)
        elif model_name == 'ProtoNet':
            self.model = ProtoModel(in_channels, out_channels, input_shape, num_hidden=num_hiddens, is_train=False)
        else:return
        self.log.debug(f"Loading model from path: {folder}")
        self.model.load_networks(self.load_dir, self.dataset_name, epoch)
        self.model.to_device()
        self.log.debug("Model has been loaded")
        
    def on_click_sample_btn(self):
        sender = self.sender()
        self.log.debug(f'The button "{sender.objectName()}" has been clicked')
        
        # hide arrow labels
        self.hide_arrows()
        
        self.log.debug(f'Sample a batch of samples to show')
        if self.sampler == None: 
            self.log.error('The sampler is NoneType, return')
            warning_txt = '请先选择模型和数据源，并点击加载按钮加载响应组件'
            QMessageBox.warning(
                self, "警告", warning_txt, QMessageBox.StandardButton.Ok, QMessageBox.StandardButton.Ok
            )
            return
        self.support_set, self.query, self.query_label = self.sampler.sample_batch(self.n_ways)
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
        Xs = self.sampler.transformer(self.support_set, self.transform)
        Xq = self.sampler.transformer(self.query, self.transform).unsqueeze(0)
        pred = self.model.pred(Xs, Xq, self.n_ways)
        self.log.debug(f"End prediction, pred value: {pred.cpu().item()}")
        self.show_pred_arrow(pred.cpu().item() + 1)
            
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