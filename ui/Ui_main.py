# Form implementation generated from reading ui file 'f:\document\post-graduation\Meta-Learning\Replay\GUI\ui\main.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(950, 495)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(950, 495))
        MainWindow.setMaximumSize(QtCore.QSize(950, 495))
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(parent=self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(160, 309, 139, 167))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.imgLabel_2 = QtWidgets.QLabel(parent=self.layoutWidget)
        self.imgLabel_2.setMinimumSize(QtCore.QSize(128, 128))
        self.imgLabel_2.setMaximumSize(QtCore.QSize(128, 128))
        self.imgLabel_2.setStyleSheet("background-color:white;\n"
"border:1px solid  #0072E3")
        self.imgLabel_2.setText("")
        self.imgLabel_2.setObjectName("imgLabel_2")
        self.verticalLayout_2.addWidget(self.imgLabel_2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_15 = QtWidgets.QLabel(parent=self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_15.sizePolicy().hasHeightForWidth())
        self.label_15.setSizePolicy(sizePolicy)
        self.label_15.setMinimumSize(QtCore.QSize(64, 28))
        self.label_15.setMaximumSize(QtCore.QSize(64, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_15.setFont(font)
        self.label_15.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.horizontalLayout_2.addWidget(self.label_15)
        self.classLabel_2 = QtWidgets.QLabel(parent=self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.classLabel_2.sizePolicy().hasHeightForWidth())
        self.classLabel_2.setSizePolicy(sizePolicy)
        self.classLabel_2.setMinimumSize(QtCore.QSize(64, 28))
        self.classLabel_2.setMaximumSize(QtCore.QSize(64, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.classLabel_2.setFont(font)
        self.classLabel_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.classLabel_2.setObjectName("classLabel_2")
        self.horizontalLayout_2.addWidget(self.classLabel_2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.layoutWidget_2 = QtWidgets.QWidget(parent=self.centralwidget)
        self.layoutWidget_2.setGeometry(QtCore.QRect(300, 309, 139, 167))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget_2)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.imgLabel_3 = QtWidgets.QLabel(parent=self.layoutWidget_2)
        self.imgLabel_3.setMinimumSize(QtCore.QSize(128, 128))
        self.imgLabel_3.setMaximumSize(QtCore.QSize(128, 128))
        self.imgLabel_3.setStyleSheet("background-color:white;\n"
"border:1px solid  #0072E3")
        self.imgLabel_3.setText("")
        self.imgLabel_3.setObjectName("imgLabel_3")
        self.verticalLayout_3.addWidget(self.imgLabel_3)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_17 = QtWidgets.QLabel(parent=self.layoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_17.sizePolicy().hasHeightForWidth())
        self.label_17.setSizePolicy(sizePolicy)
        self.label_17.setMinimumSize(QtCore.QSize(64, 28))
        self.label_17.setMaximumSize(QtCore.QSize(64, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_17.setFont(font)
        self.label_17.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_17.setObjectName("label_17")
        self.horizontalLayout_3.addWidget(self.label_17)
        self.classLabel_3 = QtWidgets.QLabel(parent=self.layoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.classLabel_3.sizePolicy().hasHeightForWidth())
        self.classLabel_3.setSizePolicy(sizePolicy)
        self.classLabel_3.setMinimumSize(QtCore.QSize(64, 28))
        self.classLabel_3.setMaximumSize(QtCore.QSize(64, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.classLabel_3.setFont(font)
        self.classLabel_3.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.classLabel_3.setObjectName("classLabel_3")
        self.horizontalLayout_3.addWidget(self.classLabel_3)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.layoutWidget_3 = QtWidgets.QWidget(parent=self.centralwidget)
        self.layoutWidget_3.setGeometry(QtCore.QRect(440, 309, 139, 167))
        self.layoutWidget_3.setObjectName("layoutWidget_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.layoutWidget_3)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.imgLabel_4 = QtWidgets.QLabel(parent=self.layoutWidget_3)
        self.imgLabel_4.setMinimumSize(QtCore.QSize(128, 128))
        self.imgLabel_4.setMaximumSize(QtCore.QSize(128, 128))
        self.imgLabel_4.setStyleSheet("background-color:white;\n"
"border:1px solid  #0072E3")
        self.imgLabel_4.setText("")
        self.imgLabel_4.setObjectName("imgLabel_4")
        self.verticalLayout_4.addWidget(self.imgLabel_4)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_19 = QtWidgets.QLabel(parent=self.layoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_19.sizePolicy().hasHeightForWidth())
        self.label_19.setSizePolicy(sizePolicy)
        self.label_19.setMinimumSize(QtCore.QSize(64, 28))
        self.label_19.setMaximumSize(QtCore.QSize(64, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_19.setFont(font)
        self.label_19.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_19.setObjectName("label_19")
        self.horizontalLayout_4.addWidget(self.label_19)
        self.classLabel_4 = QtWidgets.QLabel(parent=self.layoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.classLabel_4.sizePolicy().hasHeightForWidth())
        self.classLabel_4.setSizePolicy(sizePolicy)
        self.classLabel_4.setMinimumSize(QtCore.QSize(64, 28))
        self.classLabel_4.setMaximumSize(QtCore.QSize(64, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.classLabel_4.setFont(font)
        self.classLabel_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.classLabel_4.setObjectName("classLabel_4")
        self.horizontalLayout_4.addWidget(self.classLabel_4)
        self.verticalLayout_4.addLayout(self.horizontalLayout_4)
        self.layoutWidget_4 = QtWidgets.QWidget(parent=self.centralwidget)
        self.layoutWidget_4.setGeometry(QtCore.QRect(580, 309, 139, 167))
        self.layoutWidget_4.setObjectName("layoutWidget_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.layoutWidget_4)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.imgLabel_5 = QtWidgets.QLabel(parent=self.layoutWidget_4)
        self.imgLabel_5.setMinimumSize(QtCore.QSize(128, 128))
        self.imgLabel_5.setMaximumSize(QtCore.QSize(128, 128))
        self.imgLabel_5.setStyleSheet("background-color:white;\n"
"border:1px solid  #0072E3")
        self.imgLabel_5.setText("")
        self.imgLabel_5.setObjectName("imgLabel_5")
        self.verticalLayout_5.addWidget(self.imgLabel_5)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_21 = QtWidgets.QLabel(parent=self.layoutWidget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_21.sizePolicy().hasHeightForWidth())
        self.label_21.setSizePolicy(sizePolicy)
        self.label_21.setMinimumSize(QtCore.QSize(64, 28))
        self.label_21.setMaximumSize(QtCore.QSize(64, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_21.setFont(font)
        self.label_21.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_21.setObjectName("label_21")
        self.horizontalLayout_5.addWidget(self.label_21)
        self.classLabel_5 = QtWidgets.QLabel(parent=self.layoutWidget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.classLabel_5.sizePolicy().hasHeightForWidth())
        self.classLabel_5.setSizePolicy(sizePolicy)
        self.classLabel_5.setMinimumSize(QtCore.QSize(64, 28))
        self.classLabel_5.setMaximumSize(QtCore.QSize(64, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.classLabel_5.setFont(font)
        self.classLabel_5.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.classLabel_5.setObjectName("classLabel_5")
        self.horizontalLayout_5.addWidget(self.classLabel_5)
        self.verticalLayout_5.addLayout(self.horizontalLayout_5)
        self.layoutWidget_5 = QtWidgets.QWidget(parent=self.centralwidget)
        self.layoutWidget_5.setGeometry(QtCore.QRect(300, 39, 139, 167))
        self.layoutWidget_5.setObjectName("layoutWidget_5")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.layoutWidget_5)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.targetImageLabel = QtWidgets.QLabel(parent=self.layoutWidget_5)
        self.targetImageLabel.setMinimumSize(QtCore.QSize(128, 128))
        self.targetImageLabel.setMaximumSize(QtCore.QSize(128, 128))
        self.targetImageLabel.setStyleSheet("background-color:white;\n"
"border:1px solid  #0072E3")
        self.targetImageLabel.setText("")
        self.targetImageLabel.setObjectName("targetImageLabel")
        self.verticalLayout_6.addWidget(self.targetImageLabel)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_23 = QtWidgets.QLabel(parent=self.layoutWidget_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_23.sizePolicy().hasHeightForWidth())
        self.label_23.setSizePolicy(sizePolicy)
        self.label_23.setMinimumSize(QtCore.QSize(64, 28))
        self.label_23.setMaximumSize(QtCore.QSize(64, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_23.setFont(font)
        self.label_23.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_23.setObjectName("label_23")
        self.horizontalLayout_6.addWidget(self.label_23)
        self.targetClassLabel = QtWidgets.QLabel(parent=self.layoutWidget_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.targetClassLabel.sizePolicy().hasHeightForWidth())
        self.targetClassLabel.setSizePolicy(sizePolicy)
        self.targetClassLabel.setMinimumSize(QtCore.QSize(64, 28))
        self.targetClassLabel.setMaximumSize(QtCore.QSize(64, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.targetClassLabel.setFont(font)
        self.targetClassLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.targetClassLabel.setObjectName("targetClassLabel")
        self.horizontalLayout_6.addWidget(self.targetClassLabel)
        self.verticalLayout_6.addLayout(self.horizontalLayout_6)
        self.layoutWidget1 = QtWidgets.QWidget(parent=self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(20, 310, 139, 167))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.imgLabel_1 = QtWidgets.QLabel(parent=self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imgLabel_1.sizePolicy().hasHeightForWidth())
        self.imgLabel_1.setSizePolicy(sizePolicy)
        self.imgLabel_1.setMinimumSize(QtCore.QSize(128, 128))
        self.imgLabel_1.setMaximumSize(QtCore.QSize(128, 128))
        self.imgLabel_1.setStyleSheet("background-color:white;\n"
"border:1px solid  #0072E3")
        self.imgLabel_1.setText("")
        self.imgLabel_1.setObjectName("imgLabel_1")
        self.verticalLayout.addWidget(self.imgLabel_1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_13 = QtWidgets.QLabel(parent=self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_13.sizePolicy().hasHeightForWidth())
        self.label_13.setSizePolicy(sizePolicy)
        self.label_13.setMinimumSize(QtCore.QSize(64, 28))
        self.label_13.setMaximumSize(QtCore.QSize(64, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_13.setFont(font)
        self.label_13.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout.addWidget(self.label_13)
        self.classLabel_1 = QtWidgets.QLabel(parent=self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.classLabel_1.sizePolicy().hasHeightForWidth())
        self.classLabel_1.setSizePolicy(sizePolicy)
        self.classLabel_1.setMinimumSize(QtCore.QSize(64, 28))
        self.classLabel_1.setMaximumSize(QtCore.QSize(64, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.classLabel_1.setFont(font)
        self.classLabel_1.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.classLabel_1.setObjectName("classLabel_1")
        self.horizontalLayout.addWidget(self.classLabel_1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.layoutWidget_6 = QtWidgets.QWidget(parent=self.centralwidget)
        self.layoutWidget_6.setGeometry(QtCore.QRect(690, 79, 241, 30))
        self.layoutWidget_6.setObjectName("layoutWidget_6")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.layoutWidget_6)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_8 = QtWidgets.QLabel(parent=self.layoutWidget_6)
        self.label_8.setMinimumSize(QtCore.QSize(60, 28))
        self.label_8.setMaximumSize(QtCore.QSize(60, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_8.addWidget(self.label_8)
        self.datasetCombo = QtWidgets.QComboBox(parent=self.layoutWidget_6)
        self.datasetCombo.setMinimumSize(QtCore.QSize(120, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.datasetCombo.setFont(font)
        self.datasetCombo.setObjectName("datasetCombo")
        self.horizontalLayout_8.addWidget(self.datasetCombo)
        self.textEdit = QtWidgets.QTextEdit(parent=self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(480, 129, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.textEdit.setFont(font)
        self.textEdit.setStyleSheet("background-color: transparent;")
        self.textEdit.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.textEdit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.textEdit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.textEdit.setObjectName("textEdit")
        self.layoutWidget2 = QtWidgets.QWidget(parent=self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(690, 39, 241, 30))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_7 = QtWidgets.QLabel(parent=self.layoutWidget2)
        self.label_7.setMinimumSize(QtCore.QSize(60, 28))
        self.label_7.setMaximumSize(QtCore.QSize(60, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_7.addWidget(self.label_7)
        self.modelCombo = QtWidgets.QComboBox(parent=self.layoutWidget2)
        self.modelCombo.setMinimumSize(QtCore.QSize(120, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.modelCombo.setFont(font)
        self.modelCombo.setObjectName("modelCombo")
        self.horizontalLayout_7.addWidget(self.modelCombo)
        self.trueLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.trueLabel.setGeometry(QtCore.QRect(20, 199, 690, 110))
        self.trueLabel.setStyleSheet("background-color:transparent")
        self.trueLabel.setText("")
        self.trueLabel.setScaledContents(True)
        self.trueLabel.setObjectName("trueLabel")
        self.predLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.predLabel.setGeometry(QtCore.QRect(20, 198, 690, 110))
        self.predLabel.setStyleSheet("background-color:transparent\n"
"")
        self.predLabel.setText("")
        self.predLabel.setScaledContents(True)
        self.predLabel.setObjectName("predLabel")
        self.sampleBtn = QtWidgets.QPushButton(parent=self.centralwidget)
        self.sampleBtn.setGeometry(QtCore.QRect(750, 360, 180, 30))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sampleBtn.sizePolicy().hasHeightForWidth())
        self.sampleBtn.setSizePolicy(sizePolicy)
        self.sampleBtn.setMinimumSize(QtCore.QSize(0, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.sampleBtn.setFont(font)
        self.sampleBtn.setStyleSheet("QPushButton {\n"
"    border-radius: 10px;\n"
"    background-color: #0072E3;\n"
"    color: white;\n"
"}\n"
"QPushButton:hover {;\n"
"    background-color: #006dd3;\n"
"}\n"
"QPushButton:pressed {;\n"
"    background-color: #005eb6;\n"
"}")
        self.sampleBtn.setFlat(True)
        self.sampleBtn.setObjectName("sampleBtn")
        self.predBtn = QtWidgets.QPushButton(parent=self.centralwidget)
        self.predBtn.setGeometry(QtCore.QRect(750, 410, 180, 30))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predBtn.sizePolicy().hasHeightForWidth())
        self.predBtn.setSizePolicy(sizePolicy)
        self.predBtn.setMinimumSize(QtCore.QSize(0, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.predBtn.setFont(font)
        self.predBtn.setStyleSheet("QPushButton {\n"
"    border-radius: 10px;\n"
"    background-color: #0072E3;\n"
"    color: white;\n"
"}\n"
"QPushButton:hover {;\n"
"    background-color: #006dd3;\n"
"}\n"
"QPushButton:pressed {;\n"
"    background-color: #005eb6;\n"
"}")
        self.predBtn.setFlat(True)
        self.predBtn.setObjectName("predBtn")
        self.loadBtn = QtWidgets.QPushButton(parent=self.centralwidget)
        self.loadBtn.setGeometry(QtCore.QRect(750, 310, 180, 30))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loadBtn.sizePolicy().hasHeightForWidth())
        self.loadBtn.setSizePolicy(sizePolicy)
        self.loadBtn.setMinimumSize(QtCore.QSize(0, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.loadBtn.setFont(font)
        self.loadBtn.setStyleSheet("QPushButton {\n"
"    border-radius: 10px;\n"
"    background-color: #0072E3;\n"
"    color: white;\n"
"}\n"
"QPushButton:hover {;\n"
"    background-color: #006dd3;\n"
"}\n"
"QPushButton:pressed {;\n"
"    background-color: #005eb6;\n"
"}")
        self.loadBtn.setFlat(True)
        self.loadBtn.setObjectName("loadBtn")
        self.layoutWidget3 = QtWidgets.QWidget(parent=self.centralwidget)
        self.layoutWidget3.setGeometry(QtCore.QRect(730, 129, 201, 30))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.layoutWidget3)
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_25 = QtWidgets.QLabel(parent=self.layoutWidget3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_25.sizePolicy().hasHeightForWidth())
        self.label_25.setSizePolicy(sizePolicy)
        self.label_25.setMinimumSize(QtCore.QSize(100, 28))
        self.label_25.setMaximumSize(QtCore.QSize(100, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_25.setFont(font)
        self.label_25.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_25.setObjectName("label_25")
        self.horizontalLayout_9.addWidget(self.label_25)
        self.accLabel = QtWidgets.QLabel(parent=self.layoutWidget3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.accLabel.sizePolicy().hasHeightForWidth())
        self.accLabel.setSizePolicy(sizePolicy)
        self.accLabel.setMinimumSize(QtCore.QSize(60, 28))
        self.accLabel.setMaximumSize(QtCore.QSize(70, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.accLabel.setFont(font)
        self.accLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.accLabel.setObjectName("accLabel")
        self.horizontalLayout_9.addWidget(self.accLabel)
        self.layoutWidget.raise_()
        self.trueLabel.raise_()
        self.predLabel.raise_()
        self.layoutWidget.raise_()
        self.layoutWidget_2.raise_()
        self.layoutWidget_3.raise_()
        self.layoutWidget_4.raise_()
        self.layoutWidget_5.raise_()
        self.layoutWidget.raise_()
        self.layoutWidget_6.raise_()
        self.textEdit.raise_()
        self.layoutWidget.raise_()
        self.sampleBtn.raise_()
        self.predBtn.raise_()
        self.loadBtn.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_15.setText(_translate("MainWindow", "类别："))
        self.classLabel_2.setText(_translate("MainWindow", "2"))
        self.label_17.setText(_translate("MainWindow", "类别："))
        self.classLabel_3.setText(_translate("MainWindow", "3"))
        self.label_19.setText(_translate("MainWindow", "类别："))
        self.classLabel_4.setText(_translate("MainWindow", "4"))
        self.label_21.setText(_translate("MainWindow", "类别："))
        self.classLabel_5.setText(_translate("MainWindow", "5"))
        self.label_23.setText(_translate("MainWindow", "类别："))
        self.targetClassLabel.setText(_translate("MainWindow", "1"))
        self.label_13.setText(_translate("MainWindow", "类别："))
        self.classLabel_1.setText(_translate("MainWindow", "1"))
        self.label_8.setText(_translate("MainWindow", "数据源："))
        self.textEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:\'Microsoft YaHei UI\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt; color:#ff0004;\">---- 真实类别</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt; color:#0055ff;\">---- 预测类别</span></p></body></html>"))
        self.label_7.setText(_translate("MainWindow", "模型："))
        self.sampleBtn.setText(_translate("MainWindow", "采  样"))
        self.predBtn.setText(_translate("MainWindow", "预  测"))
        self.loadBtn.setText(_translate("MainWindow", "加  载"))
        self.label_25.setText(_translate("MainWindow", "模型准确率："))
        self.accLabel.setText(_translate("MainWindow", "1"))
