# coding=utf-8
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtWidgets import QMessageBox, QPushButton, QLabel, QVBoxLayout, QTextEdit,QComboBox
from PyQt5.QtCore import Qt


class Ui_TabWidget(object):

    def setupUi(self, TabWidget):
        TabWidget.setObjectName("TabWidget")
        TabWidget.resize(789, 619)
        self.setWindowIcon(QIcon('car.png'))

        # 通过设置组件要放置的位置来进行放置
        # "第1个子窗口"
        self.tab1 = QtWidgets.QWidget()
        self.tab1.setObjectName("tab1")
        TabWidget.addTab(self.tab1, "")

        # 第2个窗口
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        TabWidget.addTab(self.tab_2, "")

        vbox_1 = QVBoxLayout()
        vbox_1.setAlignment(Qt.AlignCenter)
        label_logo = QLabel(self.tab1)
        label_logo.setAlignment(Qt.AlignCenter)
        label_logo.setPixmap(QPixmap("ui_imgs/smart_car.png"))
        label_text = QLabel('车辆检索系统\n\n 作者：西安交通大学软件学院', self.tab1)
        label_text.setAlignment(Qt.AlignCenter)
        vbox_1.addWidget(label_logo)
        vbox_1.addWidget(label_text)
        self.tab1.setLayout(vbox_1)

        vbox_2 = QVBoxLayout()
        vbox_2.setAlignment(Qt.AlignCenter)
        self.label_img = QLabel(self.tab_2)
        self.label_result = QLabel(self.tab_2)
        self.pushButton_open = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_open.setGeometry(QtCore.QRect(10, 30, 75, 30))
        self.pushButton_open.setObjectName("infoButton")
        self.label_img.setAlignment(Qt.AlignCenter)
        vbox_2.addWidget(self.label_img)
        vbox_2.addWidget(self.label_result)
        vbox_2.addWidget(self.pushButton_open)
        self.tab_2.setLayout(vbox_2)

        # 给组件命名，设置组件的相关内容
        self.retranslateUi(TabWidget)
        TabWidget.setCurrentIndex(0)
        self.pushButton_open.clicked.connect(self.open_file)
        QtCore.QMetaObject.connectSlotsByName(TabWidget)

    # 设置按键内容
    def retranslateUi(self, TabWidget):
        _translate = QtCore.QCoreApplication.translate
        TabWidget.setWindowTitle(_translate("TabWidget", "车辆尾气检测-测试"))
        self.pushButton_open.setText(_translate("TabWidget", "打开图片"))
        TabWidget.setTabText(TabWidget.indexOf(self.tab1), _translate("TabWidget", "主页"))
        TabWidget.setTabText(TabWidget.indexOf(self.tab_2), _translate("TabWidget", "图片测试"))

    def closeEvent(self, event):
        reply = QMessageBox.question(self, '提示',
                                     "您确定退出吗？", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    pass
    # app = QtWidgets.QApplication(sys.argv)
    # window = mywindow()
    # window.show()
    # sys.exit(app.exec_())


