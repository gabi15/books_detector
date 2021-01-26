from PyQt5 import QtCore, QtGui, QtWidgets
from books_detector.detect import detect
import sys
import cv2


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1121, 733)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(570, 0, 551, 681))
        self.listWidget.setObjectName("widgetList")
        self.chooseFileBtn = QtWidgets.QPushButton(self.centralwidget)
        self.chooseFileBtn.setGeometry(QtCore.QRect(150, 90, 261, 51))
        self.chooseFileBtn.setObjectName("chooseFileBtn")
        self.getBooksBtn = QtWidgets.QPushButton(self.centralwidget)
        self.getBooksBtn.setGeometry(QtCore.QRect(150, 190, 261, 51))
        self.getBooksBtn.setObjectName("uploadFileBtn")
        self.inputphoto = QtWidgets.QFrame(self.centralwidget)
        self.inputphoto.setGeometry(QtCore.QRect(0, 330, 571, 351))
        self.inputphoto.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.inputphoto.setFrameShadow(QtWidgets.QFrame.Raised)
        self.inputphoto.setObjectName("inputphoto")
        self.imgLabel = QtWidgets.QLabel(self.inputphoto)
        self.imgLabel.setGeometry(QtCore.QRect(10, 10, 551, 331))
        self.imgLabel.setObjectName("imgLabel")

        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 320, 571, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(0, 670, 1121, 20))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(80, 10, 401, 71))
        self.label.setObjectName("label")
        self.fileNameLabel = QtWidgets.QLabel(self.centralwidget)
        self.fileNameLabel.setGeometry(QtCore.QRect(80, 160, 400, 15))
        self.fileNameLabel.setObjectName("fileNameLabel")
        self.bookCountWidget = QtWidgets.QLabel(self.centralwidget)
        self.bookCountWidget.setGeometry(QtCore.QRect(80, 260, 400, 15))
        self.bookCountWidget.setObjectName("bookCountWidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1121, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.chooseFileBtn.setText(_translate("MainWindow", "Wybierz zdjęcie"))
        self.getBooksBtn.setText(_translate("MainWindow", "Wyświetl listę książek"))
        self.label.setText(_translate("MainWindow", "Dodaj zdjęcie swojej półki z książkami, aby uzyskać listę ich tytułów!"))
        self.fileNameLabel.setText(_translate("MainWindow", "--filename--"))
        self.bookCountWidget.setText(_translate("MainWindow", "Ciekawe ile ksiazek uda nam się odnalezc..."))
        self.imgLabel.setText(_translate("MainWindow", "img placeholder"))


class AppWindow(Ui_MainWindow):
    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)
        window = QtWidgets.QMainWindow()
        self.setupUi(window)

        self.chooseFileBtn.clicked.connect(self.getPhoto)
        self.getBooksBtn.clicked.connect(self.processPhoto)

        self.fileName = ''

        window.show()
        sys.exit(app.exec_())

    def getPhoto(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "QFileDialog.getOpenFileName()",
            "",
            "Image files (*.jpg *.png)",
            options=options)

        self.fileName = fileName
        self.image = cv2.imread(fileName)
        h, w, c = self.image.shape
        if h < 2999:
            print("za male zdjecie")
            self.showPopup()
        else:
            self.fileNameLabel.setText(fileName)
            self.showPhoto(fileName)

    def showPopup(self):
        self.msg = QtWidgets.QMessageBox()
        self.msg.setIcon(QtWidgets.QMessageBox.Warning)
        self.msg.setWindowTitle("Odrzucono zdjecie")
        self.msg.setText("Minimalna wysokosc zdjecia to 3000px!")
        x = self.msg.exec_()

    def showPhoto(self, fileName):
        pixmap = QtGui.QPixmap(fileName)
        if pixmap.height() > pixmap.width():
            pixmap = pixmap.scaledToHeight(330)
        else:
            pixmap = pixmap.scaledToWidth(570)
        self.imgLabel.setPixmap(pixmap)
        self.listWidget.clear()
        self.bookCountWidget.setText("Ciekawe ile ksiazek uda nam się odnalezc...")

    def processPhoto(self):
        entries = detect(self.fileName)
        self.bookCountWidget.setText("Na twoim zdjeciu udalo nam się znalezc " + str(len(entries)) + " ksiazek!")
        self.listWidget.addItems(entries)


