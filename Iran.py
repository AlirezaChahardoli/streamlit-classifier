from PyQt5.QtWidgets import QApplication,QLabel 
import sys
app=QApplication(sys.argv)
QLabel("Qt is OK").show()
sys.exit(app.exec_())