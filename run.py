import sys
import time
import requests
from PyQt5 import QtGui
from PyQt5.QtChart import QLineSeries
from PyQt5.QtCore import pyqtSlot, QPointF
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox, QFileDialog

from MainWindow import Ui_Widget
from model import Model


class ControlApp(QDialog):
	def __init__(self):
		super(ControlApp, self).__init__()

		self.open_video = False
		self.today_max = 0

		self.video_file = ""

		self._ui = Ui_Widget()
		self._ui.setupUi(self)

		self.model = Model()

		self._ui.pushButton.clicked.connect(self.send_connect_signal)
		self._ui.pushButton_2.clicked.connect(self.start_pic)
		self._ui.pushButton_3.clicked.connect(self.stop_pic)
		self._ui.pushButton_4.clicked.connect(self.select_file)  # 选择文件

		self.model.is_connect_signal.connect(self.connect_camera)
		self.model.pic_signal.connect(self.print_pic)
		self.model.num_sum_signal.connect(self.print_info)
		self.model.timer_video.timeout.connect(self.model.make_pic)

		self.seriesArray = [0] * 10
		self.series = QLineSeries()
		for i in range(10):
			self.series.append(QPointF(i, 0))
		self.series.setName("近期人数折线图")
		# 初始化折线plot
		self._ui.graphicsView.chart().addSeries(self.series)
		self._ui.graphicsView.chart().createDefaultAxes()
		self._ui.graphicsView.chart().axisY().setMax(1)
		self._ui.graphicsView.chart().axisY().setMin(0)

		self.mytime = time.time()

	@pyqtSlot(object)
	def print_pic(self, result):
		# 画图在Label上
		self._ui.label.clear()
		img = QtGui.QImage(result.data, result.shape[1], result.shape[0], QtGui.QImage.Format_RGB888)
		self._ui.label.setPixmap(QtGui.QPixmap.fromImage(img))
		# 禁用/开放按钮
		self._ui.pushButton_2.setEnabled(False)
		self._ui.pushButton_3.setEnabled(True)

	@pyqtSlot()
	def send_connect_signal(self):
		self.model.connect_camera(self.video_file)

	@pyqtSlot(bool)
	def connect_camera(self, is_connect):
		# 弹出提示框
		if is_connect:
			msg_box = QMessageBox(QMessageBox.Information, '信息提示', "成功")
		else:
			msg_box = QMessageBox(QMessageBox.Warning, '信息提示', "失败")
		msg_box.exec_()
		self.video_file = ""
		self._ui.pushButton.setText("连接摄像头")

		# 禁用/开放按钮
		self._ui.pushButton.setEnabled(False)
		self._ui.pushButton_2.setEnabled(True)

	@pyqtSlot()
	def stop_pic(self):
		self.open_video = False
		# 关闭视频流
		self.model.timer_video.stop()
		# 禁用/开放按钮
		self._ui.pushButton.setEnabled(True)
		self._ui.pushButton_2.setEnabled(True)
		self._ui.pushButton_3.setEnabled(False)

	@pyqtSlot()
	def start_pic(self):
		if not self.open_video:
			self.model.timer_video.start(30)
		self.open_video = True
		self.model.make_pic()

	@pyqtSlot(int)
	def print_info(self, num):
		mytime = time.time()
		if mytime - self.mytime > 1:
			# 当前人数
			self._ui.lcdNumber.display(num)
			# 当日最大人数
			if num > self.today_max:
				self.today_max = num
				self._ui.lcdNumber_2.display(self.today_max)
			# 控制台信息
			self._ui.textEdit.append("{}   ---   {}人".format(time.asctime(time.localtime(mytime)), num))
			self._ui.textEdit.moveCursor(QTextCursor.End)
			# 自动滚动文本区域
			sb = self._ui.textEdit.verticalScrollBar()
			sb.setValue(sb.maximum())
			# 画图
			self.seriesArray.pop(0)
			self.seriesArray.append(num)
			self.series.replace([QPointF(i, self.seriesArray[i]) for i in range(10)])
			if max(self.seriesArray) > 0:
				self._ui.graphicsView.chart().axisY().setMax(max(self.seriesArray))
			# 更新时间
			res = requests.get(url='http://127.0.0.1:8080/index/insertPersonNum',
							   params={"num": num,
									   "area":1})
			self.mytime = mytime

	@pyqtSlot()
	def select_file(self):
		# 选择单个文件
		filename, _ = QFileDialog.getOpenFileName(self, "选取要分析的文件", "D:\\", "Vedio Files (*.mp4)")
		self.video_file = filename
		self._ui.pushButton.setText("加载视频")


if __name__ == '__main__':
	app = QApplication(sys.argv)
	window = ControlApp()
	window.show()
	sys.exit(app.exec_())
