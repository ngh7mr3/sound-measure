import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import socket

class Plot():
	def __init__(self, figure, name, x, y, z):
		self.name = name
		self.ax = figure.add_subplot(2, z,int(x+y,2)+1)
		# self.ax = figure.subplot(gs[x, y])
		self.data = []
		self.xs = []
		self.ax.set_title(self.name)
		self.ax.plot(self.xs, self.data)

	def update(self, t):
		self.data+=t
		size = len(self.xs)
		self.xs+=[i for i in range(size, len(self.data))]
		self.ax.clear()
		self.ax.set_title(self.name)
		self.ax.plot(self.xs, self.data)

class Plotter(object):
	def __init__(self, names, freqs):
		self.fig = plt.figure()
		self.names = names
		self.freqs = freqs
		self.plots = []
		for a, name in enumerate(self.names):
			for b, freq in enumerate(self.freqs):
				self.plots.append(Plot(self.fig, name+'_'+freq, str(a), str(b), len(self.names)))
		self.new_data = 0
		self.connection = None

	def animate(self, u):
		try:
			self.new_data = self.connection.recv(1024)
			
			tr = [[float(i) for i in x.split(',') if i!=''] for x in self.new_data.decode("utf-8").split('|') if x!='']
			tr_T = list(map(list, zip(*tr)))
			if self.new_data:
				self.new_data = 0
				for block, p in zip(tr_T, self.plots):
						p.update(block)
		except Exception as e:
			print("Blocked by", e)

	def run(self):
		print("start def run")
		ani = animation.FuncAnimation(self.fig, self.animate, interval=1000)
		plt.show()

if __name__ == "__main__":
	
	sock = socket.socket()
	
	if len(sys.argv) < 4:
		print("Error: include all arguments\n\
Usage: $ plotter.exe [port] [plots_names(string, sep. by comma)] [plots_freqs(string, sep. by comma)]")
		exit(1)

	print(sys.argv)

	sock.bind(('localhost', int(sys.argv[1])))
	sock.listen(1)
	print("listening on", sys.argv[1])

	conn, addr = sock.accept()
	conn.settimeout(0)
	print(conn, addr)

	plot = Plotter(sys.argv[2].split(','), sys.argv[3].split(','))
	plot.connection = conn

	plot.run()
	conn.close()
