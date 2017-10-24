"""
函数说明:梯度上升算法测试函数

求函数f(x) = -x^2 + 4x的极大值

Parameters:
    无
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-08-28
Note :
    ZJ studied in 2017-10-24
"""
def gradient_ascent_test():
	# f(x) = -x^2 + 4x ; f(x) 的导数 f'(x)= -2x+4
	def f_prime(x_old):
		return -2 * x_old + 4
	#初始值，给一个小于x_new的值
	x_old = -1
	#梯度上升算法初始值，即从(0,0)开始
	x_new = 0
	#步长，也就是学习速率，控制更新的幅度
	alpha = 0.01
	#精度，也就是更新阈值
	presision = 0.00000001 
	while abs(x_new - x_old) > presision:
		x_old = x_new
		# Xi := Xi + alpha * (导数f'(x))
		x_new =x_old + alpha * f_prime(x_old)
	print(x_new)

if __name__ == '__main__':
	gradient_ascent_test() 
	# 1.999999515279857 也就是 x 为 1.999999515279857 四舍五入2 时 可以取得函数最大值