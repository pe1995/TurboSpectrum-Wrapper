import numpy as np
import os

def conv_res(wave, flux, fwhm):
        # FWHM in mAA
	#create a file to feed to ./faltbon of spectrum that needs convolving
	f = open('original.txt', 'w')
	spud = [9.999 for i in range(len(wave))] #faltbon needs three columns of data, but only cares about the first two
	for j in range(len(wave)):
		f.write("{:f}  {:f}  {:f}\n".format(wave[j], flux[j], spud[j]))
	f.close()
	os.system("{ echo original.txt; echo convolve.txt; echo %f; echo 2; } | ./faltbon >log" % fwhm)
	wave_conv, flux_conv, spud = np.loadtxt("convolve.txt", unpack='True') #read in our new convolved spectrum
	os.remove('convolve.txt')
	os.remove('original.txt')
	#os.remove('./log')
	return wave_conv, flux_conv

def conv_macroturbulence(wave, flux, fwhm):
	#create a file to feed to ./faltbon of spectrum that needs convolving
	f = open('original.txt', 'w')
	spud = [9.999 for i in range(len(wave))] #faltbon needs three columns of data, but only cares about the first two
	for j in range(len(wave)):
		f.write("{:f}  {:f}  {:f}\n".format(wave[j], flux[j], spud[j]))
	f.close()
	os.system("{ echo original.txt; echo convolve.txt; echo %f; echo 3; } | ./faltbon > log" % -fwhm)
	wave_conv, flux_conv, spud = np.loadtxt("convolve.txt", unpack='True') #read in our new convolved spectrum
	#os.remove('convolve.txt')
	#os.remove('original.txt')
	#os.remove('./log')
	return wave_conv, flux_conv

def conv_rotation(wave, flux, fwhm):
	#create a file to feed to ./faltbon of spectrum that needs convolving
	f = open('original.txt', 'w')
	spud = [9.999 for i in range(len(wave))] #faltbon needs three columns of data, but only cares about the first two
	for j in range(len(wave)):
		f.write("{:f}  {:f}  {:f}\n".format(wave[j], flux[j], spud[j]))
	f.close()
	os.system("{ echo original.txt; echo convolve.txt; echo %f; echo 4; } | ./faltbon >log" % -fwhm)
	wave_conv, flux_conv, spud = np.loadtxt("convolve.txt", unpack='True') #read in our new convolved spectrum
	os.remove('convolve.txt')
	os.remove('original.txt')
	#os.remove('./log')
	return wave_conv, flux_conv
