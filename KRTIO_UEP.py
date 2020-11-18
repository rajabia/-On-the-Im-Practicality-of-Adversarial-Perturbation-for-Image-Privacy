

import numpy as np
import os,sys
import cv2
from PIL import Image
from matplotlib import cm
import shutil
import glob
from sewar.full_ref import uqi,ssim,msssim,mse
import matplotlib.pyplot as plt
import matplotlib

import argparse

from EncriptionWithAES   import Krtio_Permutations
import pickle


def permut_blocks(b,shape,Krtio_Perm,ovs_id,ovlays_add):
	b1=int(np.ceil(shape[0]/b))
	b2=int(np.ceil(shape[1]/b))
	add_file=glob.glob(ovlays_add+'/o'+str(ovs_id+1)+'.*')
	if len(add_file)==0:
		print('Error: Could not find overlay images '+ovlays_add+'/o'+str(ovs_id+1))
		quit()

	img=Image.open(add_file[0])

	img=np.asarray(img.resize((b2*b,b1*b)))
	new_img=np.copy(img)
	per=Krtio_Perm.randomize_permutation (b1*b2,ovs_id)
	
	for i,p in enumerate(per):
		p1=int(p//b2)
		p2=int(p%b2)
		i1=int(i//b2)
		i2=int(i%b2)
		c=img[p1*b:(p1+1)*b,p2*b:(p2+1)*b,:]
	

		new_img[i1*b:(i1+1)*b,i2*b:(i2+1)*b,:]=img[p1*b:(p1+1)*b,p2*b:(p2+1)*b,:]

	

	return new_img[:shape[0],:shape[1],:]


def Ovs_Generation(image_name,image_shape,N,k,b,ovlays_add):
	
	Krtio_Perm=Krtio_Permutations(image_name,N)
	id_ovs=Krtio_Perm.select_ovlays(k)
	ov=[]
	for i in id_ovs:
		temp=np.array(permut_blocks(b,image_shape,Krtio_Perm,i,ovlays_add),np.float)
		o=Image.fromarray(np.uint8(temp))

		if len(temp.shape)>3:
			temp=temp[:3]
			
		if temp.shape[2]==1:
			temp=np.array([temp,temp,temp])
			temp=np.transpose(temp,(2,0,1))
		
		if len(ov)==0:
			ov=temp
		else:
			ov=ov+temp
	return ov/float(k)


def compare(ImageAPath, ImageBPath):
	img1 = cv2.imread(ImageAPath)          
	img2 = cv2.imread(ImageBPath)
	# m = mse(img1, img2)
	s = ssim(img1, img2)
	
	return s

def repair(img,box=8):
	indx=np.where((img<5) | (img>250))
	indx=np.array(indx)
	
	for p in range(indx.shape[1]):
		i,j,c=indx[0,p],indx[1,p],indx[2,p] 
		if i>box and j>box and i<img.shape[0]-box and j<img.shape[1]-box:
			img[i,j,c]=np.mean(img[i-box:i+box,j-box:j+box,c])
		elif i>box/2 and j>box and i<img.shape[0]-int(box/2) and j<img.shape[1]-int(box/2):
			img[i,j,c]=np.mean(img[i-int(box/2):i+int(box/2),j-int(box/2):j+int(box/2),c])
		elif i>box/4 and j>box and i<img.shape[0]-int(box/4) and j<img.shape[1]-int(box/4):
			img[i,j,c]=np.mean(img[i-int(box/4):i+int(box/4),j-int(box/4):j+int(box/4),c])
		elif i<box or j<box or i==img.shape[0] or j==img.shape[1]:
			l1,l2=max(i-box,0),max(j-box,0)
			u1,u2=min(i+box,img.shape[0]),min(j+box,img.shape[1])
			img[i,j,c]=np.mean(img[i-l1:i+u1,j-l2:j+u2,c])
		
	for p in range(indx.shape[1]):
		if i<2*box or j<2*box or i==img.shape[0] or j==img.shape[1]:
			l1,l2=max(i-box,0),max(j-box,0)
			u1,u2=min(i+box,img.shape[0]),min(j+box,img.shape[1])
			img[i,j,c]=np.mean(img[i-l1:i+u1,j-l2:j+u2,c])
		
	
	return img

def list_files(address):
	imges_formats=['.jpeg','.jpg','.png','.JPEG','.JPG', '.PNG']
	files=glob.glob(address+'/*')
	files_final=[]
	for f in files:
		flag=len([fr for fr in imges_formats  if fr in f ])>0
		if flag:
			files_final.append(f)
	return files_final


def main():
	parser = argparse.ArgumentParser(description='KRTIO or UEP')

	parser.add_argument('--method', type=str, default='KRTIO',help='UEP or KRTIO')
	parser.add_argument('--mode', type=str, default='Enc',help='Enc or Dec')
	parser.add_argument('--alpha', type=float, default=0.45,help='alpha value for krtio')
	parser.add_argument('--overlays_folder', type=str, default='./overlays',help='path of overlays folder')
	parser.add_argument('--k', type=int, default=3, help='Number of overlay')
	parser.add_argument('--beta', type=int, default=3, help='beta values for UEP') 
	parser.add_argument('--input', type=str, default='./InputFiles', help='Input Images/ folder or an images') 
	parser.add_argument('--bl_size', type=int, default=16, help='Block size') 
	args = parser.parse_args()

	b=args.bl_size

	if not (os.path.exists('./output')):
		os.mkdir('./output')

	if not (args.mode == 'Enc' or args.mode == 'Dec'):
		print('Unknown mode error. Mode should be Enc or Dec')
		quit()
	if not (args.method == 'KRTIO' or args.method == 'UEP'):
		print('Unknown method error. method should be KRTIO or UEP')
		quit()

	
	imges_formats=['.jpeg','.jpg','.png','.JPEG','.JPG', '.PNG']
	flag=len([fr for fr in imges_formats  if fr in args.input ])>0

	if flag :
		inputs=[Image.open(args.input)]
	else:
		inputs=list_files(args.input)

	ovlays_add=list_files(args.overlays_folder)
	N=len(ovlays_add)
	dict_faces={}
	UEP=[]
	if (os.path.exists('mod.npy') and args.method == 'UEP'):
		UEP=np.load('mod.npy')
		print('UEP perturbation was loaded')
	if (args.method == 'UEP' and os.path.exists('facelocation.pkl')):
		f = open("facelocation.pkl","rb")
		dict_faces=pickle.load(f)
		f.close()

	for f in inputs:
		if args.method == 'KRTIO':
			print('processing the file:  ...  '+f)
			main_image=Image.open(f)
			main_image=np.array(main_image,np.float)
			main_image_shape=main_image.shape
			ind=[i for i in range(len(f)) if (f[i]=='/' or f[i]=='\\')]
			if len(ind)==0:
				ind=[-1]
			if len(main_image_shape)>3:
				main_image_shape=main_image_shape[:3]

			if main_image_shape[2]==1:
				main_image=np.array([main_image,main_image,main_image])
				main_image=np.transpose(main_image,(2,0,1))

			
			ovs=Ovs_Generation(f[ind[-1]:],main_image_shape,N,args.k,b,args.overlays_folder)
			if args.mode=='Enc':
				krtioImage=args.alpha*main_image+(1-args.alpha)*ovs
				krtioImage=Image.fromarray(np.uint8(np.round(krtioImage)))
				print('Saving in ... '+'./output/'+f[ind[-1]+1:])
				krtioImage.save('./output/'+f[ind[-1]+1:])
			else:
				krtioImage=(1/args.alpha)*(main_image-(1-args.alpha)*ovs)
				#face=(1/alpha)*(krtioImage-(1-alpha)ovs)
				krtioImage=repair(krtioImage)
				recovered=Image.fromarray(np.uint8(np.round(krtioImage)))
				print('Saving in ... '+'./output/'+f[ind[-1]+1:])
				recovered.save('./output/'+f[ind[-1]+1:])
		else:
			face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
			img = cv2.imread(f)
			
			ind=[i for i in range(len(f)) if (f[i]=='/' or f[i]=='\\')]
			if len(ind)==0:
				ind=[-1]
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			if  args.mode=='Enc':
				if len(UEP)==0:
					print('Error: Could not find uep perturbation file')
					quit()
				faces = face_cascade.detectMultiScale(gray, 1.1, 4)
				dict_faces[f[ind[-1]+1:]]=faces
				for (x, y, w, h) in faces:
					#cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
					
					dsize = ( h,w)

					# resize image
					pert = cv2.resize(UEP, dsize, interpolation = cv2.INTER_AREA)

					face=(img[y:y+w,x:x+h,:]/255.0)
					
					face=np.tanh((np.arctanh((face-0.5)*1.999999999999)+args.beta*pert))/2+0.5
					face=np.round(face*255).astype(int)
					img[y:y+w,x:x+h,:]=face


				cv2.imshow('img', img)
				cv2.waitKey(2000)
				cv2.imwrite('./output/'+f[ind[-1]+1:],img)
			else:
				faces=dict_faces[f[ind[-1]+1:]]
				img = cv2.imread(f)
				for (x, y, w, h) in faces:
					recoverd= np.asarray(img[y:y+w,x:x+h,:])/255.0
					dsize = ( h,w)
					# resize image
					pert = cv2.resize(UEP, dsize, interpolation = cv2.INTER_AREA)
					recoverd=np.tanh((np.arctanh((recoverd-0.5)*1.999999999999)-args.beta*pert))/2+0.5
					recoverd_b=np.round(recoverd*255).astype(int)
					recoverd_b=repair(recoverd_b,box=8)
					recoverd=np.uint8(recoverd_b)
					img[y:y+w,x:x+h,:]=recoverd

				img=repair(img,box=4)
				cv2.imshow('img', img)
				cv2.waitKey(2000)
				print('Saving in ... '+'./output/'+f[ind[-1]+1:])
				cv2.imwrite('./output/'+f[ind[-1]+1:],img)

	if (args.method=='UEP' and args.mode=='Enc'):	
		f = open("facelocation.pkl","wb")
		pickle.dump(dict_faces,f)
		f.close()


				

if __name__ == '__main__':
	main()
	