
import random,sys
from Crypto.Cipher import AES
import os

import numpy as np
import timeit
from secrets import token_bytes 

# A function to generate a random permutation of arr[] 
class Krtio_Permutations():
	
	def __init__(self, file_name,max_ovls):
		
		self.file_name = file_name

		#key lenght =32*8=256
		key= token_bytes(32)
		if ( os.path.exists('aeskey.bin')):
			with open('aeskey.bin', 'rb') as keyfile:
				key = keyfile.read()
		else:
			with open('aeskey.bin', 'wb') as keyfile:
				keyfile.write(key)

		#Random vector for encryption
		iv= token_bytes(16)
		if (os.path.exists('iv.bin')):
			with open('iv.bin', 'rb') as keyfile:
				iv = keyfile.read()
		else:
			with open('iv.bin', 'wb') as keyfile:
				keyfile.write(iv)

		self.key=key
		self.iv=iv
		self.max_ovls=max_ovls

	def randomize_permutation (self,n,ov_id): 
		arr=np.arange(n)
		# Start from the last element and swap one by one. We don't 
		# need to run for the first element that's why i > 0 
		for i in range(n-1,0,-1): 
		# Pick a random index from 0 to i 
			#j = random.randint(0,i+1) 
			j = self.genrate_random_number(i+1,ov_id)

			# Swap arr[i] with the element at random index 
			arr[i],arr[j] = arr[j],arr[i] 
		return arr 

	def genrate_random_number(self,i,ov_id):
		aes = AES.new(self.key, AES.MODE_CFB,self.iv)
		
		data=str.encode(self.file_name)+b'\x00'+str.encode(str(i))+b'\x00'+str.encode(str(ov_id))
		ciphertext = aes.encrypt(data)
		rn=int.from_bytes(ciphertext, byteorder='big', signed=True) 
		return (rn %i)

	def select_ovlays(self,k):
		aes = AES.new(self.key, AES.MODE_CFB,self.iv)
		ov_ids,j=[],0
		while len(ov_ids)<k:
			
			data=str.encode(self.file_name)+b'\x00'+str.encode(str(j))
			j=j+1
			ciphertext = aes.encrypt(data)
			rn=int.from_bytes(ciphertext, byteorder='big', signed=True) 
			id_o=rn % self.max_ovls
			if not (id_o in ov_ids):
				ov_ids=ov_ids+[id_o]
		return ov_ids

	
	



