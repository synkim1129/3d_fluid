import torch
import numpy as np
from PIL import Image
from get_param import params

"""
ask-tell interface:
ask(): ask for batch of v_cond(t),cond_mask(t),v(t),p(t)
tell(v,p): tell results for v(t+1),p(t+1) of batch
"""
submarine = torch.tensor(np.load('imgs/voxel_grid_Submarine.npy')).unsqueeze(0)
fish = torch.tensor(np.load('imgs/voxel_grid_Fish.npy')).unsqueeze(0)
cyber = torch.tensor(np.load('imgs/voxel_grid_Cyber.npy')).unsqueeze(0)
wing = torch.tensor(np.load('imgs/voxel_grid_Wing.npy')).unsqueeze(0)
two_objects = torch.tensor(np.load('imgs/voxel_grid_2_objects.npy')).unsqueeze(0)
three_objects = torch.tensor(np.load('imgs/voxel_grid_3_objects.npy')).unsqueeze(0)
img_dict = {"submarine":submarine,"fish":fish,"cyber":cyber,"wing":wing,"2_objects":two_objects,"3_objects":three_objects} # here, you can add your own custom objects

rod_size=8

class Dataset:
	def __init__(self,w,h,d,batch_size=100,dataset_size=1000,average_sequence_length=5000,interactive=False,max_speed=3,brown_damping=0.9995,brown_velocity=0.005,init_velocity=0,dt=1,types=["box"],images=["suzanne"],mu=1):
		"""
		:w,h,d: width, height, depth (sizes in x,y,z direction)
		:types: possibilities: "no_box","box","rod_y","rod_z","moving_rod_y","moving_rod_z","magnus_y","magnus_z","ball","image","benchmark"
		:images: if type is image, then there are the following possibilities: see img_dict
		"""
		self.w,self.h,self.d = w,h,d
		self.batch_size = batch_size
		self.dataset_size = dataset_size
		self.average_sequence_length = average_sequence_length
		self.v_cond = torch.zeros(dataset_size,3,w,h,d)
		self.p_cond = torch.zeros(dataset_size,1,w,h,d)
		self.T_cond = torch.zeros(dataset_size,1,w,h,d)

		self.cond_mask = torch.zeros(dataset_size,1,w,h,d)
		self.bc_mask = torch.zeros(dataset_size,1,w,h,d)
		self.env_info = [{} for _ in range(dataset_size)]
		self.interactive = interactive
		self.interactive_spring = 150#300#200#~ 1/spring constant to move object
		self.max_speed = max_speed
		self.brown_damping = brown_damping
		self.brown_velocity = brown_velocity
		self.init_velocity = init_velocity
		self.dt = dt
		self.types = types
		self.images = images
		
		self.mousex = 0
		self.mousey = 0
		self.mousez = 0
		self.mousev = 0
		self.mousew = 0
	
		self.v   = torch.zeros(dataset_size,3,w,h,d)
		self.p   = torch.zeros(dataset_size,1,w,h,d)
		self.T   = torch.zeros(dataset_size,1,w,h,d)

		for i in range(dataset_size):
			self.reset_env(i)
		
		self.t = 0
		self.i = 0

	def reset_env(self,index):
		
		self.v[index]   = 0

		self.cond_mask[index]=0
		self.cond_mask[index,:,0:5,:,:]=1
		self.cond_mask[index,:,(self.w-5):self.w,:,:]=1
		self.cond_mask[index,:,:,0:3,:]=1
		self.cond_mask[index,:,:,(self.h-3):self.h,:]=1
		self.cond_mask[index,:,:,:,0:3]=1
		self.cond_mask[index,:,:,:,(self.d-3):self.d]=1

		self.bc_mask[index]=0
		self.bc_mask[index,:,0:5,:,:]=1
		self.bc_mask[index,:,(self.w-5):self.w,:,:]=1
		self.bc_mask[index,:,:,0:3,:]=1
		self.bc_mask[index,:,:,(self.h-3):self.h,:]=1
		self.bc_mask[index,:,:,:,0:3]=1
		self.bc_mask[index,:,:,:,(self.d-3):self.d]=1

		type = np.random.choice(self.types)

#{{{types
		
		if type == "magnus_z":
			flow_v = self.max_speed*(np.random.rand()-0.5)*2
			
			if flow_v>0:
				object_x = np.random.randint(self.w//4-10,self.w//4+10)
			else:
				object_x = np.random.randint(3*self.w//4-10,3*self.w//4+10)
			object_y = np.random.randint(self.h//2-10,self.h//2+10)
			object_vx = self.init_velocity*(np.random.rand()-0.5)*2
			object_vy = self.init_velocity*(np.random.rand()-0.5)*2
			object_r = np.random.randint(5,15) # object radius (15)
			object_w = self.max_speed*(np.random.rand()-0.5)*2/object_r # object angular velocity (3/object_r)
			
			# 1. generate mesh 2 x [2r x 2r]
			y_mesh,x_mesh = torch.meshgrid([torch.arange(-object_r,object_r+1),torch.arange(-object_r,object_r+1)])
			
			# 2. generate mask
			mask_ball = ((x_mesh**2+y_mesh**2)<object_r**2).float().unsqueeze(0).unsqueeze(3)
			
			# 3. generate v_cond and multiply with mask
			v_ball = object_w*torch.cat([x_mesh.unsqueeze(0),-y_mesh.unsqueeze(0)]).unsqueeze(3)*mask_ball
			
			# 4. add masks / v_cond
			self.cond_mask[index,:,(object_x-object_r):(object_x+object_r+1),(object_y-object_r):(object_y+object_r+1),3:(self.d-3)] += mask_ball
			self.v_cond[index,0,(object_x-object_r):(object_x+object_r+1),(object_y-object_r):(object_y+object_r+1),3:(self.d-3)] = v_ball[0]+object_vx
			self.v_cond[index,1,(object_x-object_r):(object_x+object_r+1),(object_y-object_r):(object_y+object_r+1),3:(self.d-3)] = v_ball[1]+object_vy
			
			self.v_cond[index,0,0:5,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index,0,(self.w-5):self.w,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
			
			self.env_info[index]["type"] = type
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["r"] = object_r
			self.env_info[index]["w"] = object_w
			self.env_info[index]["flow_v"] = flow_v
			self.mousex = object_x
			self.mousey = object_y
			self.mousev = flow_v
			self.mousew = object_w*object_r
		
		if type == "magnus_y":
			flow_v = self.max_speed*(np.random.rand()-0.5)*2
			
			if flow_v>0:
				object_x = np.random.randint(self.w//4-10,self.w//4+10)
			else:
				object_x = np.random.randint(3*self.w//4-10,3*self.w//4+10)
			object_z = np.random.randint(self.d//2-10,self.d//2+10)
			object_vx = self.init_velocity*(np.random.rand()-0.5)*2
			object_vz = self.init_velocity*(np.random.rand()-0.5)*2
			object_r = np.random.randint(5,15) # object radius (15)
			object_w = self.max_speed*(np.random.rand()-0.5)*2/object_r # object angular velocity (3/object_r)
			
			# 1. generate mesh 2 x [2r x 2r]
			z_mesh,x_mesh = torch.meshgrid([torch.arange(-object_r,object_r+1),torch.arange(-object_r,object_r+1)])
			
			# 2. generate mask
			mask_ball = ((x_mesh**2+z_mesh**2)<object_r**2).float().unsqueeze(0).unsqueeze(2)
			
			# 3. generate v_cond and multiply with mask
			v_ball = object_w*torch.cat([x_mesh.unsqueeze(0),-z_mesh.unsqueeze(0)]).unsqueeze(2)*mask_ball
			
			# 4. add masks / v_cond
			self.cond_mask[index,:,(object_x-object_r):(object_x+object_r+1),3:(self.h-3),(object_z-object_r):(object_z+object_r+1)] += mask_ball
			self.v_cond[index,0,(object_x-object_r):(object_x+object_r+1),3:(self.h-3),(object_z-object_r):(object_z+object_r+1)] = v_ball[0]+object_vx
			self.v_cond[index,2,(object_x-object_r):(object_x+object_r+1),3:(self.h-3),(object_z-object_r):(object_z+object_r+1)] = v_ball[1]+object_vz
			
			self.v_cond[index,0,0:5,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index,0,(self.w-5):self.w,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
			
			self.env_info[index]["type"] = type
			self.env_info[index]["x"] = object_x
			self.env_info[index]["z"] = object_z
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vz"] = object_vz
			self.env_info[index]["r"] = object_r
			self.env_info[index]["w"] = object_w
			self.env_info[index]["flow_v"] = flow_v
			self.mousex = object_x
			self.mousez = object_z
			self.mousev = flow_v
			self.mousew = object_w*object_r
			
		if type == "no_box":# most simple env
			flow_v = self.max_speed*(np.random.rand()-0.5)*2
			
			self.v_cond[index,0,0:5,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index,0,(self.w-5):self.w,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
			
			self.env_info[index]["type"] = type
			self.env_info[index]["flow_v"] = flow_v
			self.mousev = flow_v
		
		if type == "rod_y":# no_box with rod (to see, if vortex street appears...)
			flow_v = self.max_speed*(np.random.rand()-0.5)*2
			
			if flow_v > 0:
				self.cond_mask[index,:,(self.w//4-rod_size):(self.w//4+rod_size),:,(self.d//2-rod_size):(self.d//2+rod_size)]=1
			else:
				self.cond_mask[index,:,(3*self.w//4-rod_size):(3*self.w//4+rod_size),:,(self.d//2-rod_size):(self.d//2+rod_size)]=1
			
			self.v_cond[index,0,0:5,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index,0,(self.w-5):self.w,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
			
			self.env_info[index]["type"] = type
			self.env_info[index]["flow_v"] = flow_v
			self.mousev = flow_v
		
		if type == "rod_z":# no_box with rod (to see, if vortex street appears...)
			flow_v = self.max_speed*(np.random.rand()-0.5)*2
			
			if flow_v > 0:
				self.cond_mask[index,:,(self.w//4-rod_size):(self.w//4+rod_size),(self.h//2-rod_size):(self.h//2+rod_size),:]=1
			else:
				self.cond_mask[index,:,(3*self.w//4-rod_size):(3*self.w//4+rod_size),(self.h//2-rod_size):(self.h//2+rod_size),:]=1
			
			self.v_cond[index,0,0:5,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index,0,(self.w-5):self.w,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
			
			self.env_info[index]["type"] = type
			self.env_info[index]["flow_v"] = flow_v
			self.mousev = flow_v
		
		if type == "moving_rod_y":
			flow_v = self.max_speed*(np.random.rand()-0.5)*2
			
			if flow_v > 0:
				object_x = np.random.randint(self.w//4-10,self.w//4+10)
			else:
				object_x = np.random.randint(3*self.w//4-10,3*self.w//4+10)
			object_z = np.random.randint(self.d//2-10,self.d//2+10)
			object_vx = self.init_velocity*(np.random.rand()-0.5)*2
			object_vz = self.init_velocity*(np.random.rand()-0.5)*2
			self.cond_mask[index,:,(object_x-rod_size):(object_x+rod_size),:,(object_z-rod_size):(object_z+rod_size)]=1
			
			self.v_cond[index,0,0:5,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index,0,(self.w-5):self.w,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
			
			self.env_info[index]["type"] = type
			self.env_info[index]["flow_v"] = flow_v
			self.env_info[index]["x"] = object_x
			self.env_info[index]["z"] = object_z
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vz"] = object_vz
			self.mousex = object_x
			self.mousez = object_z
			self.mousev = flow_v
		
		if type == "moving_rod_z":
			flow_v = self.max_speed*(np.random.rand()-0.5)*2
			
			if flow_v > 0:
				object_x = np.random.randint(self.w//4-10,self.w//4+10)
			else:
				object_x = np.random.randint(3*self.w//4-10,3*self.w//4+10)
			object_y = np.random.randint(self.h//2-10,self.h//2+10)
			object_vx = self.init_velocity*(np.random.rand()-0.5)*2
			object_vy = self.init_velocity*(np.random.rand()-0.5)*2
			self.cond_mask[index,:,(object_x-rod_size):(object_x+rod_size),(object_y-rod_size):(object_y+rod_size),:]=1
			
			self.v_cond[index,0,0:5,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index,0,(self.w-5):self.w,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
			
			self.env_info[index]["type"] = type
			self.env_info[index]["flow_v"] = flow_v
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.mousex = object_x
			self.mousey = object_y
			self.mousev = flow_v
		
		if type == "box":# block at random position
			object_w = np.random.randint(3,15) # object width / 2
			object_h = np.random.randint(3,15) # object height / 2
			object_d = np.random.randint(3,15) # object depth / 2
			flow_v = self.max_speed*(np.random.rand()-0.5)*2
			if flow_v>0:
				object_x = np.random.randint(self.w//4-10,self.w//4+10)
			else:
				object_x = np.random.randint(3*self.w//4-10,3*self.w//4+10)
			object_y = np.random.randint(self.h//2-10,self.h//2+10)
			object_z = np.random.randint(self.d//2-10,self.d//2+10)
			object_vx = self.init_velocity*(np.random.rand()-0.5)*2
			object_vy = self.init_velocity*(np.random.rand()-0.5)*2
			object_vz = self.init_velocity*(np.random.rand()-0.5)*2
			
			self.cond_mask[index,:,(object_x-object_w):(object_x+object_w),(object_y-object_h):(object_y+object_h),(object_z-object_d):(object_z+object_d)] = 1
			self.v_cond[index,0,(object_x-object_w):(object_x+object_w),(object_y-object_h):(object_y+object_h),(object_z-object_d):(object_z+object_d)] = object_vx
			self.v_cond[index,1,(object_x-object_w):(object_x+object_w),(object_y-object_h):(object_y+object_h),(object_z-object_d):(object_z+object_d)] = object_vy
			self.v_cond[index,2,(object_x-object_w):(object_x+object_w),(object_y-object_h):(object_y+object_h),(object_z-object_d):(object_z+object_d)] = object_vz
			
			self.v_cond[index,0,0:5,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index,0,(self.w-5):self.w,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
			
			self.env_info[index]["type"] = type
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["z"] = object_z
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["vz"] = object_vz
			self.env_info[index]["w"] = object_w
			self.env_info[index]["h"] = object_h
			self.env_info[index]["d"] = object_d
			self.env_info[index]["flow_v"] = flow_v
			self.mousex = object_x
			self.mousey = object_y
			self.mousez = object_z
			self.mousev = flow_v
		
		if type=="image":
			
			image = np.random.choice(self.images)
			image_mask = img_dict[image]
		
			flow_v = self.max_speed*(np.random.rand()-0.5)*2
			if flow_v>0:
				object_x = np.random.randint(self.w//4-5,self.w//4+5)
			else:
				object_x = np.random.randint(3*self.w//4-5,3*self.w//4+5)
			object_y = np.random.randint(self.h//2-5,self.h//2+5)
			object_z = np.random.randint(self.d//2-5,self.d//2+5)
			object_vx = self.init_velocity*(np.random.rand()-0.5)*2
			object_vy = self.init_velocity*(np.random.rand()-0.5)*2
			object_vz = self.init_velocity*(np.random.rand()-0.5)*2
			
			w,h,d = image_mask.shape[1],image_mask.shape[2],image_mask.shape[3]
			self.cond_mask[index,:,(object_x-w//2):(object_x-w//2+w),(object_y-h//2):(object_y-h//2+h),(object_z-d//2):(object_z-d//2+d)] = image_mask
			self.v_cond[index,0,(object_x-w//2):(object_x-w//2+w),(object_y-h//2):(object_y-h//2+h),(object_z-d//2):(object_z-d//2+d)] = object_vx
			self.v_cond[index,1,(object_x-w//2):(object_x-w//2+w),(object_y-h//2):(object_y-h//2+h),(object_z-d//2):(object_z-d//2+d)] = object_vy
			self.v_cond[index,2,(object_x-w//2):(object_x-w//2+w),(object_y-h//2):(object_y-h//2+h),(object_z-d//2):(object_z-d//2+d)] = object_vz
			
			self.v_cond[index,0,0:5,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index,0,(self.w-5):self.w,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
			
			self.env_info[index]["type"] = type
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["z"] = object_z
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["vz"] = object_vz
			self.env_info[index]["image"] = image
			self.env_info[index]["flow_v"] = flow_v
			self.mousex = object_x
			self.mousey = object_y
			self.mousez = object_z
			self.mousev = flow_v
		
		if type=="ball":
			object_r = np.random.randint(3,15) # object radius / 2
			flow_v = self.max_speed*(np.random.rand()-0.5)*2
			if flow_v>0:
				object_x = np.random.randint(self.w//4-10,self.w//4+10)
			else:
				object_x = np.random.randint(3*self.w//4-10,3*self.w//4+10)
			object_y = np.random.randint(self.h//2-10,self.h//2+10)
			object_z = np.random.randint(self.d//2-10,self.d//2+10)
			object_vx = self.init_velocity*(np.random.rand()-0.5)*2
			object_vy = self.init_velocity*(np.random.rand()-0.5)*2
			object_vz = self.init_velocity*(np.random.rand()-0.5)*2
			object_wx = self.max_speed*(np.random.rand()-0.5)*2/object_r # object angular velocity
			object_wy = self.max_speed*(np.random.rand()-0.5)*2/object_r # object angular velocity
			object_wz = self.max_speed*(np.random.rand()-0.5)*2/object_r # object angular velocity
			
			# 1. generate mesh 2 x [2r x 2r]
			x_mesh,y_mesh,z_mesh = torch.meshgrid([torch.arange(-object_r,object_r+1),torch.arange(-object_r,object_r+1),torch.arange(-object_r,object_r+1)])
			
			# 2. generate mask
			mask_ball = ((x_mesh**2+y_mesh**2+z_mesh**2)<object_r**2).float().unsqueeze(0)
			
			# 3. generate v_cond and multiply with mask
			v_ball = torch.cat([(object_wy*z_mesh-object_wz*y_mesh).unsqueeze(0),(object_wz*x_mesh-object_wx*z_mesh).unsqueeze(0),(object_wx*y_mesh-object_wy*x_mesh).unsqueeze(0)])*mask_ball
			
			self.cond_mask[index,:,(object_x-object_r):(object_x+object_r+1),(object_y-object_r):(object_y+object_r+1),(object_z-object_r):(object_z+object_r+1)] = mask_ball
			self.v_cond[index,0,(object_x-object_r):(object_x+object_r+1),(object_y-object_r):(object_y+object_r+1),(object_z-object_r):(object_z+object_r+1)] = v_ball[0]+object_vx
			self.v_cond[index,1,(object_x-object_r):(object_x+object_r+1),(object_y-object_r):(object_y+object_r+1),(object_z-object_r):(object_z+object_r+1)] = v_ball[1]+object_vy
			self.v_cond[index,2,(object_x-object_r):(object_x+object_r+1),(object_y-object_r):(object_y+object_r+1),(object_z-object_r):(object_z+object_r+1)] = v_ball[2]+object_vz
			
			self.v_cond[index,0,0:5,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index,0,(self.w-5):self.w,10:(self.h-10),10:(self.d-10)]=flow_v
			self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
			
			self.env_info[index]["type"] = type
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["z"] = object_z
			self.env_info[index]["r"] = object_r
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["vz"] = object_vz
			self.env_info[index]["wx"] = object_wx
			self.env_info[index]["wy"] = object_wy
			self.env_info[index]["wz"] = object_wz
			self.env_info[index]["flow_v"] = flow_v
			self.mousex = object_x
			self.mousey = object_y
			self.mousez = object_z
			self.mousev = flow_v
	
#}}}

		#IC, BC for p, T

		p_profile = (params.sp) / ( np.exp((np.arange(0,self.d,1)*params.dz)/params.scale_height) ) # z-coord pressure profile for scale height(p)
		rho_profile = 1.225/( np.exp((np.arange(0,self.d,1)*params.dz)/params.scale_height) )   # z-coord rho profile for scale height(p)

		self.p[index]   = torch.from_numpy(np.broadcast_to(p_profile,(self.w,self.h,self.d)))

		T_profile     = (273.15 + 15) - (10./1000.)*(np.arange(0,self.d,1)*params.dz)           # dry lapse rate
		self.T[index]   = torch.from_numpy(np.broadcast_to(T_profile,(self.w,self.h,self.d)))

		p_x   = np.linspace(1,-1,self.w) * flow_v * self.w * params.dx 
		p_X   = p_x.reshape(self.w, 1,1)
		p_pur = np.broadcast_to(p_X,(self.w,self.h,self.d))

		#self.p_cond[index] = self.p[index] + p_pur
		self.p_cond[index] = self.p[index]

		#self.T_cond[index] = self.T[index] + p_pur/(rho_profile*params.R) 
		self.T_cond[index] = self.T[index]

		self.v[index]      = flow_v

	def update_envs(self,indices):
		for index in indices:
			
			if self.env_info[index]["type"] == "magnus_z":
				object_r = self.env_info[index]["r"]
				vx_old = self.env_info[index]["vx"]
				vy_old = self.env_info[index]["vy"]
				
				if not self.interactive:
					flow_v = self.env_info[index]["flow_v"]
					object_w = self.env_info[index]["w"]
					object_vx = vx_old*self.brown_damping + self.brown_velocity*np.random.randn()
					object_vy = vy_old*self.brown_damping + self.brown_velocity*np.random.randn()
					
					object_x = self.env_info[index]["x"]+( (vx_old+object_vx)/2*self.dt ) / params.dx
					object_y = self.env_info[index]["y"]+( (vy_old+object_vy)/2*self.dt ) / params.dy
					
					if object_x < object_r + 10:
						object_x = object_r + 10
						object_vx = -object_vx
					if object_x > self.w - object_r - 10:
						object_x = self.w - object_r - 10
						object_vx = -object_vx
						
					if object_y < object_r + 10:
						object_y = object_r+10
						object_vy = -object_vy
					if object_y > self.h - object_r - 10:
						object_y = self.h - object_r - 10
						object_vy = -object_vy
					
				if self.interactive:
					flow_v = self.mousev
					object_w = self.mousew/object_r
					object_vx = max(min((self.mousex-self.env_info[index]["x"])/self.interactive_spring,self.max_speed),-self.max_speed)
					object_vy = max(min((self.mousey-self.env_info[index]["y"])/self.interactive_spring,self.max_speed),-self.max_speed)
					
					object_x = self.env_info[index]["x"]+( (vx_old+object_vx)/2*self.dt ) / params.dx
					object_y = self.env_info[index]["y"]+( (vy_old+object_vy)/2*self.dt ) / params.dy
					
					if object_x < object_r + 10:
						object_x = object_r + 10
						object_vx = 0
					if object_x > self.w - object_r - 10:
						object_x = self.w - object_r - 10
						object_vx = 0
						
					if object_y < object_r + 10:
						object_y = object_r+10
						object_vy = 0
					if object_y > self.h - object_r - 10:
						object_y = self.h - object_r - 10
						object_vy = 0
				
				
				# 1. generate mesh 2 x [2r x 2r]
				y_mesh,x_mesh = torch.meshgrid([torch.arange(-object_r,object_r+1),torch.arange(-object_r,object_r+1)])
				
				# 2. generate mask
				mask_ball = ((x_mesh**2+y_mesh**2)<object_r**2).float().unsqueeze(0).unsqueeze(3)
				
				# 3. generate v_cond and multiply with mask
				v_ball = object_w*torch.cat([x_mesh.unsqueeze(0),-y_mesh.unsqueeze(0)]).unsqueeze(3)*mask_ball
				
				
				# 4. add masks / v_cond
				self.cond_mask[index]=0
				self.cond_mask[index,:,0:5,:,:]=1
				self.cond_mask[index,:,(self.w-5):self.w,:,:]=1
				self.cond_mask[index,:,:,0:3,:]=1
				self.cond_mask[index,:,:,(self.h-3):self.h,:]=1
				self.cond_mask[index,:,:,:,0:3]=1
				self.cond_mask[index,:,:,:,(self.d-3):self.d]=1
				
				self.cond_mask[index,:,int(object_x-object_r):int(object_x+object_r+1),int(object_y-object_r):int(object_y+object_r+1),3:(self.d-3)] += mask_ball
				self.v_cond[index,0,int(object_x-object_r):int(object_x+object_r+1),int(object_y-object_r):int(object_y+object_r+1),3:(self.d-3)] = v_ball[0]+object_vx
				self.v_cond[index,1,int(object_x-object_r):int(object_x+object_r+1),int(object_y-object_r):int(object_y+object_r+1),3:(self.d-3)] = v_ball[1]+object_vy
				self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
				
				self.v_cond[index,0,0:5,10:(self.h-10),10:(self.d-10)]=flow_v
				self.v_cond[index,0,(self.w-5):self.w,10:(self.h-10),10:(self.d-10)]=flow_v
				
				self.env_info[index]["x"] = object_x
				self.env_info[index]["y"] = object_y
				self.env_info[index]["vx"] = object_vx
				self.env_info[index]["vy"] = object_vy
				self.env_info[index]["flow_v"] = flow_v
			
			if self.env_info[index]["type"] == "magnus_y":
				object_r = self.env_info[index]["r"]
				vx_old = self.env_info[index]["vx"]
				vz_old = self.env_info[index]["vz"]
				
				if not self.interactive:
					flow_v = self.env_info[index]["flow_v"]
					object_w = self.env_info[index]["w"]
					object_vx = vx_old*self.brown_damping + self.brown_velocity*np.random.randn()
					object_vz = vz_old*self.brown_damping + self.brown_velocity*np.random.randn()
					
					object_x = self.env_info[index]["x"]+( (vx_old+object_vx)/2*self.dt ) / params.dx
					object_z = self.env_info[index]["z"]+( (vz_old+object_vz)/2*self.dt ) / params.dz
					
					if object_x < object_r + 10:
						object_x = object_r + 10
						object_vx = -object_vx
					if object_x > self.w - object_r - 10:
						object_x = self.w - object_r - 10
						object_vx = -object_vx
						
					if object_z < object_r + 10:
						object_z = object_r+10
						object_vz = -object_vz
					if object_z > self.d - object_r - 10:
						object_z = self.d - object_r - 10
						object_vz = -object_vz
					
				if self.interactive:
					flow_v = self.mousev
					object_w = self.mousew/object_r
					object_vx = max(min((self.mousex-self.env_info[index]["x"])/self.interactive_spring,self.max_speed),-self.max_speed)
					object_vz = max(min((self.mousez-self.env_info[index]["z"])/self.interactive_spring,self.max_speed),-self.max_speed)
					
					object_x = self.env_info[index]["x"]+( (vx_old+object_vx)/2*self.dt ) / params.dx
					object_z = self.env_info[index]["z"]+( (vz_old+object_vz)/2*self.dt ) / params.dz
					
					if object_x < object_r + 10:
						object_x = object_r + 10
						object_vx = 0
					if object_x > self.w - object_r - 10:
						object_x = self.w - object_r - 10
						object_vx = 0
						
					if object_z < object_r + 10:
						object_z = object_r+10
						object_vz = 0
					if object_z > self.d - object_r - 10:
						object_z = self.d - object_r - 10
						object_vz = 0
				
				
				# 1. generate mesh 2 x [2r x 2r]
				z_mesh,x_mesh = torch.meshgrid([torch.arange(-object_r,object_r+1),torch.arange(-object_r,object_r+1)])
				
				# 2. generate mask
				mask_ball = ((x_mesh**2+z_mesh**2)<object_r**2).float().unsqueeze(0).unsqueeze(2)
				
				# 3. generate v_cond and multiply with mask
				v_ball = object_w*torch.cat([x_mesh.unsqueeze(0),-z_mesh.unsqueeze(0)]).unsqueeze(2)*mask_ball
				
				
				# 4. add masks / v_cond
				self.cond_mask[index]=0
				self.cond_mask[index,:,0:5,:,:]=1
				self.cond_mask[index,:,(self.w-5):self.w,:,:]=1
				self.cond_mask[index,:,:,0:3,:]=1
				self.cond_mask[index,:,:,(self.h-3):self.h,:]=1
				self.cond_mask[index,:,:,:,0:3]=1
				self.cond_mask[index,:,:,:,(self.d-3):self.d]=1
				
				self.cond_mask[index,:,int(object_x-object_r):int(object_x+object_r+1),3:(self.h-3),int(object_z-object_r):int(object_z+object_r+1)] += mask_ball
				self.v_cond[index,0,int(object_x-object_r):int(object_x+object_r+1),3:(self.h-3),int(object_z-object_r):int(object_z+object_r+1)] = v_ball[0]+object_vx
				self.v_cond[index,2,int(object_x-object_r):int(object_x+object_r+1),3:(self.h-3),int(object_z-object_r):int(object_z+object_r+1)] = v_ball[1]+object_vz
				self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
				
				self.v_cond[index,0,0:5,10:(self.h-10),10:(self.d-10)]=flow_v
				self.v_cond[index,0,(self.w-5):self.w,10:(self.h-10),10:(self.d-10)]=flow_v
				
				self.env_info[index]["x"] = object_x
				self.env_info[index]["z"] = object_z
				self.env_info[index]["vx"] = object_vx
				self.env_info[index]["vz"] = object_vz
				self.env_info[index]["flow_v"] = flow_v
			
			if self.env_info[index]["type"] == "no_box":
				
				if not self.interactive:
					flow_v = self.env_info[index]["flow_v"]
					
				if self.interactive:
					flow_v = self.mousev
				
				self.cond_mask[index]=0
				self.cond_mask[index,:,0:5,:,:]=1
				self.cond_mask[index,:,(self.w-5):self.w,:,:]=1
				self.cond_mask[index,:,:,0:3,:]=1
				self.cond_mask[index,:,:,(self.h-3):self.h,:]=1
				self.cond_mask[index,:,:,:,0:3]=1
				self.cond_mask[index,:,:,:,(self.d-3):self.d]=1
				
				self.v_cond[index,0,0:5,10:(self.h-10),10:(self.d-10)]=flow_v
				self.v_cond[index,0,(self.w-5):self.w,10:(self.h-10),10:(self.d-10)]=flow_v
				self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
				
				self.env_info[index]["flow_v"] = flow_v
				
			if self.env_info[index]["type"] == "rod_y" or self.env_info[index]["type"] == "rod_z":
				
				if not self.interactive:
					flow_v = self.env_info[index]["flow_v"]
					
				if self.interactive:
					flow_v = self.mousev
				
				self.v_cond[index,0,0:5,10:(self.h-10),10:(self.d-10)]=flow_v
				self.v_cond[index,0,(self.w-5):self.w,10:(self.h-10),10:(self.d-10)]=flow_v
				self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
				
				self.env_info[index]["flow_v"] = flow_v
				
			if self.env_info[index]["type"] == "moving_rod_y":
				vx_old = self.env_info[index]["vx"]
				vz_old = self.env_info[index]["vz"]
				
				if not self.interactive:
					flow_v = self.env_info[index]["flow_v"]
					object_vx = vx_old*self.brown_damping + self.brown_velocity*np.random.randn()
					object_vz = vz_old*self.brown_damping + self.brown_velocity*np.random.randn()
					
					object_x = self.env_info[index]["x"]+( (vx_old+object_vx)/2*self.dt ) / params.dx
					object_z = self.env_info[index]["z"]+( (vz_old+object_vz)/2*self.dt ) / params.dz
					
					if object_x < rod_size + 10:
						object_x = rod_size + 10
						object_vx = -object_vx
					if object_x > self.w - rod_size - 10:
						object_x = self.w - rod_size - 10
						object_vx = -object_vx
						
					if object_z < rod_size + 10:
						object_z = rod_size+10
						object_vz = -object_vz
					if object_z > self.d - rod_size - 10:
						object_z = self.d - rod_size - 10
						object_vz = -object_vz
					
				if self.interactive:
					flow_v = self.mousev
					object_vx = max(min((self.mousex-self.env_info[index]["x"])/self.interactive_spring,self.max_speed),-self.max_speed)
					object_vz = max(min((self.mousez-self.env_info[index]["z"])/self.interactive_spring,self.max_speed),-self.max_speed)
					
					object_x = self.env_info[index]["x"]+( (vx_old+object_vx)/2*self.dt ) / params.dx
					object_z = self.env_info[index]["z"]+( (vz_old+object_vz)/2*self.dt ) / params.dz
					
					if object_x < rod_size + 10:
						object_x = rod_size + 10
						object_vx = 0
					if object_x > self.w - rod_size - 10:
						object_x = self.w - rod_size - 10
						object_vx = 0
						
					if object_z < rod_size + 10:
						object_z = rod_size+10
						object_vz = 0
					if object_z > self.d - rod_size - 10:
						object_z = self.d - rod_size - 10
						object_vz = 0
				
				self.cond_mask[index]=0
				self.cond_mask[index,:,0:5,:,:]=1
				self.cond_mask[index,:,(self.w-5):self.w,:,:]=1
				self.cond_mask[index,:,:,0:3,:]=1
				self.cond_mask[index,:,:,(self.h-3):self.h,:]=1
				self.cond_mask[index,:,:,:,0:3]=1
				self.cond_mask[index,:,:,:,(self.d-3):self.d]=1
				
				self.cond_mask[index,:,int(object_x-rod_size):int(object_x+rod_size),3:(self.h-3),int(object_z-rod_size):int(object_z+rod_size)]=1
				
				self.v_cond[index,0,int(object_x-rod_size):int(object_x+rod_size),3:(self.h-3),int(object_z-rod_size):int(object_z+rod_size)] = object_vx
				self.v_cond[index,2,int(object_x-rod_size):int(object_x+rod_size),3:(self.h-3),int(object_z-rod_size):int(object_z+rod_size)] = object_vz
				
				self.v_cond[index,0,0:5,10:(self.h-10),10:(self.d-10)]=flow_v
				self.v_cond[index,0,(self.w-5):self.w,10:(self.h-10),10:(self.d-10)]=flow_v
				self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
				
				self.env_info[index]["x"] = object_x
				self.env_info[index]["z"] = object_z
				self.env_info[index]["vx"] = object_vx
				self.env_info[index]["vz"] = object_vz
				self.env_info[index]["flow_v"] = flow_v
				#print(f"self.env_info[index]: {self.env_info[index]}")
				
			if self.env_info[index]["type"] == "moving_rod_z":
				vx_old = self.env_info[index]["vx"]
				vy_old = self.env_info[index]["vy"]
				
				if not self.interactive:
					flow_v = self.env_info[index]["flow_v"]
					object_vx = vx_old*self.brown_damping + self.brown_velocity*np.random.randn()
					object_vy = vy_old*self.brown_damping + self.brown_velocity*np.random.randn()
					
					object_x = self.env_info[index]["x"]+( (vx_old+object_vx)/2*self.dt ) / params.dx
					object_y = self.env_info[index]["y"]+( (vy_old+object_vy)/2*self.dt ) / params.dy
					
					if object_x < rod_size + 10:
						object_x = rod_size + 10
						object_vx = -object_vx
					if object_x > self.w - rod_size - 10:
						object_x = self.w - rod_size - 10
						object_vx = -object_vx
						
					if object_y < rod_size + 10:
						object_y = rod_size+10
						object_vy = -object_vy
					if object_y > self.h - rod_size - 10:
						object_y = self.h - rod_size - 10
						object_vy = -object_vy
					
				if self.interactive:
					flow_v = self.mousev
					object_vx = max(min((self.mousex-self.env_info[index]["x"])/self.interactive_spring,self.max_speed),-self.max_speed)
					object_vy = max(min((self.mousey-self.env_info[index]["y"])/self.interactive_spring,self.max_speed),-self.max_speed)
					
					object_x = self.env_info[index]["x"]+( (vx_old+object_vx)/2*self.dt ) / params.dx
					object_y = self.env_info[index]["y"]+( (vy_old+object_vy)/2*self.dt ) / params.dy
					
					if object_x < rod_size + 10:
						object_x = rod_size + 10
						object_vx = 0
					if object_x > self.w - rod_size - 10:
						object_x = self.w - rod_size - 10
						object_vx = 0
						
					if object_y < rod_size + 10:
						object_y = rod_size+10
						object_vy = 0
					if object_y > self.h - rod_size - 10:
						object_y = self.h - rod_size - 10
						object_vy = 0
				
				self.cond_mask[index]=0
				self.cond_mask[index,:,0:5,:,:]=1
				self.cond_mask[index,:,(self.w-5):self.w,:,:]=1
				self.cond_mask[index,:,:,0:3,:]=1
				self.cond_mask[index,:,:,(self.h-3):self.h,:]=1
				self.cond_mask[index,:,:,:,0:3]=1
				self.cond_mask[index,:,:,:,(self.d-3):self.d]=1
				
				self.cond_mask[index,:,int(object_x-rod_size):int(object_x+rod_size),int(object_y-rod_size):int(object_y+rod_size),3:(self.d-3)]=1
				
				self.v_cond[index,0,int(object_x-rod_size):int(object_x+rod_size),int(object_y-rod_size):int(object_y+rod_size),3:(self.d-3)] = object_vx
				self.v_cond[index,1,int(object_x-rod_size):int(object_x+rod_size),int(object_y-rod_size):int(object_y+rod_size),3:(self.d-3)] = object_vy
				
				self.v_cond[index,0,0:5,10:(self.h-10),10:(self.d-10)]=flow_v
				self.v_cond[index,0,(self.w-5):self.w,10:(self.h-10),10:(self.d-10)]=flow_v
				self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
				
				self.env_info[index]["x"] = object_x
				self.env_info[index]["y"] = object_y
				self.env_info[index]["vx"] = object_vx
				self.env_info[index]["vy"] = object_vy
				self.env_info[index]["flow_v"] = flow_v
				#print(f"self.env_info[index]: {self.env_info[index]}")
				
			if self.env_info[index]["type"] == "box":
				object_h = self.env_info[index]["h"]
				object_w = self.env_info[index]["w"]
				object_d = self.env_info[index]["d"]
				vx_old = self.env_info[index]["vx"]
				vy_old = self.env_info[index]["vy"]
				vz_old = self.env_info[index]["vz"]
				
				if not self.interactive:
					flow_v = self.env_info[index]["flow_v"]
					object_vx = vx_old*self.brown_damping + self.brown_velocity*np.random.randn()
					object_vy = vy_old*self.brown_damping + self.brown_velocity*np.random.randn()
					object_vz = vz_old*self.brown_damping + self.brown_velocity*np.random.randn()
					
					object_x = self.env_info[index]["x"]+((vx_old+object_vx)/2*self.dt)/params.dx
					object_y = self.env_info[index]["y"]+((vy_old+object_vy)/2*self.dt)/params.dy
					object_z = self.env_info[index]["z"]+((vz_old+object_vz)/2*self.dt)/params.dz
					
					if object_x < object_w + 10:
						object_x = object_w + 10
						object_vx = -object_vx
					if object_x > self.w - object_w - 10:
						object_x = self.w - object_w - 10
						object_vx = -object_vx
						
					if object_y < object_h + 10:
						object_y = object_h+10
						object_vy = -object_vy
					if object_y > self.h - object_h - 10:
						object_y = self.h - object_h - 10
						object_vy = -object_vy
						
					if object_z < object_d + 10:
						object_z = object_d+10
						object_vz = -object_vz
					if object_z > self.d - object_d - 10:
						object_z = self.d - object_d - 10
						object_vz = -object_vz
					
				if self.interactive:
					flow_v = self.mousev
					object_vx = max(min((self.mousex-self.env_info[index]["x"])/self.interactive_spring,self.max_speed),-self.max_speed)
					object_vy = max(min((self.mousey-self.env_info[index]["y"])/self.interactive_spring,self.max_speed),-self.max_speed)
					object_vz = max(min((self.mousez-self.env_info[index]["z"])/self.interactive_spring,self.max_speed),-self.max_speed)
					
					object_x = self.env_info[index]["x"]+((vx_old+object_vx)/2*self.dt)/params.dx
					object_y = self.env_info[index]["y"]+((vy_old+object_vy)/2*self.dt)/params.dy
					object_z = self.env_info[index]["z"]+((vz_old+object_vz)/2*self.dt)/params.dz
					
					if object_x < object_w + 10:
						object_x = object_w + 10
						object_vx = 0
					if object_x > self.w - object_w - 10:
						object_x = self.w - object_w - 10
						object_vx = 0
						
					if object_y < object_h + 10:
						object_y = object_h+10
						object_vy = 0
					if object_y > self.h - object_h - 10:
						object_y = self.h - object_h - 10
						object_vy = 0
						
					if object_z < object_d + 10:
						object_z = object_d+10
						object_vz = 0
					if object_z > self.d - object_d - 10:
						object_z = self.d - object_d - 10
						object_vz = 0
				
				self.cond_mask[index]=0
				self.cond_mask[index,:,0:5,:,:]=1
				self.cond_mask[index,:,(self.w-5):self.w,:,:]=1
				self.cond_mask[index,:,:,0:3,:]=1
				self.cond_mask[index,:,:,(self.h-3):self.h,:]=1
				self.cond_mask[index,:,:,:,0:3]=1
				self.cond_mask[index,:,:,:,(self.d-3):self.d]=1
				
				self.cond_mask[index,:,int(object_x-object_w):int(object_x+object_w),int(object_y-object_h):int(object_y+object_h),int(object_z-object_d):int(object_z+object_d)] = 1
				self.v_cond[index,0,int(object_x-object_w):int(object_x+object_w),int(object_y-object_h):int(object_y+object_h),int(object_z-object_d):int(object_z+object_d)] = object_vx
				self.v_cond[index,1,int(object_x-object_w):int(object_x+object_w),int(object_y-object_h):int(object_y+object_h),int(object_z-object_d):int(object_z+object_d)] = object_vy
				self.v_cond[index,2,int(object_x-object_w):int(object_x+object_w),int(object_y-object_h):int(object_y+object_h),int(object_z-object_d):int(object_z+object_d)] = object_vz
				
				self.v_cond[index,0,0:5,10:(self.h-10),10:(self.d-10)]=flow_v
				self.v_cond[index,0,(self.w-5):self.w,10:(self.h-10),10:(self.d-10)]=flow_v
				self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
				
				self.env_info[index]["x"] = object_x
				self.env_info[index]["y"] = object_y
				self.env_info[index]["z"] = object_z
				self.env_info[index]["vx"] = object_vx
				self.env_info[index]["vy"] = object_vy
				self.env_info[index]["vz"] = object_vz
				self.env_info[index]["flow_v"] = flow_v
				
			
			if self.env_info[index]["type"] == "ball":
				object_r = self.env_info[index]["r"]
				vx_old = self.env_info[index]["vx"]
				vy_old = self.env_info[index]["vy"]
				vz_old = self.env_info[index]["vz"]
				object_wx = self.env_info[index]["wx"]
				object_wy = self.env_info[index]["wy"]
				object_wz = self.env_info[index]["wz"]
				
				if not self.interactive:
					flow_v = self.env_info[index]["flow_v"]
					object_vx = vx_old*self.brown_damping + self.brown_velocity*np.random.randn()
					object_vy = vy_old*self.brown_damping + self.brown_velocity*np.random.randn()
					object_vz = vz_old*self.brown_damping + self.brown_velocity*np.random.randn()
					
					object_x = self.env_info[index]["x"]+((vx_old+object_vx)/2*self.dt)/params.dx
					object_y = self.env_info[index]["y"]+((vy_old+object_vy)/2*self.dt)/params.dy
					object_z = self.env_info[index]["z"]+((vz_old+object_vz)/2*self.dt)/params.dz
					
					if object_x < object_r + 10:
						object_x = object_r + 10
						object_vx = -object_vx
					if object_x > self.w - object_r - 10:
						object_x = self.w - object_r - 10
						object_vx = -object_vx
						
					if object_y < object_r + 10:
						object_y = object_r+10
						object_vy = -object_vy
					if object_y > self.h - object_r - 10:
						object_y = self.h - object_r - 10
						object_vy = -object_vy
						
					if object_z < object_r + 10:
						object_z = object_r+10
						object_vz = -object_vz
					if object_z > self.d - object_r - 10:
						object_z = self.d - object_r - 10
						object_vz = -object_vz
					
				if self.interactive:
					#CODO: interactive angular velocity
					flow_v = self.mousev
					object_vx = max(min((self.mousex-self.env_info[index]["x"])/self.interactive_spring,self.max_speed),-self.max_speed)
					object_vy = max(min((self.mousey-self.env_info[index]["y"])/self.interactive_spring,self.max_speed),-self.max_speed)
					object_vz = max(min((self.mousez-self.env_info[index]["z"])/self.interactive_spring,self.max_speed),-self.max_speed)
					
					object_x = self.env_info[index]["x"]+((vx_old+object_vx)/2*self.dt)/params.dx
					object_y = self.env_info[index]["y"]+((vy_old+object_vy)/2*self.dt)/params.dy
					object_z = self.env_info[index]["z"]+((vz_old+object_vz)/2*self.dt)/params.dz
					
					if object_x < object_r + 10:
						object_x = object_r + 10
						object_vx = 0
					if object_x > self.w - object_r - 10:
						object_x = self.w - object_r - 10
						object_vx = 0
						
					if object_y < object_r + 10:
						object_y = object_r+10
						object_vy = 0
					if object_y > self.h - object_r - 10:
						object_y = self.h - object_r - 10
						object_vy = 0
						
					if object_z < object_r + 10:
						object_z = object_r+10
						object_vz = 0
					if object_z > self.d - object_r - 10:
						object_z = self.d - object_r - 10
						object_vz = 0
				
				self.cond_mask[index]=0
				self.cond_mask[index,:,0:5,:,:]=1
				self.cond_mask[index,:,(self.w-5):self.w,:,:]=1
				self.cond_mask[index,:,:,0:3,:]=1
				self.cond_mask[index,:,:,(self.h-3):self.h,:]=1
				self.cond_mask[index,:,:,:,0:3]=1
				self.cond_mask[index,:,:,:,(self.d-3):self.d]=1
				
				# 1. generate mesh 2 x [2r x 2r]
				x_mesh,y_mesh,z_mesh = torch.meshgrid([torch.arange(-object_r,object_r+1),torch.arange(-object_r,object_r+1),torch.arange(-object_r,object_r+1)])
				
				# 2. generate mask
				mask_ball = ((x_mesh**2+y_mesh**2+z_mesh**2)<object_r**2).float().unsqueeze(0)
				
				# 3. generate v_cond and multiply with mask
				v_ball = torch.cat([(object_wy*z_mesh-object_wz*y_mesh).unsqueeze(0),(object_wz*x_mesh-object_wx*z_mesh).unsqueeze(0),(object_wx*y_mesh-object_wy*x_mesh).unsqueeze(0)])*mask_ball
				
				self.cond_mask[index,:,int(object_x-object_r):int(object_x+object_r+1),int(object_y-object_r):int(object_y+object_r+1),int(object_z-object_r):int(object_z+object_r+1)] = mask_ball
				self.v_cond[index,0,int(object_x-object_r):int(object_x+object_r+1),int(object_y-object_r):int(object_y+object_r+1),int(object_z-object_r):int(object_z+object_r+1)] = v_ball[0]+object_vx
				self.v_cond[index,1,int(object_x-object_r):int(object_x+object_r+1),int(object_y-object_r):int(object_y+object_r+1),int(object_z-object_r):int(object_z+object_r+1)] = v_ball[1]+object_vy
				self.v_cond[index,2,int(object_x-object_r):int(object_x+object_r+1),int(object_y-object_r):int(object_y+object_r+1),int(object_z-object_r):int(object_z+object_r+1)] = v_ball[2]+object_vz
				
				self.v_cond[index,0,0:5,10:(self.h-10),10:(self.d-10)]=flow_v
				self.v_cond[index,0,(self.w-5):self.w,10:(self.h-10),10:(self.d-10)]=flow_v
				self.v_cond[index] = self.v_cond[index]*self.cond_mask[index]
				
				self.env_info[index]["x"] = object_x
				self.env_info[index]["y"] = object_y
				self.env_info[index]["z"] = object_z
				self.env_info[index]["vx"] = object_vx
				self.env_info[index]["vy"] = object_vy
				self.env_info[index]["vz"] = object_vz
				self.env_info[index]["wx"] = object_wx
				self.env_info[index]["wy"] = object_wy
				self.env_info[index]["wz"] = object_wz
				self.env_info[index]["flow_v"] = flow_v
			
			rho_profile = 1.225/( np.exp((np.arange(0,self.d,1)*params.dz)/params.scale_height) )

			p_profile          = (params.sp)/( np.exp((np.arange(0,self.d,1)*params.dz)/params.scale_height) )
			self.p_cond[index] = torch.from_numpy(np.broadcast_to(p_profile,(self.w,self.h,self.d)))
			p_x                = np.linspace(1,-1,self.w) * flow_v * self.w * params.dx
			p_X                = p_x.reshape(self.w, 1,1)
			p_pur              = np.broadcast_to(p_X,(self.w,self.h,self.d))
			self.p_cond[index] = self.p[index] + p_pur

			T_profile          = (273.15 + 15) - (10./1000.)*(np.arange(0,self.d,1)*params.dz)
			self.T_cond[index] = torch.from_numpy(np.broadcast_to(T_profile,(self.w,self.h,self.d)))
			self.T_cond[index] = self.T[index] + p_pur/(rho_profile*params.R)

	def ask(self):
		"""
		sample and update environemts
		:return:
		- v_cond: velocity conditions at dirichlet boundary
		- cond_mask: mask, which is 1 at dirichlet boundary and 0 elsewhere
		- a: vector potential of velocity field within fluid domain
		- p: pressure field
		"""
		if self.interactive:
			self.mousev = min(max(self.mousev,-self.max_speed),self.max_speed)
			self.mousew = min(max(self.mousew,-self.max_speed),self.max_speed)
		
		self.indices = np.random.choice(self.dataset_size,self.batch_size)
		self.update_envs(self.indices)
		return self.v_cond[self.indices], self.p_cond[self.indices], self.T_cond[self.indices], self.cond_mask[self.indices], self.bc_mask[self.indices], self.v[self.indices], self.p[self.indices],self.T[self.indices]
	
	def tell(self,v,p,T):
		self.v[self.indices]   = v.detach()
		self.p[self.indices]   = p.detach()
		self.T[self.indices]   = T.detach()
		
		self.t += 1
		if self.t % (self.average_sequence_length/self.batch_size) == 0:#ca x*batch_size steps until env gets reset
			self.reset_env(int(self.i))
			self.i = (self.i+1)%self.dataset_size
