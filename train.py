from get_param import params,get_hyperparam
import torch
from torch.optim import Adam
import numpy as np
from derivatives import toCuda,toCpu
import derivatives as d
from derivatives import vector2HSV,rot_mac
from setups import Dataset
from Logger import Logger,t_step
from pde_cnn import get_Net

torch.manual_seed(0)
torch.set_num_threads(4)
np.random.seed(0)

print(f"Parameters: {vars(params)}")

# initialize model
pde_cnn = toCuda(get_Net(params))
pde_cnn.train()

# initialize optimizer
optimizer = Adam(pde_cnn.parameters(),lr=params.lr)

# initialize logger and, if demanded, load previous model / optimizer
logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=params.log)
if params.load_latest or params.load_date_time is not None or params.load_index is not None:
	load_logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=False)
	if params.load_optimizer:
		params.load_date_time, params.load_index = logger.load_state(pde_cnn,optimizer,params.load_date_time,params.load_index)
	else:
		params.load_date_time, params.load_index = logger.load_state(pde_cnn,None,params.load_date_time,params.load_index)
	params.load_index=int(params.load_index)
	print(f"loaded: {params.load_date_time}, {params.load_index}")
params.load_index = 0 if params.load_index is None else params.load_index

# initialize dataset
dataset = Dataset(params.width,params.height,params.depth,params.batch_size,params.dataset_size,params.average_sequence_length,max_speed=params.max_speed,dt=params.dt,types=["box","moving_rod_y","moving_rod_z","magnus_y","magnus_z","ball"],mu=params.mu)

eps = 0.00000001

f   = params.f
g   = params.g
R   = params.R
mu  = params.mu
cp  = params.cp

sponge_c_h = np.zeros(params.depth)
sponge_c_z = np.zeros(params.depth)

for i in range(10):
    sponge_c_h[params.depth-i-1] = (params.dx**2/params.dt) * 0 * np.cos((np.pi/2) * i/10.)
    sponge_c_z[params.depth-i-1] = (params.dz**2/params.dt) * 0 * np.cos((np.pi/2) * i/10.)

SPONGE_h = torch.from_numpy(np.broadcast_to(sponge_c_h,(params.width, params.height, params.depth)))
SPONGE_z = torch.from_numpy(np.broadcast_to(sponge_c_h,(params.width, params.height, params.depth)))

SPONGE_h = toCuda(SPONGE_h)
SPONGE_z = toCuda(SPONGE_z)

def loss_function(x):
	if params.loss=="square":
		return torch.pow(x,2)
	if params.loss=="exp_square":
		x = torch.pow(x,2)
		return torch.exp(x/torch.max(x).detach()*5)
	if params.loss=="abs":
		return torch.abs(x)
	if params.loss=="log_square":
		return torch.log(torch.pow(x,2)+eps)

for epoch in range(params.load_index,params.n_epochs):

	for i in range(params.n_batches_per_epoch):
		# draw batch from dataset
		v_cond,p_cond,T_cond,cond_mask,bc_mask,v_old,p_old,T_old = toCuda(dataset.ask())

		#v_in = (v_old - torch.mean(v_old,dim=(2,3,4)).unsqueeze(2).unsqueeze(3).unsqueeze(4))
		#p_in = (p_old - torch.mean(p_old,dim=(1,2,3,4)).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4))
		#T_in = (T_old - torch.mean(T_old,dim=(1,2,3,4)).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4))
	
		v_in = v_old
		p_in = p_old
		T_in = T_old
	
		# map v_cond to MAC grid
		v_cond = d.normal2staggered(v_cond)
		
		# apply fluid model on fluid state / boundary conditions for given mu and rho

		v_new,p_new,T_new = pde_cnn(v_cond,p_cond,T_cond,cond_mask,bc_mask,v_in,p_in,T_in)
		#v_new,p_new,rho_new,T_new = pde_cnn(v_old,p_old,rho_old,T_old,v_cond,cond_mask)

		rho_old = (p_old)/(R*T_old)
		rho_new = (p_new)/(R*T_new)

	
		# compute masks for fluid domain / boundary conditions and map them to MAC grid:

		cond_mask_mac = (d.normal2staggered(cond_mask.repeat(1,3,1,1,1))==1).float()
		flow_mask_mac = 1-cond_mask_mac
		
		#bc_mask   = (d.normal2staggered(bc_mask.repeat(1,3,1,1,1))==1).float()
		flow_mask = 1-bc_mask

		#weight cond_mask_mac stronger at domain borders:
		cond_mask_mac = cond_mask_mac + params.loss_border * d.get_borders(cond_mask_mac)
		
		# compute loss on domain boundaries
		loss_V_bound = torch.mean(loss_function(cond_mask_mac*(v_new-v_cond))[:,:,1:-1,1:-1,1:-1],dim=(1,2,3,4))		
		loss_p_bound = torch.mean(loss_function(bc_mask*(p_new-p_cond))[:,:,1:-1,1:-1,1:-1],dim=(1,2,3,4))
		loss_T_bound = torch.mean(loss_function(bc_mask*(T_new-T_cond))[:,:,1:-1,1:-1,1:-1],dim=(1,2,3,4))

		# compute loss for Navier Stokes equations
		
		if params.integrator == "explicit":
			v = v_old
		if params.integrator == "implicit":
			v = v_new
		if params.integrator == "imex":
			v = (v_new+v_old)/2
		
#{{{derivative
		dudt = (v_new[:,0:1]-v_old[:,0:1])/params.dt
		dvdt = (v_new[:,1:2]-v_old[:,1:2])/params.dt
		dwdt = (v_new[:,2:3]-v_old[:,2:3])/params.dt

		U_grad_u = v[:,0:1]*d.dx(v[:,0:1]) +\
                   0.5*( d.map_vy2vx_p(v[:,1:2])*d.dy_p(v[:,0:1]) + d.map_vy2vx_m(v[:,1:2])*d.dy_m(v[:,0:1]) ) +\
                   0.5*( d.map_vz2vx_p(v[:,2:3])*d.dz_p(v[:,0:1]) + d.map_vz2vx_m(v[:,2:3])*d.dz_m(v[:,0:1]) )

		U_grad_v = 0.5*( d.map_vx2vy_p(v[:,0:1])*d.dx_p(v[:,1:2]) + d.map_vx2vy_m(v[:,0:1])*d.dx_m(v[:,1:2]) ) +\
                   v[:,1:2]*d.dy(v[:,1:2]) +\
                   0.5*( d.map_vz2vy_p(v[:,2:3])*d.dz_p(v[:,1:2]) + d.map_vz2vy_m(v[:,2:3])*d.dz_m(v[:,1:2]) )

		U_grad_w = 0.5*( d.map_vx2vz_p(v[:,0:1])*d.dx_p(v[:,2:3]) + d.map_vx2vz_m(v[:,0:1])*d.dx_m(v[:,2:3]) ) +\
                   0.5*( d.map_vy2vz_p(v[:,1:2])*d.dy_p(v[:,2:3]) + d.map_vy2vz_m(v[:,1:2])*d.dy_m(v[:,2:3]) ) +\
                   v[:,2:3]*d.dz(v[:,2:3])

		drhodt  = (rho_new - rho_old)/params.dt
		udrhodx = v[:,0:1]*d.dx(rho_new)
		vdrhody = v[:,1:2]*d.dy(rho_new)
		wdrhodz = v[:,2:3]*d.dz(rho_new)

		temp_p_new_0 = torch.where(p_new == 0) 
		temp_p_old_0 = torch.where(p_old == 0) 
		
		theta_new = T_new*((params.sp/p_new)**(R/cp))
		theta_old = T_old*((params.sp/p_old)**(R/cp))

		dthetadt = (theta_new - theta_old)/params.dt
		dthetadx = d.dx(theta_new)
		dthetady = d.dy(theta_new)
		dthetadz = d.dz(theta_new)

		U_grad_T = v[:,0:1]*d.dx(T_new) + v[:,1:2]*d.dx(T_new) + v[:,2:3]*d.dx(T_new)
		U_grad_p = v[:,0:1]*d.dx(p_new) + v[:,1:2]*d.dx(p_new) + v[:,2:3]*d.dx(p_new)
#}}}

		loss_nav_h =  torch.mean(loss_function(flow_mask_mac[:,0:1]*\
                      dudt + U_grad_u + d.dx_m(p_new)/rho_new - (mu/rho_new + SPONGE_h)*d.laplace(v[:,0:1]))[:,:,1:-1,1:-1,1:-1],dim=(1,2,3,4)) +\
                      torch.mean(loss_function(flow_mask_mac[:,1:2]*\
                      dvdt + U_grad_v + d.dy_m(p_new)/rho_new - (mu/rho_new + SPONGE_h)*d.laplace(v[:,1:2]))[:,:,1:-1,1:-1,1:-1],dim=(1,2,3,4))

		loss_nav_z = torch.mean(loss_function(flow_mask_mac[:,2:3]*\
                     dwdt + U_grad_w + g           + d.dz_m(p_new)/rho_new - (mu/rho_new + SPONGE_z)*d.laplace(v[:,2:3]))[:,:,1:-1,1:-1,1:-1],dim=(1,2,3,4))

#		loss_nav =  torch.mean(loss_function(flow_mask_mac[:,0:1]*\
#                    dudt + U_grad_u - f*v[:,1:2]  + d.dx_m(p_new)/rho_new - (mu/rho_new)*d.laplace(v[:,0:1]))[:,:,1:-1,1:-1,1:-1],dim=(1,2,3,4)) +\
#                    torch.mean(loss_function(flow_mask_mac[:,1:2]*\
#                    dvdt + U_grad_v + f*v[:,0:1]  + d.dy_m(p_new)/rho_new - (mu/rho_new)*d.laplace(v[:,1:2]))[:,:,1:-1,1:-1,1:-1],dim=(1,2,3,4)) +\
#                    torch.mean(loss_function(flow_mask_mac[:,2:3]*\
#                    dwdt + U_grad_w + g           + d.dz_m(p_new)/rho_new - (mu/rho_new)*d.laplace(v[:,2:3]))[:,:,1:-1,1:-1,1:-1],dim=(1,2,3,4))

		loss_mass   = torch.mean(loss_function( flow_mask *\
                      drhodt + udrhodx + vdrhody + wdrhodz + d.dx(v[:,0:1]) + d.dy(v[:,1:2]) + d.dz(v[:,2:3])
                      ),dim=(1,2,3,4))

		loss_thermal = torch.mean(loss_function( flow_mask * dthetadt + v[:,0:1]*dthetadx + v[:,1:2]*dthetady + v[:,2:3]*dthetadz),dim=(1,2,3,4))
		#loss_thermal = torch.mean(loss_function( cp*U_grad_T + U_grad_p/rho_new ), dim=(1,2,3,4))

		# combine loss terms for boundary conditions / Navier Stokes equations
		loss = params.loss_bound           * loss_V_bound +\
               20 * 1                      * loss_p_bound +\
               20 * 1                      * loss_T_bound +\
               1  * 1                      * loss_nav_h +\
               40 * 1                      * loss_nav_z +\
               50 * 1                      * loss_mass +\
               1  * 1                      * loss_thermal
		
		# evt put some extra loss on the mean of the vector potential
		if params.loss_mean_a != 0:
			loss_mean_a = torch.mean(a_new,dim=(1,2,3,4))**2
			loss = loss + params.loss_mean_a*loss_mean_a
		
		# evt put some extra loss on the mean of the pressure field
		if params.loss_mean_p != 0:
			loss_mean_p = torch.mean(p_new,dim=(1,2,3,4))**2
			loss = loss + params.loss_mean_p*loss_mean_p
		
		# evt regularize gradient of pressure field (might be useful for very high Reynolds numbers)
		if params.regularize_grad_p != 0:
			regularize_grad_p = torch.mean((dx_right(p_new)**2+dy_bottom(p_new)**2)[:,:,2:-2,2:-2,2:-2],dim=(1,2,3,4))
			loss = loss + params.regularize_grad_p*regularize_grad_p
		
		if params.loss == "log_square" or params.loss == "exp_square":
			loss = torch.mean(loss)
		elif params.loss=='square' or params.loss=='abs':
			loss = torch.mean(torch.log(loss))
		
		# compute gradients for model parameters
		optimizer.zero_grad()
		loss = loss*params.loss_multiplier
		loss.backward()
		
		# evt clip gradients
		if params.clip_grad_value is not None:
			torch.nn.utils.clip_grad_value_(pde_cnn.parameters(),3*params.clip_grad_value)
		
		if params.clip_grad_norm is not None:
			torch.nn.utils.clip_grad_norm_(pde_cnn.parameters(),params.clip_grad_norm)
		
		# perform optimization step on model
		optimizer.step()
		
		# update dataset with predicted fluid state in order to fill up dataset with more and more realistic fluid states
		dataset.tell(toCpu(v_new),toCpu(p_new),toCpu(T_new))
		
		# log losses
		loss = toCpu(loss).numpy()
		loss_V_bound = toCpu(torch.mean(loss_V_bound)).numpy()
		loss_p_bound = toCpu(torch.mean(loss_p_bound)).numpy()
		loss_T_bound = toCpu(torch.mean(loss_T_bound)).numpy()
		loss_nav_h   = toCpu(torch.mean(loss_nav_h)).numpy()
		loss_nav_z   = toCpu(torch.mean(loss_nav_z)).numpy()
		loss_mass    = toCpu(torch.mean(loss_mass)).numpy()
		loss_thermal = toCpu(torch.mean(loss_thermal)).numpy()
		
		if i%1 == 0:
			logger.log(f"loss_{params.loss}",loss,epoch*params.n_batches_per_epoch+i)
			logger.log(f"loss_V_bound_{params.loss}",loss_V_bound,epoch*params.n_batches_per_epoch+i)
			logger.log(f"loss_p_bound_{params.loss}",loss_p_bound,epoch*params.n_batches_per_epoch+i)
			logger.log(f"loss_T_bound_{params.loss}",loss_T_bound,epoch*params.n_batches_per_epoch+i)
			logger.log(f"loss_nav_h_{params.loss}",loss_nav_h,epoch*params.n_batches_per_epoch+i)
			logger.log(f"loss_nav_z_{params.loss}",loss_nav_z,epoch*params.n_batches_per_epoch+i)
			logger.log(f"loss_mass_{params.loss}",loss_mass,epoch*params.n_batches_per_epoch+i)
			logger.log(f"loss_thermal_{params.loss}",loss_thermal,epoch*params.n_batches_per_epoch+i)
			
			if params.loss_mean_a != 0:
				loss_mean_a = toCpu(torch.mean(loss_mean_a)).numpy()
				logger.log(f"loss_mean_a",loss_mean_a,epoch*params.n_batches_per_epoch+i)
			
			if params.loss_mean_p != 0:
				loss_mean_p = toCpu(torch.mean(loss_mean_p)).numpy()
				logger.log(f"loss_mean_p",loss_mean_p,epoch*params.n_batches_per_epoch+i)
			
			if params.regularize_grad_p != 0:
				regularize_grad_p = toCpu(torch.mean(regularize_grad_p)).numpy()
				logger.log(f"regularize_grad_p",regularize_grad_p,epoch*params.n_batches_per_epoch+i)
		
		if i%1 == 0:
			print(f"{epoch}: i:{i}: loss: {loss}; loss_V_bound: {loss_V_bound}; loss_p_bound: {loss_p_bound}; loss_T_bound: {loss_T_bound}; loss_nav_h: {loss_nav_h}; loss_nav_z: {loss_nav_z}; loss_mass: {loss_mass}; loss_thermal: {loss_thermal};")
	
	# save model / optimizer states
	if params.log:
		logger.save_state(pde_cnn,optimizer,epoch+1)
