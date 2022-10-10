# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt

class USM_SIM:
    def __init__(self, m = 196e-3, d = 500, c = 12e9, cd = 10e-9, rd = 1e-6, A = 0.3,
               h =  4e-3, r = 30e-3, k_r = 4e7, mu = 0.2,v_amp = 175, F_n = 170,
               T = 0, phase =1, tempMin = 20,  r_b = 30e-3, r_a = 15e-3,l = 15e-3,
               ru = 8400, sigma = 5.67e-8, h_c = 60, e = 0.8, s = 0.5e3, dt = 0,
              freq = 40, temp = 20, targetSpeed = 100, N = 9,v_amp_def=175):

   
        self.C = c                            # Modal Mass
        self.D = d                            # Modal Damping
        self.M = m                            # Modal Stiffness
        self.Cd = cd                          # Damped Capacitance
        self.A00 = A                           # Coupling Factor
        self.N = N                            # Vibration Mode
        self.Mu = mu                          # Friction Coefficient
        self.k_r = k_r                        # Rotor Stiffness
        self.rd = rd                          # Damped Resistance
        self.r = r                            # Stator Radius
        self.h = h                            # Height of contact surface
        self.tempMin = tempMin                # Ambient Temperature
        self.temp = temp                      # Current Temperature
        self.dt = dt                          # Time Step 
        self.targetSpeed = targetSpeed        # Initial target speed
        self.v_amp_def   = v_amp_def          # Default Voltage Amplitude 
    
        # Heat Trasnfer Parameters
        self.sigma = sigma                    # Boltezman constant             
        self.e = e                            # Emissivity
        self.s  = s                           # Heat Capacitance
        self.h_c = h_c                        # Convection Coefficient
    
        self.area = 2*np.pi*((r_b+r_a)*l + (r_b**2 - r_a**2)) # Stator Area
        self.mass = ru*np.pi*(r_b**2-r_a**2)*l              # Stator Mass
    
        # Control Variables
        self.freq = freq                      # Driving Frequency
        self.v_amp = v_amp                    # Voltage Amplitude
        self.F_n = F_n                        # Preload
        self.T = T                            # Load Torque
        self.phase = 1                        # Phase Difference
    
        # Zero initialization 
        self.lastSpeed = 0.0
        self.freq_action = 0
        self.pLoss = 0.0
        self.lastW0 = 0
        self.speed_noise = 0
        self.action_noise = 0
        
        self.vary_param(np.zeros(6))
    
        # Setup the min-max scaler for input-state
    
        # State ==> -- freq ,torque, FBV ,speed    
        self.max_scaler = np.array([45, 1 , 90, 300])
        self.min_scaler = np.array([39, 0 ,  50, 0])
        self.mean_scaler = (self.max_scaler + self.min_scaler)/2

    def scale_obs(self,o):
        return (o - self.mean_scaler)/(self.max_scaler - self.min_scaler)*2
  
    # Sample from Beta dist [0-1] and rescale [x_min,x_max]
    # (0.7,0.7) params give higher sampling rate of extreme states
    def sample_beta(self, x_min,x_max,a=0.7,b=0.7):
      return x_min + np.random.beta(a,b)*(x_max-x_min)
  
  
    # Sample from a unifrom dist with some extension and clip
    def sample_uniform(self,x_min,x_max,extend=0.01):
      x = x_min + (-extend+np.random.random()*(1+2*extend))*(x_max-x_min)
      return np.clip(x,x_min,x_max)
  
    # Temperature model
    # 1) Estimate heat trasnfer from convection (h*A*dT) and radiation 
    # 2) The temperature rise is calcuated from heat capacity, mass.
    def update_temp(self,pLoss=None):
      #self.h_c = self.h_c_0 + 1*(self.v_amp - self.v_amp_def) 
      if pLoss == None:
        pLoss = self.pLoss
      #pCond = 50*self.contact_area/10e-3*(self.temp - 25)
      pConv = self.h_c*self.area*(self.temp-self.tempMin)
      pRad  = self.sigma*self.e*self.area*((self.temp+273)**4 - (self.tempMin+273)**4)
      powerFlow = pLoss - pConv #- pRad
      tempRate = powerFlow/self.mass/self.s
      self.temp += self.dt*tempRate
      self.temp = np.clip(self.temp,self.tempMin,60)
      return self.get_state()
  
    # Optimization function for vibration amplitude
    # 1) Calculate rotor height (wf) and contact angle (thetaB)
    # 2) Calculate slip/stick angle (thetaA)
    # 3) Calculate contact forces (F_N, F_T) 
    # 4) Return energy error. 
    def func_amp(self,w0,*args):
      return_all, = args
      if w0 < -self.wf:
        theta_b = np.pi
        wf = self.wf
      else:
        theta_b = opt.brenth(self.func_thetaB,1e-7,np.pi,args=(w0))
        wf = w0*np.cos(theta_b)
      try:  
        theta_a = opt.brenth(self.func_thetaA,1e-7,np.pi/2,args=(w0,theta_b,wf))
      except:
        theta_a = np.pi/2
      F_N = self.k_r*(w0*(np.sin(2*theta_b)/2 + theta_b) - 2*wf*np.sin(theta_b))
      F_T = 2*self.N*self.mu*self.k_r*(self.h/self.r) *\
            (w0*(np.sin(2*theta_a)/2 -np.sin(2*theta_b)/4 + theta_a - theta_b/2) -\
            wf*(2*np.sin(theta_a)-np.sin(theta_b)))
      err= (self.A*self.v_amp)**2 - ((self.c-self.m*self.omega**2)*w0 + F_N)**2 -\
                                    (self.d*self.omega*w0 + F_T)**2
      if return_all:
        return [theta_a, theta_b, wf, F_N, F_T, err]
      else:
        return err 
    
    # Optimization function for slip/stick angle (thetaA)
    def func_thetaA(self,theta_a,*args):
      w0,theta_b,wf = args
      return self.T+self.T_max -\
       4*self.mu*self.k_r*self.r*(w0*np.sin(theta_a)-wf*theta_a)
  
    # Optimization function for contact angle (thetaB)
    def func_thetaB(self,theta_b,*args):
      w0, = args
      return self.F_n -\
       2*self.k_r*w0*(np.sin(theta_b)-theta_b*np.cos(theta_b))
  
    # Get system state at a given frequency
    def get_state(self,freq=None,return_all = False):
    
      # Get frequency and anuglar frequency
      if freq != None:    
          self.freq = freq
      freq_noise = self.freq + self.action_noise*np.random.normal(0,1)
      self.omega = 2*np.pi*(freq_noise)*1000

      # Update temperature dependent parameters
      self.c = self.c0*(1-0.0075*np.sqrt(self.temp-self.tempMin))
      self.d = self.d0*(1-0.04*np.sqrt(self.temp-self.tempMin) + 0.0145*(self.v_amp-self.v_amp_def))

      self.m = self.m0

      # Calculate vibration amplitude through bounded optimization
      self.wf = -self.F_n/(2*np.pi*self.k_r)
      omega_n = np.sqrt((self.c + np.pi*self.k_r)/self.m)

      self.w0,r = opt.brenth(self.func_amp,1e-9,1e-5,args=(False),full_output=True)
      # Calculate other states given vibration amplitude
      self.theta_a, self.theta_b, self.wf, self.F_N, self.F_T, err = self.func_amp(self.w0,True)
      # Hardcode hystresis based on last state
      cond = ~((self.omega < (1.02 - 0.00025*(self.v_amp-self.v_amp_def) - 0.00125*(self.temp - self.tempMin))*omega_n) & (self.lastSpeed < 1e-50))

      self.w0 = self.w0*cond 
      self.speed = self.N*60*self.omega*self.h*self.w0/(2*np.pi*self.r**2)*np.cos(self.theta_a)*(self.F_n > 0)* (self.T < self.F_n*self.mu*self.r)
      self.lastSpeed = self.speed
      theta0 = np.arctan2(self.d*self.omega*(self.w0 + 0.2e-6*~cond) + self.F_T, (self.c-self.m*self.omega**2)*(self.w0 + 0.2e-6*~cond) + self.F_N)
      if abs(err) > 1:
        print('freq: {}, speed: {}, FN: {}, FT: {}, w0: {}, wf: {},  theta_a: {}, err: {}'.format(freq,self.speed,self.F_N,self.F_T,self.w0,self.wf,self.theta_a,err))
      I = 1j*self.omega*(self.w0 + 0.2e-6*~cond)*np.exp(-1j*theta0)*self.A + (self.rd + 1j*self.omega*self.cd)*self.v_amp
      I_real = self.omega*self.A*(self.w0 + 0.2e-6*~cond)*np.sin(theta0) + self.rd*self.v_amp
      I_img = self.omega*(self.A*(self.w0 + 0.2e-6*~cond)*np.cos(theta0) + self.cd*self.v_amp)
      I_amp = np.abs(I)/np.sqrt(2)
      I_phase = (np.angle(I,deg=True)  + 2 * 180) % (2 * 180)
      pIn = 2*(self.v_amp/np.sqrt(2))*I_amp*np.cos(I_phase*np.pi/180)
      pOut = (self.speed/60*2*np.pi)*self.T
      FBV = (self.w0 + 0.2e-6*~cond)*self.A/self.cd/np.sqrt(2)#self.w0*self.A/self.cd*np.
      self.eff = pOut/pIn*100
      self.pLoss = pIn - pOut

      # Add noise to speed measurements
      speed_noise = self.speed + self.speed_noise*np.random.normal(0,1)
      speed_noise = round(speed_noise,2)

      # Calculate energy based on speed error
      self.err = speed_noise - self.targetSpeed
  
      if return_all:
        return np.array([self.theta_a, self.speed, self.w0, FBV , I_phase, pIn, I_amp, self.eff])
      else:
        return np.array([self.freq, self.T, I_phase, self.targetSpeed,self.temp,speed_noise, self.T])

  
    #Use an estimate of Control Lyapunov Function as a reward
    def get_reward(self,action):
      #Higher lambda -> Faster Response / Less Robustness 
      lam = 2   # [1-5]
      L = self.energyDiff + lam*self.lastEnergy
      return 1*lam - L
  
    # Reset the environment at the start of every episode
    # Use a beta distribution to increase propability of sampling edge cases for better robustness
    # Sample initial frequency - load torque - temperature - target speed
    # lastW0 -- determines the speed hystresis -- default: the motor is at stall
    def reset(self,freq=None,temp=None, lastSpeed = 0, T = None, targetSpeed=None, return_all = False, ep = 1):
      self.lastSpeed = lastSpeed
      self.freq = self.sample_beta(39,45) if freq == None else freq
      self.temp = self.sample_beta(20,60) if temp == None else temp
      self.targetSpeed = self.sample_beta(0,300) if targetSpeed == None else targetSpeed
      self.T = self.sample_beta(0,1) if T == None else T
      state = self.get_state(self.freq) 
      self.lastEnergy = 1/np.sqrt(300)*np.sqrt(np.abs(self.err))
      return state
  
  
    # Update the driving frequency given the frequency action
    # Clip the frequency within the limits
    # Update the system state and the temperature
    # Calculate the energy based on speed error (Neccessary for reward)
    def step_frequency(self,action):          
        #pi_gain = 0.01*action[0]
        #u =  pi_gain*self.err
        self.freq = self.freq + action[0]*2
        self.freq = np.clip(self.freq,39,45)
  
        state = self.get_state(self.freq)
        state = self.update_temp()
  
        energy =   1/np.sqrt(300)*np.sqrt(np.abs(self.err))
        self.energyDiff = energy - self.lastEnergy
        self.lastEnergy = energy
  
        return state, self.get_reward(action[0]*2)
  
    # Vary some USM parameters under a normal distribuation
    # Useful for studying the robustness of the controller to motor variations
    def vary_param(self,vars=[]):
      if not len(vars):
        vars = np.clip(np.random.normal(0,1,size=(6,)),-1,1)
  
      self.A0 = self.A00*(1+0*.05*vars[0])
      self.A = self.A0*((self.v_amp_def/self.v_amp)**0.3)
      self.c0 = self.C*self.A**2*(1+0.01*vars[1])
      self.d0 = self.D*self.A**2*(1+0.1*vars[2])
      self.m0 = self.M*self.A**2*(1+0.01*vars[3])
      self.cd = self.Cd*(1+0*.05*vars[4])
      self.mu = self.Mu*(1+0*.05*vars[5])
  
      self.T_max = self.F_n*self.mu*self.r
  
      params = np.array([self.A,self.c0,self.d0,self.m0,self.cd,self.mu])
      return params, vars
      
    # Set noise levels on input and output to test controller robustness
    def set_noise(self,speed_noise,action_noise):
      self.speed_noise = speed_noise
      self.action_noise = action_noise 
    
    # Return system state under changes in target speed or torque
    def set_state(self,targetSpeed=None,torque = None):
      if targetSpeed != None:
        self.targetSpeed = targetSpeed
      if torque != None:
        self.T = torque
      return self.get_state(self.freq) 
  
    # Setters and getters to access the USM class variables
    def set_temp(self,temp):
      self.temp = temp
    def get_temp(self):
        return self.temp
  
    def get_torque(self):
        return self.T
    def set_torque(self,T):
        self.T = T
    
  
    def get_targetSpeed(self):
        return self.targetSpeed
    def get_speed(self):
        return self.speed 
  
  
  
if __name__ == "__main__":
    
    def sweep_freq(freqs,return_all = False):
        states = []
        for freq in freqs:
            states.append(USM.get_state(freq,return_all))
        return np.array(states)

    USM = USM_SIM()
    
    fig, axs = plt.subplots(4, 2,figsize=(12,16))
    freqs = np.linspace(45,39,100)
    for T in [0,0.1,0.2,0.5,0.7,1]:
      USM.set_state(torque=T)
      print(T)
      states = sweep_freq(freqs, True)
      axs[0,0].plot(freqs,states[:,0]*180/np.pi, label='Torque: {} N.m'.format(T))
      for i in range(1,states.shape[1]):
        axs[i//2,i%2].plot(freqs,states[:,i], label='Torque: {} N.m'.format(T))
        axs[i//2,i%2].legend()
        axs[i//2,i%2].set_xlabel('Driving Frequency [kHz]')
    #axs[0,0].legend()
    axs[0,0].set_ylabel('Theta_a [deg]')
    axs[0,1].set_ylabel('Speed [rpm]')
    axs[1,0].set_ylabel('Vibration Amplitude [μm]')
    
    axs[1,1].set_ylabel('Current Amplitude [A]')
    axs[2,0].set_ylabel('Admittance Phase [°]')
    axs[2,1].set_ylabel('Input Power [W]')
    axs[3,0].set_ylabel('Current Amplitude [A]')
    axs[3,1].set_ylabel('Driving Efficiency [%]')
    
    USM = USM_SIM()

    plt.figure()
    n = 1000
    freqs_up = np.linspace(39,45,n) 
    freqs_down = np.linspace(45,39,n) 
    
    speeds_up = sweep_freq(freqs_up)[:,-1]
    speeds_down = sweep_freq(freqs_down)[:,-1]
    plt.figure()
    plt.plot(freqs_up,speeds_up,label='Sweep Up')
    plt.plot(freqs_down,speeds_down,label='Sweep Down')
    plt.xlabel('Frequency [kHz]');
    plt.ylabel('Speed [rpm]');
    plt.title('Speed Hystresis [' + str(n) + ' steps]');
    plt.legend();
    
    freqs = np.linspace(45,39,1000)
    temps = [20,30,40,50]
    plt.figure()
    for temp in temps:
      USM.set_temp(temp)
      speeds = sweep_freq(freqs)[:,-1]
      plt.plot(freqs,speeds,label='Temperature = ' + str(temp))
    plt.xlabel('Frequency [kHz]');
    plt.ylabel('Speed [rpm]');
    plt.title('Temperature Drift');
    plt.legend();
    
    dt = 1
    USM.dt = dt
    t = 0
    speeds = []
    times = []
    temps = []
    freq = 41
    USM.reset(temp=20,freq=freq,lastW0=1,T=0)
    for i in range(1000):
      state = USM.get_state(freq)
      speed,temp = state[-1],state[-2]
      temps.append(temp)
      speeds.append(speed)
      times.append(t)
      USM.update_temp()
      t+= dt
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(times, speeds, 'r-')
    ax2.plot(times, temps, 'b--')
    
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Speed [rpm]', color='r')
    ax2.set_ylabel('Temperature [°C]', color='b')
    ax1.set_title('Temperature Drift')
    plt.show()



    