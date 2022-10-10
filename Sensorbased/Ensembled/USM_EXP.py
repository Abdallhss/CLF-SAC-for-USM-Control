# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import time 

import serial
import struct
import pyvisa

class USM_EXP:
    def __init__(self):
        self.ser = serial.Serial()
        self.ser.port="COM1"
        self.ser.timeout=0.002
        self.ser.baudrate = 115200
        rm = pyvisa.ResourceManager()
        #print(rm.list_resources())
        self.multimeter = rm.open_resource('GPIB0::2::INSTR')
        self.temp = (self.multimeter.query_ascii_values("MEASure:TEMPerature:TCOuple?")[0])
        self.targetSpeed = 0
        self.speed = 0
        self.fbV = 0
        self.err = 0
        self.lastEnergy = np.log10(np.abs(-self.targetSpeed)+1)
        self.lastErr = -self.targetSpeed
        self.lastSpeed = 0
        self.fail = 0
        self.freq_action = 0

        self.freq = 42
        self.amp = 2.5
        self.T = 0
        self.u_T = 0
        self.T_measured = 0
        self.u3 = int(0)
        self.u4 = int(0)
        self.scope = rm.open_resource('GPIB0::1::INSTR')
        
        # State ==> -- freq ,torque, FBV , targspeed, temp    
        self.max_scaler = np.array([45, 1 , 300, 60, 300])
        self.min_scaler = np.array([39, 0 ,  0, 20, 0])
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
    
    def to2bytes(self,x):
        bin_x = bin(x)
        #bin_x = bin_x[(bin_x.find('b')+1):]
        n = len(bin_x) - 2
        if n > 8:
            x1 = int(bin_x[-8:],2)
            x2 = int(bin_x[2:-8],2)
        else:

            x1 = int(bin_x[2:],2)
            x2 = 0

        return (x1,x2)
    

    def set_torque(self,T):
        self.T = T
        self.u_T = int(255*(T))
        u_freq = int(65535*(self.freq-39)/(45-39))
        u1,u2 = self.to2bytes(u_freq)
        self.ser.open()
        self.ser.flush()
        self.ser.write(struct.pack('!BBB',u1,u2,self.u_T))
        self.ser.close()
        time.sleep(0.1)
        
    def get_state(self,freq=None,n=15,return_all = False):
        #u = int(255*(freq-39.5)/(45.5-39.5))
        #u = int(255*(freq-40)/(45-40))
        if freq != None:    
            self.freq = freq
        u_freq = int(65535*(self.freq-39)/(45-39))
        u1,u2 = self.to2bytes(u_freq)
        self.ser.open()
        self.ser.flush()
        for i in range(n):
            #self.ser.write(struct.pack('!B',u))
            self.ser.write(struct.pack('!BBB',u1,u2,self.u_T))
            time.sleep(0.001)
            #x = self.ser.read(2)
            #x = self.ser.read(7)
        x = self.ser.read(7)
        try:
            x=struct.unpack('!BBBBBBB',x)
        except:
            self.ser.write(struct.pack('!BBB',u1,u2,self.u_T))
            time.sleep(0.001)
            #x = self.ser.read(2)
            x = self.ser.read(7)
            x=struct.unpack('!BBBBBBB',x)
            if x[1] == 255:
                print('Max',x)
        self.ser.close()
        #print(x)
        #self.freq = 40+x[0]/255*(45-40)
        self.speed = (x[0] + x[1]*256)*500/65535
        self.speed = self.speed*(self.speed < 350)
        
        self.fbV = (x[3])*100/255
        self.fbV = self.fbV*(self.fbV < 80)
    
        self.T_measured = x[2]/255*1.2
        #I_amp = 500e-3*(x[4] + x[5]*256)/65535
        #if I_amp != 0 and I_amp != 255:
        self.I_amp = 500e-3*(x[4] + x[5]*256)/65535
        self.p_in = 2*(x[6]/255)*75/5
        #self.p_in = 2*self.v_amp*self.I_amp*np.cos(self.I_phase*np.pi/180)
        #self.I_phase = 90*x[6]/255
        self.I_phase = np.arccos(self.p_in/(2*self.v_amp*self.I_amp))*180/np.pi
        self.p_out = (self.speed/60*2*np.pi)*self.T
        self.eff = self.p_out/(self.p_in+0.01)*100
        self.pLoss = self.p_in - self.p_out
        self.err = self.speed - self.targetSpeed
        
        if return_all:
            return np.array([self.freq, self.temp, self.speed, self.fbV, self.T, self.T_measured, self.I_amp, self.I_phase, self.p_in, self.eff])
        else:
            return np.array([self.freq, self.T, self.targetSpeed, self.temp, self.speed, self.T_measured])

        
    def start(self, amp,freq):
        self.amp = amp
        self.scope.write(":OUTP1:STATe 1")
        self.scope.write(":OUTP1:SYNC:TYPE SFCT")
        self.scope.write(":SOUR1:FREQ " + str(freq*1000))
        self.v_amp = amp*100/np.sqrt(2)
        amp1 = amp*100/107
        amp2 = amp*100/122
        self.scope.write(":SOUR1:VOLT:AMPL " + str(round(amp1,2)))
        self.scope.write(":SOUR1:SCH:VOLT:AMPL " + str(round(amp2,2)))
        
    def stop(self):
        self.scope.write(":OUTP:STATe 0")
        #self.scope.write(":OUTP1:SYNC:TYPE OFF")
        self.scope.write(":SOUR1:SCH:VOLT:AMPL 0")
        
    def update_temp(self):
        self.temp = (self.multimeter.query_ascii_values("MEASure:TEMPerature:TCOuple?")[0])
        
    def set_targetSpeed(self,speed):
        self.targetSpeed = speed
        return self.get_state(self.freq) 
    
    def reset(self,freq=None,targetSpeed=None,T = None, temp=None,ep=1):
        self.update_temp()
        self.freq = self.sample_beta(39,45) if freq == None else freq
        self.targetSpeed = self.sample_beta(0,300) if targetSpeed == None else targetSpeed
        if ep % 10 == 0:
            self.T = self.sample_beta(0,1,0.7,1) if T == None else T
            self.set_torque(self.T)
        state = self.get_state(self.freq,n=50)
        self.lastEnergy = 1/np.sqrt(300)*np.sqrt(np.abs(self.err))
        return state

    
    def get_reward(self,action):
        lam = 2
        L = self.energyDiff + lam*self.lastEnergy
        return lam - L

    
    def step_frequency(self,action):          
        
      act1 = 2*action[0]
      act2 = 0.1*action[1]*self.err
      act_w = (action[2]+1)/2
      self.freq = self.freq + act1*act_w + act2*(1-act_w)
      self.freq = np.clip(self.freq,39,45)
      
      state = self.get_state(self.freq,n=20)
      
      energy =   1/np.sqrt(300)*np.sqrt(np.abs(self.err))
      self.energyDiff = energy - self.lastEnergy
      self.lastEnergy = energy
      
      #action[0] = (self.freq - lastFreq)/2
      return state, self.get_reward(action[0]*2)
  
    def set_state(self,targetSpeed=None,torque = None):
        if targetSpeed != None:
            self.targetSpeed = targetSpeed
        if (torque != None) and (torque != self.T):
            self.set_torque(torque)
        return self.get_state(self.freq) 
    
    def get_temp(self):
        return self.temp
    def get_amp(self):
        return self.amp
    def get_targetSpeed(self):
        return self.targetSpeed
    def get_speed(self):
        return self.speed
    def get_torque(self):
      return self.T

    def set_noise(self,speed_noise,action_noise):
        self.speed_noise = speed_noise
        self.action_noise = action_noise 
  
  
