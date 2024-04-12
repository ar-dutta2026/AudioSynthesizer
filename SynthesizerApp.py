import tkinter as tk
from tkinter import ttk
import numpy as np
import sounddevice as sa
import math
import itertools
import numpy as np

SR = 44_100 # Sample rate
figsize=(25, 6.25)

from abc import ABC, abstractmethod

class Oscillator(ABC):
    def __init__(self, freq=440, phase=0, amp=1, \
                 sample_rate=44_100, wave_range=(-1, 1)):
        self._freq = freq
        self._amp = amp
        self._phase = phase
        self._sample_rate = sample_rate
        self._wave_range = wave_range
        
        # Properties that will be changed
        self._f = freq
        self._a = amp
        self._p = phase
        
    @property
    def init_freq(self):
        return self._freq
    
    @property
    def init_amp(self):
        return self._amp
    
    @property
    def init_phase(self):
        return self._phase
    
    @property
    def freq(self):
        return self._f
    
    @freq.setter
    def freq(self, value):
        self._f = value
        self._post_freq_set()
        
    @property
    def amp(self):
        return self._a
    
    @amp.setter
    def amp(self, value):
        self._a = value
        self._post_amp_set()
        
    @property
    def phase(self):
        return self._p
    
    @phase.setter
    def phase(self, value):
        self._p = value
        self._post_phase_set()
    
    def _post_freq_set(self):
        pass
    
    def _post_amp_set(self):
        pass
    
    def _post_phase_set(self):
        pass
    
    @abstractmethod
    def _initialize_osc(self):
        pass
    
    @staticmethod
    def squish_val(val, min_val=0, max_val=1):
        return (((val + 1) / 2 ) * (max_val - min_val)) + min_val
    
    @abstractmethod
    def __next__(self):
        return None
    
    def __iter__(self):
        self.freq = self._freq
        self.phase = self._phase
        self.amp = self._amp
        self._initialize_osc()
        return self


def get_val(osc, sample_rate=SR):
    return [next(osc) for i in range(sample_rate)]


to_16 = lambda wav, amp: np.int16(wav * amp * (2**15 - 1))
    

class SineOscillator(Oscillator):
    def _post_freq_set(self):
        self._step = (2 * math.pi * self._f) / self._sample_rate
        
    def _post_phase_set(self):
        self._p = (self._p / 360) * 2 * math.pi
        
    def _initialize_osc(self):
        self._i = 0
        
    def __next__(self):
        val = math.sin(self._i + self._p)
        self._i = self._i + self._step
        if self._wave_range != (-1, 1):
            val = self.squish_val(val, *self._wave_range)
        return val * self._a

class SquareOscillator(SineOscillator):
    def __init__(self, freq=440, phase=0, amp=1, \
                 sample_rate=44_100, wave_range=(-1, 1), threshold=0):
        super().__init__(freq, phase, amp, sample_rate, wave_range)
        self.threshold = threshold
    
    def __next__(self):
        val = math.sin(self._i + self._p)
        self._i = self._i + self._step
        if val < self.threshold:
            val = self._wave_range[0]
        else:
            val = self._wave_range[1]
        return val * self._a

class SawtoothOscillator(Oscillator):
    def _post_freq_set(self):
        self._period = self._sample_rate / self._f
        self._post_phase_set
        
    def _post_phase_set(self):
        self._p = ((self._p + 90)/ 360) * self._period
    
    def _initialize_osc(self):
        self._i = 0
    
    def __next__(self):
        div = (self._i + self._p )/self._period
        val = 2 * (div - math.floor(0.5 + div))
        self._i = self._i + 1
        if self._wave_range != (-1, 1):
            val = self.squish_val(val, *self._wave_range)
        return val * self._a

class TriangleOscillator(SawtoothOscillator):
    def __next__(self):
        div = (self._i + self._p)/self._period
        val = 2 * (div - math.floor(0.5 + div))
        val = (abs(val) - 0.5) * 2
        self._i = self._i + 1
        if self._wave_range != (-1, 1):
            val = self.squish_val(val, *self._wave_range)
        return val * self._a


class ADSREnvelope:
    def __init__(self, attack_duration=0.05, decay_duration=0.2, sustain_level=0.7, \
                 release_duration=0.3, sample_rate=SR):
        self.attack_duration = attack_duration
        self.decay_duration = decay_duration
        self.sustain_level = sustain_level
        self.release_duration = release_duration
        self._sample_rate = sample_rate
        
    def get_ads_stepper(self):
        steppers = []
        if self.attack_duration > 0:
            steppers.append(itertools.count(start=0, \
                step= 1 / (self.attack_duration * self._sample_rate)))
        if self.decay_duration > 0:
            steppers.append(itertools.count(start=1, \
            step=-(1 - self.sustain_level) / (self.decay_duration  * self._sample_rate)))
        while True:
            l = len(steppers)
            if l > 0:
                val = next(steppers[0])
                if l == 2 and val > 1:
                    steppers.pop(0)
                    val = next(steppers[0])
                elif l == 1 and val < self.sustain_level:
                    steppers.pop(0)
                    val = self.sustain_level
            else:
                val = self.sustain_level
            yield val
    
    def get_r_stepper(self):
        val = 1
        if self.release_duration > 0:
            release_step = - self.val / (self.release_duration * self._sample_rate)
            stepper = itertools.count(self.val, step=release_step)
        else:
            val = -1
        while True:
            if val <= 0:
                self.ended = True
                val = 0
            else:
                val = next(stepper)
            yield val
    
    def __iter__(self):
        self.val = 0
        self.ended = False
        self.stepper = self.get_ads_stepper()
        return self
    
    def __next__(self):
        self.val = next(self.stepper)
        return self.val
        
    def trigger_release(self):
        self.stepper = self.get_r_stepper()

def amp_mod(init_amp, env):
    return env * init_amp

def freq_mod(init_freq, env, mod_amt=0.01, sustain_level=0.7):
    return init_freq + ((env - sustain_level) * init_freq * mod_amt)


class ModulatedOscillator:
    def __init__(self, oscillator, *modulators, amp_mod=None, freq_mod=None, phase_mod=None):
        self.oscillator = oscillator
        self.modulators = modulators # list
        self.amp_mod = amp_mod
        self.freq_mod = freq_mod
        self.phase_mod = phase_mod
        self._modulators_count = len(modulators)
    
    def __iter__(self):
        iter(self.oscillator)
        [iter(modulator) for modulator in self.modulators]
        return self
    
    def _modulate(self, mod_vals):
        if self.amp_mod is not None:
            new_amp = self.amp_mod(self.oscillator.init_amp, mod_vals[0])
            self.oscillator.amp = new_amp
            
        if self.freq_mod is not None:
            if self._modulators_count == 2:
                mod_val = mod_vals[1]
            else:
                mod_val = mod_vals[0]
            new_freq = self.freq_mod(self.oscillator.init_freq, mod_val)
            self.oscillator.freq = new_freq
            
        if self.phase_mod is not None:
            if self._modulators_count == 3:
                mod_val = mod_vals[2]
            else:
                mod_val = mod_vals[-1]
            new_phase = self.phase_mod(self.oscillator.init_phase, mod_val)
            self.oscillator.phase = new_phase
    
    def trigger_release(self):
        tr = "trigger_release"
        for modulator in self.modulators:
            if hasattr(modulator, tr):
                modulator.trigger_release()
        if hasattr(self.oscillator, tr):
            self.oscillator.trigger_release()
            
    @property
    def ended(self):
        e = "ended"
        ended = []
        for modulator in self.modulators:
            if hasattr(modulator, e):
                ended.append(modulator.ended)
        if hasattr(self.oscillator, e):
            ended.append(self.oscillator.ended)
        return all(ended)

    def __next__(self):
        mod_vals = [next(modulator) for modulator in self.modulators]
        self._modulate(mod_vals)
        return next(self.oscillator)

def gettrig(gen, downtime, sample_rate=SR):
    gen = iter(gen)
    down = int(downtime * sample_rate)
    vals = get_val(gen, down)
    gen.trigger_release()
    while not gen.ended:
        vals.append(next(gen))
    return vals

def getadsr(a, d, sl, sd, r, Osc=None, mod = None):
    if mod is None:
        mod = ModulatedOscillator(
            Osc,
            ADSREnvelope(a,d,sl,r),
            amp_mod=amp_mod
        )
    #sd = 0.4
    downtime = a + d + sd
    return gettrig(mod, downtime)





# Main GUI application
class OscillatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Synthesizer")
        self.geometry("600x400")
        global attack_var, decay_var, release_var, sustain_var, waveform_selected, note_selected
        attack_var = tk.DoubleVar() #removed value= 1, because it defaulted to that value and slider value didnt update
        decay_var = tk.DoubleVar()
        sustain_var = tk.DoubleVar()
        release_var = tk.DoubleVar()
        waveform_selected = tk.StringVar()
        note_selected = tk.StringVar()


        

        self.waveform_selected_var = tk.StringVar()
        # Create the waveform selector Combobox
        ttk.Label(self, text="Waveform:").grid(row=0, column=0, padx=5, pady=5)
        self.waveform_selector = ttk.Combobox(self, textvariable=self.waveform_selected_var, values=["Sine", "Square", "Triangle", "Sawtooth"])
        self.waveform_selector.grid(row=0, column=1, padx=5, pady=5)
        self.waveform_selector.bind("<<ComboboxSelected>>", self.on_waveform_selected)

        # Create the note selector Combobox
        self.note_selected_var = tk.StringVar()
        ttk.Label(self, text="Notes:").grid(row=1, column=0, padx=5, pady=5)
        self.note_selector = ttk.Combobox(self, textvariable=self.note_selected_var)
        self.note_selector['values'] = ["C", "D", "E", "F", "G", "A", "B", "C2"]
        self.note_selector.grid(row=1, column=1, padx=5, pady=5)
        self.note_selector.bind("<<ComboboxSelected>>", self.on_note_selected)

        # button
        play_button = tk.Button(self, text="Play Sound", command= self.play2)
        play_button.grid(row=4, column=0, padx=5, pady=5)

        # Attack control
        ttk.Label(self, text="Attack:").grid(row=0, column=2, padx=5, pady=5)
        tk.Scale(self, from_=0, to_=1,resolution=0.01, variable=attack_var, orient='horizontal').grid(row=1, column=2, padx=5, pady=5)

        # Decay control
        ttk.Label(self, text="Decay:").grid(row=2, column=2, padx=5, pady=5)
        tk.Scale(self, from_=0, to_=1, resolution=0.01, variable=decay_var, orient='horizontal').grid(row=3, column=2, padx=5, pady=5)

        # Sustain control
        ttk.Label(self, text="Sustain:").grid(row=4, column=2, padx=5, pady=5)
        tk.Scale(self, from_=0, to_=1, resolution=0.01, variable=sustain_var, orient='horizontal').grid(row=5, column=2, padx=5, pady=5)

        # Release
        ttk.Label(self, text="Release:").grid(row=6, column=2, padx=5, pady=5)
        tk.Scale(self, from_=0, to_=1, resolution=0.01, variable=release_var, orient='horizontal').grid(row=7, column=2, padx=5, pady=5)

        #attack_var = attack_var.get()
        #decay_var = decay_var.get()
        #sustain_var = sustain_var.get()
        #release_var= release_var.get()

        #change the 0.7 value

    def on_waveform_selected(self, event):
        # This method is called when the user selects an item from the dropdown
        global waveform_selected
        waveform_selected = self.waveform_selected_var.get()

    def on_note_selected(self, event):
        # Using the global variable
        global note_selected
        note_selected = self.note_selected_var.get()
    


    def play2(self):
        if(waveform_selected == "Sine" and note_selected == "C"):
            osc = SineOscillator(262)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Sine" and note_selected == "D"):
            osc = SineOscillator(294)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Sine" and note_selected == "E"):
            osc = SineOscillator(330)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Sine" and note_selected == "F"):
            osc = SineOscillator(349)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Sine" and note_selected == "G"):
            osc = SineOscillator(392)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Sine" and note_selected == "A"):
            osc = SineOscillator(440)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Sine" and note_selected == "B"):
            osc = SineOscillator(494)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Sine" and note_selected == "C2"):
            osc = SineOscillator(523)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()



        if(waveform_selected == "Square" and note_selected == "C"):
            osc = SquareOscillator(262)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Square" and note_selected == "D"):
            osc = SquareOscillator(294)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()
                
        if(waveform_selected == "Square" and note_selected == "E"):
            osc = SquareOscillator(330)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Square" and note_selected == "F"):
            osc = SquareOscillator(349)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Square" and note_selected == "G"):
            osc = SquareOscillator(392)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Square" and note_selected == "A"):
            osc = SquareOscillator(440)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()
        
        if(waveform_selected == "Square" and note_selected == "B"):
            osc = SquareOscillator(494)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Square" and note_selected == "C2"):
            osc = SquareOscillator(523)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()


        if(waveform_selected == "Triangle" and note_selected == "C"):
            osc = TriangleOscillator(262)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Triangle" and note_selected == "D"):
            osc = TriangleOscillator(294)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Triangle" and note_selected == "E"):
            osc = TriangleOscillator(330)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Triangle" and note_selected == "F"):
            osc = TriangleOscillator(349)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Triangle" and note_selected == "G"):
            osc = TriangleOscillator(392)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Triangle" and note_selected == "A"):
            osc = TriangleOscillator(440)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()


        if(waveform_selected == "Triangle" and note_selected == "B"):
            osc = TriangleOscillator(494)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()


        if(waveform_selected == "Triangle" and note_selected == "C2"):
            osc = TriangleOscillator(523)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()



        if(waveform_selected == "Sawtooth" and note_selected == "C"):
            osc = SawtoothOscillator(262)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait() 


        if(waveform_selected == "Sawtooth" and note_selected == "D"):
            osc = SawtoothOscillator(294)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Sawtooth" and note_selected == "E"):
            osc = SawtoothOscillator(330)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Sawtooth" and note_selected == "F"):
            osc = SawtoothOscillator(349)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()


        if(waveform_selected == "Sawtooth" and note_selected == "G"):
            osc = SawtoothOscillator(392)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()


        if(waveform_selected == "Sawtooth" and note_selected == "A"):
            osc = SawtoothOscillator(440)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Sawtooth" and note_selected == "B"):
            osc = SawtoothOscillator(494)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()

        if(waveform_selected == "Sawtooth" and note_selected == "C2"):
            osc = SawtoothOscillator(523)
            mod = ModulatedOscillator(
                osc, #for oscillator
                ADSREnvelope(attack_var.get(),decay_var.get(),sustain_var.get(),release_var.get()), #ADSR envelope
                amp_mod=amp_mod,
                freq_mod=lambda env, init_freq: freq_mod(env, init_freq, mod_amt=1, sustain_level=sustain_var.get())
            )
            vals = getadsr(attack_var.get(), decay_var.get(), sustain_var.get(), 0.4, release_var.get(), Osc = osc, mod=mod)
            vals = np.array(vals) 
            vals = to_16(vals, 0.1)
            sa.play(vals, SR)
            sa.wait()









if __name__ == "__main__":
    osc_classes = {
        "Sine": SineOscillator,
        "Square": SquareOscillator,
        "Sawtooth": SawtoothOscillator,
        "Triangle": TriangleOscillator
    }
    
    app = OscillatorApp()
    app.mainloop()
