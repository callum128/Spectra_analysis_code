#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:08:03 2026

@author: ccl128
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import toml
import plotly.express as px
import plotly.tools as tls
import plotly.offline as pyo


from scipy import sparse
from scipy.sparse import linalg
from numpy.linalg import norm
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
from pathlib import Path

plt.rcParams['font.family'] = 'Nimbus Roman'

SITE_2 = {
    'Jon_emission_names': {
     '1D2-3H4':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Jon Data/600nm-650nm 0.01nm step D2 to H4.dat', 
     '1D2-3H5':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Jon Data/690-720 1D2-3H5.dat', 
     '1D2-3H6':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Jon Data/810-850 1D2-3H6.dat', 
     '3P0-3H5':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Jon Data/540nm-570nm 0.01 step P0 to h5.dat', 
     '3P0-3H6':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Jon Data/598nm-683nm 0.1nm step P0 to H6.dat',
     '3P0-3F2':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Jon Data/598nm-683nm 0.1nm step P0 to H6.dat',
     '3P0-3F3+3F4':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Jon Data/683nm-760nm 0.1nm step P0 to 3F4.dat', 
     },
 'Scope_emission_names':{
     '1D2-3H4':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/1D2-3H4_580.2_emission_scope_amp_13_07_2026_12_03_47_480.toml',
     '1D2-3H5':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/1D2-3H5_606.35_emission_scope_amp_22_06_2026_14_10_14_891.toml',
     '1D2-3H6':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/1D2-3H6_606.45_emission_scope_amp_13_07_2026_14_53_02_724.toml',
     '1D2-3F2':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/1D2-3F2_606.5_emission_scope_amp_bigslits_18_08_2026_10_02_52_585.toml',
     '1D2-3F3+3F4':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/1D2-3F3_606.5_emission_scope_17_08_2026_13_02_34_028.toml',
     '1D2-1G4':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/1D2-1G4_emission_scope_30_06_2026_11_26_28_142.toml', #This may be site 1, but moved -147cm
     '3P0-3H5':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/3P0-3H5_488.3_emission_scope_14_07_2026_14_40_44_075.toml', 
     '3P0-3H6':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/3P0-3H6_488.35_emission_scope_amp_15_07_2026_10_12_10_468.toml',
     '3P0-3F2':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/3P0-3F2_488.35_emission_scope_amp_15_07_2026_13_43_32_484.toml',
     '3P0-3F3+3F4':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/3P0-3F33F4_488.7_emission_scope_amp_21_07_2026_10_38_53_510.toml',
     },
  'em_move_factors': {'1D2-3H4':0.01, '1D2-3H5':1.13, '1D2-3H6':1.68, '1D2-3F2':3.16, '1D2-3F3+3F4':3.16, '1D2-1G4':2.73, '3P0-3H5':-3.14, '3P0-3H6':-2.53, '3P0-3F2':-2.22, 
                '3P0-3F3+3F4':-1.93}, #in nm
 'Scope_excitation_names':{
     '1D2:R590':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Excitation/Site 2/EX_S2_1D2_from_1D2-1G4_excitation_amp_26_08_2026_09_35_06_735.toml',
     '1D2:R610':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Excitation/Site 2/EX_S2_1D2_from_1D2-1G4_excitation_R610dye_26_08_2026_15_05_25_131.toml',
     '3P0:C481':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Excitation/Site 2/EX_S2_3P0_from_1D2-1G4_excitation_C481_27_08_2026_11_36_09_371.toml',
     '1I6:C460':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Excitation/Site 2/EX_S2_1I6_from_1D2-1G4_excitation_C460_07_09_2026_11_15_51_423.toml',
     '3P2:C440':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Excitation/Site 2/EX_S2_3P2_from_1D2-1G4_excitation_C440TEST_07_09_2026_16_06_14_032.toml' #test
     }
         }   

SITE_1 = {'Scope_emission_names':{    
    '1D2-3H4':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/Site1/Site1_1D2-3H4_577.0_emission_scope_amp_24_07_2026_09_50_05_490.toml',
    '1D2-3H5':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/Site1/Site1_1D2-3H5_577.1_emission_scope_amp_27_07_2026_09_24_43_045.toml',
    '1D2-3H6':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/Site1/Site1_1D2-3H6_577.1_emission_scope_bigslits_28_07_2026_10_28_08_353.toml',
    '1D2-3F2':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/Site1/Site1_1D2-3F2_576.9_emission_scope_amp_bigslits_18_08_2026_14_27_48_441.toml',
    '1D2-3F3+3F4':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/Site1/Site1_1D2-3F3_577.1_emission_scope2_31_07_2026_11_23_04_930.toml',
    '1D2-1G4':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/Site1/Site1_1D2-1G4_577.1_emission_scope_03_08_2026_11_09_25_530.toml',
    '3P0-3H5':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/Site1/Site1_3P0-3H5_483.8_emission_scope_amp_14_08_2026_09_58_42_698.toml',
    '3P0-3H6':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/Site1/Site1_3P0-3H6_483.8_emission_scope_amp_14_08_2026_14_37_25_765.toml',
    '3P0-3F2':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/Site1/Site1_3P0-3F2_483.8_emission_scope_amp_smallgates_04_08_2026_11_06_03_392.toml',
    '3P0-3F3+3F4':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Emission/Site1/Site1_3P0-3F3_483.8_emission_scope_amp_05_08_2026_11_14_59_235.toml',
    },
     'em_move_factors' : {'1D2-3H4':0.05, '1D2-3H5':1.08, '1D2-3H6':1.83, '1D2-3F2':3.04, '1D2-3F3+3F4':3.49, '1D2-1G4':3.34, '3P0-3H5':-0.59, '3P0-3H6':-0.37, 
                '3P0-3F2':1.73, '3P0-3F3+3F4':-0.60},
    'Scope_excitation_names':{
        '1D2:R590':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Excitation/Site 1/EX_S1_1D2_from_1D2-1G4_excitationTEST_25_08_2026_14_04_17_027.toml',
        '1D2:R610':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Excitation/Site 1/EX_S1_1D2_from_1D2-1G4_excitation_R610dye_27_08_2026_09_19_18_610.toml',
        '3P0:C481':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Excitation/Site 1/EX_S1_3P0_from_1D2-1G4_excitation_C481_27_08_2026_15_04_57_328.toml',
        '1I6:C460':'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Scope Excitation/Site 1/EX_S1_1I6_from_1D2-1G4_excitation_C460_07_09_2026_13_45_43_970.toml',
        }
        }

ABSORBTION = {'FTIR_absorbtion_names_9':{
    '3H5':['/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/FTIR/20260224_YSO_Pr_MCT_KBr_9K.0.dpt', 1e11, [2050,3540] ],
    '3H6':['/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/FTIR/20260224_YSO_Pr_MCT_KBr_9K.0.dpt', 1e10, [4000,4900] ],
    '3F2':['/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/FTIR/20260224_YSO_Pr_InGaAs_CaF2_9K.0.dpt', 1e10, [4900,5900] ],
    '3F3+3F4':['/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/FTIR/20260224_YSO_Pr_InGaAs_CaF2_9K.0.dpt', 1e10, [5900,7800] ],
    '1G4':['/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/FTIR/20260224_YSO_Pr_InGaAs_CaF2_9K.0.dpt', 1e9, [9450,10450] ],
    '1D2':['/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/FTIR/20260224_YSO_Pr_SiDiode_CaF2_9K.1.dpt', 1e11, [16200, 17700] ],
    '3P0':['/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/FTIR/20260312_YSO_Pr_GaP_CaF2UVVIS_9K_512_0.2res.0.dpt',1e10, [20050, 22800] ]
    }
        }


def loader(site, top, bottom, normalize=False, laser=False, move_factor=0):
    laser_wave = 0
    
    if site == 2:
        filename = SITE_2['Scope_emission_names'][f'{top}-{bottom}']
        
        data = toml.load(filename)
        areas = np.array(data['device']['DPO7104_TekTronix_scope']['data']['area']['data'])
        wavelengths = np.array(data['device']['iHR550']['data']['wavelength (nm)']['data'])
        
        if laser:
            laser_wave = (488.4 + move_factor) if top == '3P0' else 602.6 + move_factor #not laser, from fitting the ground state to zero
            
    elif site == 1:
        filename = SITE_1['Scope_emission_names'][f'{top}-{bottom}']
        
        data = toml.load(filename)
        areas = np.array(data['device']['DPO7104_TekTronix_scope']['data']['area']['data'])
        wavelengths = np.array(data['device']['iHR550']['data']['wavelength (nm)']['data'])
        
        if laser:
            laser_wave = (483.8 + move_factor) if top == '3P0' else 604.5 + move_factor #lowest state, not the highest where the laser was (577), from fitting the ground state to zero
            
    if laser_wave == 0:
        adjusted_wavenumbers = 1.0e7 / (wavelengths)  #+ move_factor
        
    else:
        adjusted_wavenumbers =1e7/ laser_wave *np.ones_like(wavelengths) - 1.0e7 / (wavelengths)  
    
    #print(laser_wave)
    
    if normalize:
        areas = (areas-min(areas))/max(areas)
        
    return adjusted_wavenumbers, areas


def abs_loader(level, normalize=True):
    info = ABSORBTION['FTIR_absorbtion_names_9'][level]
    filename = info[0]
    baseline_lam = info[1]
    fit_range = info[2]
    
    data = np.loadtxt(filename, skiprows=0, delimiter=',')
    x = data[:,0]
    y = data[:,1]
    
    cut_wavenumber, abs_coeff = process_spectrum(x, y, fit_range[0], fit_range[1], baseline_lam, 0.178)
    
    if normalize:
      abs_coeff = (abs_coeff-min(abs_coeff))/max(abs_coeff)
      
    return  cut_wavenumber, abs_coeff


def exc_loader(filename, start, stop, step, n=2):    
    data = toml.load(filename)
    areas = np.array(data['device']['DPO7104_TekTronix_scope']['data']['area']['data'])

    
    wavelengths = np.arange(start, stop + n*step, step)
    print(f'Laser wavelength range: {wavelengths[0]} - {wavelengths[-1]}')

    wavenumbers = 1e7 / wavelengths

    return wavenumbers, areas


def exc_loader2(site, level, dye, normalize=False, move=0):
    if site == 2:
        filename = SITE_2['Scope_excitation_names'][f'{level}:{dye}']

    elif site == 1:
        filename = SITE_1['Scope_excitation_names'][f'{level}:{dye}']

    data = toml.load(filename)
    areas = np.array(data['device']['DPO7104_TekTronix_scope']['data']['area']['data'])
    i_laser = int(data['device']['GL100_Dye_Laser']['initial_position'])
    f_laser = int(data['device']['GL100_Dye_Laser']['end_position'])
    step = float(data['device']['GL100_Dye_Laser']['step_size'])
    
    wavelengths = np.linspace(i_laser, f_laser, len(areas))
   
    expected_len = int(round((f_laser - i_laser) / step)) + 1
    if abs(len(areas) - expected_len) > 2:  # Allow small rounding tolerance
         raise ValueError(f"Data length mismatch. Expected ~{expected_len}, got {len(areas)}")
    

    areas = (areas-min(areas))/max(areas) if normalize else areas

    return 1e7/(wavelengths+move), areas

def UV_loader(level, normalize=False, move=0):
    
        if level == '3P':
            filename = '/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Cary UV/3P_10K_0.csv'
            end = 5000
        
        elif level == '1S0':
            filename = '/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Cary UV/1S0_10K_0.csv'
            end = 4000

        data = np.loadtxt(filename, skiprows=2, delimiter=',', usecols=(0,1), max_rows=end)

        wavelenght = data[:,0]
        transmision = data[:,1]

        base =0 
        alpha = -1/0.178 * np.log((transmision-base)/100)
        wavenumber = 1e7/(wavelenght+move)
        
        alpha = (alpha-min(alpha))/max(alpha) if normalize else alpha
        
        return wavenumber, alpha

def process_spectrum(
    wavenumber,
    intensity,
    fitmin,
    fitmax,
    lam,
    thickness,
    displaymin=None,
    displaymax=None
):

    mask = (wavenumber > fitmin) & (wavenumber < fitmax)

    cut_w = wavenumber[mask]
    cut_i = intensity[mask]

    baseline = -baseline_arPLS(-cut_i, lam)

    abs_coeff = -(1/thickness) * np.log(cut_i / baseline)

    if displaymin is not None and displaymax is not None:

        mask = (cut_w > displaymin) & (cut_w < displaymax)

        cut_w = cut_w[mask]
        abs_coeff = abs_coeff[mask]
    
  

    return cut_w, abs_coeff


def baseline_arPLS(y, lam, ratio=1e-10, niter=10, full_output=False):
    """
    Modified asymetric least squares smoothing algorithm. 
    This may throw warnings, evaluating plot is only way to tell
    if input parameters are valid, will likely exceed input params, 
    do not worry about this. 
    Inspection and tinkering with input values is critical for a baseline. 
    lam input should be large, for the test data "1e10" works well. 
    """
    #print("*** Don't get too worried about 'iterations exceeded', or overflow in exp, etc. ***")
    L = len(y)
    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)
    H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    crit = 1
    count = 0
    while crit > ratio:
        z = linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        w_new = 1 / (1 + np.exp(2 * (d - (2*s - m))/s))
        crit = norm(w_new - w) / norm(w)
        w = w_new
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values
        count += 1
        if count > niter:
            print('Maximum number of iterations exceeded')
            break
    if full_output:
        info = {'num_iter': count, 'stop_criterion': crit}
        return z, d, info
    else:
        return z


def cmnminvert(x):
    x = np.array(x).astype(float)
    near_zero = np.isclose(x, 0)
    x[near_zero] = np.inf
    x[~near_zero] = 1.0e7 / x[~near_zero]
    return x

def detect_peaks(ax, x, y, color='k', alpha=1.0, width=2, distance=5, prominence=0.05, absorp=False):
    peaks, properties = find_peaks(
        y,
        width=width,
        distance=distance,
        prominence=prominence
    )
    
    if not absorp:
        for x, y in zip(x[peaks], y[peaks]):
            ax.annotate(
                f"{x:.1f}",
                (x, y),
                textcoords="offset points",
                xytext=(0,8),
                ha='center',
                fontsize=9,
                color=color,
                alpha=alpha
            )
            
    if absorp:
        for x, y in zip(x[peaks], y[peaks]):
            ax.annotate(
                f"{x:.1f}",
                (x, -y),
                textcoords="offset points",
                xytext=(0,-8),
                ha='center',
                fontsize=9,
                color=color,
                alpha=alpha
            )

def plot_details(fig, ax, no_invert=False):
    ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize = 20, labelpad = 3)
    secax = ax.secondary_xaxis('top', functions=(cmnminvert, cmnminvert))  #to put wavelength on top
    secax.set_xlabel('Wavelength (nm)', fontsize = 20, labelpad = 9)
    secax.tick_params(axis = 'x', labelsize = 20, width = 1, length = 7)
    ax.set_ylabel('Normalised Intensity (arb. units)', fontsize = 20, labelpad = 3)
    ax.tick_params(axis = 'x', labelsize = 18, width = 1, length = 7)
    ax.tick_params(axis = 'y', labelsize = 18, width = 1, length = 7)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    if not no_invert:
        ax.invert_xaxis()
        secax.invert_xaxis()
    
    #plt.legend(fontsize='small',bbox_to_anchor=(1.0, 1.0))
    plt.legend(fontsize='small')
    plt.tight_layout()


def super_plot(site, ax, laser_fit=True):
    
    if site == 2:
        bottom_states_from_1D2 = ['3H4', '3H5', '3H6', '3F2', '3F3+3F4', '1G4']
        bottom_states_from_3P0 = ['3H5', '3H6', '3F2', '3F3+3F4']
        move_factors = {'1D2-3H4':0.01, '1D2-3H5':1.13, '1D2-3H6':1.68, '1D2-3F2':3.16, '1D2-3F3+3F4':3.16, '1D2-1G4':2.73, '3P0-3H5':-3.14, '3P0-3H6':-2.53, '3P0-3F2':-2.22, 
                        '3P0-3F3+3F4':-1.93} #in nm
        
        end_title = 'Moved' if laser_fit else 'Raw'
        title = f'Super Plot of Pr:YSO Site 2 (CN 7) Emission ({end_title})'
        
    else:
        bottom_states_from_1D2 = ['3H4', '3H5', '3H6', '3F2', '3F3+3F4', '1G4']
        bottom_states_from_3P0 = ['3H5', '3H6', '3F2', '3F3+3F4']
        move_factors = {'1D2-3H4':0.05, '1D2-3H5':1.08, '1D2-3H6':1.83, '1D2-3F2':3.04, '1D2-3F3+3F4':3.49, '1D2-1G4':3.34, '3P0-3H5':-0.59, '3P0-3H6':-0.37, 
                        '3P0-3F2':1.73, '3P0-3F3+3F4':-0.60} #in nm
        
        end_title = 'Moved' if laser_fit else 'Raw'
        title = f'Super Plot of Pr:YSO Site 1 (CN 6) Emission ({end_title})'

    for bottom in bottom_states_from_1D2:
        top = '1D2'
        normalize = True
        name_end = ' (bigslits)' if bottom == '3F2' else ''
        name = f'Site {site} {top}-{bottom}'+name_end
        move = move_factors[f'{top}-{bottom}']
        waves, areas = loader(site, top, bottom, normalize, laser_fit, move)
        line, = ax.plot(waves, areas, label=name)
        color = line.get_color()
        detect_peaks(ax, waves, areas, color)
        
    for bottom in bottom_states_from_3P0:
        top = '3P0'
        normalize = True
        name = f'Site {site} {top}-{bottom} (includes overlaping 1D2-X)'
        move = move_factors[f'{top}-{bottom}']
        waves, areas = loader(site, top, bottom, normalize, laser_fit, move)
        alpha = 0.3
        line, = ax.plot(waves, areas, label=name, alpha=alpha)
        color = line.get_color()
        detect_peaks(ax, waves, areas, color, alpha)

    
    if laser_fit:
        scale = -0.2
        loaders = ['3H5', '3H6', '3F2', '3F3+3F4', '1G4']
        
        for i, load_id in enumerate(loaders):
            label = '9K Absorption' if i == 0 else None
            
            cut_w, abs_c = abs_loader(load_id)
            ax.plot(cut_w, scale * abs_c, 'k', label=label)
            detect_peaks(ax, cut_w, -scale * abs_c, absorp=True)
            
    ax.set_title(title)
    return title

def all_plot(ax):
    bottom_states_from_1D2 = ['3H4', '3H5', '3H6', '3F2', '3F3+3F4', '1G4']
    bottom_states_from_3P0 = ['3H5', '3H6', '3F2', '3F3+3F4']
    laser_fit=False
    
    label_formats = {
        '1D2-3H4': r'$^1D_2 \rightarrow ^3H_4$',
        '1D2-3H5': r'$^1D_2 \rightarrow ^3H_5$',
        '1D2-3H6': r'$^1D_2 \rightarrow ^3H_6$',
        '1D2-3F2': r'$^1D_2 \rightarrow ^3H_4$ (bigslits)',
        '1D2-3F3+3F4': r'$^1D_2 \rightarrow ^3F_3 + ^3F_4$ (includes 1G4-3H4?)',
        '1D2-1G4': r'$^1D_2 \rightarrow ^1G_4$ (includes 1G4-3H5?)',
        '3P0-3H5': r'$^3P_0 \rightarrow ^3H_5$',
        '3P0-3H6': r'$^3P_0 \rightarrow ^3H_6$ (includes 1D2-3H4)',
        '3P0-3F2': r'$^3P_0 \rightarrow ^3F_2$ (includes 1D2-3H4)',
        '3P0-3F3+3F4': r'$^3P_0 \rightarrow ^3F_3 + ^3F_4$ (includes 1D2-3H5)'
    }
    
    for site in [2,1]:
        if site == 1:
            move_factors = {'1D2-3H4':0.05, '1D2-3H5':1.08, '1D2-3H6':1.83, '1D2-3F2':3.04, '1D2-3F3+3F4':3.49, '1D2-1G4':3.34, '3P0-3H5':-0.59, '3P0-3H6':-0.37, 
                            '3P0-3F2':1.73, '3P0-3F3+3F4':-0.60} #in nm
            scale = -1
            style='dashed'
            flip=True
        else:
            move_factors = {'1D2-3H4':0.01, '1D2-3H5':1.13, '1D2-3H6':1.68, '1D2-3F2':3.16, '1D2-3F3+3F4':3.16, '1D2-1G4':2.73, '3P0-3H5':-0.92, '3P0-3H6':-2.53, '3P0-3F2':-2.22, 
                            '3P0-3F3+3F4':-1.93} #in nm
            scale = 1
            style ='solid'
            flip=False
            
        for bottom in bottom_states_from_1D2:
            top = '1D2'
            normalize = True
            key = f'{top}-{bottom}'
            name = f'Site {site}: {label_formats[key]}'

            move = move_factors[f'{top}-{bottom}']
            waves, areas = loader(site, top, bottom, normalize, laser_fit, move)
            line, = ax.plot(waves, scale*areas, label=name, linestyle=style)
            color = line.get_color()
            detect_peaks(ax, waves, areas, color, absorp=flip)
            
        for bottom in bottom_states_from_3P0:
            top = '3P0'
            normalize = True
            key = f'{top}-{bottom}'
            name = f'Site {site}: {label_formats[key]}'
            move = move_factors[f'{top}-{bottom}']
            waves, areas = loader(site, top, bottom, normalize, laser_fit, move)
            alpha = 0.3
            line, = ax.plot(waves, scale*areas, label=name, alpha=alpha, linestyle=style)
            color = line.get_color()
            detect_peaks(ax, waves, areas, color, alpha, absorp=flip)
            
    title = 'Super Plot of Pr:YSO both sites Emission (raw)'
    ax.set_title(title)
    return title

def excitation_plotter(site, ax, fit=True):
    levels = ['1D2', '3P0', '1I6']
    dyes_dict = {'1D2':['R590', 'R610'], '3P0':['C481'], '1I6':['C460']}
    if site == 2:
        fit_factors = {'R590':-1.9, 'R610':-2.57, 'C481':-0.94, 'C460':1.41}
        up_factor = 0.1
    else:
        fit_factors = {'R590':-2.14, 'R610':-2.16, 'C481':-0.9, 'C460':1.57}
        up_factor = 0
        
    for level in levels:
        dyes = dyes_dict[level]
        for dye in dyes:
            move = fit_factors[dye] if fit else 0
            w, a = exc_loader2(site, level, dye, True, move)
            ax.plot(w, a+up_factor, label=f'Site {site}: {level}, {dye} dye excitation')
            
    scale = -0.2
    loaders = ['1D2', '3P0']
    
    for i, load_id in enumerate(loaders):
        label = '10K FTIR Absorption' if i == 0 else None
        
        cut_w, abs_c = abs_loader(load_id)
        ax.plot(cut_w, scale * abs_c, 'k', label=label)
        detect_peaks(ax, cut_w, -scale * abs_c, absorp=True)
        
    w, a = UV_loader('3P', True)
    ax.plot(w, (scale-0.5) * a, 'k', alpha=0.3, label='10K Cary UV Absorption')
    detect_peaks(ax, w, -(scale-0.5) * a, color='k', alpha=0.3, absorp=True)    


def locate_peaks2(x, y, x_guesses, delta=2):
    # Sort x and y together based on x values
    sort_indices = np.argsort(x)
    x = x[sort_indices]
    y = y[sort_indices]
    
    x_peaks = []
    y_peaks = []
    
    for x_guess in x_guesses:
        i_min = np.searchsorted(x, x_guess - delta)
        i_max = np.searchsorted(x, x_guess + delta)
        
        i_min = max(0, i_min)
        i_max = min(len(x), i_max) 
        
        if i_min >= i_max:
            continue
            
        local_slice = y[i_min:i_max]
        if len(local_slice) == 0:
            continue
            
        i_peak = i_min + np.argmax(local_slice)
        x_peak = x[i_peak]
        y_peak = y[i_peak]
        
        x_peaks.append(x_peak)
        y_peaks.append(y_peak)
        
    return {
        'x': np.array(x_peaks),
        'y': np.array(y_peaks),
    }

def simple_table(file):
    # Load data, skipping the comment/header line
    data = np.loadtxt(file, comments="#")
    
    levels = data[:, 0].astype(int)
    energies = data[:, 1]
    
    # Create table
    fig, ax = plt.subplots(figsize=(5, 15))
    ax.axis("off")
    
    table_data = [[f"{int(level)}", f"{energy:.1f}"]
              for level, energy in zip(levels, energies)]
    
    table = ax.table(
        cellText=table_data,
        colLabels=["Level", "Energy"],
        loc="center",
        cellLoc="center"
    )
    
    # Formatting
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(0.8, 1)
    

    #plt.title("Pr YSO Energy Levels")
    plt.show()


def fluorescence_table(site, top, bottom, peaks, labels):
    fig, ax = plt.subplots(figsize=(3, 5))
    ax.axis("off")
    
    table_data = [[f"{lab}", f"{energy:.1f}"]
              for lab, energy in zip(labels, peaks)]
    
    table = ax.table(
        cellText=table_data,
        colLabels=["Process", "Raw Fluorescence"],
        loc="center",
        cellLoc="center"
    )
    
    # Formatting
    table.auto_set_font_size(True)
    #table.set_fontsize(8)
    #table.scale(0.8, 1)
    
    plt.title(f"Site {site}: {top}→{bottom} Raw Fluorescence")
    plt.show()


def energy_table(site, top, bottom, peaks, labels):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis("off")
    
    mask = np.char.startswith(labels, '1→')
    select_peaks = np.array(peaks)[mask]
    select_labels = np.array(labels)[mask]
    
    wavelengths = 1e7/select_peaks
    
    if site == 2:
        move_factor = SITE_2['em_move_factors'][f'{top}-{bottom}']
        laser_wave = (488.4 + move_factor) if top == '3P0' else 602.6 + move_factor
    else:
        move_factor = SITE_1['em_move_factors'][f'{top}-{bottom}']
        laser_wave = (483.8 + move_factor) if top == '3P0' else 604.5 + move_factor
        
        
    energies = 1e7/ laser_wave *np.ones_like(wavelengths) - 1.0e7 / (wavelengths) #these are the emission energies, not FTIR!
    #print(true_energies)
    
    all_true_energies = np.loadtxt(f'/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Energy Levels/Assignment/site_{site}.txt')[:,1]
    #print(all_true_energies)
    cost = np.abs(all_true_energies[:, None]-energies[None, :])
    i, j = linear_sum_assignment(cost)
    
    matched1 = all_true_energies[i]
    matched2 = energies[j]
    
    # Differences
    differences = np.abs(matched1 - matched2)
    
    if (differences > 20).any():
        print('Check energies assignment!')
        print(matched1)
        print(matched2)
        print(differences)
        print('')
        
    true_energies = matched1
    
    table_data = [[f"{level[2:]}", f"{energy:.1f}"]
              for level, energy in zip(select_labels, true_energies)]
    
    table = ax.table(
        cellText=table_data,
        colLabels=["Level", "FTIR-fit Energy"],
        loc="center",
        cellLoc="center"
    )
    
    # Formatting
    table.auto_set_font_size(True)
    #table.set_fontsize(8)
    #table.scale(0.8, 1)
    
    plt.title(f"Site {site}: {bottom} Energy Levels")
    plt.show()


def phonon_hunter(all_peaks, suspected_parents, site, top, bottom):
    phonons = np.array([55.7, 139, 176, 222, 240, 277, 324, 370, 408, 435, 500, 528, 565, 593, 908, 935, 982, 1010, 1038]*20) #20 sets to allow repeats
    
    all_data =[]
    
    for parent in suspected_parents:
        difs = np.ones_like(all_peaks)*parent - np.array(all_peaks)
        
        
        cost = np.abs(difs[:, None]-phonons[None, :])
        i, j = linear_sum_assignment(cost)
        
        matched1 = difs[i]
        matched2 = phonons[j]
        errors = abs(matched2 - matched1)
        closest_phonons = matched2
        
        color_guide = np.full((len(all_peaks), 3), 'w')
        color_guide[:,1][errors<6] = 'y'
        color_guide[:,1][errors<1.5] = 'g'
        color_guide[:,1][difs<=0.5] = 'r'
        #print(color_guide)
        
        all_data.append([difs, closest_phonons, errors, color_guide])
    
    # ---------------------------------------
    # Make ONE wide table
    # ---------------------------------------

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    table_data = []
    table_colours = []

    # Build each row by combining the results
    # from every parent
    for row in range(len(all_peaks)):
        row_data = [f"{all_peaks[row]:.0f}"] # <-- first column = peak 
        row_colours = ["blue"]

        for difs, closest_phonons, errors, color_guide in all_data:
            row_data.extend([
                f"{difs[row]:.0f}",
                f"{closest_phonons[row]:.0f}",
                f"{errors[row]:.0f}"
            ])

            row_colours.extend(color_guide[row])

        table_data.append(row_data)
        table_colours.append(row_colours)

    # Three columns for each parent
    col_labels = ["Peak"]
    for parent in suspected_parents:
        col_labels.extend([
            f"{parent}",
            "DOS",
            "Diff"
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        cellColours=table_colours
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    
    for row in range(len(all_peaks) + 1): # +1 for header 
        table[(row, 0)].get_text().set_color("white")

    plt.title(f"Site {site}: {top}→{bottom} Phonon Hunter")
    plt.show()


def publish_plot_em(site, top, bottom, marks=False):
    if site == 2:
        x_guesses = toml.load('/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Energy Levels/Guesses/all_raw_peaks_guess.toml')[bottom]['guesses']
        labels = toml.load('/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Energy Levels/Guesses/all_raw_peaks_guess.toml')[bottom]['labels']
    else:
        x_guesses = toml.load('/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Energy Levels/Guesses/site1_all_raw_peaks_guess.toml')[bottom]['guesses']
        labels = toml.load('/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Energy Levels/Guesses/site1_all_raw_peaks_guess.toml')[bottom]['labels']    
        
    wavenumbers, areas = loader(site, top, bottom, normalize=True)    
    peaks = locate_peaks2(wavenumbers, areas, x_guesses)    
    
    fluorescence_table(site, top, bottom, peaks['x'], labels)
    energy_table(site, top, bottom, peaks['x'], labels)
    
    suspected_parents = [14419, 14393, 14315, 14266, 14255, 14243, 14205] #change this for each spectra when hunting hotlines
    phonon_hunter(peaks['x'], suspected_parents, site, top, bottom)
    
    
    fig, ax = plt.subplots(figsize=(14,8))
    ax.plot(wavenumbers, areas, 'k')
    for x, y, label in zip(peaks['x'], peaks['y'], labels):
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(1,12),
            ha='center',
            fontsize=11,
            rotation='vertical'
        )
    if marks:
        for x, y in zip(peaks['x'], peaks['y']):
            ax.annotate(
                f'{x:.0f}',
                (x, y),
                textcoords="offset points",
                xytext=(0,60),
                ha='center',
                fontsize=8,
                color='r',
                rotation='vertical'
            )
    
    ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize = 14, labelpad = 3)
    ax.set_title(f'Site {site}: $^{top[0]}${top[1]}$_{top[2]}$'+r'$ \rightarrow $ '+f'$^{bottom[0]}${bottom[1]}$_{bottom[2]}$', y=1, x=0.1, fontweight='bold', fontsize=18)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False) 
    ax.spines['right'].set_visible(False) 

marks = True
#publish_plot_em(2, '1D2', '3H4', marks)
#publish_plot_em(1, '1D2', '3H4', marks)

publish_plot_em(2, '1D2', '3H5', marks)

#publish_plot_em(2, '1D2', '3F2', marks)
#publish_plot_em(1, '1D2', '3F2', marks)
plt.show()


#simple_table('/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Energy Levels/Assignment/site_2.txt')

'''    
fig, ax = plt.subplots(figsize=(16,8))
site = 2
fit = True
excitation_plotter(site, ax, fit)

w, a = exc_loader2(2, '3P2', 'C440', True, 2.58)
ax.plot(w, a+0.1, label='Site 2: 3P2, C440 dye excitation TEST')

site = 1
excitation_plotter(site, ax, fit)
plot_details(fig, ax, True)
ax.grid()
title = 'Fit Excitation Spectra. Spec fixed on (1D2-1G4)'
ax.set_title(title)
savename = title
#plt.savefig('/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Plots_Scope_Exc/' +savename +'.pdf', bbox_inches='tight', dpi=600, format='pdf')
plt.show()
    
#==================================

'''
'''

#super plot one site
fig, ax = plt.subplots(figsize=(16,8))
laser_fit = True
site = 1
title = super_plot(site, ax, laser_fit)
plot_details(fig, ax, laser_fit)
#plotly_fig = tls.mpl_to_plotly(fig)
#pyo.plot(plotly_fig, filename="interactive_plot_site1.html")
ax.grid()
savename = title
#plt.savefig('/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Plots_Scope_Emission/' +savename +'.pdf', bbox_inches='tight', dpi=600, format='pdf')
plt.show()


fig, ax = plt.subplots(figsize=(16,8))
laser_fit = True
site = 2
title = super_plot(site, ax, laser_fit)
plot_details(fig, ax, laser_fit)
#plotly_fig = tls.mpl_to_plotly(fig)
#pyo.plot(plotly_fig, filename="interactive_plot_site2.html")
ax.grid()
savename = title
#plt.savefig('/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Plots_Scope_Emission/' +savename +'.pdf', bbox_inches='tight', dpi=600, format='pdf')
plt.show()



#plot both sites
fig, ax = plt.subplots(figsize=(16,8))
title = all_plot(ax)
plot_details(fig, ax)
#plotly_fig = tls.mpl_to_plotly(fig)
#pyo.plot(plotly_fig, filename="interactive_plot_raw_emission.html")
ax.grid()
savename = title
#plt.savefig('/home/users/ccl128/Documents/Spectra/2026Y2SiO5_Pr/Plots_Scope_Emission/' +savename +'.pdf', bbox_inches='tight', dpi=600, format='pdf')
plt.show()

'''