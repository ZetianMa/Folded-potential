#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt
from scipy.special import spherical_jn, legendre
from scipy.integrate import simpson as simps
from scipy.optimize import minimize_scalar
from jitr import reactions, rmatrix
from jitr.reactions.potentials import (
    woods_saxon_potential,  woods_saxon_prime,  thomas_safe,
    coulomb_charged_sphere
)
from jitr.reactions.system   import spin_half_orbit_coupling
from jitr.utils              import mass, constants, kinematics, delta
from jitr.utils.kinematics   import ChannelKinematics
from jitr.xs.elastic         import DifferentialWorkspace

# === Density functions ===
def gaussian_density(r, rms, A):
    """高斯密度 distribution, normalized to A"""
    sigma_r = rms / np.sqrt(1.5)
    rho = np.exp(-r**2 / sigma_r**2)
    norm = 4.0 * np.pi * simps(r**2 * rho, x=r)
    return rho * (A / norm)

def fermi3_density(r, c, a, w, A):
    """3-parameter Fermi density"""
    rho = (1 + w * (r**2) / c**2) / (1 + np.exp((r - c) / a))
    norm = 4.0 * np.pi * simps(r**2 * rho, x=r)
    return rho * (A / norm)

# === Fourier transforms ===
def radial_to_momentum(rho_r):
    """ρ(r) -> rho_tilde(k)"""
    rho_tilde = np.zeros_like(k)
    for i, kv in enumerate(k):
        rho_tilde[i] = 4.0 * np.pi * simps(r**2 * rho_r * spherical_jn(0, kv * r), x=r)
    return rho_tilde

def momentum_to_radial(U_tilde):
    """U_tilde(k) -> U(r)"""
    U_r = np.zeros_like(r)
    for i, Rval in enumerate(r):
        integrand = k**2 * U_tilde * spherical_jn(0, k * Rval)
        U_r[i] = (1.0 / (2.0 * np.pi**2)) * simps(integrand, x=k)
    return U_r

# === M3Y nucleon-nucleon effective interaction in k-space ===
def m3y_vtilde(k, E_lab, A_proj):
    V1, mu1 = 7999.0, 4.0
    V2, mu2 = -2134.0, 2.5
    J00 = -276.0 * (1 - 0.005 * E_lab / A_proj)
    term1 = 4.0 * np.pi * V1 / (mu1 * (k**2 + mu1**2))
    term2 = 4.0 * np.pi * V2 / (mu2 * (k**2 + mu2**2))
    return term1 + term2 + J00

# === Folding potentials ===
def double_folding(rho1, rho2):
    """Double-folding nuclear potential U_nuc(r)"""
    t1 = radial_to_momentum(rho1)
    t2 = radial_to_momentum(rho2)
    U_t = t1 * t2 * m3y_vtilde(k, Elab, p[0])
    return momentum_to_radial(U_t)



def coulomb_folding(rho1, rho2):
    """Double-folding Coulomb potential U_coul(r)"""
    t1 = radial_to_momentum(rho1)
    t2 = radial_to_momentum(rho2)
    vC = 4.0 * np.pi * 1.44 / (k**2)
    U_t = t1 * t2 * vC
    return momentum_to_radial(U_t)

# radial and momentum grids
R_MAX, N_R = 20.0, 300
K_MAX, N_K = 5.0, 300
r = np.linspace(1e-6, R_MAX, N_R)
k = np.linspace(1e-6, K_MAX, N_K)


# ---------- 体系 ----------
Elab = 29.0 
Ca48, p = (40, 20), (4, 2)
Zz = Ca48[1]*p[1]
#imag_scale=0.6

    # build densities
rho_alpha     = gaussian_density(r, 1.46, 4)
rho_Ca40      = fermi3_density(r, 3.808, 0.512, -0.166, 40)
rho_charge_a  = gaussian_density(r, 1.46, 2)
rho_charge_ca = gaussian_density(r, 3.478, 20)

    # compute potentials
U_real = double_folding(rho_alpha, rho_Ca40)
U_coul = coulomb_folding(rho_charge_a, rho_charge_ca)

sys = reactions.ProjectileTargetSystem(
    channel_radius = 30,
    lmax           = 50,        # k·a ≈ 25 → 给 30
    mass_target    = mass.mass(*Ca48)[0],
    mass_projectile= constants.MASS_P,
    Ztarget=Ca48[1], Zproj=p[1],
    coupling = spin_half_orbit_coupling
)

kin = ChannelKinematics(*kinematics.classical_kinematics(
        sys.mass_target, sys.mass_projectile, Elab, Zz))
solver = rmatrix.Solver(nbasis=60)
chs, asym = sys.get_partial_wave_channels(kin.Ecm, kin.mu, kin.k, kin.eta)
θ_deg = np.linspace(1,179,360)

def compute_cross_section(scale, theta_exp):
    U_imag = scale * U_real
    U_tot  = U_real + 1j * U_imag + U_coul
    def V_central(rr):

        return np.interp(rr, r, U_tot)

    def V_ls(rr):

        return np.zeros_like(rr)

# ---------- 相移检查 ----------
    _, S0, _  = solver.solve(chs[0], asym[0], V_central, local_args=())
    ph0, att0 = delta(S0[0,0])
    print(f"δ₀ = {ph0:.2f}°,  attenuation = {att0:.2f}°")

# ---------- differential XS ----------
    dw = DifferentialWorkspace.build_from_system(
            p, Ca48, sys, kin, solver, np.deg2rad(theta_exp), smatrix_abs_tol=0.0)

    xs = dw.xs(V_central, V_ls, args_central=(), args_spin_orbit=())
#xs = dw.xs(V_central, lambda r, *args: np.zeros_like(r), args_central=(), args_spin_orbit=())
    ratio = xs.dsdo / xs.rutherford
    return ratio

data = np.loadtxt('data_29MeV.txt')
theta_exp, sigma_exp = data[:, 0], data[:, 1]
def chi2(scale):
    sigma_th = compute_cross_section(scale, theta_exp)
    return np.sum((( np.log(sigma_th)- np.log(sigma_exp)) )**2)

    # find best scale in [0,1]
res = minimize_scalar(chi2, bounds=(0, 1), method='bounded')
best_scale = res.x
print("Best imaginary scale =", best_scale)

    # compute theory with best scale
theta_plot = np.linspace(0, 180, 181)
sigma_fit = compute_cross_section(best_scale,θ_deg)
# ---------- plot σ/σR ----------
plt.figure(figsize=(6,4))
plt.semilogy(θ_deg, sigma_fit, 'r-', label='jitr')
plt.semilogy(theta_exp, sigma_exp, 'ro', ms=3, label='exp')
plt.xlabel(r'$\theta$ (deg)')
plt.ylabel(r'$\sigma/\sigma_R$')
plt.legend()
plt.grid(alpha=.3); plt.tight_layout(); plt.show()



# === Density functions ===
def gaussian_density(r, rms, A):
    """高斯密度 distribution, normalized to A"""
    sigma_r = rms / np.sqrt(1.5)
    rho = np.exp(-r**2 / sigma_r**2)
    norm = 4.0 * np.pi * simps(r**2 * rho, x=r)
    return rho * (A / norm)

def fermi3_density(r, c, a, w, A):
    """3-parameter Fermi density"""
    rho = (1 + w * (r**2) / c**2) / (1 + np.exp((r - c) / a))
    norm = 4.0 * np.pi * simps(r**2 * rho, x=r)
    return rho * (A / norm)

# === Fourier transforms ===
def radial_to_momentum(rho_r):
    """ρ(r) -> rho_tilde(k)"""
    rho_tilde = np.zeros_like(k)
    for i, kv in enumerate(k):
        rho_tilde[i] = 4.0 * np.pi * simps(r**2 * rho_r * spherical_jn(0, kv * r), x=r)
    return rho_tilde

def momentum_to_radial(U_tilde):
    """U_tilde(k) -> U(r)"""
    U_r = np.zeros_like(r)
    for i, Rval in enumerate(r):
        integrand = k**2 * U_tilde * spherical_jn(0, k * Rval)
        U_r[i] = (1.0 / (2.0 * np.pi**2)) * simps(integrand, x=k)
    return U_r

# === M3Y nucleon-nucleon effective interaction in k-space ===
def m3y_vtilde(k, E_lab, A_proj):
    V1, mu1 = 7999.0, 4.0
    V2, mu2 = -2134.0, 2.5
    J00 = -276.0 * (1 - 0.005 * E_lab / A_proj)
    term1 = 4.0 * np.pi * V1 / (mu1 * (k**2 + mu1**2))
    term2 = 4.0 * np.pi * V2 / (mu2 * (k**2 + mu2**2))
    return term1 + term2 + J00

# === Folding potentials ===
def double_folding(rho1, rho2):
    """Double-folding nuclear potential U_nuc(r)"""
    t1 = radial_to_momentum(rho1)
    t2 = radial_to_momentum(rho2)
    U_t = t1 * t2 * m3y_vtilde(k, ELAB, A_PROJ)
    return momentum_to_radial(U_t)



def coulomb_folding(rho1, rho2):
    """Double-folding Coulomb potential U_coul(r)"""
    t1 = radial_to_momentum(rho1)
    t2 = radial_to_momentum(rho2)
    vC = 4.0 * np.pi * 1.44 / (k**2)
    U_t = t1 * t2 * vC
    return momentum_to_radial(U_t)