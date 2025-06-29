#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.special   import spherical_jn
from scipy.integrate import simpson as simps
from scipy.optimize  import minimize_scalar

from jitr import reactions, rmatrix
from jitr.reactions.system import spin_half_orbit_coupling
from jitr.utils            import mass, constants, kinematics, delta
from jitr.utils.kinematics import ChannelKinematics
from jitr.xs.elastic       import DifferentialWorkspace

# === 1. 全局参数 & 网格 ===
ELAB      = 29.0           # MeV (实验能量)
PROJECTILE = (4, 2)        # α 粒子
TARGET     = (40, 20)      # 40Ca
ZZ         = PROJECTILE[1] * TARGET[1]

R_MAX, N_R = 20.0, 300
K_MAX, N_K =  5.0, 300
r  = np.linspace(1e-6, R_MAX, N_R)
k  = np.linspace(1e-6, K_MAX, N_K)
th = np.linspace(1, 179, 360)

# === 2. 核密度 & 傅里叶变换 ===
def gaussian_density(r, rms, A):
    sigma = rms/np.sqrt(1.5)
    rho   = np.exp(-r**2/sigma**2)
    norm  = 4*np.pi*simps(r**2*rho, x=r)
    return rho * (A/norm)

def fermi3_density(r, c, a, w, A):
    rho  = (1 + w*(r**2)/c**2)/(1 + np.exp((r-c)/a))
    norm = 4*np.pi*simps(r**2*rho, x=r)
    return rho*(A/norm)

def r2k(rho_r):
    return np.array([
        4*np.pi*simps(r**2 * rho_r * spherical_jn(0, kv*r), x=r)
        for kv in k
    ])

def k2r(U_k):
    return np.array([
        (1/(2*np.pi**2)) * simps(k**2 * U_k * spherical_jn(0, k*R), x=k)
        for R in r
    ])

# === 3. M3Y & 折叠势 ===
def m3y_vtilde(k, E_lab, A_proj):
    V1, mu1 = 7999.0, 4.0
    V2, mu2 = -2134.0, 2.5
    J00     = -276.0*(1 - 0.005*E_lab/A_proj)
    return (4*np.pi*V1)/(mu1*(k**2+mu1**2)) \
         + (4*np.pi*V2)/(mu2*(k**2+mu2**2)) \
         + J00

def double_folding(rho1, rho2):
    U_k = r2k(rho1)*r2k(rho2)*m3y_vtilde(k, ELAB, PROJECTILE[0])
    return k2r(U_k)

def coulomb_folding(rho1, rho2):
    U_k = r2k(rho1)*r2k(rho2)*(4*np.pi*1.44/(k**2))
    return k2r(U_k)

# === 4. 预计算：密度 & 实部势 & 库仑 ===
rho_p   = gaussian_density(r, 1.46, PROJECTILE[0])
rho_t   = fermi3_density(r, 3.808, 0.512, -0.166, TARGET[1]) + fermi3_density(r, 3.748, 0.512, -0.166, TARGET[0] - TARGET[1])
rho_cp  = gaussian_density(r, 1.46, PROJECTILE[1])
rho_t   = fermi3_density(r, 3.808, 0.512, -0.166, TARGET[1])

U_real = double_folding(rho_p, rho_t)
U_coul = coulomb_folding(rho_cp, rho_ct)

# === 5. 构造 R-矩阵系统 ===
sys = reactions.ProjectileTargetSystem(
    channel_radius = 30.0,
    lmax           = 60,
    mass_target    = mass.mass(*TARGET)[0],
    mass_projectile= constants.MASS_P,
    Ztarget=TARGET[1], Zproj=PROJECTILE[1],
    coupling = spin_half_orbit_coupling
)
Ecm, mu, k0, eta = kinematics.classical_kinematics(
    sys.mass_target, sys.mass_projectile, ELAB, ZZ
)
kin    = ChannelKinematics(Ecm, mu, k0, eta)
solver = rmatrix.Solver(nbasis=60)

# === 6. 核心：给定 imag_scale, 计算 σ/σ_R ===
def compute_ratio(imag_scale, theta_deg):
    U_im  = imag_scale * U_real
    U_tot = U_real + 1j*U_im + U_coul

    # 中央势 & 无自旋-轨道
    Vc = lambda rr, *a: np.interp(rr, r, U_tot)
    Vls= lambda rr, *a: np.zeros_like(rr)

    dw = DifferentialWorkspace.build_from_system(
        PROJECTILE, TARGET, sys, kin, solver,
        np.deg2rad(theta_deg), smatrix_abs_tol=0.0
    )
    xs = dw.xs(Vc, Vls, args_central=(), args_spin_orbit=())
    return xs.dsdo / xs.rutherford

# === 7. 拟合 & 作图 ===
if __name__ == "__main__":
    data       = np.loadtxt("data_29MeV.txt")
    theta_exp, ratio_exp = data[:,0], data[:,1]

    # χ² 拟合（取对数残差）
    res = minimize_scalar(
        lambda s: np.sum((np.log(compute_ratio(s, theta_exp)) - np.log(ratio_exp))**2),
        bounds=(0,1), method="bounded")
    #best_scale = 0.03
    print(f"Best imaginary scale = {best_scale:.3f}")

    # 画图
    ratio_fit = compute_ratio(best_scale, th)
    plt.semilogy(th,        ratio_fit, 'r-', label="jitr-folded")
    plt.semilogy(theta_exp, ratio_exp, 'ko', ms=3, label="exp")
    plt.xlabel(r"$\theta$ (deg)")
    plt.ylabel(r"$\sigma/\sigma_R$")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
