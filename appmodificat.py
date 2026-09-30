import streamlit as st
import numpy as np
from scipy.optimize import minimize, root_scalar
from scipy.stats import poisson, nbinom, erlang, wasserstein_distance
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pure Market Engine 9x9 (Optimized + Copula + Toothprints)", layout="wide")

class PureMarketEngine9x9:
    def __init__(self, max_goals=8):
        self.max_goals = max_goals
        self.weights = self._build_weight_matrix()

    def _build_weight_matrix(self):
        W = np.full((self.max_goals + 1, self.max_goals + 1), 0.02)
        anchor_scores = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), (2,1), (1,2), (3,0), (0,3)]
        for h, a in anchor_scores:
            if h <= self.max_goals and a <= self.max_goals:
                W[h, a] = 1.0
        secondary_scores = [(3,1), (1,3), (2,2), (3,2), (2,3), (4,0), (0,4), (4,1), (1,4), (4,2), (2,4)]
        for h, a in secondary_scores:
            if h <= self.max_goals and a <= self.max_goals:
                W[h, a] = 0.45
        return W

    def _power_unmargin_binary(self, odd_1, odd_2):
        if odd_1 <= 1.0 or odd_2 <= 1.0:
            return None, None
        p1_raw, p2_raw = 1.0 / odd_1, 1.0 / odd_2
        try:
            res = root_scalar(lambda k: (p1_raw ** k + p2_raw ** k) - 1.0, bracket=[0.001, 5.0], method='brentq')
            return (p1_raw ** res.root), (p2_raw ** res.root)
        except:
            clean = self._shin_unmargin({'1': p1_raw, '2': p2_raw})
            return clean['1'], clean['2']

    def _shin_unmargin(self, raw_probs):
        n = len(raw_probs)
        if n == 0: return {}
        sum_pi = sum(raw_probs.values())
        if sum_pi <= 1.0: return {k: v / sum_pi for k, v in raw_probs.items()}
        z = (sum_pi - 1.0) / max(1, n - 1)
        clean_probs = {}
        for k, pi in raw_probs.items():
            num = np.sqrt(z**2 + 4 * (1 - z) * (pi**2 / sum_pi)) - z
            den = 2 * (1 - z)
            clean_probs[k] = max(1e-12, num / max(den, 1e-12))
        total_clean = sum(clean_probs.values())
        return {k: v / total_clean for k, v in clean_probs.items()}

    def _shin_unmargin_asymmetric(self, raw_probs, z_mri_x=0.0):
        if z_mri_x <= 1.8:
            return self._shin_unmargin(raw_probs)
        
        n = len(raw_probs)
        if n == 0: return {}
        sum_pi = sum(raw_probs.values())
        if sum_pi <= 1.0: return {k: v / sum_pi for k, v in raw_probs.items()}
        
        z_base = (sum_pi - 1.0) / max(1, n - 1)
        clean_probs = {}
        asym_boost = 1.0 + 0.12 * min(z_mri_x - 1.8, 2.5)
        
        for k, pi in raw_probs.items():
            z_eff = z_base * (asym_boost if (isinstance(k, tuple) and k[0] == k[1]) else 1.0)
            num = np.sqrt(z_eff**2 + 4 * (1 - z_eff) * (pi**2 / sum_pi)) - z_eff
            den = 2 * (1 - z_eff)
            clean_probs[k] = max(1e-12, num / max(den, 1e-12))
            
        total_clean = sum(clean_probs.values())
        return {k: v / total_clean for k, v in clean_probs.items()}

    def calculate_x_stress_z_score(self, kl_div, x_gap, mri_index):
        mri_x_component = mri_index * (x_gap / max(x_gap + kl_div + 0.1, 1e-4))
        z_mri_x = (mri_x_component - 15.0) / 12.0
        return float(max(0.0, z_mri_x))

    def _apply_shannon_bayes_noise_filter(self, raw_probs):
        if not raw_probs: return {}
        sum_pi = sum(raw_probs.values())
        payout = 1.0 / sum_pi if sum_pi > 0 else 1.0
        if payout < 0.92:
            filtered_probs = {}
            margin_penalty = (0.92 - payout) * 2.0
            for k, p in raw_probs.items():
                sigmoid_weight = 1.0 / (1.0 + np.exp(-15.0 * (p - 0.03)))
                adj_p = p * (1.0 - margin_penalty * (1.0 - sigmoid_weight))
                filtered_probs[k] = max(1e-12, adj_p)
            return filtered_probs
        return raw_probs

    def _get_shin_alpha(self, raw_probs):
        n = len(raw_probs)
        if n <= 1: return 0.0
        sum_pi = sum(raw_probs.values())
        if sum_pi <= 1.0: return 0.0
        return float((sum_pi - 1.0) / max(1, n - 1))

    def _apply_copula_density(self, u, v, copula_type="Frank", theta=1.5):
        u = np.clip(u, 1e-6, 1.0 - 1e-6)
        v = np.clip(v, 1e-6, 1.0 - 1e-6)
        
        if copula_type == "Frank":
            if abs(theta) < 1e-4:
                return np.ones_like(u) if isinstance(u, np.ndarray) else 1.0
            num = -theta * (np.exp(-theta) - 1.0) * np.exp(-theta * (u + v))
            den = ((np.exp(-theta * u) - 1.0) * (np.exp(-theta * v) - 1.0) + (np.exp(-theta) - 1.0)) ** 2
            return np.maximum(1e-6, num / np.maximum(den, 1e-12))
            
        elif copula_type == "Gumbel":
            if theta <= 1.0:
                return np.ones_like(u) if isinstance(u, np.ndarray) else 1.0
            x = -np.log(u)
            y = -np.log(v)
            A = (x**theta + y**theta)**(1.0 / theta)
            C = np.exp(-A)
            density = (C / (u * v)) * ((x * y)**(theta - 1.0) / (x**theta + y**theta)**(2.0 - 1.0/theta)) * (A + theta - 1.0)
            return np.maximum(1e-6, density)

        elif copula_type == "Clayton":
            if theta <= 0.0:
                return np.ones_like(u) if isinstance(u, np.ndarray) else 1.0
            density = (1.0 + theta) * ((u * v) ** (-1.0 - theta)) * ((u ** (-theta) + v ** (-theta) - 1.0) ** (-2.0 - 1.0 / theta))
            return np.maximum(1e-6, density)
            
        return np.ones_like(u) if isinstance(u, np.ndarray) else 1.0

    def _get_marginal_pmf_cdf(self, mu, k_val, phi=0.0):
        if phi <= 1e-4:
            pmf = poisson.pmf(k_val, mu)
            cdf = poisson.cdf(k_val, mu)
        else:
            r = 1.0 / phi
            p_param = r / (r + mu)
            pmf = nbinom.pmf(k_val, r, p_param)
            cdf = nbinom.cdf(k_val, r, p_param)
        return pmf, cdf

    def _generate_matrix(self, lambda_h, mu_a, rho=0.0, pi_zero=0.0, copula_type="Fără", copula_theta=1.5, phi_dispersion=0.0):
        h_arr = np.arange(self.max_goals + 1)
        a_arr = np.arange(self.max_goals + 1)
        
        p_h, cdf_h = self._get_marginal_pmf_cdf(lambda_h, h_arr, phi_dispersion)
        p_a, cdf_a = self._get_marginal_pmf_cdf(mu_a, a_arr, phi_dispersion)
        
        P_H, P_A = np.meshgrid(p_h, p_a, indexing='ij')
        CDF_H, CDF_A = np.meshgrid(cdf_h, cdf_a, indexing='ij')
        
        xg_ratio = lambda_h / max(mu_a, 1e-5)
        if xg_ratio > 2.5:
            asym_factor = 1.0 + 0.15 * np.tanh(xg_ratio - 2.5)
        elif xg_ratio < 0.4:
            asym_factor = 1.0 + 0.15 * np.tanh((1.0 / xg_ratio) - 2.5)
        else:
            asym_factor = 1.0
        eff_rho = rho * asym_factor

        adj = np.ones((self.max_goals + 1, self.max_goals + 1))
        if self.max_goals >= 1:
            adj[0, 0] = 1.0 - (lambda_h * mu_a * eff_rho)
            adj[1, 0] = 1.0 + (mu_a * eff_rho)
            adj[0, 1] = 1.0 + (lambda_h * eff_rho)
            adj[1, 1] = 1.0 - eff_rho

        base_matrix = P_H * P_A * adj
        
        if copula_type != "Fără":
            u_mid = np.clip(CDF_H - 0.5 * P_H, 1e-6, 1.0 - 1e-6)
            v_mid = np.clip(CDF_A - 0.5 * P_A, 1e-6, 1.0 - 1e-6)
            c_density = self._apply_copula_density(u_mid, v_mid, copula_type, copula_theta)
            base_matrix *= c_density

        diag_mask = np.eye(self.max_goals + 1, dtype=bool)
        diag_boost = 1.0 + (pi_zero * np.exp(-0.5 * h_arr))
        
        matrix = np.where(diag_mask, base_matrix * diag_boost, (1.0 - pi_zero * 0.2) * base_matrix)
        matrix = np.maximum(1e-12, matrix)
                
        total_p = np.sum(matrix)
        return matrix / total_p if total_p > 0 else matrix

    def _huber_loss(self, y_true, y_pred, delta=0.001):
        error = y_pred - y_true
        return np.where(np.abs(error) <= delta, 0.5 * (error ** 2), delta * (np.abs(error) - 0.5 * delta))

    # =========================================================================
    # CELE 3 MODULE NOI DE DETECȚIE A AMPRENTELOR (TOOTHPRINTS)
    # =========================================================================

    def calculate_cross_market_stress_kl(self, mat_decoupled, mat_close, main_ou_close):
        """1. Indicele CMSI bazat pe Divergența Kullback-Leibler (1X2 vs O/U)"""
        p_decoupled = np.clip(mat_decoupled, 1e-12, 1.0)
        p_close = np.clip(mat_close, 1e-12, 1.0)
        
        # Proiecția 1X2 din decoupled
        p_home_dec = np.sum(np.tril(p_decoupled, -1))
        p_draw_dec = np.trace(p_decoupled)
        p_away_dec = np.sum(np.triu(p_decoupled, 1))
        p_1x2_dec = np.array([p_home_dec, p_draw_dec, p_away_dec])
        p_1x2_dec /= np.sum(p_1x2_dec)

        # Proiecția 1X2 din close
        p_home_cls = np.sum(np.tril(p_close, -1))
        p_draw_cls = np.trace(p_close)
        p_away_cls = np.sum(np.triu(p_close, 1))
        p_1x2_cls = np.array([p_home_cls, p_draw_cls, p_away_cls])
        p_1x2_cls /= np.sum(p_1x2_cls)

        kl_1x2 = np.sum(p_1x2_cls * np.log(p_1x2_cls / p_1x2_dec))

        # Proiecția O/U
        p_over_dec = self.calculate_ou_probability(p_decoupled, main_ou_close['line'], is_over=True)
        p_under_dec = 1.0 - p_over_dec
        p_ou_dec = np.array([p_over_dec, p_under_dec])

        p_over_cls = self.calculate_ou_probability(p_close, main_ou_close['line'], is_over=True)
        p_under_cls = 1.0 - p_over_cls
        p_ou_cls = np.array([p_over_cls, p_under_cls])

        kl_ou = np.sum(p_ou_cls * np.log(p_ou_cls / p_ou_dec))

        cmsi_kl = abs(kl_1x2 - kl_ou) * 100.0
        return float(np.clip(cmsi_kl, 0.0, 1.0))

    def calculate_asymmetric_pressure_field(self, mat_close):
        """2. Gradientul APF de ordinul 2 pe Diagonala Remizelor"""
        grad_y, grad_x = np.gradient(mat_close)
        grad_2_y, grad_2_x = np.gradient(grad_y)[0], np.gradient(grad_x)[1]
        
        diag_curvature = np.diag(grad_2_y + grad_2_x)
        
        # Masurare asimetrie stanga vs dreapta pe scorurile mici
        home_side_pressure = np.sum(grad_y[np.tril_indices(self.max_goals + 1, -1)])
        away_side_pressure = np.sum(grad_x[np.triu_indices(self.max_goals + 1, 1)])
        
        apf_score = (home_side_pressure - away_side_pressure) / (abs(home_side_pressure + away_side_pressure) + 1e-6)
        diag_asymmetry = float(np.std(diag_curvature))
        
        return float(apf_score), diag_asymmetry, diag_curvature

    def calculate_cell_level_vig_squeeze(self, mat_decoupled, mat_close_raw):
        """3. Indicatorul Vig Squeeze / Taxă pe Celulă (P_Close / P_Decoupled)"""
        p_dec = np.maximum(1e-6, mat_decoupled)
        p_cls = np.maximum(1e-6, mat_close_raw)
        
        vig_squeeze_matrix = p_cls / p_dec
        max_squeeze = float(np.max(vig_squeeze_matrix))
        squeezed_cells_idx = np.argwhere(vig_squeeze_matrix > 1.25)
        squeezed_cells = [(int(c[0]), int(c[1]), float(vig_squeeze_matrix[c[0], c[1]])) for c in squeezed_cells_idx]
        
        return vig_squeeze_matrix, max_squeeze, squeezed_cells

    # =========================================================================
    # MODULELE COMPLEMENTARE
    # =========================================================================

    def calculate_vector_field_flow(self, mat_open, mat_close):
        delta_p = mat_close - mat_open
        v_y, u_x = np.gradient(delta_p)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(delta_p, cmap="coolwarm", origin="upper", vmin=-0.03, vmax=0.03)
        
        x, y = np.meshgrid(np.arange(self.max_goals + 1), np.arange(self.max_goals + 1))
        ax.quiver(x, y, u_x, v_y, color="black", angles="xy", scale_units="xy", scale=0.5, pivot="middle")
        
        ax.set_title("Vector Field Flow (ΔP & Gradient Stream)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Goluri Oaspeți")
        ax.set_ylabel("Goluri Gazde")
        ax.set_xticks(range(self.max_goals + 1))
        ax.set_yticks(range(self.max_goals + 1))
        fig.colorbar(cax, ax=ax, label="Shift Probabilitate (ΔP)")
        plt.tight_layout()
        plt.close(fig)
        return u_x, v_y, fig

    def calculate_surface_laplacian(self, mat_p):
        kernel = np.array([[0,  1, 0],
                           [1, -4, 1],
                           [0,  1, 0]])
        
        laplacian_map = ndimage.convolve(mat_p, kernel, mode='constant', cval=0.0)
        max_curvature = float(np.max(np.abs(laplacian_map)))
        surface_stiffness = float(np.std(laplacian_map))
        return laplacian_map, max_curvature, surface_stiffness

    def calculate_topological_bimodal_index(self, mat_p, threshold_relative=0.25):
        local_max = ndimage.maximum_filter(mat_p, size=3) == mat_p
        max_p = np.max(mat_p)
        significant_peaks = local_max & (mat_p >= (max_p * threshold_relative))
        
        coords = np.argwhere(significant_peaks)
        num_peaks = len(coords)
        is_bimodal = num_peaks >= 2
        peaks_info = [(int(c[0]), int(c[1]), float(mat_p[c[0], c[1]])) for c in coords]
        return is_bimodal, num_peaks, peaks_info

    def calculate_bivariate_moments(self, mat_p):
        grid = np.arange(self.max_goals + 1)
        x_grid, y_grid = np.meshgrid(grid, grid)
        
        lambda_h = np.sum(mat_p * y_grid)
        mu_a = np.sum(mat_p * x_grid)
        
        sigma_h = np.sqrt(np.sum(mat_p * ((y_grid - lambda_h) ** 2)))
        sigma_a = np.sqrt(np.sum(mat_p * ((x_grid - mu_a) ** 2)))
        
        sigma_h = max(sigma_h, 1e-6)
        sigma_a = max(sigma_a, 1e-6)
        
        coskew_ha = np.sum(mat_p * (y_grid - lambda_h) * ((x_grid - mu_a) ** 2)) / (sigma_h * (sigma_a ** 2))
        
        if coskew_ha > 0.10:
            regime = "Meci Răzbunător (Tit-for-Tat)"
            regime_desc = "Un gol primit determină o replică ofensivă imediată."
        elif coskew_ha < -0.10:
            regime = "Meci de Blocaj Defensiv"
            regime_desc = "Un gol marcat închide complet jocul."
        else:
            regime = "Meci Simetric / Echilibrat"
            regime_desc = "Evoluția scorului urmează dinamica Poisson standard."
            
        return float(coskew_ha), regime, regime_desc

    def calculate_first_passage_time(self, lambda_h, mu_a):
        total_rate = lambda_h + mu_a
        if total_rate <= 0:
            return 90.0, 0.0, 0.0
        
        expected_min = (1.0 / total_rate) * 90.0
        p_0_15 = (1.0 - np.exp(-total_rate * (15.0 / 90.0))) * 100
        p_0_30 = (1.0 - np.exp(-total_rate * (30.0 / 90.0))) * 100
        return float(expected_min), float(p_0_15), float(p_0_30)

    def calculate_svi_9x9(self, mat_open, mat_close, grad_volatilitate_t=0.15):
        matrice_delta = mat_close - mat_open
        svi_score = np.std(matrice_delta) * (1.0 + grad_volatilitate_t)
        return float(svi_score), matrice_delta

    def calculate_residual_heatmap_9x9(self, mat_open, mat_close):
        matrice_reziduuri = mat_close - mat_open
        max_devier_pozitiva = float(np.max(matrice_reziduuri))
        scor_anomalie_idx = np.unravel_index(np.argmax(matrice_reziduuri), (9, 9))
        scor_anomalie = (int(scor_anomalie_idx[0]), int(scor_anomalie_idx[1]))
        
        este_contaminat = False
        mesaj_status = "Piață 9x9 Structurată / Fără Zgomot Anomalic"
        
        if max_devier_pozitiva > 0.04:
            este_contaminat = True
            mesaj_status = (f"⚠️ Anomalie Detectată la Scor Corect {scor_anomalie[0]}-{scor_anomalie[1]} "
                            f"| Abatere: +{max_devier_pozitiva*100:.2f}%")
            
        return {
            "matrice_reziduuri": matrice_reziduuri,
            "max_devier": max_devier_pozitiva,
            "scor_anomalie": scor_anomalie,
            "este_contaminat": este_contaminat,
            "mesaj": mesaj_status
        }

    def calculate_micro_price_skew(self, mat_close, main_ou):
        p_over_calc = self.calculate_ou_probability(mat_close, main_ou['line'], is_over=True)
        micro_odd = 1.0 / max(p_over_calc, 1e-5)
        return float(micro_odd)

    def calculate_obi_s(self, mass_analysis, main_ou):
        delta_p_over = abs(mass_analysis.get('under_shift_pct', 0.0)) / 100.0
        delta_odd = abs(main_ou['over_odd'] - 1.90) / 1.90
        return float(delta_p_over / max(delta_odd, 1e-4))

    def calculate_phantom_shift(self, mat_open, mat_close, main_ou):
        p_over_open = self.calculate_ou_probability(mat_open, main_ou['line'], is_over=True)
        p_over_close = self.calculate_ou_probability(mat_close, main_ou['line'], is_over=True)
        delta_p_mat = p_over_close - p_over_open
        
        p_over_odd_clean, _ = self._power_unmargin_binary(main_ou['over_odd'], main_ou['under_odd'])
        delta_p_line = (p_over_odd_clean if p_over_odd_clean else 0.5) - 0.5
        return float(abs(delta_p_mat - delta_p_line))

    def calculate_marginal_overround_asymmetry(self, main_ou):
        if not main_ou or main_ou.get('over_odd', 0) <= 1.0 or main_ou.get('under_odd', 0) <= 1.0:
            return 0.0
        p_over_raw = 1.0 / main_ou['over_odd']
        p_under_raw = 1.0 / main_ou['under_odd']
        total_overround = (p_over_raw + p_under_raw) - 1.0
        if total_overround <= 0: return 0.0
        p_over_clean, p_under_clean = self._power_unmargin_binary(main_ou['over_odd'], main_ou['under_odd'])
        margin_over = p_over_raw - (p_over_clean if p_over_clean else p_over_raw)
        return float(margin_over / max(total_overround, 1e-6))

    def calculate_wasserstein_emd_1d(self, mat_open, mat_close):
        goals_open = np.sum(mat_open, axis=1)
        goals_close = np.sum(mat_close, axis=1)
        grid = np.arange(self.max_goals + 1)
        emd_goals = wasserstein_distance(grid, grid, goals_open, goals_close)
        
        diff_grid = np.arange(-self.max_goals, self.max_goals + 1)
        diff_open = np.zeros(len(diff_grid))
        diff_close = np.zeros(len(diff_grid))
        
        for h in range(self.max_goals + 1):
            for a in range(self.max_goals + 1):
                idx = (h - a) + self.max_goals
                diff_open[idx] += mat_open[h, a]
                diff_close[idx] += mat_close[h, a]
                
        emd_diff = wasserstein_distance(diff_grid, diff_grid, diff_open, diff_close)
        return float(emd_goals), float(emd_diff)

    def calculate_price_elasticity_compression(self, under_shift_pct, main_ou):
        delta_p = abs(under_shift_pct) / 100.0
        if delta_p < 1e-4: return 0.0
        delta_odd = abs(main_ou['under_odd'] - 1.90) / 1.90 
        return float(delta_odd / delta_p)

    def calculate_conditional_entropy(self, matrix, main_ah_line):
        h_grid, a_grid = np.indices(matrix.shape)
        ah_mask = (h_grid - a_grid) >= main_ah_line
        p_x1 = np.sum(matrix[ah_mask])
        p_x0 = 1.0 - p_x1
        
        if p_x1 <= 1e-6 or p_x0 <= 1e-6: return 0.0
        
        total_goals = h_grid + a_grid
        ou_median = np.median(total_goals)
        ou_mask = total_goals > ou_median
        
        p_y1_x1 = np.sum(matrix[ah_mask & ou_mask]) / p_x1
        p_y0_x1 = 1.0 - p_y1_x1
        
        p_y1_x0 = np.sum(matrix[(~ah_mask) & ou_mask]) / p_x0
        p_y0_x0 = 1.0 - p_y1_x0
        
        h_y_x1 = - (p_y1_x1 * np.log2(max(p_y1_x1, 1e-12)) + p_y0_x1 * np.log2(max(p_y0_x1, 1e-12)))
        h_y_x0 = - (p_y1_x0 * np.log2(max(p_y1_x0, 1e-12)) + p_y0_x0 * np.log2(max(p_y0_x0, 1e-12)))
        return float(p_x1 * h_y_x1 + p_x0 * h_y_x0)

    def calculate_gini_index(self, matrix):
        flat = np.sort(matrix.flatten())
        n = len(flat)
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * flat)) / (n * np.sum(flat)) - (n + 1) / n)

    def calculate_top3_density(self, matrix):
        flat_sorted = np.sort(matrix.flatten())[::-1]
        return float(np.sum(flat_sorted[:3]) * 100.0)

    def calculate_modal_skewness(self, lambda_h, mu_a):
        skew_h = 1.0 / np.sqrt(max(lambda_h, 1e-5))
        skew_a = 1.0 / np.sqrt(max(mu_a, 1e-5))
        return float(skew_h - skew_a)

    def calculate_mvi(self, lambda_h, mu_a):
        margin = abs(lambda_h - mu_a)
        variance = lambda_h + mu_a
        return float(margin / max(variance, 1e-5))

    def calculate_kl_divergence(self, mat_open, mat_close):
        p = np.clip(mat_open.flatten(), 1e-12, 1.0)
        q = np.clip(mat_close.flatten(), 1e-12, 1.0)
        return float(np.sum(p * np.log(p / q)))

    def calculate_jsd(self, mat_open, mat_close):
        p = np.clip(mat_open.flatten(), 1e-12, 1.0)
        q = np.clip(mat_close.flatten(), 1e-12, 1.0)
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log2(p / m))
        kl_qm = np.sum(q * np.log2(q / m))
        return float(0.5 * (kl_pm + kl_qm))

    def calculate_market_refractive_index(self, jsd_div, ah_line_open, ah_line_close):
        delta_line = abs(ah_line_close - ah_line_open)
        return float(jsd_div / (delta_line + 1e-4))

    def calculate_erlang_adjusted_srp(self, jsd_div, ah_line_open, ah_line_close, total_xg):
        delta_line = abs(ah_line_close - ah_line_open)
        k_shape = 2
        beta_scale = max(0.1, total_xg / 2.0)
        erlang_pdf = erlang.pdf(delta_line + 0.1, k_shape, scale=beta_scale)
        srp_raw = jsd_div / (delta_line + 1e-4)
        return float(srp_raw * (1.0 + erlang_pdf))

    def calculate_bivariate_skewness_tensor(self, matrix):
        h_grid, a_grid = np.indices(matrix.shape)
        mean_h = np.sum(h_grid * matrix)
        mean_a = np.sum(a_grid * matrix)
        var_h = np.sum(((h_grid - mean_h) ** 2) * matrix)
        var_a = np.sum(((a_grid - mean_a) ** 2) * matrix)
        skew_h = np.sum(((h_grid - mean_h) ** 3) * matrix) / (var_h ** 1.5 + 1e-6)
        skew_a = np.sum(((a_grid - mean_a) ** 3) * matrix) / (var_a ** 1.5 + 1e-6)
        return float(skew_h - skew_a)

    def calculate_analytical_tail_dependence(self, matrix, min_goals=3):
        h_grid, a_grid = np.indices(matrix.shape)
        mask = (h_grid + a_grid) >= min_goals
        if not np.any(mask): return 0.0
        weights = matrix[mask]
        total_w = np.sum(weights)
        if total_w <= 0: return 0.0
        w_norm = weights / total_w
        h_vals, a_vals = h_grid[mask], a_grid[mask]
        mean_h = np.sum(h_vals * w_norm)
        mean_a = np.sum(a_vals * w_norm)
        cov = np.sum((h_vals - mean_h) * (a_vals - mean_a) * w_norm)
        var_h = np.sum(((h_vals - mean_h)**2) * w_norm)
        var_a = np.sum(((a_vals - mean_a)**2) * w_norm)
        denom = np.sqrt(var_h * var_a)
        return float(cov / denom) if denom > 1e-6 else 0.0

    def calculate_shin_alpha_variance(self, cs_odds_dict, main_ah_input, main_ou_input, main_x_odd):
        alphas = []
        valid_cs = {k: 1.0 / v for k, v in cs_odds_dict.items() if 1.0 < v <= 100.0}
        if valid_cs: alphas.append(self._get_shin_alpha(valid_cs))
        if main_ah_input and main_ah_input.get('home_odd', 0) > 1.0 and main_ah_input.get('away_odd', 0) > 1.0:
            alphas.append(self._get_shin_alpha({'h': 1.0/main_ah_input['home_odd'], 'a': 1.0/main_ah_input['away_odd']}))
        if main_ou_input and main_ou_input.get('over_odd', 0) > 1.0 and main_ou_input.get('under_odd', 0) > 1.0:
            alphas.append(self._get_shin_alpha({'o': 1.0/main_ou_input['over_odd'], 'u': 1.0/main_ou_input['under_odd']}))
        if len(alphas) < 2: return 0.0
        return float(np.var(alphas))

    def calculate_cross_market_stress_index(self, kl_div, shin_var, x_gap):
        stress = (0.40 * min(kl_div * 10.0, 1.0)) + (0.35 * min(shin_var * 1000.0, 1.0)) + (0.25 * min(x_gap / 3.0, 1.0))
        return float(np.clip(stress, 0.0, 1.0))

    def calculate_ah_probability(self, matrix, ah_line, is_home=True):
        rem = abs(ah_line) % 0.5
        if abs(rem - 0.25) < 1e-4:
            line_1 = ah_line - 0.25
            line_2 = ah_line + 0.25
            
            p_win1, p_push1, _ = self._single_line_ah_outcomes(matrix, line_1, is_home)
            p_win2, p_push2, _ = self._single_line_ah_outcomes(matrix, line_2, is_home)
            
            p_eff = 0.5 * (p_win1 + 0.5 * p_push1 + p_win2 + 0.5 * p_push2)
            return float(np.clip(p_eff, 1e-5, 1.0))
        else:
            p_win, p_push, _ = self._single_line_ah_outcomes(matrix, ah_line, is_home)
            denom = 1.0 - p_push
            return float(p_win / denom) if denom > 0 else 0.5

    def _single_line_ah_outcomes(self, matrix, ah_line, is_home=True):
        h_grid, a_grid = np.indices(matrix.shape)
        diff = (h_grid - a_grid) if is_home else (a_grid - h_grid)
        score_eff = diff + ah_line
        
        prob_win = np.sum(matrix[score_eff > 1e-5])
        prob_push = np.sum(matrix[np.abs(score_eff) <= 1e-5])
        prob_loss = np.sum(matrix[score_eff < -1e-5])
        return prob_win, prob_push, prob_loss

    def calculate_ou_probability(self, matrix, ou_line, is_over=True):
        rem = abs(ou_line) % 0.5
        if abs(rem - 0.25) < 1e-4:
            line_1 = ou_line - 0.25
            line_2 = ou_line + 0.25
            
            p_win1, p_push1, _ = self._single_line_ou_outcomes(matrix, line_1, is_over)
            p_win2, p_push2, _ = self._single_line_ou_outcomes(matrix, line_2, is_over)
            
            p_eff = 0.5 * (p_win1 + 0.5 * p_push1 + p_win2 + 0.5 * p_push2)
            return float(np.clip(p_eff, 1e-5, 1.0))
        else:
            p_win, p_push, _ = self._single_line_ou_outcomes(matrix, ou_line, is_over)
            denom = 1.0 - p_push
            return float(p_win / denom) if denom > 0 else 0.5

    def _single_line_ou_outcomes(self, matrix, ou_line, is_over=True):
        h_grid, a_grid = np.indices(matrix.shape)
        total_goals = h_grid + a_grid
        diff = (total_goals - ou_line) if is_over else (ou_line - total_goals)
        
        prob_win = np.sum(matrix[diff > 1e-5])
        prob_push = np.sum(matrix[np.abs(diff) <= 1e-5])
        prob_loss = np.sum(matrix[diff < -1e-5])
        return prob_win, prob_push, prob_loss

    def calculate_draw_probability(self, matrix):
        return float(np.trace(matrix))

    def analyze_mass_shifts(self, cs_open_dict, cs_close_dict, ou_line=2.5, z_mri_x=0.0):
        open_raw = {k: 1.0/v for k, v in cs_open_dict.items() if 1.0 < v <= 100.0}
        close_raw = {k: 1.0/v for k, v in cs_close_dict.items() if 1.0 < v <= 100.0}
        
        open_clean = self._shin_unmargin_asymmetric(self._apply_shannon_bayes_noise_filter(open_raw), z_mri_x)
        close_clean = self._shin_unmargin_asymmetric(self._apply_shannon_bayes_noise_filter(close_raw), z_mri_x)
        shift_report, total_home_shift, total_away_shift, total_under_shift = {}, 0.0, 0.0, 0.0

        for score, p_open in open_clean.items():
            if score in close_clean:
                p_close = close_clean[score]
                delta_p = p_close - p_open
                shift_report[score] = delta_p
                h, a = score
                if h > a: total_home_shift += delta_p
                elif a > h: total_away_shift += delta_p
                if (h + a) < ou_line: total_under_shift += delta_p

        top_inflows = sorted(shift_report.items(), key=lambda x: x[1], reverse=True)[:3]
        top_outflows = sorted(shift_report.items(), key=lambda x: x[1])[:3]

        return {
            'home_shift_pct': round(total_home_shift * 100, 2),
            'away_shift_pct': round(total_away_shift * 100, 2),
            'under_shift_pct': round(total_under_shift * 100, 2),
            'top_inflows': [(f"{s[0]}-{s[1]}", round(d * 100, 2)) for s, d in top_inflows],
            'top_outflows': [(f"{s[0]}-{s[1]}", round(d * 100, 2)) for s, d in top_outflows]
        }

    def extract_pure_xg(self, cs_odds_dict, main_ah_input=None, main_ou_input=None, main_x_odd=None, mode="Decoupled", copula_type="Fără", copula_theta=1.5, auto_fit_copula=False, use_nbinom=False, z_mri_x=0.0):
        valid_odds = {k: v for k, v in cs_odds_dict.items() if 1.0 < v <= 100.0}
        raw_probs = {k: 1.0 / v for k, v in valid_odds.items()}
        filtered_probs = self._apply_shannon_bayes_noise_filter(raw_probs)
        clean_probs = self._shin_unmargin_asymmetric(filtered_probs, z_mri_x)

        target_p_ah, target_p_ou, target_p_x = None, None, None
        if main_ah_input and main_ah_input.get('home_odd', 0) > 1.0 and main_ah_input.get('away_odd', 0) > 1.0:
            p_h_clean, _ = self._power_unmargin_binary(main_ah_input['home_odd'], main_ah_input['away_odd'])
            target_p_ah = p_h_clean
        if main_ou_input and main_ou_input.get('over_odd', 0) > 1.0 and main_ou_input.get('under_odd', 0) > 1.0:
            p_over_clean, _ = self._power_unmargin_binary(main_ou_input['over_odd'], main_ou_input['under_odd'])
            target_p_ou = p_over_clean
        if main_x_odd and main_x_odd > 1.0:
            target_p_x = (1.0 / main_x_odd) / 1.05

        def loss_function(params):
            lambda_h, mu_a, rho, pi_zero = params[:4]
            current_theta = params[4] if auto_fit_copula and len(params) > 4 else copula_theta
            current_phi = params[5] if use_nbinom and len(params) > 5 else 0.0

            theo_matrix = self._generate_matrix(lambda_h, mu_a, rho, pi_zero, copula_type, current_theta, current_phi)
            sum_theo_subset = sum(theo_matrix[h, a] for (h, a) in clean_probs.keys() if h <= self.max_goals and a <= self.max_goals)
            if sum_theo_subset <= 0: return 1e6
                
            total_loss = 0.0
            for (h, a), p_clean in clean_probs.items():
                if h <= self.max_goals and a <= self.max_goals:
                    p_theo_scaled = theo_matrix[h, a] / sum_theo_subset
                    weight = self.weights[h, a]
                    huber = self._huber_loss(p_clean, p_theo_scaled, delta=0.001)
                    total_loss += weight * huber

            if mode == "Anchored (5.0 Weight)":
                if target_p_ou and main_ou_input:
                    p_ou_calc = self.calculate_ou_probability(theo_matrix, main_ou_input['line'], is_over=True)
                    total_loss += 5.0 * ((p_ou_calc - target_p_ou) ** 2)

                if target_p_ah and main_ah_input:
                    p_ah_calc = self.calculate_ah_probability(theo_matrix, main_ah_input['home_line'], is_home=True)
                    total_loss += 5.0 * ((p_ah_calc - target_p_ah) ** 2)

            if target_p_x:
                p_x_calc = self.calculate_draw_probability(theo_matrix)
                total_loss += 2.5 * ((p_x_calc - target_p_x) ** 2)
                    
            return total_loss

        init_guess = [1.40, 1.10, 0.0, 0.01]
        bounds = [(0.1, 4.5), (0.1, 4.5), (-0.25, 0.25), (0.0, 0.25)]

        if auto_fit_copula and copula_type != "Fără":
            t_min = 1.01 if copula_type == "Gumbel" else (0.01 if copula_type == "Clayton" else 0.1)
            t_max = 5.0 if copula_type != "Frank" else 10.0
            init_guess.append(copula_theta)
            bounds.append((t_min, t_max))

        if use_nbinom:
            init_guess.append(0.01)
            bounds.append((0.0, 0.5))
        
        res = minimize(loss_function, init_guess, bounds=bounds, method='L-BFGS-B')
        if not res.success or res.x[0] >= 4.3 or res.x[1] >= 4.3:
            res = minimize(loss_function, init_guess, bounds=bounds, method='Nelder-Mead')

        lambda_pure, mu_pure, rho_pure, pi_zero_pure = res.x[:4]
        
        idx = 4
        if auto_fit_copula and copula_type != "Fără":
            fitted_theta = res.x[idx]
            idx += 1
        else: fitted_theta = copula_theta

        if use_nbinom: fitted_phi = res.x[idx]
        else: fitted_phi = 0.0

        matrix = self._generate_matrix(lambda_pure, mu_pure, rho_pure, pi_zero_pure, copula_type, fitted_theta, fitted_phi)
        
        top_4_prob = np.sum(np.sort(matrix.flatten())[-4:])
        cs_ratio = top_4_prob / (1.0 - top_4_prob + 1e-6)

        flat_m = matrix.flatten()
        flat_m = flat_m[flat_m > 0]
        shannon_entropy = -np.sum(flat_m * np.log2(flat_m))

        gini_index = self.calculate_gini_index(matrix)
        top3_density = self.calculate_top3_density(matrix)
        modal_skew = self.calculate_modal_skewness(lambda_pure, mu_pure)
        mvi_index = self.calculate_mvi(lambda_pure, mu_pure)

        return lambda_pure, mu_pure, rho_pure, pi_zero_pure, cs_ratio, shannon_entropy, gini_index, top3_density, modal_skew, mvi_index, matrix, fitted_theta, fitted_phi

    def decode_comparative(self, cs_open, cs_close, main_ah_open, main_ah_close, main_ou_open, main_ou_close, main_x_open=None, main_x_close=None, mode="Decoupled", copula_type="Fără", copula_theta=1.5, auto_fit_copula=False, use_nbinom=False):
        l_open_pre, m_open_pre, _, _, _, _, _, _, _, _, mat_open_pre, _, _ = self.extract_pure_xg(cs_open, main_ah_open, main_ou_open, main_x_open, mode, copula_type, copula_theta, auto_fit_copula, use_nbinom, z_mri_x=0.0)
        l_close_pre, m_close_pre, _, _, _, _, _, _, _, _, mat_close_pre, _, _ = self.extract_pure_xg(cs_close, main_ah_close, main_ou_close, main_x_close, mode, copula_type, copula_theta, auto_fit_copula, use_nbinom, z_mri_x=0.0)

        kl_pre = self.calculate_kl_divergence(mat_open_pre, mat_close_pre)
        jsd_pre = self.calculate_jsd(mat_open_pre, mat_close_pre)
        mri_pre = self.calculate_market_refractive_index(jsd_pre, main_ah_open['home_line'], main_ah_close['home_line'])
        p_draw_pre = self.calculate_draw_probability(mat_close_pre)
        fair_x_pre = 1.0 / max(p_draw_pre, 1e-5)
        x_gap_pre = abs(fair_x_pre - (main_x_close if main_x_close and main_x_close > 1.0 else fair_x_pre))
        
        z_mri_x = self.calculate_x_stress_z_score(kl_pre, x_gap_pre, mri_pre)

        l_open, m_open, r_open, pi_open, cs_ratio_open, ent_open, gini_open, top3_open, skew_open, mvi_open, mat_open, theta_open, phi_open = self.extract_pure_xg(cs_open, main_ah_open, main_ou_open, main_x_open, mode, copula_type, copula_theta, auto_fit_copula, use_nbinom, z_mri_x)
        l_close, m_close, r_close, pi_close, cs_ratio_close, ent_close, gini_close, top3_close, skew_close, mvi_close, mat_close, theta_close, phi_close = self.extract_pure_xg(cs_close, main_ah_close, main_ou_close, main_x_close, mode, copula_type, copula_theta, auto_fit_copula, use_nbinom, z_mri_x)

        delta_xg = (l_close + m_close) - (l_open + m_open)
        delta_ent = ent_close - ent_open
        delta_cs = cs_ratio_close - cs_ratio_open
        delta_gini = gini_close - gini_open
        delta_top3 = top3_close - top3_open

        kl_div = self.calculate_kl_divergence(mat_open, mat_close)
        jsd_div = self.calculate_jsd(mat_open, mat_close)
        mri_index = self.calculate_market_refractive_index(jsd_div, main_ah_open['home_line'], main_ah_close['home_line'])
        ear_srp = self.calculate_erlang_adjusted_srp(jsd_div, main_ah_open['home_line'], main_ah_close['home_line'], l_close + m_close)
        biv_skew = self.calculate_bivariate_skewness_tensor(mat_close)
        tail_corr = self.calculate_analytical_tail_dependence(mat_close)
        shin_var = self.calculate_shin_alpha_variance(cs_close, main_ah_close, main_ou_close, main_x_close)

        svi_score_9x9, matrice_delta_9x9 = self.calculate_svi_9x9(mat_open, mat_close)
        res_heatmap_9x9 = self.calculate_residual_heatmap_9x9(mat_open, mat_close)

        u_x, v_y, fig_quiver = self.calculate_vector_field_flow(mat_open, mat_close)
        lap_map, max_curv, stiffness = self.calculate_surface_laplacian(mat_close)
        is_bimodal, num_peaks, peaks = self.calculate_topological_bimodal_index(mat_close)
        coskew_ha, regime_name, regime_desc = self.calculate_bivariate_moments(mat_close)
        exp_min, p15, p30 = self.calculate_first_passage_time(l_close, m_close)

        # RULARE MODUL 6: TOOTHPRINTS
        cmsi_kl_val = self.calculate_cross_market_stress_kl(mat_close_pre, mat_close, main_ou_close)
        apf_score, diag_asym, diag_curv = self.calculate_asymmetric_pressure_field(mat_close)
        
        # Matrice brută din cotele de închidere pentru raportul Vig Squeeze
        raw_close_matrix = np.zeros((9, 9))
        for (h, a), odd in cs_close.items():
            if h <= 8 and a <= 8 and odd > 1.0: raw_close_matrix[h, a] = 1.0 / odd
        sum_raw = np.sum(raw_close_matrix)
        if sum_raw > 0: raw_close_matrix /= sum_raw

        vig_sq_mat, max_squeeze, squeezed_cells = self.calculate_cell_level_vig_squeeze(mat_close, raw_close_matrix)

        moa_index = self.calculate_marginal_overround_asymmetry(main_ou_close)
        emd_goals, emd_diff = self.calculate_wasserstein_emd_1d(mat_open, mat_close)
        mass_analysis = self.analyze_mass_shifts(cs_open, cs_close, ou_line=main_ou_close['line'], z_mri_x=z_mri_x)
        pec_index = self.calculate_price_elasticity_compression(mass_analysis['under_shift_pct'], main_ou_close)
        cond_entropy = self.calculate_conditional_entropy(mat_close, main_ah_close['home_line'])

        micro_skew = self.calculate_micro_price_skew(mat_close, main_ou_close)
        obi_s_val = self.calculate_obi_s(mass_analysis, main_ou_close)
        phantom_val = self.calculate_phantom_shift(mat_open, mat_close, main_ou_close)

        p_ah_home = self.calculate_ah_probability(mat_close, main_ah_close['home_line'], is_home=True)
        fair_ah_home = 1.0 / max(p_ah_home, 1e-5)
        edge_ah_home = (main_ah_close['home_odd'] / fair_ah_home) - 1.0 if main_ah_close['home_odd'] > 1.0 else 0.0

        p_ah_away = self.calculate_ah_probability(mat_close, -main_ah_close['home_line'], is_home=False)
        fair_ah_away = 1.0 / max(p_ah_away, 1e-5)
        edge_ah_away = (main_ah_close['away_odd'] / fair_ah_away) - 1.0 if main_ah_close['away_odd'] > 1.0 else 0.0

        p_ou_over = self.calculate_ou_probability(mat_close, main_ou_close['line'], is_over=True)
        fair_ou_over = 1.0 / max(p_ou_over, 1e-5)
        edge_ou_over = (main_ou_close['over_odd'] / fair_ou_over) - 1.0 if main_ou_close['over_odd'] > 1.0 else 0.0

        p_ou_under = self.calculate_ou_probability(mat_close, main_ou_close['line'], is_over=False)
        fair_ou_under = 1.0 / max(p_ou_under, 1e-5)
        edge_ou_under = (main_ou_close['under_odd'] / fair_ou_under) - 1.0 if main_ou_close['under_odd'] > 1.0 else 0.0

        p_draw = self.calculate_draw_probability(mat_close)
        fair_x_odd = 1.0 / max(p_draw, 1e-5)
        edge_x_odd = (main_x_close / fair_x_odd) - 1.0 if main_x_close and main_x_close > 1.0 else 0.0

        x_gap = abs(fair_x_odd - (main_x_close if main_x_close and main_x_close > 1.0 else fair_x_odd))
        stress_index = self.calculate_cross_market_stress_index(kl_div, shin_var, x_gap)

        scenario = "Scenariul D: Sharp Re-evaluation (Piață Recalibrată)"
        signal = "✅ PIAȚĂ ECHILIBRATĂ"
        explanation = f"Piața s-a recalibrat natural. xG Total s-a mutat cu {round(delta_xg, 2)} goluri."

        if stress_index > 0.75 or cmsi_kl_val > 0.65:
            scenario = "Scenariul S: Cross-Market Anomaly (Stress Elevat)"
            signal = "🚨 ALERTĂ MAXIMĂ: INEFIENȚĂ STRUCTURALĂ DE PIAȚĂ (TOOTHPRINTS DETECTED)"
            explanation = f"CMSI a atins {round(cmsi_kl_val, 2)}. Există o ruptură masivă între Correct Score, Handicap și 1X2 acoperită artificial prin marjă."
        elif x_gap > 2.50 and main_x_close and main_x_close > 1.0:
            scenario = "Scenariul E: Structural Cement Fracture (Anomalie Cota X)"
            signal = "⚠️ CAPCANĂ PE FAVORIT / PRĂPASTIE STRUCTURALĂ"
            explanation = f"Prăpastia dintre Cotă Egal Fair ({round(fair_x_odd, 2)}) și Cotă Egal Piață ({main_x_close}) este uriașă ({round(x_gap, 2)}). Favoritul este overpriced!"

        return {
            'open_l': round(l_open, 2), 'open_m': round(m_open, 2), 'open_xg': round(l_open+m_open, 2),
            'close_l': round(l_close, 2), 'close_m': round(m_close, 2), 'close_xg': round(l_close+m_close, 2),
            'delta_l': round(l_close - l_open, 2), 'delta_m': round(m_close - m_open, 2), 'delta_xg': round(delta_xg, 2),
            'cs_ratio_close': round(cs_ratio_close, 2), 'delta_cs': round(delta_cs, 2),
            'ent_close': round(ent_close, 2), 'delta_ent': round(delta_ent, 2),
            'gini_close': round(gini_close, 3), 'delta_gini': round(delta_gini, 3),
            'top3_close': round(top3_close, 1), 'delta_top3': round(delta_top3, 1),
            'skew_close': round(skew_close, 2), 'mvi_close': round(mvi_close, 2),
            'kl_div': round(kl_div, 4), 'jsd_div': round(jsd_div, 4), 'mri_index': round(mri_index, 4),
            'ear_srp': round(ear_srp, 4), 'biv_skew': round(biv_skew, 3),
            'tail_corr': round(tail_corr, 3), 'shin_var': round(shin_var, 5), 'x_gap': round(x_gap, 2),
            'stress_index': round(stress_index, 3), 'fitted_theta': round(theta_close, 2), 'fitted_phi': round(phi_close, 3),
            'moa_index': round(moa_index, 3), 'emd_goals': round(emd_goals, 3), 'emd_diff': round(emd_diff, 3),
            'pec_index': round(pec_index, 3), 'cond_entropy': round(cond_entropy, 3),
            'micro_skew': round(micro_skew, 2), 'obi_s_val': round(obi_s_val, 3), 'phantom_val': round(phantom_val, 4),
            'z_mri_x': round(z_mri_x, 2),
            'svi_score_9x9': round(svi_score_9x9, 4),
            'res_heatmap_9x9': res_heatmap_9x9,
            'matrice_delta_9x9': matrice_delta_9x9,
            'fair_ah_home': round(fair_ah_home, 2), 'edge_ah_home': round(edge_ah_home * 100, 2),
            'fair_ah_away': round(fair_ah_away, 2), 'edge_ah_away': round(edge_ah_away * 100, 2),
            'fair_ou_over': round(fair_ou_over, 2), 'edge_ou_over': round(edge_ou_over * 100, 2),
            'fair_ou_under': round(fair_ou_under, 2), 'edge_ou_under': round(edge_ou_under * 100, 2),
            'fair_x_odd': round(fair_x_odd, 2), 'edge_x_odd': round(edge_x_odd * 100, 2),
            'scenario': scenario, 'signal': signal, 'explanation': explanation, 'matrix_close': mat_close,
            'mass_analysis': mass_analysis,
            'fig_quiver': fig_quiver, 'lap_map': lap_map, 'max_curv': round(max_curv, 4), 'stiffness': round(stiffness, 4),
            'is_bimodal': is_bimodal, 'num_peaks': num_peaks, 'peaks': peaks,
            'coskew_ha': round(coskew_ha, 4), 'regime_name': regime_name, 'regime_desc': regime_desc,
            'exp_min': round(exp_min, 1), 'p15': round(p15, 1), 'p30': round(p30, 1),
            # REZULTATE MODULUL 6
            'cmsi_kl_val': round(cmsi_kl_val, 4),
            'apf_score': round(apf_score, 4), 'diag_asym': round(diag_asym, 4),
            'vig_sq_mat': vig_sq_mat, 'max_squeeze': round(max_squeeze, 2), 'squeezed_cells': squeezed_cells
        }


# ==========================================
# FUNCȚIE CU CACHE PENTRU CALCUL RAPID
# ==========================================

@st.cache_data(show_spinner="Calculare matrice și detectare amprente în curs...")
def run_cached_engine_decode(cs_open_input, cs_close_input, main_ah_open, main_ah_close, main_ou_open, main_ou_close, main_x_open, main_x_close, mode, copula_type, copula_theta, auto_fit_copula, use_nbinom):
    engine = PureMarketEngine9x9()
    return engine.decode_comparative(
        cs_open_input, cs_close_input, main_ah_open, main_ah_close, main_ou_open, main_ou_close, main_x_open, main_x_close, 
        mode=mode, copula_type=copula_type, copula_theta=copula_theta, 
        auto_fit_copula=auto_fit_copula, use_nbinom=use_nbinom
    )


# ==========================================
# INTERFAȚĂ STREAMLIT 
# ==========================================

st.title("🕵️ Pure Market Engine 9x9 (Optimized + Copula + Market Toothprints)")

ah_options = [round(x, 2) for x in np.arange(-3.50, 3.75, 0.25)]
ou_options = [round(x, 2) for x in np.arange(1.25, 5.25, 0.25)]

cs_main = {
    (0,0): 12.0, (1,1): 6.8, (2,2): 13.5, (3,3): 50.0,
    (1,0): 9.0, (2,0): 15.0, (2,1): 11.0, (3,0): 35.0, (3,1): 22.0, (3,2): 28.0, (4,0): 80.0, (4,1): 65.0, (4,2): 70.0,
    (0,1): 8.5, (0,2): 10.5, (1,2): 8.35, (0,3): 24.0, (1,3): 15.0, (2,3): 20.0, (0,4): 60.0, (1,4): 50.0, (2,4): 55.0
}

cs_ext_home = {(5,0): 90.0, (5,1): 75.0, (5,2): 85.0, (6,0): 100.0, (6,1): 100.0, (6,2): 100.0}
cs_ext_away = {(0,5): 90.0, (1,5): 75.0, (2,5): 85.0, (0,6): 100.0, (1,6): 100.0, (2,6): 100.0}

st.sidebar.header("⚙️ Setări Motor Optimizare")
engine_mode = st.sidebar.radio("Mod Optimizare Solver:", ["Decoupled (Pure CS Matrix)", "Anchored (5.0 Weight)"])

st.sidebar.header("🔬 Setări Avansate Modelare")
use_nbinom = st.sidebar.checkbox("Activare Negative Binomial (Overdispersion)", value=False)

st.sidebar.header("🌀 Strat Copula (Dependență Non-Liniară)")
copula_type = st.sidebar.selectbox("Selectează Model Copula:", ["Fără", "Frank", "Gumbel", "Clayton"])
auto_fit_copula = st.sidebar.checkbox("Auto-Fit Optim Parametru Θ (Copula)", value=True)
copula_theta = 1.5

if copula_type != "Fără" and not auto_fit_copula:
    if copula_type == "Frank": min_t, max_t, default_t = 0.1, 10.0, 1.5
    elif copula_type == "Gumbel": min_t, max_t, default_t = 1.01, 5.0, 1.2
    elif copula_type == "Clayton": min_t, max_t, default_t = 0.1, 5.0, 1.0
        
    copula_theta = st.sidebar.slider(
        f"Parametru Manual Θ ({copula_type})", 
        min_value=float(min_t), max_value=float(max_t), value=float(default_t), step=0.1
    )

st.sidebar.header("🎯 1. Matrice Scor Corect (Open vs Close)")
fav_option = st.sidebar.radio("Extensie Favorit Extrem:", ["Fără", "Favorit Gazde (5-0..6-2)", "Favorit Oaspeți (0-5..2-6)"])

cs_open_input, cs_close_input = {}, {}

with st.sidebar.expander("📌 Correct Score OPEN", expanded=False):
    for score, default_odd in cs_main.items():
        cs_open_input[score] = st.number_input(f"Open {score[0]}-{score[1]}", value=default_odd, step=0.25, key=f"op_m_{score}")
    if fav_option == "Favorit Gazde (5-0..6-2)":
        for score, default_odd in cs_ext_home.items():
            cs_open_input[score] = st.number_input(f"Open {score[0]}-{score[1]}", value=default_odd, step=0.5, key=f"op_eh_{score}")
    elif fav_option == "Favorit Oaspeți (0-5..2-6)":
        for score, default_odd in cs_ext_away.items():
            cs_open_input[score] = st.number_input(f"Open {score[0]}-{score[1]}", value=default_odd, step=0.5, key=f"op_ea_{score}")

with st.sidebar.expander("📌 Correct Score CLOSE", expanded=True):
    for score, default_odd in cs_main.items():
        cs_close_input[score] = st.number_input(f"Close {score[0]}-{score[1]}", value=default_odd, step=0.25, key=f"cl_m_{score}")
    if fav_option == "Favorit Gazde (5-0..6-2)":
        for score, default_odd in cs_ext_home.items():
            cs_close_input[score] = st.number_input(f"Close {score[0]}-{score[1]}", value=default_odd, step=0.5, key=f"cl_eh_{score}")
    elif fav_option == "Favorit Oaspeți (0-5..2-6)":
        for score, default_odd in cs_ext_away.items():
            cs_close_input[score] = st.number_input(f"Close {score[0]}-{score[1]}", value=default_odd, step=0.5, key=f"cl_ea_{score}")

st.sidebar.header("⚖️ 2. Anchors Sharp OPEN & CLOSE (AH, O/U & 1X2 X)")
col_a1, col_a2 = st.sidebar.columns(2)

with col_a1:
    st.markdown("#### 🔓 Deschidere (OPEN)")
    home_ah_line_op = st.selectbox("AH Gazde Open", ah_options, index=13, key="ah_line_op")
    home_ah_odd_op = st.number_input("Cotă AH Gazde Open", value=1.95, step=0.01, key="ah_h_op")
    away_ah_odd_op = st.number_input("Cotă AH Oaspeți Open", value=1.95, step=0.01, key="ah_a_op")
    main_ou_line_op = st.selectbox("O/U Open", ou_options, index=5, key="ou_line_op")
    main_ou_odd_op = st.number_input("Cotă Over Open", value=1.90, step=0.01, key="ou_o_op")
    under_ou_odd_op = st.number_input("Cotă Under Open", value=1.90, step=0.01, key="ou_u_op")
    main_x_odd_op = st.number_input("Cotă X Open", value=3.40, step=0.05, key="x_op")

with col_a2:
    st.markdown("#### 🔒 Închidere (CLOSE)")
    home_ah_line_cl = st.selectbox("AH Gazde Close", ah_options, index=13, key="ah_line_cl")
    home_ah_odd_cl = st.number_input("Cotă AH Gazde Close", value=1.95, step=0.01, key="ah_h_cl")
    away_ah_odd_cl = st.number_input("Cotă AH Oaspeți Close", value=1.95, step=0.01, key="ah_a_cl")
    main_ou_line_cl = st.selectbox("O/U Close", ou_options, index=5, key="ou_line_cl")
    main_ou_odd_cl = st.number_input("Cotă Over Close", value=1.90, step=0.01, key="ou_o_cl")
    under_ou_odd_cl = st.number_input("Cotă Under Close", value=1.90, step=0.01, key="ou_u_cl")
    main_x_odd_cl = st.number_input("Cotă X Close", value=3.40, step=0.05, key="x_cl")

main_ah_open = {'home_line': home_ah_line_op, 'home_odd': home_ah_odd_op, 'away_odd': away_ah_odd_op}
main_ah_close = {'home_line': home_ah_line_cl, 'home_odd': home_ah_odd_cl, 'away_odd': away_ah_odd_cl}

main_ou_open = {'line': main_ou_line_op, 'over_odd': main_ou_odd_op, 'under_odd': under_ou_odd_op}
main_ou_close = {'line': main_ou_line_cl, 'over_odd': main_ou_odd_cl, 'under_odd': under_ou_odd_cl}

res = run_cached_engine_decode(
    cs_open_input, cs_close_input, 
    main_ah_open, main_ah_close, 
    main_ou_open, main_ou_close, 
    main_x_odd_op, main_x_odd_cl, 
    engine_mode, copula_type, copula_theta, 
    auto_fit_copula, use_nbinom
)

if res['is_bimodal']:
    st.error(f"⚠️ **Meci Bifurcat (Schizofrenie de Piață)**: Două Scenarii Incompatibile Cotate Simultanal ({res['num_peaks']} vârfuri detectate)!")

st.subheader("1. Metricile xG Pure & Structură (Shin Unmargined)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("xG Pur Gazde (λ)", f"{res['close_l']}", delta=f"{res['delta_l']} vs Open")
c2.metric("xG Pur Oaspeți (μ)", f"{res['close_m']}", delta=f"{res['delta_m']} vs Open")
c3.metric("xG Pur Total", f"{res['close_xg']}", delta=f"{res['delta_xg']} vs Open")
c4.metric("CS Ratio (Concentrare)", f"{res['cs_ratio_close']}", delta=f"{res['delta_cs']} vs Open")
c5.metric("Entropie Shannon (Haos)", f"{res['ent_close']}", delta=f"{res['delta_ent']} vs Open")

st.markdown("---")

# AFISARE TAB-URI MODULE (INCLUSIV NOUL MODUL 6)
st.subheader("🌊 Analiza Dinamică, Topologică & Amprente de Piață (Toothprints)")
tab_mod1, tab_mod2, tab_mod4, tab_mod5, tab_mod6 = st.tabs([
    "Modulul 1: Vector Field Flow", 
    "Modulul 2: Surface Laplacian", 
    "Modulul 4: Bivariate Co-Skewness", 
    "Modulul 5: First Passage Time",
    "🔥 Modulul 6: Market Toothprints & Stress"
])

with tab_mod1:
    st.pyplot(res['fig_quiver'])

with tab_mod2:
    mc1, mc2 = st.columns(2)
    mc1.metric("Max Local Curvature (Tensiune Max)", f"{res['max_curv']}")
    mc2.metric("Surface Stiffness Index", f"{res['stiffness']}")
    st.dataframe(np.round(res['lap_map'], 4), use_container_width=True)

with tab_mod4:
    st.metric("CoSkewness HA Score", f"{res['coskew_ha']:+.4f}")
    st.info(f"**Regim Detectat:** {res['regime_name']}\n\n_{res['regime_desc']}_")

with tab_mod5:
    fpt1, fpt2, fpt3 = st.columns(3)
    fpt1.metric("Minut Așteptat Prim Gol", f"{res['exp_min']}'")
    fpt2.metric("Probabilitate Gol 0-15 Min", f"{res['p15']}%")
    fpt3.metric("Probabilitate Gol 0-30 Min", f"{res['p30']}%")

# NOUL TAB - MODULUL 6
with tab_mod6:
    st.markdown("### 🦷 Detecția Urmelor de Dinți ai Algoritmului")
    
    t1, t2, t3 = st.columns(3)
    t1.metric("CMSI (KL Divergence 1X2 vs O/U)", f"{res['cmsi_kl_val']}")
    t2.metric("Asymmetric Pressure Field (APF)", f"{res['apf_score']}")
    t3.metric("Max Vig Squeeze (Taxă Max Celulă)", f"{res['max_squeeze']}x")

    if res['cmsi_kl_val'] > 0.65:
        st.error("🚨 **ALERTĂ CMSI CRITICĂ:** Algoritmul casei a fost prins în foarfecă! Există o dezaliniere structurală între 1X2 și Over/Under.")
    else:
        st.success("✅ **CMSI Normal:** Piața 1X2 și Over/Under sunt în armonie matematică.")

    st.markdown("#### 1. Heatmap 9x9: Taxa Dinamică de Protecție pe Celulă (Vig Squeeze)")
    st.caption("Valori > 1.25 (colorate intensiv) reprezintă exact celulele unde casa a umflat artificial marja pentru a opri arbitrajul.")
    st.dataframe(np.round(res['vig_sq_mat'], 2), use_container_width=True)

    if res['squeezed_cells']:
        st.warning(f"⚠️ **Scoruri Afectate de Squeeze Dinamic (>1.25x):** {[(c[0], c[1]) for c in res['squeezed_cells']]}")

st.markdown("---")

st.subheader("5. Decizie Tactică & Scenariu")
st.success(f"**SCENARIU DETECTAT:** {res['scenario']}")
st.info(f"**SEMNAL:** {res['signal']}\n\n**Explicație:** {res['explanation']}")

st.markdown("---")

st.subheader("6. Matricea Pură 9x9 Fără Marjă (%)")
st.dataframe(np.round(res['matrix_close'] * 100, 2), use_container_width=True)