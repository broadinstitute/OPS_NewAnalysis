__doc__ = ""


import numpy as np
import pandas as pd
import statsmodels.api as sm2
from scipy.optimize import minimize
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import ImageName
from cellprofiler_core.image import Image
from matplotlib import cm, colors





def safelog(vals):
    vals = np.array(vals)
    minpos = vals[vals > 0].min()
    return np.log2(vals + minpos / 2)


def perp_objective(params, x, y):
    m, b = params
    return np.abs(m*x + b - y).sum() / np.sqrt(m**2 + 1)

def rescale(x):
    mn = x.min()
    mx = x.max()
    return (x - mn) / (mx - mn)

# def rescale_clip(x, low_q=0.01, high_q=0.99):
#     # clip to percentiles so outliers and background floor don't dominate
#     lo = np.quantile(x, low_q)
#     hi = np.quantile(x, high_q)
#     x_clipped = np.clip(x, lo, hi)
#     return (x_clipped - lo) / (hi - lo)


# def adaptive_upper_cutoff(s, *, bg_frac=0.60, bg_quantile=0.20, inner=0.90, cap=0.995):
#     """
#     s: pandas Series (e.g., df_pat['mito'])
#     If >= bg_frac of values lie at/below the bg_quantile, treat image as background-heavy
#     and compute thresholds on non-background only. Otherwise use all pixels.
#     Uses a simple 90% 'fence' and caps at the 99.5th percentile.
#     """
#     qbg = s.quantile(bg_quantile)
#     frac_bg = (s <= qbg).mean()

#     s_use = s[s > qbg] if frac_bg >= bg_frac else s

#     # If everything got filtered (degenerate case), fall back to original series
#     if s_use.empty:
#         s_use = s

#     m50 = s_use.quantile(0.50)
#     m90 = s_use.quantile(inner)
#     # 90%-fence, capped to avoid a single spike setting the bar too high
#     thr = min(m90 + 2.0 * (m90 - m50), s_use.quantile(cap))
#     return float(thr)

class NMFpixle(ImageProcessing):

    module_name = "NMFpixle"
    variable_revision_number = 1

    def create_settings(self):
        super(NMFpixle, self).create_settings()

        self.bodipy_image = ImageSubscriber("bodipy", "None")
        self.mito_image = ImageSubscriber("mito", "None")

        self.red_name = ImageName("red", "")
        self.green_name = ImageName("green", "")
        self.yellow_name = ImageName("yellow", "")

    def settings(self):
        return [
            self.bodipy_image,
            self.mito_image,
            self.red_name,
            self.green_name,
            self.yellow_name,
        ]

    def visible_settings(self):
        return [
            self.bodipy_image,
            self.mito_image,
            self.red_name,
            self.green_name,
            self.yellow_name,
        ]




    def _get_input_arrays(self, workspace):
        image_set = workspace.image_set
        bodipy_img = image_set.get_image(self.bodipy_image.value)
        mito_img = image_set.get_image(self.mito_image.value)
        bp = bodipy_img.pixel_data.astype(np.float32)
        bp *= 65535
        mt = mito_img.pixel_data.astype(np.float32)
        mt *= 65535
        bp = safelog(bp)
        mt = safelog(mt)

        return bp, mt, bodipy_img

    def _downsample(self, bp, mt, step):
        h, w = bp.shape[:2]
        bp_ds = bp[::step, ::step]
        mt_ds = mt[::step, ::step]
        return bp_ds, mt_ds, h, w

    def _coords_downsampled(self, h, w, step):
        i_full, j_full = np.indices((h, w))
        i_ds = i_full[::step, ::step] / float(h)
        j_ds = j_full[::step, ::step] / float(w)
        return i_ds, j_ds

    def _build_df_pat(self, i_ds, j_ds, bp_ds, mt_ds):
        return pd.DataFrame({
            "i": i_ds.ravel(),
            "j": j_ds.ravel(),
            "bodipy": bp_ds.ravel(),
            "mito": mt_ds.ravel(),
        })
    


    # def _head_tail_table(self, df_pat, k=5):
    #     n = df_pat.shape[0]
    #     head_idx = np.arange(min(k, n))
    #     tail_idx = np.arange(max(0, n - k), n)
    #     combo_idx = np.concatenate([head_idx, tail_idx]) if n > k else head_idx

    #     col_labels = ["i", "j", "bodipy", "mito"]

    #     table_rows = [
    #         [
    #             float(df_pat.iloc[idx]["i"]),
    #             float(df_pat.iloc[idx]["j"]),
    #             float(df_pat.iloc[idx]["bodipy"]),
    #             float(df_pat.iloc[idx]["mito"]),
    #         ]
    #         for idx in combo_idx
    #     ]

    #     bodipy_vals = df_pat["bodipy"]
    #     mito_vals = df_pat["mito"]

    #     bmin_idx = np.argmin(bodipy_vals)
    #     bmax_idx = np.argmax(bodipy_vals)
    #     mmin_idx = np.argmin(mito_vals)
    #     mmax_idx = np.argmax(mito_vals)

    #     extrema_indices = [
    #         ("bodipy_min", bmin_idx),
    #         ("bodipy_max", bmax_idx),
    #         ("mito_min", mmin_idx),
    #         ("mito_max", mmax_idx),
    #     ]

    #     for label, idx in extrema_indices:
    #         row = df_pat.iloc[idx]
    #         table_rows.append([
    #             float(row["i"]),
    #             float(row["j"]),
    #             float(row["bodipy"]),
    #             float(row["mito"]),
    #         ])

    #     return col_labels, table_rows



    def run(self, workspace):
        step = 3

        # --- load inputs (full-res, log-scaled) ---
        bp, mt, bodipy_img = self._get_input_arrays(workspace)


        # --- downsample & build df for robust fitting ---
        bp_ds, mt_ds, h, w = self._downsample(bp, mt, step)
        i_ds, j_ds = self._coords_downsampled(h, w, step)
        df_pat = self._build_df_pat(i_ds, j_ds, bp_ds, mt_ds)
        print(df_pat)
        print(np.unique(bp_ds))

        # print(df_pat.shape, df_pat[['mito','bodipy']].isfinite().sum())


        # col_labels, table_rows = self._head_tail_table(df_pat, k=5)

        # lo_b = df_pat['bodipy'].quantile(0.01)
        # lo_m = df_pat['mito'  ].quantile(0.01)

        # df_pat = df_pat[
        #     (df_pat['bodipy'] > lo_b) &
        #     (df_pat['mito'  ] > lo_m)
        # ]


        #remove high-value extremes
        # mito_threshold   = df_pat['mito'].quantile(0.75)
        # bodipy_threshold = df_pat['bodipy'].quantile(0.75)

        # df_pat = df_pat[(df_pat['mito'] <= mito_threshold) & (df_pat['bodipy'] <= bodipy_threshold)]
        print(df_pat)
        # print("after trim:", df_pat.shape)
        # assert len(df_pat) > 10, "No pixels left after trimming"




        #initial linear fit: to be optimized later
        corrected_mito = df_pat['mito']
        corrected_bodipy = df_pat['bodipy']

        X_corr = sm2.add_constant(corrected_mito)
        rlm_model_corr = sm2.RLM(corrected_bodipy, X_corr, M=sm2.robust.norms.Hampel())
        rlm_results_corr = rlm_model_corr.fit()
        intercept_corr = rlm_results_corr.params.iloc[0]
        slope_corr = rlm_results_corr.params.iloc[1]




        #use minimize for orthogonal l1 fit.
        init = [slope_corr, intercept_corr]
        res = minimize(perp_objective, init, args=(corrected_mito.values, corrected_bodipy.values), method='Nelder-Mead')
        m_opt, b_opt = res.x

        #apply the fit to full resolution images
        b_pred_new     = m_opt * mt + b_opt
        m_inferred_new = (bp - b_opt) / m_opt


        #yellow channel
        yellow_scalar = np.minimum(bp, b_pred_new) + np.minimum(mt, m_inferred_new)
        # yellow = np.minimum(bp, b_pred_new) + np.minimum(mt, m_inferred_new)
        # split_corrected_yellow = np.dstack([rescale(yellow_scalar), rescale(yellow_scalar), np.zeros_like(yellow_scalar)])
        # split_corrected_yellow = np.dstack([yellow_scalar, yellow_scalar, np.zeros_like(yellow_scalar)])
        

        #red and blue channel
        # bodip_over      = np.clip(bp - b_pred_new, 0, None)
        bodipy_over     = np.clip(bp - b_pred_new, 0, None)
        mito_over       = np.clip(mt - m_inferred_new, 0, None)

        y_gray  = rescale(np.exp2(yellow_scalar)).astype(np.float32)
        g_gray  = rescale(np.exp2(bodipy_over)).astype(np.float32)
        r_gray  = rescale(np.exp2(mito_over)).astype(np.float32)
                          
        split_corrected_yellow = np.dstack([y_gray, y_gray, np.zeros_like(y_gray)])
        
        # y_gray  = yellow_scalar.astype(np.float32)
        # g_gray  = bodipy_over.astype(np.float32)
        # r_gray  = mito_over.astype(np.float32)

        split_corrected_green = np.dstack([
            np.zeros_like(g_gray),
            g_gray,
            # bodipy_over,
            np.zeros_like(g_gray)
        ])

        split_corrected_red = np.dstack([
            r_gray,
            # mito_over,
            np.zeros_like(r_gray),
            np.zeros_like(r_gray)
        ])



        # --- stash arrays for display() ---
        workspace.display_data.orig_bp = bp
        workspace.display_data.orig_mt = mt
        workspace.display_data.split_y = split_corrected_yellow
        workspace.display_data.split_g = split_corrected_green
        workspace.display_data.split_r = split_corrected_red
        workspace.display_data.dimensions = 2  

        workspace.display_data.hex_x = corrected_mito.values
        workspace.display_data.hex_y = corrected_bodipy.values
        workspace.display_data.m_opt = float(m_opt)
        workspace.display_data.b_opt = float(b_opt)


         
        image_set = workspace.image_set
        image_set.add(self.yellow_name.value, Image(y_gray, parent_image=bodipy_img, convert=False, dimensions=2))
        image_set.add(self.green_name.value,  Image(g_gray,  parent_image=bodipy_img, convert=False, dimensions=2))
        image_set.add(self.red_name.value,    Image(r_gray,    parent_image=bodipy_img, convert=False, dimensions=2))

    def display(self, workspace, figure):
        # 3×3 grid:
        # (0,0) Bodipy (grayscale + cb)   (0,1) Split: Yellow (RGB)   (0,2) Yellow (grayscale + cb)
        # (1,0) Mito (grayscale + cb)     (1,1) Split: Green (RGB)    (1,2) Green  (grayscale + cb)
        # (2,0) Hexbin + x=y + fit + cb   (2,1) Split: Red (RGB)      (2,2) Red    (grayscale + cb)
        figure.set_subplots(dimensions=workspace.display_data.dimensions, subplots=(3, 3))

        # Consistent colorbar sizing everywhere ➜ keeps columns aligned
        cb_kwargs = dict(fraction=0.046, pad=0.04)

        # --- Inputs (true-intensity colorbars) ---
        vmin_bp = workspace.display_data.orig_bp.min()
        vmax_bp = workspace.display_data.orig_bp.max()
        figure.subplot_imshow(
            x=0, y=0,
            image=workspace.display_data.orig_bp,
            title="Bodipy (input; log_2 scaled)",
            colormap="gray",
            normalize=False,
            vmin=vmin_bp, vmax=vmax_bp,
        )
        from matplotlib import cm, colors
        ax00 = figure.subplot(0, 0)
        figure.figure.colorbar(
            cm.ScalarMappable(norm=colors.Normalize(vmin=vmin_bp, vmax=vmax_bp), cmap="gray"),
            ax=ax00, **cb_kwargs
        )

        vmin_mt = workspace.display_data.orig_mt.min()
        vmax_mt = workspace.display_data.orig_mt.max()
        figure.subplot_imshow(
            x=1, y=0,
            image=workspace.display_data.orig_mt,
            title="Mito (input;log_2 scaled)",
            colormap="gray",
            normalize=False,
            vmin=vmin_mt, vmax=vmax_mt,
        )
        ax10 = figure.subplot(1, 0)
        figure.figure.colorbar(
            cm.ScalarMappable(norm=colors.Normalize(vmin=vmin_mt, vmax=vmax_mt), cmap="gray"),
            ax=ax10, **cb_kwargs
        )

        figure.subplot_imshow(image=workspace.display_data.split_y, title="Split: Yellow;exponentiated", x=0, y=1)
        figure.subplot_imshow(image=workspace.display_data.split_g, title="Split: Green;exponentiated",  x=1, y=1)
        figure.subplot_imshow(image=workspace.display_data.split_r, title="Split: Red;exponentiated",    x=2, y=1)

        y_gray = workspace.display_data.split_y[:, :, 0]
        g_gray = workspace.display_data.split_g[:, :, 1]
        r_gray = workspace.display_data.split_r[:, :, 0]

        vmin_y, vmax_y = y_gray.min(), y_gray.max()
        figure.subplot_imshow(x=0, y=2, image=y_gray, title="Yellow (grayscale);exponentiated",
                            colormap="gray", normalize=False, vmin=vmin_y, vmax=vmax_y)
        ax02 = figure.subplot(0, 2)
        figure.figure.colorbar(
            cm.ScalarMappable(norm=colors.Normalize(vmin=vmin_y, vmax=vmax_y), cmap="gray"),
            ax=ax02, **cb_kwargs
        )

        vmin_g, vmax_g = g_gray.min(), g_gray.max()
        figure.subplot_imshow(x=1, y=2, image=g_gray, title="Green (grayscale);exponentiated",
                            colormap="gray", normalize=False, vmin=vmin_g, vmax=vmax_g)
        ax12 = figure.subplot(1, 2)
        figure.figure.colorbar(
            cm.ScalarMappable(norm=colors.Normalize(vmin=vmin_g, vmax=vmax_g), cmap="gray"),
            ax=ax12, **cb_kwargs
        )

        vmin_r, vmax_r = r_gray.min(), r_gray.max()
        figure.subplot_imshow(x=2, y=2, image=r_gray, title="Red (grayscale);exponentiated",
                            colormap="gray", normalize=False, vmin=vmin_r, vmax=vmax_r)
        ax22 = figure.subplot(2, 2)
        figure.figure.colorbar(
            cm.ScalarMappable(norm=colors.Normalize(vmin=vmin_r, vmax=vmax_r), cmap="gray"),
            ax=ax22, **cb_kwargs
        )

        ax20 = figure.subplot(2, 0)
        hex_x = workspace.display_data.hex_x
        hex_y = workspace.display_data.hex_y
        m_opt = workspace.display_data.m_opt
        b_opt = workspace.display_data.b_opt

        hb = ax20.hexbin(hex_x, hex_y, bins='log') 
        lo = min(hex_x.min(), hex_y.min())
        hi = max(hex_x.max(), hex_y.max())
        ax20.plot([lo, hi], [lo, hi], color='black', linewidth=2, zorder=2)  

        import numpy as np
        x_line = np.linspace(hex_x.min(), hex_x.max(), 200)
        y_line = m_opt * x_line + b_opt
        ax20.plot(x_line, y_line, color='red', linewidth=2, zorder=3)

        fit_label = (rf"Bodipy = {m_opt:.2f}·Mito "
                    f"{'-' if b_opt < 0 else '+'} {abs(b_opt):.2f}")
        ax20.set_title(f"After correction (filtered)\n;log_2 scaled\n{fit_label}")

        ax20.set_xlabel("Mito intensity")
        ax20.set_ylabel("Bodipy intensity")
        figure.figure.colorbar(hb, ax=ax20, fraction=0.046, pad=0.04)



    def volumetric(self):
        return False
