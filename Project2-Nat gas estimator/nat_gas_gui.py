import numpy as np
from datetime import datetime, date
import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkfont

# ---------------------------------------------------------------------------
# Model (same OLS logic as main.py)
# ---------------------------------------------------------------------------
RAW_DATA = [
    ("10/31/20", 10.10), ("11/30/20", 10.30), ("12/31/20", 11.00),
    ("01/31/21", 10.90), ("02/28/21", 10.90), ("03/31/21", 10.90),
    ("04/30/21", 10.40), ("05/31/21",  9.84), ("06/30/21", 10.00),
    ("07/31/21", 10.10), ("08/31/21", 10.30), ("09/30/21", 10.20),
    ("10/31/21", 10.10), ("11/30/21", 11.20), ("12/31/21", 11.40),
    ("01/31/22", 11.50), ("02/28/22", 11.80), ("03/31/22", 11.50),
    ("04/30/22", 10.70), ("05/31/22", 10.70), ("06/30/22", 10.40),
    ("07/31/22", 10.50), ("08/31/22", 10.40), ("09/30/22", 10.80),
    ("10/31/22", 11.00), ("11/30/22", 11.60), ("12/31/22", 11.60),
    ("01/31/23", 12.10), ("02/28/23", 11.70), ("03/31/23", 12.00),
    ("04/30/23", 11.50), ("05/31/23", 11.20), ("06/30/23", 10.90),
    ("07/31/23", 11.40), ("08/31/23", 11.10), ("09/30/23", 11.50),
    ("10/31/23", 11.80), ("11/30/23", 12.20), ("12/31/23", 12.80),
    ("01/31/24", 12.60), ("02/29/24", 12.40), ("03/31/24", 12.70),
    ("04/30/24", 12.10), ("05/31/24", 11.40), ("06/30/24", 11.50),
    ("07/31/24", 11.60), ("08/31/24", 11.50), ("09/30/24", 11.80),
]

T0           = datetime(2020, 10, 31)
DATA_END     = datetime(2024,  9, 30)
FORECAST_END = datetime(2025,  9, 30)

def _to_years(dt):
    return (dt - T0).days / 365.25

def _build_row(t):
    import math
    w = 2 * math.pi * t
    return [1.0, t, math.sin(w), math.cos(w)]

_ts     = [_to_years(datetime.strptime(d, "%m/%d/%y")) for d, _ in RAW_DATA]
_prices = [p for _, p in RAW_DATA]
X = np.array([_build_row(t) for t in _ts])
y = np.array(_prices)
BETA, *_ = np.linalg.lstsq(X, y, rcond=None)

def estimate_price(query_date_str):
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(query_date_str.strip(), fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError("Use format YYYY-MM-DD")
    if dt < T0:
        raise ValueError(f"Before data start ({T0.date()})")
    if dt > FORECAST_END:
        raise ValueError(f"Beyond forecast limit ({FORECAST_END.date()})")
    t = _to_years(dt)
    price = float(np.array(_build_row(t)) @ BETA)
    is_forecast = dt > DATA_END
    return price, is_forecast, dt

# ---------------------------------------------------------------------------
# Colours & style constants
# ---------------------------------------------------------------------------
BG        = "#0f1923"
PANEL     = "#162030"
BORDER    = "#1e3048"
ACCENT    = "#00c2ff"
ACCENT2   = "#ff6b35"
TEXT      = "#e8f4f8"
MUTED     = "#5a7a8a"
HIST_CLR  = "#00c2ff"
FORE_CLR  = "#ff6b35"
SUCCESS   = "#00e5a0"

# ---------------------------------------------------------------------------
# Mini canvas chart
# ---------------------------------------------------------------------------
class SparkChart(tk.Canvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, highlightthickness=0, **kw)
        self._marker = None
        self._draw_base()

    def _draw_base(self):
        self.update_idletasks()
        w, h = self.winfo_width() or 560, self.winfo_height() or 160
        pad = 36
        xs = _ts
        ys = _prices
        min_x, max_x = min(xs), _to_years(FORECAST_END)
        min_y, max_y = 9.0, 14.0

        def sx(x): return pad + (x - min_x) / (max_x - min_x) * (w - 2*pad)
        def sy(y): return h - pad - (y - min_y) / (max_y - min_y) * (h - 2*pad)

        self._sx = sx; self._sy = sy
        self._min_x = min_x; self._max_x = max_x

        # grid lines
        for yv in [10, 11, 12, 13]:
            yc = sy(yv)
            self.create_line(pad, yc, w-pad, yc, fill=BORDER, dash=(4,4))
            self.create_text(pad-4, yc, text=f"${yv}", anchor="e",
                             fill=MUTED, font=("Courier", 8))

        # fitted curve (full range incl forecast)
        import math
        steps = 120
        pts_hist, pts_fore = [], []
        for i in range(steps+1):
            t = min_x + (max_x - min_x) * i / steps
            row = _build_row(t)
            price = float(np.array(row) @ BETA)
            xc, yc = sx(t), sy(price)
            if t <= _to_years(DATA_END):
                pts_hist.extend([xc, yc])
            else:
                if not pts_fore:   # bridge gap
                    pts_fore.extend([xc, yc])
                pts_fore.extend([xc, yc])

        if len(pts_hist) >= 4:
            self.create_line(*pts_hist, fill=HIST_CLR, width=2, smooth=True)
        if len(pts_fore) >= 4:
            self.create_line(*pts_fore, fill=FORE_CLR, width=2,
                             smooth=True, dash=(6,3))

        # actual dots
        for t, p in zip(_ts, _prices):
            xc, yc = sx(t), sy(p)
            self.create_oval(xc-3, yc-3, xc+3, yc+3,
                             fill=HIST_CLR, outline="")

        # divider line at data end
        xd = sx(_to_years(DATA_END))
        self.create_line(xd, pad//2, xd, h-pad//2, fill=MUTED,
                         dash=(3,3), width=1)
        self.create_text(xd+3, pad//2+2, text="data end", anchor="w",
                         fill=MUTED, font=("Courier", 7))

        # axis labels
        self.create_text(pad, h-10, text="2020", anchor="w",
                         fill=MUTED, font=("Courier", 8))
        self.create_text(w-pad, h-10, text="2025", anchor="e",
                         fill=MUTED, font=("Courier", 8))

        # legend
        lx, ly = w - pad - 120, pad + 4
        self.create_line(lx, ly+5, lx+18, ly+5, fill=HIST_CLR, width=2)
        self.create_text(lx+22, ly+5, text="fitted", anchor="w",
                         fill=HIST_CLR, font=("Courier", 8))
        self.create_line(lx, ly+18, lx+18, ly+18, fill=FORE_CLR,
                         width=2, dash=(5,2))
        self.create_text(lx+22, ly+18, text="forecast", anchor="w",
                         fill=FORE_CLR, font=("Courier", 8))

    def mark_date(self, query_dt, price):
        if self._marker:
            for item in self._marker:
                self.delete(item)
        t = _to_years(query_dt)
        xc = self._sx(t)
        yc = self._sy(price)
        color = FORE_CLR if query_dt > DATA_END else SUCCESS
        items = []
        items.append(self.create_line(xc, 0, xc, 9999,
                                      fill=color, dash=(4,2), width=1))
        items.append(self.create_oval(xc-6, yc-6, xc+6, yc+6,
                                      fill=color, outline=BG, width=2))
        self._marker = items

    def clear_marker(self):
        if self._marker:
            for item in self._marker:
                self.delete(item)
            self._marker = None

# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Natural Gas Price Estimator")
        self.configure(bg=BG)
        self.resizable(False, False)

        # ── header ──────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=30, pady=(28, 0))

        tk.Label(hdr, text="N A T   G A S", bg=BG, fg=ACCENT,
                 font=("Courier", 10, "bold")).pack(anchor="w")
        tk.Label(hdr, text="Price Estimator", bg=BG, fg=TEXT,
                 font=("Georgia", 26, "bold")).pack(anchor="w")
        tk.Label(hdr,
                 text="Trend + seasonality model  ·  Oct 2020 – Sep 2024  ·  +1 yr forecast",
                 bg=BG, fg=MUTED, font=("Courier", 9)).pack(anchor="w", pady=(4,0))

        sep = tk.Frame(self, bg=BORDER, height=1)
        sep.pack(fill="x", padx=30, pady=16)

        # ── chart ────────────────────────────────────────────────────────────
        chart_frame = tk.Frame(self, bg=PANEL, bd=0,
                               highlightbackground=BORDER, highlightthickness=1)
        chart_frame.pack(fill="x", padx=30, pady=(0,16))

        self.chart = SparkChart(chart_frame, width=560, height=160)
        self.chart.pack(padx=2, pady=2)
        self.after(50, self.chart._draw_base)   # redraw once geometry is set

        # ── input row ────────────────────────────────────────────────────────
        inp_frame = tk.Frame(self, bg=BG)
        inp_frame.pack(fill="x", padx=30)

        tk.Label(inp_frame, text="ENTER DATE", bg=BG, fg=MUTED,
                 font=("Courier", 8, "bold")).pack(anchor="w")

        row = tk.Frame(inp_frame, bg=BG)
        row.pack(fill="x", pady=(6,0))

        # spinboxes for day / month / year
        self._day   = tk.StringVar(value="15")
        self._month = tk.StringVar(value="06")
        self._year  = tk.StringVar(value="2025")

        spin_style = dict(bg=PANEL, fg=TEXT, insertbackground=ACCENT,
                          relief="flat", font=("Courier", 14, "bold"),
                          highlightbackground=BORDER, highlightthickness=1,
                          buttonbackground=PANEL, bd=0)

        tk.Label(row, text="YYYY", bg=BG, fg=MUTED,
                 font=("Courier", 7)).pack(side="left")
        year_box = tk.Spinbox(row, from_=2020, to=2025, width=5,
                              textvariable=self._year, **spin_style,
                              command=self._on_spin)
        year_box.pack(side="left", padx=(2,0))
        year_box.bind("<KeyRelease>", lambda e: self._on_spin())

        tk.Label(row, text=" – ", bg=BG, fg=MUTED,
                 font=("Courier", 14)).pack(side="left")

        tk.Label(row, text="MM", bg=BG, fg=MUTED,
                 font=("Courier", 7)).pack(side="left")
        mon_box = tk.Spinbox(row, from_=1, to=12, width=3,
                             textvariable=self._month, format="%02.0f",
                             **spin_style, command=self._on_spin)
        mon_box.pack(side="left", padx=(2,0))
        mon_box.bind("<KeyRelease>", lambda e: self._on_spin())

        tk.Label(row, text=" – ", bg=BG, fg=MUTED,
                 font=("Courier", 14)).pack(side="left")

        tk.Label(row, text="DD", bg=BG, fg=MUTED,
                 font=("Courier", 7)).pack(side="left")
        day_box = tk.Spinbox(row, from_=1, to=31, width=3,
                             textvariable=self._day, format="%02.0f",
                             **spin_style, command=self._on_spin)
        day_box.pack(side="left", padx=(2,0))
        day_box.bind("<KeyRelease>", lambda e: self._on_spin())

        # estimate button
        self.btn = tk.Button(row, text="ESTIMATE →",
                             bg=ACCENT, fg="#0f1923",
                             font=("Courier", 11, "bold"),
                             relief="flat", cursor="hand2",
                             padx=18, pady=6,
                             activebackground=ACCENT2,
                             activeforeground="#fff",
                             command=self._estimate)
        self.btn.pack(side="left", padx=(20,0))
        self.btn.bind("<Enter>", lambda e: self.btn.config(bg=ACCENT2, fg="#fff"))
        self.btn.bind("<Leave>", lambda e: self.btn.config(bg=ACCENT, fg="#0f1923"))

        sep2 = tk.Frame(self, bg=BORDER, height=1)
        sep2.pack(fill="x", padx=30, pady=16)

        # ── result panel ─────────────────────────────────────────────────────
        res_frame = tk.Frame(self, bg=PANEL,
                             highlightbackground=BORDER, highlightthickness=1)
        res_frame.pack(fill="x", padx=30)

        inner = tk.Frame(res_frame, bg=PANEL)
        inner.pack(padx=20, pady=16, fill="x")

        self.lbl_tag   = tk.Label(inner, text="AWAITING INPUT", bg=PANEL,
                                  fg=MUTED, font=("Courier", 9, "bold"))
        self.lbl_tag.pack(anchor="w")

        self.lbl_price = tk.Label(inner, text="—", bg=PANEL, fg=TEXT,
                                  font=("Georgia", 42, "bold"))
        self.lbl_price.pack(anchor="w")

        self.lbl_sub   = tk.Label(inner, text="per MMBtu", bg=PANEL,
                                  fg=MUTED, font=("Courier", 10))
        self.lbl_sub.pack(anchor="w")

        self.lbl_note  = tk.Label(inner, text="", bg=PANEL, fg=MUTED,
                                  font=("Courier", 9), wraplength=480,
                                  justify="left")
        self.lbl_note.pack(anchor="w", pady=(8,0))

        # ── model coefficients ───────────────────────────────────────────────
        sep3 = tk.Frame(self, bg=BORDER, height=1)
        sep3.pack(fill="x", padx=30, pady=16)

        coef_frame = tk.Frame(self, bg=BG)
        coef_frame.pack(padx=30, pady=(0,24), fill="x")

        tk.Label(coef_frame, text="MODEL COEFFICIENTS", bg=BG, fg=MUTED,
                 font=("Courier", 8, "bold")).pack(anchor="w", pady=(0,8))

        coefs = [
            ("a  intercept",    f"{BETA[0]:.4f}"),
            ("b  trend / yr",   f"+{BETA[1]:.4f}"),
            ("A  sin coeff",    f"{BETA[2]:.4f}"),
            ("B  cos coeff",    f"{BETA[3]:.4f}"),
            ("amplitude",       f"{np.sqrt(BETA[2]**2+BETA[3]**2):.4f}"),
        ]
        cf_row = tk.Frame(coef_frame, bg=BG)
        cf_row.pack(fill="x")
        for label, val in coefs:
            box = tk.Frame(cf_row, bg=PANEL,
                           highlightbackground=BORDER, highlightthickness=1)
            box.pack(side="left", padx=(0,8))
            tk.Label(box, text=label.upper(), bg=PANEL, fg=MUTED,
                     font=("Courier", 7), padx=10, pady=4).pack(anchor="w")
            tk.Label(box, text=val, bg=PANEL, fg=ACCENT,
                     font=("Courier", 13, "bold"), padx=10).pack(anchor="w", pady=(0,6))

        self.geometry("620x680")
        self.after(100, self.chart._draw_base)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _date_string(self):
        y = self._year.get().zfill(4)
        m = self._month.get().zfill(2)
        d = self._day.get().zfill(2)
        return f"{y}-{m}-{d}"

    def _on_spin(self):
        self.chart.clear_marker()
        self.lbl_tag.config(text="AWAITING INPUT", fg=MUTED)
        self.lbl_price.config(text="—", fg=TEXT)
        self.lbl_note.config(text="")

    def _estimate(self):
        ds = self._date_string()
        try:
            price, is_forecast, dt = estimate_price(ds)
        except ValueError as e:
            messagebox.showerror("Invalid Date", str(e))
            return

        color  = FORE_CLR if is_forecast else SUCCESS
        tag    = "FORECAST  ·  beyond training data" if is_forecast else "HISTORICAL ESTIMATE"
        note   = ("Extrapolated beyond the training window. "
                  "Uncertainty grows with distance from Sep 2024."
                  if is_forecast else
                  "Interpolated from the fitted trend + seasonality model.")

        self.lbl_tag.config(text=tag, fg=color)
        self.lbl_price.config(text=f"${price:.2f}", fg=color)
        self.lbl_note.config(text=note)
        self.chart.mark_date(dt, price)


if __name__ == "__main__":
    app = App()
    app.mainloop()
