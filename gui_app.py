import os
import sys
import threading
import time
import queue
import numpy as np
import pandas as pd
import joblib

import customtkinter as ctk
from xgboost import XGBClassifier

# Local backend imports
from data_preparation import prepare_stock_data
from model_training import get_all_nasdaq_tickers, train_single_stock_models

# Set global CustomTkinter appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# Theme Palette Definitions
THEME = {
    "bg": "#0b0f19",
    "card_bg": "#151c2c",
    "card_border": "#1e293b",
    "header_bg": "#111827",
    "text_main": "#f3f4f6",
    "text_muted": "#9ca3af",
    "accent": "#6366f1",
    "accent_hover": "#4f46e5",
    "buy": "#10b981",
    "buy_bg": "#064e3b",
    "buy_border": "#059669",
    "sell": "#ef4444",
    "sell_bg": "#7f1d1d",
    "sell_border": "#dc2626",
    "warning": "#f59e0b",
    "input_bg": "#1f2937",
    "item_hover": "#1e293b",
    "item_select": "#312e81",
}

UNIVERSAL_MODELS_CFG = {
    "xgboost": {"file": "xgboost_universal_nasdaq.json", "name": "XGBoost", "icon": "🚀", "type": "xgboost"},
    "lightgbm": {"file": "lgbm_universal_nasdaq.joblib", "name": "LightGBM", "icon": "⚡", "type": "joblib"},
    "random_forest": {"file": "rf_universal_nasdaq.joblib", "name": "Random Forest", "icon": "🌲", "type": "joblib"},
}

FEATURE_DESCRIPTIONS = {
    'RSI_14': "RSI (14) - Aşırı Alım/Satım",
    'MACD_Hist': "MACD Hist - Momentum",
    'SMA_20_Ratio': "Fiyat / SMA(20)",
    'SMA_50_Ratio': "Fiyat / SMA(50)",
    'Hourly_Return': "Son Saatlik Getiri (%)",
    'Return_lag_1': "1 Saat Önceki Getiri (%)",
    'Return_lag_2': "2 Saat Önceki Getiri (%)",
    'Return_lag_3': "3 Saat Önceki Getiri (%)",
}

class NasdaqDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Configuration
        self.title("🔮 NASDAQ AI Tahmin Merkezi (Masaüstü Uygulaması)")
        self.geometry("1280x850")
        self.minsize(1024, 720)
        self.configure(fg_color=THEME["bg"])

        # App State
        self.all_tickers = []
        self.filtered_tickers = []
        self.selected_ticker = None
        self.msg_queue = queue.Queue()
        self.current_specific_result = None

        # Build GUI Layout
        self._build_header()
        self._build_main_split()

        # Thread-safe Queue Poller
        self.after(100, self._process_queue)

        # Load Tickers Asynchronously
        threading.Thread(target=self._load_tickers_async, daemon=True).start()

    # ── Header Bar ────────────────────────────────────────────────────────────
    def _build_header(self):
        header_frame = ctk.CTkFrame(self, fg_color=THEME["header_bg"], corner_radius=0, height=70)
        header_frame.pack(fill="x", side="top")
        header_frame.pack_propagate(False)

        inner = ctk.CTkFrame(header_frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=20, pady=10)

        # Logo & Title
        logo_label = ctk.CTkLabel(inner, text="🔮", font=ctk.CTkFont(size=26))
        logo_label.pack(side="left", padx=(0, 12))

        title_box = ctk.CTkFrame(inner, fg_color="transparent")
        title_box.pack(side="left")

        lbl_title = ctk.CTkLabel(
            title_box,
            text="NASDAQ AI Tahmin Merkezi",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=THEME["text_main"]
        )
        lbl_title.pack(anchor="w")

        lbl_sub = ctk.CTkLabel(
            title_box,
            text="Native Python Desktop Application · Multi-Model Ensemble AI",
            font=ctk.CTkFont(size=11),
            text_color=THEME["text_muted"]
        )
        lbl_sub.pack(anchor="w")

        # Stats Badges
        stats_box = ctk.CTkFrame(inner, fg_color="transparent")
        stats_box.pack(side="right")

        self.lbl_stat_count = ctk.CTkLabel(
            stats_box,
            text="— Hisse",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1f2937",
            text_color="#ffffff",
            corner_radius=8,
            padx=12,
            pady=4
        )
        self.lbl_stat_count.pack(side="left", padx=6)

        lbl_stat_active = ctk.CTkLabel(
            stats_box,
            text="● 3 Model Aktif",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["buy_bg"],
            text_color=THEME["buy"],
            corner_radius=8,
            padx=12,
            pady=4
        )
        lbl_stat_active.pack(side="left", padx=6)

    # ── Main Layout Split ─────────────────────────────────────────────────────
    def _build_main_split(self):
        main_box = ctk.CTkFrame(self, fg_color="transparent")
        main_box.pack(fill="both", expand=True, padx=15, pady=15)

        # Left Panel (Stock Picker - Fixed Width ~340px)
        self.left_panel = ctk.CTkFrame(
            main_box,
            fg_color=THEME["card_bg"],
            corner_radius=12,
            border_width=1,
            border_color=THEME["card_border"],
            width=340
        )
        self.left_panel.pack(side="left", fill="y", padx=(0, 15))
        self.left_panel.pack_propagate(False)

        # Right Panel (Results Dashboard - Flexible Width)
        self.right_panel = ctk.CTkFrame(main_box, fg_color="transparent")
        self.right_panel.pack(side="left", fill="both", expand=True)

        self._build_left_picker()
        self._build_right_dashboard()

    # ── Left Stock Picker Panel ───────────────────────────────────────────────
    def _build_left_picker(self):
        # Panel Header
        p_head = ctk.CTkFrame(self.left_panel, fg_color="transparent", height=45)
        p_head.pack(fill="x", padx=15, pady=(15, 5))

        lbl_p_title = ctk.CTkLabel(p_head, text="📋 Hisse Seçimi", font=ctk.CTkFont(size=15, weight="bold"), text_color=THEME["text_main"])
        lbl_p_title.pack(side="left")

        self.lbl_filtered_count = ctk.CTkLabel(p_head, text="0 sonuç", font=ctk.CTkFont(size=11), text_color=THEME["text_muted"])
        self.lbl_filtered_count.pack(side="right")

        # Search Entry
        search_box = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        search_box.pack(fill="x", padx=15, pady=5)

        self.entry_search = ctk.CTkEntry(
            search_box,
            placeholder_text="🔍 Hisse ara... (AAPL, NVDA)",
            font=ctk.CTkFont(size=13),
            fg_color=THEME["input_bg"],
            border_color=THEME["card_border"],
            height=38,
            corner_radius=8
        )
        self.entry_search.pack(fill="x")
        self.entry_search.bind("<KeyRelease>", lambda e: self._on_search_change())

        # Sector Filter Dropdown
        sector_box = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        sector_box.pack(fill="x", padx=15, pady=5)

        self.option_sector = ctk.CTkOptionMenu(
            sector_box,
            values=["Tüm Sektörler"],
            command=lambda v: self._on_search_change(),
            fg_color=THEME["input_bg"],
            button_color=THEME["accent"],
            button_hover_color=THEME["accent_hover"],
            dropdown_fg_color=THEME["card_bg"],
            height=36,
            corner_radius=8
        )
        self.option_sector.pack(fill="x")

        # Scrollable Stock List Frame
        self.stock_list_scroll = ctk.CTkScrollableFrame(
            self.left_panel,
            fg_color=THEME["bg"],
            corner_radius=8,
            border_width=1,
            border_color=THEME["card_border"]
        )
        self.stock_list_scroll.pack(fill="both", expand=True, padx=15, pady=10)

        # Status Label inside list
        self.lbl_list_status = ctk.CTkLabel(
            self.stock_list_scroll,
            text="NASDAQ Hisseleri Yükleniyor...",
            font=ctk.CTkFont(size=12),
            text_color=THEME["accent"]
        )
        self.lbl_list_status.pack(pady=40)

        # Bottom Action Bar
        action_bar = ctk.CTkFrame(self.left_panel, fg_color="#111827", corner_radius=10, border_width=1, border_color=THEME["card_border"])
        action_bar.pack(fill="x", side="bottom", padx=15, pady=15)

        self.lbl_selected_symbol = ctk.CTkLabel(action_bar, text="Hisse Seçiniz", font=ctk.CTkFont(size=14, weight="bold"), text_color=THEME["text_main"])
        self.lbl_selected_symbol.pack(anchor="w", padx=12, pady=(10, 0))

        self.lbl_selected_name = ctk.CTkLabel(action_bar, text="Analiz etmek için listeden hisse seçin", font=ctk.CTkFont(size=10), text_color=THEME["text_muted"])
        self.lbl_selected_name.pack(anchor="w", padx=12, pady=(0, 8))

        self.btn_analyze = ctk.CTkButton(
            action_bar,
            text="🚀 Analiz Et",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=THEME["accent"],
            hover_color=THEME["accent_hover"],
            height=42,
            corner_radius=8,
            state="disabled",
            command=self._start_analysis
        )
        self.btn_analyze.pack(fill="x", padx=12, pady=(0, 10))

    # ── Right Dashboard Panel ────────────────────────────────────────────────
    def _build_right_dashboard(self):
        self.dash_scroll = ctk.CTkScrollableFrame(self.right_panel, fg_color="transparent")
        self.dash_scroll.pack(fill="both", expand=True)

        # 1. Empty State (Visible by default)
        self._build_empty_state()

        # 2. Results Container (Hidden by default)
        self.results_container = ctk.CTkFrame(self.dash_scroll, fg_color="transparent")

        self._build_results_header()
        self._build_price_card()
        self._build_universal_card()
        self._build_specific_card()

    def _build_empty_state(self):
        self.empty_frame = ctk.CTkFrame(self.dash_scroll, fg_color=THEME["card_bg"], corner_radius=12, border_width=1, border_color=THEME["card_border"])
        self.empty_frame.pack(fill="both", expand=True, padx=5, pady=5)

        lbl_icon = ctk.CTkLabel(self.empty_frame, text="📊", font=ctk.CTkFont(size=56))
        lbl_icon.pack(pady=(40, 10))

        lbl_title = ctk.CTkLabel(self.empty_frame, text="Multi-Model Analiz Sonuçları", font=ctk.CTkFont(size=20, weight="bold"), text_color=THEME["text_main"])
        lbl_title.pack()

        lbl_sub = ctk.CTkLabel(self.empty_frame, text="Sol panelden bir NASDAQ hissesi seçin ve Analiz Et butonuna tıklayın.", font=ctk.CTkFont(size=12), text_color=THEME["text_muted"])
        lbl_sub.pack(pady=(5, 30))

        features_box = ctk.CTkFrame(self.empty_frame, fg_color="transparent")
        features_box.pack(pady=(0, 40))

        items = [
            ("⚡", "3 Universal Model (XGBoost, LightGBM, Random Forest) ile anlık enstrüman tahmini"),
            ("🎯", "Hisseye özel 4 Model (XGBoost, LightGBM, Random Forest, SVM) eğitimi ve metrikleri"),
            ("🤖", "Ensemble Konsensüs kararları & Teknik gösterge paneli"),
        ]
        for icon, text in items:
            f_row = ctk.CTkFrame(features_box, fg_color="transparent")
            f_row.pack(anchor="w", pady=4)
            ctk.CTkLabel(f_row, text=icon, font=ctk.CTkFont(size=16)).pack(side="left", padx=(0, 10))
            ctk.CTkLabel(f_row, text=text, font=ctk.CTkFont(size=12), text_color=THEME["text_main"]).pack(side="left")

    def _build_results_header(self):
        res_header = ctk.CTkFrame(self.results_container, fg_color=THEME["card_bg"], corner_radius=12, border_width=1, border_color=THEME["card_border"])
        res_header.pack(fill="x", pady=(0, 15))

        self.lbl_res_ticker_symbol = ctk.CTkLabel(res_header, text="AAPL", font=ctk.CTkFont(size=22, weight="bold"), text_color=THEME["accent"])
        self.lbl_res_ticker_symbol.pack(side="left", padx=(20, 10), pady=15)

        self.lbl_res_ticker_name = ctk.CTkLabel(res_header, text="Apple Inc.", font=ctk.CTkFont(size=14), text_color=THEME["text_muted"])
        self.lbl_res_ticker_name.pack(side="left", pady=15)

        self.lbl_res_date = ctk.CTkLabel(res_header, text="", font=ctk.CTkFont(size=11), text_color=THEME["text_muted"])
        self.lbl_res_date.pack(side="right", padx=20, pady=15)

    def _build_price_card(self):
        self.card_price = ctk.CTkFrame(self.results_container, fg_color=THEME["card_bg"], corner_radius=12, border_width=1, border_color=THEME["card_border"])
        self.card_price.pack(fill="x", pady=(0, 15))

        p_head = ctk.CTkFrame(self.card_price, fg_color="transparent")
        p_head.pack(fill="x", padx=20, pady=(15, 5))
        ctk.CTkLabel(p_head, text="💰 Güncel Fiyat Bilgisi (yfinance 1h)", font=ctk.CTkFont(size=14, weight="bold"), text_color=THEME["text_main"]).pack(side="left")

        p_body = ctk.CTkFrame(self.card_price, fg_color="transparent")
        p_body.pack(fill="x", padx=20, pady=(0, 15))

        self.lbl_price_current = ctk.CTkLabel(p_body, text="$—", font=ctk.CTkFont(size=28, weight="bold"), text_color="#ffffff")
        self.lbl_price_current.pack(anchor="w")

        self.lbl_price_time = ctk.CTkLabel(p_body, text="Son mum: —", font=ctk.CTkFont(size=11), text_color=THEME["text_muted"])
        self.lbl_price_time.pack(anchor="w", pady=(0, 12))

        m_grid = ctk.CTkFrame(p_body, fg_color="transparent")
        m_grid.pack(fill="x")

        self.lbl_p_open = self._add_metric_card(m_grid, "Açılış", "$—")
        self.lbl_p_high = self._add_metric_card(m_grid, "En Yüksek", "$—")
        self.lbl_p_low = self._add_metric_card(m_grid, "En Düşük", "$—")
        self.lbl_p_volume = self._add_metric_card(m_grid, "Hacim", "—")

    def _add_metric_card(self, parent, label, default_val):
        box = ctk.CTkFrame(parent, fg_color=THEME["input_bg"], corner_radius=8, border_width=1, border_color=THEME["card_border"])
        box.pack(side="left", fill="x", expand=True, padx=4)

        ctk.CTkLabel(box, text=label, font=ctk.CTkFont(size=10), text_color=THEME["text_muted"]).pack(anchor="w", padx=10, pady=(6, 0))
        val_lbl = ctk.CTkLabel(box, text=default_val, font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["text_main"])
        val_lbl.pack(anchor="w", padx=10, pady=(0, 6))
        return val_lbl

    def _build_universal_card(self):
        self.card_universal = ctk.CTkFrame(self.results_container, fg_color=THEME["card_bg"], corner_radius=12, border_width=1, border_color=THEME["card_border"])
        self.card_universal.pack(fill="x", pady=(0, 15))

        u_head = ctk.CTkFrame(self.card_universal, fg_color="transparent")
        u_head.pack(fill="x", padx=20, pady=(15, 10))
        ctk.CTkLabel(u_head, text="🌐 Universal Modeller Tahminleri (Tüm NASDAQ Ensemble)", font=ctk.CTkFont(size=14, weight="bold"), text_color=THEME["text_main"]).pack(side="left")
        ctk.CTkLabel(u_head, text="⚡ Anlık Inference", font=ctk.CTkFont(size=11, weight="bold"), fg_color=THEME["accent"], text_color="#ffffff", corner_radius=6, padx=8, pady=2).pack(side="right")

        # Consensus Banner
        self.universal_consensus_frame = ctk.CTkFrame(self.card_universal, fg_color=THEME["buy_bg"], corner_radius=10, border_width=1, border_color=THEME["buy_border"])
        self.universal_consensus_frame.pack(fill="x", padx=20, pady=(0, 15))

        self.lbl_u_consensus_title = ctk.CTkLabel(self.universal_consensus_frame, text="Universal Konsensüs: AL (BUY)", font=ctk.CTkFont(size=15, weight="bold"), text_color="#ffffff")
        self.lbl_u_consensus_title.pack(anchor="w", padx=15, pady=(10, 0))

        self.lbl_u_consensus_desc = ctk.CTkLabel(self.universal_consensus_frame, text="Modellerin ortak kararı hesaplanıyor...", font=ctk.CTkFont(size=11), text_color="#d1fae5")
        self.lbl_u_consensus_desc.pack(anchor="w", padx=15, pady=(0, 10))

        # 3 Universal Models Grid Container
        self.u_models_container = ctk.CTkFrame(self.card_universal, fg_color="transparent")
        self.u_models_container.pack(fill="x", padx=20, pady=(0, 15))

        # Technical Feature Gauges Grid Container
        gauges_head = ctk.CTkFrame(self.card_universal, fg_color="transparent")
        gauges_head.pack(fill="x", padx=20, pady=(5, 5))
        ctk.CTkLabel(gauges_head, text="🔬 Modele Giren Teknik Göstergeler (Son Mum)", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["text_main"]).pack(anchor="w")

        self.gauges_grid = ctk.CTkFrame(self.card_universal, fg_color="transparent")
        self.gauges_grid.pack(fill="x", padx=20, pady=(0, 15))

    def _build_specific_card(self):
        self.card_specific = ctk.CTkFrame(self.results_container, fg_color=THEME["card_bg"], corner_radius=12, border_width=1, border_color=THEME["card_border"])
        self.card_specific.pack(fill="x", pady=(0, 15))

        s_head = ctk.CTkFrame(self.card_specific, fg_color="transparent")
        s_head.pack(fill="x", padx=20, pady=(15, 10))
        ctk.CTkLabel(s_head, text="🎯 Hisseye Özel 4 Model Eğitimi", font=ctk.CTkFont(size=14, weight="bold"), text_color=THEME["text_main"]).pack(side="left")

        self.lbl_specific_badge = ctk.CTkLabel(s_head, text="🔄 Eğitiliyor", font=ctk.CTkFont(size=11, weight="bold"), fg_color=THEME["warning"], text_color="#ffffff", corner_radius=6, padx=8, pady=2)
        self.lbl_specific_badge.pack(side="right")

        # Training Progress Section
        self.progress_box = ctk.CTkFrame(self.card_specific, fg_color="transparent")
        self.progress_box.pack(fill="x", padx=20, pady=(0, 15))

        self.prog_bar = ctk.CTkProgressBar(self.progress_box, fg_color=THEME["input_bg"], progress_color=THEME["accent"], height=14)
        self.prog_bar.pack(fill="x", pady=(5, 8))
        self.prog_bar.set(0.0)

        p_info = ctk.CTkFrame(self.progress_box, fg_color="transparent")
        p_info.pack(fill="x")
        self.lbl_prog_pct = ctk.CTkLabel(p_info, text="0%", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["accent"])
        self.lbl_prog_pct.pack(side="left")
        self.lbl_prog_stage = ctk.CTkLabel(p_info, text="Başlatılıyor...", font=ctk.CTkFont(size=12), text_color=THEME["text_muted"])
        self.lbl_prog_stage.pack(side="left", padx=10)

        # Training Done Result Frame
        self.specific_result_box = ctk.CTkFrame(self.card_specific, fg_color="transparent")

        # Specific Consensus Banner
        self.specific_consensus_frame = ctk.CTkFrame(self.specific_result_box, fg_color=THEME["buy_bg"], corner_radius=10, border_width=1, border_color=THEME["buy_border"])
        self.specific_consensus_frame.pack(fill="x", padx=20, pady=(0, 15))

        self.lbl_s_consensus_title = ctk.CTkLabel(self.specific_consensus_frame, text="Hisseye Özel Consensus: —", font=ctk.CTkFont(size=15, weight="bold"), text_color="#ffffff")
        self.lbl_s_consensus_title.pack(anchor="w", padx=15, pady=(10, 0))

        self.lbl_s_consensus_desc = ctk.CTkLabel(self.specific_consensus_frame, text="—", font=ctk.CTkFont(size=11), text_color="#d1fae5")
        self.lbl_s_consensus_desc.pack(anchor="w", padx=15, pady=(0, 10))

        # 4 Specific Model Cards Grid Container
        self.s_models_container = ctk.CTkFrame(self.specific_result_box, fg_color="transparent")
        self.s_models_container.pack(fill="x", padx=20, pady=(0, 15))

        # Feature Importance Tabview
        fi_head = ctk.CTkFrame(self.specific_result_box, fg_color="transparent")
        fi_head.pack(fill="x", padx=20, pady=(5, 5))
        ctk.CTkLabel(fi_head, text="📊 Öznitelik Önem Sıralaması (Feature Importance)", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["text_main"]).pack(anchor="w")

        self.fi_tabview = ctk.CTkTabview(
            self.specific_result_box,
            fg_color=THEME["input_bg"],
            segmented_button_selected_color=THEME["accent"],
            segmented_button_selected_hover_color=THEME["accent_hover"],
            corner_radius=10
        )
        self.fi_tabview.pack(fill="x", padx=20, pady=(0, 15))
        self.fi_tabview.add("🚀 XGBoost")
        self.fi_tabview.add("⚡ LightGBM")
        self.fi_tabview.add("🌲 Random Forest")

    # ── Async Ticker Loading ──────────────────────────────────────────────────
    def _load_tickers_async(self):
        try:
            tickers = get_all_nasdaq_tickers()
            self.msg_queue.put(("TICKERS_LOADED", tickers))
        except Exception as ex:
            self.msg_queue.put(("TICKERS_ERROR", str(ex)))

    def _on_tickers_loaded(self, tickers):
        self.all_tickers = tickers
        self.filtered_tickers = list(tickers)

        self.lbl_stat_count.configure(text=f"{len(tickers):,} Hisse")

        sectors = sorted(list(set(t.get("sector") for t in tickers if t.get("sector") and t.get("sector") != "None")))
        self.option_sector.configure(values=["Tüm Sektörler"] + sectors)

        self.lbl_list_status.pack_forget()
        self._render_stock_list()

    # ── Filter & Search Logic ─────────────────────────────────────────────────
    def _on_search_change(self):
        query = self.entry_search.get().strip().lower()
        sector = self.option_sector.get()

        def match(t):
            m_q = not query or (query in t["symbol"].lower() or query in t["name"].lower())
            m_s = (sector == "Tüm Sektörler") or (t.get("sector") == sector)
            return m_q and m_s

        self.filtered_tickers = [t for t in self.all_tickers if match(t)]
        self._render_stock_list()

    def _render_stock_list(self):
        # Clear existing items
        for child in self.stock_list_scroll.winfo_children():
            child.destroy()

        self.lbl_filtered_count.configure(text=f"{len(self.filtered_tickers)} sonuç")

        limit = 60
        for t in self.filtered_tickers[:limit]:
            item_btn = ctk.CTkButton(
                self.stock_list_scroll,
                text=f"{t['symbol']}  ·  {t['name'][:24]}",
                anchor="w",
                font=ctk.CTkFont(size=12),
                fg_color=THEME["input_bg"],
                hover_color=THEME["item_select"],
                text_color=THEME["text_main"],
                height=34,
                corner_radius=6,
                command=lambda ticker=t: self._select_stock(ticker)
            )
            item_btn.pack(fill="x", pady=2)

    def _select_stock(self, ticker):
        self.selected_ticker = ticker
        self.lbl_selected_symbol.configure(text=ticker["symbol"])
        self.lbl_selected_name.configure(text=ticker["name"])
        self.btn_analyze.configure(state="normal")

    # ── Analysis Control ──────────────────────────────────────────────────────
    def _start_analysis(self):
        if not self.selected_ticker:
            return

        ticker = self.selected_ticker["symbol"]
        name = self.selected_ticker["name"]

        self.btn_analyze.configure(state="disabled", text="⏳ Hesaplanıyor...")
        self.lbl_res_ticker_symbol.configure(text=ticker)
        self.lbl_res_ticker_name.configure(text=name)

        # Switch views
        self.empty_frame.pack_forget()
        self.results_container.pack(fill="both", expand=True)

        self._reset_results_cards()

        threading.Thread(target=self._run_analysis_async, args=(ticker,), daemon=True).start()

    def _reset_results_cards(self):
        self.lbl_price_current.configure(text="$—")
        self.lbl_price_time.configure(text="Son mum: —")
        self.lbl_p_open.configure(text="$—")
        self.lbl_p_high.configure(text="$—")
        self.lbl_p_low.configure(text="$—")
        self.lbl_p_volume.configure(text="—")

        self.universal_consensus_frame.configure(fg_color=THEME["buy_bg"], border_color=THEME["buy_border"])
        self.lbl_u_consensus_title.configure(text="Universal Konsensüs Hesaplanıyor...")
        self.lbl_u_consensus_desc.configure(text="Lütfen bekleyin...")

        for child in self.u_models_container.winfo_children():
            child.destroy()
        for child in self.gauges_grid.winfo_children():
            child.destroy()

        self.lbl_specific_badge.configure(text="🔄 Eğitiliyor", fg_color=THEME["warning"])
        self.progress_box.pack(fill="x", padx=20, pady=(0, 15))
        self.specific_result_box.pack_forget()
        self.prog_bar.set(0.0)
        self.lbl_prog_pct.configure(text="0%")
        self.lbl_prog_stage.configure(text="Başlatılıyor...")

    # ── Async Execution Thread ────────────────────────────────────────────────
    def _run_analysis_async(self, ticker):
        # 1. Fetch yfinance live price info
        price_info = None
        try:
            import yfinance as yf
            raw_df = yf.download(ticker, period="5d", interval="1h", auto_adjust=True, progress=False)
            if isinstance(raw_df.columns, pd.MultiIndex):
                raw_df.columns = raw_df.columns.get_level_values(0)
            if not raw_df.empty:
                last_candle = raw_df.iloc[-1]
                last_ts = raw_df.index[-1]
                price_info = {
                    "current_price": round(float(last_candle["Close"]), 2),
                    "open": round(float(last_candle["Open"]), 2),
                    "high": round(float(last_candle["High"]), 2),
                    "low": round(float(last_candle["Low"]), 2),
                    "volume": int(last_candle["Volume"]),
                    "last_candle_time": last_ts.strftime('%Y-%m-%d %H:%M') if hasattr(last_ts, 'strftime') else str(last_ts),
                }
        except Exception:
            pass

        self.msg_queue.put(("PRICE_INFO", price_info))

        # 2. Universal Inference
        try:
            df = prepare_stock_data(ticker, for_training=False)
            last_row = df.iloc[[-1]].copy()
            last_date = last_row.index[0]
            last_date_str = last_date.strftime('%Y-%m-%d %H:%M') if hasattr(last_date, 'strftime') else str(last_date)

            feature_values = {col: round(float(last_row[col].iloc[0]), 6) for col in last_row.columns if col != 'Target'}

            if 'Target' in last_row.columns:
                last_row.drop(columns=['Target'], inplace=True)

            X_latest_np = np.ascontiguousarray(last_row.to_numpy(), dtype=np.float32)

            universal_models_res = {}
            buy_count = 0
            sell_count = 0
            confidences = []

            for key, cfg in UNIVERSAL_MODELS_CFG.items():
                file_path = cfg["file"]
                if os.path.exists(file_path):
                    try:
                        if cfg["type"] == "xgboost":
                            m = XGBClassifier()
                            m.load_model(file_path)
                            pred = int(m.predict(X_latest_np)[0])
                            proba = float(m.predict_proba(X_latest_np)[0][pred])
                        else:
                            m = joblib.load(file_path)
                            pred = int(m.predict(last_row)[0])
                            proba = float(m.predict_proba(last_row)[0][pred])

                        conf_pct = round(proba * 100, 2)
                        signal = "BUY" if pred == 1 else "SELL"
                        if pred == 1:
                            buy_count += 1
                        else:
                            sell_count += 1
                        confidences.append(conf_pct)

                        universal_models_res[key] = {
                            "name": cfg["name"],
                            "icon": cfg["icon"],
                            "signal": signal,
                            "confidence": conf_pct,
                            "model_file": file_path,
                        }
                    except Exception as ex:
                        universal_models_res[key] = {"name": cfg["name"], "icon": cfg["icon"], "error": str(ex)}

            consensus_signal = "BUY" if buy_count >= sell_count else "SELL"
            avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

            u_res = {
                "signal": consensus_signal,
                "consensus": {
                    "signal": consensus_signal,
                    "buy_count": buy_count,
                    "sell_count": sell_count,
                    "total_models": len(confidences),
                    "avg_confidence": avg_confidence,
                },
                "models": universal_models_res,
                "last_date": last_date_str,
                "features": feature_values,
            }
            self.msg_queue.put(("UNIVERSAL_RESULTS", u_res))

        except Exception as e:
            self.msg_queue.put(("UNIVERSAL_ERROR", str(e)))

        # 3. Stock-Specific Multi-Model Training
        def progress_cb(stage, pct, msg):
            self.msg_queue.put(("TRAINING_PROGRESS", (pct, msg)))

        try:
            res = train_single_stock_models(ticker, progress_callback=progress_cb)
            self.msg_queue.put(("TRAINING_DONE", res))
        except Exception as e:
            self.msg_queue.put(("TRAINING_ERROR", str(e)))

    # ── Queue Dispatcher ──────────────────────────────────────────────────────
    def _process_queue(self):
        try:
            while True:
                msg_type, payload = self.msg_queue.get_nowait()

                if msg_type == "TICKERS_LOADED":
                    self._on_tickers_loaded(payload)

                elif msg_type == "TICKERS_ERROR":
                    self.lbl_list_status.configure(text=f"❌ Hata: {payload}", text_color=THEME["sell"])

                elif msg_type == "PRICE_INFO":
                    if payload:
                        self.lbl_price_current.configure(text=f"${payload['current_price']:,.2f}")
                        self.lbl_price_time.configure(text=f"Son mum: {payload['last_candle_time']}")
                        self.lbl_p_open.configure(text=f"${payload['open']:.2f}")
                        self.lbl_p_high.configure(text=f"${payload['high']:.2f}")
                        self.lbl_p_low.configure(text=f"${payload['low']:.2f}")
                        self.lbl_p_volume.configure(text=f"{payload['volume']:,}")

                elif msg_type == "UNIVERSAL_RESULTS":
                    self._render_universal_results(payload)

                elif msg_type == "UNIVERSAL_ERROR":
                    self.lbl_u_consensus_title.configure(text=f"⚠️ Hata: {payload}")

                elif msg_type == "TRAINING_PROGRESS":
                    pct, msg = payload
                    self.prog_bar.set(pct / 100.0)
                    self.lbl_prog_pct.configure(text=f"{pct}%")
                    self.lbl_prog_stage.configure(text=msg)

                elif msg_type == "TRAINING_DONE":
                    self._on_training_done(payload)

                elif msg_type == "TRAINING_ERROR":
                    self.lbl_specific_badge.configure(text="❌ Hata", fg_color=THEME["sell"])
                    self.lbl_prog_stage.configure(text=f"Hata: {payload}", text_color=THEME["sell"])
                    self.btn_analyze.configure(state="normal", text="🚀 Analiz Et")

        except queue.Empty:
            pass

        self.after(100, self._process_queue)

    # ── Render Universal Results ──────────────────────────────────────────────
    def _render_universal_results(self, data):
        self.lbl_res_date.configure(text=f"Son Veri Tarihi: {data.get('last_date', '')}")

        c = data.get("consensus", {})
        is_buy = c.get("signal") == "BUY"
        bg_col = THEME["buy_bg"] if is_buy else THEME["sell_bg"]
        border_col = THEME["buy_border"] if is_buy else THEME["sell_border"]

        self.universal_consensus_frame.configure(fg_color=bg_col, border_color=border_col)
        sign_tr = "AL (BUY)" if is_buy else "SAT (SELL)"
        self.lbl_u_consensus_title.configure(text=f"Universal Konsensüs: {sign_tr}")
        self.lbl_u_consensus_desc.configure(
            text=f"{c.get('buy_count',0)} / {c.get('total_models',0)} Model AL Yönünde · Ort. Güven: %{c.get('avg_confidence',0)}"
        )

        models = data.get("models", {})
        for key, m in models.items():
            card = ctk.CTkFrame(self.u_models_container, fg_color=THEME["input_bg"], corner_radius=10, border_width=1, border_color=THEME["card_border"])
            card.pack(side="left", fill="both", expand=True, padx=4)

            ctk.CTkLabel(card, text=f"{m['icon']} {m['name']}", font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["text_main"]).pack(anchor="w", padx=12, pady=(10, 4))

            if "error" in m:
                ctk.CTkLabel(card, text="Henüz eğitilmedi", font=ctk.CTkFont(size=11), text_color=THEME["text_muted"]).pack(anchor="w", padx=12, pady=(0, 10))
                continue

            m_is_buy = m["signal"] == "BUY"
            s_color = THEME["buy"] if m_is_buy else THEME["sell"]
            s_bg = THEME["buy_bg"] if m_is_buy else THEME["sell_bg"]

            lbl_sig = ctk.CTkLabel(
                card,
                text=f"{'▲' if m_is_buy else '▼'} {m['signal']}",
                font=ctk.CTkFont(size=12, weight="bold"),
                fg_color=s_bg,
                text_color=s_color,
                corner_radius=6,
                padx=8,
                pady=2
            )
            lbl_sig.pack(anchor="w", padx=12, pady=(4, 6))

            ctk.CTkLabel(card, text=f"Güven Oranı: %{m['confidence']}", font=ctk.CTkFont(size=11), text_color=THEME["text_main"]).pack(anchor="w", padx=12, pady=(0, 10))

        # Render Technical Gauges
        features = data.get("features", {})
        col = 0
        row_frame = None
        for f_name, f_val in features.items():
            if col % 4 == 0:
                row_frame = ctk.CTkFrame(self.gauges_grid, fg_color="transparent")
                row_frame.pack(fill="x", pady=2)

            box = ctk.CTkFrame(row_frame, fg_color=THEME["input_bg"], corner_radius=6, border_width=1, border_color=THEME["card_border"])
            box.pack(side="left", fill="x", expand=True, padx=2)

            desc = FEATURE_DESCRIPTIONS.get(f_name, f_name)
            ctk.CTkLabel(box, text=desc, font=ctk.CTkFont(size=9), text_color=THEME["text_muted"]).pack(anchor="w", padx=8, pady=(4, 0))
            ctk.CTkLabel(box, text=f"{f_val:.4f}", font=ctk.CTkFont(size=11, weight="bold"), text_color=THEME["accent"]).pack(anchor="w", padx=8, pady=(0, 4))
            col += 1

    # ── Render Training Done ──────────────────────────────────────────────────
    def _on_training_done(self, result):
        self.current_specific_result = result

        self.lbl_specific_badge.configure(text="✅ Tamamlandı", fg_color=THEME["buy"])
        self.progress_box.pack_forget()
        self.specific_result_box.pack(fill="x", padx=20, pady=(0, 15))
        self.btn_analyze.configure(state="normal", text="🚀 Analiz Et")

        c = result.get("consensus", {})
        is_buy = c.get("signal") == "BUY"
        bg_col = THEME["buy_bg"] if is_buy else THEME["sell_bg"]
        border_col = THEME["buy_border"] if is_buy else THEME["sell_border"]

        self.specific_consensus_frame.configure(fg_color=bg_col, border_color=border_col)
        sign_tr = "AL (BUY)" if is_buy else "SAT (SELL)"
        self.lbl_s_consensus_title.configure(text=f"Hisseye Özel Consensus: {sign_tr}")
        self.lbl_s_consensus_desc.configure(
            text=f"{result['ticker']} için {c.get('buy_count',0)} / 3 Model AL Sinyali Veriyor · Ort. Güven: %{c.get('avg_confidence',0)} · Veri: {result.get('total_records',0):,} Mum"
        )

        for child in self.s_models_container.winfo_children():
            child.destroy()

        models = result.get("models", {})
        for key, m in models.items():
            card = ctk.CTkFrame(self.s_models_container, fg_color=THEME["input_bg"], corner_radius=10, border_width=1, border_color=THEME["card_border"])
            card.pack(side="left", fill="both", expand=True, padx=3)

            ctk.CTkLabel(card, text=f"{m['icon']} {m['name']}", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["text_main"]).pack(anchor="w", padx=10, pady=(8, 2))

            m_is_buy = m["signal"] == "BUY"
            s_color = THEME["buy"] if m_is_buy else THEME["sell"]
            s_bg = THEME["buy_bg"] if m_is_buy else THEME["sell_bg"]

            lbl_sig = ctk.CTkLabel(
                card,
                text=f"{'▲' if m_is_buy else '▼'} {m['signal']}",
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color=s_bg,
                text_color=s_color,
                corner_radius=6,
                padx=6,
                pady=2
            )
            lbl_sig.pack(anchor="w", padx=10, pady=4)

            ctk.CTkLabel(card, text=f"Güven: %{m['confidence']}", font=ctk.CTkFont(size=10), text_color=THEME["text_main"]).pack(anchor="w", padx=10)
            ctk.CTkLabel(card, text=f"Accuracy: %{m['accuracy']*100:.1f}", font=ctk.CTkFont(size=10), text_color=THEME["text_muted"]).pack(anchor="w", padx=10)
            ctk.CTkLabel(card, text=f"Buy Prec.: %{m['precision_buy']*100:.1f}", font=ctk.CTkFont(size=10), text_color=THEME["text_muted"]).pack(anchor="w", padx=10, pady=(0, 8))

        # Populate Feature Importance Tabview
        self._populate_fi_tabs(result.get("models", {}))

    def _populate_fi_tabs(self, models):
        tab_names = [("🚀 XGBoost", "xgboost"), ("⚡ LightGBM", "lightgbm"), ("🌲 Random Forest", "random_forest")]

        for tab_title, model_key in tab_names:
            tab = self.fi_tabview.tab(tab_title)
            for child in tab.winfo_children():
                child.destroy()

            m_data = models.get(model_key, {})
            fi_list = m_data.get("feature_importance", [])

            if not fi_list:
                ctk.CTkLabel(tab, text="Bu model için feature importance verisi mevcut değil.", font=ctk.CTkFont(size=11), text_color=THEME["text_muted"]).pack(pady=15)
                continue

            max_imp = max(item["importance"] for item in fi_list) if fi_list else 1.0

            for item in fi_list:
                row = ctk.CTkFrame(tab, fg_color="transparent")
                row.pack(fill="x", pady=2)

                ctk.CTkLabel(row, text=item["feature"], font=ctk.CTkFont(size=11), text_color=THEME["text_main"], width=130, anchor="w").pack(side="left")

                pbar = ctk.CTkProgressBar(row, fg_color=THEME["bg"], progress_color=THEME["accent"], height=10)
                pbar.pack(side="left", fill="x", expand=True, padx=8)
                pbar.set(item["importance"] / max_imp if max_imp > 0 else 0)

                ctk.CTkLabel(row, text=f"{item['importance']*100:.1f}%", font=ctk.CTkFont(size=11, weight="bold"), text_color=THEME["accent"], width=50, anchor="e").pack(side="right")


if __name__ == "__main__":
    app = NasdaqDesktopApp()
    app.mainloop()
