"""CustomTkinter GUI for the MEDIDOR internet speed collector."""

from __future__ import annotations

import datetime as dt
import csv
import logging
import math
import pathlib
import queue
import sys
import threading
from dataclasses import asdict
from tkinter import filedialog, ttk

try:
    import customtkinter as ctk
except ImportError as exc:  # pragma: no cover - runtime dependency check
    message = (
        "Missing dependency 'customtkinter'. Install with:\n"
        "  pip install customtkinter"
    )
    raise RuntimeError(message) from exc

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover - optional dependency
    Figure = None
    FigureCanvasTkAgg = None

try:
    import psycopg
except ImportError:  # pragma: no cover - optional dependency
    psycopg = None

from main import DEFAULT_OUTPUT, SpeedSample, append_sample_csv, collect_sample
from main import (
    DEFAULT_DB_HOST,
    DEFAULT_DB_NAME,
    DEFAULT_DB_PASSWORD,
    DEFAULT_DB_PORT,
    DEFAULT_DB_USER,
    append_prediction_postgres,
    append_sample_postgres,
)
from analyze import (
    add_outlier_flags,
    build_report,
    canonical_server_names,
    deduplicate_rows,
    load_data,
    write_clean_csv,
)
from train_forecast import (
    DEFAULT_INPUT as TRAIN_DEFAULT_INPUT,
    DEFAULT_MODEL_DIR as TRAIN_DEFAULT_MODEL_DIR,
    TARGET_COLUMNS as TRAIN_TARGET_COLUMNS,
    build_training_frame as train_build_training_frame,
    evaluate_forecast as train_evaluate_forecast,
    load_data as train_load_data,
    save_outputs as train_save_outputs,
    train_test_split_time as train_train_test_split_time,
)
from predict_next import (
    DEFAULT_MODEL as PREDICT_DEFAULT_MODEL,
    TARGET_SERVER_COUNTRY as PREDICT_TARGET_SERVER_COUNTRY,
    TARGET_SERVER_NAME as PREDICT_TARGET_SERVER_NAME,
    build_latest_feature_row as predict_build_latest_feature_row,
    load_data as predict_load_data,
    load_model as predict_load_model,
)

APP_LOG_PATH = pathlib.Path(__file__).with_name("app.log")
MODEL_REGISTRY_PATH = TRAIN_DEFAULT_MODEL_DIR / "model_registry.csv"
LOGGER = logging.getLogger("medidor.app")


def _configure_error_logging() -> None:
    if LOGGER.handlers:
        return
    APP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(APP_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
        ),
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(file_handler)
    LOGGER.propagate = False


def _install_exception_hooks() -> None:
    def _log_uncaught_exception(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: object,
    ) -> None:
        LOGGER.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    def _log_uncaught_thread_exception(args: threading.ExceptHookArgs) -> None:
        thread_name = args.thread.name if args.thread is not None else "unknown"
        LOGGER.critical(
            f"Uncaught thread exception in {thread_name}",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _log_uncaught_exception
    threading.excepthook = _log_uncaught_thread_exception


class MedidorApp(ctk.CTk):
    """Desktop app to collect and visualize internet speed measurements."""

    def __init__(self) -> None:
        super().__init__()
        self.title("MEDIDOR | Internet Speed")
        self.geometry("820x560")
        self.minsize(780, 520)

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.ui_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.running = False
        self.training_running = False
        self.predict_running = False

        self.output_var = ctk.StringVar(value=str(DEFAULT_OUTPUT))
        self.timeout_var = ctk.StringVar(value="20")
        self.interval_var = ctk.StringVar(value="60")
        self.storage_var = ctk.StringVar(value="postgres")
        self.db_host_var = ctk.StringVar(value=DEFAULT_DB_HOST)
        self.db_port_var = ctk.StringVar(value=str(DEFAULT_DB_PORT))
        self.db_name_var = ctk.StringVar(value=DEFAULT_DB_NAME)
        self.db_user_var = ctk.StringVar(value=DEFAULT_DB_USER)
        self.db_password_var = ctk.StringVar(value=DEFAULT_DB_PASSWORD)
        self.status_var = ctk.StringVar(value="Idle")
        self.last_down_var = ctk.StringVar(value="-")
        self.last_pred_down_var = ctk.StringVar(value="-")
        self.last_pred_valid_var = ctk.StringVar(value="-")
        self.last_pred_model_version_var = ctk.StringVar(value="-")
        self.last_up_var = ctk.StringVar(value="-")
        self.last_ping_var = ctk.StringVar(value="-")
        self.last_time_var = ctk.StringVar(value="-")
        self.last_server_var = ctk.StringVar(value="-")
        self.last_prediction_valid_until_utc: dt.datetime | None = None
        self.report_input_var = ctk.StringVar(value=str(DEFAULT_OUTPUT))
        self.report_clean_var = ctk.StringVar(value=str(self._default_clean_output()))
        self.train_input_var = ctk.StringVar(value=str(TRAIN_DEFAULT_INPUT))
        self.train_target_var = ctk.StringVar(value=TRAIN_TARGET_COLUMNS[0])
        self.train_horizon_var = ctk.StringVar(value="1")
        self.train_lags_var = ctk.StringVar(value="5")
        self.train_test_size_var = ctk.StringVar(value="0.2")
        self.train_model_output_var = ctk.StringVar(
            value=str(TRAIN_DEFAULT_MODEL_DIR / "medidor_forecast.joblib"),
        )
        self.train_metrics_output_var = ctk.StringVar(
            value=str(TRAIN_DEFAULT_MODEL_DIR / "medidor_forecast_metrics.json"),
        )
        self.train_status_var = ctk.StringVar(value="Idle")
        self.predict_input_var = ctk.StringVar(value=str(TRAIN_DEFAULT_INPUT))
        self.predict_model_var = ctk.StringVar(value=str(PREDICT_DEFAULT_MODEL))
        self.predict_status_var = ctk.StringVar(value="Idle")
        self.predict_chart_status_var = ctk.StringVar(
            value="Chart: waiting for prediction.",
        )
        self.predict_combined_chart_status_var = ctk.StringVar(
            value="Combined chart: waiting for data.",
        )
        self.model_health_status_var = ctk.StringVar(value="Model health: waiting for data.")
        self.model_health_mae_var = ctk.StringVar(value="-")
        self.model_health_rmse_var = ctk.StringVar(value="-")
        self.model_health_baseline_mae_var = ctk.StringVar(value="-")
        self.model_health_baseline_rmse_var = ctk.StringVar(value="-")
        self.model_health_improvement_var = ctk.StringVar(value="-")
        self.model_health_badge_var = ctk.StringVar(value="-")
        self.predict_history_chart_status_var = ctk.StringVar(
            value="Prediction history chart: waiting for data.",
        )
        self.predict_figure: Figure | None = None
        self.predict_axis = None
        self.predict_canvas = None
        self.predict_history_figure: Figure | None = None
        self.predict_history_axis = None
        self.predict_history_canvas = None
        self.predict_combined_figure: Figure | None = None
        self.predict_combined_axis = None
        self.predict_combined_canvas = None
        self.model_registry_status_var = ctk.StringVar(value="Model registry: waiting for data.")
        self.model_registry_tree: ttk.Treeview | None = None

        self._build_layout()
        self._poll_ui_queue()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        content_host = ctk.CTkFrame(self, corner_radius=12)
        content_host.grid(row=0, column=0, padx=14, pady=(14, 8), sticky="nsew")
        content_host.grid_columnconfigure(1, weight=1)
        content_host.grid_rowconfigure(0, weight=1)

        nav_panel = ctk.CTkFrame(content_host, width=170, corner_radius=10)
        nav_panel.grid(row=0, column=0, padx=(10, 8), pady=10, sticky="ns")
        nav_panel.grid_propagate(False)
        nav_panel.grid_columnconfigure(0, weight=1)

        pages_host = ctk.CTkFrame(content_host, corner_radius=10)
        pages_host.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        pages_host.grid_columnconfigure(0, weight=1)
        pages_host.grid_rowconfigure(0, weight=1)

        view_names = ["Settings", "Monitor", "Resultados", "Entrenar", "Predicciones", "Modelos"]
        self.view_frames: dict[str, ctk.CTkFrame] = {}
        self.view_buttons: dict[str, ctk.CTkButton] = {}
        for index, view_name in enumerate(view_names):
            button = ctk.CTkButton(
                nav_panel,
                text=view_name,
                command=lambda name=view_name: self._show_view(name),
                anchor="w",
            )
            button.grid(row=index, column=0, padx=10, pady=(10 if index == 0 else 6, 0), sticky="ew")
            self.view_buttons[view_name] = button

            frame = ctk.CTkFrame(pages_host, corner_radius=10)
            frame.grid(row=0, column=0, sticky="nsew")
            self.view_frames[view_name] = frame

        settings_tab = self.view_frames["Settings"]
        settings_tab.grid_columnconfigure(0, weight=1)
        settings_tab.grid_rowconfigure(0, weight=1)
        self._build_settings_form(settings_tab)

        monitor_tab = self.view_frames["Monitor"]
        monitor_tab.grid_columnconfigure(0, weight=1)
        monitor_tab.grid_rowconfigure(1, weight=1)

        body = ctk.CTkFrame(monitor_tab, corner_radius=12)
        body.grid(row=1, column=0, padx=14, pady=8, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(4, weight=1)

        self._metric_card(body, "Last Download (Mbps)", self.last_down_var, 0, 0)
        self._metric_card(body, "Predicted Download (Mbps)", self.last_pred_down_var, 0, 1)
        self._metric_card(body, "Last Upload (Mbps)", self.last_up_var, 1, 0)
        self._metric_card(body, "Last Ping (ms)", self.last_ping_var, 1, 1)
        self._metric_card(body, "Last Time (UTC)", self.last_time_var, 2, 0)
        self._metric_card(body, "Prediction Valid For", self.last_pred_valid_var, 2, 1)
        self._metric_card(body, "Model Version Used", self.last_pred_model_version_var, 3, 0)

        server_card = ctk.CTkFrame(body, corner_radius=10)
        server_card.grid(row=4, column=0, columnspan=2, padx=10, pady=(8, 0), sticky="nsew")
        server_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(server_card, text="Server").grid(
            row=0,
            column=0,
            padx=12,
            pady=(10, 2),
            sticky="w",
        )
        ctk.CTkLabel(
            server_card,
            textvariable=self.last_server_var,
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=1, column=0, padx=12, pady=(0, 10), sticky="w")

        log_panel = ctk.CTkFrame(monitor_tab, corner_radius=12)
        log_panel.grid(row=2, column=0, padx=14, pady=8, sticky="nsew")
        log_panel.grid_columnconfigure(0, weight=1)
        log_panel.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(log_panel, text="Event Log").grid(
            row=0,
            column=0,
            padx=12,
            pady=(10, 4),
            sticky="w",
        )
        self.log_box = ctk.CTkTextbox(log_panel, height=130)
        self.log_box.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        self.log_box.configure(state="disabled")

        results_tab = self.view_frames["Resultados"]
        results_tab.grid_columnconfigure(0, weight=1)
        results_tab.grid_rowconfigure(1, weight=1)

        result_controls = ctk.CTkFrame(results_tab, corner_radius=10)
        result_controls.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="ew")
        result_controls.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            result_controls,
            textvariable=self.report_input_var,
            anchor="w",
        ).grid(row=0, column=0, padx=12, pady=(8, 2), sticky="ew")
        ctk.CTkLabel(
            result_controls,
            textvariable=self.report_clean_var,
            anchor="w",
        ).grid(row=1, column=0, padx=12, pady=(0, 8), sticky="ew")
        ctk.CTkButton(
            result_controls,
            text="Actualizar resultados",
            command=self._refresh_analysis_report,
            width=170,
        ).grid(row=0, column=1, rowspan=2, padx=12, pady=8, sticky="e")

        self.report_box = ctk.CTkTextbox(results_tab)
        self.report_box.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.report_box.configure(state="disabled")

        train_tab = self.view_frames["Entrenar"]
        train_tab.grid_columnconfigure(0, weight=1)
        train_tab.grid_rowconfigure(1, weight=1)

        train_controls = ctk.CTkFrame(train_tab, corner_radius=10)
        train_controls.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="ew")
        train_controls.grid_columnconfigure(1, weight=1)
        train_controls.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(train_controls, text="Input CSV").grid(
            row=0,
            column=0,
            padx=(10, 6),
            pady=(10, 6),
            sticky="w",
        )
        ctk.CTkEntry(train_controls, textvariable=self.train_input_var).grid(
            row=0,
            column=1,
            padx=(0, 10),
            pady=(10, 6),
            sticky="ew",
        )

        ctk.CTkLabel(train_controls, text="Target").grid(
            row=0,
            column=2,
            padx=(6, 6),
            pady=(10, 6),
            sticky="w",
        )
        ctk.CTkOptionMenu(
            train_controls,
            variable=self.train_target_var,
            values=list(TRAIN_TARGET_COLUMNS),
            width=140,
        ).grid(row=0, column=3, padx=(0, 10), pady=(10, 6), sticky="w")

        ctk.CTkLabel(train_controls, text="Lags").grid(
            row=1,
            column=0,
            padx=(10, 6),
            pady=6,
            sticky="w",
        )
        ctk.CTkEntry(train_controls, textvariable=self.train_lags_var, width=90).grid(
            row=1,
            column=1,
            padx=(0, 10),
            pady=6,
            sticky="w",
        )

        ctk.CTkLabel(train_controls, text="Horizon").grid(
            row=1,
            column=2,
            padx=(6, 6),
            pady=6,
            sticky="w",
        )
        ctk.CTkEntry(
            train_controls,
            textvariable=self.train_horizon_var,
            width=90,
        ).grid(row=1, column=3, padx=(0, 10), pady=6, sticky="w")

        ctk.CTkLabel(train_controls, text="Test size").grid(
            row=2,
            column=0,
            padx=(10, 6),
            pady=6,
            sticky="w",
        )
        ctk.CTkEntry(
            train_controls,
            textvariable=self.train_test_size_var,
            width=90,
        ).grid(row=2, column=1, padx=(0, 10), pady=6, sticky="w")

        ctk.CTkLabel(train_controls, text="Model output").grid(
            row=3,
            column=0,
            padx=(10, 6),
            pady=6,
            sticky="w",
        )
        ctk.CTkEntry(train_controls, textvariable=self.train_model_output_var).grid(
            row=3,
            column=1,
            columnspan=3,
            padx=(0, 10),
            pady=6,
            sticky="ew",
        )

        ctk.CTkLabel(train_controls, text="Metrics output").grid(
            row=4,
            column=0,
            padx=(10, 6),
            pady=(6, 10),
            sticky="w",
        )
        ctk.CTkEntry(train_controls, textvariable=self.train_metrics_output_var).grid(
            row=4,
            column=1,
            columnspan=2,
            padx=(0, 10),
            pady=(6, 10),
            sticky="ew",
        )
        ctk.CTkLabel(train_controls, textvariable=self.train_status_var).grid(
            row=4,
            column=3,
            padx=(6, 10),
            pady=(6, 10),
            sticky="e",
        )

        self.btn_train = ctk.CTkButton(
            train_controls,
            text="Entrenar modelo",
            command=self._start_training,
            width=140,
        )
        self.btn_train.grid(row=5, column=0, columnspan=4, padx=10, pady=(0, 10), sticky="ew")

        self.train_box = ctk.CTkTextbox(train_tab)
        self.train_box.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.train_box.configure(state="disabled")

        predict_tab = self.view_frames["Predicciones"]
        predict_tab.grid_columnconfigure(0, weight=1)
        predict_tab.grid_rowconfigure(1, weight=0)
        predict_tab.grid_rowconfigure(2, weight=2)
        predict_tab.grid_rowconfigure(3, weight=2)
        predict_tab.grid_rowconfigure(4, weight=2)
        predict_tab.grid_rowconfigure(5, weight=1)

        predict_controls = ctk.CTkFrame(predict_tab, corner_radius=10)
        predict_controls.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="ew")
        predict_controls.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(predict_controls, text="Input CSV").grid(
            row=0,
            column=0,
            padx=(10, 6),
            pady=(10, 6),
            sticky="w",
        )
        ctk.CTkEntry(predict_controls, textvariable=self.predict_input_var).grid(
            row=0,
            column=1,
            padx=(0, 10),
            pady=(10, 6),
            sticky="ew",
        )

        ctk.CTkLabel(predict_controls, text="Model file").grid(
            row=1,
            column=0,
            padx=(10, 6),
            pady=(0, 6),
            sticky="w",
        )
        self.predict_model_picker = ctk.CTkOptionMenu(
            predict_controls,
            variable=self.predict_model_var,
            values=[str(PREDICT_DEFAULT_MODEL)],
        )
        self.predict_model_picker.grid(
            row=1,
            column=1,
            padx=(0, 10),
            pady=(0, 6),
            sticky="ew",
        )
        ctk.CTkButton(
            predict_controls,
            text="Refresh",
            command=self._refresh_model_picker,
            width=90,
        ).grid(row=1, column=2, padx=(0, 10), pady=(0, 6), sticky="e")

        self.btn_predict = ctk.CTkButton(
            predict_controls,
            text="Predecir siguiente",
            command=self._start_prediction,
            width=160,
        )
        self.btn_predict.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="w")
        ctk.CTkLabel(predict_controls, textvariable=self.predict_status_var).grid(
            row=2,
            column=1,
            padx=(0, 10),
            pady=(0, 10),
            sticky="e",
        )

        health_panel = ctk.CTkFrame(predict_tab, corner_radius=10)
        health_panel.grid(row=1, column=0, padx=10, pady=(0, 6), sticky="ew")
        health_panel.grid_columnconfigure((0, 1, 2), weight=1)
        ctk.CTkLabel(
            health_panel,
            textvariable=self.model_health_status_var,
            anchor="w",
        ).grid(row=0, column=0, columnspan=3, padx=10, pady=(8, 4), sticky="ew")
        ctk.CTkLabel(
            health_panel,
            textvariable=self.model_health_mae_var,
            anchor="w",
        ).grid(row=1, column=0, padx=10, pady=(0, 6), sticky="w")
        ctk.CTkLabel(
            health_panel,
            textvariable=self.model_health_rmse_var,
            anchor="w",
        ).grid(row=1, column=1, padx=10, pady=(0, 6), sticky="w")
        ctk.CTkLabel(
            health_panel,
            textvariable=self.model_health_improvement_var,
            anchor="w",
        ).grid(row=1, column=2, padx=10, pady=(0, 6), sticky="w")
        ctk.CTkLabel(
            health_panel,
            textvariable=self.model_health_baseline_mae_var,
            anchor="w",
        ).grid(row=2, column=0, padx=10, pady=(0, 8), sticky="w")
        ctk.CTkLabel(
            health_panel,
            textvariable=self.model_health_baseline_rmse_var,
            anchor="w",
        ).grid(row=2, column=1, padx=10, pady=(0, 8), sticky="w")
        self.model_health_badge_label = ctk.CTkLabel(
            health_panel,
            textvariable=self.model_health_badge_var,
            anchor="w",
            text_color="#8d8d8d",
        )
        self.model_health_badge_label.grid(row=2, column=2, padx=10, pady=(0, 8), sticky="w")

        predict_chart_panel = ctk.CTkFrame(predict_tab, corner_radius=10)
        predict_chart_panel.grid(row=2, column=0, padx=10, pady=(0, 6), sticky="nsew")
        predict_chart_panel.grid_columnconfigure(0, weight=1)
        predict_chart_panel.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            predict_chart_panel,
            textvariable=self.predict_chart_status_var,
            anchor="w",
        ).grid(row=0, column=0, padx=10, pady=(8, 4), sticky="ew")

        if Figure is None or FigureCanvasTkAgg is None:
            ctk.CTkLabel(
                predict_chart_panel,
                text="Install matplotlib to display download history chart.",
                anchor="w",
            ).grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        else:
            self.predict_figure = Figure(figsize=(6.8, 2.6), dpi=100)
            self.predict_axis = self.predict_figure.add_subplot(111)
            self.predict_axis.set_title("Download speed history")
            self.predict_axis.set_ylabel("Mbps")
            self.predict_axis.grid(alpha=0.25)

            self.predict_canvas = FigureCanvasTkAgg(
                self.predict_figure,
                master=predict_chart_panel,
            )
            self.predict_canvas.get_tk_widget().grid(
                row=1,
                column=0,
                padx=10,
                pady=(0, 10),
                sticky="nsew",
            )

        predict_history_chart_panel = ctk.CTkFrame(predict_tab, corner_radius=10)
        predict_history_chart_panel.grid(row=3, column=0, padx=10, pady=(0, 6), sticky="nsew")
        predict_history_chart_panel.grid_columnconfigure(0, weight=1)
        predict_history_chart_panel.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            predict_history_chart_panel,
            textvariable=self.predict_history_chart_status_var,
            anchor="w",
        ).grid(row=0, column=0, padx=10, pady=(8, 4), sticky="ew")

        if Figure is None or FigureCanvasTkAgg is None:
            ctk.CTkLabel(
                predict_history_chart_panel,
                text="Install matplotlib to display prediction history chart.",
                anchor="w",
            ).grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        else:
            self.predict_history_figure = Figure(figsize=(6.8, 2.6), dpi=100)
            self.predict_history_axis = self.predict_history_figure.add_subplot(111)
            self.predict_history_axis.set_title("Download prediction history")
            self.predict_history_axis.set_ylabel("Mbps")
            self.predict_history_axis.grid(alpha=0.25)

            self.predict_history_canvas = FigureCanvasTkAgg(
                self.predict_history_figure,
                master=predict_history_chart_panel,
            )
            self.predict_history_canvas.get_tk_widget().grid(
                row=1,
                column=0,
                padx=10,
                pady=(0, 10),
                sticky="nsew",
            )

        predict_combined_chart_panel = ctk.CTkFrame(predict_tab, corner_radius=10)
        predict_combined_chart_panel.grid(row=4, column=0, padx=10, pady=(0, 6), sticky="nsew")
        predict_combined_chart_panel.grid_columnconfigure(0, weight=1)
        predict_combined_chart_panel.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            predict_combined_chart_panel,
            textvariable=self.predict_combined_chart_status_var,
            anchor="w",
        ).grid(row=0, column=0, padx=10, pady=(8, 4), sticky="ew")

        if Figure is None or FigureCanvasTkAgg is None:
            ctk.CTkLabel(
                predict_combined_chart_panel,
                text="Install matplotlib to display combined chart.",
                anchor="w",
            ).grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        else:
            self.predict_combined_figure = Figure(figsize=(6.8, 2.6), dpi=100)
            self.predict_combined_axis = self.predict_combined_figure.add_subplot(111)
            self.predict_combined_axis.set_title("Combined download + prediction history")
            self.predict_combined_axis.set_ylabel("Mbps")
            self.predict_combined_axis.grid(alpha=0.25)

            self.predict_combined_canvas = FigureCanvasTkAgg(
                self.predict_combined_figure,
                master=predict_combined_chart_panel,
            )
            self.predict_combined_canvas.get_tk_widget().grid(
                row=1,
                column=0,
                padx=10,
                pady=(0, 10),
                sticky="nsew",
            )

        self.predict_box = ctk.CTkTextbox(predict_tab)
        self.predict_box.grid(row=5, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.predict_box.configure(state="disabled")

        models_tab = self.view_frames["Modelos"]
        models_tab.grid_columnconfigure(0, weight=1)
        models_tab.grid_rowconfigure(1, weight=1)

        models_controls = ctk.CTkFrame(models_tab, corner_radius=10)
        models_controls.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="ew")
        models_controls.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            models_controls,
            textvariable=self.model_registry_status_var,
            anchor="w",
        ).grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(
            models_controls,
            text="Refresh",
            command=self._refresh_model_registry_table,
            width=100,
        ).grid(row=0, column=1, padx=(6, 10), pady=10, sticky="e")

        models_table_frame = ctk.CTkFrame(models_tab, corner_radius=10)
        models_table_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        models_table_frame.grid_columnconfigure(0, weight=1)
        models_table_frame.grid_rowconfigure(0, weight=1)

        columns = ("id", "model_name", "model_path")
        self.model_registry_tree = ttk.Treeview(
            models_table_frame,
            columns=columns,
            show="headings",
        )
        self.model_registry_tree.heading("id", text="ID")
        self.model_registry_tree.heading("model_name", text="Model Name")
        self.model_registry_tree.heading("model_path", text="Model Path")
        self.model_registry_tree.column("id", width=70, anchor="center", stretch=False)
        self.model_registry_tree.column("model_name", width=220, anchor="w", stretch=False)
        self.model_registry_tree.column("model_path", width=760, anchor="w", stretch=True)
        self.model_registry_tree.grid(row=0, column=0, padx=(10, 0), pady=10, sticky="nsew")

        models_scroll = ttk.Scrollbar(
            models_table_frame,
            orient="vertical",
            command=self.model_registry_tree.yview,
        )
        models_scroll.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ns")
        self.model_registry_tree.configure(yscrollcommand=models_scroll.set)

        self._show_view("Monitor")

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.grid(row=1, column=0, padx=14, pady=(4, 14), sticky="ew")
        actions.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.btn_one = ctk.CTkButton(
            actions,
            text="Measure Once",
            command=self._run_once,
        )
        self.btn_one.grid(row=0, column=0, padx=6, sticky="ew")

        self.btn_start = ctk.CTkButton(
            actions,
            text="Start Monitoring",
            command=self._start_monitoring,
        )
        self.btn_start.grid(row=0, column=1, padx=6, sticky="ew")

        self.btn_stop = ctk.CTkButton(
            actions,
            text="Stop",
            command=self._stop_monitoring,
            state="disabled",
        )
        self.btn_stop.grid(row=0, column=2, padx=6, sticky="ew")

        ctk.CTkButton(actions, text="Clear Log", command=self._clear_log).grid(
            row=0,
            column=3,
            padx=6,
            sticky="ew",
        )
        self._refresh_model_picker()
        self._refresh_analysis_report()
        self._refresh_prediction_chart(pathlib.Path(self.predict_input_var.get()))
        self._refresh_prediction_history_chart()
        self._refresh_prediction_combined_chart(pathlib.Path(self.predict_input_var.get()))
        self._refresh_model_health()
        self._refresh_last_predicted_download()
        self._refresh_model_registry_table()

    def _build_settings_form(self, parent: ctk.CTkFrame) -> None:
        form = ctk.CTkFrame(parent, corner_radius=12)
        form.grid(row=0, column=0, padx=14, pady=10, sticky="nsew")
        form.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(form, text="Output CSV").grid(
            row=0,
            column=0,
            padx=(12, 8),
            pady=12,
            sticky="w",
        )
        ctk.CTkEntry(form, textvariable=self.output_var).grid(
            row=0,
            column=1,
            padx=8,
            pady=12,
            sticky="ew",
        )
        ctk.CTkButton(form, text="Browse", width=90, command=self._pick_output).grid(
            row=0,
            column=2,
            padx=(8, 12),
            pady=12,
            sticky="e",
        )

        ctk.CTkLabel(form, text="Timeout (s)").grid(
            row=1,
            column=0,
            padx=(12, 8),
            pady=(0, 12),
            sticky="w",
        )
        ctk.CTkEntry(form, textvariable=self.timeout_var, width=90).grid(
            row=1,
            column=1,
            padx=8,
            pady=(0, 12),
            sticky="w",
        )

        ctk.CTkLabel(form, text="Interval (s)").grid(
            row=1,
            column=1,
            padx=(130, 8),
            pady=(0, 12),
            sticky="w",
        )
        ctk.CTkEntry(form, textvariable=self.interval_var, width=90).grid(
            row=1,
            column=1,
            padx=(220, 8),
            pady=(0, 12),
            sticky="w",
        )

        ctk.CTkLabel(form, textvariable=self.status_var).grid(
            row=1,
            column=2,
            padx=(8, 12),
            pady=(0, 12),
            sticky="e",
        )

        ctk.CTkLabel(form, text="Storage").grid(
            row=2,
            column=0,
            padx=(12, 8),
            pady=(0, 8),
            sticky="w",
        )
        ctk.CTkOptionMenu(
            form,
            variable=self.storage_var,
            values=["csv", "postgres", "both"],
        ).grid(row=2, column=1, padx=8, pady=(0, 8), sticky="w")

        ctk.CTkLabel(form, text="DB host").grid(
            row=3,
            column=0,
            padx=(12, 8),
            pady=(0, 6),
            sticky="w",
        )
        ctk.CTkEntry(form, textvariable=self.db_host_var, width=120).grid(
            row=3,
            column=1,
            padx=8,
            pady=(0, 6),
            sticky="w",
        )

        ctk.CTkLabel(form, text="DB port").grid(
            row=3,
            column=1,
            padx=(140, 8),
            pady=(0, 6),
            sticky="w",
        )
        ctk.CTkEntry(form, textvariable=self.db_port_var, width=80).grid(
            row=3,
            column=1,
            padx=(210, 8),
            pady=(0, 6),
            sticky="w",
        )

        ctk.CTkLabel(form, text="DB name").grid(
            row=3,
            column=1,
            padx=(310, 8),
            pady=(0, 6),
            sticky="w",
        )
        ctk.CTkEntry(form, textvariable=self.db_name_var, width=100).grid(
            row=3,
            column=1,
            padx=(380, 8),
            pady=(0, 6),
            sticky="w",
        )

        ctk.CTkLabel(form, text="DB user").grid(
            row=4,
            column=0,
            padx=(12, 8),
            pady=(0, 12),
            sticky="w",
        )
        ctk.CTkEntry(form, textvariable=self.db_user_var, width=120).grid(
            row=4,
            column=1,
            padx=8,
            pady=(0, 12),
            sticky="w",
        )
        ctk.CTkLabel(form, text="DB password").grid(
            row=4,
            column=1,
            padx=(140, 8),
            pady=(0, 12),
            sticky="w",
        )
        ctk.CTkEntry(form, textvariable=self.db_password_var, width=160, show="*").grid(
            row=4,
            column=1,
            padx=(230, 8),
            pady=(0, 12),
            sticky="w",
        )

    def _show_view(self, view_name: str) -> None:
        for name, frame in self.view_frames.items():
            if name == view_name:
                frame.grid()
            else:
                frame.grid_remove()

        for name, button in self.view_buttons.items():
            button.configure(state="disabled" if name == view_name else "normal")

    def _metric_card(
        self,
        parent: ctk.CTkFrame,
        title: str,
        variable: ctk.StringVar,
        row: int,
        column: int,
    ) -> None:
        card = ctk.CTkFrame(parent, corner_radius=10)
        card.grid(row=row, column=column, padx=10, pady=8, sticky="nsew")
        ctk.CTkLabel(card, text=title).pack(anchor="w", padx=12, pady=(10, 2))
        ctk.CTkLabel(
            card,
            textvariable=variable,
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w", padx=12, pady=(0, 10))

    def _pick_output(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Select output CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            initialfile=pathlib.Path(self.output_var.get()).name,
            initialdir=str(pathlib.Path(self.output_var.get()).parent),
        )
        if selected:
            self.output_var.set(selected)
            self._refresh_analysis_report()

    def _default_clean_output(self) -> pathlib.Path:
        output_path = pathlib.Path(self.output_var.get())
        return output_path.with_name(f"{output_path.stem}_clean{output_path.suffix}")

    def _refresh_analysis_report(self) -> None:
        output_path = pathlib.Path(self.output_var.get())
        clean_path = self._default_clean_output()
        self.report_input_var.set(f"Input: {output_path}")
        self.report_clean_var.set(f"Clean CSV: {clean_path}")

        if not output_path.exists():
            self._set_report_text(f"No CSV found yet at:\n{output_path}")
            return

        try:
            rows = load_data(output_path)
            if not rows:
                self._set_report_text("CSV has no valid rows to analyze.")
                return
            rows, dedup_dropped = deduplicate_rows(rows)
            server_name_replacements = canonical_server_names(rows)
            outlier_counts = add_outlier_flags(rows)
            write_clean_csv(clean_path, rows)
            report = build_report(
                rows=rows,
                rolling_window=5,
                dedup_dropped=dedup_dropped,
                server_name_replacements=server_name_replacements,
                outlier_counts=outlier_counts,
                output_clean=clean_path,
            )
            self._set_report_text(report)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("CSV analysis failed")
            self._set_report_text(f"Could not analyze CSV:\n{exc}")

    def _set_running_ui(self, running: bool) -> None:
        self.running = running
        self.btn_one.configure(state="disabled" if running else "normal")
        self.btn_start.configure(state="disabled" if running else "normal")
        self.btn_stop.configure(state="normal" if running else "disabled")
        self.status_var.set("Running..." if running else "Idle")

    def _set_training_ui(self, running: bool) -> None:
        self.training_running = running
        self.btn_train.configure(state="disabled" if running else "normal")
        self.train_status_var.set("Training..." if running else "Idle")

    def _set_prediction_ui(self, running: bool) -> None:
        self.predict_running = running
        self.btn_predict.configure(state="disabled" if running else "normal")
        self.predict_status_var.set("Predicting..." if running else "Idle")

    def _list_model_paths(self) -> list[pathlib.Path]:
        model_dir = TRAIN_DEFAULT_MODEL_DIR
        model_paths = [path for path in model_dir.glob("*.joblib") if path.is_file()]
        return sorted(model_paths, key=lambda path: path.stat().st_mtime, reverse=True)

    def _refresh_model_picker(self, preferred: str | None = None) -> None:
        self._sync_model_registry_with_files()
        rows = self._load_model_registry_rows()
        rows_sorted = sorted(
            rows,
            key=lambda row: int(row.get("id", "0")) if row.get("id", "0").isdigit() else 0,
            reverse=True,
        )
        options = [row["model_name"] for row in rows_sorted if row.get("model_name")]
        if not options:
            options = ["No models found"]

        self.predict_model_picker.configure(values=options)
        selected = preferred if preferred in options else self.predict_model_var.get()
        if selected not in options:
            selected = options[0]
        self.predict_model_var.set(selected)

    def _find_model_registry_row_by_name(self, model_name: str) -> dict[str, str] | None:
        self._sync_model_registry_with_files()
        rows = self._load_model_registry_rows()
        for row in rows:
            if row.get("model_name") == model_name:
                return row
        return None

    def _load_model_registry_rows(self) -> list[dict[str, str]]:
        if not MODEL_REGISTRY_PATH.exists():
            return []
        rows: list[dict[str, str]] = []
        with MODEL_REGISTRY_PATH.open("r", encoding="utf-8", newline="") as file_handle:
            reader = csv.DictReader(file_handle)
            for row in reader:
                rows.append(
                    {
                        "id": str(row.get("id", "")).strip(),
                        "model_name": str(row.get("model_name", "")).strip(),
                        "model_path": str(row.get("model_path", "")).strip(),
                    },
                )
        return rows

    def _save_model_registry_rows(self, rows: list[dict[str, str]]) -> None:
        MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MODEL_REGISTRY_PATH.open("w", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=["id", "model_name", "model_path"])
            writer.writeheader()
            writer.writerows(rows)

    def _next_download_model_name(self, rows: list[dict[str, str]]) -> str:
        versions: list[tuple[int, int, int]] = []
        prefix = "Download Model v"
        for row in rows:
            model_name = row.get("model_name", "")
            if not model_name.startswith(prefix):
                continue
            raw_version = model_name.removeprefix(prefix)
            parts = raw_version.split(".")
            if len(parts) != 3 or not all(part.isdigit() for part in parts):
                continue
            versions.append((int(parts[0]), int(parts[1]), int(parts[2])))

        if not versions:
            return "Download Model v01.00.00"

        major, minor, patch = max(versions)
        patch += 1
        if patch > 99:
            patch = 0
            minor += 1
        if minor > 99:
            minor = 0
            major += 1
        return f"Download Model v{major:02d}.{minor:02d}.{patch:02d}"

    def _register_model_in_registry(self, model_path_raw: str) -> dict[str, str]:
        rows = self._load_model_registry_rows()
        model_path = str(pathlib.Path(model_path_raw).expanduser().resolve())
        for row in rows:
            if row.get("model_path") == model_path:
                return row

        max_id = 0
        for row in rows:
            try:
                max_id = max(max_id, int(row.get("id", "0")))
            except ValueError:
                continue
        new_row = {
            "id": str(max_id + 1),
            "model_name": self._next_download_model_name(rows),
            "model_path": model_path,
        }
        rows.append(new_row)
        self._save_model_registry_rows(rows)
        return new_row

    def _sync_model_registry_with_files(self) -> None:
        rows = self._load_model_registry_rows()
        known_paths = {row.get("model_path", "") for row in rows}
        changed = False
        for model_file in self._list_model_paths():
            resolved = str(model_file.resolve())
            if resolved in known_paths:
                continue
            next_id = 1
            for row in rows:
                try:
                    next_id = max(next_id, int(row.get("id", "0")) + 1)
                except ValueError:
                    continue
            new_row = {
                "id": str(next_id),
                "model_name": self._next_download_model_name(rows),
                "model_path": resolved,
            }
            rows.append(new_row)
            known_paths.add(resolved)
            changed = True
        if changed:
            self._save_model_registry_rows(rows)

    def _refresh_model_registry_table(self) -> None:
        if self.model_registry_tree is None:
            return
        try:
            self._sync_model_registry_with_files()
            rows = self._load_model_registry_rows()
            for item_id in self.model_registry_tree.get_children():
                self.model_registry_tree.delete(item_id)
            for row in rows:
                self.model_registry_tree.insert(
                    "",
                    "end",
                    values=(row["id"], row["model_name"], row["model_path"]),
                )
            self.model_registry_status_var.set(
                f"Model registry: {len(rows)} models | {MODEL_REGISTRY_PATH.resolve()}",
            )
            self._refresh_model_picker(preferred=self.predict_model_var.get())
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Model registry refresh failed")
            self.model_registry_status_var.set(f"Model registry error: {exc}")

    def _parse_positive_int(self, raw: str, field_name: str) -> int:
        value = int(raw)
        if value <= 0:
            message = f"{field_name} must be > 0"
            raise ValueError(message)
        return value

    def _parse_test_size(self, raw: str) -> float:
        value = float(raw)
        if not 0 < value < 0.5:
            raise ValueError("Test size must be > 0 and < 0.5")
        return value

    def _run_once(self) -> None:
        try:
            timeout = self._parse_positive_int(self.timeout_var.get(), "Timeout")
        except ValueError as exc:
            self._append_log(f"[error] {exc}")
            return

        self._set_running_ui(True)
        self.stop_event.clear()
        self.worker_thread = threading.Thread(
            target=self._worker_single,
            args=(timeout,),
            daemon=True,
        )
        self.worker_thread.start()

    def _start_monitoring(self) -> None:
        try:
            timeout = self._parse_positive_int(self.timeout_var.get(), "Timeout")
            interval = self._parse_positive_int(self.interval_var.get(), "Interval")
        except ValueError as exc:
            self._append_log(f"[error] {exc}")
            return

        self._set_running_ui(True)
        self.stop_event.clear()
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(timeout, interval),
            daemon=True,
        )
        self.worker_thread.start()
        self._append_log(f"[info] Monitoring started (interval={interval}s).")

    def _stop_monitoring(self) -> None:
        if not self.running:
            return
        self.stop_event.set()
        self._append_log("[info] Stop requested.")

    def _start_training(self) -> None:
        if self.running:
            self._append_log("[error] Stop monitoring before training.")
            return
        if self.training_running:
            return

        try:
            lags = self._parse_positive_int(self.train_lags_var.get(), "Lags")
            horizon = self._parse_positive_int(self.train_horizon_var.get(), "Horizon")
            test_size = self._parse_test_size(self.train_test_size_var.get())
            target = self.train_target_var.get()
            input_path = pathlib.Path(self.train_input_var.get())
            model_output = pathlib.Path(self.train_model_output_var.get())
            metrics_output = pathlib.Path(self.train_metrics_output_var.get())
        except ValueError as exc:
            self._append_log(f"[error] {exc}")
            return

        self._set_training_ui(True)
        self._append_log(
            "[info] Training started "
            f"(target={target}, lags={lags}, horizon={horizon}, test_size={test_size}).",
        )
        self.worker_thread = threading.Thread(
            target=self._worker_train,
            args=(input_path, target, lags, horizon, test_size, model_output, metrics_output),
            daemon=True,
        )
        self.worker_thread.start()

    def _start_prediction(self) -> None:
        if self.running:
            self._append_log("[error] Stop monitoring before prediction.")
            return
        if self.training_running:
            self._append_log("[error] Wait for training to finish before prediction.")
            return
        if self.predict_running:
            return

        input_path = pathlib.Path(self.predict_input_var.get())
        selected_model_name = self.predict_model_var.get().strip()
        if not selected_model_name or selected_model_name == "No models found":
            self._append_log("[error] No model selected. Train or refresh model list first.")
            return
        selected_row = self._find_model_registry_row_by_name(selected_model_name)
        if selected_row is None:
            self._append_log(
                f"[error] Selected model name was not found in registry: {selected_model_name}",
            )
            self._refresh_model_picker()
            return
        model_path_raw = str(selected_row.get("model_path", "")).strip()
        if not model_path_raw:
            self._append_log(
                f"[error] Selected model has no path in registry: {selected_model_name}",
            )
            self._refresh_model_picker()
            return
        model_path = pathlib.Path(model_path_raw)
        if not model_path.exists():
            self._append_log(f"[error] Selected model file not found: {model_path}")
            self._refresh_model_picker()
            return

        self._set_prediction_ui(True)
        self._append_log(
            "[info] Prediction started "
            f"(input={input_path}, model_name={selected_model_name}, model_path={model_path}).",
        )
        self.worker_thread = threading.Thread(
            target=self._worker_predict,
            args=(input_path, model_path),
            daemon=True,
        )
        self.worker_thread.start()

    def _worker_single(self, timeout: int) -> None:
        self._collect_and_store(timeout=timeout)
        self.ui_queue.put(("done", None))

    def _worker_loop(self, timeout: int, interval: int) -> None:
        while not self.stop_event.is_set():
            self._collect_and_store(timeout=timeout)
            if self.stop_event.wait(interval):
                break
        self.ui_queue.put(("done", None))

    def _worker_train(
        self,
        input_path: pathlib.Path,
        target: str,
        lags: int,
        horizon: int,
        test_size: float,
        model_output: pathlib.Path,
        metrics_output: pathlib.Path,
    ) -> None:
        try:
            frame = train_load_data(input_path)
            x, y = train_build_training_frame(
                frame=frame,
                target=target,
                lags=lags,
                horizon=horizon,
            )
            x_train, x_test, y_train, y_test = train_train_test_split_time(
                x=x,
                y=y,
                test_size=test_size,
            )
            model, metrics = train_evaluate_forecast(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                target=target,
            )
            outputs = train_save_outputs(
                model=model,
                x_train=x_train,
                metrics=metrics,
                target=target,
                horizon=horizon,
                lags=lags,
                model_output=model_output,
                metrics_output=metrics_output,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Model training failed")
            self.ui_queue.put(("training_error", str(exc)))
            return

        self.ui_queue.put(
            (
                "training_done",
                {
                    "target": target,
                    "samples_total": metrics.samples_total,
                    "samples_train": metrics.samples_train,
                    "samples_test": metrics.samples_test,
                    "mae_baseline": metrics.mae_baseline,
                    "rmse_baseline": metrics.rmse_baseline,
                    "mae_model": metrics.mae_model,
                    "rmse_model": metrics.rmse_model,
                    "model_version": outputs["model_version"],
                    "model_output": outputs["model_output"],
                    "metrics_output": outputs["metrics_output"],
                    "versioned_model_output": outputs["versioned_model_output"],
                    "versioned_metrics_output": outputs["versioned_metrics_output"],
                },
            ),
        )

    def _worker_predict(self, input_path: pathlib.Path, model_path: pathlib.Path) -> None:
        try:
            payload = predict_load_model(model_path)
            frame = predict_load_data(input_path)

            model = payload["model"]
            feature_columns = payload["feature_columns"]
            target = payload["target"]
            horizon = payload["horizon"]
            lags = payload["lags"]
            model_version = str(payload.get("model_version", pathlib.Path(model_path).stem))

            features = predict_build_latest_feature_row(
                frame=frame,
                feature_columns=feature_columns,
                lags=lags,
            )
            prediction = float(model.predict(features)[0])
            latest_ts = frame.iloc[-1]["timestamp_utc"]
            predicted_sample, expected_step_seconds = self._build_predicted_sample(
                frame=frame,
                target=str(target),
                horizon=int(horizon),
                prediction=prediction,
            )

            db_status = "Prediction not saved to PostgreSQL."
            try:
                db_port = self._parse_positive_int(self.db_port_var.get(), "DB port")
                append_prediction_postgres(
                    sample=predicted_sample,
                    host=self.db_host_var.get(),
                    port=db_port,
                    database=self.db_name_var.get(),
                    user=self.db_user_var.get(),
                    password=self.db_password_var.get(),
                    predicted_from_timestamp_utc=latest_ts.isoformat(),
                    target_metric=str(target),
                    horizon_steps=int(horizon),
                    expected_step_seconds=expected_step_seconds,
                    model_version=model_version,
                    input_source=str(input_path),
                )
                db_status = (
                    "Saved to PostgreSQL table "
                    "internet_speed_predictions "
                    f"({self.db_host_var.get()}:{self.db_port_var.get()}/"
                    f"{self.db_name_var.get()})"
                )
            except Exception as db_exc:  # noqa: BLE001
                LOGGER.exception("Saving prediction to PostgreSQL failed")
                db_status = f"Could not save prediction to PostgreSQL: {db_exc}"
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Prediction failed")
            self.ui_queue.put(("prediction_error", str(exc)))
            return

        self.ui_queue.put(
            (
                "prediction_done",
                {
                    "target": str(target),
                    "horizon": int(horizon),
                    "latest_ts": latest_ts.isoformat(),
                    "prediction": prediction,
                    "model_version": model_version,
                    "model_path": str(model_path),
                    "input_path": str(input_path),
                    "predicted_timestamp_utc": predicted_sample.timestamp_utc,
                    "db_status": db_status,
                },
            ),
        )

    def _collect_and_store(self, timeout: int) -> None:
        try:
            sample = collect_sample(timeout=timeout)
            storage = self.storage_var.get()
            output_path = pathlib.Path(self.output_var.get())
            if storage in {"csv", "both"}:
                append_sample_csv(sample=sample, output_path=output_path)
            if storage in {"postgres", "both"}:
                db_port = self._parse_positive_int(self.db_port_var.get(), "DB port")
                append_sample_postgres(
                    sample=sample,
                    host=self.db_host_var.get(),
                    port=db_port,
                    database=self.db_name_var.get(),
                    user=self.db_user_var.get(),
                    password=self.db_password_var.get(),
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Sample collection/storage failed")
            self.ui_queue.put(("error", str(exc)))
            return

        self.ui_queue.put(("sample", sample))
        if storage == "csv":
            self.ui_queue.put(("log", f"[ok] Saved sample to {output_path}"))
        elif storage == "postgres":
            self.ui_queue.put(
                (
                    "log",
                    "[ok] Saved sample to PostgreSQL "
                    f"({self.db_host_var.get()}:{self.db_port_var.get()}/"
                    f"{self.db_name_var.get()})",
                ),
            )
        else:
            self.ui_queue.put(
                ("log", f"[ok] Saved sample to CSV + PostgreSQL ({output_path})"),
            )

    def _poll_ui_queue(self) -> None:
        try:
            while True:
                event, payload = self.ui_queue.get_nowait()
                if event == "sample":
                    sample = payload
                    if isinstance(sample, SpeedSample):
                        self._apply_sample(sample)
                elif event == "log":
                    self._append_log(str(payload))
                elif event == "error":
                    self._append_log(f"[error] {payload}")
                    self._set_running_ui(False)
                elif event == "done":
                    self._set_running_ui(False)
                elif event == "training_error":
                    self._append_log(f"[error] Training failed: {payload}")
                    self._set_training_report(f"Training failed:\n{payload}")
                    self._set_training_ui(False)
                elif event == "training_done":
                    if isinstance(payload, dict):
                        explanation = self._build_training_explanation(payload)
                        try:
                            registry_row = self._register_model_in_registry(
                                str(payload["versioned_model_output"]),
                            )
                            self._refresh_model_picker(preferred=registry_row["model_name"])
                            self._refresh_model_registry_table()
                            self._append_log(
                                "[info] Model registry updated "
                                f"(id={registry_row['id']}, name={registry_row['model_name']}).",
                            )
                        except Exception as registry_exc:  # noqa: BLE001
                            LOGGER.exception("Model registry update failed")
                            self._append_log(f"[error] Model registry update failed: {registry_exc}")
                        self._append_log(
                            (
                                "[ok] Training completed "
                                f"(version={payload['model_version']}, "
                                f"model={payload['versioned_model_output']}, "
                                f"metrics={payload['metrics_output']})."
                            ),
                        )
                        self._set_training_report(
                            "Training completed\n"
                            f"Model version: {payload['model_version']}\n"
                            f"Target: {payload['target']}\n"
                            "Samples used: "
                            f"total={payload['samples_total']}, "
                            f"train={payload['samples_train']}, "
                            f"test={payload['samples_test']}\n"
                            "Baseline -> "
                            f"MAE={payload['mae_baseline']:.3f}, "
                            f"RMSE={payload['rmse_baseline']:.3f}\n"
                            "Model    -> "
                            f"MAE={payload['mae_model']:.3f}, "
                            f"RMSE={payload['rmse_model']:.3f}\n\n"
                            f"{explanation}\n"
                            f"Model saved (latest alias): {payload['model_output']}\n"
                            f"Versioned model saved: {payload['versioned_model_output']}\n"
                            f"Metrics saved (latest alias): {payload['metrics_output']}\n"
                            f"Versioned metrics saved: {payload['versioned_metrics_output']}",
                        )
                    self._set_training_ui(False)
                elif event == "prediction_error":
                    self._append_log(f"[error] Prediction failed: {payload}")
                    self._set_prediction_report(f"Prediction failed:\n{payload}")
                    self._set_prediction_ui(False)
                elif event == "prediction_done":
                    if isinstance(payload, dict):
                        self.last_pred_model_version_var.set(
                            str(payload.get("model_version", "-")),
                        )
                        if str(payload.get("target", "")) == "download_mbps":
                            self.last_pred_down_var.set(f"{float(payload['prediction']):.3f}")
                            self._set_prediction_valid_until(
                                str(payload.get("predicted_timestamp_utc", "")),
                            )
                        self._refresh_prediction_chart(
                            pathlib.Path(payload["input_path"]),
                            prediction_payload=payload,
                        )
                        self._refresh_prediction_history_chart()
                        self._refresh_prediction_combined_chart(
                            pathlib.Path(payload["input_path"]),
                            prediction_payload=payload,
                        )
                        self._refresh_model_health()
                        self._append_log(
                            (
                                "[ok] Prediction completed "
                                f"(target={payload['target']}, value={payload['prediction']:.3f})."
                            ),
                        )
                        self._append_log(f"[info] {payload['db_status']}")
                        self._set_prediction_report(
                            "Prediccion completada\n"
                            f"Target: {payload['target']}\n"
                            f"Horizon (steps): {payload['horizon']}\n"
                            f"Latest timestamp (UTC): {payload['latest_ts']}\n"
                            f"Predicted timestamp (UTC): {payload['predicted_timestamp_utc']}\n"
                            f"Predicted next value: {payload['prediction']:.3f}\n"
                            f"Model version: {payload['model_version']}\n"
                            f"PostgreSQL: {payload['db_status']}\n"
                            f"Model: {payload['model_path']}\n"
                            f"Input: {payload['input_path']}",
                        )
                    self._set_prediction_ui(False)
        except queue.Empty:
            pass
        self._refresh_prediction_validity_display()
        self.after(250, self._poll_ui_queue)

    def _apply_sample(self, sample: SpeedSample) -> None:
        row = asdict(sample)
        self.last_down_var.set(f"{row['download_mbps']}")
        self.last_up_var.set(f"{row['upload_mbps']}")
        self.last_ping_var.set(f"{row['ping_ms']}")
        self.last_time_var.set(str(row["timestamp_utc"]))
        self.last_server_var.set(f"{row['server_name']} ({row['server_country']})")
        self._append_log(
            "[sample] "
            f"down={row['download_mbps']} Mbps | "
            f"up={row['upload_mbps']} Mbps | "
            f"ping={row['ping_ms']} ms",
        )
        self._refresh_analysis_report()
        self._refresh_model_health()

    def _build_predicted_sample(
        self,
        frame,
        target: str,
        horizon: int,
        prediction: float,
    ) -> tuple[SpeedSample, int]:
        latest = frame.iloc[-1]
        predicted_timestamp, step_seconds = self._estimate_prediction_timestamp(
            frame=frame,
            horizon=horizon,
        )

        download = float(latest["download_mbps"])
        upload = float(latest["upload_mbps"])
        ping = float(latest["ping_ms"])

        if target == "download_mbps":
            download = prediction
        elif target == "upload_mbps":
            upload = prediction
        elif target == "ping_ms":
            ping = prediction

        sample = SpeedSample(
            timestamp_utc=predicted_timestamp.isoformat(),
            download_mbps=round(download, 3),
            upload_mbps=round(upload, 3),
            ping_ms=round(ping, 3),
            server_name=PREDICT_TARGET_SERVER_NAME,
            server_country=PREDICT_TARGET_SERVER_COUNTRY,
            isp=str(latest.get("isp", "predicted")),
            public_ip=str(latest.get("public_ip", "127.0.0.1")),
            local_hostname=str(latest.get("local_hostname", "predicted-host")),
            machine=str(latest.get("machine", "predicted-machine")),
        )
        return sample, step_seconds

    def _estimate_prediction_timestamp(self, frame, horizon: int):
        latest_ts = frame.iloc[-1]["timestamp_utc"]
        if len(frame) >= 2:
            delta = latest_ts - frame.iloc[-2]["timestamp_utc"]
            if delta <= dt.timedelta(0):
                delta = dt.timedelta(minutes=1)
        else:
            delta = dt.timedelta(minutes=1)
        step_seconds = max(int(delta.total_seconds()), 60)
        return latest_ts + (delta * max(horizon, 1)), step_seconds

    def _set_report_text(self, message: str) -> None:
        self.report_box.configure(state="normal")
        self.report_box.delete("1.0", "end")
        self.report_box.insert("end", f"{message}\n")
        self.report_box.configure(state="disabled")

    def _set_training_report(self, message: str) -> None:
        self.train_box.configure(state="normal")
        self.train_box.delete("1.0", "end")
        self.train_box.insert("end", f"{message}\n")
        self.train_box.configure(state="disabled")

    def _set_prediction_report(self, message: str) -> None:
        self.predict_box.configure(state="normal")
        self.predict_box.delete("1.0", "end")
        self.predict_box.insert("end", f"{message}\n")
        self.predict_box.configure(state="disabled")

    def _refresh_prediction_chart(
        self,
        input_path: pathlib.Path,
        prediction_payload: dict[str, object] | None = None,
    ) -> None:
        if (
            self.predict_figure is None
            or self.predict_axis is None
            or self.predict_canvas is None
        ):
            self.predict_chart_status_var.set("Chart unavailable (matplotlib not installed).")
            return
        try:
            frame = predict_load_data(input_path)
            if frame.empty:
                raise ValueError("CSV has no rows.")

            latest_ts_all = frame.iloc[-1]["timestamp_utc"]
            window_start = latest_ts_all - dt.timedelta(hours=1)
            history = frame[frame["timestamp_utc"] >= window_start].copy()
            if history.empty:
                history = frame.tail(1).copy()
            timestamps = history["timestamp_utc"]
            downloads = history["download_mbps"]

            self.predict_axis.clear()
            self.predict_axis.plot(
                timestamps,
                downloads,
                color="#0a84ff",
                linewidth=1.8,
            )
            self.predict_axis.set_title("Download speed history (last 1 hour)")
            self.predict_axis.set_ylabel("Mbps")
            self.predict_axis.set_xlabel("Timestamp (UTC)")
            self.predict_axis.grid(alpha=0.25)
            self.predict_axis.tick_params(axis="x", labelrotation=20, labelsize=8)
            chart_status = f"Chart refreshed (last 1 hour) from: {input_path}"

            if prediction_payload is not None:
                target = str(prediction_payload.get("target", ""))
                prediction = float(prediction_payload.get("prediction", 0.0))
                horizon = int(prediction_payload.get("horizon", 1))
                if target == "download_mbps":
                    latest_ts = history.iloc[-1]["timestamp_utc"]
                    latest_download = float(history.iloc[-1]["download_mbps"])
                    if len(history) >= 2:
                        delta = latest_ts - history.iloc[-2]["timestamp_utc"]
                        if delta <= dt.timedelta(0):
                            delta = dt.timedelta(minutes=1)
                    else:
                        delta = dt.timedelta(minutes=1)
                    predicted_ts = latest_ts + (delta * max(horizon, 1))
                    self.predict_axis.plot(
                        [latest_ts, predicted_ts],
                        [latest_download, prediction],
                        color="#d62828",
                        linewidth=2.0,
                        marker="o",
                    )
                    self.predict_axis.annotate(
                        f"{prediction:.3f}",
                        xy=(predicted_ts, prediction),
                        xytext=(8, 6),
                        textcoords="offset points",
                        color="#d62828",
                        fontsize=8,
                        weight="bold",
                    )
                    self.predict_axis.legend(
                        ["history", "prediction"],
                        loc="upper left",
                    )
                    chart_status = (
                        f"Chart updated with red prediction line at {predicted_ts.isoformat()}"
                    )
                else:
                    chart_status = (
                        "Chart refreshed. Red prediction line only applies when target is "
                        "'download_mbps'."
                    )

            self.predict_figure.tight_layout()
            self.predict_canvas.draw_idle()
            self.predict_chart_status_var.set(chart_status)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Prediction chart refresh failed")
            self.predict_chart_status_var.set(f"Chart error: {exc}")

    def _load_prediction_history(self, limit: int = 1000) -> tuple[list, list[float]]:
        if psycopg is None:
            raise RuntimeError('Missing dependency "psycopg[binary]".')

        port = self._parse_positive_int(self.db_port_var.get(), "DB port")
        conninfo = (
            f"host={self.db_host_var.get()} "
            f"port={port} "
            f"dbname={self.db_name_var.get()} "
            f"user={self.db_user_var.get()} "
            f"password={self.db_password_var.get()}"
        )
        query = """
        WITH latest AS (
            SELECT MAX(timestamp_utc) AS max_ts
            FROM internet_speed_predictions
            WHERE server_name = %s
              AND server_country = %s
        )
        SELECT p.timestamp_utc, p.download_mbps
        FROM internet_speed_predictions p
        CROSS JOIN latest l
        WHERE p.server_name = %s
          AND p.server_country = %s
          AND l.max_ts IS NOT NULL
          AND p.timestamp_utc >= (l.max_ts - interval '1 hour')
        ORDER BY p.timestamp_utc ASC
        LIMIT %s
        """
        with psycopg.connect(conninfo=conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        PREDICT_TARGET_SERVER_NAME,
                        PREDICT_TARGET_SERVER_COUNTRY,
                        PREDICT_TARGET_SERVER_NAME,
                        PREDICT_TARGET_SERVER_COUNTRY,
                        limit,
                    ),
                )
                rows = cur.fetchall()

        if not rows:
            return [], []
        timestamps = [row[0] for row in rows]
        values = [float(row[1]) for row in rows]
        return timestamps, values

    def _load_latest_download_prediction(self) -> tuple[float, str, str]:
        if psycopg is None:
            raise RuntimeError('Missing dependency "psycopg[binary]".')

        port = self._parse_positive_int(self.db_port_var.get(), "DB port")
        conninfo = (
            f"host={self.db_host_var.get()} "
            f"port={port} "
            f"dbname={self.db_name_var.get()} "
            f"user={self.db_user_var.get()} "
            f"password={self.db_password_var.get()}"
        )
        query = """
        SELECT download_mbps, timestamp_utc, COALESCE(model_version, '')
        FROM internet_speed_predictions
        WHERE target_metric = 'download_mbps'
          AND server_name = %s
          AND server_country = %s
        ORDER BY timestamp_utc DESC
        LIMIT 1
        """
        with psycopg.connect(conninfo=conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (PREDICT_TARGET_SERVER_NAME, PREDICT_TARGET_SERVER_COUNTRY),
                )
                row = cur.fetchone()

        if row is None or row[0] is None or row[1] is None:
            raise ValueError("No download prediction rows yet.")
        return float(row[0]), row[1].isoformat(), str(row[2] or "-")

    def _load_model_health(self, limit: int = 60) -> dict[str, float]:
        if psycopg is None:
            raise RuntimeError('Missing dependency "psycopg[binary]".')

        port = self._parse_positive_int(self.db_port_var.get(), "DB port")
        conninfo = (
            f"host={self.db_host_var.get()} "
            f"port={port} "
            f"dbname={self.db_name_var.get()} "
            f"user={self.db_user_var.get()} "
            f"password={self.db_password_var.get()}"
        )
        query = """
        WITH preds AS (
            SELECT
                timestamp_utc,
                download_mbps,
                predicted_from_timestamp_utc,
                COALESCE(expected_step_seconds, 60) AS expected_step_seconds
            FROM internet_speed_predictions
            WHERE target_metric = 'download_mbps'
              AND server_name = %s
              AND server_country = %s
            ORDER BY timestamp_utc DESC
            LIMIT %s
        )
        SELECT
            p.timestamp_utc AS predicted_ts,
            p.download_mbps AS predicted_download,
            a.download_mbps AS actual_download,
            b.download_mbps AS baseline_download
        FROM preds p
        LEFT JOIN LATERAL (
            SELECT s.download_mbps
            FROM internet_speed_samples s
            WHERE s.timestamp_utc >= p.timestamp_utc
              AND s.timestamp_utc <= (
                p.timestamp_utc + make_interval(secs => GREATEST(p.expected_step_seconds, 60))
              )
              AND s.server_name = %s
              AND s.server_country = %s
            ORDER BY s.timestamp_utc ASC
            LIMIT 1
        ) a ON TRUE
        LEFT JOIN LATERAL (
            SELECT s.download_mbps
            FROM internet_speed_samples s
            WHERE s.timestamp_utc <= COALESCE(p.predicted_from_timestamp_utc, p.timestamp_utc)
              AND s.server_name = %s
              AND s.server_country = %s
            ORDER BY s.timestamp_utc DESC
            LIMIT 1
        ) b ON TRUE
        """
        with psycopg.connect(conninfo=conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        PREDICT_TARGET_SERVER_NAME,
                        PREDICT_TARGET_SERVER_COUNTRY,
                        limit,
                        PREDICT_TARGET_SERVER_NAME,
                        PREDICT_TARGET_SERVER_COUNTRY,
                        PREDICT_TARGET_SERVER_NAME,
                        PREDICT_TARGET_SERVER_COUNTRY,
                    ),
                )
                rows = cur.fetchall()

        model_errors: list[float] = []
        baseline_errors: list[float] = []
        for _, predicted_download, actual_download, baseline_download in rows:
            if (
                predicted_download is None
                or actual_download is None
                or baseline_download is None
            ):
                continue
            p = float(predicted_download)
            a = float(actual_download)
            b = float(baseline_download)
            model_errors.append(abs(a - p))
            baseline_errors.append(abs(a - b))

        if not model_errors or not baseline_errors:
            raise ValueError("Not enough matched prediction/actual rows yet.")

        n = len(model_errors)
        mae_model = sum(model_errors) / n
        mae_baseline = sum(baseline_errors) / n
        rmse_model = math.sqrt(sum(e * e for e in model_errors) / n)
        rmse_baseline = math.sqrt(sum(e * e for e in baseline_errors) / n)
        improvement = 0.0
        if mae_baseline > 0:
            improvement = ((mae_baseline - mae_model) / mae_baseline) * 100.0

        return {
            "samples": float(n),
            "mae_model": mae_model,
            "rmse_model": rmse_model,
            "mae_baseline": mae_baseline,
            "rmse_baseline": rmse_baseline,
            "improvement_pct": improvement,
        }

    def _refresh_model_health(self) -> None:
        try:
            metrics = self._load_model_health(limit=60)
            self.model_health_status_var.set(
                f"Model health (last {int(metrics['samples'])} matched predictions).",
            )
            self.model_health_mae_var.set(f"MAE model: {metrics['mae_model']:.3f}")
            self.model_health_rmse_var.set(f"RMSE model: {metrics['rmse_model']:.3f}")
            self.model_health_baseline_mae_var.set(
                f"MAE baseline: {metrics['mae_baseline']:.3f}",
            )
            self.model_health_baseline_rmse_var.set(
                f"RMSE baseline: {metrics['rmse_baseline']:.3f}",
            )
            self.model_health_improvement_var.set(
                f"MAE improvement: {metrics['improvement_pct']:+.1f}%",
            )
            if metrics["mae_model"] < metrics["mae_baseline"]:
                self.model_health_badge_var.set("Status: BETTER than baseline")
                self.model_health_badge_label.configure(text_color="#2e7d32")
            elif metrics["mae_model"] > metrics["mae_baseline"]:
                self.model_health_badge_var.set("Status: WORSE than baseline")
                self.model_health_badge_label.configure(text_color="#c62828")
            else:
                self.model_health_badge_var.set("Status: TIED with baseline")
                self.model_health_badge_label.configure(text_color="#8d8d8d")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Model health refresh failed")
            self.model_health_status_var.set(f"Model health unavailable: {exc}")
            self.model_health_mae_var.set("MAE model: -")
            self.model_health_rmse_var.set("RMSE model: -")
            self.model_health_baseline_mae_var.set("MAE baseline: -")
            self.model_health_baseline_rmse_var.set("RMSE baseline: -")
            self.model_health_improvement_var.set("MAE improvement: -")
            self.model_health_badge_var.set("Status: -")
            self.model_health_badge_label.configure(text_color="#8d8d8d")

    def _refresh_last_predicted_download(self) -> None:
        try:
            value, predicted_ts, model_version = self._load_latest_download_prediction()
            self.last_pred_down_var.set(f"{value:.3f}")
            self.last_pred_model_version_var.set(model_version)
            self._set_prediction_valid_until(predicted_ts)
        except Exception:
            LOGGER.exception("Latest prediction refresh failed")
            self.last_pred_down_var.set("-")
            self.last_pred_valid_var.set("-")
            self.last_pred_model_version_var.set("-")
            self.last_prediction_valid_until_utc = None

    def _set_prediction_valid_until(self, predicted_timestamp_utc: str) -> None:
        if not predicted_timestamp_utc:
            self.last_prediction_valid_until_utc = None
            self.last_pred_valid_var.set("-")
            return
        try:
            predicted_dt = dt.datetime.fromisoformat(predicted_timestamp_utc)
            if predicted_dt.tzinfo is None:
                predicted_dt = predicted_dt.replace(tzinfo=dt.timezone.utc)
            self.last_prediction_valid_until_utc = predicted_dt
            self._refresh_prediction_validity_display()
        except Exception:
            LOGGER.exception("Prediction validity parsing failed")
            self.last_prediction_valid_until_utc = None
            self.last_pred_valid_var.set("-")

    def _refresh_prediction_validity_display(self) -> None:
        if self.last_prediction_valid_until_utc is None:
            self.last_pred_valid_var.set("-")
            return
        now_utc = dt.datetime.now(dt.timezone.utc)
        delta_seconds = int((self.last_prediction_valid_until_utc - now_utc).total_seconds())
        if delta_seconds >= 0:
            hours, rem = divmod(delta_seconds, 3600)
            minutes, seconds = divmod(rem, 60)
            if hours > 0:
                self.last_pred_valid_var.set(f"{hours}h {minutes:02d}m {seconds:02d}s")
            else:
                self.last_pred_valid_var.set(f"{minutes}m {seconds:02d}s")
        else:
            elapsed = abs(delta_seconds)
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)
            if hours > 0:
                self.last_pred_valid_var.set(f"Expired {hours}h {minutes:02d}m ago")
            else:
                self.last_pred_valid_var.set(f"Expired {minutes}m {seconds:02d}s ago")

    def _refresh_prediction_history_chart(self) -> None:
        if (
            self.predict_history_figure is None
            or self.predict_history_axis is None
            or self.predict_history_canvas is None
        ):
            self.predict_history_chart_status_var.set(
                "Prediction history chart unavailable (matplotlib not installed).",
            )
            return
        try:
            timestamps, values = self._load_prediction_history(limit=1000)
            if not timestamps:
                self.predict_history_axis.clear()
                self.predict_history_axis.set_title("Download prediction history (last 1 hour)")
                self.predict_history_axis.set_ylabel("Mbps")
                self.predict_history_axis.set_xlabel("Predicted timestamp (UTC)")
                self.predict_history_axis.grid(alpha=0.25)
                self.predict_history_figure.tight_layout()
                self.predict_history_canvas.draw_idle()
                self.predict_history_chart_status_var.set(
                    "Prediction history chart: no rows yet in internet_speed_predictions.",
                )
                return

            self.predict_history_axis.clear()
            self.predict_history_axis.plot(
                timestamps,
                values,
                color="#d62828",
                linewidth=1.8,
                marker="o",
                markersize=3,
            )
            last_ts = timestamps[-1]
            last_value = values[-1]
            self.predict_history_axis.annotate(
                f"{last_value:.3f}",
                xy=(last_ts, last_value),
                xytext=(8, 6),
                textcoords="offset points",
                color="#d62828",
                fontsize=8,
                weight="bold",
            )
            self.predict_history_axis.set_title("Download prediction history (last 1 hour)")
            self.predict_history_axis.set_ylabel("Mbps")
            self.predict_history_axis.set_xlabel("Predicted timestamp (UTC)")
            self.predict_history_axis.grid(alpha=0.25)
            self.predict_history_axis.tick_params(axis="x", labelrotation=20, labelsize=8)
            self.predict_history_figure.tight_layout()
            self.predict_history_canvas.draw_idle()
            self.predict_history_chart_status_var.set(
                "Prediction history chart refreshed (last 1 hour) from PostgreSQL.",
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Prediction history chart refresh failed")
            self.predict_history_chart_status_var.set(f"Prediction history error: {exc}")

    def _refresh_prediction_combined_chart(
        self,
        input_path: pathlib.Path,
        prediction_payload: dict[str, object] | None = None,
    ) -> None:
        if (
            self.predict_combined_figure is None
            or self.predict_combined_axis is None
            or self.predict_combined_canvas is None
        ):
            self.predict_combined_chart_status_var.set(
                "Combined chart unavailable (matplotlib not installed).",
            )
            return
        try:
            frame = predict_load_data(input_path)
            if frame.empty:
                raise ValueError("CSV has no rows.")

            latest_ts_all = frame.iloc[-1]["timestamp_utc"]
            window_start = latest_ts_all - dt.timedelta(hours=1)
            history = frame[frame["timestamp_utc"] >= window_start].copy()
            if history.empty:
                history = frame.tail(1).copy()

            self.predict_combined_axis.clear()
            self.predict_combined_axis.plot(
                history["timestamp_utc"],
                history["download_mbps"],
                color="#0a84ff",
                linewidth=1.8,
                label="download history",
            )

            prediction_history_loaded = False
            try:
                pred_timestamps, pred_values = self._load_prediction_history(limit=1000)
                if pred_timestamps:
                    self.predict_combined_axis.plot(
                        pred_timestamps,
                        pred_values,
                        color="#d62828",
                        linewidth=1.4,
                        marker="o",
                        markersize=2.5,
                        alpha=0.9,
                        label="prediction history",
                    )
                    prediction_history_loaded = True
            except Exception:
                prediction_history_loaded = False

            if prediction_payload is not None:
                target = str(prediction_payload.get("target", ""))
                prediction = float(prediction_payload.get("prediction", 0.0))
                horizon = int(prediction_payload.get("horizon", 1))
                if target == "download_mbps":
                    latest_ts = history.iloc[-1]["timestamp_utc"]
                    latest_download = float(history.iloc[-1]["download_mbps"])
                    if len(history) >= 2:
                        delta = latest_ts - history.iloc[-2]["timestamp_utc"]
                        if delta <= dt.timedelta(0):
                            delta = dt.timedelta(minutes=1)
                    else:
                        delta = dt.timedelta(minutes=1)
                    predicted_ts = latest_ts + (delta * max(horizon, 1))
                    self.predict_combined_axis.plot(
                        [latest_ts, predicted_ts],
                        [latest_download, prediction],
                        color="#ff7f11",
                        linewidth=2.0,
                        marker="o",
                        label="latest prediction",
                    )

            self.predict_combined_axis.set_title(
                "Combined chart: download + prediction history (last 1 hour)",
            )
            self.predict_combined_axis.set_ylabel("Mbps")
            self.predict_combined_axis.set_xlabel("Timestamp (UTC)")
            self.predict_combined_axis.grid(alpha=0.25)
            self.predict_combined_axis.tick_params(axis="x", labelrotation=20, labelsize=8)
            self.predict_combined_axis.legend(loc="upper left")
            self.predict_combined_figure.tight_layout()
            self.predict_combined_canvas.draw_idle()
            if prediction_history_loaded:
                self.predict_combined_chart_status_var.set(
                    "Combined chart refreshed with CSV download + PostgreSQL prediction history.",
                )
            else:
                self.predict_combined_chart_status_var.set(
                    "Combined chart refreshed with CSV download history (no prediction history).",
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Combined prediction chart refresh failed")
            self.predict_combined_chart_status_var.set(f"Combined chart error: {exc}")

    def report_callback_exception(
        self,
        exc: type[BaseException],
        value: BaseException,
        traceback: object,
    ) -> None:
        LOGGER.error(
            "Tk callback exception",
            exc_info=(exc, value, traceback),
        )
        super().report_callback_exception(exc, value, traceback)

    def _build_training_explanation(self, metrics: dict[str, object]) -> str:
        mae_baseline = float(metrics["mae_baseline"])
        rmse_baseline = float(metrics["rmse_baseline"])
        mae_model = float(metrics["mae_model"])
        rmse_model = float(metrics["rmse_model"])

        mae_delta = mae_model - mae_baseline
        rmse_delta = rmse_model - rmse_baseline
        mae_pct = (mae_delta / mae_baseline * 100) if mae_baseline else 0.0
        rmse_pct = (rmse_delta / rmse_baseline * 100) if rmse_baseline else 0.0

        mae_trend = "mejora" if mae_delta < 0 else "empeora"
        rmse_trend = "mejora" if rmse_delta < 0 else "empeora"

        return (
            "Explicacion:\n"
            f"- MAE: el modelo {mae_trend} vs baseline en {abs(mae_pct):.1f}% "
            f"({mae_baseline:.3f} -> {mae_model:.3f}).\n"
            f"- RMSE: el modelo {rmse_trend} vs baseline en {abs(rmse_pct):.1f}% "
            f"({rmse_baseline:.3f} -> {rmse_model:.3f}).\n"
            "- Menor MAE/RMSE es mejor; RMSE penaliza mas los errores grandes."
        )

    def _append_log(self, message: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{message}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def _on_close(self) -> None:
        self.stop_event.set()
        self.destroy()


def main() -> None:
    """Launch MEDIDOR GUI."""
    _configure_error_logging()
    _install_exception_hooks()
    LOGGER.info("MEDIDOR GUI startup")
    app = MedidorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
