"""
Digantara – Data Scientist I Assignment (Updated)
------------------------------------------------
This module implements a clean, explainable, and review-friendly pipeline to detect
photometric deviations in satellite observations by comparing reference and current
measurement windows.

Updates in this version:
- Explicit data cleaning & null handling step
- Step-wise data summaries for reporting
- Optional plotting helpers for exploratory analysis

Design principles:
- No black-box ML for anomaly detection
- Physics-aware handling of glinting effects
- Robust, explainable statistical methods
- Code clarity > algorithmic cleverness
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PhotometricChangeDetector:
    """
    Main class expected by the assignment.

    The class exposes a single public method `run`, which:
    - Reads input configuration (JSON)
    - Processes each era independently
    - Cleans and summarizes data step-by-step
    - Detects photometric deviations per NORAD object
    - Writes anomaly summaries as JSON output
    """

    def __init__(self,
                 glint_phase_threshold: float = 5.0,
                 phase_bin_size: float = 5.0,
                 min_event_points: int = 3):
        """
        Parameters
        ----------
        glint_phase_threshold : float
            Absolute SEPA value (degrees) below which glinting is expected.
        phase_bin_size : float
            Bin width (degrees) used to model reference behavior vs phase angle.
        min_event_points : int
            Minimum number of consecutive anomalous points to form an event.
        """
        self.glint_phase_threshold = glint_phase_threshold
        self.phase_bin_size = phase_bin_size
        self.min_event_points = min_event_points

        # Store summaries for reporting
        self.data_summaries: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, input_json_path: str, output_json_path: str) -> None:
        """
        Entry point required by the assignment.

        Parameters
        ----------
        input_json_path : str
            Path to JSON describing era folders and time windows.
        output_json_path : str
            Path where detected anomalies will be written as JSON.
        """
        with open(input_json_path, 'r') as f:
            config = json.load(f)

        all_era_results = []

        for era_cfg in config["data"]:
            era_results = self._process_single_era(era_cfg)
            all_era_results.extend(era_results)

        with open(output_json_path, 'w') as f:
            json.dump(all_era_results, f, indent=2)

    # ------------------------------------------------------------------
    # Era-level processing
    # ------------------------------------------------------------------
    def _process_single_era(self, era_cfg: Dict) -> List[Dict]:
        # store current era name for downstream attribution
        self._current_era = era_cfg.get("era_name", "unknown")
        """
        Process one era independently (no cross-era leakage).
        """
        df_raw = self._load_era_csvs(era_cfg["data_path"])
        self._summarize(df_raw, era_cfg["era_name"], "raw_loaded")

        df_clean = self._clean_data(df_raw)
        self._summarize(df_clean, era_cfg["era_name"], "after_cleaning")

        df = self._prepare_dataframe(df_clean)

        reference_df, current_df = self._split_reference_current(
            df,
            era_cfg["reference_start"],
            era_cfg["reference_end"],
            era_cfg["current_start"],
            era_cfg["current_end"],
        )

        self._summarize(reference_df, era_cfg["era_name"], "reference_window")
        self._summarize(current_df, era_cfg["era_name"], "current_window")

        era_anomalies = []

        for norad_id, ref_obj_df in reference_df.groupby("norad_id"):
            cur_obj_df = current_df[current_df["norad_id"] == norad_id]
            if cur_obj_df.empty:
                continue

            ref_no_outliers = self._handle_outliers(ref_obj_df)
            self._summarize(ref_no_outliers, era_cfg["era_name"], f"ref_no_outliers_{norad_id}")

            ref_profile = self._build_reference_profile(ref_no_outliers)

            anomalies = self._detect_anomalies(
                ref_profile,
                cur_obj_df,
                norad_id
            )
            era_anomalies.extend(anomalies)

        return era_anomalies

    # ------------------------------------------------------------------
    # Data loading & cleaning
    # ------------------------------------------------------------------
    def _load_era_csvs(self, folder_path: str) -> pd.DataFrame:
        """
        Load all CSV files belonging to a single era.
        """
        frames = []
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                frames.append(pd.read_csv(os.path.join(folder_path, file)))
        return pd.concat(frames, ignore_index=True)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values and obvious data quality issues.
        This step ensures downstream logic is not affected by invalid rows.
        """
        df = df.copy()

        # Drop rows with critical missing values
        df = df.dropna(subset=[
            "norad_id",
            "timestamp",
            "equatorial_phase",
            "magnitude"
        ])

        # Remove non-physical magnitude uncertainty values
        if "magnitude_unc" in df.columns:
            df = df[df["magnitude_unc"] >= 0]

        return df

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column types and sort data.
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.sort_values("timestamp", inplace=True)
        return df

    def _split_reference_current(self,
                                 df: pd.DataFrame,
                                 ref_start: str,
                                 ref_end: str,
                                 cur_start: str,
                                 cur_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into reference and current windows.
        """
        ref_mask = (df["timestamp"] >= pd.to_datetime(ref_start, utc=True)) & \
                   (df["timestamp"] < pd.to_datetime(ref_end, utc=True))

        cur_mask = (df["timestamp"] >= pd.to_datetime(cur_start, utc=True)) & \
                   (df["timestamp"] < pd.to_datetime(cur_end, utc=True))

        return df[ref_mask], df[cur_mask]

    # ------------------------------------------------------------------
    # Summaries & visualization helpers
    # ------------------------------------------------------------------
    def _summarize(self, df: pd.DataFrame, era: str, step: str) -> None:
        """
        Capture step-wise data summaries for reporting.
        """
        summary = {
            "era": era,
            "step": step,
            "rows": len(df),
            "nulls": df.isnull().sum().to_dict()
        }
        self.data_summaries.append(summary)
        

    def plot_distribution(self, df: pd.DataFrame, title: str) -> None:
        """
        Plot magnitude vs equatorial phase to visualize data spread and outliers.
        """
        plt.figure(figsize=(8, 5))
        plt.scatter(df["equatorial_phase"], df["magnitude"], s=5, alpha=0.4)
        plt.xlabel("Equatorial Phase (deg)")
        plt.ylabel("Magnitude")
        plt.title(title)
        plt.grid(True)
        plt.show()

    # ------------------------------------------------------------------
    # Reference modeling
    # ------------------------------------------------------------------
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove extreme magnitude outliers from reference data using IQR filtering.
        Outliers are visualized and analyzed, but not blindly treated as anomalies.
        """
        q1 = df["magnitude"].quantile(0.25)
        q3 = df["magnitude"].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        return df[(df["magnitude"] >= lower) & (df["magnitude"] <= upper)]

    def _build_reference_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a phase-binned reference envelope (median + IQR), excluding glint region.
        """
        df = df.copy()
        df = df[np.abs(df["equatorial_phase"]) > self.glint_phase_threshold]

        df["phase_bin"] = (
            df["equatorial_phase"] // self.phase_bin_size
        ) * self.phase_bin_size

        profile = df.groupby("phase_bin")["magnitude"].agg(
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75)
        ).reset_index()

        return profile

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------
    def _detect_anomalies(self,
                          ref_profile: pd.DataFrame,
                          cur_df: pd.DataFrame,
                          norad_id: int) -> List[Dict]:
        """
        Compare current measurements against reference envelope
        and group sustained deviations into anomaly events.
        """
        anomalies = []

        cur_df = cur_df.copy()
        cur_df = cur_df[np.abs(cur_df["equatorial_phase"]) > self.glint_phase_threshold]

        cur_df["phase_bin"] = (
            cur_df["equatorial_phase"] // self.phase_bin_size
        ) * self.phase_bin_size

        merged = cur_df.merge(ref_profile, on="phase_bin", how="left")
        merged.dropna(inplace=True)

        merged["deviation"] = (
            merged["magnitude"] - merged["median"]
        ) / (merged["q3"] - merged["q1"] + 1e-6)

        merged["is_anomaly"] = np.abs(merged["deviation"]) > 1.5

        merged.sort_values("timestamp", inplace=True)
        merged["group"] = (merged["is_anomaly"] != merged["is_anomaly"].shift()).cumsum()

        for _, grp in merged.groupby("group"):
            if grp["is_anomaly"].all() and len(grp) >= self.min_event_points:
                anomalies.append({
                    "era": self._current_era,
                    "norad_id": int(norad_id),
                    "equatorial_phase": [
                        float(grp["equatorial_phase"].min()),
                        float(grp["equatorial_phase"].max())
                    ],
                    "timestamp": [
                        grp["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
                        grp["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S")
                    ],
                    "deviation_score": float(1 - np.exp(-np.mean(np.abs(grp["deviation"]))))
                })

        return anomalies


detector = PhotometricChangeDetector()
detector.run("input_config.json", "anomalies1.json")
