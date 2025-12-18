import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PhotometricChangeDetector:
    """
    Main class.

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
                 min_event_points: int = 3,
                 plot_enabled: bool = False):
        """
        Parameters
        ----------
        glint_phase_threshold : float
            Absolute SEPA value (degrees) below which glinting is expected.
        phase_bin_size : float
            Bin width (degrees) used to model reference behavior vs phase angle.
        min_event_points : int
            Minimum number of consecutive anomalous points to form an event.
        plot_enabled : bool
            Enable optional visualization for explanation.
        """
        self.glint_phase_threshold = glint_phase_threshold
        self.phase_bin_size = phase_bin_size
        self.min_event_points = min_event_points
        self.plot_enabled = plot_enabled

        self.data_summaries: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, input_json_path: str, output_json_path: str) -> None:
        with open(input_json_path, 'r') as f:
            config = json.load(f)

        all_era_results = []

        for era_cfg in config["data"]:
            era_results = self._process_single_era(era_cfg)
            all_era_results.extend(era_results)

        with open(output_json_path, 'w') as f:
            json.dump(all_era_results, f, indent=2)

        if self.plot_enabled:
            self._plot_deviation_vs_phase(all_era_results)
            self._plot_data_volume()

    # ------------------------------------------------------------------
    # Era-level processing
    # ------------------------------------------------------------------
    def _process_single_era(self, era_cfg: Dict) -> List[Dict]:
        self._current_era = era_cfg.get("era_name", "unknown")

        df_raw = self._load_era_csvs(era_cfg["data_path"])
        self._summarize(df_raw, self._current_era, "raw_loaded")

        df_clean = self._clean_data(df_raw)
        self._summarize(df_clean, self._current_era, "after_cleaning")

        df = self._prepare_dataframe(df_clean)

        reference_df, current_df = self._split_reference_current(
            df,
            era_cfg["reference_start"],
            era_cfg["reference_end"],
            era_cfg["current_start"],
            era_cfg["current_end"],
        )

        self._summarize(reference_df, self._current_era, "reference_window")
        self._summarize(current_df, self._current_era, "current_window")

        era_anomalies = []

        for norad_id, ref_obj_df in reference_df.groupby("norad_id"):
            cur_obj_df = current_df[current_df["norad_id"] == norad_id]
            if cur_obj_df.empty:
                continue

            ref_no_outliers = self._handle_outliers(ref_obj_df)
            self._summarize(
                ref_no_outliers,
                self._current_era,
                f"ref_no_outliers_{norad_id}"
            )

            ref_profile = self._build_reference_profile(ref_no_outliers)

            if self.plot_enabled:
                self._plot_reference_vs_current(
                    ref_no_outliers,
                    cur_obj_df,
                    self._current_era,
                    norad_id
                )

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
        frames = []
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                frames.append(pd.read_csv(os.path.join(folder_path, file)))
        return pd.concat(frames, ignore_index=True)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.dropna(subset=[
            "norad_id",
            "timestamp",
            "equatorial_phase",
            "magnitude"
        ])

        if "magnitude_unc" in df.columns:
            df = df[df["magnitude_unc"] >= 0]

        return df

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
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
        ref_mask = (df["timestamp"] >= pd.to_datetime(ref_start, utc=True)) & \
                   (df["timestamp"] < pd.to_datetime(ref_end, utc=True))

        cur_mask = (df["timestamp"] >= pd.to_datetime(cur_start, utc=True)) & \
                   (df["timestamp"] < pd.to_datetime(cur_end, utc=True))

        return df[ref_mask], df[cur_mask]

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    def _summarize(self, df: pd.DataFrame, era: str, step: str) -> None:
        self.data_summaries.append({
            "era": era,
            "step": step,
            "rows": len(df),
            "nulls": df.isnull().sum().to_dict()
        })

    # ------------------------------------------------------------------
    # Reference modeling
    # ------------------------------------------------------------------
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        q1 = df["magnitude"].quantile(0.25)
        q3 = df["magnitude"].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        return df[(df["magnitude"] >= lower) & (df["magnitude"] <= upper)]

    def _build_reference_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df[np.abs(df["equatorial_phase"]) > self.glint_phase_threshold]

        df["phase_bin"] = (
            df["equatorial_phase"] // self.phase_bin_size
        ) * self.phase_bin_size

        return df.groupby("phase_bin")["magnitude"].agg(
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75)
        ).reset_index()

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------
    def _detect_anomalies(self,
                          ref_profile: pd.DataFrame,
                          cur_df: pd.DataFrame,
                          norad_id: int) -> List[Dict]:
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
                    "deviation_score": float(
                        1 - np.exp(-np.mean(np.abs(grp["deviation"])))
                    )
                })

        return anomalies

    # ------------------------------------------------------------------
    # Visualization (Optional, Explanation Only)
    # ------------------------------------------------------------------
    def _plot_reference_vs_current(self,
                                   ref_df: pd.DataFrame,
                                   cur_df: pd.DataFrame,
                                   era: str,
                                   norad_id: int) -> None:
        plt.figure(figsize=(8, 5))
        plt.scatter(ref_df["equatorial_phase"],
                    ref_df["magnitude"],
                    s=6, alpha=0.3, label="Reference")
        plt.scatter(cur_df["equatorial_phase"],
                    cur_df["magnitude"],
                    s=8, alpha=0.6, label="Current")
        plt.xlabel("Equatorial Phase (deg)")
        plt.ylabel("Magnitude")
        plt.title(f"{era} | NORAD {norad_id}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _plot_deviation_vs_phase(self, anomalies: List[Dict]) -> None:
        if not anomalies:
            return

        df = pd.DataFrame(anomalies)
        df["mean_phase"] = df["equatorial_phase"].apply(
            lambda x: (x[0] + x[1]) / 2
        )

        plt.figure(figsize=(8, 5))
        plt.scatter(df["mean_phase"], df["deviation_score"], s=30)
        plt.xlabel("Equatorial Phase (deg)")
        plt.ylabel("Deviation Score")
        plt.title("Deviation Severity vs Phase")
        plt.tight_layout()
        plt.show()

    def _plot_data_volume(self) -> None:
        df = pd.DataFrame(self.data_summaries)
        pivot = df.pivot_table(
            index="step",
            columns="era",
            values="rows",
            aggfunc="first"
        )
        pivot.plot(kind="bar", figsize=(10, 5))
        plt.ylabel("Number of Rows")
        plt.title("Data Volume Across Processing Stages")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()


# -------------------------------
# Execution
# -------------------------------
if __name__ == "__main__":
    detector = PhotometricChangeDetector(plot_enabled=False)
    detector.run("input_config.json", "anomalies.json")
